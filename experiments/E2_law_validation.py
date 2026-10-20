"""E2: prospective test of the first-order drift-compensation laws as a design-time predictor.

Frozen plan: experiments/E2_config.json (Chinese description: docs/E2_protocol.md).

Phase 1 (predict): nominal and +-1 sigma single-device simulations only -> predictions.json
    python experiments/E2_law_validation.py --phase predict [--smoke]
Phase 2 (mc): Monte Carlo with the PDK mismatch model, compared with the frozen predictions
    python experiments/E2_law_validation.py --phase mc --predictions results/E2_predict_<stamp>/predictions.json [--smoke]

Outputs (never overwritten): results/<run_prefix>_predict_<stamp>[_smoke]/ and results/<run_prefix>_<stamp>[_smoke]/
(run_prefix from the config: "E2" for the test; "E2cal_E1" for the pre-freeze calibration on the E1 designs)
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pcm_acam.branch_post import IVCurve, Mapping, bound, g_of_x, summarize, switching_voltage  # noqa: E402
from pcm_acam.io_utils import environment_info, make_logger, make_run_dir, now_jst, sha256_file, write_json  # noqa: E402
from pcm_acam.law_predict import divider_sensitivity, predict, pulldown_sensitivity  # noqa: E402
from pcm_acam.spice_iv import MM_KEYS, branch_dc, iv_sweep, ngspice_version  # noqa: E402

NODES = ("none", "dac_remap", "supply", "ideal")
SETTINGS = ("pdk", "div_only", "pd_only")
LSB_PER_X = 15.0
I_FLOOR = 1e-9


def lib_for(cfg: dict, corner: str) -> Path:
    return ROOT / cfg["models"]["subset_dir"] / cfg["models"]["lib_pattern"].format(corner=corner)


def drift_factors(cfg: dict) -> Dict[str, float]:
    d = cfg["drift"]
    return {n: float((t / d["t0_s"]) ** (-d["nu"])) for n, t in d["times_s"].items()}


def write_csv(path: Path, rows: List[dict]) -> None:
    if rows:
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)


def hashes(cfg_path: Path, cfg: dict) -> Dict[str, str]:
    files = [cfg_path, Path(__file__), ROOT / "src/pcm_acam/law_predict.py", ROOT / "src/pcm_acam/spice_iv.py",
             ROOT / "src/pcm_acam/branch_post.py", ROOT / "src/pcm_acam/sky130_models.py"]
    files += sorted((ROOT / cfg["models"]["subset_dir"]).glob("*"))
    return {str(p.relative_to(ROOT)): sha256_file(p) for p in files}


def conditions(cfg: dict, smoke: bool) -> List[dict]:
    return [c for c in cfg["conditions"] if (not smoke) or c["id"] in cfg["smoke"]["conditions"]]


# ------------------------------------------------------------------ predict phase
def phase_predict(cfg: dict, cfg_path: Path, smoke: bool) -> int:
    run = make_run_dir(ROOT / "results", cfg.get("run_prefix", "E2") + "_predict", smoke)
    log = make_logger(run / "run.log")
    info = {"experiment": cfg.get("run_prefix", "E2"), "phase": "predict", "smoke": smoke, "start_jst": now_jst().isoformat(timespec="seconds"),
            "environment": environment_info(), "ngspice": ngspice_version(), "sha256": hashes(cfg_path, cfg)}
    write_json(run / "run_info.json", info)
    write_json(run / "config_used.json", cfg)
    circ, sw = cfg["circuit"], cfg["sweep"]
    a_of = drift_factors(cfg)
    theta = np.arange(16) / 15.0
    xg = np.arange(cfg["x_grid"][0], cfg["x_grid"][1] + 1e-12, cfg["x_grid"][2])
    out, rows = {}, []
    pd_cache: Dict[tuple, dict] = {}
    for c in conditions(cfg, smoke):
        t0 = time.time()
        lib = lib_for(cfg, c["corner"])
        pd = circ["pulldowns"][c["pulldown"]]
        key = (c["pulldown"], c["corner"], c["temp_c"])
        if key not in pd_cache:
            pd_cache[key] = pulldown_sensitivity(lib, pd["device"], pd["w_um"], pd["l_um"], circ["vdd"],
                                                 100e-9 * pd["w_um"] / pd["l_um"], sw, temp=c["temp_c"])
        pds = pd_cache[key]
        vsw_nom = pds["vsw_nom"]
        v_sl = vsw_nom + c["d_v"]
        vg, cur, curves = divider_sensitivity(lib, circ["divider_device"], c["w_div_um"], c["l_div_um"], vsw_nom, sw,
                                              temp=c["temp_c"], i_floor=I_FLOOR)
        nom = curves[0]
        b30 = float(bound(nom, np.array([30.0]), v_sl, vsw_nom)[0])
        m0 = bound(nom, g_of_x(xg, circ["g_lo_us"], circ["g_hi_us"]), v_sl, vsw_nom)
        in_need = xg <= 1.0
        feasible = bool(np.isfinite(b30) and b30 <= 1.75 and np.all(np.isfinite(m0[in_need])))
        entry = {"condition": c, "vsw_nom_V": vsw_nom, "v_sl_nom_V": v_sl, "pulldown_sensitivity": pds,
                 "bound_at_30uS_V": b30 if np.isfinite(b30) else None, "feasible": feasible,
                 "bound_x0_V": float(m0[np.argmin(abs(xg))]), "bound_x1_V": float(m0[np.argmin(abs(xg - 1))]) if np.isfinite(m0[np.argmin(abs(xg - 1))]) else None,
                 "nominal_ln_i_sha256": None, "predictions": {}}
        entry["nominal_ln_i_sha256"] = hashlib.sha256(np.ascontiguousarray(nom.ln_i).tobytes()).hexdigest()
        np.save(run / f"nominal_curve_{c['id']}.npy", np.vstack([cur[0], vg]))
        if feasible:
            for tn, a in a_of.items():
                pr = predict(curves, theta, circ["g_lo_us"], circ["g_hi_us"], c["d_v"], a, pds["sigma_delta"])
                entry["predictions"][tn] = pr
                for k, v in pr["rms_lsb"].items():
                    s, node = k.split("|")
                    rows.append({"condition": c["id"], "set": c["set"], "time": tn, "setting": s, "node": node,
                                 "pred_rms_lsb": v, "sigma_delta_mV": 1e3 * pds["sigma_delta"]})
        out[c["id"]] = entry
        log(f"{c['id']}: V_sw,nom {vsw_nom:.4f} V, sigma_delta {1e3 * pds['sigma_delta']:.2f} mV, bound(30 uS) "
            f"{b30:.3f} V, feasible {feasible}, {time.time() - t0:.1f} s")
    pred_path = run / "predictions.json"
    with open(pred_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    write_csv(run / "predictions.csv", rows)
    info["end_jst"] = now_jst().isoformat(timespec="seconds")
    info["predictions_sha256"] = sha256_file(pred_path)
    write_json(run / "run_info.json", info)
    log(f"predictions written; sha256 {info['predictions_sha256']}")
    return 0


# ------------------------------------------------------------------ decision (criteria A1-A3)
def decide(compare: List[dict]) -> dict:
    """Frozen criteria on the prediction-versus-simulation table (rows of compare.csv)."""
    sel = [r for r in compare if r["time"] == "1y" and r["setting"] == "pdk" and r["node"] in ("dac_remap", "supply")]
    a1_rate = float(np.mean([r["pass"] for r in sel]))
    a1 = a1_rate >= 0.9
    a2_rows = []
    for cid in sorted({r["condition"] for r in sel}):
        pr = {r["node"]: r for r in sel if r["condition"] == cid}
        pd_, ps_ = pr["dac_remap"]["pred_rms_lsb"], pr["supply"]["pred_rms_lsb"]
        sd_, ss_ = pr["dac_remap"]["sim_rms_lsb"], pr["supply"]["sim_rms_lsb"]
        clear = abs(pd_ - ps_) >= 0.2 * max(pd_, ps_)
        a2_rows.append({"condition": cid, "pred_better": "dac_remap" if pd_ < ps_ else "supply",
                        "sim_better": "dac_remap" if sd_ < ss_ else "supply", "clear_prediction": bool(clear),
                        "agree": bool((pd_ < ps_) == (sd_ < ss_))})
    a2 = all(r["agree"] for r in a2_rows if r["clear_prediction"])
    sel3 = [r for r in compare if r["time"] == "1y" and ((r["setting"] == "div_only" and r["node"] == "dac_remap")
                                                        or (r["setting"] == "pd_only" and r["node"] in ("dac_remap", "supply")))]
    a3_rate = float(np.mean([r["pass"] for r in sel3]))
    return {"A1": {"pass_rate": a1_rate, "pass": bool(a1), "n": len(sel),
                   "failures": [f"{r['condition']}/{r['node']}" for r in sel if not r["pass"]]},
            "A2": {"pass": bool(a2), "n_clear": sum(r["clear_prediction"] for r in a2_rows), "rows": a2_rows},
            "A3_secondary": {"pass_rate": a3_rate, "pass": bool(a3_rate >= 0.9), "n": len(sel3),
                             "failures": [f"{r['condition']}/{r['setting']}/{r['node']}" for r in sel3 if not r["pass"]]},
            "laws_validated": bool(a1 and a2)}


# ------------------------------------------------------------------ Monte Carlo phase
def phase_mc(cfg: dict, cfg_path: Path, smoke: bool, pred_path: Path) -> int:
    run = make_run_dir(ROOT / "results", cfg.get("run_prefix", "E2"), smoke)
    log = make_logger(run / "run.log")
    preds = json.loads(pred_path.read_text())
    info = {"experiment": cfg.get("run_prefix", "E2"), "phase": "mc", "smoke": smoke, "start_jst": now_jst().isoformat(timespec="seconds"),
            "environment": environment_info(), "ngspice": ngspice_version(), "sha256": hashes(cfg_path, cfg),
            "predictions_file": str(pred_path.relative_to(ROOT)), "predictions_sha256": sha256_file(pred_path)}
    write_json(run / "run_info.json", info)
    write_json(run / "config_used.json", cfg)
    log(f"E2 Monte Carlo phase; predictions {pred_path.relative_to(ROOT)} sha256 {info['predictions_sha256']}")
    circ, sw = cfg["circuit"], cfg["sweep"]
    a_of = drift_factors(cfg)
    times = list(a_of)
    theta = np.arange(16) / 15.0
    xg = np.arange(cfg["x_grid"][0], cfg["x_grid"][1] + 1e-12, cfg["x_grid"][2])
    g_lo, g_hi = circ["g_lo_us"], circ["g_hi_us"]
    mmc = cfg["mismatch"]
    rng = np.random.default_rng([mmc["base_seed"], mmc["rng_stream"]])
    z_div_all = rng.standard_normal((mmc["n_instances"], 4))
    z_pd_all = rng.standard_normal((mmc["n_instances"], 4))
    n = cfg["smoke"]["n_instances"] if smoke else mmc["n_instances"]
    z_div, z_pd = z_div_all[:n], z_pd_all[:n]
    mm_div = [{k: float(v) for k, v in zip(MM_KEYS, r)} for r in z_div]
    mm_pd = [{k: float(v) for k, v in zip(MM_KEYS, r)} for r in z_pd]

    checks: Dict[str, dict] = {k: {} for k in ("C1", "C2", "C3", "C4", "C5", "C6", "C7")}
    failed: List[str] = []
    summary, profile, compare = [], [], []
    sim_index: Dict[tuple, float] = {}
    vsw_nom_by: Dict[tuple, float] = {}

    for c in conditions(cfg, smoke):
        cid = c["id"]
        entry = preds[cid]
        if not entry["feasible"]:
            log(f"{cid}: infeasible in the predict phase -> skipped")
            continue
        t0 = time.time()
        lib = lib_for(cfg, c["corner"])
        pd = circ["pulldowns"][c["pulldown"]]
        i_sw = 100e-9 * pd["w_um"] / pd["l_um"]
        # pass A
        vg, cur = iv_sweep(lib, pd["device"], pd["w_um"], pd["l_um"], [circ["vdd"]] * (n + 1), [None] + mm_pd,
                           sw["vg_start"], sw["vg_stop"], sw["vg_step"], temp=c["temp_c"])
        vsw = np.array([switching_voltage(vg, cur[k], i_sw) for k in range(n + 1)])
        vsw_nom = float(vsw[0])
        vsw_mc = vsw[1:]
        vsw_nom_by[(c["pulldown"], c["corner"], c["temp_c"])] = vsw_nom
        if abs(vsw_nom - entry["vsw_nom_V"]) > 1e-9:
            failed.append(f"C7 {cid}: V_sw,nom differs from the predict phase")
        sig = float(np.std(vsw_mc, ddof=1))
        tol = max(0.15, 3.0 / np.sqrt(2.0 * (n - 1)))
        sd_pred = entry["pulldown_sensitivity"]["sigma_delta"]
        ok = abs(sig / sd_pred - 1) <= tol
        checks["C4"][cid] = {"sigma_mc_mV": 1e3 * sig, "sigma_pred_mV": 1e3 * sd_pred, "tolerance": tol, "pass": bool(ok)}
        if not ok:
            failed.append(f"C4 {cid}")
        d_v = c["d_v"]
        v_sl = vsw_nom + d_v
        # pass B
        runs = {"pdk": ([vsw_nom] + list(vsw_mc), [None] + mm_div),
                "div_only": ([vsw_nom] * (n + 1), [None] + mm_div),
                "pd_only": ([vsw_nom] + list(vsw_mc), [None] * (n + 1))}
        curves, vg_div = {}, None
        for s, (vds, mm) in runs.items():
            vg_div, cur_b = iv_sweep(lib, circ["divider_device"], c["w_div_um"], c["l_div_um"], vds, mm,
                                     sw["vg_start"], sw["vg_stop"], sw["vg_step"], temp=c["temp_c"])
            try:
                curves[s] = [IVCurve.from_arrays(vg_div, cur_b[k], I_FLOOR) for k in range(n + 1)]
            except ValueError as e:
                failed.append(f"C5 {cid} {s}: {e}")
                curves[s] = None
        nom = curves["div_only"][0]
        if hashlib.sha256(np.ascontiguousarray(nom.ln_i).tobytes()).hexdigest() != entry["nominal_ln_i_sha256"]:
            saved = np.load(pred_path.parent / f"nominal_curve_{cid}.npy")
            diff = float(np.max(np.abs(np.log(np.maximum(saved[0], 1e-30))[-nom.ln_i.size:] - nom.ln_i)))
            checks["C7"][cid] = {"max_abs_dlnI": diff, "pass": diff <= 1e-9}
            if diff > 1e-9:
                failed.append(f"C7 {cid}")
        else:
            checks["C7"][cid] = {"max_abs_dlnI": 0.0, "pass": True}
        # maps
        m0 = Mapping(xg, bound(nom, g_of_x(xg, g_lo, g_hi), v_sl, vsw_nom))
        maps_a = {tn: Mapping(xg, bound(nom, a * g_of_x(xg, g_lo, g_hi), v_sl, vsw_nom)) for tn, a in a_of.items()}
        b_target = bound(nom, g_of_x(theta, g_lo, g_hi), v_sl, vsw_nom)
        # C1
        c1 = []
        for vs in (v_sl, vsw_nom + d_v / a_of["1y"]):
            vdl, vx = branch_dc(lib, circ["divider_device"], c["w_div_um"], c["l_div_um"], [2.5, 12.5, 22.5], vs,
                                sw["vg_start"], sw["vg_stop"], sw["vg_step"], temp=c["temp_c"])
            for k, gg in enumerate([2.5, 12.5, 22.5]):
                c1.append(1e3 * abs(float(np.interp(-vsw_nom, -vx[k], vdl)) - float(bound(nom, np.array([gg]), vs, vsw_nom)[0])))
        checks["C1"][cid] = {"max_abs_diff_mV": max(c1), "pass": max(c1) <= 0.05}
        if max(c1) > 0.05:
            failed.append(f"C1 {cid}")

        def evaluate(cv, vs_arr, g_max):
            g_prog = np.vstack([cv[k].i_at(b_target) / (v_sl - vs_arr[k]) * 1e6 for k in range(len(cv))])
            g_c = np.clip(g_prog, 0.0, g_max) if np.isfinite(g_max) else g_prog
            clipped = (np.abs(g_c - g_prog) > 1e-9) | ~np.isfinite(g_prog)
            g_c = np.where(np.isfinite(g_c), g_c, 0.0)
            res = {}
            for tn, a in a_of.items():
                v_sl_t = vsw_nom + d_v / a
                b_drift = np.vstack([cv[k].v_at(a * g_c[k] * 1e-6 * (v_sl - vs_arr[k])) for k in range(len(cv))])
                b_sup = np.vstack([cv[k].v_at(a * g_c[k] * 1e-6 * (v_sl_t - vs_arr[k])) for k in range(len(cv))])
                b_ideal = np.vstack([cv[k].v_at(g_c[k] * 1e-6 * (v_sl - vs_arr[k])) for k in range(len(cv))])
                for node, (mp, bv) in {"none": (m0, b_drift), "dac_remap": (maps_a[tn], b_drift),
                                       "supply": (m0, b_sup), "ideal": (m0, b_ideal)}.items():
                    x_eff, oor = mp.inverse(bv)
                    e = LSB_PER_X * (x_eff - theta[None, :])
                    valid = (~clipped) & (~oor) & np.isfinite(bv)
                    res[(tn, node)] = (e, valid, clipped)
            return res

        # zero-mismatch check C2
        res0 = evaluate([nom] * 4, np.full(4, vsw_nom), np.inf)
        worst = max(float(np.nanmax(np.abs(res0[(tn, nd)][0]))) for tn in times for nd in ("dac_remap", "supply", "ideal"))
        checks["C2"][cid] = {"max_abs_lsb": worst, "pass": worst <= 0.001}
        if worst > 0.001:
            failed.append(f"C2 {cid}")
        arrays = {"theta": theta, "b_target": b_target, "vsw_mc": vsw_mc}
        for s in SETTINGS:
            if curves[s] is None:
                continue
            cv = curves[s][1:]
            vs_arr = {"pdk": vsw_mc, "div_only": np.full(n, vsw_nom), "pd_only": vsw_mc}[s]
            for variant, gmax in (("noclip", np.inf), ("clip25", circ["g_max_secondary_us"])):
                res = evaluate(cv, vs_arr, gmax)
                for (tn, node), (e, valid, clipped) in res.items():
                    st = summarize(e, valid)
                    frac_invalid = 1.0 - float(np.mean(valid))
                    summary.append({"condition": cid, "set": c["set"], "setting": s, "variant": variant, "node": node,
                                    "time": tn, **st, "invalid_frac": frac_invalid, "clip_frac": float(np.mean(clipped))})
                    if variant == "noclip":
                        sim_index[(cid, s, node, tn)] = st["rms"]
                        if node != "none" and frac_invalid > 0.01:
                            failed.append(f"C5 {cid} {s} {node} {tn}: {100 * frac_invalid:.2f} % invalid pairs")
                        if node == "ideal":
                            w = float(np.nanmax(np.abs(e[valid]))) if valid.any() else np.nan
                            checks["C3"][f"{cid}/{s}/{tn}"] = {"max_abs_lsb": w, "pass": bool(np.isfinite(w) and w <= 0.001)}
                            if not (np.isfinite(w) and w <= 0.001):
                                failed.append(f"C3 {cid} {s} {tn}")
                        for j, th in enumerate(theta):
                            ej = e[valid[:, j], j]
                            profile.append({"condition": cid, "setting": s, "node": node, "time": tn,
                                            "theta": round(float(th), 6),
                                            "sim_rms": float(np.sqrt(np.mean(ej ** 2))) if ej.size else np.nan})
                        arrays[f"err_{s}_{node}_{tn}"] = e.astype(np.float32)
        np.savez_compressed(run / f"arrays_{cid}.npz", **arrays)
        log(f"{cid}: Monte Carlo done in {time.time() - t0:.1f} s; V_sw,nom {vsw_nom:.4f} V, sigma {1e3 * sig:.2f} mV")

    # C6 ordering
    tt = vsw_nom_by.get(("pd_min", "tt", 27))
    ordering = {}
    if tt is not None:
        ss, ff = vsw_nom_by.get(("pd_min", "ss", 27)), vsw_nom_by.get(("pd_min", "ff", 27))
        cold, hot = vsw_nom_by.get(("pd_min", "tt", -40)), vsw_nom_by.get(("pd_min", "tt", 85))
        if ss is not None and ff is not None:
            ordering["corner"] = bool(ss > tt > ff)
        if cold is not None and hot is not None:
            ordering["temperature"] = bool(cold > tt > hot)
    else:
        # V_sw at tt / 27 C from the predict phase when no tt condition with pd_min was simulated here
        for cid_, e_ in preds.items():
            cc = e_["condition"]
            if (cc["pulldown"], cc["corner"], cc["temp_c"]) == ("pd_min", "tt", 27):
                tt = e_["vsw_nom_V"]
    checks["C6"] = {"vsw_nom": {f"{k[0]}/{k[1]}/{k[2]}C": v for k, v in vsw_nom_by.items()}, "ordering": ordering,
                    "pass": all(ordering.values()) if ordering else True}
    if ordering and not all(ordering.values()):
        failed.append("C6 ordering")

    checks["failed"] = failed
    checks["all_pass"] = not failed
    write_json(run / "checks.json", checks)
    write_csv(run / "summary.csv", summary)
    write_csv(run / "profile.csv", profile)

    # comparison with the frozen predictions
    for cid, entry in preds.items():
        if not entry["feasible"] or (smoke and cid not in cfg["smoke"]["conditions"]):
            continue
        for tn in times:
            pr = entry["predictions"][tn]["rms_lsb"]
            for key, pv in pr.items():
                s, node = key.split("|")
                sv = sim_index.get((cid, s, node, tn))
                if sv is None:
                    continue
                tolv = max(0.15 * sv, 0.02)
                compare.append({"condition": cid, "set": entry["condition"]["set"], "time": tn, "setting": s,
                                "node": node, "pred_rms_lsb": pv, "sim_rms_lsb": sv, "abs_diff_lsb": abs(pv - sv),
                                "rel_diff": (pv - sv) / sv if sv > 0 else np.nan, "tolerance_lsb": tolv,
                                "pass": bool(abs(pv - sv) <= tolv)})
    write_csv(run / "compare.csv", compare)

    if failed:
        log("CHECK FAILURE -> stop before criteria: " + "; ".join(failed))
        info["status"] = "stopped_by_checks"
        info["end_jst"] = now_jst().isoformat(timespec="seconds")
        write_json(run / "run_info.json", info)
        return 2
    log("checks C1-C7 passed")
    if not smoke:
        decision = decide(compare)
        write_json(run / "decision.json", decision)
        log(f"decision: A1 {decision['A1']['pass_rate']:.0%} ({'pass' if decision['A1']['pass'] else 'fail'}), "
            f"A2 {'pass' if decision['A2']['pass'] else 'fail'}, A3 {decision['A3_secondary']['pass_rate']:.0%}; "
            f"laws validated: {decision['laws_validated']}")
    info["status"] = "complete"
    info["end_jst"] = now_jst().isoformat(timespec="seconds")
    write_json(run / "run_info.json", info)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=("predict", "mc"), required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--predictions")
    ap.add_argument("--config", default=str(ROOT / "experiments" / "E2_config.json"))
    args = ap.parse_args()
    cfg_path = Path(args.config).resolve()
    cfg = json.loads(cfg_path.read_text())
    if args.phase == "predict":
        return phase_predict(cfg, cfg_path, args.smoke)
    if not args.predictions:
        sys.exit("--predictions is required for the mc phase")
    return phase_mc(cfg, cfg_path, args.smoke, (ROOT / args.predictions) if not Path(args.predictions).is_absolute() else Path(args.predictions))


if __name__ == "__main__":
    sys.exit(main())
