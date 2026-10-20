"""E3b: upper-bound branch with the inverter -- prospective test of the first-order laws (sky130, PDK mismatch).

Frozen plan: experiments/E3b_config.json (Chinese description: docs/E3b_protocol.md).

Phase 1 (predict): nominal and +-1 sigma single-device / single-inverter simulations only -> predictions.json
    python experiments/E3b_upper_branch.py --phase predict [--smoke]
Phase 2 (mc): Monte Carlo with the PDK mismatch model, compared with the frozen predictions
    python experiments/E3b_upper_branch.py --phase mc --predictions results/E3b_predict_<stamp>/predictions.json [--smoke]

Outputs (never overwritten): results/E3b_predict_<stamp>[_smoke]/ and results/E3b_<stamp>[_smoke]/
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

from pcm_acam.branch_post import IVCurve, bound, g_of_x, switching_voltage  # noqa: E402
from pcm_acam.inverter import input_at_output, transfer, upper_branch_bounds, vy_star_sensitivity  # noqa: E402
from pcm_acam.io_utils import environment_info, make_logger, make_run_dir, now_jst, sha256_file, write_json  # noqa: E402
from pcm_acam.law_predict import divider_sensitivity, eps_sensitivities  # noqa: E402
from pcm_acam.spice_iv import MM_KEYS, iv_sweep, ngspice_version  # noqa: E402

NODES = ("none", "dac_remap", "supply", "ideal")
SETTINGS = ("pdk", "div_only", "y_only")
LSB_PER_X = 15.0
I_FLOOR = 1e-9


def lib_path(cfg: dict) -> Path:
    return ROOT / cfg["models"]["subset_dir"] / cfg["models"]["lib"]


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
    files = [cfg_path, Path(__file__), ROOT / "src/pcm_acam/inverter.py", ROOT / "src/pcm_acam/law_predict.py",
             ROOT / "src/pcm_acam/spice_iv.py", ROOT / "src/pcm_acam/branch_post.py", ROOT / "src/pcm_acam/sky130_models.py"]
    files += sorted((ROOT / cfg["models"]["subset_dir"]).glob("*"))
    return {str(p.relative_to(ROOT)): sha256_file(p) for p in files}


def conditions(cfg: dict, smoke: bool) -> List[dict]:
    return [c for c in cfg["conditions"] if (not smoke) or c["id"] in cfg["smoke"]["conditions"]]


def e_rm1_sq(d_v: float, a: float, sigma: float, n: int = 4001) -> float:
    """E[(r - 1)^2], r = (D - a delta)/(D - delta), delta ~ N(0, sigma^2); trapezoid on [-6 sigma, 6 sigma],
    delta >= 0.95 D dropped and the weights renormalized."""
    delta = np.linspace(-6.0 * sigma, 6.0 * sigma, n)
    w = np.exp(-0.5 * (delta / sigma) ** 2)
    keep = delta < 0.95 * d_v
    delta, w = delta[keep], w[keep]
    r = (d_v - a * delta) / (d_v - delta)
    return float(np.trapezoid(w * (r - 1.0) ** 2, delta) / np.trapezoid(w, delta))


def predict_upper(curves, theta, g_lo, g_hi, d_v, a, sigma_y) -> Dict[str, float]:
    dg = g_hi - g_lo
    g = g_lo + theta * dg
    i_t = g * 1e-6 * d_v
    s1 = eps_sensitivities(curves, i_t)
    s2 = eps_sensitivities(curves, a * i_t)
    var_div = sum((np.asarray(s2[k]) - np.asarray(s1[k])) ** 2 for k in MM_KEYS)
    var_lam = (np.asarray(s2["lambda"]) - np.asarray(s1["lambda"])) ** 2 * sigma_y ** 2
    var_sup = np.full_like(theta, e_rm1_sq(d_v, a, sigma_y))
    w2 = (g / dg) ** 2
    ms = {"pdk|dac_remap": w2 * (var_div + var_lam), "pdk|supply": w2 * var_sup,
          "div_only|dac_remap": w2 * var_div, "div_only|supply": np.zeros_like(theta),
          "y_only|dac_remap": w2 * var_lam, "y_only|supply": w2 * var_sup}
    return {k: float(15.0 * np.sqrt(np.mean(v))) for k, v in ms.items()}


# ------------------------------------------------------------------ predict phase
def phase_predict(cfg: dict, cfg_path: Path, smoke: bool) -> int:
    run = make_run_dir(ROOT / "results", cfg["run_prefix"] + "_predict", smoke)
    log = make_logger(run / "run.log")
    info = {"experiment": cfg["run_prefix"], "phase": "predict", "smoke": smoke,
            "start_jst": now_jst().isoformat(timespec="seconds"), "environment": environment_info(),
            "ngspice": ngspice_version(), "sha256": hashes(cfg_path, cfg)}
    write_json(run / "run_info.json", info)
    write_json(run / "config_used.json", cfg)
    circ, sw = cfg["circuit"], cfg["sweep"]
    lib = lib_path(cfg)
    a_of = drift_factors(cfg)
    theta = np.arange(16) / 15.0
    pd = circ["pulldown"]
    i_sw = 100e-9 * pd["w_um"] / pd["l_um"]
    out, rows = {}, []
    cache = {}
    for c in conditions(cfg, smoke):
        t0 = time.time()
        inv = circ["inverters"][c["inverter"]]
        if c["inverter"] not in cache:
            cache[c["inverter"]] = vy_star_sensitivity(lib, inv, pd, circ["vdd"], i_sw, sw)
        ys = cache[c["inverter"]]
        vy = ys["vy_star_nom"]
        d_v = c["d_v"]
        v_sl = vy + d_v
        vg, cur, curves = divider_sensitivity(lib, circ["divider_device"], c["w_div_um"], c["l_div_um"], vy, sw,
                                              i_floor=I_FLOOR)
        nom = curves[0]
        b30 = float(bound(nom, np.array([30.0]), v_sl, vy)[0])
        entry = {"condition": c, "vy_star_nom_V": vy, "v_sl_nom_V": v_sl, "y_sensitivity": ys,
                 "bound_at_30uS_V": b30 if np.isfinite(b30) else None, "feasible": bool(np.isfinite(b30) and b30 <= 1.75),
                 "nominal_ln_i_sha256": hashlib.sha256(np.ascontiguousarray(nom.ln_i).tobytes()).hexdigest(),
                 "predictions": {}}
        np.save(run / f"nominal_curve_{c['id']}.npy", np.vstack([cur[0], vg]))
        for tn, a in a_of.items():
            pr = predict_upper(curves, theta, circ["g_lo_us"], circ["g_hi_us"], d_v, a, ys["sigma_delta"])
            entry["predictions"][tn] = pr
            for k, v in pr.items():
                s, node = k.split("|")
                rows.append({"condition": c["id"], "time": tn, "setting": s, "node": node, "pred_rms_lsb": v,
                             "sigma_delta_y_mV": 1e3 * ys["sigma_delta"]})
        out[c["id"]] = entry
        log(f"{c['id']}: V_Y* {vy:.4f} V, sigma_dY {1e3 * ys['sigma_delta']:.2f} mV, gain {ys['inverter_gain']:.1f}, "
            f"bound(30 uS) {b30:.3f} V, {time.time() - t0:.1f} s")
    pred_path = run / "predictions.json"
    with open(pred_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    write_csv(run / "predictions.csv", rows)
    info["end_jst"] = now_jst().isoformat(timespec="seconds")
    info["predictions_sha256"] = sha256_file(pred_path)
    write_json(run / "run_info.json", info)
    log(f"predictions written; sha256 {info['predictions_sha256']}")
    return 0


# ------------------------------------------------------------------ decision
def decide(compare: List[dict]) -> dict:
    sel = [r for r in compare if r["time"] == "1y" and r["setting"] == "pdk" and r["node"] in ("dac_remap", "supply")]
    a1_rate = float(np.mean([r["pass"] for r in sel]))
    a2_rows = []
    for cid in sorted({r["condition"] for r in sel}):
        pr = {r["node"]: r for r in sel if r["condition"] == cid}
        pdv, psv = pr["dac_remap"]["pred_rms_lsb"], pr["supply"]["pred_rms_lsb"]
        sdv, ssv = pr["dac_remap"]["sim_rms_lsb"], pr["supply"]["sim_rms_lsb"]
        clear = abs(pdv - psv) >= 0.2 * max(pdv, psv)
        a2_rows.append({"condition": cid, "pred_better": "dac_remap" if pdv < psv else "supply",
                        "sim_better": "dac_remap" if sdv < ssv else "supply", "clear_prediction": bool(clear),
                        "agree": bool((pdv < psv) == (sdv < ssv))})
    a2 = all(r["agree"] for r in a2_rows if r["clear_prediction"])
    sel3 = [r for r in compare if r["time"] == "1y" and ((r["setting"] == "div_only" and r["node"] == "dac_remap")
                                                        or (r["setting"] == "y_only" and r["node"] in ("dac_remap", "supply")))]
    a3_rate = float(np.mean([r["pass"] for r in sel3]))
    return {"A1": {"pass_rate": a1_rate, "pass": bool(a1_rate >= 0.9), "n": len(sel),
                   "failures": [f"{r['condition']}/{r['node']}" for r in sel if not r["pass"]]},
            "A2": {"pass": bool(a2), "n_clear": sum(r["clear_prediction"] for r in a2_rows), "rows": a2_rows},
            "A3_secondary": {"pass_rate": a3_rate, "pass": bool(a3_rate >= 0.9), "n": len(sel3),
                             "failures": [f"{r['condition']}/{r['setting']}/{r['node']}" for r in sel3 if not r["pass"]]},
            "laws_validated": bool(a1_rate >= 0.9 and a2)}


# ------------------------------------------------------------------ Monte Carlo phase
def phase_mc(cfg: dict, cfg_path: Path, smoke: bool, pred_path: Path) -> int:
    run = make_run_dir(ROOT / "results", cfg["run_prefix"], smoke)
    log = make_logger(run / "run.log")
    preds = json.loads(pred_path.read_text())
    info = {"experiment": cfg["run_prefix"], "phase": "mc", "smoke": smoke,
            "start_jst": now_jst().isoformat(timespec="seconds"), "environment": environment_info(),
            "ngspice": ngspice_version(), "sha256": hashes(cfg_path, cfg),
            "predictions_file": str(pred_path.relative_to(ROOT)), "predictions_sha256": sha256_file(pred_path)}
    write_json(run / "run_info.json", info)
    write_json(run / "config_used.json", cfg)
    log(f"{cfg['run_prefix']} Monte Carlo phase; predictions {pred_path.relative_to(ROOT)} sha256 {info['predictions_sha256']}")
    circ, sw = cfg["circuit"], cfg["sweep"]
    lib = lib_path(cfg)
    a_of = drift_factors(cfg)
    times = list(a_of)
    theta = np.arange(16) / 15.0
    g_lo, g_hi = circ["g_lo_us"], circ["g_hi_us"]
    dg = g_hi - g_lo
    g_t = g_of_x(theta, g_lo, g_hi)
    pd = circ["pulldown"]
    i_sw = 100e-9 * pd["w_um"] / pd["l_um"]
    mmc = cfg["mismatch"]
    rng = np.random.default_rng([mmc["base_seed"], mmc["rng_stream"]])
    draws = {k: rng.standard_normal((mmc["n_instances"], 4)) for k in ("div", "pd", "invn", "invp")}
    n = cfg["smoke"]["n_instances"] if smoke else mmc["n_instances"]
    mm = {k: [{q: float(v) for q, v in zip(MM_KEYS, r)} for r in z[:n]] for k, z in draws.items()}

    checks: Dict[str, dict] = {k: {} for k in ("C1", "C2", "C3", "C4", "C5", "C7")}
    failed: List[str] = []
    summary, compare, profile = [], [], []
    sim_index: Dict[tuple, float] = {}
    # pass A shared by all conditions with the same inverter: pull-down and inverter Monte Carlo
    pass_a: Dict[str, dict] = {}
    for c in conditions(cfg, smoke):
        cid = c["id"]
        entry = preds[cid]
        if not entry["feasible"]:
            log(f"{cid}: infeasible -> skipped")
            continue
        t0 = time.time()
        inv = circ["inverters"][c["inverter"]]
        if c["inverter"] not in pass_a:
            vg, cur = iv_sweep(lib, pd["device"], pd["w_um"], pd["l_um"], [circ["vdd"]] * (n + 1), [None] + mm["pd"],
                               sw["vg_start"], sw["vg_stop"], sw["vg_step"])
            vsw = np.array([switching_voltage(vg, cur[k], i_sw) for k in range(n + 1)])
            y, z = transfer(lib, inv, [None] + mm["invn"], [None] + mm["invp"], circ["vdd"], sw)
            vy = np.array([input_at_output(y, z[k], vsw[k]) for k in range(n + 1)])
            pass_a[c["inverter"]] = {"vsw": vsw, "vy": vy}
        vsw, vy_all = pass_a[c["inverter"]]["vsw"], pass_a[c["inverter"]]["vy"]
        vy_nom, vy_mc = float(vy_all[0]), vy_all[1:]
        if abs(vy_nom - entry["vy_star_nom_V"]) > 1e-9:
            failed.append(f"C7 {cid}: V_Y* nominal differs from the predict phase")
        sig = float(np.std(vy_mc, ddof=1))
        tol = 0.5                                     # setup check only (see config C4)
        sd_pred = entry["y_sensitivity"]["sigma_delta"]
        ok = abs(sig / sd_pred - 1) <= tol
        checks["C4"][cid] = {"sigma_mc_mV": 1e3 * sig, "sigma_pred_mV": 1e3 * sd_pred, "rel_diff": sig / sd_pred - 1,
                             "tolerance": tol, "pass": bool(ok)}
        if not ok:
            failed.append(f"C4 {cid}")
        d_v = c["d_v"]
        v_sl = vy_nom + d_v
        runs = {"pdk": ([vy_nom] + list(vy_mc), [None] + mm["div"]),
                "div_only": ([vy_nom] * (n + 1), [None] + mm["div"]),
                "y_only": ([vy_nom] + list(vy_mc), [None] * (n + 1))}
        curves = {}
        for s, (vds, mmd) in runs.items():
            vg_div, cur_b = iv_sweep(lib, circ["divider_device"], c["w_div_um"], c["l_div_um"], vds, mmd,
                                     sw["vg_start"], sw["vg_stop"], sw["vg_step"])
            try:
                curves[s] = [IVCurve.from_arrays(vg_div, cur_b[k], I_FLOOR) for k in range(n + 1)]
            except ValueError as e:
                failed.append(f"C5 {cid} {s}: {e}")
                curves[s] = None
        nom = curves["div_only"][0]
        h = hashlib.sha256(np.ascontiguousarray(nom.ln_i).tobytes()).hexdigest()
        checks["C7"][cid] = {"identical": h == entry["nominal_ln_i_sha256"], "pass": h == entry["nominal_ln_i_sha256"]}
        if h != entry["nominal_ln_i_sha256"]:
            saved = np.load(pred_path.parent / f"nominal_curve_{cid}.npy")
            diff = float(np.max(np.abs(np.log(np.maximum(saved[0], 1e-30))[-nom.ln_i.size:] - nom.ln_i)))
            checks["C7"][cid].update({"max_abs_dlnI": diff, "pass": diff <= 1e-9})
            if diff > 1e-9:
                failed.append(f"C7 {cid}")
        b_target = bound(nom, g_t, v_sl, vy_nom)

        def x_of_bound(b, a_map):
            return (nom.i_at(b) / (a_map * d_v) * 1e6 - g_lo) / dg

        # C1: nominal full upper branch (PCM resistor, divider, inverter) versus the I-V inversion
        c1 = []
        div = {"device": circ["divider_device"], "w_um": c["w_div_um"], "l_um": c["l_div_um"]}
        for vs in (v_sl, vy_nom + d_v / a_of["1y"]):
            bb = upper_branch_bounds(lib, div, inv, [2.5, 12.5, 22.5], vs, circ["vdd"], float(vsw[0]), sw)
            inv_b = bound(nom, np.array([2.5, 12.5, 22.5]), vs, vy_nom)
            c1.append(float(np.max(1e3 * np.abs(bb - inv_b))))
        checks["C1"][cid] = {"max_abs_diff_mV": max(c1), "pass": bool(max(c1) <= 0.05)}
        if max(c1) > 0.05:
            failed.append(f"C1 {cid}")

        def evaluate(cv, vy_arr):
            i_prog = np.vstack([cv[k].i_at(b_target) for k in range(len(cv))])
            res = {}
            for tn, a in a_of.items():
                v_sl_t = vy_nom + d_v / a
                cur = {"none": a * i_prog, "dac_remap": a * i_prog, "ideal": i_prog,
                       "supply": a * i_prog * ((v_sl_t - vy_arr) / (v_sl - vy_arr))[:, None]}
                for node, cn in cur.items():
                    b = np.vstack([cv[k].v_at(cn[k]) for k in range(len(cv))])
                    e = LSB_PER_X * (x_of_bound(b, a if node == "dac_remap" else 1.0) - theta[None, :])
                    res[(tn, node)] = e
            return res

        res0 = evaluate([nom] * 4, np.full(4, vy_nom))
        worst = max(float(np.nanmax(np.abs(res0[(tn, nd)]))) for tn in times for nd in ("dac_remap", "supply", "ideal"))
        checks["C2"][cid] = {"max_abs_lsb": worst, "pass": bool(worst <= 0.001)}
        if worst > 0.001:
            failed.append(f"C2 {cid}")
        arrays = {"theta": theta, "b_target": b_target, "vy_mc": vy_mc, "vsw_pd_mc": vsw[1:]}
        c3 = 0.0
        for s in SETTINGS:
            if curves[s] is None:
                continue
            vy_arr = {"pdk": vy_mc, "div_only": np.full(n, vy_nom), "y_only": vy_mc}[s]
            res = evaluate(curves[s][1:], vy_arr)
            for (tn, node), e in res.items():
                valid = np.isfinite(e)
                ev = e[valid]
                inv_frac = 1.0 - float(np.mean(valid))
                st = {"n": int(ev.size), "rms": float(np.sqrt(np.mean(ev ** 2))) if ev.size else np.nan,
                      "mean": float(np.mean(ev)) if ev.size else np.nan,
                      "max_abs": float(np.max(np.abs(ev))) if ev.size else np.nan}
                summary.append({"condition": cid, "setting": s, "node": node, "time": tn, **st, "invalid_frac": inv_frac})
                sim_index[(cid, s, node, tn)] = st["rms"]
                if node != "none" and inv_frac > 0.01:
                    failed.append(f"C5 {cid} {s} {node} {tn}: {100 * inv_frac:.2f} % invalid pairs")
                if node == "ideal":
                    c3 = max(c3, float(np.max(np.abs(ev))) if ev.size else 0.0)
                if tn == "1y":
                    for j, th in enumerate(theta):
                        ej = e[valid[:, j], j]
                        profile.append({"condition": cid, "setting": s, "node": node, "theta": round(float(th), 6),
                                        "sim_rms": float(np.sqrt(np.mean(ej ** 2))) if ej.size else np.nan})
                    arrays[f"err_{s}_{node}_{tn}"] = e.astype(np.float32)
        checks["C3"][cid] = {"max_abs_lsb": c3, "pass": bool(c3 <= 0.001)}
        if c3 > 0.001:
            failed.append(f"C3 {cid}")
        np.savez_compressed(run / f"arrays_{cid}.npz", **arrays)
        log(f"{cid}: done in {time.time() - t0:.1f} s; V_Y* {vy_nom:.4f} V, sigma {1e3 * sig:.2f} mV, C1 {max(c1):.2e} mV")

    for cid, entry in preds.items():
        if not entry["feasible"] or (smoke and cid not in cfg["smoke"]["conditions"]):
            continue
        for tn in times:
            for key, pv in entry["predictions"][tn].items():
                s, node = key.split("|")
                sv = sim_index.get((cid, s, node, tn))
                if sv is None:
                    continue
                tolv = max(0.15 * sv, 0.02)
                compare.append({"condition": cid, "time": tn, "setting": s, "node": node, "pred_rms_lsb": pv,
                                "sim_rms_lsb": sv, "rel_diff": (pv - sv) / sv if sv > 0 else np.nan,
                                "tolerance_lsb": tolv, "pass": bool(abs(pv - sv) <= tolv)})
    checks["failed"] = failed
    checks["all_pass"] = not failed
    write_json(run / "checks.json", checks)
    write_csv(run / "summary.csv", summary)
    write_csv(run / "profile.csv", profile)
    write_csv(run / "compare.csv", compare)
    if failed:
        log("CHECK FAILURE -> stop before criteria: " + "; ".join(failed[:20]))
        info["status"] = "stopped_by_checks"
        info["end_jst"] = now_jst().isoformat(timespec="seconds")
        write_json(run / "run_info.json", info)
        return 2
    log("checks passed")
    if not smoke:
        dec = decide(compare)
        write_json(run / "decision.json", dec)
        log(f"decision: A1 {dec['A1']['pass_rate']:.0%} ({dec['A1']['pass']}), A2 {dec['A2']['pass']} "
            f"({dec['A2']['n_clear']} clear), A3 {dec['A3_secondary']['pass_rate']:.0%}; validated {dec['laws_validated']}")
    info["status"] = "complete"
    info["end_jst"] = now_jst().isoformat(timespec="seconds")
    write_json(run / "run_info.json", info)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=("predict", "mc"), required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--predictions")
    ap.add_argument("--config", default=str(ROOT / "experiments" / "E3b_config.json"))
    args = ap.parse_args()
    cfg_path = Path(args.config).resolve()
    cfg = json.loads(cfg_path.read_text())
    if args.phase == "predict":
        return phase_predict(cfg, cfg_path, args.smoke)
    if not args.predictions:
        sys.exit("--predictions is required for the mc phase")
    p = Path(args.predictions)
    return phase_mc(cfg, cfg_path, args.smoke, p if p.is_absolute() else (ROOT / p))


if __name__ == "__main__":
    sys.exit(main())
