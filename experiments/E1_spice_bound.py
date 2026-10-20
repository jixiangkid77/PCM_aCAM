"""E1: transistor-level (sky130, ngspice) check of the aCAM lower-bound branch.

Questions, settings, checks and decision rules are frozen in experiments/E1_config.json
(Chinese description: docs/E1_protocol.md).

Method (quasi-static, exact for a linear PCM conductance):
  1. pull-down I-V at V_DS = 1.8 V for every instance -> V_sw,k (I = I_sw = 100 nA * W/L)
  2. divider I-V at V_DS = V_sw,k for every instance
  3. bound B_k(G; V_SL) = I_div,k^{-1}( G (V_SL - V_sw,k) )
  4. threshold verify, uniform drift a, compensation nodes, input-referred error in LSB

Usage
-----
    python experiments/E1_spice_bound.py            # full run (300 instances, 9 designs)
    python experiments/E1_spice_bound.py --smoke    # 20 instances, P1 and P4, no decisions

Outputs: results/E1_<YYYYMMDD_HHMM>[_smoke]/ (never overwritten)
    config_used.json, run_info.json, checks.json, design_info.csv, summary.csv, profile.csv,
    regime.csv, sigma.csv, decision.json (full run only), arrays_<design>.npz, run.log
"""

from __future__ import annotations

import argparse
import csv
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
from pcm_acam.spice_iv import MM_KEYS, branch_dc, iv_sweep, ngspice_version, op_param_sweep  # noqa: E402

EXPERIMENT = "E1"
NODES = ("none", "dac_remap", "dac_gain", "supply", "ideal")
SETTINGS = ("none", "pdk", "div_only", "pd_only", "e02")
LSB_PER_X = 15.0


def mm_rows(z: np.ndarray) -> List[Dict[str, float]]:
    return [{k: float(v) for k, v in zip(MM_KEYS, row)} for row in z]


def drift_factors(cfg: dict) -> Dict[str, float]:
    d = cfg["drift"]
    return {name: float((t / d["t0_s"]) ** (-d["nu"])) for name, t in d["times_s"].items()}


def write_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--config", default=str(ROOT / "experiments" / "E1_config.json"))
    args = ap.parse_args()

    cfg_path = Path(args.config)
    cfg = json.loads(cfg_path.read_text())
    run_dir = make_run_dir(ROOT / "results", EXPERIMENT, args.smoke)
    log = make_logger(run_dir / "run.log")
    write_json(run_dir / "config_used.json", cfg)
    t_start = time.time()

    lib = ROOT / cfg["models"]["subset"]
    hashed = [cfg_path, Path(__file__), ROOT / "src/pcm_acam/spice_iv.py", ROOT / "src/pcm_acam/branch_post.py",
              ROOT / "src/pcm_acam/sky130_models.py"] + sorted((ROOT / "third_party/sky130_fd_pr_subset").glob("*"))
    run_info = {
        "experiment": EXPERIMENT, "smoke": args.smoke, "start_jst": now_jst().isoformat(timespec="seconds"),
        "environment": environment_info(), "ngspice": ngspice_version(),
        "sha256": {str(p.relative_to(ROOT)): sha256_file(p) for p in hashed},
    }
    write_json(run_dir / "run_info.json", run_info)
    log(f"E1 start; run dir {run_dir.name}; ngspice {run_info['ngspice']}")
    log(f"config sha256 {run_info['sha256'][str(cfg_path.relative_to(ROOT))]}")

    mm_cfg = cfg["mismatch"]
    n_full = mm_cfg["n_instances"]
    rng = np.random.default_rng([mm_cfg["base_seed"], 0])
    z_div_all = rng.standard_normal((n_full, 4))
    z_pd_all = rng.standard_normal((n_full, 4))
    n = cfg["smoke"]["n_instances"] if args.smoke else n_full
    z_div, z_pd = z_div_all[:n], z_pd_all[:n]
    designs = [d for d in cfg["designs"] if (not args.smoke) or d["id"] in cfg["smoke"]["designs"]]

    circ = cfg["circuit"]
    sw = cfg["sweep"]
    vdd = circ["vdd"]
    g_lo, g_hi, g_max = circ["g_lo_us"], circ["g_hi_us"], circ["g_max_us"]
    a_of = drift_factors(cfg)
    times = list(a_of)
    theta = np.arange(16) / 15.0
    xg = np.arange(cfg["metric"]["x_grid"][0], cfg["metric"]["x_grid"][1] + 1e-12, cfg["metric"]["x_grid"][2])
    i_floor = 1e-9  # smallest current queried is about 1e-8 A (x = -0.12, 1 year, D = 0.2 V)

    checks: Dict[str, dict] = {"C1": {}, "C2": {}, "C3": {}, "C4": {}, "C5": {}}
    failed: List[str] = []

    # ---------------- pass A: pull-downs ----------------
    pd_info: Dict[str, dict] = {}
    for pd_name in sorted({d["pulldown"] for d in designs}):
        p = circ["pulldowns"][pd_name]
        i_sw = 100e-9 * p["w_um"] / p["l_um"]
        t0 = time.time()
        vg, cur = iv_sweep(lib, p["device"], p["w_um"], p["l_um"], [vdd] * (n + 1),
                           [None] + mm_rows(z_pd), sw["vg_start"], sw["vg_stop"], sw["vg_step"])
        vsw = np.array([switching_voltage(vg, cur[k], i_sw) for k in range(n + 1)])
        vsw_nom = float(vsw[0])
        pd_info[pd_name] = {"vsw_nom": vsw_nom, "vsw_pdk": vsw[1:], "vsw_e02": vsw_nom + 0.020 * z_pd[:, 0],
                            "i_sw": i_sw}
        sig = float(np.std(vsw[1:], ddof=1))
        expect = 5.456e-3 / np.sqrt(p["w_um"] * p["l_um"])
        tol = max(cfg["checks"]["C4_sigma_tolerance"], 3.0 / np.sqrt(2.0 * (n - 1)))  # sampling error of a std
        ok = abs(sig / expect - 1) <= tol
        checks["C4"][pd_name] = {"sigma_vsw_mV": 1e3 * sig, "expected_mV": 1e3 * expect, "tolerance": tol, "pass": bool(ok)}
        if not ok:
            failed.append(f"C4 {pd_name}")
        log(f"pass A {pd_name}: V_sw,nom = {vsw_nom:.5f} V, sigma(pdk) = {1e3 * sig:.2f} mV "
            f"(expected {1e3 * expect:.2f}), {time.time() - t0:.1f} s")

    design_rows, summary_rows, profile_rows, regime_rows, sigma_rows = [], [], [], [], []
    summary_index: Dict[tuple, dict] = {}

    for dsg in designs:
        did = dsg["id"]
        pdi = pd_info[dsg["pulldown"]]
        vsw_nom, d_v = pdi["vsw_nom"], dsg["d_v"]
        v_sl_nom = vsw_nom + d_v
        dv = circ["divider"]
        L = dsg["l_div_um"]
        t0 = time.time()

        # ---------------- pass B: dividers ----------------
        runs = {
            "pdk": ([vsw_nom] + list(pdi["vsw_pdk"]), [None] + mm_rows(z_div)),
            "div_only": ([vsw_nom] * (n + 1), [None] + mm_rows(z_div)),
            "pd_only": ([vsw_nom] + list(pdi["vsw_pdk"]), [None] * (n + 1)),
            "e02": ([vsw_nom] + list(pdi["vsw_e02"]), [None] * (n + 1)),
        }
        curves: Dict[str, List[IVCurve]] = {}
        vg_div = None
        for s, (vds, mm) in runs.items():
            vg_div, cur = iv_sweep(lib, dv["device"], dv["w_um"], L, vds, mm, sw["vg_start"], sw["vg_stop"], sw["vg_step"])
            try:
                curves[s] = [IVCurve.from_arrays(vg_div, cur[k], i_floor) for k in range(n + 1)]
            except ValueError as e:
                failed.append(f"C5 {did} {s}: {e}")
                curves[s] = None
        nom = curves["div_only"][0]
        nom_cur = nom.i_at(vg_div)
        log(f"{did}: pass B done ({time.time() - t0:.1f} s); V_sw,nom {vsw_nom:.5f} V, V_SL,nom {v_sl_nom:.5f} V")

        # ---------------- nominal maps ----------------
        m0_v = bound(nom, g_of_x(xg, g_lo, g_hi), v_sl_nom, vsw_nom)
        maps = {"M0": Mapping(xg, m0_v)}
        in_unit = (xg >= 0) & (xg <= 1)
        slope, v0 = np.polyfit(g_of_x(xg[in_unit], g_lo, g_hi), m0_v[in_unit], 1)
        for tn, a in a_of.items():
            maps[f"Ma_{tn}"] = Mapping(xg, bound(nom, a * g_of_x(xg, g_lo, g_hi), v_sl_nom, vsw_nom))
            maps[f"Mg_{tn}"] = Mapping(xg, v0 + a * (m0_v - v0))
        b_target = bound(nom, g_of_x(theta, g_lo, g_hi), v_sl_nom, vsw_nom)
        span = float(b_target[-1] - b_target[0])
        design_rows.append({"design": did, "role": dsg["role"], "l_div_um": L, "d_v": d_v, "pulldown": dsg["pulldown"],
                            "vsw_nom_V": vsw_nom, "v_sl_nom_V": v_sl_nom, "bound_x0_V": float(b_target[0]),
                            "bound_x1_V": float(b_target[-1]), "span_mV": 1e3 * span, "lsb_mV": 1e3 * span / 15,
                            "dac_gain_v0_V": float(v0), "v_sl_1y_V": vsw_nom + d_v / a_of["1y"],
                            "m0_x_min": float(maps["M0"].x[0]), "m0_x_max": float(maps["M0"].x[-1])})

        # ---------------- C1: full branch vs I-V inversion ----------------
        g_chk = [2.5, 7.5, 12.5, 17.5, 22.5]
        c1 = []
        for v_sl in (v_sl_nom, vsw_nom + d_v / a_of["1y"]):
            vdl, vx = branch_dc(lib, dv["device"], dv["w_um"], L, g_chk, v_sl, sw["vg_start"], sw["vg_stop"], sw["vg_step"])
            for k, g in enumerate(g_chk):
                b_branch = float(np.interp(-vsw_nom, -vx[k], vdl))
                b_iv = float(bound(nom, np.array([g]), v_sl, vsw_nom)[0])
                c1.append(abs(b_branch - b_iv) * 1e3)
        c1max = float(max(c1))
        ok = c1max <= cfg["checks"]["C1_branch_vs_iv_mV"]
        checks["C1"][did] = {"max_abs_diff_mV": c1max, "pass": bool(ok)}
        if not ok:
            failed.append(f"C1 {did}")

        # ---------------- regime of the nominal divider ----------------
        op = op_param_sweep(lib, dv["device"], dv["w_um"], L, vsw_nom, sw["vg_start"], sw["vg_stop"], sw["vg_step"])

        def region(vgs: np.ndarray) -> List[str]:
            vth = np.interp(vgs, op["vg"], op["vth"])
            vdsat = np.interp(vgs, op["vg"], op["vdsat"])
            return ["sub" if v < t else ("triode" if vsw_nom < s_ else "sat") for v, t, s_ in zip(vgs, vth, vdsat)]

        reg_t0 = region(b_target)
        b_1y_remap = bound(nom, a_of["1y"] * g_of_x(theta, g_lo, g_hi), v_sl_nom, vsw_nom)
        reg_1y = region(b_1y_remap)
        for j, th in enumerate(theta):
            regime_rows.append({"design": did, "theta": round(float(th), 6), "vgs_t0_V": float(b_target[j]),
                                "region_t0_and_supply": reg_t0[j], "vgs_1y_dac_remap_V": float(b_1y_remap[j]),
                                "region_1y_dac_remap": reg_1y[j]})

        # ---------------- mismatch statistics (information) ----------------
        i_ref = g_of_x(0.5, g_lo, g_hi) * 1e-6 * d_v
        dvg = np.array([c.v_at(np.array([i_ref]))[0] for c in curves["div_only"][1:]]) - nom.v_at(np.array([i_ref]))[0]
        sigma_rows.append({"design": did, "sigma_div_gate_shift_mV_at_mid_current": 1e3 * float(np.std(dvg, ddof=1)),
                           "sigma_vsw_pdk_mV": 1e3 * float(np.std(pdi["vsw_pdk"], ddof=1)),
                           "pelgrom_vth0_div_mV": 1e3 * 3.356e-3 / np.sqrt(dv["w_um"] * L)})

        # ---------------- programming, drift, nodes ----------------
        arrays = {"theta": theta, "b_target": b_target, "vg_div": vg_div, "nom_ln_i": np.log(np.maximum(nom_cur, 1e-30))}
        for s in SETTINGS:
            if s == "none":
                cv = [nom] * n
                vsw = np.full(n, vsw_nom)
                off = np.zeros(n)
            else:
                if curves[s] is None:
                    continue
                cv = curves[s][1:]
                vsw = {"pdk": pdi["vsw_pdk"], "div_only": np.full(n, vsw_nom), "pd_only": pdi["vsw_pdk"],
                       "e02": pdi["vsw_e02"]}[s]
                off = 0.020 * z_div[:, 0] if s == "e02" else np.zeros(n)
            # threshold verify (conductance in uS)
            g_prog = np.vstack([cv[k].i_at(b_target - off[k]) / (v_sl_nom - vsw[k]) * 1e6 for k in range(n)])
            g_c = np.clip(g_prog, 0.0, g_max)
            clipped = (np.abs(g_c - g_prog) > 1e-9) | ~np.isfinite(g_prog)
            g_c = np.where(np.isfinite(g_c), g_c, 0.0)
            err = np.full((len(NODES), len(times), n, 16), np.nan)
            valid = np.zeros((len(NODES), len(times), n, 16), dtype=bool)
            for ti, tn in enumerate(times):
                a = a_of[tn]
                v_sl_t = vsw_nom + d_v / a
                b_drift = np.vstack([cv[k].v_at(a * g_c[k] * 1e-6 * (v_sl_nom - vsw[k])) + off[k] for k in range(n)])
                b_sup = np.vstack([cv[k].v_at(a * g_c[k] * 1e-6 * (v_sl_t - vsw[k])) + off[k] for k in range(n)])
                b_ideal = np.vstack([cv[k].v_at(g_c[k] * 1e-6 * (v_sl_nom - vsw[k])) + off[k] for k in range(n)])
                per_node = {"none": (maps["M0"], b_drift), "dac_remap": (maps[f"Ma_{tn}"], b_drift),
                            "dac_gain": (maps[f"Mg_{tn}"], b_drift), "supply": (maps["M0"], b_sup),
                            "ideal": (maps["M0"], b_ideal)}
                for ni, node in enumerate(NODES):
                    mp, bv = per_node[node]
                    x_eff, oor = mp.inverse(bv)
                    e = LSB_PER_X * (x_eff - theta[None, :])
                    v = (~clipped) & (~oor) & np.isfinite(bv)
                    err[ni, ti] = e
                    valid[ni, ti] = v
                    nan_unclipped = int(np.sum((~clipped) & ~np.isfinite(bv)))
                    if nan_unclipped:
                        failed.append(f"C5 {did} {s} {node} {tn}: {nan_unclipped} NaN bounds on unclipped pairs")
                    st = summarize(e, v)
                    row = {"design": did, "role": dsg["role"], "setting": s, "node": node, "time": tn, "a": a, **st,
                           "clip_frac": float(np.mean(clipped)), "oor_count": int(np.sum(oor & ~clipped))}
                    summary_rows.append(row)
                    summary_index[(did, s, node, tn)] = row
                    for j, th in enumerate(theta):
                        vj = v[:, j]
                        ej = e[vj, j]
                        profile_rows.append({"design": did, "setting": s, "node": node, "time": tn,
                                             "theta": round(float(th), 6), "n": int(vj.sum()),
                                             "rms": float(np.sqrt(np.mean(ej ** 2))) if ej.size else np.nan,
                                             "mean": float(np.mean(ej)) if ej.size else np.nan})
            arrays[f"err_{s}"] = err.astype(np.float32)
            arrays[f"valid_{s}"] = valid
            arrays[f"gprog_{s}"] = g_prog.astype(np.float32)
            arrays[f"vsw_{s}"] = vsw
            # C2 / C3
            if s == "none":
                worst = 0.0
                for node in ("dac_remap", "supply", "ideal"):
                    ni = NODES.index(node)
                    worst = max(worst, float(np.nanmax(np.abs(err[ni][valid[ni]]))))
                ok = worst <= cfg["checks"]["C2_zero_mismatch_lsb"]
                checks["C2"][did] = {"max_abs_lsb": worst, "pass": bool(ok)}
                if not ok:
                    failed.append(f"C2 {did}")
            ni = NODES.index("ideal")
            worst_ideal = float(np.nanmax(np.abs(err[ni][valid[ni]]))) if valid[ni].any() else np.nan
            ok = bool(np.isfinite(worst_ideal) and worst_ideal <= cfg["checks"]["C3_verify_lsb"])
            checks["C3"][f"{did}/{s}"] = {"max_abs_lsb": worst_ideal, "pass": ok,
                                          "clip_frac": float(np.mean(clipped))}
            if not ok:
                failed.append(f"C3 {did} {s}")
        np.savez_compressed(run_dir / f"arrays_{did}.npz", **arrays)
        log(f"{did}: done in {time.time() - t0:.1f} s; span {1e3 * span:.1f} mV; regions t0 "
            f"{ {r: reg_t0.count(r) for r in set(reg_t0)} }")

    checks["C5"]["failures"] = [f for f in failed if f.startswith("C5")]
    checks["failed"] = failed
    checks["all_pass"] = not failed
    write_json(run_dir / "checks.json", checks)
    write_csv(run_dir / "design_info.csv", design_rows)
    write_csv(run_dir / "summary.csv", summary_rows)
    write_csv(run_dir / "profile.csv", profile_rows)
    write_csv(run_dir / "regime.csv", regime_rows)
    write_csv(run_dir / "sigma.csv", sigma_rows)

    if failed:
        log("CHECK FAILURE -> stop before decisions: " + "; ".join(failed))
        run_info["end_jst"] = now_jst().isoformat(timespec="seconds")
        run_info["status"] = "stopped_by_checks"
        write_json(run_dir / "run_info.json", run_info)
        return 2
    log("checks C1-C5 passed")

    if not args.smoke:
        prim = [d["id"] for d in designs if d["role"] == "primary"]
        g = lambda dd, s, node: summary_index[(dd, s, node, "1y")]["rms"]  # noqa: E731
        d1a_supply = {dd: g(dd, "div_only", "supply") for dd in prim}
        d1a_dac = {dd: g(dd, "div_only", "dac_remap") for dd in prim}
        d1a = all(v <= 0.01 for v in d1a_supply.values()) and any(v >= 0.05 for v in d1a_dac.values())
        d1b_supply = {dd: g(dd, "pd_only", "supply") for dd in prim}
        d1b = any(v >= 0.05 for v in d1b_supply.values())
        pdk_dac = {dd: g(dd, "pdk", "dac_remap") for dd in prim}
        pdk_sup = {dd: g(dd, "pdk", "supply") for dd in prim}
        if all(pdk_sup[dd] < pdk_dac[dd] for dd in prim):
            d2 = "supply_better_everywhere"
        elif all(pdk_dac[dd] < pdk_sup[dd] for dd in prim):
            d2 = "dac_better_everywhere"
        else:
            d2 = "design_dependent"
        gap = max(abs(pdk_dac[dd] - pdk_sup[dd]) for dd in prim)
        d3 = "matters" if gap >= 0.25 else "minor"
        decision = {"D1a": {"pass": d1a, "div_only_supply_rms": d1a_supply, "div_only_dac_remap_rms": d1a_dac},
                    "D1b": {"pass": d1b, "pd_only_supply_rms": d1b_supply},
                    "N2_holds_at_transistor_level": bool(d1a and d1b),
                    "D2": {"class": d2, "pdk_dac_remap_rms": pdk_dac, "pdk_supply_rms": pdk_sup},
                    "D3": {"class": d3, "max_gap_lsb": gap},
                    "note": "decisions follow the frozen rules in E1_config.json; secondary results are not used"}
        write_json(run_dir / "decision.json", decision)
        log(f"decisions: N2 {'holds' if d1a and d1b else 'not confirmed'} (D1a {d1a}, D1b {d1b}); D2 {d2}; D3 {d3} (gap {gap:.3f} LSB)")

    run_info["end_jst"] = now_jst().isoformat(timespec="seconds")
    run_info["elapsed_s"] = round(time.time() - t_start, 1)
    run_info["status"] = "complete"
    write_json(run_dir / "run_info.json", run_info)
    log(f"E1 finished in {time.time() - t_start:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
