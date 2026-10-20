"""E3a: PCM read nonlinearity at the two drift-compensation nodes (lower-bound branch, sky130, PDK mismatch).

Frozen plan: experiments/E3a_config.json (Chinese description: docs/E3a_protocol.md).

Phase 1 (predict): nominal and +-1 sigma single-device simulations only -> predictions.json
    python experiments/E3a_pcm_nonlinearity.py --phase predict [--smoke]
Phase 2 (mc): Monte Carlo with the PDK mismatch model, compared with the frozen predictions
    python experiments/E3a_pcm_nonlinearity.py --phase mc --predictions results/E3a_predict_<stamp>/predictions.json [--smoke]

Outputs (never overwritten): results/E3a_predict_<stamp>[_smoke]/ and results/E3a_<stamp>[_smoke]/
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
from pcm_acam.io_utils import environment_info, make_logger, make_run_dir, now_jst, sha256_file, write_json  # noqa: E402
from pcm_acam.law_predict import divider_sensitivity, pulldown_sensitivity  # noqa: E402
from pcm_acam.pcm_iv import (grid_label, predict_dac, predict_supply, replica_vt, rho, s_of_v,  # noqa: E402
                             v0_of_g)
from pcm_acam.spice_iv import MM_KEYS, iv_sweep, ngspice_version, run_netlist  # noqa: E402

NODES = ("none", "dac_remap", "supply_lin", "supply_rep", "ideal")
B1_CONFIGS = ((np.inf, 0.0), (0.3, 0.0), (0.3, 0.5), (0.15, -0.5))
LSB_PER_X = 15.0
I_FLOOR = 1e-9


def lib_for(cfg: dict, corner: str) -> Path:
    return ROOT / cfg["models"]["subset_dir"] / cfg["models"]["lib_pattern"].format(corner=corner)


def drift_factors(cfg: dict) -> Dict[str, float]:
    d = cfg["drift"]
    return {n: float((t / d["t0_s"]) ** (-d["nu"])) for n, t in d["times_s"].items()}


def grid_points(cfg: dict):
    pts = []
    for v in cfg["grid"]["v0_ref_v"]:
        v0 = np.inf if v == "inf" else float(v)
        if not np.isfinite(v0):
            pts.append((v0, 0.0))
        else:
            for b in cfg["grid"]["beta"]:
                pts.append((v0, float(b)))
    return pts


def write_csv(path: Path, rows: List[dict]) -> None:
    if rows:
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)


def hashes(cfg_path: Path, cfg: dict) -> Dict[str, str]:
    files = [cfg_path, Path(__file__), ROOT / "src/pcm_acam/pcm_iv.py", ROOT / "src/pcm_acam/law_predict.py",
             ROOT / "src/pcm_acam/spice_iv.py", ROOT / "src/pcm_acam/branch_post.py", ROOT / "src/pcm_acam/sky130_models.py"]
    files += sorted((ROOT / cfg["models"]["subset_dir"]).glob("*"))
    return {str(p.relative_to(ROOT)): sha256_file(p) for p in files}


def designs(cfg: dict, smoke: bool) -> List[dict]:
    return [d for d in cfg["designs"] if (not smoke) or d["id"] in cfg["smoke"]["designs"]]


def supply_vts(cfg: dict, d_v: float, a: float, v0_ref: float, beta: float) -> Dict[str, float]:
    g_rep = cfg["pcm_model"]["g_ref_us"]
    v0_rep = float(v0_of_g(np.array([g_rep]), v0_ref, beta, cfg["pcm_model"]["g_ref_us"])[0])
    return {"supply_lin": d_v / a, "supply_rep": replica_vt(d_v, a, v0_rep)}


# ------------------------------------------------------------------ predict phase
def phase_predict(cfg: dict, cfg_path: Path, smoke: bool) -> int:
    run = make_run_dir(ROOT / "results", cfg["run_prefix"] + "_predict", smoke)
    log = make_logger(run / "run.log")
    info = {"experiment": cfg["run_prefix"], "phase": "predict", "smoke": smoke,
            "start_jst": now_jst().isoformat(timespec="seconds"), "environment": environment_info(),
            "ngspice": ngspice_version(), "sha256": hashes(cfg_path, cfg)}
    write_json(run / "run_info.json", info)
    write_json(run / "config_used.json", cfg)
    circ, sw, pcm = cfg["circuit"], cfg["sweep"], cfg["pcm_model"]
    a_of = drift_factors(cfg)
    theta = np.arange(16) / 15.0
    g_t = g_of_x(theta, circ["g_lo_us"], circ["g_hi_us"])
    out, rows = {}, []
    for dsg in designs(cfg, smoke):
        t0 = time.time()
        lib = lib_for(cfg, dsg["corner"])
        pd = circ["pulldowns"][dsg["pulldown"]]
        pds = pulldown_sensitivity(lib, pd["device"], pd["w_um"], pd["l_um"], circ["vdd"],
                                   100e-9 * pd["w_um"] / pd["l_um"], sw, temp=dsg["temp_c"])
        vsw_nom, d_v = pds["vsw_nom"], dsg["d_v"]
        vg, cur, curves = divider_sensitivity(lib, circ["divider_device"], dsg["w_div_um"], dsg["l_div_um"], vsw_nom, sw,
                                              temp=dsg["temp_c"], i_floor=I_FLOOR)
        nom = curves[0]
        b30 = float(bound(nom, np.array([30.0]), vsw_nom + d_v, vsw_nom)[0])
        entry = {"design": dsg, "vsw_nom_V": vsw_nom, "v_sl_nom_V": vsw_nom + d_v, "pulldown_sensitivity": pds,
                 "bound_at_30uS_V": b30 if np.isfinite(b30) else None,
                 "nominal_ln_i_sha256": hashlib.sha256(np.ascontiguousarray(nom.ln_i).tobytes()).hexdigest(),
                 "pred": {}}
        np.save(run / f"nominal_curve_{dsg['id']}.npy", np.vstack([cur[0], vg]))
        for tn, a in a_of.items():
            pdac = predict_dac(curves, theta, circ["g_lo_us"], circ["g_hi_us"], d_v, a, pds["sigma_delta"])
            entry["pred"][tn] = {}
            for v0_ref, beta in grid_points(cfg):
                lab = grid_label(v0_ref, beta)
                v0t = v0_of_g(g_t, v0_ref, beta, pcm["g_ref_us"])
                res = {"dac_remap": {"rms_lsb": pdac["rms_lsb"]}}
                for node, vt in supply_vts(cfg, d_v, a, v0_ref, beta).items():
                    ps = predict_supply(curves, theta, circ["g_lo_us"], circ["g_hi_us"], d_v, a, pds["sigma_delta"], vt, v0t)
                    res[node] = {"rms_lsb": ps["rms_lsb"], "rms_systematic_lsb": ps["rms_systematic_lsb"], "vt_V": vt,
                                 "v_sl_V": vsw_nom + vt, "rho0": ps["rho0"].tolist()}
                entry["pred"][tn][lab] = res
                for node in ("dac_remap", "supply_lin", "supply_rep"):
                    rows.append({"design": dsg["id"], "time": tn, "grid": lab, "v0_ref_V": v0_ref, "beta": beta, "node": node,
                                 "pred_rms_lsb": res[node]["rms_lsb"],
                                 "pred_systematic_lsb": res[node].get("rms_systematic_lsb", 0.0),
                                 "v_sl_V": res[node].get("v_sl_V", vsw_nom + d_v)})
        out[dsg["id"]] = entry
        log(f"{dsg['id']}: V_sw,nom {vsw_nom:.4f} V, sigma_delta {1e3 * pds['sigma_delta']:.2f} mV, bound(30 uS) {b30:.3f} V, "
            f"{time.time() - t0:.1f} s")
    pred_path = run / "predictions.json"
    with open(pred_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    write_csv(run / "predictions.csv", rows)
    info["end_jst"] = now_jst().isoformat(timespec="seconds")
    info["predictions_sha256"] = sha256_file(pred_path)
    write_json(run / "run_info.json", info)
    log(f"predictions written; sha256 {info['predictions_sha256']}")
    return 0


# ------------------------------------------------------------------ full-branch check with a nonlinear PCM
def branch_bounds(lib: Path, dev: str, w: float, l: float, temp, pcms: List[tuple], v_sl: float, vsw: float,
                  sw: dict) -> np.ndarray:
    """Bound (V_DL at V_X = vsw) of nominal branches whose PCM is I = A s(V; V0) (linear for V0 = inf)."""
    lines = ["* branches with nonlinear PCM", f'.include "{Path(lib).resolve()}"', f"vsl sl 0 {v_sl:.12g}", "vdl dl 0 0"]
    for k, (amp, v0) in enumerate(pcms):
        if np.isfinite(v0):
            expr = f"{amp:.12g}*{v0:.12g}*sinh(v(sl,n{k})/{v0:.12g})"
        else:
            expr = f"{amp:.12g}*v(sl,n{k})"
        lines.append(f"bp{k} sl n{k} i={expr}")
        lines.append(f"x{k} n{k} dl 0 0 sky130_fd_pr__{dev} w={w:.6g} l={l:.6g}")
    if temp is not None:
        lines.append(f".temp {float(temp):.6g}")
    lines += [f".dc vdl {sw['vg_start']:.6g} {sw['vg_stop']:.6g} {sw['vg_step']:.6g}", ".end"]
    vec = run_netlist("\n".join(lines) + "\n")
    vdl = vec["v(dl)"]
    out = []
    for k in range(len(pcms)):
        vx = vec[f"v(n{k})"]
        out.append(float(np.interp(-vsw, -vx, vdl)))
    return np.array(out)


def branch_vx(lib: Path, dev: str, w: float, l: float, temp, amp: float, v0: float, v_sl: float, v_dl: float) -> float:
    """Node voltage of one nominal branch at a fixed input voltage (operating point)."""
    expr = f"{amp:.12g}*{v0:.12g}*sinh(v(sl,n0)/{v0:.12g})" if np.isfinite(v0) else f"{amp:.12g}*v(sl,n0)"
    lines = ["* branch operating point", f'.include "{Path(lib).resolve()}"', f"vsl sl 0 {v_sl:.12g}",
             f"vdl dl 0 {v_dl:.12g}", f"bp0 sl n0 i={expr}",
             f"x0 n0 dl 0 0 sky130_fd_pr__{dev} w={w:.6g} l={l:.6g}"]
    if temp is not None:
        lines.append(f".temp {float(temp):.6g}")
    lines += [f".dc vdl {v_dl:.12g} {v_dl:.12g} 0.001", ".end"]
    vec = run_netlist("\n".join(lines) + "\n")
    return float(vec["v(n0)"][0])


# ------------------------------------------------------------------ decision
def decide(compare: List[dict], b1_rows: List[dict]) -> dict:
    b1_worst = max((r["max_abs_diff_from_linear_lsb"] for r in b1_rows), default=float("nan"))
    b1 = bool(b1_rows) and bool(b1_worst <= 0.001)
    sel = [r for r in compare if r["time"] == "1y" and r["node"] in ("supply_lin", "supply_rep")]
    usable = [r for r in sel if r["invalid_frac"] <= 0.01 and np.isfinite(r["pred_rms_lsb"])]
    b2_rate = float(np.mean([r["pass"] for r in usable])) if usable else float("nan")
    b2 = bool(usable) and b2_rate >= 0.9
    b3_rows = []
    by = {(r["design"], r["grid"], r["node"]): r for r in compare if r["time"] == "1y"}
    for (dsn, grid, node), r in by.items():
        if node != "supply_rep":
            continue
        rd = by[(dsn, grid, "dac_remap")]
        if r["invalid_frac"] > 0.01 or not np.isfinite(r["pred_rms_lsb"]):
            continue
        ps, pdv = r["pred_rms_lsb"], rd["pred_rms_lsb"]
        ss, sd = r["sim_rms_lsb"], rd["sim_rms_lsb"]
        clear = abs(ps - pdv) >= 0.2 * max(ps, pdv)
        b3_rows.append({"design": dsn, "grid": grid, "pred_better": "supply_rep" if ps < pdv else "dac_remap",
                        "sim_better": "supply_rep" if ss < sd else "dac_remap", "clear_prediction": bool(clear),
                        "agree": bool((ps < pdv) == (ss < sd))})
    b3 = all(r["agree"] for r in b3_rows if r["clear_prediction"])
    return {"B1": {"pass": b1, "max_abs_diff_from_linear_lsb": b1_worst},
            "B2": {"pass": bool(b2), "pass_rate": b2_rate, "n_usable": len(usable), "n_excluded": len(sel) - len(usable),
                   "failures": [f"{r['design']}/{r['grid']}/{r['node']}" for r in usable if not r["pass"]]},
            "B3": {"pass": bool(b3), "n_clear": sum(r["clear_prediction"] for r in b3_rows),
                   "disagreements": [r for r in b3_rows if not r["agree"]], "rows": b3_rows},
            "laws_validated": bool(b1 and b2 and b3)}


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
    circ, sw, pcm = cfg["circuit"], cfg["sweep"], cfg["pcm_model"]
    a_of = drift_factors(cfg)
    theta = np.arange(16) / 15.0
    g_lo, g_hi = circ["g_lo_us"], circ["g_hi_us"]
    dg = g_hi - g_lo
    g_t = g_of_x(theta, g_lo, g_hi)
    mmc = cfg["mismatch"]
    rng = np.random.default_rng([mmc["base_seed"], mmc["rng_stream"]])
    z_div_all = rng.standard_normal((mmc["n_instances"], 4))
    z_pd_all = rng.standard_normal((mmc["n_instances"], 4))
    n = cfg["smoke"]["n_instances"] if smoke else mmc["n_instances"]
    mm_div = [{k: float(v) for k, v in zip(MM_KEYS, r)} for r in z_div_all[:n]]
    mm_pd = [{k: float(v) for k, v in zip(MM_KEYS, r)} for r in z_pd_all[:n]]

    checks: Dict[str, dict] = {k: {} for k in ("C1", "C2", "C3", "C4", "C5", "C7")}
    failed: List[str] = []
    summary, compare, b1_rows, dac_rows, stress = [], [], [], [], []
    for dsg in designs(cfg, smoke):
        did = dsg["id"]
        entry = preds[did]
        t0 = time.time()
        lib = lib_for(cfg, dsg["corner"])
        pd = circ["pulldowns"][dsg["pulldown"]]
        i_sw = 100e-9 * pd["w_um"] / pd["l_um"]
        d_v = dsg["d_v"]
        # pass A: pull-down
        vg, cur = iv_sweep(lib, pd["device"], pd["w_um"], pd["l_um"], [circ["vdd"]] * (n + 1), [None] + mm_pd,
                           sw["vg_start"], sw["vg_stop"], sw["vg_step"], temp=dsg["temp_c"])
        vsw = np.array([switching_voltage(vg, cur[k], i_sw) for k in range(n + 1)])
        vsw_nom, vsw_mc = float(vsw[0]), vsw[1:]
        delta = vsw_mc - vsw_nom
        if abs(vsw_nom - entry["vsw_nom_V"]) > 1e-9:
            failed.append(f"C7 {did}: V_sw,nom differs from the predict phase")
        sig = float(np.std(vsw_mc, ddof=1))
        tol = max(0.15, 3.0 / np.sqrt(2.0 * (n - 1)))
        sd_pred = entry["pulldown_sensitivity"]["sigma_delta"]
        ok = abs(sig / sd_pred - 1) <= tol
        checks["C4"][did] = {"sigma_mc_mV": 1e3 * sig, "sigma_pred_mV": 1e3 * sd_pred, "tolerance": tol, "pass": bool(ok)}
        if not ok:
            failed.append(f"C4 {did}")
        v_sl_nom = vsw_nom + d_v
        # pass B: divider, PDK mismatch at V_DS = V_sw,k (instance 0 nominal at V_sw,nom)
        vg_div, cur_b = iv_sweep(lib, circ["divider_device"], dsg["w_div_um"], dsg["l_div_um"], [vsw_nom] + list(vsw_mc),
                                 [None] + mm_div, sw["vg_start"], sw["vg_stop"], sw["vg_step"], temp=dsg["temp_c"])
        try:
            curves = [IVCurve.from_arrays(vg_div, cur_b[k], I_FLOOR) for k in range(n + 1)]
        except ValueError as e:
            failed.append(f"C5 {did}: {e}")
            continue
        nom, cv = curves[0], curves[1:]
        h = hashlib.sha256(np.ascontiguousarray(nom.ln_i).tobytes()).hexdigest()
        checks["C7"][did] = {"identical": h == entry["nominal_ln_i_sha256"], "pass": h == entry["nominal_ln_i_sha256"]}
        if h != entry["nominal_ln_i_sha256"]:
            saved = np.load(pred_path.parent / f"nominal_curve_{did}.npy")
            diff = float(np.max(np.abs(np.log(np.maximum(saved[0], 1e-30))[-nom.ln_i.size:] - nom.ln_i)))
            checks["C7"][did].update({"max_abs_dlnI": diff, "pass": diff <= 1e-9})
            if diff > 1e-9:
                failed.append(f"C7 {did}")
        b_target = bound(nom, g_t, v_sl_nom, vsw_nom)                     # M0(theta)
        i_prog = np.vstack([c.i_at(b_target) for c in cv])               # (n, 16) programmed switching currents
        i_prog0 = nom.i_at(b_target)                                     # nominal cell

        def x_of_bound(b, a_map):
            i0 = nom.i_at(b)
            return (i0 / (a_map * d_v) * 1e6 - g_lo) / dg

        # C1: full-branch DC with the nonlinear PCM element
        c1 = []
        a1y = a_of["1y"]
        for v0_ref, beta in ((np.inf, 0.0), (0.3, 0.0), (0.3, 0.5)):
            v0g = v0_of_g(np.array([2.5, 12.5, 22.5]), v0_ref, beta, pcm["g_ref_us"])
            vts = supply_vts(cfg, d_v, a1y, v0_ref, beta)
            for v_sl in (v_sl_nom, vsw_nom + vts["supply_lin"], vsw_nom + vts["supply_rep"]):
                pcms = [(gg * 1e-6 * d_v / float(s_of_v(d_v, v0)), float(v0)) for gg, v0 in zip([2.5, 12.5, 22.5], v0g)]
                bb = branch_bounds(lib, circ["divider_device"], dsg["w_div_um"], dsg["l_div_um"], dsg["temp_c"], pcms,
                                   v_sl, vsw_nom, sw)
                inv = np.array([float(nom.v_at(np.array([amp * float(s_of_v(v_sl - vsw_nom, v0))]))[0]) for amp, v0 in pcms])
                c1.append(float(np.nanmax(1e3 * np.abs(bb - inv))))
        checks["C1"][did] = {"max_abs_diff_mV": max(c1), "pass": bool(max(c1) <= 0.05)}
        if not max(c1) <= 0.05:
            failed.append(f"C1 {did}")

        # B1: full-branch input-remap test with the nonlinear PCM element (first 20 instances, 4 thresholds, 1 year)
        kk = list(range(min(20, n)))
        jj = [1, 5, 10, 14]
        a1y = a_of["1y"]
        fb = {}
        for v0_ref, beta in B1_CONFIGS:
            v0t = v0_of_g(g_t, v0_ref, beta, pcm["g_ref_us"])
            lines = ["* B1 full-branch input remap", f'.include "{lib.resolve()}"', f"vsl sl 0 {v_sl_nom:.12g}", "vdl dl 0 0"]
            idx = []
            for k in kk:
                for j in jj:
                    amp = a1y * i_prog[k, j] / float(s_of_v(d_v - delta[k], v0t[j]))
                    v0 = float(v0t[j])
                    expr = f"{amp:.12g}*{v0:.12g}*sinh(v(sl,n{len(idx)})/{v0:.12g})" if np.isfinite(v0) else f"{amp:.12g}*v(sl,n{len(idx)})"
                    lines.append(f"bp{len(idx)} sl n{len(idx)} i={expr}")
                    mm = mm_div[k]
                    lines.append(f"x{len(idx)} n{len(idx)} dl 0 0 sky130_fd_pr__{circ['divider_device']} w={dsg['w_div_um']:.6g} "
                                 f"l={dsg['l_div_um']:.6g} " + " ".join(f"mm_{q}={mm[q]:.9g}" for q in MM_KEYS))
                    idx.append((k, j))
            if dsg["temp_c"] is not None:
                lines.append(f".temp {float(dsg['temp_c']):.6g}")
            lines += [f".dc vdl {sw['vg_start']:.6g} {sw['vg_stop']:.6g} {sw['vg_step']:.6g}", ".end"]
            vec = run_netlist("\n".join(lines) + "\n")
            vdl = vec["v(dl)"]
            errs = np.full((len(kk), len(jj)), np.nan)
            for m, (k, j) in enumerate(idx):
                vx = vec[f"v(n{m})"]
                b = float(np.interp(-vsw_mc[k], -vx, vdl))
                errs[kk.index(k), jj.index(j)] = LSB_PER_X * (float(x_of_bound(np.array([b]), a1y)[0]) - theta[j])
            fb[grid_label(v0_ref, beta)] = errs
        lin = fb["linear"]
        # consistency of the full branch with the I-V inversion (linear PCM)
        b_iv = np.vstack([cv[k].v_at(a1y * i_prog[k]) for k in kk])
        e_iv = LSB_PER_X * (x_of_bound(b_iv, a1y) - theta[None, :])[:, jj]
        for lab, errs in fb.items():
            b1_rows.append({"design": did, "grid": lab, "n_pairs": int(errs.size),
                            "max_abs_diff_from_linear_lsb": float(np.max(np.abs(errs - lin))),
                            "max_abs_diff_from_iv_inversion_lsb": float(np.max(np.abs(errs - e_iv))),
                            "rms_lsb": float(np.sqrt(np.mean(errs ** 2)))})
        arrays = {"theta": theta, "b_target": b_target, "vsw_mc": vsw_mc}
        arrays.update({f"b1_fullbranch_{k_.replace('|', '_').replace('=', '')}": v_.astype(np.float32) for k_, v_ in fb.items()})
        c2_worst, c3_worst = 0.0, 0.0
        for tn, a in a_of.items():
            dac_lin = None
            for v0_ref, beta in grid_points(cfg):
                lab = grid_label(v0_ref, beta)
                v0t = v0_of_g(g_t, v0_ref, beta, pcm["g_ref_us"])
                vts = supply_vts(cfg, d_v, a, v0_ref, beta)
                cur_n = {"none": a * i_prog, "dac_remap": a * i_prog, "ideal": i_prog.copy()}
                cur_0 = {"none": a * i_prog0, "dac_remap": a * i_prog0, "ideal": i_prog0.copy()}
                for node, vt in vts.items():
                    fac = a * s_of_v(vt - delta[:, None], v0t[None, :]) / s_of_v(d_v - delta[:, None], v0t[None, :])
                    cur_n[node] = i_prog * fac
                    cur_0[node] = i_prog0 * a * s_of_v(vt, v0t) / s_of_v(d_v, v0t)
                for node in NODES:
                    b = np.vstack([cv[k].v_at(cur_n[node][k]) for k in range(n)])
                    a_map = a if node == "dac_remap" else 1.0
                    e = LSB_PER_X * (x_of_bound(b, a_map) - theta[None, :])
                    valid = np.isfinite(e)
                    ev = e[valid]
                    inv_frac = 1.0 - float(np.mean(valid))
                    st = {"n": int(ev.size), "rms": float(np.sqrt(np.mean(ev ** 2))) if ev.size else np.nan,
                          "mean": float(np.mean(ev)) if ev.size else np.nan,
                          "max_abs": float(np.max(np.abs(ev))) if ev.size else np.nan}
                    summary.append({"design": did, "time": tn, "grid": lab, "v0_ref_V": v0_ref, "beta": beta, "node": node,
                                    **st, "invalid_frac": inv_frac})
                    # zero-mismatch cell (C2) and ideal (C3)
                    b0 = nom.v_at(cur_0[node])
                    e0 = LSB_PER_X * (x_of_bound(b0, a_map) - theta)
                    if node in ("dac_remap", "ideal"):
                        w0 = float(np.nanmax(np.abs(e0)))
                    elif node in ("supply_lin", "supply_rep"):
                        rho0 = np.array([float(rho(0.0, vts[node], d_v, a, v0)) for v0 in v0t])
                        ref = LSB_PER_X * (g_t / dg) * (rho0 - 1.0)
                        ok0 = np.isfinite(e0)
                        w0 = float(np.max(np.abs(e0[ok0] - ref[ok0]))) if ok0.any() else 0.0
                    else:
                        w0 = 0.0
                    c2_worst = max(c2_worst, w0)
                    if node == "ideal":
                        c3_worst = max(c3_worst, float(np.nanmax(np.abs(ev))) if ev.size else 0.0)
                    if node == "dac_remap" and inv_frac > 0.01:
                        failed.append(f"C5 {did} {tn} {lab} {node}: {100 * inv_frac:.2f} % invalid pairs")
                    if node == "dac_remap":
                        if lab == "linear":
                            dac_lin = st["rms"]
                        dac_rows.append({"design": did, "time": tn, "grid": lab, "rms_lsb": st["rms"],
                                         "abs_diff_from_linear_lsb": abs(st["rms"] - dac_lin)})
                    if node in ("dac_remap", "supply_lin", "supply_rep"):
                        pr = entry["pred"][tn][lab][node]["rms_lsb"]
                        sv = st["rms"]
                        tolv = max(0.15 * sv, 0.02) if np.isfinite(sv) else np.nan
                        compare.append({"design": did, "time": tn, "grid": lab, "v0_ref_V": v0_ref, "beta": beta, "node": node,
                                        "pred_rms_lsb": pr, "sim_rms_lsb": sv,
                                        "rel_diff": (pr - sv) / sv if np.isfinite(sv) and sv > 0 else np.nan,
                                        "tolerance_lsb": tolv, "invalid_frac": inv_frac,
                                        "pass": bool(np.isfinite(pr) and np.isfinite(sv) and abs(pr - sv) <= tolv)})
                    if tn == "1y" and lab in ("linear", "V0=0.3|beta=+0", "V0=0.3|beta=+0.5", "V0=0.3|beta=-0.5"):
                        arrays[f"err_{node}_{tn}_{lab.replace('|', '_').replace('=', '')}"] = e.astype(np.float32)
        checks["C2"][did] = {"max_abs_lsb": c2_worst, "pass": bool(c2_worst <= 0.001)}
        checks["C3"][did] = {"max_abs_lsb": c3_worst, "pass": bool(c3_worst <= 0.001)}
        if c2_worst > 0.001:
            failed.append(f"C2 {did}")
        if c3_worst > 0.001:
            failed.append(f"C3 {did}")

        # voltage stress (secondary): nominal cell at the lowest level, input at the top of the range
        a1y = a_of["1y"]
        for v0_ref, beta in ((np.inf, 0.0), (0.3, 0.0)):
            v0_lo = float(v0_of_g(np.array([g_lo]), v0_ref, beta, pcm["g_ref_us"])[0])
            amp = g_lo * 1e-6 * d_v / float(s_of_v(d_v, v0_lo))
            vts = supply_vts(cfg, d_v, a1y, v0_ref, beta)
            m0_top = float(bound(nom, np.array([g_hi]), v_sl_nom, vsw_nom)[0])
            ma_top = float(nom.v_at(np.array([a1y * g_hi * 1e-6 * d_v]))[0])
            cases = [("t0", amp, v_sl_nom, m0_top, d_v), ("dac_remap_1y", a1y * amp, v_sl_nom, ma_top, d_v),
                     ("supply_lin_1y", a1y * amp, vsw_nom + vts["supply_lin"], m0_top, vts["supply_lin"]),
                     ("supply_rep_1y", a1y * amp, vsw_nom + vts["supply_rep"], m0_top, vts["supply_rep"])]
            for name, am, v_sl, v_dl, v_sp in cases:
                vx = branch_vx(lib, circ["divider_device"], dsg["w_div_um"], dsg["l_div_um"], dsg["temp_c"], am, v0_lo, v_sl, v_dl)
                stress.append({"design": did, "grid": grid_label(v0_ref, beta), "case": name, "v_sl_V": v_sl, "v_dl_top_V": v_dl,
                               "v_pcm_at_switching_point_V": v_sp, "v_pcm_at_top_input_lowest_level_V": v_sl - vx})
        np.savez_compressed(run / f"arrays_{did}.npz", **arrays)
        log(f"{did}: done in {time.time() - t0:.1f} s; V_sw,nom {vsw_nom:.4f} V, sigma {1e3 * sig:.2f} mV, C1 {max(c1):.2e} mV, "
            f"C2 {c2_worst:.1e} LSB")

    checks["failed"] = failed
    checks["all_pass"] = not failed
    write_json(run / "checks.json", checks)
    write_csv(run / "summary.csv", summary)
    write_csv(run / "compare.csv", compare)
    write_csv(run / "b1_fullbranch_dac_immunity.csv", b1_rows)
    write_csv(run / "dac_rms_by_grid.csv", dac_rows)
    write_csv(run / "voltage_stress.csv", stress)
    if failed:
        log("CHECK FAILURE -> stop before decisions: " + "; ".join(failed[:20]))
        info["status"] = "stopped_by_checks"
        info["end_jst"] = now_jst().isoformat(timespec="seconds")
        write_json(run / "run_info.json", info)
        return 2
    log("checks passed")
    if not smoke:
        dec = decide(compare, b1_rows)
        write_json(run / "decision.json", dec)
        log(f"decision: B1 {dec['B1']['pass']} (max {dec['B1']['max_abs_diff_from_linear_lsb']:.2e} LSB), "
            f"B2 {dec['B2']['pass_rate']:.0%} of {dec['B2']['n_usable']} ({dec['B2']['n_excluded']} excluded), "
            f"B3 {dec['B3']['pass']} ({dec['B3']['n_clear']} clear); validated {dec['laws_validated']}")
    info["status"] = "complete"
    info["end_jst"] = now_jst().isoformat(timespec="seconds")
    write_json(run / "run_info.json", info)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=("predict", "mc"), required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--predictions")
    ap.add_argument("--config", default=str(ROOT / "experiments" / "E3a_config.json"))
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
