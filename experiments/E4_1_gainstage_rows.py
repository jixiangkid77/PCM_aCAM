"""E4.1: full-row transient check of the E4 row model for the gain-stage cell set (GS); check V1 of E4 repeated for GS.

Frozen plan: experiments/E4_1_config.json. I reuse the E4 machinery (instrument, cell model, forest state, sense time)
by loading experiments/E4_forest_system.py as a module, so that the cases are built exactly as the gs arm of E4 sees
them; only the case selection for the GS cell set and the netlist of the gain-stage lower branch are new here.

    python experiments/E4_1_gainstage_rows.py [--smoke]

Output (never overwritten): results/E4_1_<stamp>[_smoke]/
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pcm_acam.acam import Scheme  # noqa: E402
from pcm_acam.io_utils import environment_info, make_logger, make_run_dir, now_jst, sha256_file, write_json  # noqa: E402
from pcm_acam.ml_transient import MLSpec, header, run_meas, tran_card  # noqa: E402
from pcm_acam.pcm_model import PCMParams  # noqa: E402
from pcm_acam.row_model import crosstalk_match, map_voltage, solve_nodes, violation_match  # noqa: E402
from pcm_acam.spice_iv import mm_string, ngspice_version  # noqa: E402
from pcm_acam.trees import dataset_bundle, quantize  # noqa: E402

_spec = importlib.util.spec_from_file_location("e4_forest_system", ROOT / "experiments" / "E4_forest_system.py")
E4 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(E4)

LSB = 15.0
CHUNK = 12


def hashes(cfg_path: Path, e4cfg: dict) -> Dict[str, str]:
    files = [cfg_path, Path(__file__), ROOT / "experiments" / "E4_forest_system.py", ROOT / "experiments" / "E4_config.json"]
    files += [ROOT / f"src/pcm_acam/{m}.py" for m in ("row_model", "spice_iv", "ml_transient", "acam", "pcm_model",
                                                       "trees", "io_utils", "inverter", "branch_post")]
    files += [ROOT / e4cfg["models"]["lib"]]
    return {str(p.relative_to(ROOT)): sha256_file(p) for p in files}


def build_cases(cfg: dict, e4cfg: dict, inst, per_group: int, log) -> List[dict]:
    """Violation-matched (sample, row) pairs of the GS cell set with model leakage S in [0.3, 3], as in V1."""
    c = cfg["cases"]
    names = list(e4cfg["datasets"])
    b = dataset_bundle(c["dataset"], c["split"])
    rows, x = b["rows"], b["x_te"]
    k = rows.cons_thr.size
    theta = quantize(rows.cons_thr, e4cfg["bits"])
    kind = rows.cons_kind
    p = PCMParams.from_dict(e4cfg["pcm_model"])
    rep = Scheme(name="replica", kind="absolute", devices=1, g_lo=e4cfg["circuit"]["g_lo_us"],
                 g_hi=e4cfg["circuit"]["g_hi_us"])
    n_pool = e4cfg["pools"]["n_instances"]
    draws, idx = E4.forest_state(e4cfg, names.index(c["dataset"]), c["split"], c["seed"], k, n_pool)
    times = list(e4cfg["times_inference_s"])
    pts = []
    for tlab, node in c["points"]:
        t_inf = 0.0 if tlab == "t0" else float(e4cfg["times_inference_s"][tlab])
        pts.append((tlab, 0 if tlab == "t0" else times.index(tlab), t_inf, node))
    n_feat = x.shape[1]
    w0 = inst.chains["L6T"].w0()
    cases = []
    for did in c["designs"]:
        d = next(dd for dd in e4cfg["designs"] if dd["id"] == did)
        cm = E4.CellModel(inst, e4cfg, did, c["cell_set"])
        res, _, _ = E4.theta_effs(cm, theta, kind, idx, draws, c["condition"], pts, p, rep)
        for tlab, node, a_hat, th, g_now in res:
            m_v = violation_match(x, rows, th)
            lo_st, up_st = cm.branch_states(g_now, idx, node, a_hat)
            _, s, (n_idx, r_idx), (pair, kk_all, w_all) = crosstalk_match(x, rows, th, m_v, lo_st, up_st, n_feat, w0)
            sel = np.where((s >= 0.3) & (s <= 3.0))[0]
            log(f"{did} {tlab} {node}: {s.size} matched pairs, {sel.size} with S in [0.3, 3]")
            if sel.size == 0:
                continue
            order = sel[np.argsort(np.log(s[sel]))]
            pick = order[np.unique(np.round(np.linspace(0, order.size - 1, min(per_group, order.size))).astype(int))]
            gdiv, d_eff, m = cm.node_setting(node, a_hat)
            for q in pick:
                bounds = []
                for kk, ww in zip(kk_all[pair == q], w_all[pair == q]):
                    xv = float(x[n_idx[q], rows.cons_feat[kk]])
                    v_dl = float(map_voltage(cm.map_curve, np.array([xv]), m, cm.d_v, cm.g_lo, cm.g_hi)[0])
                    bounds.append({"kind": int(kind[kk]), "feat": int(rows.cons_feat[kk]), "x": xv,
                                   "theta_eff": float(th[kk]), "g_us": float(g_now[kk] / gdiv), "idx": int(idx[kk]),
                                   "v_dl": v_dl, "w_model": float(ww)})
                cases.append({"design": did, "time": tlab, "node": node, "a_hat": float(a_hat), "sample": int(n_idx[q]),
                              "row": int(r_idx[q]), "S": float(s[q]), "n_feat": n_feat, "bounds": bounds,
                              "v_sl_lo": cm.v_sp_nom[0] + d_eff, "v_sl_hi": cm.v_sp_nom[1] + d_eff,
                              "w_div_um": d["w_div_um"], "l_div_um": d["l_div_um"]})
    return cases


def netlist(e4cfg: dict, lib: Path, cases: List[dict], z: Dict[str, np.ndarray], spec: MLSpec, t_stop: float,
            n_feat: int) -> Tuple[List[str], List[str]]:
    c = e4cfg["circuit"]
    inv = c["inverter"]
    pd = f"sky130_fd_pr__{c['pulldown']['device']} w={c['pulldown']['w_um']:.6g} l={c['pulldown']['l_um']:.6g}"
    p_inv = f"sky130_fd_pr__{inv['p_dev']} w={inv['wp']:.6g} l={inv['lp']:.6g}"
    n_inv = f"sky130_fd_pr__{inv['n_dev']} w={inv['wn']:.6g} l={inv['ln']:.6g}"

    def mm(key: str, q: int) -> str:
        return mm_string(E4.mm_of(z[key][q]))

    lines = header(lib, "E4.1 full-row transients, gain-stage cell set") + [f"vdd vdd 0 {c['vdd']:.9g}"]
    names: List[str] = []
    for i, cs in enumerate(cases):
        dv = f"sky130_fd_pr__{c['divider_device']} w={cs['w_div_um']:.6g} l={cs['l_div_um']:.6g}"
        lines += [f"vsllo{i} sllo{i} 0 {cs['v_sl_lo']:.12g}", f"vslhi{i} slhi{i} 0 {cs['v_sl_hi']:.12g}",
                  f"cml{i} ml{i} 0 {spec.c_lumped(n_feat):.9g}"]
        for j, b in enumerate(cs["bounds"]):
            q = b["idx"] - 1
            lines.append(f"vdl{i}_{j} dl{i}_{j} 0 {b['v_dl']:.12g}")
            r = 1.0 / (max(b["g_us"], 1e-6) * 1e-6)
            gate = f"z{i}_{j}"
            if b["kind"] == 0:
                # lower bound: X -> INV_S (b1) -> W -> INV_S (b2) -> Z -> pull-down
                lines += [f"rp{i}_{j} sllo{i} x{i}_{j} {r:.12g}",
                          f"xd{i}_{j} x{i}_{j} dl{i}_{j} 0 0 {dv}{mm('div_lo', q)}",
                          f"xb1p{i}_{j} w{i}_{j} x{i}_{j} vdd vdd {p_inv}{mm('b1p', q)}",
                          f"xb1n{i}_{j} w{i}_{j} x{i}_{j} 0 0 {n_inv}{mm('b1n', q)}",
                          f"xb2p{i}_{j} z{i}_{j} w{i}_{j} vdd vdd {p_inv}{mm('b2p', q)}",
                          f"xb2n{i}_{j} z{i}_{j} w{i}_{j} 0 0 {n_inv}{mm('b2n', q)}",
                          f"xpd{i}_{j} ml{i} z{i}_{j} 0 0 {pd}{mm('pd_lo', q)}"]
            else:
                # upper bound: Y -> INV_S -> Z -> pull-down (as in the 6T2M cell)
                lines += [f"rp{i}_{j} slhi{i} y{i}_{j} {r:.12g}",
                          f"xd{i}_{j} y{i}_{j} dl{i}_{j} 0 0 {dv}{mm('div_hi', q)}",
                          f"xip{i}_{j} z{i}_{j} y{i}_{j} vdd vdd {p_inv}{mm('up', q)}",
                          f"xin{i}_{j} z{i}_{j} y{i}_{j} 0 0 {n_inv}{mm('un', q)}",
                          f"xpd{i}_{j} ml{i} z{i}_{j} 0 0 {pd}{mm('pd_hi', q)}"]
            lines.append(f".meas tran g{i}_{j} find v({gate}) at=2e-9")
            names.append(f"g{i}_{j}")
        for j in range(2 * n_feat - len(cs["bounds"])):
            lines.append(f"xpo{i}_{j} ml{i} 0 0 0 {pd}")
        lines += [f".ic v(ml{i})={spec.v_pre:.9g}", f".meas tran tc{i} when v(ml{i})={spec.v_ref:.9g} fall=1"]
    return lines + [tran_card(spec, t_stop)], names


def model_gates(e4cfg: dict, inst, cs: dict, cell_set: str) -> List[float]:
    """Model gate voltage of every constrained pull-down of a case (diagnosis only)."""
    cm = E4.CellModel(inst, e4cfg, cs["design"], cell_set)
    out = []
    for b in cs["bounds"]:
        k = b["kind"]
        v_sl = cs["v_sl_lo"] if k == 0 else cs["v_sl_hi"]
        idx = np.array([b["idx"]])
        node = solve_nodes(cm.surf, cm.pool[k], idx, np.array([b["g_us"]]), np.array([v_sl]), np.array([b["v_dl"]]))
        out.append(float(cm.chain[k].gate(idx, node)[0]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "experiments" / "E4_1_config.json"))
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    cfg_path = Path(args.config).resolve()
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    e4path = ROOT / cfg["inputs"]["e4_plan"]["path"]
    ipath = ROOT / cfg["inputs"]["instrument"]["path"]
    for pth, key in ((e4path, "e4_plan"), (ipath, "instrument")):
        if sha256_file(pth) != cfg["inputs"][key]["sha256"]:
            raise SystemExit(f"SHA-256 of {pth} differs from the plan")
    e4cfg = json.loads(e4path.read_text(encoding="utf-8"))
    run = make_run_dir(ROOT / "results", cfg["run_prefix"], args.smoke)
    log = make_logger(run / "run.log")
    info = {"experiment": "E4.1", "smoke": args.smoke, "start_jst": now_jst().isoformat(timespec="seconds"),
            "environment": environment_info(), "ngspice": ngspice_version(), "sha256": hashes(cfg_path, e4cfg)}
    write_json(run / "run_info.json", info)
    write_json(run / "config_used.json", cfg)
    t_start = time.time()
    inst = E4.Instrument.load(ipath)
    lib = ROOT / e4cfg["models"]["lib"]
    per_group = cfg["smoke"]["per_group"] if args.smoke else cfg["cases"]["per_group"]
    cases = build_cases(cfg, e4cfg, inst, per_group, log)
    n_feat = cases[0]["n_feat"]
    m = e4cfg["ml"]
    pdc = e4cfg["circuit"]["pulldown"]
    spec = MLSpec(v_pre=m["v_pre"], v_ref=m["v_ref"], c_sa=m["c_sa_f"], c_wire=m["c_wire_per_cell_f"],
                  pd_w=pdc["w_um"], pd_l=pdc["l_um"])
    t_s = E4.t_s_of(e4cfg, lib, spec, inst.vsw_nom, n_feat)
    w0 = inst.chains["L6T"].w0()
    log(f"{len(cases)} cases; N = {n_feat}; t_s(N) = {1e9 * t_s:.3f} ns; w0 = {w0:.4e}")
    z = E4.pool_draws(e4cfg, e4cfg["pools"]["n_instances"])
    tc = np.full(len(cases), np.nan)
    gates_sim: List[List[float]] = []
    for s0 in range(0, len(cases), CHUNK):
        sub = cases[s0:s0 + CHUNK]
        lines, gnames = netlist(e4cfg, lib, sub, z, spec, 6.0 * t_s, n_feat)
        vals = run_meas(lines, [f"tc{i}" for i in range(len(sub))] + gnames)
        tc[s0:s0 + len(sub)] = vals[:len(sub)]
        it = iter(vals[len(sub):])
        for cs in sub:
            gates_sim.append([float(next(it)) for _ in cs["bounds"]])
        log(f"cases {s0}-{s0 + len(sub) - 1} simulated ({time.time() - t_start:.0f} s)")

    thr = 1 + (2 * n_feat - 1) * w0
    rows, bounds = [], []
    for ci, (cs, t_sim, gsim) in enumerate(zip(cases, tc, gates_sim)):
        t_pred = t_s * thr / cs["S"]
        r = float(np.log(t_sim / t_pred)) if np.isfinite(t_sim) else float("inf")
        s_lo = sum(b["w_model"] for b in cs["bounds"] if b["kind"] == 0)
        s_up = sum(b["w_model"] for b in cs["bounds"] if b["kind"] == 1)
        rows.append({"case": ci, "design": cs["design"], "time": cs["time"], "node": cs["node"], "sample": cs["sample"],
                     "row": cs["row"], "n_bounds": len(cs["bounds"]), "S": cs["S"], "S_lower_LBUF": s_lo,
                     "S_upper_UINV": s_up, "t_s_ns": 1e9 * t_s, "t_pred_ns": 1e9 * t_pred,
                     "t_sim_ns": 1e9 * t_sim if np.isfinite(t_sim) else float("nan"), "ln_ratio": r,
                     "pass": bool(abs(r) <= 0.05), "decision_pred_mismatch": bool(cs["S"] >= thr),
                     "decision_sim_mismatch": bool(np.isfinite(t_sim) and t_sim <= t_s)})
        for b, gm, gs in zip(cs["bounds"], model_gates(e4cfg, inst, cs, cfg["cases"]["cell_set"]), gsim):
            u = LSB * ((b["x"] - b["theta_eff"]) if b["kind"] == 0 else (b["theta_eff"] - b["x"]))
            bounds.append({"case": ci, "kind": b["kind"], "u_lsb": u, "w_model": b["w_model"], "gate_model_V": gm,
                           "gate_sim_2ns_V": gs, "gate_diff_mV": 1e3 * (gm - gs)})
    E4.write_csv(run / "rows.csv", rows)
    E4.write_csv(run / "bounds.csv", bounds)
    write_json(run / "cases.json", {"t_s_s": t_s, "n_feat": n_feat, "w0": w0, "cases": cases})

    frac = float(np.mean([r["pass"] for r in rows]))
    dis = [r for r in rows if r["decision_pred_mismatch"] != r["decision_sim_mismatch"]]
    by_group = {}
    for r in rows:
        key = f"{r['design']}|{r['time']}|{r['node']}"
        by_group.setdefault(key, []).append(abs(r["ln_ratio"]))
    checks = {"V1GS": {"cases": len(rows), "pass_fraction": frac, "max_abs_ln_ratio": max(abs(r["ln_ratio"]) for r in rows),
                       "max_abs_ln_ratio_by_group": {k: max(v) for k, v in by_group.items()},
                       "decision_disagreements": len(dis),
                       "decision_disagreements_near_tie": int(sum(abs(np.log(r["S"])) < 0.05 for r in dis)),
                       "t_s_ns": 1e9 * t_s, "pass": frac >= 0.9},
              "gates_2ns": {"max_abs_diff_mV": float(max(abs(b["gate_diff_mV"]) for b in bounds)),
                            "max_abs_diff_mV_lower": float(max([abs(b["gate_diff_mV"]) for b in bounds if b["kind"] == 0] or [0])),
                            "max_abs_diff_mV_upper": float(max([abs(b["gate_diff_mV"]) for b in bounds if b["kind"] == 1] or [0]))}}
    write_json(run / "checks.json", checks)
    decision = {"V1GS": {"rule": cfg["decision"]["V1GS"], "pass_fraction": frac, "cases": len(rows),
                         "outcome": "PASS" if frac >= 0.9 else "FAIL"}, "smoke": args.smoke}
    write_json(run / "decision.json", decision)
    info["end_jst"] = now_jst().isoformat(timespec="seconds")
    info["elapsed_s"] = round(time.time() - t_start, 1)
    write_json(run / "run_info.json", info)
    log(f"V1GS: {len(rows)} cases, pass fraction {frac:.3f}, largest |ln ratio| {checks['V1GS']['max_abs_ln_ratio']:.4f}, "
        f"decision disagreements {len(dis)}; {decision['V1GS']['outcome']}; {info['elapsed_s']} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
