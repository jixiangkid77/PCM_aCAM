"""E3c: match-line transient sensing and row crosstalk at the two drift-compensation nodes (lower-bound branch).

Frozen plan: experiments/E3c_config.json (Chinese description: docs/E3c_protocol.md).

Phase 1 (predict): pull-down-only transients and single-branch DC operating points -> predictions.json
    python experiments/E3c_ml_transient.py --phase predict [--smoke]
Phase 2 (sim): full-branch transients of the row (cell under test and leaker cells as full branches)
    python experiments/E3c_ml_transient.py --phase sim --predictions results/E3c_predict_<stamp>/predictions.json [--smoke]

Outputs (never overwritten): results/E3c_predict_<stamp>[_smoke]/ and results/E3c_<stamp>[_smoke]/
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

from pcm_acam.branch_post import IVCurve, bound, g_of_x, switching_voltage  # noqa: E402
from pcm_acam.io_utils import environment_info, make_logger, make_run_dir, now_jst, sha256_file, write_json  # noqa: E402
from pcm_acam.law_predict import divider_sensitivity, eps_sensitivities, pulldown_sensitivity  # noqa: E402
from pcm_acam.ml_transient import MLSpec, bisect, branch_lines, crossing_times, header, row_devices, run_meas, tran_card  # noqa: E402
from pcm_acam.spice_iv import MM_KEYS, iv_sweep, ngspice_version, run_netlist  # noqa: E402

I_FLOOR = 1e-9


def write_csv(path: Path, rows: List[dict]) -> None:
    if rows:
        keys: List[str] = []
        for r in rows:
            for k in r:
                if k not in keys:
                    keys.append(k)
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)


def hashes(cfg_path: Path, cfg: dict) -> Dict[str, str]:
    files = [cfg_path, Path(__file__), ROOT / "src/pcm_acam/ml_transient.py", ROOT / "src/pcm_acam/law_predict.py",
             ROOT / "src/pcm_acam/spice_iv.py", ROOT / "src/pcm_acam/branch_post.py", ROOT / cfg["models"]["lib"]]
    return {str(p.relative_to(ROOT)): sha256_file(p) for p in files}


class Setup:
    """Quantities shared by both phases (nominal devices only)."""

    def __init__(self, cfg: dict, smoke: bool):
        self.cfg, self.smoke = cfg, smoke
        self.lib = ROOT / cfg["models"]["lib"]
        c = cfg["circuit"]
        m = cfg["ml"]
        self.spec = MLSpec(v_pre=m["v_pre"], v_ref=m["v_ref"], c_sa=m["c_sa_f"], c_wire=m["c_wire_per_cell_f"],
                           pd_w=c["pulldown"]["w_um"], pd_l=c["pulldown"]["l_um"], t_step=m["t_step_s"],
                           t_max=m["t_max_s"])
        self.g_lo, self.g_hi = c["g_lo_us"], c["g_hi_us"]
        self.dg = self.g_hi - self.g_lo
        self.d_v = c["d_v"]
        sw = cfg["sweep"]
        self.sw = sw
        pd = c["pulldown"]
        self.i_sw = 100e-9 * pd["w_um"] / pd["l_um"]
        vg, cur = iv_sweep(self.lib, c["pulldown_device"], pd["w_um"], pd["l_um"], [c["vdd"]], [None],
                           sw["vg_start"], sw["vg_stop"], sw["vg_step"])
        self.vsw = switching_voltage(vg, cur[0], self.i_sw)
        self.a = {n: float((t / cfg["drift"]["t0_s"]) ** (-cfg["drift"]["nu"])) for n, t in cfg["drift"]["times_s"].items()}
        self.designs = [d for d in cfg["designs"] if (not smoke) or d["id"] in cfg["smoke"]["designs"]]
        self.curves = {}
        for d in self.designs:
            vgd, curd = iv_sweep(self.lib, c["divider_device"], d["w_div_um"], d["l_div_um"], [self.vsw], [None],
                                 sw["vg_start"], sw["vg_stop"], sw["vg_step"])
            self.curves[d["id"]] = IVCurve.from_arrays(vgd, curd[0], I_FLOOR)
        self.thetas = np.array(cfg["smoke"]["thetas"] if smoke else cfg["thetas"], float)
        self.nodes = cfg["nodes"]
        self.rows = cfg["rows"]

    # ---- node definitions -------------------------------------------------------------------------------------
    def node_params(self, node: str):
        """(PCM conductance factor, V_SL, map factor for x_eff, kappa of the crosstalk law)."""
        if node == "t0":
            return 1.0, self.vsw + self.d_v, 1.0, 1.0
        kind, tn = node.split("_")
        a = self.a[tn]
        if kind == "dac":
            return a, self.vsw + self.d_v, a, 1.0
        if kind == "supply":
            return a, self.vsw + self.d_v / a, 1.0, a
        raise ValueError(node)

    def qs_bound(self, did: str, theta: float, node: str) -> float:
        gf, v_sl, _, _ = self.node_params(node)
        g = g_of_x(np.array([theta]), self.g_lo, self.g_hi) * gf
        return float(bound(self.curves[did], g, v_sl, self.vsw)[0])

    def x_eff(self, did: str, b: np.ndarray, node: str) -> np.ndarray:
        _, _, mapf, _ = self.node_params(node)
        i0 = self.curves[did].i_at(np.asarray(b))
        return (i0 / (mapf * self.d_v) * 1e6 - self.g_lo) / self.dg

    def leaker_vdl(self, did: str, x: float, node: str) -> float:
        """Input voltage of a leaker cell for input x: nominal map M0 (t0, supply) or Ma (dac)."""
        _, _, mapf, _ = self.node_params(node)
        g = g_of_x(np.array([x]), self.g_lo, self.g_hi) * mapf
        return float(self.curves[did].v_at(g * 1e-6 * self.d_v)[0])

    def t_stop(self, t_s: float) -> float:
        return self.cfg["ml"]["t_stop_factor"] * t_s


def leaker_vx(st: Setup, did: str, d: dict, node: str, x_j: float, theta_j: float) -> float:
    """DC node voltage of a leaker branch (PCM + divider; the pull-down gate draws no current)."""
    gf, v_sl, _, _ = st.node_params(node)
    g = float(g_of_x(np.array([theta_j]), st.g_lo, st.g_hi)[0]) * gf
    lines = header(st.lib, "leaker dc") + [f"vsl sl 0 {v_sl:.12g}"]
    lines += branch_lines(0, 0, d["w_div_um"], d["l_div_um"], "sl", g, st.leaker_vdl(did, x_j, node), "x0")
    lines += [".op", ".end"]
    vec = run_netlist("\n".join(lines) + "\n")
    return float(vec["v(x0)"][0])


def ts_for_rows(st: Setup) -> Dict[int, float]:
    out = {}
    for n in st.rows:
        def build(n=n):
            ln = header(st.lib, "t_s") + [f"vgc gc 0 {st.vsw:.12g}"] + row_devices(0, st.spec, n, "gc", [])
            return ln + [tran_card(st.spec, 5e-6)]
        out[n] = float(crossing_times(build, 1)[0])
    return out


def vsw_tr_bisect(st: Setup, n: int, t_s: float, cases: List[dict], iters: int, win: float) -> np.ndarray:
    """Pull-down-only: CUT gate voltage giving a crossing at t_s; each case may set 'mm' (CUT mismatch) and
    'leak' (list of leaker gate voltages)."""
    k = len(cases)

    def build(vals):
        ln = header(st.lib, "pull-down only")
        for i, (cs, v) in enumerate(zip(cases, vals)):
            ln.append(f"vgc{i} gc{i} 0 {v:.12g}")
            leak_nodes = []
            for j, vl in enumerate(cs.get("leak", [])):
                ln.append(f"vgl{i}_{j} gl{i}_{j} 0 {vl:.12g}")
                leak_nodes.append(f"gl{i}_{j}")
            ln += row_devices(i, st.spec, n, f"gc{i}", leak_nodes, cs.get("mm"))
        return ln + [tran_card(st.spec, st.t_stop(t_s))]

    lo = np.full(k, st.vsw - win)
    hi = np.full(k, st.vsw + win)
    return bisect(build, k, lo, hi, np.full(k, t_s), iters, increasing_means_faster=True)


def false_mismatch(st: Setup, n: int, t_s: float, leak_sets: List[List[float]]) -> np.ndarray:
    """Crossing time with the CUT pull-down off (gate 0) and the given leakers; <= t_s means the row fails."""
    k = len(leak_sets)

    def build():
        ln = header(st.lib, "false mismatch")
        for i, leak in enumerate(leak_sets):
            ln.append(f"vgc{i} gc{i} 0 0")
            nodes = []
            for j, vl in enumerate(leak):
                ln.append(f"vgl{i}_{j} gl{i}_{j} 0 {vl:.12g}")
                nodes.append(f"gl{i}_{j}")
            ln += row_devices(i, st.spec, n, f"gc{i}", nodes)
        return ln + [tran_card(st.spec, st.t_stop(t_s))]

    return crossing_times(build, k)


# ------------------------------------------------------------------ predict phase
def phase_predict(cfg: dict, cfg_path: Path, smoke: bool) -> int:
    run = make_run_dir(ROOT / "results", "E3c_predict", smoke)
    log = make_logger(run / "run.log")
    info = {"experiment": "E3c", "phase": "predict", "smoke": smoke, "start_jst": now_jst().isoformat(timespec="seconds"),
            "environment": environment_info(), "ngspice": ngspice_version(), "sha256": hashes(cfg_path, cfg)}
    write_json(run / "run_info.json", info)
    write_json(run / "config_used.json", cfg)
    st = Setup(cfg, smoke)
    it = cfg["bisection"]["iters_smoke"] if smoke else cfg["bisection"]["iters"]
    win = cfg["bisection"]["gate_window_v"]
    out: Dict[str, object] = {"vsw_nom_V": st.vsw, "a": st.a}
    t0 = time.time()
    ts = ts_for_rows(st)
    out["t_s_s"] = {str(n): v for n, v in ts.items()}
    log(f"V_sw,nom {st.vsw:.6f} V; t_s {', '.join(f'N={n}: {v * 1e9:.3f} ns' for n, v in ts.items())} ({time.time() - t0:.0f} s)")

    # T1: transient sensitivities of the pull-down
    pd = cfg["circuit"]["pulldown"]
    qs = pulldown_sensitivity(st.lib, cfg["circuit"]["pulldown_device"], pd["w_um"], pd["l_um"], cfg["circuit"]["vdd"],
                              st.i_sw, st.sw)
    out["sigma_qs_V"] = qs["sigma_delta"]
    out["t1"] = {}
    for n in st.rows:
        cases = [{}] + [{"mm": {k: s}} for k in MM_KEYS for s in (+1.0, -1.0)]
        v = vsw_tr_bisect(st, n, ts[n], cases, it, win)
        sens = {k: float((v[1 + 2 * j] - v[2 + 2 * j]) / 2.0) for j, k in enumerate(MM_KEYS)}
        sig = float(np.sqrt(sum(s * s for s in sens.values())))
        out["t1"][str(n)] = {"vsw_tr_nominal_V": float(v[0]), "sensitivities_V": sens, "sigma_tr_V": sig,
                             "ratio_tr_over_qs": sig / qs["sigma_delta"]}
        log(f"T1 N={n}: V_sw,tr nominal {v[0]:.6f} V (V_sw {st.vsw:.6f}); sigma_tr {1e3 * sig:.3f} mV vs "
            f"sigma_qs {1e3 * qs['sigma_delta']:.3f} mV ({time.time() - t0:.0f} s)")

    # T3 predictions: leaker gate voltages, false-mismatch flags, delta_l and predicted shifts
    n3 = cfg["crosstalk"]["n_cells"]
    configs = cfg["smoke"]["crosstalk_configs"] if smoke else cfg["crosstalk"]["configs"]
    theta_j = cfg["crosstalk"]["leaker_theta"]
    preds = {}
    lam_out = {}
    g_th = g_of_x(st.thetas, st.g_lo, st.g_hi)
    for d in st.designs:
        did = d["id"]
        # divider drain-voltage sensitivity lambda = -d lnI / d V_DS at V_DS = V_sw (E2 predictor), at I and a I
        _, _, dcurves = divider_sensitivity(st.lib, cfg["circuit"]["divider_device"], d["w_div_um"], d["l_div_um"],
                                            st.vsw, st.sw, i_floor=I_FLOOR)
        lam = {"I": eps_sensitivities(dcurves, g_th * 1e-6 * st.d_v)["lambda"],
               "aI": eps_sensitivities(dcurves, st.a["1y"] * g_th * 1e-6 * st.d_v)["lambda"]}
        lam_out[did] = {k: v.tolist() for k, v in lam.items()}
        vx = {}
        for c in configs:
            for node in st.nodes:
                x_j = theta_j + c["d_lsb"] / 15.0
                vx[(c["id"], node)] = leaker_vx(st, did, d, node, x_j, theta_j)
        keys = list(vx)
        fm_tc = false_mismatch(st, n3, ts[n3], [[vx[k]] * next(cc["k"] for cc in configs if cc["id"] == k[0]) for k in keys])
        cases = [{"leak": [vx[k]] * next(cc["k"] for cc in configs if cc["id"] == k[0])} for k in keys]
        vtr = vsw_tr_bisect(st, n3, ts[n3], cases, it, cfg["bisection"]["gate_window_leak_v"])
        for i, key in enumerate(keys):
            cid, node = key
            fm = bool(np.isfinite(fm_tc[i]) and fm_tc[i] <= ts[n3])
            _, _, _, kappa = st.node_params(node)
            delta_l = float(vtr[i] - out["t1"][str(n3)]["vsw_tr_nominal_V"])  # relative to the isolated row of the same N
            lam_b = lam["aI"] if node.startswith("dac") else lam["I"]
            shift = (15.0 * (g_th / st.dg) * ((1.0 - kappa * delta_l / st.d_v) * np.exp(lam_b * delta_l) - 1.0)).tolist()
            preds[f"{did}|{cid}|{node}"] = {"design": did, "config": cid, "node": node, "leaker_vx_V": vx[key],
                                            "leaker_vx_minus_vsw_mV": 1e3 * (vx[key] - st.vsw),
                                            "false_mismatch": fm, "t_cross_cut_off_s": None if not np.isfinite(fm_tc[i]) else float(fm_tc[i]),
                                            "vsw_tr_with_leakers_V": None if fm else float(vtr[i]),
                                            "delta_l_mV": None if fm else 1e3 * delta_l, "kappa": kappa,
                                            "pred_shift_lsb": None if fm else shift}
        log(f"T3 predictions for {did} done ({time.time() - t0:.0f} s)")
    out["t3"] = preds
    out["divider_lambda_per_V"] = lam_out
    out["thetas"] = st.thetas.tolist()

    # secondary: false-mismatch map (largest number of leakers that does not discharge the ML before t_s)
    if not smoke:
        fmap = []
        d0 = st.designs[0]
        for dl in cfg["crosstalk"]["map_d_lsb"]:
            for node in ("t0", "supply_1y"):
                v = leaker_vx(st, d0["id"], d0, node, theta_j + dl / 15.0, theta_j)
                tcs = false_mismatch(st, n3, ts[n3], [[v] * k for k in range(1, n3)])
                ok = [k for k, tc in zip(range(1, n3), tcs) if not (np.isfinite(tc) and tc <= ts[n3])]
                kmax = 0
                for k in range(1, n3):
                    if k in ok:
                        kmax = k
                    else:
                        break
                fmap.append({"d_lsb": dl, "node": node, "leaker_vx_minus_vsw_mV": 1e3 * (v - st.vsw), "k_max_no_false_mismatch": kmax})
                log(f"map d {dl} LSB {node}: V_X - V_sw {1e3 * (v - st.vsw):.1f} mV, k_max {kmax}")
        out["false_mismatch_map"] = fmap

    pred_path = run / "predictions.json"
    with open(pred_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    info["end_jst"] = now_jst().isoformat(timespec="seconds")
    info["predictions_sha256"] = sha256_file(pred_path)
    write_json(run / "run_info.json", info)
    log(f"predictions written; sha256 {info['predictions_sha256']}")
    return 0


# ------------------------------------------------------------------ sim phase
def full_branch_bounds(st: Setup, did: str, d: dict, n: int, t_s: float, cases: List[dict], iters: int, win: float) -> np.ndarray:
    """Transient bound of the cell under test (full branch) for each case: theta, node, optional leakers
    (list of (x_j, theta_j)) modelled as full branches."""
    k = len(cases)
    qs = np.array([st.qs_bound(did, c["theta"], c["node"]) for c in cases])
    leak_vdl = {}
    for c in cases:
        for (xj, thj) in c.get("leak", []):
            leak_vdl[(xj, thj, c["node"])] = st.leaker_vdl(did, xj, c["node"])

    def build(vals):
        ln = header(st.lib, "full branch row")
        for i, (c, v) in enumerate(zip(cases, vals)):
            gf, v_sl, _, _ = st.node_params(c["node"])
            ln.append(f"vsl{i} sl{i} 0 {v_sl:.12g}")
            g = float(g_of_x(np.array([c["theta"]]), st.g_lo, st.g_hi)[0]) * gf
            ln += branch_lines(i, 0, d["w_div_um"], d["l_div_um"], f"sl{i}", g, v, f"xc{i}")
            leak_nodes = []
            for j, (xj, thj) in enumerate(c.get("leak", [])):
                gj = float(g_of_x(np.array([thj]), st.g_lo, st.g_hi)[0]) * gf
                ln += branch_lines(i, j + 1, d["w_div_um"], d["l_div_um"], f"sl{i}", gj, leak_vdl[(xj, thj, c["node"])], f"xl{i}_{j}")
                leak_nodes.append(f"xl{i}_{j}")
            ln += row_devices(i, st.spec, n, f"xc{i}", leak_nodes)
        return ln + [tran_card(st.spec, st.t_stop(t_s))]

    return bisect(build, k, qs - win, qs + win, np.full(k, t_s), iters, increasing_means_faster=False)


def phase_sim(cfg: dict, cfg_path: Path, smoke: bool, pred_path: Path) -> int:
    run = make_run_dir(ROOT / "results", "E3c", smoke)
    log = make_logger(run / "run.log")
    preds = json.loads(pred_path.read_text())
    info = {"experiment": "E3c", "phase": "sim", "smoke": smoke, "start_jst": now_jst().isoformat(timespec="seconds"),
            "environment": environment_info(), "ngspice": ngspice_version(), "sha256": hashes(cfg_path, cfg),
            "predictions_file": str(pred_path.relative_to(ROOT)), "predictions_sha256": sha256_file(pred_path)}
    write_json(run / "run_info.json", info)
    write_json(run / "config_used.json", cfg)
    log(f"E3c sim phase; predictions {pred_path.relative_to(ROOT)} sha256 {info['predictions_sha256']}")
    st = Setup(cfg, smoke)
    it = cfg["bisection"]["iters_smoke"] if smoke else cfg["bisection"]["iters"]
    win = cfg["bisection"]["input_window_v"]
    ts = {int(k): v for k, v in preds["t_s_s"].items()}
    checks: Dict[str, object] = {}
    failed: List[str] = []
    if abs(st.vsw - preds["vsw_nom_V"]) > 1e-12:
        failed.append("V_sw,nom differs from the predict phase")
    t0 = time.time()

    # T2: isolated cell, full branch, transient vs quasi-static
    t2_rows, iso = [], {}
    for d in st.designs:
        did = d["id"]
        for n in st.rows:
            cases = [{"theta": float(th), "node": node} for th in st.thetas for node in st.nodes]
            b = full_branch_bounds(st, did, d, n, ts[n], cases, it, win)
            for c, bb in zip(cases, b):
                e = 15.0 * (float(st.x_eff(did, np.array([bb]), c["node"])[0]) - c["theta"])
                iso[(did, n, round(c["theta"], 9), c["node"])] = e
                t2_rows.append({"design": did, "n_cells": n, "theta": c["theta"], "node": c["node"], "bound_tr_V": float(bb),
                                "bound_qs_V": st.qs_bound(did, c["theta"], c["node"]), "error_lsb": e,
                                "pass": bool(abs(e) <= cfg["criteria"]["t2_tol_lsb"])})
            log(f"T2 {did} N={n}: max |error| {max(abs(r['error_lsb']) for r in t2_rows if r['design'] == did and r['n_cells'] == n):.2e} LSB ({time.time() - t0:.0f} s)")
    write_csv(run / "t2_isolated.csv", t2_rows)

    # T3: leakers as full branches
    n3 = cfg["crosstalk"]["n_cells"]
    configs = cfg["smoke"]["crosstalk_configs"] if smoke else cfg["crosstalk"]["configs"]
    theta_j = cfg["crosstalk"]["leaker_theta"]
    t3_rows = []
    for d in st.designs:
        did = d["id"]
        for c in configs:
            cases = []
            for node in st.nodes:
                p = preds["t3"][f"{did}|{c['id']}|{node}"]
                if p["false_mismatch"]:
                    continue
                for th in st.thetas:
                    cases.append({"theta": float(th), "node": node, "leak": [(theta_j + c["d_lsb"] / 15.0, theta_j)] * c["k"]})
            if not cases:
                continue
            b = full_branch_bounds(st, did, d, n3, ts[n3], cases, it, cfg["bisection"]["input_window_leak_v"])
            for cs, bb in zip(cases, b):
                e = 15.0 * (float(st.x_eff(did, np.array([bb]), cs["node"])[0]) - cs["theta"])
                base = iso[(did, n3, round(cs["theta"], 9), cs["node"])]
                p = preds["t3"][f"{did}|{c['id']}|{cs['node']}"]
                j = int(np.argmin(np.abs(np.array(preds["thetas"]) - cs["theta"])))
                pv = p["pred_shift_lsb"][j]
                sv = e - base
                tol = max(cfg["criteria"]["t3_rel_tol"] * abs(sv), cfg["criteria"]["t3_abs_tol_lsb"])
                t3_rows.append({"design": did, "config": c["id"], "k": c["k"], "d_lsb": c["d_lsb"], "node": cs["node"],
                                "theta": cs["theta"], "bound_tr_V": float(bb), "error_lsb": e, "isolated_error_lsb": base,
                                "sim_shift_lsb": sv, "pred_shift_lsb": pv, "abs_diff_lsb": abs(pv - sv), "tolerance_lsb": tol,
                                "pass": bool(abs(pv - sv) <= tol)})
            log(f"T3 {did} {c['id']}: done ({time.time() - t0:.0f} s)")
    write_csv(run / "t3_crosstalk.csv", t3_rows)

    # false-mismatch consistency for excluded configurations: leakers as full branches, cell under test far inside
    fm_rows = []
    for key, p in preds["t3"].items():
        if not p["false_mismatch"] or (smoke and p["design"] not in cfg["smoke"]["designs"]):
            continue
        d = next(dd for dd in st.designs if dd["id"] == p["design"])
        c = next(cc for cc in configs if cc["id"] == p["config"])
        case = {"theta": 0.0, "node": p["node"], "leak": [(theta_j + c["d_lsb"] / 15.0, theta_j)] * c["k"]}
        vdl_far = cfg["circuit"]["vdd"]  # cell under test strongly matched (input at the top of the DL range)

        def build_fm(d=d, case=case):
            gf, v_sl, _, _ = st.node_params(case["node"])
            ln = header(st.lib, "false mismatch full") + [f"vsl0 sl0 0 {v_sl:.12g}"]
            ln += branch_lines(0, 0, d["w_div_um"], d["l_div_um"], "sl0", st.g_lo * gf, vdl_far, "xc0")
            nodes = []
            for j, (xj, thj) in enumerate(case["leak"]):
                gj = float(g_of_x(np.array([thj]), st.g_lo, st.g_hi)[0]) * gf
                ln += branch_lines(0, j + 1, d["w_div_um"], d["l_div_um"], "sl0", gj, st.leaker_vdl(d["id"], xj, case["node"]), f"xl0_{j}")
                nodes.append(f"xl0_{j}")
            ln += row_devices(0, st.spec, n3, "xc0", nodes)
            return ln + [tran_card(st.spec, st.t_stop(ts[n3]))]

        tc = float(crossing_times(build_fm, 1)[0])
        ok = bool(np.isfinite(tc) and tc <= ts[n3])
        fm_rows.append({"key": key, "t_cross_s": tc if np.isfinite(tc) else None, "full_sim_false_mismatch": ok})
        if not ok:
            failed.append(f"false-mismatch prediction not reproduced: {key}")
    write_csv(run / "false_mismatch_check.csv", fm_rows)

    # checks and decision
    checks["t2_max_abs_error_lsb"] = max(abs(r["error_lsb"]) for r in t2_rows)
    checks["failed"] = failed
    write_json(run / "checks.json", checks)
    if failed:
        log("CHECK FAILURE: " + "; ".join(failed))
        info["status"] = "stopped_by_checks"
        info["end_jst"] = now_jst().isoformat(timespec="seconds")
        write_json(run / "run_info.json", info)
        return 2
    if not smoke:
        t1 = {n: v for n, v in preds["t1"].items()}
        t1_pass = all(abs(v["ratio_tr_over_qs"] - 1) <= cfg["criteria"]["t1_rel_tol"] for v in t1.values())
        t2_pass = all(r["pass"] for r in t2_rows)
        t3_rate = float(np.mean([r["pass"] for r in t3_rows])) if t3_rows else float("nan")
        t3_pass = bool(t3_rows) and t3_rate >= cfg["criteria"]["t3_pass_rate"]
        t4_rows = []
        for d in st.designs:
            for c in configs:
                for th in st.thetas:
                    sel = {r["node"]: r for r in t3_rows if r["design"] == d["id"] and r["config"] == c["id"] and abs(r["theta"] - th) < 1e-9}
                    if "dac_1y" in sel and "supply_1y" in sel:
                        t4_rows.append({"design": d["id"], "config": c["id"], "theta": float(th),
                                        "dac_shift": sel["dac_1y"]["sim_shift_lsb"], "supply_shift": sel["supply_1y"]["sim_shift_lsb"],
                                        "supply_smaller": bool(abs(sel["supply_1y"]["sim_shift_lsb"]) < abs(sel["dac_1y"]["sim_shift_lsb"]))})
        t4_pass = bool(t4_rows) and all(r["supply_smaller"] for r in t4_rows)
        decision = {"T1": {"pass": t1_pass, "ratios": {n: v["ratio_tr_over_qs"] for n, v in t1.items()}},
                    "T2": {"pass": t2_pass, "max_abs_error_lsb": checks["t2_max_abs_error_lsb"], "n": len(t2_rows)},
                    "T3": {"pass": t3_pass, "pass_rate": t3_rate, "n": len(t3_rows),
                           "failures": [f"{r['design']}/{r['config']}/{r['node']}/{r['theta']:.3f}" for r in t3_rows if not r["pass"]]},
                    "T4": {"pass": t4_pass, "n": len(t4_rows), "rows": t4_rows},
                    "validated": bool(t2_pass and t3_pass and t4_pass)}
        write_json(run / "decision.json", decision)
        log(f"decision: T1 {t1_pass}, T2 {t2_pass}, T3 {t3_rate:.0%} ({t3_pass}), T4 {t4_pass}; validated {decision['validated']}")
    info["status"] = "complete"
    info["end_jst"] = now_jst().isoformat(timespec="seconds")
    write_json(run / "run_info.json", info)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=("predict", "sim"), required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--predictions")
    ap.add_argument("--config", default=str(ROOT / "experiments" / "E3c_config.json"))
    args = ap.parse_args()
    cfg_path = Path(args.config).resolve()
    cfg = json.loads(cfg_path.read_text())
    if args.phase == "predict":
        return phase_predict(cfg, cfg_path, args.smoke)
    if not args.predictions:
        sys.exit("--predictions is required for the sim phase")
    p = Path(args.predictions)
    return phase_sim(cfg, cfg_path, args.smoke, p if p.is_absolute() else (ROOT / p))


if __name__ == "__main__":
    sys.exit(main())
