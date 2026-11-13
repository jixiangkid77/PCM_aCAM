"""Task 1 (SPICE side): my own full-row transient netlists for the first case of each
(design, time, node) group of the E4.1 run of record (6 cases: P1 and P3 at t0, 1 y input remap
and 1 y divider supply), crossing time of 0.9 V compared with t_sim_ns of rows.csv (threshold
0.5 %, as the E4 check) and with t_pred_ns; the sense time t_s(N) by SPICE compared with t_s_ns;
the gate voltages of the constrained pull-downs 2 ns after release compared with the SPICE
gates of bounds.csv (diagnosis).

Row netlist (written from the frozen plan, experiments/E4_1_config.json, 'netlist'):
- lower bound (LBUF): PCM resistor 1/G from SL_lo to X; divider nfet_01v8 (design W/L, div_lo
  draw) X-DL-0-0; INV_S 1 (b1n, b1p) X -> W; INV_S 2 (b2n, b2p) W -> Z; pull-down nfet_01v8_lvt
  0.42/0.15 (pd_lo) ML-Z-0-0
- upper bound (UINV): PCM resistor from SL_hi to Y; divider (div_hi) Y-DL-0-0; INV_S (un, up)
  Y -> Z; pull-down (pd_hi) ML-Z-0-0
- INV_S: pfet_01v8 0.42/4 from VDD, nfet_01v8 2.0/0.15 to ground (E4 plan, 'circuit.inverter')
- the other 2N - n_c pull-downs nominal with gate 0 V; ML capacitor C_SA + N C_wire; precharge
  1.8 V by .ic without UIC (every other node starts at its DC operating point); crossing of 0.9 V
Conductances g_us, data-line voltages v_dl, instance indices idx and supplies v_sl_lo / v_sl_hi
are taken from cases.json (task 2 recomputes them with my model); the mismatch draws of the
instances are those of the E4 pools, reproduced by e4i_spice.pool_draws.

Extra (not asked by the task, reported separately and not part of the pass flag): in the six
first cases the lower (LBUF) bounds carry at most 1.1e-4 of S, so they do not test the LBUF chain
near its switching point. I also simulate every case whose LBUF bounds carry at least half of S
(S_lower_LBUF / S >= 0.5 in rows.csv), with the same netlist, settings and threshold.

Usage: python experiments/E4_1_independent/task1_spice.py --out <new folder>
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True           # no __pycache__ folders in the repository

import argparse            # noqa: E402
import json                # noqa: E402
import os                  # noqa: E402
import time                # noqa: E402
from pathlib import Path   # noqa: E402

import numpy as np         # noqa: E402

import e41i_common as K    # noqa: E402
from e41i_common import C, S   # noqa: E402

TRAN_OPTS = "reltol=1e-4 abstol=1e-15 vntol=1e-8"     # as the E4 check
TMAX = 0.1e-9                                          # maximum step (s), as the E4 check
T_GATE = "2n"                                          # gate voltages recorded 2 ns after release
LBUF_SHARE_EXTRA = 0.5                                 # extra cases: S_lower_LBUF / S >= 0.5


def run_ngspice(cir: Path) -> dict:
    """One ngspice process at a time with one OpenMP thread; returns the 'meas' values."""
    os.environ["OMP_NUM_THREADS"] = "1"
    return S.meas_values(S.run_ngspice(cir))


def header(title: str, work: Path, options: str) -> str:
    lines = [f"* {title}", f'.include "{K.lib_rel_from(work)}"']
    if options:
        lines.append(f".options {options}")
    return "\n".join(lines) + "\n"


def case_elements(case: dict, n_feat: int, draws: dict, c_ml: float, sfx: str = "", nd: str = ""):
    """Elements of one row. sfx: suffix of the element names, nd: prefix of the node names (both
    empty for a one-case netlist). Returns (lines, measures, gate node names)."""
    L = [f"VSLLO{sfx} {nd}sl_lo 0 {case['v_sl_lo']:.17g}",
         f"VSLHI{sfx} {nd}sl_hi 0 {case['v_sl_hi']:.17g}"]
    wd, ld = case["w_div_um"], case["l_div_um"]
    meas = [f"meas tran tc{sfx} when v({nd}ml)=0.9 fall=1"]
    gates = []
    for i, bd in enumerate(case["bounds"]):
        k = int(bd["idx"])
        g = bd["g_us"] * 1e-6
        r_pcm = 1.0 / g if g > 0 else 1e18
        z = lambda grp: S.mm(S.inst_z(draws, grp, k))        # noqa: E731
        L.append(f"VDL{i}{sfx} {nd}dl{i} 0 {bd['v_dl']:.17g}")
        if bd["kind"] == 0:                                   # LBUF: X -> INV_S -> W -> INV_S -> Z
            L += [f"RPCM{i}{sfx} {nd}sl_lo {nd}x{i} {r_pcm:.17g}",
                  f"XDIV{i}{sfx} {nd}x{i} {nd}dl{i} 0 0 {S.NFET} w={wd} l={ld} {z('div_lo')}",
                  f"XB1P{i}{sfx} {nd}w{i} {nd}x{i} vdd vdd {S.PFET} w={S.INV_WP} l={S.INV_LP} {z('b1p')}",
                  f"XB1N{i}{sfx} {nd}w{i} {nd}x{i} 0 0 {S.NFET} w={S.INV_WN} l={S.INV_LN} {z('b1n')}",
                  f"XB2P{i}{sfx} {nd}z{i} {nd}w{i} vdd vdd {S.PFET} w={S.INV_WP} l={S.INV_LP} {z('b2p')}",
                  f"XB2N{i}{sfx} {nd}z{i} {nd}w{i} 0 0 {S.NFET} w={S.INV_WN} l={S.INV_LN} {z('b2n')}",
                  f"XPD{i}{sfx} {nd}ml {nd}z{i} 0 0 {S.NFET_LVT} w={S.PD_W} l={S.PD_L} {z('pd_lo')}"]
            meas += [f"meas tran n{i}{sfx}_2ns find v({nd}x{i}) at={T_GATE}",
                     f"meas tran m{i}{sfx}_2ns find v({nd}w{i}) at={T_GATE}"]
        else:                                                 # UINV: Y -> INV_S -> Z
            L += [f"RPCM{i}{sfx} {nd}sl_hi {nd}y{i} {r_pcm:.17g}",
                  f"XDIV{i}{sfx} {nd}y{i} {nd}dl{i} 0 0 {S.NFET} w={wd} l={ld} {z('div_hi')}",
                  f"XUP{i}{sfx} {nd}z{i} {nd}y{i} vdd vdd {S.PFET} w={S.INV_WP} l={S.INV_LP} {z('up')}",
                  f"XUN{i}{sfx} {nd}z{i} {nd}y{i} 0 0 {S.NFET} w={S.INV_WN} l={S.INV_LN} {z('un')}",
                  f"XPD{i}{sfx} {nd}ml {nd}z{i} 0 0 {S.NFET_LVT} w={S.PD_W} l={S.PD_L} {z('pd_hi')}"]
            meas.append(f"meas tran n{i}{sfx}_2ns find v({nd}y{i}) at={T_GATE}")
        meas.append(f"meas tran g{i}{sfx}_2ns find v({nd}z{i}) at={T_GATE}")
        gates.append(f"{nd}z{i}")
    for j in range(2 * n_feat - len(case["bounds"])):
        L.append(f"XW{j}{sfx} {nd}ml 0 0 0 {S.NFET_LVT} w={S.PD_W} l={S.PD_L}")
    L.append(f"CML{sfx} {nd}ml 0 {c_ml:.6g}")
    return L, meas, gates


def full_netlist(title: str, work: Path, options: str, body: list, ics: list, meas: list,
                 tstop: float, tmax: float) -> str:
    lines = [header(title, work, options), f"VDD vdd 0 {S.VDD}"] + body
    lines.append(".ic " + " ".join(ics))
    lines += [".control", f"tran {tmax:.4g} {tstop:.6g} 0 {tmax:.4g}"] + meas + ["quit", ".endc", ".end"]
    return "\n".join(lines) + "\n"


def row_netlist(case, n_feat, draws, c_ml, tstop, tmax, options, tag, work):
    body, meas, gates = case_elements(case, n_feat, draws, c_ml)
    return full_netlist(f"E4.1 independent GS row {tag}", work, options, body, ["v(ml)=1.8"],
                        meas, tstop, tmax), gates


def ts_netlist(n_feat, vsw, c_ml, tstop, tmax, options, work):
    body = [f"VG g 0 {vsw:.17g}", f"XON ml g 0 0 {S.NFET_LVT} w={S.PD_W} l={S.PD_L}"]
    body += [f"XW{j} ml 0 0 0 {S.NFET_LVT} w={S.PD_W} l={S.PD_L}" for j in range(2 * n_feat - 1)]
    body.append(f"CML ml 0 {c_ml:.6g}")
    return full_netlist(f"E4.1 independent: sense time t_s({n_feat})", work, options, body, ["v(ml)=1.8"],
                        ["meas tran tc when v(ml)=0.9 fall=1"], tstop, tmax)


def run(out_dir: Path, log=print, sensitivity: bool = True) -> dict:
    t_start = time.time()
    cfg = C.load_config()
    main = K.load_main()
    cases = main["cases"]["cases"]
    rows = main["rows"]
    n_feat = int(main["cases"]["n_feat"])
    assert all(int(c["n_feat"]) == n_feat for c in cases)
    c_ml = cfg["ml"]["c_sa_f"] + n_feat * cfg["ml"]["c_wire_per_cell_f"]
    work = out_dir / "spice_task1"
    S.write_spiceinit(work)
    draws = S.pool_draws()
    inst = C.Instrument()

    # ---------------------------------------------------------------- sense time t_s(N)
    cir = work / "ts.cir"
    cir.write_text(ts_netlist(n_feat, inst.vsw_nom, c_ml, 1.2e-6, TMAX, TRAN_OPTS, work))
    t_s = run_ngspice(cir)["tc"]
    t_s_main = float(rows[0]["t_s_ns"]) * 1e-9
    assert all(float(r["t_s_ns"]) == float(rows[0]["t_s_ns"]) for r in rows)
    log(f"[task1] t_s({n_feat}) = {t_s * 1e9:.4f} ns (rows.csv {t_s_main * 1e9:.3f} ns, "
        f"cases.json {main['cases']['t_s_s'] * 1e9:.3f} ns)")

    # ---------------------------------------------------------------- the six rows (+ extra rows)
    sel = K.first_of_groups(cases)
    extra = [i for i, r in enumerate(rows)
             if float(r["S_lower_LBUF"]) / float(r["S"]) >= LBUF_SHARE_EXTRA and i not in sel]
    out, gate_rows = [], []
    for ci in sel + extra:
        case, row = cases[ci], rows[ci]
        t0 = time.time()
        t_pred = float(row["t_pred_ns"]) * 1e-9
        tstop = 2.0 * t_pred + 100e-9
        net, gates = row_netlist(case, n_feat, draws, c_ml, tstop, TMAX, TRAN_OPTS, f"case{ci}", work)
        cir = work / f"row_case{ci:02d}.cir"
        cir.write_text(net)
        mv = run_ngspice(cir)
        t_sim = mv["tc"]
        t_sim_main = float(row["t_sim_ns"]) * 1e-9
        rec = {"case": ci, "set": "first_of_group" if ci in sel else "extra_LBUF_dominated",
               "design": case["design"], "time": case["time"], "node": case["node"],
               "sample": case["sample"], "row": case["row"], "n_bounds": len(case["bounds"]),
               "n_lower_LBUF": sum(1 for b in case["bounds"] if b["kind"] == 0),
               "n_upper_UINV": sum(1 for b in case["bounds"] if b["kind"] == 1),
               "S_main": case["S"], "S_lower_LBUF_share": float(row["S_lower_LBUF"]) / float(row["S"]),
               "t_pred_main_ns": t_pred * 1e9,
               "t_sim_main_ns": t_sim_main * 1e9, "t_sim_indep_ns": t_sim * 1e9,
               "rel_diff_vs_main": t_sim / t_sim_main - 1.0,
               "pass": bool(abs(t_sim / t_sim_main - 1.0) <= K.TOL["t_sim_rel"]),
               "ratio_indep_to_pred_main": t_sim / t_pred,
               "ln_ratio_indep_to_pred_main": float(np.log(t_sim / t_pred)),
               "ln_ratio_main": float(np.log(t_sim_main / t_pred)),
               "tstop_ns": tstop * 1e9, "elapsed_s": round(time.time() - t0, 1)}
        bmain = main["bounds_by_case"][ci]
        for i, bd in enumerate(case["bounds"]):
            g_sp = mv[f"g{i}_2ns"]
            g_main = float(bmain[i]["gate_sim_2ns_V"])
            gate_rows.append({"case": ci, "set": rec["set"], "bound": i, "kind": bd["kind"],
                              "type": "LBUF" if bd["kind"] == 0 else "UINV", "feat": bd["feat"],
                              "idx": bd["idx"], "w_model_main": bd["w_model"],
                              "divider_node_2ns_V": mv[f"n{i}_2ns"],
                              "mid_node_W_2ns_V": mv.get(f"m{i}_2ns", float("nan")),
                              "gate_spice_indep_2ns_V": g_sp, "gate_spice_main_2ns_V": g_main,
                              "diff_mV": (g_sp - g_main) * 1e3})
        dg = [g["diff_mV"] for g in gate_rows if g["case"] == ci]
        rec["max_abs_gate_diff_vs_main_spice_mV"] = K.max_abs(dg)
        out.append(rec)
        log(f"[task1] case {ci} ({K.group_key(case)}, {rec['n_lower_LBUF']} LBUF + {rec['n_upper_UINV']} UINV, "
            f"LBUF share of S {rec['S_lower_LBUF_share']:.2g}{'' if ci in sel else ', extra'}): "
            f"t_sim {t_sim * 1e9:.3f} ns, main {t_sim_main * 1e9:.3f} ns, diff {100 * rec['rel_diff_vs_main']:+.4f} %, "
            f"ratio to t_pred {rec['ratio_indep_to_pred_main']:.4f} ({rec['elapsed_s']} s)")

    # ---------------------------------------------------------------- numerical sensitivity
    sens = []
    if sensitivity:
        ref = {o["case"]: o["t_sim_indep_ns"] * 1e-9 for o in out}
        ci = sel[0]
        case = cases[ci]
        t_pred = float(rows[ci]["t_pred_ns"]) * 1e-9
        for label, opts, tm in (("default_tolerances", "", TMAX), ("tmax_0.4ns", TRAN_OPTS, 0.4e-9)):
            t0 = time.time()
            net, _ = row_netlist(case, n_feat, draws, c_ml, 2.0 * t_pred + 100e-9, tm, opts,
                                 f"case{ci} {label}", work)
            cir = work / f"row_case{ci:02d}_{label}.cir"
            cir.write_text(net)
            t_alt = run_ngspice(cir)["tc"]
            sens.append({"variant": label, "case": ci, "t_sim_ns": t_alt * 1e9,
                         "rel_change_vs_my_main_setting": t_alt / ref[ci] - 1.0,
                         "rel_diff_vs_main_run": t_alt / (float(rows[ci]["t_sim_ns"]) * 1e-9) - 1.0,
                         "elapsed_s": round(time.time() - t0, 1)})
        # the six rows in one netlist with the plan's maximum step (0.2 ns) and stop time 6 t_s(N)
        # (the main run put 12 rows into each netlist); separate ML nodes, shared VDD only
        t0 = time.time()
        body, meas, ics = [], [], []
        for ci in sel:
            b_, m_, _ = case_elements(cases[ci], n_feat, draws, c_ml, sfx=f"_c{ci}", nd=f"c{ci}_")
            body += b_
            meas += m_
            ics.append(f"v(c{ci}_ml)=1.8")
        label = "six_rows_one_netlist_tmax_0.2ns_tstop_6ts"
        cir = work / "rows_combined.cir"
        cir.write_text(full_netlist("E4.1 independent: the six rows in one netlist", work, TRAN_OPTS,
                                    body, ics, meas, 6.0 * t_s_main, 0.2e-9))
        mv = run_ngspice(cir)
        el = round(time.time() - t0, 1)
        for ci in sel:
            t_alt = mv[f"tc_c{ci}"]
            sens.append({"variant": label, "case": ci, "t_sim_ns": t_alt * 1e9,
                         "rel_change_vs_my_main_setting": t_alt / ref[ci] - 1.0,
                         "rel_diff_vs_main_run": t_alt / (float(rows[ci]["t_sim_ns"]) * 1e-9) - 1.0,
                         "elapsed_s": el})
        log(f"[task1] numerical sensitivity: largest |change| "
            f"{100 * K.max_abs(s['rel_change_vs_my_main_setting'] for s in sens):.4f} %")

    K.write_csv(out_dir / "task1_spice_cases.csv", out)
    K.write_csv(out_dir / "task1_gates_2ns.csv", gate_rows)
    if sens:
        K.write_csv(out_dir / "task1_sensitivity.csv", sens)

    req = [o for o in out if o["set"] == "first_of_group"]
    ext = [o for o in out if o["set"] != "first_of_group"]
    by_type = {}
    for typ in ("LBUF", "UINV"):
        d = [g["diff_mV"] for g in gate_rows if g["type"] == typ]
        by_type[typ] = {"n_bounds": len(d), "max_abs_diff_mV": K.max_abs(d)}
    t_s_ok = abs(t_s / t_s_main - 1.0) <= K.TOL["t_s_rel"]
    summary = {
        "selected_cases": sel,
        "groups": [K.group_key(cases[i]) for i in sel],
        "extra_cases_LBUF_dominated": extra,
        "extra_rule": f"S_lower_LBUF / S >= {LBUF_SHARE_EXTRA} (rows.csv); reported, not part of the pass flag",
        "threshold_rel": K.TOL["t_sim_rel"],
        "c_ml_F": c_ml,
        "tran": {"options": TRAN_OPTS, "tmax_s": TMAX, "tstop": "2 t_pred + 100 ns",
                 "precharge": ".ic v(ml)=1.8 without UIC", "crossing_V": 0.9},
        "t_s_indep_ns": t_s * 1e9, "t_s_main_ns": t_s_main * 1e9,
        "t_s_rel_diff": t_s / t_s_main - 1.0,
        "t_s_pass": bool(t_s_ok),
        "max_abs_rel_diff_vs_main": K.max_abs(o["rel_diff_vs_main"] for o in req),
        "ratio_indep_to_pred_main_range": [float(min(o["ratio_indep_to_pred_main"] for o in req)),
                                           float(max(o["ratio_indep_to_pred_main"] for o in req))],
        "max_abs_ln_ratio_indep_to_pred_main": K.max_abs(o["ln_ratio_indep_to_pred_main"] for o in req),
        "max_S_lower_LBUF_share_first_cases": max(o["S_lower_LBUF_share"] for o in req),
        "extra": {"cases": extra,
                  "max_abs_rel_diff_vs_main": K.max_abs(o["rel_diff_vs_main"] for o in ext),
                  "max_abs_ln_ratio_indep_to_pred_main": K.max_abs(o["ln_ratio_indep_to_pred_main"] for o in ext),
                  "all_within_threshold": bool(all(o["pass"] for o in ext))},
        "gate_2ns_vs_main_spice": by_type,
        "cases": out,
        "numerical_sensitivity": sens,
        "pass": bool(all(o["pass"] for o in req) and t_s_ok),
        "elapsed_s": round(time.time() - t_start, 1),
    }
    with open(out_dir / "task1_summary.json", "w") as f:
        json.dump(summary, f, indent=1)
    log(f"[task1] max |rel diff| vs main {100 * summary['max_abs_rel_diff_vs_main']:.4f} % (6 cases), "
        f"{100 * summary['extra']['max_abs_rel_diff_vs_main']:.4f} % ({len(ext)} extra LBUF-dominated cases); "
        f"t_s rel diff {summary['t_s_rel_diff']:+.2e}; PASS={summary['pass']} ({summary['elapsed_s']} s)")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="output folder (created; must not exist)")
    ap.add_argument("--no-sensitivity", action="store_true")
    a = ap.parse_args()
    o = Path(a.out).resolve()
    o.mkdir(parents=True, exist_ok=False)
    run(o, sensitivity=not a.no_sensitivity)
