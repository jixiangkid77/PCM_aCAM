#!/usr/bin/env python3
"""E6 independent check, phase 2 step A: additional case group 1d, computed before opening any main-run file.

1d: the gain-stage lower branch LBUF of the GS cell set (X -> INV_S -> INV_S -> pull-down gate), designs P1, P3,
nodes t0 and dac_remap at 1 year, theta = 1 (level 15), x_prev = 0, C_par = 0.5 fF at the divider node X.
GS map on the nominal divider curve at V_DS = V_X*: M_f(x) = I_0^-1(f G(x) D); programming
G = I_div(M0(theta), V_DS = V_X*) / D; V_SL = V_X* + D (t0 and input remap); conductance a G with the remap.
Transient definitions and numerics as item 1 of run_check.py (plan numerics, stop at 600 ns), plus the
half-step, quarter-step and Gear reruns.  Reports the settling time to the plan's band (+-0.01 LSB) and to
+-0.1 LSB, and time constants of the last part of the approach of X, of the first inverter's output and of
the decision node.

Writes only new files into the phase-1 results folder: extra_lbuf.json, extra_lbuf.csv, extra_lbuf_run.log,
netlists_extra_lbuf/.  Run:  python experiments/E6_independent/run_extra_lbuf.py
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np                      # noqa: E402
import run_check as rc                  # noqa: E402

DEFAULT_OUT = rc.RESULTS / "E6_independent_check_20261024_0936"
NEW_FILES = ["extra_lbuf.json", "extra_lbuf.csv", "extra_lbuf_run.log", "netlists_extra_lbuf"]
TAU_WINDOWS_LSB = [(0.003, 0.03), (0.0003, 0.003)]     # input-referred distance from the final DC value


def lbuf_lines(m: rc.Model, tag: str, l_div: float, g: float, vsl: float, vdl: str,
               cpar: float = 0.0) -> tuple[list[str], str, str, str]:
    """LBUF: PCM 1/G from V_SL to X, divider (gate = DL), X -> INV_S -> mid -> INV_S -> pull-down gate."""
    n, mid, dec = f"n_{tag}", f"m_{tag}", f"o_{tag}"
    lines = [f"vsl_{tag} sl_{tag} 0 {rc.fmt(vsl)}",
             f"r_{tag} sl_{tag} {n} {rc.fmt(1.0 / g)}",
             rc.dev(f"div_{tag}", n, f"dl_{tag}", "0", "0", rc.NFET, 0.42, l_div),
             f"vdl_{tag} dl_{tag} 0 {vdl}",
             f"vml_{tag} ml_{tag} 0 {rc.fmt(m.vdd)}",
             f"vinva_{tag} via_{tag} 0 {rc.fmt(m.vdd)}",
             f"vinvb_{tag} vib_{tag} 0 {rc.fmt(m.vdd)}"]
    lines += rc.inv_lines(m, f"a{tag}", n, mid, f"via_{tag}")
    lines += rc.inv_lines(m, f"b{tag}", mid, dec, f"vib_{tag}")
    lines.append(rc.dev(f"pd_{tag}", f"ml_{tag}", dec, "0", "0", rc.NFET_LVT, m.pd_w, m.pd_l))
    if cpar > 0:
        lines.append(f"cpar_{tag} {n} 0 {rc.fmt(cpar)}")
    return lines, n, mid, dec


def maps_at(sp: rc.Spice, m: rc.Model, design: str, vds: float, a1y: float, xs: dict, log) -> tuple[dict, dict]:
    """M0 and M_a(1 y) on the nominal divider of the design at V_DS = vds; .op check at every root."""
    w, l = m.designs[design]
    body = "\n".join([f"vd d 0 {rc.fmt(vds)}", "vg g 0 0", rc.dev("dv", "d", "g", "0", "0", rc.NFET, w, l)])
    targets = {f"{fk}|{xk}": f * m.G(x) * m.d for fk, f in (("M0", 1.0), ("Ma", a1y)) for xk, x in xs.items()}
    r = rc.roots_by_sweeps(sp, f"mapGS_{design}", body, "vg", lambda raw: -raw["i(vd)"], targets, 0.0, m.vdd)
    keys = list(targets)
    maps = {"M0": {}, "Ma": {}}
    lines = []
    for j, key in enumerate(keys):
        fk, xk = key.split("|")
        maps[fk][xk] = {"x": xs[xk], "V_DL": r[key]["root"], "I_target_A": targets[key], "gm_S": r[key]["slope"]}
        lines += [f"vd{j} d{j} 0 {rc.fmt(vds)}", f"vg{j} g{j} 0 {rc.fmt(r[key]['root'])}",
                  rc.dev(f"dv{j}", f"d{j}", f"g{j}", "0", "0", rc.NFET, w, l)]
    o = sp.run(f"mapGS_{design}_verify", "\n".join(lines), "op\nwrite @RAW(o)@ all")["o"]
    worst = 0.0
    for j, key in enumerate(keys):
        fk, xk = key.split("|")
        e = maps[fk][xk]
        e["I_at_root_A"] = float(-o[f"i(vd{j})"][0])
        e["rel_residual"] = e["I_at_root_A"] / e["I_target_A"] - 1.0
        e["voltage_residual_V"] = (e["I_target_A"] - e["I_at_root_A"]) / e["gm_S"]
        worst = max(worst, abs(e["voltage_residual_V"]))
    # programming G = I_div(M0(theta), V_DS = V_X*) / D: the .op current at the M0(theta = 1) root, drain at V_X*
    prog = {"theta": 1.0, "I_div_A": maps["M0"]["1"]["I_at_root_A"],
            "G_S": maps["M0"]["1"]["I_at_root_A"] / m.d, "G_over_G_theta": maps["M0"]["1"]["I_at_root_A"] / m.d / m.G(1.0)}
    log(f"{design}: GS map at V_DS = V_X* verified, worst |voltage residual| = {worst:.2e} V; "
        f"G(theta=1) = {prog['G_S'] * 1e6:.9f} uS")
    return maps, prog


def tail_taus(t: np.ndarray, v: np.ndarray, v_final: float, slope_v_per_lsb: float, t_edge: float) -> list[dict]:
    """Straight-line fits of ln|v - v_final| on the samples after the edge whose input-referred distance
    |v - v_final| / |slope| lies in each window (LSB)."""
    sel = t > t_edge + 0.1e-9
    tt = t[sel]
    d = np.abs(v[sel] - v_final) / abs(slope_v_per_lsb)
    out = []
    for lo, hi in TAU_WINDOWS_LSB:
        win = (d > lo) & (d < hi)
        rec = {"window_LSB": [lo, hi], "n": int(win.sum()), "tau_s": None, "t_from_s": None, "t_to_s": None}
        if win.sum() >= 5:
            p = np.polyfit(tt[win], np.log(d[win]), 1)
            rec.update({"tau_s": float(-1.0 / p[0]) if p[0] < 0 else None,
                        "t_from_s": float(tt[win][0] - t_edge), "t_to_s": float(tt[win][-1] - t_edge)})
        out.append(rec)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="the phase-1 results folder (new files only)")
    ap.add_argument("--tmp", default=None, help="base directory for the ngspice temporary folder")
    args = ap.parse_args()
    out = Path(args.out).resolve()
    if not (out / "values.json").exists():
        print(f"{out} is not a phase-1 results folder", file=sys.stderr)
        return 2
    for f in NEW_FILES:
        if (out / f).exists():
            print(f"refusing to overwrite {out / f}", file=sys.stderr)
            return 2
    (out / "netlists_extra_lbuf").mkdir()
    log = rc.Log(out / "extra_lbuf_run.log")
    t_start = rc.now_jst()
    log(f"E6 independent check phase 2 step A (1d, LBUF); output folder {out}")
    cfg = json.loads(rc.CONFIG.read_text())
    m = rc.Model(cfg)
    lib = (rc.ROOT / cfg["models"]["lib"]).resolve()
    sp = rc.Spice(lib, log, out / "netlists_extra_lbuf", args.tmp)
    a1y = m.a(m.t_1y)
    res = {"meta": {}, "cases": []}
    try:
        sw = rc.switching_points(sp, m, log)
        p1 = json.loads((out / "values.json").read_text())["item0_switching_points"]
        for k in ("V_sw", "V_Y_star", "V_X_star"):
            if sw[k]["value_V"] != p1[k]["value_V"]:
                raise RuntimeError(f"{k} differs from phase 1: {sw[k]['value_V']} vs {p1[k]['value_V']}")
        vsw, vx = sw["V_sw"]["value_V"], sw["V_X_star"]["value_V"]
        log(f"switching points identical to phase 1 (V_X* = {vx:.9f} V)")
        lsb = m.lsb
        xs = {"0": 0.0, "1": 1.0, "1-d": 1 - 0.01 * lsb, "1+d": 1 + 0.01 * lsb, "1-D": 1 - 0.1 * lsb, "1+D": 1 + 0.1 * lsb}
        dd = {d: maps_at(sp, m, d, vx, a1y, xs, log) for d in ("P1", "P3")}
        cases = [dict(id=f"d{j}", group="1d", design=d, cell_set="GS", btype="LBUF", node=nd, theta_key="1", level=15,
                      x_prev=0.0) for j, (d, nd) in enumerate([(d, nd) for d in ("P1", "P3") for nd in ("t0", "dac_remap_1y")])]

        def setting(c):
            maps, prog = dd[c["design"]]
            if c["node"] == "t0":
                return {"G_S": prog["G_S"], "V_SL": vx + m.d, "map": "M0", "a": 1.0}, maps["M0"]
            return {"G_S": a1y * prog["G_S"], "V_SL": vx + m.d, "map": "Ma", "a": a1y}, maps["Ma"]

        # DC points (plan options; tight options as a check)
        pts = ["0", "1-D", "1-d", "1", "1+d", "1+D"]
        dc = {}
        for opts, okey in ((rc.OPTS_PLAN, "plan"), (rc.OPTS_TIGHT, "tight")):
            lines, nodes = [], {}
            for c in cases:
                ns, mp = setting(c)
                for pk in pts:
                    tag = f"{c['id']}{pts.index(pk)}"
                    ls, n, mid, dec = lbuf_lines(m, tag, m.designs[c["design"]][1], ns["G_S"], ns["V_SL"],
                                                 rc.fmt(mp[pk]["V_DL"]))
                    lines += ls
                    nodes[(c["id"], pk)] = (n, mid, dec)
            o = sp.run(f"lbuf_dc_{okey}", "\n".join(lines), "op\nwrite @RAW(o)@ all", options=opts)["o"]
            dc[okey] = {k: tuple(float(o[f"v({q})"][0]) for q in v) for k, v in nodes.items()}
        dc_diff = max(abs(a - b) for k in dc["plan"] for a, b in zip(dc["plan"][k], dc["tight"][k]))
        log(f"DC points: max |plan - tight options| = {dc_diff:.2e} V")

        for c in cases:
            ns, mp = setting(c)
            P = {pk: dc["plan"][(c["id"], pk)] for pk in pts}           # (X, mid, dec)
            node_names = ["X", "mid", "dec"]
            slopes = {nm: (P["1+d"][i] - P["1-d"][i]) / (2 * 0.01) for i, nm in enumerate(node_names)}   # V per LSB
            band001 = (min(P["1-d"][2], P["1+d"][2]), max(P["1-d"][2], P["1+d"][2]))
            band01 = (min(P["1-D"][2], P["1+D"][2]), max(P["1-D"][2], P["1+D"][2]))
            t_stop = 6e-7
            pwl = (f"pwl(0 {rc.fmt(mp['0']['V_DL'])} {rc.fmt(m.t_edge)} {rc.fmt(mp['0']['V_DL'])} "
                   f"{rc.fmt(m.t_edge + m.t_rise)} {rc.fmt(mp['1']['V_DL'])} {rc.fmt(t_stop * 1.01)} {rc.fmt(mp['1']['V_DL'])})")
            rec = dict(c)
            rec.update({"theta": 1.0, "G_uS": ns["G_S"] * 1e6, "V_SL_V": ns["V_SL"], "map": ns["map"], "a": ns["a"],
                        "map_V_DS_V": vx, "V_DL_prev_V": mp["0"]["V_DL"], "V_DL_theta_V": mp["1"]["V_DL"],
                        "V_DL_theta_pm_0p01_V": [mp["1-d"]["V_DL"], mp["1+d"]["V_DL"]],
                        "V_DL_theta_pm_0p1_V": [mp["1-D"]["V_DL"], mp["1+D"]["V_DL"]],
                        "DC_X_mid_dec_V": {pk: list(P[pk]) for pk in pts},
                        "band_0p01_lo_V": band001[0], "band_0p01_hi_V": band001[1],
                        "band_0p1_lo_V": band01[0], "band_0p1_hi_V": band01[1],
                        "V_dec_DC_theta_V": P["1"][2], "DC_theta_minus_V_sw_LSB": (P["1"][2] - vsw) / slopes["dec"],
                        "slope_V_per_LSB": slopes, "t_stop_s": t_stop, "C_par_F": m.cpar})
            runs = [("plan", m.t_step, m.t_max, rc.OPTS_PLAN),
                    ("half_step", m.t_step / 2, m.t_max / 2, rc.OPTS_PLAN),
                    ("quarter_step", m.t_step / 4, m.t_max / 4, rc.OPTS_PLAN),
                    ("gear", m.t_step, m.t_max, rc.OPTS_PLAN + "\n.options method=gear")]
            for rkey, tstep, tmax, topts in runs:
                ls, n, mid, dec = lbuf_lines(m, "t", m.designs[c["design"]][1], ns["G_S"], ns["V_SL"], pwl, cpar=m.cpar)
                raw = sp.run(f"tran_{c['id']}_{rkey}", "\n".join(ls),
                             f"tran {rc.fmt(tstep)} {rc.fmt(t_stop)} 0 {rc.fmt(tmax)}\n"
                             f"write @RAW(tr)@ v({n}) v({mid}) v({dec}) v(dl_t)", options=topts)["tr"]
                t = raw["time"]
                wv = {"X": raw[f"v({n})"], "mid": raw[f"v({mid})"], "dec": raw[f"v({dec})"]}
                r = {"tstep_s": tstep, "tmax_s": tmax, "n_points": int(len(t))}
                for bkey, (lo, hi) in (("0p01", band001), ("0p1", band01)):
                    ts, st = rc.settle_time(t, wv["dec"], lo, hi, m.t_edge)
                    r[f"t_settle_{bkey}_s"] = ts
                    r[f"status_{bkey}"] = st
                    r[f"crossings_{bkey}"] = rc.band_crossings(t, wv["dec"], lo, hi, m.t_edge)
                for i, nm in enumerate(node_names):
                    r[f"tau_{nm}"] = tail_taus(t, wv[nm], P["1"][i], slopes[nm], m.t_edge)
                    r[f"end_minus_DC_{nm}_LSB"] = float((wv[nm][-1] - P["1"][i]) / slopes[nm])
                r["end_inside_band_0p01"] = bool(band001[0] <= wv["dec"][-1] <= band001[1])
                r["t0_op_minus_DC_prev_V"] = [float(wv[nm][0] - P["0"][i]) for i, nm in enumerate(node_names)]
                r["dec_min_max_after_edge_V"] = [float(wv["dec"][t >= m.t_edge].min()), float(wv["dec"][t >= m.t_edge].max())]
                rec[rkey] = r
            q, h = rec["quarter_step"], rec["half_step"]
            for bkey in ("0p01", "0p1"):
                a_, b_, c_ = rec["plan"][f"t_settle_{bkey}_s"], h[f"t_settle_{bkey}_s"], q[f"t_settle_{bkey}_s"]
                rec[f"extrapolated_t_settle_{bkey}_s"] = (4 * c_ - b_) / 3 if None not in (b_, c_) else None
                rec[f"conv_plan_vs_half_{bkey}_s"] = abs(a_ - b_) if None not in (a_, b_) else None
                rec[f"conv_half_vs_quarter_{bkey}_s"] = abs(b_ - c_) if None not in (b_, c_) else None
                g_ = rec["gear"][f"t_settle_{bkey}_s"]
                rec[f"gear_minus_plan_{bkey}_s"] = g_ - a_ if None not in (a_, g_) else None
            pr = rec["plan"]

            def tau_str(nm):
                return ", ".join("-" if w["tau_s"] is None else f"{w['tau_s'] * 1e9:.3f}" for w in pr[f"tau_{nm}"])
            ts1, ts2 = pr["t_settle_0p01_s"], pr["t_settle_0p1_s"]
            log(f"{c['id']} {c['design']} LBUF {c['node']:<12}: t_settle(+-0.01 LSB) = "
                f"{'unsettled' if ts1 is None else f'{ts1 * 1e9:.4f}'} ns (half {h['t_settle_0p01_s'] * 1e9:.4f}, quarter "
                f"{q['t_settle_0p01_s'] * 1e9:.4f}, gear {rec['gear']['t_settle_0p01_s'] * 1e9:.4f}); t_settle(+-0.1 LSB) = "
                f"{'unsettled' if ts2 is None else f'{ts2 * 1e9:.4f}'} ns; tau (ns) X [{tau_str('X')}], mid [{tau_str('mid')}], "
                f"dec [{tau_str('dec')}]; band {band001[0]:.6f}-{band001[1]:.6f} V, DC(theta) {P['1'][2]:.6f} V")
            res["cases"].append(rec)
        res["dc_plan_vs_tight_max_abs_V"] = dc_diff
        res["maps"] = {d: dd[d][0] for d in dd}
        res["programming"] = {d: dd[d][1] for d in dd}
        res["switching_points_V"] = {k: sw[k]["value_V"] for k in ("V_sw", "V_Y_star", "V_X_star")}
    finally:
        sp.cleanup()
    t_end = rc.now_jst()
    res["meta"] = {"started_jst": t_start.strftime("%Y-%m-%d %H:%M:%S"), "finished_jst": t_end.strftime("%Y-%m-%d %H:%M:%S"),
                   "wall_s": (t_end - t_start).total_seconds(), "ngspice_runs": sp.n, "ngspice_cpu_s": sp.cpu,
                   "ngspice_warnings": sp.warnings, "script_sha256": rc.sha256(Path(__file__).resolve()),
                   "run_check_sha256": rc.sha256(HERE / "run_check.py"), "a_1y": a1y,
                   "notes": ["written before any main-run file was opened",
                             "tau windows: input-referred distance |v - v_DC(theta)| / |slope| of each node (slope from "
                             "its DC values at M(theta +- 0.01 LSB)) in [0.003, 0.03] LSB (last part of the approach, "
                             "around the +-0.01 LSB band edge) and [0.0003, 0.003] LSB (asymptotic tail)",
                             "the +-0.1 LSB band uses the DC decision-node voltages at M(theta -+ 0.1 LSB)",
                             "each INV_S has its own 1.8 V source (pMOS source and body)"]}
    (out / "extra_lbuf.json").write_text(json.dumps(rc.jsonable(res), indent=1) + "\n")
    cols = ["id", "design", "node", "G_uS", "V_SL_V", "map", "V_DL_prev_V", "V_DL_theta_V", "band_0p01_lo_V", "band_0p01_hi_V",
            "band_0p1_lo_V", "band_0p1_hi_V", "V_dec_DC_theta_V", "plan_t_settle_0p01_ns", "half_t_settle_0p01_ns",
            "quarter_t_settle_0p01_ns", "extrap_t_settle_0p01_ns", "gear_t_settle_0p01_ns", "plan_t_settle_0p1_ns",
            "extrap_t_settle_0p1_ns", "tau_X_ns", "tau_mid_ns", "tau_dec_ns"]
    rows = []
    for r in res["cases"]:
        f = lambda v: None if v is None else v * 1e9          # noqa: E731
        rows.append({**{k: r.get(k) for k in cols[:13]},
                     "plan_t_settle_0p01_ns": f(r["plan"]["t_settle_0p01_s"]),
                     "half_t_settle_0p01_ns": f(r["half_step"]["t_settle_0p01_s"]),
                     "quarter_t_settle_0p01_ns": f(r["quarter_step"]["t_settle_0p01_s"]),
                     "extrap_t_settle_0p01_ns": f(r["extrapolated_t_settle_0p01_s"]),
                     "gear_t_settle_0p01_ns": f(r["gear"]["t_settle_0p01_s"]),
                     "plan_t_settle_0p1_ns": f(r["plan"]["t_settle_0p1_s"]),
                     "extrap_t_settle_0p1_ns": f(r["extrapolated_t_settle_0p1_s"]),
                     "tau_X_ns": ";".join("" if w["tau_s"] is None else repr(w["tau_s"] * 1e9) for w in r["plan"]["tau_X"]),
                     "tau_mid_ns": ";".join("" if w["tau_s"] is None else repr(w["tau_s"] * 1e9) for w in r["plan"]["tau_mid"]),
                     "tau_dec_ns": ";".join("" if w["tau_s"] is None else repr(w["tau_s"] * 1e9) for w in r["plan"]["tau_dec"])})
    rc.write_csv(out / "extra_lbuf.csv", rows, cols)
    log(f"done: {sp.n} ngspice runs, {sp.cpu:.1f} s ngspice CPU, wall {(t_end - t_start).total_seconds():.0f} s")
    log.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
