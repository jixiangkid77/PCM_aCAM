#!/usr/bin/env python3
"""E6 independent check, phase 2 step B: compare the independent values with the main run of record.

Reads, from the main-run folder given as the argument, only settle.csv, power_curves.npz, energy_dl_ml.csv,
overhead.csv, checks.json and run.log; from the independent results folder values.json (phase 1) and
extra_lbuf.json (phase 2 step A).  Writes compare.json and report.md into the independent results folder.

Run:  python experiments/E6_independent/compare.py results/E6_20261024_1032
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DEFAULT_MINE = ROOT / "results" / "E6_independent_check_20261024_0936"
MAIN_FILES = ["settle.csv", "power_curves.npz", "energy_dl_ml.csv", "overhead.csv", "checks.json", "run.log"]
JST = dt.timezone(dt.timedelta(hours=9))

TOL_SETTLE_REL, TOL_SETTLE_ABS_NS = 0.02, 0.1
TOL_POWER_REL = 0.005
NODE_MAP = {"t0": ("t0", "t0"), "dac_remap_1y": ("dac_remap", "1y"), "supply_1y": ("supply", "1y")}


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read_csv(p: Path) -> list[dict]:
    with open(p, newline="") as f:
        return list(csv.DictReader(f))


def settle_rows(main_rows: list[dict], design: str, cell_set: str, btype: str, node: str, level: int,
                x_prev: float) -> list[dict]:
    nd, tm = NODE_MAP[node]
    return [r for r in main_rows
            if r["design"] == design and r["cell_set"] == cell_set and r["type"] == btype and r["node"] == nd
            and r["time"] == tm and int(r["level"]) == level and float(r["x_prev"]) == x_prev
            and float(r["c_par_f"]) == 5e-16]


def compare_settle(mine: dict, main_rows: list[dict]) -> dict:
    """mine: id, group, design, cell_set, btype, node, level, x_prev, t_ns, band_lo, band_hi, v_dc, v_end, slope, a."""
    rows = settle_rows(main_rows, mine["design"], mine["cell_set"], mine["btype"], mine["node"], mine["level"],
                       mine["x_prev"])
    out = dict(mine)
    out["n_main_rows"] = len(rows)
    if len(rows) != 1:
        out["pass"] = False
        out["note"] = "no unique main-run row"
        return out
    r = rows[0]
    t_main = None if r["t_settle_ns"] in ("", "nan", "None") else float(r["t_settle_ns"])
    t_mine = mine["t_ns"]
    out["t_main_ns"] = t_main
    if t_main is None or t_mine is None or not math.isfinite(t_main):
        out["pass"] = (t_main is None) == (t_mine is None)
    else:
        d = t_mine - t_main
        out["abs_diff_ns"] = d
        out["rel_diff"] = abs(d) / max(abs(t_main), abs(t_mine)) if max(abs(t_main), abs(t_mine)) > 0 else 0.0
        out["pass"] = bool(abs(d) <= TOL_SETTLE_ABS_NS or out["rel_diff"] <= TOL_SETTLE_REL)
    s = abs(float(r["slope_v_per_lsb"]))
    out["main_slope_v_per_lsb"] = s
    for k, mk in (("band_lo", "band_lo_v"), ("band_hi", "band_hi_v"), ("v_dc", "v_dc_v"), ("v_end", "v_end_v")):
        mv = float(r[mk])
        out[f"main_{k}_V"] = mv
        out[f"diff_{k}_V"] = mine[k] - mv
        out[f"diff_{k}_LSB"] = (mine[k] - mv) / s
    out["main_a"] = float(r["a"])
    out["diff_a"] = mine["a"] - float(r["a"])
    out["slope_rel_diff"] = abs(abs(mine["slope"]) - s) / s
    return out


def power_on_main_grid(v1: dict, cases: list[dict], tmp: str | None) -> dict:
    """Explanation aid for item 2: my own DC power at the two 1 mV data-line grid points that bracket V_DL
    (the main run sweeps the DL in 1 mV steps and interpolates linearly), interpolated the same way."""
    sys.path.insert(0, str(HERE))
    import run_check as rc                                   # own phase-1 helpers
    cfg = json.loads(rc.CONFIG.read_text())
    m = rc.Model(cfg)
    lib = (rc.ROOT / cfg["models"]["lib"]).resolve()

    class _Quiet:
        def __call__(self, msg):
            pass
    sp = rc.Spice(lib, _Quiet(), None, tmp)
    sw = v1["item0_switching_points"]
    dd = v1["maps_and_programming"]
    a1y = v1["drift"]["a_1y"]
    mine = {(r["design"], r["btype"], r["node"], r["xkey"]): r for r in v1["item2_power"]["cases"]}
    lines, meta = [], []
    try:
        for j, c in enumerate(cases):
            r = mine[(c["design"], c["btype"], c["node"], c["x"])]
            ns = rc.node_setting(m, sw, dd[c["design"]], c["btype"], c["node"], "8/15", a1y)
            k = math.floor(r["V_DL_V"] / 1e-3)
            for s, vdl in enumerate((k * 1e-3, (k + 1) * 1e-3)):
                tag = f"g{j}{'ab'[s]}"
                ls, n, dec = rc.branch_lines(m, tag, c["btype"], m.designs[c["design"]][1], ns["G_S"], ns["V_SL"], rc.fmt(vdl))
                lines += ls
            meta.append((j, k, r["V_DL_V"], ns))
        o = sp.run("power_grid", "\n".join(lines), "op\nwrite @RAW(o)@ all")["o"]
    finally:
        sp.cleanup()
    out = {}
    for (j, k, vdl, ns), c in zip(meta, cases):
        ps = []
        for s in "ab":
            tag = f"g{j}{s}"
            p = ns["V_SL"] * float(-o[f"i(vsl_{tag})"][0])
            if c["btype"] == "UINV":
                p += m.vdd * float(-o[f"i(vinv_{tag})"][0])
            ps.append(p * 1e6)
        w = (vdl - k * 1e-3) / 1e-3
        p_int = ps[0] + w * (ps[1] - ps[0])
        out[j] = {"grid_V": [k * 1e-3, (k + 1) * 1e-3], "P_grid_uW": ps, "P_mine_interp_uW": p_int,
                  "interp_bias_rel": p_int / c["P_mine_uW"] - 1.0,
                  "rel_diff_main_vs_mine_interp": abs(c["P_main_uW"] - p_int) / abs(p_int)}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("main_dir", help="main-run folder (run of record)")
    ap.add_argument("--mine", default=str(DEFAULT_MINE), help="independent results folder")
    ap.add_argument("--tmp", default=None, help="base directory for the ngspice temporary folder (power-grid aid)")
    args = ap.parse_args()
    main_dir = Path(args.main_dir).resolve()
    mine_dir = Path(args.mine).resolve()
    for f in ("compare.json", "report.md"):
        if (mine_dir / f).exists():
            print(f"refusing to overwrite {mine_dir / f}", file=sys.stderr)
            return 2
    v1 = json.loads((mine_dir / "values.json").read_text())
    v2 = json.loads((mine_dir / "extra_lbuf.json").read_text())
    settle = read_csv(main_dir / "settle.csv")
    curves = np.load(main_dir / "power_curves.npz")
    energy = read_csv(main_dir / "energy_dl_ml.csv")
    overhead = read_csv(main_dir / "overhead.csv")
    checks = json.loads((main_dir / "checks.json").read_text())
    runlog = (main_dir / "run.log").read_text()
    def rel(p: Path) -> str:
        return str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)

    res = {"meta": {"main_dir": rel(main_dir), "mine_dir": rel(mine_dir),
                    "created_jst": dt.datetime.now(JST).strftime("%Y-%m-%d %H:%M:%S"),
                    "main_files_read": MAIN_FILES,
                    "main_file_sha256": {f: sha256(main_dir / f) for f in MAIN_FILES},
                    "mine_file_sha256": {f: sha256(mine_dir / f) for f in ("values.json", "extra_lbuf.json")},
                    "script_sha256": sha256(Path(__file__).resolve())}}

    # ---- items 1a-1c (phase 1) and 1d (phase 2 step A)
    a1y = v1["drift"]["a_1y"]
    mine_cases = []
    for c in v1["item1_settling"]["cases"]:
        mine_cases.append({"id": c["id"], "group": c["group"], "design": c["design"], "cell_set": "6T2M",
                           "btype": c["btype"], "node": c["node"], "level": int(c["level"]), "x_prev": float(c["x_prev"]),
                           "t_ns": None if c["plan_t_settle_s"] is None else c["plan_t_settle_s"] * 1e9,
                           "t_converged_ns": None if c.get("richardson_t_settle_s") is None else c["richardson_t_settle_s"] * 1e9,
                           "band_lo": c["band_lo_V"], "band_hi": c["band_hi_V"], "v_dc": c["V_dec_DC_theta_V"],
                           "v_end": c["plan_V_dec_end_V"], "slope": c["slope_V_per_LSB"], "a": c["a"]})
    for c in v2["cases"]:
        mine_cases.append({"id": c["id"], "group": "1d", "design": c["design"], "cell_set": "GS", "btype": "LBUF",
                           "node": c["node"], "level": 15, "x_prev": 0.0,
                           "t_ns": None if c["plan"]["t_settle_0p01_s"] is None else c["plan"]["t_settle_0p01_s"] * 1e9,
                           "t_converged_ns": None if c["extrapolated_t_settle_0p01_s"] is None else c["extrapolated_t_settle_0p01_s"] * 1e9,
                           "band_lo": c["band_0p01_lo_V"], "band_hi": c["band_0p01_hi_V"], "v_dc": c["V_dec_DC_theta_V"],
                           "v_end": None, "slope": c["slope_V_per_LSB"]["dec"], "a": c["a"]})
    # v_end of 1d from the plan transient: end deviation in LSB x slope + DC
    for mc, c in zip(mine_cases[-len(v2["cases"]):], v2["cases"]):
        mc["v_end"] = c["V_dec_DC_theta_V"] + c["plan"]["end_minus_DC_dec_LSB"] * c["slope_V_per_LSB"]["dec"]
    settle_cmp = [compare_settle(mc, settle) for mc in mine_cases]
    res["item1"] = {}
    for g in ("1a", "1b", "1c", "1d"):
        rr = [r for r in settle_cmp if r["group"] == g]
        res["item1"][g] = {
            "frozen_plan": g == "1a", "n": len(rr), "pass": all(r["pass"] for r in rr), "cases": rr,
            "max_abs_diff_ns": max((abs(r["abs_diff_ns"]) for r in rr if "abs_diff_ns" in r), default=None),
            "max_rel_diff": max((r["rel_diff"] for r in rr if "rel_diff" in r and r["t_main_ns"]), default=None),
            "max_band_diff_LSB": max(max(abs(r["diff_band_lo_LSB"]), abs(r["diff_band_hi_LSB"])) for r in rr if "diff_band_lo_LSB" in r),
            "max_band_diff_V": max(max(abs(r["diff_band_lo_V"]), abs(r["diff_band_hi_V"])) for r in rr if "diff_band_lo_V" in r),
            "max_v_dc_diff_V": max(abs(r["diff_v_dc_V"]) for r in rr if "diff_v_dc_V" in r),
            "max_v_dc_diff_LSB": max(abs(r["diff_v_dc_LSB"]) for r in rr if "diff_v_dc_LSB" in r),
            "max_v_end_diff_LSB": max(abs(r["diff_v_end_LSB"]) for r in rr if "diff_v_end_LSB" in r),
            "max_abs_diff_a": max(abs(r["diff_a"]) for r in rr if "diff_a" in r)}

    # ---- item 2: static power at row 8, x = 0.25 and 0.75
    x = curves["x"]
    idx = {"0.25": 250, "0.75": 750}
    for k, i in idx.items():
        if abs(x[i] - float(k)) > 1e-15:
            raise RuntimeError(f"grid point {i} is {x[i]}, not {k}")
    pc = []
    for r in v1["item2_power"]["cases"]:
        if r["xkey"] not in idx:
            continue
        nd, tm = NODE_MAP[r["node"]]
        key = f"{r['design']}|6T2M|{r['btype']}|{nd}|{tm}"
        p_main = float(curves[key][8, idx[r["xkey"]]]) * 1e6
        rel = abs(r["P_uW"] - p_main) / abs(p_main)
        pc.append({"design": r["design"], "btype": r["btype"], "node": r["node"], "x": r["xkey"], "key": key,
                   "P_mine_uW": r["P_uW"], "P_main_uW": p_main, "rel_diff": rel, "pass": rel <= TOL_POWER_REL})
    aid = power_on_main_grid(v1, pc, args.tmp)
    for j, c in enumerate(pc):
        c.update(aid[j])
    res["item2"] = {"frozen_plan": True, "n": len(pc), "pass": all(c["pass"] for c in pc),
                    "max_rel_diff": max(c["rel_diff"] for c in pc), "cases": pc,
                    "max_rel_diff_after_main_grid_interpolation": max(c["rel_diff_main_vs_mine_interp"] for c in pc),
                    "aid_note": "my exact DC power at the two bracketing 1 mV data-line grid points, linearly "
                                "interpolated at V_DL as the main run does (explanation only; not the check)"}

    # ---- item 3: counts
    c0 = [r for r in v1["item3_counts"]["splits"] if r["split"] == 0][0]
    e0 = [r for r in energy if r["dataset"] == "iris" and int(r["split"]) == 0]
    nr = sorted(set(int(r["n_rows"]) for r in e0))
    nf = sorted(set(int(r["n_features"]) for r in e0))
    if len(nr) != 1 or len(nf) != 1:
        raise RuntimeError(f"iris split 0 rows not unique in energy_dl_ml.csv: {nr}, {nf}")
    main_cells = nr[0] * nf[0]
    i3 = {"frozen_plan_split0": {
        "rows": [c0["rows"], nr[0]], "features": [c0["features"], nf[0]], "cells": [c0["cells"], main_cells],
        "replica_share_256": [c0["replica_share_256"], 256 / (2 * main_cells)], "n_energy_rows_checked": len(e0)}}
    i3["frozen_plan_split0"]["pass"] = (c0["rows"] == nr[0] and c0["features"] == nf[0] and c0["cells"] == main_cells
                                         and c0["replica_share_256"] == 256 / (2 * main_cells))
    oh = [r for r in overhead if r["dataset"] == "iris"]
    if len(oh) != 1:
        raise RuntimeError("no unique iris row in overhead.csv")
    oh = oh[0]
    mn = v1["item3_counts"]["mean_over_splits"]
    pairs = {"rows_mean": (mn["rows"], float(oh["rows_mean"])), "features": (mn["features"], float(oh["features"])),
             "cells": (mn["cells"], float(oh["cells"])), "pcm_devices": (mn["pcm_devices"], float(oh["pcm_devices"])),
             "programmed_bounds_mean": (mn["programmed_bounds"], float(oh["programmed_bounds_mean"])),
             "replica_64_share (64 / mean devices)": (64 / mn["pcm_devices"], float(oh["replica_64_share"])),
             "replica_256_share (256 / mean devices)": (256 / mn["pcm_devices"], float(oh["replica_256_share"]))}
    i3["additional_means"] = {k: {"mine": a, "main": b, "identical": a == b, "rel_diff": abs(a - b) / abs(b)}
                              for k, a, b in ((k, *p) for k, p in pairs.items())}
    i3["additional_means_pass"] = all(e["identical"] for e in i3["additional_means"].values())
    i3["note_mean_of_shares"] = {"replica_256_mean_of_split_shares_mine": mn["replica_share_256"],
                                 "replica_64_mean_of_split_shares_mine": mn["replica_share_64"],
                                 "comment": "overhead.csv uses 256 (64) / mean PCM devices, not the mean of the per-split shares"}
    res["item3"] = i3

    # ---- switching points from run.log
    m = re.search(r"V_sw\s+([0-9.]+)\s*V,\s*V_Y\*\s+([0-9.]+)\s*V,\s*V_X\*\s+([0-9.]+)\s*V", runlog)
    sw_mine = v1["item0_switching_points"]
    sp = {}
    if m:
        for k, val in zip(("V_sw", "V_Y_star", "V_X_star"), m.groups()):
            nd = len(val.split(".")[1])
            mv = float(val)
            mine_v = sw_mine[k]["value_V"]
            half = 0.5 * 10 ** (-nd)
            sp[k] = {"main_printed_V": val, "mine_V": mine_v, "mine_rounded": f"{mine_v:.{nd}f}",
                     "diff_to_printed_V": mine_v - mv,
                     "implied_diff_range_V": [mine_v - (mv + half), mine_v - (mv - half)],
                     "consistent_within_print_resolution": abs(mine_v - mv) <= half}
    res["switching_points"] = sp
    res["main_checks"] = {k: v.get("pass") for k, v in checks.items()}
    res["overall_pass_frozen_plan"] = bool(res["item1"]["1a"]["pass"] and res["item2"]["pass"]
                                           and res["item3"]["frozen_plan_split0"]["pass"])
    res["overall_pass_additional"] = bool(all(res["item1"][g]["pass"] for g in ("1b", "1c", "1d"))
                                          and res["item3"]["additional_means_pass"])
    (mine_dir / "compare.json").write_text(json.dumps(res, indent=1) + "\n")
    write_report(mine_dir / "report.md", res, v2)
    print(json.dumps({"1a": res["item1"]["1a"]["pass"], "1b": res["item1"]["1b"]["pass"], "1c": res["item1"]["1c"]["pass"],
                      "1d": res["item1"]["1d"]["pass"], "item2": res["item2"]["pass"],
                      "item3": res["item3"]["frozen_plan_split0"]["pass"], "item3_means": res["item3"]["additional_means_pass"]}))
    return 0


def fnum(v, nd=4):
    return "–" if v is None else f"{v:.{nd}f}"


def write_report(path: Path, res: dict, v2: dict) -> None:
    L = ["# E6 independent check: comparison with the run of record", ""]
    mt = res["meta"]
    L.append(f"- Main run of record: `{mt['main_dir']}`; files read: {', '.join(mt['main_files_read'])} (nothing else)")
    L.append(f"- Independent values: `{mt['mine_dir']}/values.json` (phase 1, 09:37 JST) and `extra_lbuf.json` (1d, written "
             f"10:48 JST before any main-run file was opened)")
    L.append(f"- Created {mt['created_jst']} JST by `experiments/E6_independent/compare.py` (SHA-256 `{mt['script_sha256']}`)")
    L.append(f"- Main-run checks (checks.json): " + ", ".join(f"{k} {'pass' if v else 'FAIL'}" for k, v in res["main_checks"].items()))
    L.append("")
    L.append("## Summary")
    L.append("")
    L.append("| Item | Status | Criterion | Result | Largest difference |")
    L.append("|---|---|---|---|---|")
    for g, label in (("1a", "1a settling (frozen plan)"), ("1b", "1b settling, L6T θ = 1 (additional)"),
                     ("1c", "1c settling, UINV θ = 0 (additional)"), ("1d", "1d settling, LBUF θ = 1 (additional)")):
        e = res["item1"][g]
        L.append(f"| {label} | {'frozen plan' if e['frozen_plan'] else 'additional'} | ≤ 2 % or ≤ 0.1 ns | "
                 f"{'PASS' if e['pass'] else 'FAIL'} ({e['n']}) | {e['max_abs_diff_ns'] * 1e3:.3f} ps; rel {e['max_rel_diff'] * 100:.4f} %; "
                 f"band ≤ {e['max_band_diff_LSB']:.1e} LSB, v_dc ≤ {e['max_v_dc_diff_LSB']:.1e} LSB |")
    e = res["item2"]
    L.append(f"| 2 static power (frozen plan) | frozen plan | ≤ 0.5 % | {'PASS' if e['pass'] else 'FAIL'} ({e['n']}) | "
             f"rel {e['max_rel_diff'] * 100:.5f} % |")
    e = res["item3"]["frozen_plan_split0"]
    L.append(f"| 3 iris split 0 counts (frozen plan) | frozen plan | identical | {'PASS' if e['pass'] else 'FAIL'} | "
             f"rows {e['rows'][0]}/{e['rows'][1]}, features {e['features'][0]}/{e['features'][1]}, cells {e['cells'][0]}/{e['cells'][1]}, "
             f"256/(2 cells) {e['replica_share_256'][0]:.6f}/{e['replica_share_256'][1]:.6f} |")
    L.append(f"| 3 means over splits (additional) | additional | identical | "
             f"{'PASS' if res['item3']['additional_means_pass'] else 'FAIL'} | see below |")
    L.append("")
    L.append("## Item 1: settling times (ns; mine at the plan's numerics; main = settle.csv, c_par_f = 5e-16)")
    L.append("")
    L.append("| Group | Case | Design | Type | Node | Level | x_prev | Mine | Main | Diff (ps) | Rel | Pass | Δband lo/hi (µV) | Δv_dc (µV) | Mine converged |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for g in ("1a", "1b", "1c", "1d"):
        for r in res["item1"][g]["cases"]:
            L.append(f"| {g} | {r['id']} | {r['design']} | {r['btype']} | {r['node']} | {r['level']} | {r['x_prev']:.0f} | "
                     f"{fnum(r['t_ns'])} | {fnum(r.get('t_main_ns'))} | {r.get('abs_diff_ns', float('nan')) * 1e3:.3f} | "
                     f"{r.get('rel_diff', float('nan')) * 100:.4f} % | {'yes' if r['pass'] else 'NO'} | "
                     f"{r['diff_band_lo_V'] * 1e6:+.3f} / {r['diff_band_hi_V'] * 1e6:+.3f} | {r['diff_v_dc_V'] * 1e6:+.3f} | "
                     f"{fnum(r.get('t_converged_ns'))} |")
    L.append("")
    L.append("Δ = mine − main. 'Mine converged' = extrapolated from the 10 ps and 5 ps maximum-step runs (trapezoidal); "
             "the comparison uses the plan's 20 ps value.")
    L.append("")
    L.append("## 1d additional values (LBUF, GS cell set, θ = 1, x_prev = 0)")
    L.append("")
    L.append("| Case | Design | Node | G (µS) | V_SL (V) | t_settle ±0.01 LSB (ns) | t_settle ±0.1 LSB (ns) | τ mid node (ns) | τ decision node (ns) | τ X (ns) |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for c in v2["cases"]:
        p = c["plan"]

        def taus(nm):
            return " / ".join("–" if w["tau_s"] is None else f"{w['tau_s'] * 1e9:.2f}" for w in p[f"tau_{nm}"])
        L.append(f"| {c['id']} | {c['design']} | {c['node']} | {c['G_uS']:.6f} | {c['V_SL_V']:.6f} | "
                 f"{p['t_settle_0p01_s'] * 1e9:.4f} | {p['t_settle_0p1_s'] * 1e9:.4f} | {taus('mid')} | {taus('dec')} | {taus('X')} |")
    L.append("")
    L.append("τ: fit of ln|v − v_DC(θ)| where the node's input-referred distance is 0.003–0.03 LSB (last part of the approach) / "
             "0.0003–0.003 LSB (tail). The approach is set by one slow pole of about 51 ns at the first INV_S output (it drives the "
             "second INV_S input); the decision node follows it, X is fast and only dragged through the first INV_S's gate-drain "
             "capacitance.")
    L.append("")
    L.append("## Item 2: static power (µW), row 8 (θ = 8/15), x = 0.25 (index 250) and 0.75 (index 750)")
    L.append("")
    L.append("| Design | Type | Node | x | Mine | Main | Rel diff | Mine on the main's 1 mV DL grid | Rel diff to that |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for c in res["item2"]["cases"]:
        L.append(f"| {c['design']} | {c['btype']} | {c['node']} | {c['x']} | {c['P_mine_uW']:.7f} | {c['P_main_uW']:.7f} | "
                 f"{c['rel_diff'] * 100:.5f} % | {c['P_mine_interp_uW']:.7f} | {c['rel_diff_main_vs_mine_interp'] * 100:.6f} % |")
    L.append("")
    L.append(f"Explanation aid (not the check): the main run sweeps the data line in 1 mV steps and interpolates linearly at "
             f"V_DL = M(x). Doing the same with my own DC power at the two bracketing grid points reduces the largest "
             f"difference from {res['item2']['max_rel_diff'] * 100:.5f} % to "
             f"{res['item2']['max_rel_diff_after_main_grid_interpolation'] * 100:.6f} %: the differences are mainly the "
             f"interpolation bias of a convex P(V_DL) (main always higher), not a model difference; the rest is at the level "
             f"of the sub-µV differences in the switching points and data-line maps.")
    L.append("")
    L.append("## Item 3: counts")
    L.append("")
    L.append("| Quantity (iris, mean over splits 0–2) | Mine | Main (overhead.csv) | Identical |")
    L.append("|---|---|---|---|")
    for k, e in res["item3"]["additional_means"].items():
        L.append(f"| {k} | {e['mine']!r} | {e['main']!r} | {'yes' if e['identical'] else 'no (rel ' + format(e['rel_diff'], '.1e') + ')'} |")
    L.append("")
    nm = res["item3"]["note_mean_of_shares"]
    L.append(f"overhead.csv's shares are 256 (64) / mean PCM devices; the mean of the per-split shares would be "
             f"{nm['replica_256_mean_of_split_shares_mine']:.6f} ({nm['replica_64_mean_of_split_shares_mine']:.6f}).")
    L.append("")
    L.append("## Switching points (run.log prints 6 decimals)")
    L.append("")
    L.append("| Quantity | Main (printed) | Mine | Mine − printed (µV) | Implied range of mine − main (µV) | Within print resolution |")
    L.append("|---|---|---|---|---|---|")
    for k, e in res["switching_points"].items():
        L.append(f"| {k} | {e['main_printed_V']} | {e['mine_V']:.9f} | {e['diff_to_printed_V'] * 1e6:+.3f} | "
                 f"{e['implied_diff_range_V'][0] * 1e6:+.3f} to {e['implied_diff_range_V'][1] * 1e6:+.3f} | "
                 f"{'yes' if e['consistent_within_print_resolution'] else 'no'} |")
    L.append("")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
