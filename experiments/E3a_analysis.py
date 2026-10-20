"""E3a post-run analysis (secondary quantities named in the frozen plan; no new simulation).

Usage: python experiments/E3a_analysis.py results/E3a_<stamp>
Writes into the run folder (new files only):
  analysis_crossover.csv   for each design, supply node and beta: the grid interval (going from the linear PCM
                           towards stronger nonlinearity) in which the node stops being better than the input remap,
                           from the Monte Carlo and from the frozen predictions
  analysis_systematic.csv  per-theta systematic error 15 (G/dG)(rho0 - 1) of the supply nodes (frozen predictions)
  analysis_times.csv       B2-type pass rates and B3-type agreement at 1 day and 30 days
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def rcsv(p: Path):
    with open(p, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def wcsv(p: Path, rows):
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    run = Path(sys.argv[1])
    outs = [run / n for n in ("analysis_crossover.csv", "analysis_systematic.csv", "analysis_times.csv")]
    for p in outs:
        if p.exists():
            sys.exit(f"{p} exists; refusing to overwrite")
    info = json.loads((run / "run_info.json").read_text())
    preds = json.loads((ROOT / info["predictions_file"]).read_text())
    cfg = json.loads((run / "config_used.json").read_text())
    cmp_rows = rcsv(run / "compare.csv")
    v0_order = [np.inf if v == "inf" else float(v) for v in cfg["grid"]["v0_ref_v"]]  # linear first, then decreasing V0

    def get(design, time, grid, node):
        for r in cmp_rows:
            if (r["design"], r["time"], r["grid"], r["node"]) == (design, time, grid, node):
                return r
        raise KeyError((design, time, grid, node))

    def lab(v0, beta):
        return "linear" if not np.isfinite(v0) else f"V0={v0:g}|beta={beta:+g}"

    cross = []
    for d in cfg["designs"]:
        did = d["id"]
        for node in ("supply_rep", "supply_lin"):
            for beta in cfg["grid"]["beta"]:
                for src, key in (("monte_carlo", "sim_rms_lsb"), ("prediction", "pred_rms_lsb")):
                    dac = float(get(did, "1y", "linear", "dac_remap")[key])
                    prev, found = None, None
                    seq = []
                    for v0 in v0_order:
                        r = get(did, "1y", lab(v0, beta), node)
                        val = float(r[key])
                        seq.append((v0, val))
                        better = np.isfinite(val) and val < dac
                        if prev is not None and prev[1] and not better and found is None:
                            found = (prev[0], v0)
                        prev = (v0, better)
                    first_better = seq[0][1] < dac
                    always_better = all(np.isfinite(v) and v < dac for _, v in seq)
                    never_better = all(not (np.isfinite(v) and v < dac) for _, v in seq)
                    cross.append({"design": did, "node": node, "beta": beta, "source": src, "dac_rms_lsb": dac,
                                  "better_than_dac_for_linear_pcm": bool(first_better),
                                  "better_than_dac_on_whole_grid": bool(always_better),
                                  "never_better_than_dac": bool(never_better),
                                  "loses_between_v0_ref_V": "" if found is None else f"{found[0]:g} -> {found[1]:g}",
                                  "rms_by_v0_ref": "; ".join(f"{v0:g}:{v:.3f}" for v0, v in seq)})
    syst = []
    theta = np.arange(16) / 15.0
    g_lo, g_hi = cfg["circuit"]["g_lo_us"], cfg["circuit"]["g_hi_us"]
    w = (g_lo + theta * (g_hi - g_lo)) / (g_hi - g_lo)
    for did, e in preds.items():
        for grid, res in e["pred"]["1y"].items():
            for node in ("supply_lin", "supply_rep"):
                r0 = np.array(res[node]["rho0"])
                for j in range(16):
                    syst.append({"design": did, "grid": grid, "node": node, "theta": round(float(theta[j]), 6),
                                 "rho0": float(r0[j]), "systematic_error_lsb": float(15.0 * w[j] * (r0[j] - 1.0))})
    times = []
    for tn in cfg["drift"]["times_s"]:
        sel = [r for r in cmp_rows if r["time"] == tn and r["node"] in ("supply_lin", "supply_rep")
               and float(r["invalid_frac"]) <= 0.01 and r["pred_rms_lsb"] not in ("nan", "")]
        rate = float(np.mean([r["pass"] == "True" for r in sel])) if sel else float("nan")
        rel = [abs(float(r["rel_diff"])) for r in sel if r["rel_diff"] not in ("nan", "")]
        agree = clear = 0
        for d in cfg["designs"]:
            for r in [x for x in cmp_rows if x["time"] == tn and x["design"] == d["id"] and x["node"] == "supply_rep"]:
                if float(r["invalid_frac"]) > 0.01:
                    continue
                rd = get(d["id"], tn, r["grid"], "dac_remap")
                ps, pd_ = float(r["pred_rms_lsb"]), float(rd["pred_rms_lsb"])
                ss, sd = float(r["sim_rms_lsb"]), float(rd["sim_rms_lsb"])
                if abs(ps - pd_) >= 0.2 * max(ps, pd_):
                    clear += 1
                    agree += (ps < pd_) == (ss < sd)
        times.append({"time": tn, "n_usable": len(sel), "pass_rate": rate, "max_abs_rel_diff": max(rel) if rel else np.nan,
                      "n_clear": clear, "n_clear_agree": agree})
    for p, rows in zip(outs, (cross, syst, times)):
        wcsv(p, rows)
    for r in cross:
        if r["source"] == "monte_carlo":
            print(f"{r['design']} {r['node']:10} beta {r['beta']:+.1f}: dac {r['dac_rms_lsb']:.3f}  "
                  f"better(linear) {r['better_than_dac_for_linear_pcm']}  loses {r['loses_between_v0_ref_V'] or '-'}  "
                  f"always {r['better_than_dac_on_whole_grid']} never {r['never_better_than_dac']}")
    for r in times:
        print(r)
    return 0


if __name__ == "__main__":
    sys.exit(main())
