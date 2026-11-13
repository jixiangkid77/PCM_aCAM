"""E5 post-hoc analysis (not part of the frozen E5 plan): time until an rms error budget is exceeded, as a function of
the budget.

The E5 plan evaluated the budgets 0.5, 1.0 and 1.5 LSB. Here I repeat its interpolation (linear in the error versus
log t between the 13 stored times, 10 s to 10 years) on a grid of budgets from 0.60 to 3.00 LSB, from the stored
error-versus-time curves of the run of record. No bound is recomputed.

Check R1: the interpolation must reproduce every rms entry of e54_budget.csv of the run of record (same status and,
where interpolated, the same time to within 1e-9 relative).

The array has to be reprogrammed as soon as one of its branch types exceeds the budget, so I also report, for each cell
set, the earlier of the times of its lower-bound and upper-bound type (the reprogramming interval of the array).

Usage (from the repository root):
    python experiments/E5_posthoc_budget.py
Output (never overwritten): results/E5_posthoc_budget_<stamp>/
    budget_sweep.csv    every (design, cell set, branch type, condition, node, budget): status, time, reprogrammings/year
    array_interval.csv  every (design, cell set, condition, node, budget): earlier of the two branch types
    check_R1.json       reproduction of the stored budget table
    summary.md          lower bounds and arrays of both cell sets, condition 'all' and 'unif', selected budgets
    run_info.json       environment and SHA-256 of the inputs and of this script
"""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pcm_acam.io_utils import environment_info, make_logger, make_run_dir, now_jst, sha256_file, write_json  # noqa: E402

RUN = ROOT / "results" / "E5_20261024_0210"
CURVES = RUN / "e54_curves.csv"
BUDGET = RUN / "e54_budget.csv"
YEAR_S = 365.0 * 86400.0
GRID = np.round(np.arange(0.60, 3.00 + 1e-9, 0.05), 2)
SUMMARY_BUDGETS = (1.0, 1.25, 1.5, 1.75, 2.0, 2.5)


def rcsv(path: Path) -> List[dict]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[dict]) -> None:
    keys: List[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def time_to_budget(t_s: List[float], vals: List[float], v_t0: float, budget: float) -> Tuple[str, float]:
    """The E5 rule: exceeded at t0, before the first stored time, interpolated, or beyond the last stored time."""
    if v_t0 > budget:
        return "exceeded at t0", 0.0
    for i, v in enumerate(vals):
        if v > budget:
            if i == 0:
                return "< 10 s", math.nan
            lt0, lt1 = math.log(t_s[i - 1]), math.log(t_s[i])
            return "interpolated", math.exp(lt0 + (budget - vals[i - 1]) / (v - vals[i - 1]) * (lt1 - lt0))
    return "> 10 y", math.inf


def fmt(t: float) -> str:
    if t < 60:
        return f"{t:.3g} s"
    if t < 3600:
        return f"{t / 60:.3g} min"
    if t < 86400:
        return f"{t / 3600:.3g} h"
    if t < YEAR_S:
        return f"{t / 86400:.3g} d"
    return f"{t / YEAR_S:.3g} y"


def main() -> int:
    run = make_run_dir(ROOT / "results", "E5_posthoc_budget", False)
    log = make_logger(run / "run.log")
    info = {"analysis": "E5 post hoc: time to an rms budget versus the budget", "start_jst": now_jst().isoformat(timespec="seconds"),
            "environment": environment_info(),
            "sha256": {str(p.relative_to(ROOT)): sha256_file(p) for p in (CURVES, BUDGET, Path(__file__))}}
    write_json(run / "run_info.json", info)

    curves: Dict[tuple, Dict[str, dict]] = {}
    for r in rcsv(CURVES):
        key = (r["design"], r["cell_set"], r["type"], r["condition"])
        curves.setdefault(key, {}).setdefault(r["node"], {})[r["time"]] = r
    groups = sorted(curves)

    def series(key: tuple, node: str) -> Tuple[List[float], List[float], float]:
        c = curves[key]
        v_t0 = float(c["t0"]["t0"]["rms_noext_lsb"])
        pts = sorted(((float(r["t_s"]), float(r["rms_noext_lsb"])) for lab, r in c[node].items()), key=lambda p: p[0])
        return [p[0] for p in pts], [p[1] for p in pts], v_t0

    # check R1: reproduce the stored table
    mism = []
    n_cmp = 0
    for r in rcsv(BUDGET):
        if r["metric"] != "rms_noext":
            continue
        key = (r["design"], r["cell_set"], r["type"], r["condition"])
        t_s, vals, v_t0 = series(key, r["node"])
        st, tb = time_to_budget(t_s, vals, v_t0, float(r["budget"]))
        n_cmp += 1
        stored = r["status"]
        ok = (st == stored) if st != "interpolated" else (stored == "interpolated" and
                                                           abs(tb / float(r["t_budget_s"]) - 1.0) <= 1e-9)
        if st == "< 10 s" and stored.startswith("<"):
            ok = True
        if st == "> 10 y" and stored.startswith(">"):
            ok = True
        if not ok:
            mism.append({"row": r, "recomputed": [st, tb]})
    write_json(run / "check_R1.json", {"compared": n_cmp, "mismatches": len(mism), "pass": not mism, "details": mism[:20]})
    log(f"R1: {n_cmp} stored rms entries compared, {len(mism)} mismatches")
    if mism:
        log("R1 failed; no sweep written")
        return 2

    rows = []
    for key in groups:
        for node in ("none", "dac_remap", "supply", "ideal"):
            t_s, vals, v_t0 = series(key, node)
            for b in GRID:
                st, tb = time_to_budget(t_s, vals, v_t0, float(b))
                rows.append({"design": key[0], "cell_set": key[1], "type": key[2], "condition": key[3], "node": node,
                             "budget_lsb": float(b), "rms_t0_lsb": v_t0, "status": st,
                             "t_budget_s": tb if st == "interpolated" else "",
                             "reprogrammings_per_year": YEAR_S / tb if st == "interpolated" else ""})
    write_csv(run / "budget_sweep.csv", rows)
    log(f"budget sweep: {len(groups)} groups x 4 nodes x {GRID.size} budgets")

    get = {(r["design"], r["cell_set"], r["type"], r["condition"], r["node"], r["budget_lsb"]): r for r in rows}

    # reprogramming interval of the array: the earlier of its lower-bound and upper-bound branch type
    def as_time(r: dict) -> float:
        if r["status"] == "interpolated":
            return float(r["t_budget_s"])
        if r["status"].startswith(">"):
            return math.inf
        return 0.0                      # exceeded at t0, or before the first stored time

    lower_type = {"6T2M": "L6T", "GS": "LBUF"}
    arr_rows = []
    for (d, cs, typ, cond) in groups:
        if typ != lower_type[cs]:
            continue
        for node in ("none", "dac_remap", "supply", "ideal"):
            for b in GRID:
                rl, ru = get[(d, cs, typ, cond, node, float(b))], get[(d, cs, "UINV", cond, node, float(b))]
                tl, tu = as_time(rl), as_time(ru)
                first, r_first = ("lower", rl) if tl <= tu else ("upper", ru)
                tmin = min(tl, tu)
                st = "interpolated" if 0 < tmin < math.inf else r_first["status"]
                arr_rows.append({"design": d, "cell_set": cs, "condition": cond, "node": node, "budget_lsb": float(b),
                                 "first_branch": first, "status": st, "t_budget_s": tmin if st == "interpolated" else "",
                                 "reprogrammings_per_year": YEAR_S / tmin if st == "interpolated" else ""})
    write_csv(run / "array_interval.csv", arr_rows)
    aget = {(r["design"], r["cell_set"], r["condition"], r["node"], r["budget_lsb"]): r for r in arr_rows}

    lines = ["# Time to an rms budget versus the budget (post hoc, from e54_curves.csv of E5_20261024_0210)", "",
             "Lower bounds; rms over all bounds without the bound failures (|e| > 15 LSB), as in E5.", ""]
    for cond, title in (("all", "with the PCM noise model"), ("unif", "without PCM noise (uniform drift)")):
        for cs, typ in (("6T2M", "L6T"), ("GS", "LBUF")):
            lines += [f"## {cs} cell, {typ}, {title}", "",
                      "| design | budget (LSB) | rms at t0 | input remap | divider supply | ideal scaling | supply / remap |",
                      "|---|---|---|---|---|---|---|"]
            for d in ("P1", "P3"):
                for b in SUMMARY_BUDGETS:
                    cells = []
                    t = {}
                    for node in ("dac_remap", "supply", "ideal"):
                        r = get[(d, cs, typ, cond, node, b)]
                        if r["status"] == "interpolated":
                            t[node] = float(r["t_budget_s"])
                            cells.append(fmt(t[node]))
                        else:
                            t[node] = math.inf if r["status"].startswith(">") else 0.0
                            cells.append(r["status"])
                    ratio = (f"{t['supply'] / t['dac_remap']:.2f}" if 0 < t["dac_remap"] < math.inf and
                             0 < t["supply"] < math.inf else "-")
                    r0 = get[(d, cs, typ, cond, "dac_remap", b)]
                    lines.append(f"| {d} | {b:.2f} | {float(r0['rms_t0_lsb']):.3f} | " + " | ".join(cells) + f" | {ratio} |")
            lines.append("")
    lines += ["# Reprogramming interval of the array (earlier of the lower-bound and upper-bound branch type)", ""]
    for cond, title in (("all", "with the PCM noise model"), ("unif", "without PCM noise (uniform drift)")):
        for cs in ("6T2M", "GS"):
            lines += [f"## {cs} cell, {title}", "",
                      "| design | budget (LSB) | input remap | divider supply | ideal scaling | better node, factor |",
                      "|---|---|---|---|---|---|"]
            for d in ("P1", "P3"):
                for b in SUMMARY_BUDGETS:
                    cells, t = [], {}
                    for node in ("dac_remap", "supply", "ideal"):
                        r = aget[(d, cs, cond, node, b)]
                        t[node] = as_time(r) if r["status"] != "interpolated" else float(r["t_budget_s"])
                        cells.append((fmt(t[node]) if r["status"] == "interpolated" else r["status"]) + f" ({r['first_branch']})")
                    a_, s_ = t["dac_remap"], t["supply"]
                    if 0 < a_ < math.inf and 0 < s_ < math.inf:
                        better = f"{'supply' if s_ > a_ else 'remap'}, {max(a_, s_) / min(a_, s_):.2f}"
                    else:
                        better = "-"
                    lines.append(f"| {d} | {b:.2f} | " + " | ".join(cells) + f" | {better} |")
            lines.append("")
    (run / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    info["end_jst"] = now_jst().isoformat(timespec="seconds")
    write_json(run / "run_info.json", info)
    log(f"done: {run.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
