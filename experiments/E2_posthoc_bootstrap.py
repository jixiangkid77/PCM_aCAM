"""Post-hoc analysis (not part of the frozen plans): sampling uncertainty of the Monte Carlo rms bound errors of the
transistor-level tests, and the spread of the pull-down offsets between the random draws of the experiments.

For every test case I resample the 300 instances with replacement (10,000 resamples, fixed seed) and report the
percentile 95 % interval of the rms error over instances and the 16 thresholds, one year after programming, at the
input remap and the divider supply. The frozen predictions are compared with these intervals for information only;
the decisions of the plans are unchanged.

Cases: E2 (14 prospective conditions, lower branch, PDK mismatch), E3b (5 upper-branch conditions), E1 (design points
P1-P4, run of record), the E2 calibration on the E1 design points, and the linear-read case of E3a.

Check R1: the rms of every case, recomputed from the stored per-instance arrays, must equal the stored summary value
to within 1e-6 LSB.

Usage (from the repository root):
    python experiments/E2_posthoc_bootstrap.py
Output (never overwritten): results/E2_posthoc_bootstrap_<stamp>/
    bootstrap_ci.csv   one row per (experiment, condition, node)
    offsets.csv        standard deviation of the switching-point offsets of the pull-downs in each draw
    check_R1.json      reproduction of the stored rms values
    summary.md         short tables
    run_info.json      environment and SHA-256 of the inputs and of this script
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pcm_acam.io_utils import environment_info, make_logger, make_run_dir, now_jst, sha256_file, write_json  # noqa: E402

RES = ROOT / "results"
E1 = RES / "E1_20261014_0156"
E2 = RES / "E2_20261014_0316"
E2CAL = RES / "E2cal_E1_20261014_0303"
E3A = RES / "E3a_20261014_0442"
E3B = RES / "E3b_20261014_0532"
E1_NODES = ("none", "dac_remap", "dac_gain", "supply", "ideal")
E1_TIMES = ("1d", "30d", "1y")
N_BOOT = 10000
SEED = 20261113


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


def rms_of(e: np.ndarray, valid: np.ndarray) -> float:
    return float(np.sqrt(np.mean(e[valid] ** 2)))


def bootstrap(e: np.ndarray, valid: np.ndarray, rng: np.random.Generator) -> Tuple[float, float, float]:
    """Resample instances (rows); each resample keeps the valid pairs of the drawn instances."""
    sq = np.where(valid, e.astype(float) ** 2, 0.0).sum(axis=1)
    cnt = valid.sum(axis=1).astype(float)
    n = e.shape[0]
    idx = rng.integers(0, n, size=(N_BOOT, n))
    boot = np.sqrt(sq[idx].sum(axis=1) / cnt[idx].sum(axis=1))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return rms_of(e, valid), float(lo), float(hi)


def main() -> int:
    run = make_run_dir(RES, "E2_posthoc_bootstrap", False)
    log = make_logger(run / "run.log")
    inputs = [E1 / "summary.csv", E2 / "compare.csv", E2 / "summary.csv", E2CAL / "compare.csv", E3A / "summary.csv",
              E3B / "compare.csv", Path(__file__)]
    inputs += sorted(E1.glob("arrays_P*.npz")) + sorted(E2.glob("arrays_*.npz")) + sorted(E2CAL.glob("arrays_*.npz"))
    inputs += sorted(E3A.glob("arrays_*.npz")) + sorted(E3B.glob("arrays_*.npz"))
    info = {"analysis": "post hoc: bootstrap intervals of the Monte Carlo rms errors; pull-down offset spread",
            "start_jst": now_jst().isoformat(timespec="seconds"), "environment": environment_info(),
            "n_boot": N_BOOT, "seed": SEED,
            "sha256": {str(p.relative_to(ROOT)): sha256_file(p) for p in inputs}}
    write_json(run / "run_info.json", info)
    rng = np.random.default_rng(SEED)
    rows, offs, r1 = [], [], []

    def add(exp: str, cond: str, node: str, e: np.ndarray, valid: np.ndarray, stored: float,
            pred: Optional[float], tol: Optional[float]) -> None:
        m, lo, hi = bootstrap(e, valid, rng)
        r1.append({"experiment": exp, "condition": cond, "node": node, "recomputed": m, "stored": stored,
                   "abs_diff": abs(m - stored)})
        row = {"experiment": exp, "condition": cond, "node": node, "n_instances": int(e.shape[0]),
               "rms_lsb": m, "ci_lo_lsb": lo, "ci_hi_lsb": hi, "ci_rel_lo": lo / m - 1.0, "ci_rel_hi": hi / m - 1.0}
        if pred is not None:
            row.update({"pred_rms_lsb": pred, "pred_in_ci": bool(lo <= pred <= hi),
                        "tolerance_lsb": tol, "pred_minus_mc_lsb": pred - m})
        rows.append(row)

    # E1 design points, PDK mismatch, one year
    e1 = {(r["design"], r["setting"], r["node"], r["time"]): float(r["rms"]) for r in rcsv(E1 / "summary.csv")}
    for d in ("P1", "P2", "P3", "P4"):
        z = np.load(E1 / f"arrays_{d}.npz")
        ti = E1_TIMES.index("1y")
        for node in ("dac_remap", "supply"):
            ni = E1_NODES.index(node)
            add("E1", d, node, z["err_pdk"][ni, ti], z["valid_pdk"][ni, ti], e1[(d, "pdk", node, "1y")], None, None)
        offs.append({"experiment": "E1", "condition": d, "branch": "lower", "sigma_mV": 1e3 * float(np.std(z["vsw_pdk"], ddof=1))})

    # E2 prospective test and the E2 calibration on the E1 design points
    for exp, folder in (("E2", E2), ("E2cal", E2CAL)):
        comp = {(r["condition"], r["node"]): r for r in rcsv(folder / "compare.csv")
                if r["time"] == "1y" and r["setting"] == "pdk"}
        for f in sorted(folder.glob("arrays_*.npz")):
            cond = f.stem.split("_", 1)[1]
            z = np.load(f)
            for node in ("dac_remap", "supply"):
                e = z[f"err_pdk_{node}_1y"]
                valid = np.isfinite(e)
                c = comp.get((cond, node))
                if c is None:
                    continue
                add(exp, cond, node, e, valid, float(c["sim_rms_lsb"]), float(c["pred_rms_lsb"]), float(c["tolerance_lsb"]))
            offs.append({"experiment": exp, "condition": cond, "branch": "lower",
                         "sigma_mV": 1e3 * float(np.std(z["vsw_mc"], ddof=1))})

    # E3a, linear read (the reference case of the nonlinearity study)
    e3a = {(r["design"], r["node"]): float(r["rms"]) for r in rcsv(E3A / "summary.csv")
           if r["time"] == "1y" and r["grid"] == "linear"}
    for f in sorted(E3A.glob("arrays_*.npz")):
        cond = f.stem.split("_", 1)[1]
        z = np.load(f)
        for node, key in (("dac_remap", "dac_remap"), ("supply", "supply_lin")):
            e = z[f"err_{key}_1y_linear"]
            add("E3a_linear", cond, node, e, np.isfinite(e), e3a[(cond, key)], None, None)
        offs.append({"experiment": "E3a", "condition": cond, "branch": "lower",
                     "sigma_mV": 1e3 * float(np.std(z["vsw_mc"], ddof=1))})

    # E3b upper-bound branch
    comp = {(r["condition"], r["node"]): r for r in rcsv(E3B / "compare.csv") if r["time"] == "1y" and r["setting"] == "pdk"}
    for f in sorted(E3B.glob("arrays_*.npz")):
        cond = f.stem.split("_", 1)[1]
        z = np.load(f)
        for node in ("dac_remap", "supply"):
            e = z[f"err_pdk_{node}_1y"]
            c = comp[(cond, node)]
            add("E3b", cond, node, e, np.isfinite(e), float(c["sim_rms_lsb"]), float(c["pred_rms_lsb"]), float(c["tolerance_lsb"]))
        offs.append({"experiment": "E3b", "condition": cond, "branch": "upper (trip point)",
                     "sigma_mV": 1e3 * float(np.std(z["vy_mc"], ddof=1))})

    worst = max(r["abs_diff"] for r in r1)
    ok = worst <= 1e-6
    write_json(run / "check_R1.json", {"cases": len(r1), "max_abs_diff_lsb": worst, "pass": ok, "rows": r1})
    log(f"R1: {len(r1)} cases, largest |recomputed - stored| {worst:.2e} LSB")
    if not ok:
        log("R1 failed; no intervals written")
        return 2
    write_csv(run / "bootstrap_ci.csv", rows)
    write_csv(run / "offsets.csv", offs)

    lines = ["# Bootstrap 95 % intervals of the Monte Carlo rms errors (one year) and pull-down offset spread (post hoc)", ""]
    for exp in ("E1", "E2cal", "E2", "E3a_linear", "E3b"):
        sub = [r for r in rows if r["experiment"] == exp]
        if not sub:
            continue
        lines += [f"## {exp}", "", "| condition | node | rms (LSB) | 95 % interval | relative | prediction | in interval |",
                  "|---|---|---|---|---|---|---|"]
        for r in sub:
            pred = f"{r['pred_rms_lsb']:.3f}" if "pred_rms_lsb" in r else "-"
            inside = str(r["pred_in_ci"]) if "pred_in_ci" in r else "-"
            lines.append(f"| {r['condition']} | {r['node']} | {r['rms_lsb']:.3f} | [{r['ci_lo_lsb']:.3f}, {r['ci_hi_lsb']:.3f}] | "
                         f"{100 * r['ci_rel_lo']:+.1f} / {100 * r['ci_rel_hi']:+.1f} % | {pred} | {inside} |")
        for node in ("dac_remap", "supply"):
            s = [r for r in sub if r["node"] == node]
            lines.append(f"- {node}: relative half-width {100 * min((r['ci_rel_hi'] - r['ci_rel_lo']) / 2 for r in s):.1f}"
                         f"-{100 * max((r['ci_rel_hi'] - r['ci_rel_lo']) / 2 for r in s):.1f} %")
        lines.append("")
    lines += ["## Pull-down offsets (standard deviation of the switching-point offset in each draw)", "",
              "| experiment | condition | branch | sigma (mV) |", "|---|---|---|---|"]
    lines += [f"| {o['experiment']} | {o['condition']} | {o['branch']} | {o['sigma_mV']:.2f} |" for o in offs]
    (run / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    info["end_jst"] = now_jst().isoformat(timespec="seconds")
    write_json(run / "run_info.json", info)
    log(f"done: {run.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
