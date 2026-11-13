"""E4 post-hoc analysis (not part of the frozen E4 plan): the node difference of the forest accuracy loss for each
dataset, with a 95 % interval over its 3 splits x 20 device seeds, and the mean over the datasets without the synthetic
one.

Difference: loss(input remap) - loss(divider supply), percentage points, paired over identical draws, one year after
programming; positive means that the supply node is better. The per-dataset interval is mean +- 1.96 SE over the 60
(split, seed) units of the dataset, the same unit and normal approximation as the plan uses for the mean over datasets.

Check R1: the per-dataset means and the means over datasets must equal deltas.csv of the run of record to within
1e-9 pp.

Usage (from the repository root):
    python experiments/E4_posthoc_datasets.py
Output (never overwritten): results/E4_posthoc_datasets_<stamp>/
    per_dataset.csv   one row per (design, arm, condition, dataset)
    check_R1.json, summary.md, run_info.json
"""

from __future__ import annotations

import collections
import csv
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pcm_acam.io_utils import environment_info, make_logger, make_run_dir, now_jst, sha256_file, write_json  # noqa: E402

RUN = ROOT / "results" / "E4_20261017_0618"
DATASETS = ("iris", "wine", "breast_cancer", "digits", "synthetic")
ARMS = ("xt", "is", "gs", "gi")


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


def main() -> int:
    run = make_run_dir(ROOT / "results", "E4_posthoc_datasets", False)
    log = make_logger(run / "run.log")
    files = [RUN / "per_eval.csv", RUN / "deltas.csv", Path(__file__)]
    info = {"analysis": "E4 post hoc: node difference per dataset", "start_jst": now_jst().isoformat(timespec="seconds"),
            "environment": environment_info(), "sha256": {str(p.relative_to(ROOT)): sha256_file(p) for p in files}}
    write_json(run / "run_info.json", info)

    loss: Dict[tuple, float] = {}
    for r in rcsv(RUN / "per_eval.csv"):
        if r["time"] != "1y":
            continue
        loss[(r["design"], r["arm"], r["condition"], r["node"], r["dataset"], r["split"], r["seed"])] = float(r["loss_pp"])
    units: Dict[tuple, List[float]] = collections.defaultdict(list)
    for (d, arm, cond, node, ds, split, seed), v in loss.items():
        if node != "dac_remap":
            continue
        other = loss.get((d, arm, cond, "supply", ds, split, seed))
        if other is not None:
            units[(d, arm, cond, ds)].append(v - other)

    stored = {(r["design"], r["arm"], r["condition"]): r for r in rcsv(RUN / "deltas.csv")
              if r["time"] == "1y" and r["contrast"] == "dac_minus_supply" and r["arm"] in ARMS}
    rows, worst = [], 0.0
    for (d, arm, cond), s in sorted(stored.items()):
        means = []
        for ds in DATASETS:
            a = np.asarray(units[(d, arm, cond, ds)], float)
            m, se = float(a.mean()), float(a.std(ddof=1) / np.sqrt(a.size))
            means.append(m)
            worst = max(worst, abs(m - float(s[f"d_{ds}"])))
            rows.append({"design": d, "arm": arm, "condition": cond, "dataset": ds, "n_units": int(a.size),
                         "mean_pp": m, "ci_lo": m - 1.96 * se, "ci_hi": m + 1.96 * se,
                         "sign": "supply better" if m - 1.96 * se > 0 else ("remap better" if m + 1.96 * se < 0 else "n.s.")})
        worst = max(worst, abs(float(np.mean(means)) - float(s["mean"])))
        rows.append({"design": d, "arm": arm, "condition": cond, "dataset": "mean without synthetic", "n_units": "",
                     "mean_pp": float(np.mean(means[:4])), "ci_lo": "", "ci_hi": "", "sign": ""})
    ok = worst <= 1e-9
    write_json(run / "check_R1.json", {"max_abs_diff_pp": worst, "pass": ok})
    log(f"R1: largest |recomputed - stored| {worst:.2e} pp")
    if not ok:
        return 2
    write_csv(run / "per_dataset.csv", rows)
    lines = ["# Node difference per dataset, one year (loss(remap) - loss(supply), pp; post hoc)", ""]
    for cond in ("all", "unif"):
        lines += [f"## condition {cond}", "", "| design | arm | " + " | ".join(DATASETS) + " | mean without synthetic |",
                  "|---|---|" + "---|" * (len(DATASETS) + 1)]
        for d in ("P1", "P3"):
            for arm in ARMS:
                sel = [r for r in rows if (r["design"], r["arm"], r["condition"]) == (d, arm, cond)]
                cells = [f"{r['mean_pp']:+.2f} [{r['ci_lo']:+.2f}, {r['ci_hi']:+.2f}]" for r in sel[:5]]
                lines.append(f"| {d} | {arm} | " + " | ".join(cells) + f" | {sel[5]['mean_pp']:+.2f} |")
        lines.append("")
    (run / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    info["end_jst"] = now_jst().isoformat(timespec="seconds")
    write_json(run / "run_info.json", info)
    log(f"done: {run.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
