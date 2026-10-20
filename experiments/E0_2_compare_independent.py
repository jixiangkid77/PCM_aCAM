"""Compare the main E0.2 run with the independent recomputation (primary analysis).

Usage
-----
    python experiments/E0_2_compare_independent.py <main_run_dir> <independent_dir>

Joins the primary evaluations (mismatch m20, threshold programming) of both runs on
(dataset, split, device_seed, node, condition, t), reports exact agreement of accuracy, path
integrity and replica factor, the reference accuracies, the clipping counts, the summary losses
and the decision. Writes results/E0_2_compare_independent_<YYYYMMDD_HHMM>/comparison.json.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pcm_acam.io_utils import make_run_dir, now_jst, sha256_file, write_json  # noqa: E402

KEY = ("dataset", "split", "device_seed", "node", "condition", "t")


def read_csv(path: Path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> int:
    main_dir, ind_dir = Path(sys.argv[1]), Path(sys.argv[2])
    m_rows = [r for r in read_csv(main_dir / "per_eval.csv") if r["mismatch"] == "m20" and r["programming"] == "threshold"]
    ind = {tuple(r[k] for k in KEY): r for r in read_csv(ind_dir / "per_eval.csv")}
    n = same = missing = 0
    d_acc = d_int = d_a = 0.0
    for r in m_rows:
        k = tuple(r[c] for c in KEY)
        if k not in ind:
            missing += 1
            continue
        o = ind[k]
        n += 1
        x = abs(float(r["acc"]) - float(o["acc"]))
        same += x == 0.0
        d_acc = max(d_acc, x)
        d_int = max(d_int, abs(float(r["integrity"]) - float(o["integrity"])))
        d_a = max(d_a, abs(float(r["a_hat"]) - float(o["a_hat"])))

    ref_m = {(r["dataset"], r["split"]): float(r["acc_ref"]) for r in read_csv(main_dir / "refs.csv")}
    ref_i = {(r["dataset"], r["split"]): float(r["acc_ref"]) for r in read_csv(ind_dir / "refs.csv")}
    d_ref = max(abs(ref_m[k] - ref_i[k]) for k in ref_i)

    clip_m = sum(int(r["clipped"]) for r in read_csv(main_dir / "clipping.csv")
                 if r["mismatch"] == "m20" and r["programming"] == "threshold")
    clip_rows_i = read_csv(ind_dir / "clipping.csv")
    clip_col = next(c for c in clip_rows_i[0].keys() if "clip" in c.lower())
    clip_i = sum(int(r[clip_col]) for r in clip_rows_i)

    s_m = {(r["node"], r["condition"], r["t"]): r for r in read_csv(main_dir / "summary.csv")
           if r["mismatch"] == "m20" and r["programming"] == "threshold"}
    s_i = {(r["node"], r["condition"], r["t"]): r for r in read_csv(ind_dir / "summary.csv")}
    d_mean = max(abs(float(s_m[k]["mean_loss_pp"]) - float(s_i[k]["mean_loss_pp"])) for k in s_i)
    d_max = max(abs(float(s_m[k]["max_loss_pp"]) - float(s_i[k]["max_loss_pp"])) for k in s_i)

    dec_m = json.loads((main_dir / "decision.json").read_text(encoding="utf-8"))
    gates_i = json.loads((ind_dir / "gates.json").read_text(encoding="utf-8"))
    verdict_i = gates_i.get("verdict")  # explicit field; the rule texts also mention every label

    out = {"main_run": main_dir.name, "independent_run": ind_dir.name,
           "main_decision_sha256": sha256_file(main_dir / "decision.json"),
           "independent_gates_sha256": sha256_file(ind_dir / "gates.json"),
           "evaluations_compared": n, "evaluations_missing_in_independent": missing,
           "acc_identical": same, "acc_max_abs_diff": d_acc, "integrity_max_abs_diff": d_int,
           "a_hat_max_abs_diff": d_a, "acc_ref_max_abs_diff": d_ref,
           "clipped_main": clip_m, "clipped_independent": clip_i,
           "summary_cells_compared": len(s_i), "mean_loss_pp_max_abs_diff": d_mean, "max_loss_pp_max_abs_diff": d_max,
           "verdict_main": dec_m["verdict"], "verdict_independent": verdict_i,
           "verdict_identical": dec_m["verdict"] == verdict_i, "compared_at_jst": now_jst().isoformat()}
    run_dir = make_run_dir(ROOT / "results", "E0_2_compare_independent", False)
    write_json(run_dir / "comparison.json", out)
    for k, v in out.items():
        print(f"{k}: {v}")
    print(f"written to {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
