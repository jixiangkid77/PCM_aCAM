"""Compare the main E0.1 run with the independent recomputation.

Usage
-----
    python experiments/E0_1_compare_independent.py <main_run_dir> <independent_dir>

Joins the 4-bit evaluations of both runs on (dataset, split, device_seed,
scheme, track, condition, t) and reports exact agreement of accuracy,
path integrity and replica factor, the reference accuracies, the
per-scheme mean/max losses and the gate results. Writes
results/E0_1_compare_independent_<YYYYMMDD_HHMM>/comparison.json (new folder).
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pcm_acam.io_utils import make_run_dir, now_jst, sha256_file, write_json  # noqa: E402

KEY = ("dataset", "split", "device_seed", "scheme", "track", "condition", "t")


def read_csv(path: Path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> int:
    main_dir, ind_dir = Path(sys.argv[1]), Path(sys.argv[2])
    main_rows = [r for r in read_csv(main_dir / "per_eval.csv") if r["bits"] == "4"]
    ind_rows = read_csv(ind_dir / "per_eval.csv")
    ind = {tuple(r[k] for k in KEY): r for r in ind_rows}
    n, same_acc, max_acc, max_int, max_ahat, missing = 0, 0, 0.0, 0.0, 0.0, 0
    for r in main_rows:
        k = tuple(r[k] for k in KEY)
        if k not in ind:
            if r["scheme"] != "V2d_abs_high_dual":
                missing += 1
            continue
        o = ind[k]
        n += 1
        d_acc = abs(float(r["acc"]) - float(o["acc"]))
        same_acc += d_acc == 0.0
        max_acc = max(max_acc, d_acc)
        max_int = max(max_int, abs(float(r["integrity"]) - float(o["integrity"])))
        max_ahat = max(max_ahat, abs(float(r["a_hat"]) - float(o["a_hat"])))

    ref_main = {(r["dataset"], r["split"]): float(r["acc_ref"]) for r in read_csv(main_dir / "refs.csv") if r["bits"] == "4"}
    ref_ind = {(r["dataset"], r["split"]): float(r["acc_ref"]) for r in read_csv(ind_dir / "refs.csv")}
    ref_diff = max(abs(ref_main[k] - ref_ind[k]) for k in ref_ind)

    summ_main = {(r["scheme"], r["track"], r["condition"], r["t"]): r
                 for r in read_csv(main_dir / "scheme_summary.csv") if r["bits"] == "4"}
    summ_ind = {(r["scheme"], r["track"], r["condition"], r["t"]): r for r in read_csv(ind_dir / "summary.csv")}
    max_mean, max_max = 0.0, 0.0
    for k, o in summ_ind.items():
        m = summ_main[k]
        max_mean = max(max_mean, abs(float(m["mean_loss_pp"]) - float(o["mean_loss_pp"])))
        max_max = max(max_max, abs(float(m["max_loss_pp"]) - float(o["max_loss_pp"])))

    dec_main = json.loads((main_dir / "decision.json").read_text(encoding="utf-8"))
    gates_ind = json.loads((ind_dir / "gates.json").read_text(encoding="utf-8"))

    def gate_ind(g):
        v = gates_ind.get(g)
        if isinstance(v, dict):
            v = v.get("pass", v.get("result"))
        return v in (True, "PASS")

    gates_same = all(dec_main[g]["pass"] == gate_ind(g) for g in ("gate_A", "gate_B", "gate_C"))
    verdict_same = dec_main["verdict"] == gates_ind.get("verdict")

    out = {
        "main_run": main_dir.name, "independent_run": ind_dir.name,
        "main_decision_sha256": sha256_file(main_dir / "decision.json"),
        "independent_gates_sha256": sha256_file(ind_dir / "gates.json"),
        "evaluations_compared": n, "evaluations_missing_in_independent": missing,
        "acc_identical": same_acc, "acc_max_abs_diff": max_acc,
        "integrity_max_abs_diff": max_int, "a_hat_max_abs_diff": max_ahat,
        "acc_ref_max_abs_diff": ref_diff,
        "summary_cells_compared": len(summ_ind),
        "mean_loss_pp_max_abs_diff": max_mean, "max_loss_pp_max_abs_diff": max_max,
        "gates_identical": gates_same, "verdict_main": dec_main["verdict"],
        "verdict_independent": gates_ind.get("verdict"), "verdict_identical": verdict_same,
        "compared_at_jst": now_jst().isoformat(),
    }
    run_dir = make_run_dir(ROOT / "results", "E0_1_compare_independent", False)
    write_json(run_dir / "comparison.json", out)
    for k, v in out.items():
        print(f"{k}: {v}")
    print(f"written to {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
