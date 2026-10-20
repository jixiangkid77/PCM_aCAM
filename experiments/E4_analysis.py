"""Tables for the E4 results record (report only; no decision is taken here).

Usage: python experiments/E4_analysis.py results/E4_<stamp>
Writes into the run folder:
  analysis_main.csv          loss at t0 and at 1 year per node, and the paired node difference, per design / arm / condition
  analysis_crosstalk.csv     crosstalk loss (xt minus is, gs minus gi) per time and node, with leaf-loss rates
  analysis_cell_errors.csv   rms / mean input-referred bound error of the forest constraints at t0 and 1 year
  analysis_cell_errors_robust.csv  post-hoc (not pre-registered): bound errors with PCM noise (condition all) for
                             split 0 and device seed 0 of every dataset, with the rare cells whose conductance
                             reaches 0 (x_eff = -10 by convention, error about -150 LSB) counted separately
  analysis_ties.csv          post-hoc: ideal 4-bit CAM accuracy when every tie x == theta goes to the other branch
                             (theta - 1e-9: lower bounds satisfied, upper bounds violated) or keeps the ideal
                             convention (theta + 1e-9), and the share of constraints with a tie
  analysis_no_digits.csv     post-hoc sensitivity check (not pre-registered): the paired differences of the
                             hypotheses recomputed without the digits dataset, whose integer pixel values often
                             equal a quantized threshold exactly (x = theta = 0), so that ties decide matches when
                             the bound errors are tiny (condition unif at t0 and with the ideal node)
"""

from __future__ import annotations

import collections
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def rcsv(p: Path):
    with open(p, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def wcsv(p: Path, rows):
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    run = Path(sys.argv[1])
    summary = rcsv(run / "summary.csv")
    deltas = rcsv(run / "deltas.csv")
    cells = rcsv(run / "cell_errors.csv")
    s_idx = {(r["design"], r["arm"], r["condition"], r["time"], r["node"]): r for r in summary}
    d_idx = {(r["design"], r["arm"], r["condition"], r["time"], r["contrast"]): r for r in deltas}

    main_rows = []
    for cond in ("all", "unif"):
        for did in ("P1", "P3"):
            for arm in ("xt", "is", "gs", "gi"):
                if (did, arm, cond, "t0", "t0") not in s_idx:
                    continue
                d = d_idx[(did, arm, cond, "1y", "dac_minus_supply")]
                main_rows.append({
                    "condition": cond, "design": did, "arm": arm,
                    "loss_t0": float(s_idx[(did, arm, cond, "t0", "t0")]["mean_loss_pp"]),
                    **{f"loss_1y_{n}": float(s_idx[(did, arm, cond, "1y", n)]["mean_loss_pp"])
                       for n in ("dac_remap", "supply", "ideal")},
                    "delta_1y_dac_minus_supply": float(d["mean"]), "ci_lo": float(d["ci_lo"]), "ci_hi": float(d["ci_hi"]),
                    "leaf_loss_t0": float(s_idx[(did, arm, cond, "t0", "t0")]["leaf_loss_mean"]),
                    **{f"leaf_loss_1y_{n}": float(s_idx[(did, arm, cond, "1y", n)]["leaf_loss_mean"])
                       for n in ("dac_remap", "supply")},
                })
    wcsv(run / "analysis_main.csv", main_rows)

    xt_rows = []
    for cond in ("all", "unif"):
        for did in ("P1", "P3"):
            for lab in ("xt_minus_is", "gs_minus_gi"):
                for t, node in [("t0", "t0")] + [(t, n) for t in ("1d", "30d", "1y") for n in ("dac_remap", "supply", "ideal")]:
                    d = d_idx.get((did, lab, cond, t, node))
                    if d is None:
                        continue
                    xt_rows.append({"condition": cond, "design": did, "contrast": lab, "time": t, "node": node,
                                    "mean_pp": float(d["mean"]), "ci_lo": float(d["ci_lo"]), "ci_hi": float(d["ci_hi"]),
                                    **{k: float(v) for k, v in d.items() if k.startswith("d_")}})
    wcsv(run / "analysis_crosstalk.csv", xt_rows)

    cell_rows = [r for r in cells if r["time"] in ("1y", "t0")]
    wcsv(run / "analysis_cell_errors.csv", cell_rows)

    # post-hoc: paired statistics without digits
    ev = rcsv(run / "per_eval.csv")
    unit = {(e["design"], e["arm"], e["condition"], e["time"], e["node"], e["dataset"], e["split"], e["seed"]):
            float(e["loss_pp"]) for e in ev}
    datasets = [d for d in ("iris", "wine", "breast_cancer", "digits", "synthetic") if d != "digits"]

    def paired(a, b):
        per = collections.defaultdict(list)
        for key, v in unit.items():
            if key[:5] == a and key[5] in datasets:
                w = unit.get(b + key[5:])
                if w is not None:
                    per[key[5]].append(v - w)
        m = [np.mean(v) for v in per.values()]
        se = np.sqrt(sum(np.var(v, ddof=1) / len(v) for v in per.values())) / len(m)
        return float(np.mean(m)), float(se)

    nd_rows = []
    for cond in ("all", "unif"):
        for did in ("P1", "P3"):
            for arm in ("xt", "is", "gs", "gi"):
                m, se = paired((did, arm, cond, "1y", "dac_remap"), (did, arm, cond, "1y", "supply"))
                nd_rows.append({"condition": cond, "design": did, "contrast": f"{arm}: dac_remap - supply (1y)",
                                "mean_pp": m, "ci_lo": m - 1.96 * se, "ci_hi": m + 1.96 * se})
            for a_arm, b_arm in (("xt", "is"), ("gs", "gi")):
                for t, node in (("t0", "t0"), ("1y", "dac_remap"), ("1y", "supply")):
                    m, se = paired((did, a_arm, cond, t, node), (did, b_arm, cond, t, node))
                    nd_rows.append({"condition": cond, "design": did, "contrast": f"{a_arm} - {b_arm} ({t}, {node})",
                                    "mean_pp": m, "ci_lo": m - 1.96 * se, "ci_hi": m + 1.96 * se})
    wcsv(run / "analysis_no_digits.csv", nd_rows)

    # post-hoc: ties x == theta in the ideal 4-bit CAM (all datasets and splits)
    from pcm_acam.acam import cam_predict
    from pcm_acam.trees import dataset_bundle as _bundle, quantize as _quant
    tie_rows = []
    for ds in ("iris", "wine", "breast_cancer", "digits", "synthetic"):
        for split in (0, 1, 2):
            b = _bundle(ds, split)
            rows_, x, y = b["rows"], b["x_te"], b["y_te"]
            cls = b["rf"].classes_
            th = _quant(rows_.cons_thr, 4)

            def acc_of(t):
                pr, _ = cam_predict(x, rows_, t)
                return float(np.mean(np.where(pr >= 0, cls[np.maximum(pr, 0)], -1) == y))
            xv = x[:, rows_.cons_feat]
            tie_rows.append({"dataset": ds, "split": split, "acc_ref": acc_of(th),
                             "acc_ties_matched": acc_of(th - 1e-9), "acc_ties_violated": acc_of(th + 1e-9),
                             "share_constraints_with_tie": float(np.mean(np.any(xv == th[None, :], axis=0)))})
    wcsv(run / "analysis_ties.csv", tie_rows)

    # post-hoc: robust bound-error statistics with PCM noise (split 0, seed 0 of every dataset)
    import importlib.util
    import json
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("e4main", root / "experiments" / "E4_forest_system.py")
    e4 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(e4)
    from pcm_acam.acam import Scheme
    from pcm_acam.pcm_model import PCMParams
    from pcm_acam.trees import dataset_bundle, quantize
    info = json.loads((run / "run_info.json").read_text())
    cfg = json.loads((run / "config_used.json").read_text())
    inst = e4.Instrument.load(root / info["instrument_dir"] / "instrument.npz")
    p = PCMParams.from_dict(cfg["pcm_model"])
    rep = Scheme(name="replica", kind="absolute", devices=1, g_lo=cfg["circuit"]["g_lo_us"], g_hi=cfg["circuit"]["g_hi_us"])
    pts = e4.eval_points(cfg)
    names = list(cfg["datasets"])
    acc = collections.defaultdict(lambda: [0.0, 0, 0, [], 0, 0])
    for ds in names:
        b = dataset_bundle(ds, 0)
        rows = b["rows"]
        th = quantize(rows.cons_thr, cfg["bits"])
        kind = rows.cons_kind
        draws, idx = e4.forest_state(cfg, names.index(ds), 0, 0, th.size, cfg["pools"]["n_instances"])
        for did in ("P1", "P3"):
            for cs in ("6T2M", "GS"):
                cm = e4.CellModel(inst, cfg, did, cs)
                res, _, _ = e4.theta_effs(cm, th, kind, idx, draws, "all", pts, p, rep)
                for tlab, node, _, te, g_now in res:
                    e = 15.0 * (te - th)
                    for kd in (0, 1):
                        sel = kind == kd
                        big = np.abs(e[sel]) > 15.0
                        a = acc[(did, cs, cm.types[kd], tlab, node)]
                        a[0] += float(np.sum(e[sel][~big] ** 2))
                        a[1] += int((~big).sum())
                        a[2] += int(big.sum())
                        a[3].append(np.abs(e[sel][~big]))
                        if big.any():
                            a[4] = max(a[4], int(np.round(15 * th[sel][big].max())))
                            a[5] += int(np.sum(g_now[sel][big] == 0))
    rob = []
    for key, (ss, n_ok, n_big, absvals, th_max, n_zero) in sorted(acc.items()):
        rob.append({"design": key[0], "cell_set": key[1], "type": key[2], "time": key[3], "node": key[4],
                    "n": n_ok + n_big, "n_extreme": n_big, "extreme_max_theta_x15": th_max if n_big else "",
                    "extreme_with_zero_conductance": n_zero, "rms_lsb_without_extreme": float(np.sqrt(ss / n_ok)),
                    "median_abs_lsb": float(np.median(np.concatenate(absvals)))})
    wcsv(run / "analysis_cell_errors_robust.csv", rob)
    print(f"wrote analysis_main.csv ({len(main_rows)}), analysis_crosstalk.csv ({len(xt_rows)}), "
          f"analysis_cell_errors.csv ({len(cell_rows)}), analysis_no_digits.csv ({len(nd_rows)}), "
          f"analysis_cell_errors_robust.csv ({len(rob)}), analysis_ties.csv ({len(tie_rows)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
