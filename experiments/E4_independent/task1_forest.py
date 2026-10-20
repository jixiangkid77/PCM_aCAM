"""Task 1: recompute acc and leaf_loss of 640 E4 evaluations (wine, breast_cancer; split 0;
seeds 0-1; P1, P3; arms xt, is, gs, gi; conditions all, unif; t0 and 1d/30d/1y x
dac_remap/supply/ideal) from instrument.npz and compare them with per_eval.csv.

Usage (from anywhere):  python experiments/E4_independent/task1_forest.py --out <dir>
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

import e4i_core as C

ARMS = {"6T2M": ("is", "xt"), "GS": ("gi", "gs")}       # (ideal sensing, crosstalk)
POINTS = [("t0", "t0")] + [(t, n) for t in ("1d", "30d", "1y")
                           for n in ("dac_remap", "supply", "ideal")]


def row_matrices(rows):
    n_rows = rows.row_tree.size
    nz = np.where(rows.row_ncons > 0)[0]
    tree_onehot = np.zeros((n_rows, rows.n_trees))
    tree_onehot[np.arange(n_rows), rows.row_tree] = 1.0
    return nz, tree_onehot


def violated(x_te, rows, x_eff):
    xv = x_te[:, rows.cons_feat]                       # (n_s, K)
    lo = rows.cons_kind == 0
    return np.where(lo[None, :], xv <= x_eff[None, :], xv > x_eff[None, :])


def row_ok_from_viol(viol, rows, nz):
    n_s = viol.shape[0]
    cnt = np.zeros((n_s, rows.row_tree.size), dtype=np.int64)
    if nz.size:
        cnt[:, nz] = np.add.reduceat(viol.astype(np.int64), rows.row_start[nz], axis=1)
    return cnt == 0


def score_eval(match, rows, y_te, classes, tree_onehot):
    m = match.astype(float)
    score = m @ rows.row_value
    anym = match.any(axis=1)
    pred = classes[np.argmax(score, axis=1)]
    correct = anym & (pred == y_te)
    tcount = m @ tree_onehot
    # runner-up margin of the class scores (diagnosis of argmax near-ties)
    srt = np.sort(score, axis=1)
    margin = srt[:, -1] - srt[:, -2] if score.shape[1] > 1 else np.full(score.shape[0], np.inf)
    return {
        "n_correct": int(correct.sum()),
        "acc": float(correct.mean()),
        "leaf_loss": float(np.mean(tcount == 0)),
        "multi_match": float(np.mean(tcount > 1)),
        "correct": correct,
        "margin": margin,
    }


def run(out_dir: Path, datasets=("wine", "breast_cancer"), split: int = 0, seeds=(0, 1),
        designs=("P1", "P3"), conditions=("all", "unif"), log=print, n_inv: int | None = None,
        tag: str = "", surface_scheme: str = "lnvg_ivds"):
    """n_inv: bisection steps of the bound-voltage inversion (None = e4i_core.N_INV = 40);
    surface_scheme: interpolation of the I-V surface ('lnvg_ivds' = main run, 'bilinear_ln');
    tag: suffix of the output files (e.g. '_n64')."""
    t_start = time.time()
    cfg = C.load_config()
    trees = C.trees_module()
    make_draws = C.make_draws_fn()
    inst = C.Instrument(surface_scheme=surface_scheme)
    pcm = C.PCM(cfg)
    times = cfg["times_inference_s"]
    w0 = inst.w0
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []          # one dict per evaluation
    near = []             # near-threshold diagnostics per crosstalk evaluation
    refs = {}
    for ds in datasets:
        ds_idx = C.DATASETS.index(ds)
        b = trees.dataset_bundle(ds, split)
        rows = b["rows"]
        x_te, y_te, rf = b["x_te"], b["y_te"], b["rf"]
        classes = np.asarray(rf.classes_)
        theta = trees.quantize(rows.cons_thr, 4)
        nz, tree_onehot = row_matrices(rows)
        n_feat = x_te.shape[1]
        k_cons = rows.cons_row.size
        # ideal 4-bit CAM reference
        viol_ref = violated(x_te, rows, theta)
        ref = score_eval(row_ok_from_viol(viol_ref, rows, nz), rows, y_te, classes, tree_onehot)
        # exact ties: test inputs equal to a quantized threshold of a constraint
        xv = x_te[:, rows.cons_feat]
        ts, tc = np.nonzero(xv == theta[None, :])
        tie_vals = sorted({int(round(v * 15)) for v in xv[ts, tc]})
        refs[ds] = {"acc_ref": ref["acc"], "n_test": int(y_te.size), "n_rows": int(rows.row_tree.size),
                    "n_constraints": int(k_cons), "n_lower": int(np.sum(rows.cons_kind == 0)),
                    "n_features": int(n_feat),
                    "exact_ties": {"pairs": int(ts.size), "x_times_15": tie_vals,
                                   "features": sorted({int(f) for f in rows.cons_feat[tc]})}}
        log(f"[task1{tag}] {ds} split {split}: n_test {y_te.size}, rows {rows.row_tree.size}, "
            f"constraints {k_cons}, acc_ref {ref['acc']:.6f}")
        thr_lhs_const = 1.0 + (2 * n_feat - 1) * w0
        for seed in seeds:
            draws = make_draws([C.BASE_SEED, ds_idx, split, seed], k_cons, 5, 64)
            idx = np.random.default_rng([C.BASE_SEED, ds_idx, split, seed, 4]).integers(1, 2001, k_cons)
            for design in designs:
                for cell_set, (arm_i, arm_x) in ARMS.items():
                    cs = C.CellSetBounds(inst, design, cell_set, rows.cons_kind, theta, idx, n_inv=n_inv)
                    for cond in conditions:
                        for tname, node in POINTS:
                            g_now, a_hat = C.condition_state(pcm, cond, tname, float(times[tname]),
                                                             cs.g_target, draws)
                            g_eff, v_sl, m = cs.node_params(node, g_now, a_hat)
                            x_eff, v_b = cs.x_eff(g_eff, v_sl, m)
                            viol = violated(x_te, rows, x_eff)
                            ok = row_ok_from_viol(viol, rows, nz)
                            key = dict(dataset=ds, split=split, seed=seed, design=design,
                                       condition=cond, time=tname, node=node, a_hat=a_hat)
                            # ideal sensing arm
                            r_i = score_eval(ok, rows, y_te, classes, tree_onehot)
                            results.append(dict(key, arm=arm_i, acc=r_i["acc"], n_correct=r_i["n_correct"],
                                                n_test=int(y_te.size), loss_pp=100 * (ref["acc"] - r_i["acc"]),
                                                leaf_loss=r_i["leaf_loss"], multi_match=r_i["multi_match"],
                                                removed_by_crosstalk=0, min_rel_gap=np.nan,
                                                n_tie_x_eq_theta=0))
                            # crosstalk arm
                            s_idx, r_idx = np.nonzero(ok)
                            # constraints of the matched (sample, row) pairs
                            ncons = rows.row_ncons[r_idx]
                            rep_s = np.repeat(s_idx, ncons)
                            rep_r = np.repeat(r_idx, ncons)
                            offs = np.arange(rep_s.size) - np.repeat(np.cumsum(ncons) - ncons, ncons)
                            rep_c = rows.row_start[rep_r] + offs
                            xs = x_te[rep_s, rows.cons_feat[rep_c]]
                            w = cs.leakage(rep_c, xs, x_eff, g_eff, v_sl, m)
                            n_rows = rows.row_tree.size
                            ssum = np.bincount(rep_s * n_rows + rep_r, weights=w,
                                               minlength=x_te.shape[0] * n_rows).reshape(x_te.shape[0], n_rows)
                            lhs = ssum + (2 * n_feat - rows.row_ncons[None, :]) * w0
                            xt_mis = ok & (lhs >= thr_lhs_const)
                            match_x = ok & ~xt_mis
                            r_x = score_eval(match_x, rows, y_te, classes, tree_onehot)
                            # closeness of the matched pairs to the crosstalk threshold
                            gaps = np.abs(lhs[ok] - thr_lhs_const) / thr_lhs_const
                            results.append(dict(key, arm=arm_x, acc=r_x["acc"], n_correct=r_x["n_correct"],
                                                n_test=int(y_te.size), loss_pp=100 * (ref["acc"] - r_x["acc"]),
                                                leaf_loss=r_x["leaf_loss"], multi_match=r_x["multi_match"],
                                                removed_by_crosstalk=int(xt_mis.sum()),
                                                min_rel_gap=float(gaps.min()) if gaps.size else np.nan,
                                                n_tie_x_eq_theta=0))
                            # exact ties x == x_eff (possible for clipped/discrete inputs when
                            # x_eff equals theta up to rounding): count (sample, bound) pairs
                            ties = np.abs(x_te[:, rows.cons_feat] - x_eff[None, :]) <= 1e-9
                            for rr in results[-2:]:
                                rr["n_tie_x_eq_theta"] = int(ties.sum())
            log(f"[task1{tag}] {ds} seed {seed} done ({time.time() - t_start:.1f} s)")
    # write my results
    fields = ["dataset", "split", "seed", "design", "arm", "condition", "time", "node", "a_hat",
              "acc", "n_correct", "n_test", "loss_pp", "leaf_loss", "multi_match",
              "removed_by_crosstalk", "min_rel_gap", "n_tie_x_eq_theta"]
    with open(out_dir / f"task1_independent_eval{tag}.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader()
        for r in results:
            wr.writerow({k: r[k] for k in fields})
    return compare(results, refs, out_dir, log, time.time() - t_start, tag,
                   C.N_INV if n_inv is None else n_inv, surface_scheme)


def compare(results, refs, out_dir: Path, log, elapsed, tag: str = "", n_inv: int = C.N_INV,
            surface_scheme: str = "lnvg_ivds"):
    main = {}
    with open(C.FOREST_DIR / "per_eval.csv") as f:
        for r in csv.DictReader(f):
            main[(r["dataset"], int(r["split"]), int(r["seed"]), r["design"], r["arm"],
                  r["condition"], r["time"], r["node"])] = r
    ref_main = {}
    with open(C.FOREST_DIR / "refs.csv") as f:
        for r in csv.DictReader(f):
            ref_main[(r["dataset"], int(r["split"]))] = r
    rows_out = []
    n_acc_diff = 0
    n_acc_diff_1 = 0
    max_sample_diff = 0
    n_leaf_diff = 0
    max_leaf_diff = 0.0
    max_ahat_rel = 0.0
    n_multi_diff = 0
    n_removed_diff = 0
    for r in results:
        key = (r["dataset"], r["split"], r["seed"], r["design"], r["arm"], r["condition"], r["time"], r["node"])
        m = main[key]
        acc_m = float(m["acc"])
        n_m = int(round(acc_m * r["n_test"]))
        dn = r["n_correct"] - n_m
        leaf_m = float(m["leaf_loss"])
        d_leaf = r["leaf_loss"] - leaf_m
        ahat_m = float(m["a_hat"])
        d_ahat = abs(r["a_hat"] - ahat_m) / ahat_m
        max_ahat_rel = max(max_ahat_rel, d_ahat)
        if dn != 0:
            n_acc_diff += 1
            if abs(dn) == 1:
                n_acc_diff_1 += 1
        max_sample_diff = max(max_sample_diff, abs(dn))
        if abs(d_leaf) > 1e-12:
            n_leaf_diff += 1
        max_leaf_diff = max(max_leaf_diff, abs(d_leaf))
        d_multi = r["multi_match"] - float(m["multi_match"])
        d_rem = r["removed_by_crosstalk"] - int(m["removed_by_crosstalk"])
        n_multi_diff += abs(d_multi) > 1e-12
        n_removed_diff += d_rem != 0
        rows_out.append({
            "dataset": r["dataset"], "split": r["split"], "seed": r["seed"], "design": r["design"],
            "arm": r["arm"], "condition": r["condition"], "time": r["time"], "node": r["node"],
            "a_hat_indep": r["a_hat"], "a_hat_main": ahat_m,
            "acc_indep": r["acc"], "acc_main": acc_m, "d_samples": dn,
            "leaf_loss_indep": r["leaf_loss"], "leaf_loss_main": leaf_m, "d_leaf_loss": d_leaf,
            "loss_pp_indep": r["loss_pp"], "loss_pp_main": float(m["loss_pp"]),
            "multi_match_indep": r["multi_match"], "multi_match_main": float(m["multi_match"]),
            "removed_indep": r["removed_by_crosstalk"], "removed_main": int(m["removed_by_crosstalk"]),
            "min_rel_gap_to_crosstalk_threshold": r["min_rel_gap"],
            "n_tie_pairs_x_eq_xeff": r["n_tie_x_eq_theta"],
        })
    fields = list(rows_out[0].keys())
    with open(out_dir / f"task1_compare{tag}.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader()
        wr.writerows(rows_out)
    ref_cmp = {}
    for ds, v in refs.items():
        rm = ref_main[(ds, 0)]
        ref_cmp[ds] = {"exact_ties_x_eq_theta": v["exact_ties"],
                       "acc_ref_indep": v["acc_ref"], "acc_ref_main": float(rm["acc_ref"]),
                       "n_rows_indep": v["n_rows"], "n_rows_main": int(rm["n_rows"]),
                       "n_constraints_indep": v["n_constraints"], "n_constraints_main": int(rm["n_constraints"]),
                       "n_lower_indep": v["n_lower"], "n_lower_main": int(rm["n_lower"]),
                       "equal": (v["acc_ref"] == float(rm["acc_ref"]) and v["n_rows"] == int(rm["n_rows"])
                                 and v["n_constraints"] == int(rm["n_constraints"])
                                 and v["n_lower"] == int(rm["n_lower"]))}
    n_eval = len(results)
    allowed = int(np.floor(0.01 * n_eval))
    by_point = {}
    for ro in rows_out:
        p = f"{ro['condition']}/{ro['time']}/{ro['node']}"
        d = by_point.setdefault(p, {"acc": 0, "leaf_loss": 0, "multi_match": 0, "removed_by_crosstalk": 0})
        d["acc"] += int(ro["d_samples"] != 0)
        d["leaf_loss"] += int(abs(ro["d_leaf_loss"]) > 1e-12)
        d["multi_match"] += int(abs(ro["multi_match_indep"] - ro["multi_match_main"]) > 1e-12)
        d["removed_by_crosstalk"] += int(ro["removed_indep"] != ro["removed_main"])
    by_point = {p: d for p, d in by_point.items() if any(d.values())}
    n_trees = 16
    max_leaf_pairs = max(int(round(abs(ro["d_leaf_loss"]) * r["n_test"] * n_trees))
                         for ro, r in zip(rows_out, results))
    acc_pass = (max_sample_diff <= 1) and (n_acc_diff <= allowed)
    leaf_pass = n_leaf_diff == 0
    summary = {
        "bound_inversion_bisection_steps": n_inv,
        "surface_scheme": surface_scheme,
        "n_evaluations": n_eval,
        "acc_identical": n_eval - n_acc_diff,
        "acc_differing": n_acc_diff,
        "acc_differing_by_one_sample": n_acc_diff_1,
        "max_sample_difference": max_sample_diff,
        "allowed_one_sample_differences": allowed,
        "leaf_loss_differing": n_leaf_diff,
        "max_abs_leaf_loss_difference": max_leaf_diff,
        "max_sample_tree_pairs_difference": max_leaf_pairs,
        "a_hat_max_rel_difference": max_ahat_rel,
        "multi_match_differing": int(n_multi_diff),
        "removed_by_crosstalk_differing": int(n_removed_diff),
        "differences_by_point": by_point,
        "min_rel_gap_to_crosstalk_threshold": float(np.nanmin([ro["min_rel_gap_to_crosstalk_threshold"]
                                                               for ro in rows_out])),
        "max_tie_pairs_x_eq_xeff_per_eval": int(max(ro["n_tie_pairs_x_eq_xeff"] for ro in rows_out)),
        "refs": ref_cmp,
        "pass_acc": bool(acc_pass),
        "pass_leaf_loss": bool(leaf_pass),
        "pass": bool(acc_pass and leaf_pass and all(v["equal"] for v in ref_cmp.values())),
        "elapsed_s": round(elapsed, 1),
    }
    with open(out_dir / f"task1_summary{tag}.json", "w") as f:
        json.dump(summary, f, indent=1)
    log(f"[task1{tag}] {n_eval} evaluations: acc identical {n_eval - n_acc_diff}, differing {n_acc_diff} "
        f"(max {max_sample_diff} samples); leaf_loss differing {n_leaf_diff} (max {max_leaf_diff:.3g}); "
        f"a_hat max rel diff {max_ahat_rel:.2e}; PASS={summary['pass']}")
    return summary


def c2_reproduction(out_dir: Path, n_inv: int | None = None, log=print):
    """Recompute check C2 of checks.json (nominal instance, no PCM noise, uniform drift at
    1h, 1d, 30d, 1y; max |x_eff - theta| in LSB over the 16 levels) and compare bit for bit.
    Used to identify the bisection convention (step count, midpoint) of the bound inversion."""
    cfg = C.load_config()
    inst = C.Instrument()
    with open(C.INSTR_DIR / "checks.json") as f:
        c2_main = json.load(f)["C2"]
    theta = np.arange(16) / 15.0
    times = [float(cfg["times_inference_s"][t]) for t in ("1h", "1d", "30d", "1y")]
    out = {}
    n_equal = n_total = 0
    for design in ("P1", "P3"):
        for cell_set, (lo_t, up_t) in C.CELL_SETS.items():
            for typ, kind in ((lo_t, 0), (up_t, 1)):
                cs = C.CellSetBounds(inst, design, cell_set, np.full(16, kind), theta,
                                     np.zeros(16, dtype=int), n_inv=n_inv)
                res = {}
                for node in ("t0", "supply", "ideal", "dac_remap"):
                    m_err = 0.0
                    for t_inf in ([0.0] if node == "t0" else times):
                        a = 1.0 if node == "t0" else ((t_inf + 20.0) / 20.0) ** (-0.049)
                        g_eff, v_sl, m = cs.node_params(node, a * cs.g_target, a)
                        xe, _ = cs.x_eff(g_eff, v_sl, m)
                        m_err = max(m_err, float(np.max(np.abs(xe - theta)) * 15))
                    res[node] = m_err
                key = f"{design}/{cell_set}/{typ}"
                main = c2_main[key]["max_abs_lsb"]
                eq = {nd: (res[nd] == main[nd]) for nd in res}
                n_equal += sum(eq.values())
                n_total += len(eq)
                out[key] = {"indep_max_abs_lsb": res, "main_max_abs_lsb": main, "bit_identical": eq}
    summary = {"bound_inversion_bisection_steps": C.N_INV if n_inv is None else n_inv,
               "values_bit_identical": n_equal, "values_total": n_total, "per_set": out}
    with open(out_dir / f"task1_c2_reproduction_n{summary['bound_inversion_bisection_steps']}.json", "w") as f:
        json.dump(summary, f, indent=1)
    log(f"[task1] C2 reproduction ({summary['bound_inversion_bisection_steps']} bisection steps): "
        f"{n_equal}/{n_total} values bit-identical")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--n-inv", type=int, default=None, help="bisection steps of the bound inversion")
    ap.add_argument("--tag", default="", help="suffix of the output files")
    ap.add_argument("--surface-scheme", default="lnvg_ivds", choices=["lnvg_ivds", "bilinear_ln"])
    a = ap.parse_args()
    run(Path(a.out), n_inv=a.n_inv, tag=a.tag, surface_scheme=a.surface_scheme)
