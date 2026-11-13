"""Task 2 (model side): the 48 cases of the E4.1 run of record recomputed with my model (e4i_core
of the E4 independent check, cell set GS: lower bounds LBUF, upper bounds UINV).

For every case (breast_cancer, split 0, seed 0, condition all; P1 and P3 at t0, 1 y input remap,
1 y divider supply) I rebuild the forest, the instance indices, the PCM draws and the threshold-
verify programming, and recompute per constrained bound: instance index, input x, conductance,
effective threshold, data-line voltage, divider node (load line), gate, leakage weight w; per
case: a_hat, supplies, S = sum_k w_k + (2N - n_c) w0, its lower (LBUF) and upper (UINV) parts and
t_c,pred = t_s(N) (1 + (2N - 1) w0) / S, with t_s(N) of the run of record (task 3 also uses my
SPICE t_s). They are compared with cases.json, rows.csv and bounds.csv (tolerances in
e41i_common.TOL).

Extra (not asked by the task): the case selection of the plan (matched (sample, row) pairs with
S in [0.3, 3], 8 per group evenly spread in ln S) is redone from scratch and compared with the 48
(sample, row) pairs and with the pair counts that the main run wrote into its run.log.

Usage: python experiments/E4_1_independent/task2_model.py --out <new folder>
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True           # no __pycache__ folders in the repository

import argparse            # noqa: E402
import csv                 # noqa: E402
import json                # noqa: E402
import re                  # noqa: E402
import time                # noqa: E402
from pathlib import Path   # noqa: E402

import numpy as np         # noqa: E402

import e41i_common as K    # noqa: E402
from e41i_common import C  # noqa: E402

S_RANGE = (0.3, 3.0)
PER_GROUP = 8
H_SLOPE = 1e-7          # step (V) of the numerical d ln w / d node (diagnosis only)
WINDOW = 0.06           # stored inverter curves cover +-60 mV around each switching input


class Forest:
    """breast_cancer, split 0, seed 0: rows, test inputs, quantized thresholds, PCM draws,
    instance indices (as the E4 check, task 1)."""

    def __init__(self):
        cfg = C.load_config()
        trees = C.trees_module()
        make_draws = C.make_draws_fn()
        self.cfg = cfg
        self.inst = C.Instrument()
        self.pcm = C.PCM(cfg)
        ds_idx = C.DATASETS.index(K.DATASET)
        b = trees.dataset_bundle(K.DATASET, K.SPLIT)
        self.rows, self.x_te = b["rows"], b["x_te"]
        self.theta = trees.quantize(self.rows.cons_thr, 4)
        k_cons = self.rows.cons_row.size
        self.draws = make_draws([C.BASE_SEED, ds_idx, K.SPLIT, K.SEED], k_cons, 5, 64)
        self.idx = np.random.default_rng([C.BASE_SEED, ds_idx, K.SPLIT, K.SEED, 4]).integers(1, 2001, k_cons)
        self.n_feat = self.x_te.shape[1]
        self.w0 = self.inst.w0
        self.cs = {d: C.CellSetBounds(self.inst, d, K.CELL_SET, self.rows.cons_kind, self.theta, self.idx)
                   for d in ("P1", "P3")}

    def point(self, design: str, tname: str, node: str):
        """State of all bounds at one (design, time, node)."""
        cs = self.cs[design]
        g_now, a_hat = C.condition_state(self.pcm, K.CONDITION, tname,
                                         float(self.cfg["times_inference_s"][tname]), cs.g_target, self.draws)
        g_eff, v_sl, m = cs.node_params(node, g_now, a_hat)
        x_eff, _ = cs.x_eff(g_eff, v_sl, m)
        return {"cs": cs, "a_hat": a_hat, "g_eff": g_eff, "v_sl": v_sl, "m": m, "x_eff": x_eff}

    def supplies(self, design: str, node: str, a_hat: float):
        d_eff = C.D_V / a_hat if node == "supply" else C.D_V
        lo_t, up_t = C.CELL_SETS[K.CELL_SET]
        return (self.inst.pools[(design, lo_t)].v_sp[0] + d_eff,
                self.inst.pools[(design, up_t)].v_sp[0] + d_eff)


def gate_detail(inst, typ: str, k: int, node: float):
    """Uncut gate of the chain, the LBUF middle node W and the window class."""
    ch = inst.chains[typ]
    kk = np.array([k])
    gate = float(ch.gate(kk, np.array([node]))[0])
    d1 = node - ch.v_sp[k]
    if typ == "LBUF":
        mid = float(C.interp_rows(np.array([d1]), ch.rel_grid, ch.t_first[kk])[0])
        d2 = mid - ch.v_mid_in[k]
        inside = abs(d1) <= WINDOW and abs(d2) <= WINDOW
    else:
        mid, d2 = float("nan"), float("nan")
        inside = abs(d1) <= WINDOW
    return gate, mid, d1, d2, inside


def ln_w_slope(inst, typ: str, k: int, node: float) -> float:
    """d ln w / d node (1/V) of the uncut chain (central difference)."""
    ch = inst.chains[typ]
    kk = np.array([k])
    f = lambda v: float(ch.ln_w(kk, ch.gate(kk, np.array([v])))[0])    # noqa: E731
    return (f(node + H_SLOPE) - f(node - H_SLOPE)) / (2.0 * H_SLOPE)


def model_side(fo: Forest, main: dict, log=print):
    cases = main["cases"]["cases"]
    rows_main = main["rows"]
    t_s = float(main["cases"]["t_s_s"])
    n_feat, w0 = fo.n_feat, fo.w0
    rows = fo.rows
    out_c, out_b = [], []
    cache = {}
    for ci, c in enumerate(cases):
        key = (c["design"], c["time"], c["node"])
        if key not in cache:
            cache[key] = fo.point(*key)
        p = cache[key]
        cs, x_eff = p["cs"], p["x_eff"]
        r, smp = int(c["row"]), int(c["sample"])
        cons = np.arange(rows.row_start[r], rows.row_start[r] + rows.row_ncons[r])
        x = fo.x_te[smp, rows.cons_feat[cons]]
        viol = np.where(rows.cons_kind[cons] == 0, x <= x_eff[cons], x > x_eff[cons])
        w, det = cs.leakage(cons, x, x_eff, p["g_eff"], p["v_sl"], p["m"], return_detail=True)
        lo = rows.cons_kind[cons] == 0
        s_sum = float(w.sum() + (2 * n_feat - cons.size) * w0)
        t_pred = t_s * (1.0 + (2 * n_feat - 1) * w0) / s_sum
        v_sl_lo, v_sl_hi = fo.supplies(c["design"], c["node"], p["a_hat"])
        set_indep = sorted((int(rows.cons_feat[j]), int(rows.cons_kind[j])) for j in cons)
        set_main = sorted((int(b["feat"]), int(b["kind"])) for b in c["bounds"])
        rm = rows_main[ci]
        out_c.append({
            "case": ci, "design": c["design"], "time": c["time"], "node": c["node"],
            "sample": smp, "row": r, "n_bounds_indep": int(cons.size), "n_bounds_main": len(c["bounds"]),
            "bound_set_equal": set_indep == set_main, "any_violated_indep": bool(viol.any()),
            "a_hat_indep": p["a_hat"], "a_hat_main": c["a_hat"],
            "v_sl_lo_indep": v_sl_lo, "v_sl_lo_main": c["v_sl_lo"],
            "v_sl_hi_indep": v_sl_hi, "v_sl_hi_main": c["v_sl_hi"],
            "S_indep": s_sum, "S_main": c["S"], "S_rows_csv": float(rm["S"]),
            "S_rel_diff": s_sum / c["S"] - 1.0,
            "S_lower_LBUF_indep": float(w[lo].sum()), "S_lower_LBUF_main": float(rm["S_lower_LBUF"]),
            "S_upper_UINV_indep": float(w[~lo].sum()), "S_upper_UINV_main": float(rm["S_upper_UINV"]),
            "t_pred_ns_indep": t_pred * 1e9, "t_pred_ns_main": float(rm["t_pred_ns"]),
            "t_pred_rel_diff": t_pred * 1e9 / float(rm["t_pred_ns"]) - 1.0,
            "decision_pred_mismatch_indep": bool(s_sum >= 1.0 + (2 * n_feat - 1) * w0),
            "decision_pred_mismatch_main": rm["decision_pred_mismatch"] == "True",
        })
        bmain = main["bounds_by_case"][ci]
        for bi, bd in enumerate(c["bounds"]):
            j = np.where((rows.cons_feat[cons] == bd["feat"]) & (rows.cons_kind[cons] == bd["kind"]))[0]
            assert j.size == 1
            j = int(j[0])
            cc = int(cons[j])
            typ = cs.types[bd["kind"]]
            k = int(fo.idx[cc])
            u = (x[j] - x_eff[cc]) if bd["kind"] == 0 else (x_eff[cc] - x[j])
            cut = bool(u >= C.U_CUT[typ])
            node_v = float(det["node"][j])
            gate_uncut, mid, d1, d2, inside = gate_detail(fo.inst, typ, k, node_v)
            gclass = "u_cut_gate_0V" if cut else ("in_window" if inside else "node_outside_transfer_window")
            dln = float(np.log(bd["w_model"]) - np.log(w[j]))
            slope = 0.0 if cut else ln_w_slope(fo.inst, typ, k, node_v)
            bm = bmain[bi]
            out_b.append({
                "case": ci, "bound": bi, "design": c["design"], "time": c["time"], "node": c["node"],
                "kind": bd["kind"], "type": typ, "feat": bd["feat"],
                "idx_indep": k, "idx_main": int(bd["idx"]),
                "x_indep": float(x[j]), "x_main": bd["x"],
                "g_us_indep": float(p["g_eff"][cc]), "g_us_main": bd["g_us"],
                "theta_eff_indep": float(x_eff[cc]), "theta_eff_main": bd["theta_eff"],
                "v_dl_indep": float(det["v_dl"][j]), "v_dl_main": bd["v_dl"],
                "u_lsb_indep": float(u * 15.0), "u_lsb_main": float(bm["u_lsb"]),
                "u_cut": cut, "gate_class": gclass,
                "divider_node_V": node_v, "node_minus_vsp_mV": d1 * 1e3,
                "mid_node_W_V": mid, "mid_minus_vmid_mV": d2 * 1e3,
                "gate_uncut_V": gate_uncut, "gate_used_V": float(det["gate"][j]),
                "gate_model_main_V": float(bm["gate_model_V"]),
                "gate_uncut_minus_main_uV": (gate_uncut - float(bm["gate_model_V"])) * 1e6,
                "w_indep": float(w[j]), "w_main": bd["w_model"], "w_rel_diff": float(w[j] / bd["w_model"] - 1.0),
                "dlnw_dnode_per_V": slope,
                "implied_node_diff_V": dln / slope if abs(slope) > 1.0 else float("nan"),
            })
    log(f"[task2] model side recomputed for {len(out_c)} cases, {len(out_b)} bounds")
    return out_c, out_b


def selection(fo: Forest, main: dict, log=print):
    """The plan's case selection redone from scratch (extra)."""
    rows, x_te = fo.rows, fo.x_te
    n_s, n_rows = x_te.shape[0], rows.row_tree.size
    nz = np.where(rows.row_ncons > 0)[0]
    lo_mask = rows.cons_kind == 0
    logged = {}
    for ln in (K.MAIN_DIR / "run.log").read_text().splitlines():
        m = re.search(r"\] (P\d) (\S+) (\S+): (\d+) matched pairs, (\d+) with S in", ln)
        if m:
            logged[f"{m.group(1)}|{m.group(2)}|{m.group(3)}"] = (int(m.group(4)), int(m.group(5)))
    groups = {}
    for design in ("P1", "P3"):
        for tname, node in K.POINTS:
            p = fo.point(design, tname, node)
            xv = x_te[:, rows.cons_feat]
            viol = np.where(lo_mask[None, :], xv <= p["x_eff"][None, :], xv > p["x_eff"][None, :])
            cnt = np.zeros((n_s, n_rows), dtype=np.int64)
            cnt[:, nz] = np.add.reduceat(viol.astype(np.int64), rows.row_start[nz], axis=1)
            s_idx, r_idx = np.nonzero(cnt == 0)
            ncons = rows.row_ncons[r_idx]
            rep_s, rep_r = np.repeat(s_idx, ncons), np.repeat(r_idx, ncons)
            offs = np.arange(rep_s.size) - np.repeat(np.cumsum(ncons) - ncons, ncons)
            rep_c = rows.row_start[rep_r] + offs
            w = p["cs"].leakage(rep_c, x_te[rep_s, rows.cons_feat[rep_c]], p["x_eff"], p["g_eff"], p["v_sl"], p["m"])
            pair = np.repeat(np.arange(s_idx.size), ncons)
            s_pair = np.bincount(pair, weights=w, minlength=s_idx.size) + (2 * fo.n_feat - ncons) * fo.w0
            cand = np.where((s_pair >= S_RANGE[0]) & (s_pair <= S_RANGE[1]))[0]
            order = cand[np.argsort(np.log(s_pair[cand]), kind="stable")]
            pick = list(dict.fromkeys(np.round(np.linspace(0, order.size - 1, PER_GROUP)).astype(int).tolist()))
            mine = [(int(s_idx[order[i]]), int(r_idx[order[i]])) for i in pick]
            key = f"{design}|{tname}|{node}"
            theirs = [(int(c["sample"]), int(c["row"])) for c in main["cases"]["cases"] if K.group_key(c) == key]
            srt = np.sort(s_pair[cand])
            groups[key] = {
                "matched_pairs_indep": int(s_idx.size), "candidates_indep": int(cand.size),
                "matched_pairs_main_log": logged.get(key, (None, None))[0],
                "candidates_main_log": logged.get(key, (None, None))[1],
                "selected_indep": mine, "selected_main": theirs, "selection_identical": mine == theirs,
                "min_rel_gap_between_sorted_candidate_S": float(np.min(np.diff(srt) / srt[1:])) if srt.size > 1 else None,
                "min_rel_distance_of_any_S_to_0.3_or_3": float(min(np.min(np.abs(s_pair / S_RANGE[0] - 1.0)),
                                                                    np.min(np.abs(s_pair / S_RANGE[1] - 1.0)))),
            }
    ok = all(g["selection_identical"] and g["matched_pairs_indep"] == g["matched_pairs_main_log"]
             and g["candidates_indep"] == g["candidates_main_log"] for g in groups.values())
    log(f"[task2] case selection redone: " + ", ".join(
        f"{k} {g['matched_pairs_indep']}/{g['candidates_indep']}" for k, g in groups.items())
        + f"; identical to the main run: {ok}")
    return {"groups": groups, "all_identical": bool(ok)}


def run(out_dir: Path, log=print) -> dict:
    t0 = time.time()
    main = K.load_main()
    fo = Forest()
    mc, mb = model_side(fo, main, log)
    K.write_csv(out_dir / "task2_model_cases.csv", mc)
    K.write_csv(out_dir / "task2_model_bounds.csv", mb)
    sel = selection(fo, main, log)
    with open(out_dir / "task2_selection.json", "w") as f:
        json.dump(sel, f, indent=1)

    T = K.TOL
    rel = lambda a, b: abs(a / b - 1.0) if b != 0 else (0.0 if a == 0 else float("inf"))   # noqa: E731
    lbuf = [b for b in mb if b["type"] == "LBUF"]
    uinv = [b for b in mb if b["type"] == "UINV"]
    m = {
        "n_cases": len(mc), "n_bounds": len(mb), "n_bounds_LBUF": len(lbuf), "n_bounds_UINV": len(uinv),
        "idx_all_equal": all(b["idx_indep"] == b["idx_main"] for b in mb),
        "bound_sets_equal": all(c["bound_set_equal"] and c["n_bounds_indep"] == c["n_bounds_main"] for c in mc),
        "no_violated_bound": all(not c["any_violated_indep"] for c in mc),
        "max_abs_x_diff": K.max_abs(b["x_indep"] - b["x_main"] for b in mb),
        "max_rel_g_us": max(rel(b["g_us_indep"], b["g_us_main"]) for b in mb),
        "max_abs_theta_eff": K.max_abs(b["theta_eff_indep"] - b["theta_eff_main"] for b in mb),
        "max_abs_v_dl_V": K.max_abs(b["v_dl_indep"] - b["v_dl_main"] for b in mb),
        "max_abs_u_lsb": K.max_abs(b["u_lsb_indep"] - b["u_lsb_main"] for b in mb),
        "max_abs_a_hat": K.max_abs(c["a_hat_indep"] - c["a_hat_main"] for c in mc),
        "max_abs_v_sl_V": K.max_abs([c["v_sl_lo_indep"] - c["v_sl_lo_main"] for c in mc]
                                    + [c["v_sl_hi_indep"] - c["v_sl_hi_main"] for c in mc]),
        "max_rel_w": K.max_abs(b["w_rel_diff"] for b in mb),
        "max_rel_w_LBUF": K.max_abs(b["w_rel_diff"] for b in lbuf),
        "max_rel_w_UINV": K.max_abs(b["w_rel_diff"] for b in uinv),
        "max_rel_S": K.max_abs(c["S_rel_diff"] for c in mc),
        "max_abs_S_part_over_S": K.max_abs([(c["S_lower_LBUF_indep"] - c["S_lower_LBUF_main"]) / c["S_main"] for c in mc]
                                           + [(c["S_upper_UINV_indep"] - c["S_upper_UINV_main"]) / c["S_main"] for c in mc]),
        "max_rel_t_pred": K.max_abs(c["t_pred_rel_diff"] for c in mc),
        "S_cases_json_equals_rows_csv": all(c["S_main"] == c["S_rows_csv"] for c in mc),
        "decision_pred_equal": all(c["decision_pred_mismatch_indep"] == c["decision_pred_mismatch_main"] for c in mc),
        "max_abs_implied_node_diff_V": K.max_abs(b["implied_node_diff_V"] for b in mb
                                                 if not np.isnan(b["implied_node_diff_V"])),
        "median_abs_dlnw_dnode_in_window_per_V": {
            t: float(np.median([abs(b["dlnw_dnode_per_V"]) for b in mb
                                if b["type"] == t and b["gate_class"] == "in_window"])) for t in ("LBUF", "UINV")},
        "max_abs_gate_uncut_minus_main_uV": K.max_abs(b["gate_uncut_minus_main_uV"] for b in mb),
        "max_abs_gate_uncut_minus_main_uV_by_class": {
            cl: K.max_abs([b["gate_uncut_minus_main_uV"] for b in mb if b["gate_class"] == cl])
            for cl in ("in_window", "node_outside_transfer_window", "u_cut_gate_0V")},
        "n_bounds_by_class": {cl: {t: sum(1 for b in mb if b["gate_class"] == cl and b["type"] == t)
                                   for t in ("LBUF", "UINV")}
                              for cl in ("in_window", "node_outside_transfer_window", "u_cut_gate_0V")},
        "w_main_equals_w_at_gate_0V_for_u_cut_bounds": all(b["w_rel_diff"] == 0.0 for b in mb if b["u_cut"]),
    }
    flags = {
        "idx": m["idx_all_equal"],
        "bound_sets": m["bound_sets_equal"],
        "no_violated_bound": m["no_violated_bound"],
        "x": m["max_abs_x_diff"] <= T["unsolved_rel"],
        "g_us": m["max_rel_g_us"] <= T["unsolved_rel"],
        "theta_eff": m["max_abs_theta_eff"] <= T["unsolved_rel"],
        "v_dl": m["max_abs_v_dl_V"] <= T["unsolved_rel"],
        "u_lsb": m["max_abs_u_lsb"] <= T["unsolved_rel"],
        "a_hat": m["max_abs_a_hat"] <= T["unsolved_rel"],
        "v_sl": m["max_abs_v_sl_V"] <= T["unsolved_rel"],
        "w": m["max_rel_w"] <= T["solved_rel"],
        "S": m["max_rel_S"] <= T["solved_rel"],
        "S_parts": m["max_abs_S_part_over_S"] <= T["solved_rel"],
        "t_pred": m["max_rel_t_pred"] <= T["solved_rel"],
        "decision_pred": m["decision_pred_equal"],
    }
    # the largest deviations, for the report
    worst_w = sorted(mb, key=lambda b: -abs(b["w_rel_diff"]))[:5]
    worst_s = sorted(mc, key=lambda c: -abs(c["S_rel_diff"]))[:3]
    with open(K.E4_CHECK_DIR / "compare.json") as f:
        e4_ref = json.load(f)["task2"]["model_side_all_48_cases"]
    e4_ref = {k: e4_ref[k] for k in ("max_rel_g_us", "max_abs_theta_eff", "max_abs_v_dl_V", "max_rel_w", "max_rel_S",
                                     "max_abs_a_hat", "max_abs_v_sl_diff_V", "max_abs_x_diff")}
    # where the largest 6T2M deviations sat (per-bound output of the E4 check)
    with open(K.E4_CHECK_DIR / "task2_model_bounds.csv") as f:
        e4b = list(csv.DictReader(f))
    e4_ref["max_rel_w_by_type"] = {t: K.max_abs(float(b["w_indep"]) / float(b["w_main"]) - 1.0
                                                for b in e4b if b["type"] == t) for t in ("L6T", "UINV")}
    summary = {
        "tolerances": {k: T[k] for k in ("unsolved_rel", "solved_rel")},
        "e4_check_6T2M_reference": e4_ref,
        "model_side_all_48_cases": m,
        "largest_w_deviations": [{k: b[k] for k in ("case", "bound", "type", "idx_indep", "w_indep", "w_main",
                                                     "w_rel_diff", "node_minus_vsp_mV", "implied_node_diff_V")}
                                 for b in worst_w],
        "largest_S_deviations": [{k: c[k] for k in ("case", "design", "time", "node", "S_indep", "S_main",
                                                     "S_rel_diff", "t_pred_rel_diff")} for c in worst_s],
        "flags": flags,
        "selection_extra": {"all_identical": sel["all_identical"],
                            "per_group": {k: {kk: g[kk] for kk in ("matched_pairs_indep", "matched_pairs_main_log",
                                                                    "candidates_indep", "candidates_main_log",
                                                                    "selection_identical")}
                                          for k, g in sel["groups"].items()}},
        "pass": bool(all(flags.values())),
        "elapsed_s": round(time.time() - t0, 1),
    }
    with open(out_dir / "task2_summary.json", "w") as f:
        json.dump(summary, f, indent=1)
    log(f"[task2] max rel diff: w {m['max_rel_w']:.2e} (LBUF {m['max_rel_w_LBUF']:.2e}, UINV {m['max_rel_w_UINV']:.2e}), "
        f"S {m['max_rel_S']:.2e}, t_pred {m['max_rel_t_pred']:.2e}; unsolved inputs: g {m['max_rel_g_us']:.1e}, "
        f"theta_eff {m['max_abs_theta_eff']:.1e}, V_DL {m['max_abs_v_dl_V']:.1e}; PASS={summary['pass']} "
        f"({summary['elapsed_s']} s)")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="output folder (created; must not exist)")
    a = ap.parse_args()
    o = Path(a.out).resolve()
    o.mkdir(parents=True, exist_ok=False)
    run(o)
