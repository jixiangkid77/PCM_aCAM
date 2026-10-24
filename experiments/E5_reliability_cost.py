"""E5: reliability statistics of the input-referred bound errors, errors of the drift compensation itself and time to
an error budget, for the transistor-level bounds of the E4 forests (no new SPICE).

Plan: experiments/E5_config.json (Chinese description: docs/E5_protocol.md). Reuses, unchanged, the E4 instrument
(Monte Carlo pools of the three branch types) and the E4 forest machinery (CellModel, theta_effs, pcm_states,
forest_state, eval_points), so that every bound is the E4 bound of the same forest, instance and PCM draw.

Parts
  e51  bound-error statistics on the full E4 set (5 datasets x 3 splits x 20 device seeds): rms with and without
       bound failures, mean, shares of |e| > 0.5, 1, 2 LSB, 99 % and 99.9 % quantiles of |e|, clipping, failures
  e52  error of the global factor: replica sampling (N = 16 ... 1024 against a 1e6-replica limit) and gain error
  e53  resolution of the adjustable quantity: data-line DAC (input remap; t0 and the supply node's fixed map for
       reference) and supply DAC (supply node)
  e54  time to an error budget at 13 times from 10 s to 10 years: no compensation, the two nodes, ideal scaling
Checks C1-C5 (plan section 9) are evaluated in the same run; a failed check marks the run "checks_failed".

    python experiments/E5_reliability_cost.py [--part e51|e52|e53|e54|all] [--smoke] [--workers 2]
Outputs (never overwritten): results/E5_<stamp>[_smoke]/
"""

from __future__ import annotations

import argparse
import collections
import csv
import importlib.util
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pcm_acam.acam import Scheme, replica_factor  # noqa: E402
from pcm_acam.io_utils import environment_info, make_logger, make_run_dir, now_jst, sha256_file, write_json  # noqa: E402
from pcm_acam.pcm_model import PCMParams  # noqa: E402
from pcm_acam.row_model import map_voltage  # noqa: E402
from pcm_acam.trees import dataset_bundle, quantize  # noqa: E402

_spec = importlib.util.spec_from_file_location("e4_forest_system", ROOT / "experiments" / "E4_forest_system.py")
E4 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(E4)

LSB = E4.LSB
PARTS = ("e51", "e52", "e53", "e54")
YEAR_S = 31536000.0
G: Dict[str, object] = {}          # set in main() before the worker processes are forked


# --------------------------------------------------------------------------------------------- helpers
def pcm_setup(cfg4: dict) -> Tuple[PCMParams, Scheme]:
    p = PCMParams.from_dict(cfg4["pcm_model"])
    rep = Scheme(name="replica", kind="absolute", devices=1, g_lo=cfg4["circuit"]["g_lo_us"],
                 g_hi=cfg4["circuit"]["g_hi_us"])
    return p, rep


def forest_bounds(ds: str, split: int, bits: int) -> Tuple[np.ndarray, np.ndarray]:
    rows = dataset_bundle(ds, split)["rows"]
    return quantize(rows.cons_thr, bits), rows.cons_kind


def bound_voltage(cm, g_now: np.ndarray, kind: np.ndarray, idx: np.ndarray, node: str, a_hat: float,
                  d_eff: float = None) -> Tuple[np.ndarray, float]:
    """Data-line voltage V* at which every bound switches (E4 effective_x before the input map) and the map factor.
    d_eff replaces V_SL - V_sp,nom of the node setting (supply DAC)."""
    gdiv, d_set, m = cm.node_setting(node, a_hat)
    if d_eff is not None:
        d_set = d_eff
    v = np.empty(g_now.shape)
    for k in (0, 1):
        sel = kind == k
        if sel.any():
            pool = cm.pool[k]
            v_sl = cm.v_sp_nom[k] + d_set
            cur = np.asarray(g_now[sel] / gdiv, float) * 1e-6 * (v_sl - pool.v_sp[idx[sel]])
            v[sel] = pool.v_at(idx[sel], cur)
    return v, m


def x_of_v(cm, v: np.ndarray, m: float, x_cap: float = 10.0) -> np.ndarray:
    """Input-referred bound of a bound voltage (E4 effective_x after the pool inversion)."""
    finite = np.isfinite(v)
    i0 = cm.map_curve.i_at(np.where(finite, v, 0.0))
    x = (i0 / (m * cm.d_v) * 1e6 - cm.g_lo) / (cm.g_hi - cm.g_lo)
    x = np.where(finite, x, np.where(v > 0, x_cap, -x_cap))
    return np.clip(x, -x_cap, x_cap)


def quantized_switch(v: np.ndarray, kind: np.ndarray, off: float, step: float) -> np.ndarray:
    """Effective switching voltage with read-time inputs rounded to the grid off + j step: lower bounds switch at
    V_g - step/2 (V_g lowest grid point >= v), upper bounds at V_g' + step/2 (V_g' highest grid point <= v)."""
    finite = np.isfinite(v)
    pos = (np.where(finite, v, 0.0) - off) / step
    lower = off + np.ceil(pos) * step - 0.5 * step
    upper = off + np.floor(pos) * step + 0.5 * step
    return np.where(finite, np.where(kind == 0, lower, upper), v)


def replica_draws(cfg4: dict, cfg5: dict, ds_idx: int, split: int, seed: int, n: int) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng([cfg4["base_seed"], ds_idx, split, seed, cfg5["e52"]["rng_stream"], n])
    n_t = len(cfg4["times_inference_s"])
    return {"z_rep_prog": rng.standard_normal(n), "z_rep_nu": rng.standard_normal(n),
            "z_rep_read": rng.standard_normal((n_t, n))}


def add_into(dst: dict, src: dict, max_idx: Tuple[int, ...] = ()) -> None:
    for key, v in src.items():
        if key not in dst:
            dst[key] = list(v)
            continue
        a = dst[key]
        for i, x in enumerate(v):
            a[i] = max(a[i], x) if i in max_idx else a[i] + x


def fmt_duration(t: float) -> str:
    if not np.isfinite(t):
        return ""
    for unit, s in (("y", YEAR_S), ("d", 86400.0), ("h", 3600.0), ("min", 60.0)):
        if t >= s:
            return f"{t / s:.3g} {unit}"
    return f"{t:.3g} s"


def loops(G_: dict):
    cfg5 = G_["cfg5"]
    return cfg5["designs"], cfg5["cell_sets"], cfg5["conditions"]


# --------------------------------------------------------------------------------------------- E5.1
def job_e51(args) -> dict:
    order, ds, ds_idx, split, seeds = args
    cfg4, cfg5, inst = G["cfg4"], G["cfg5"], G["inst"]
    designs, cell_sets, conditions = loops(G)
    p, rep = pcm_setup(cfg4)
    pts = E4.eval_points(cfg4)
    theta, kind = forest_bounds(ds, split, cfg4["bits"])
    n_pool = cfg4["pools"]["n_instances"]
    thr = cfg5["e51"]["exceed_lsb"]
    ext = cfg5["extreme_lsb"]
    sp = cfg5["e51"]["spot"]
    acc: Dict[tuple, list] = {}
    absv: Dict[tuple, list] = collections.defaultdict(list)
    clip: Dict[tuple, list] = {}
    robust: Dict[tuple, list] = {}
    robust_abs: Dict[tuple, list] = collections.defaultdict(list)
    spot: dict = {}
    for seed in seeds:
        draws, idx = E4.forest_state(cfg4, ds_idx, split, seed, theta.size, n_pool)
        for did in designs:
            for cs in cell_sets:
                cm = E4.CellModel(inst, cfg4, did, cs)
                for ci, cond in enumerate(conditions):
                    res, g_t, clipped = E4.theta_effs(cm, theta, kind, idx, draws, cond, pts, p, rep)
                    if ci == 0:                                   # clipping does not depend on the condition
                        for kd in (0, 1):
                            c = clip.setdefault((did, cs, cm.types[kd]), [0, 0])
                            c[0] += int(np.sum(clipped[kind == kd]))
                            c[1] += int(np.sum(kind == kd))
                    for tlab, node, a_hat, th, g_now in res:
                        e = LSB * (th - theta)
                        for kd in (0, 1):
                            sel = kind == kd
                            ek, gk = e[sel], g_now[sel]
                            ae = np.abs(ek)
                            big = ae > ext
                            key = (did, cs, cm.types[kd], cond, tlab, node)
                            a = acc.setdefault(key, [0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0] + [0] * len(thr) + [-1])
                            a[0] += int(sel.sum())
                            a[1] += int(big.sum())
                            a[2] += int(np.sum(gk[big] == 0))
                            a[3] += float(np.sum(ek ** 2))
                            a[4] += float(np.sum(ek))
                            a[5] += float(np.sum(ek[~big] ** 2))
                            a[6] += float(np.sum(ek[~big]))
                            a[7] += int((~big).sum())
                            for j, k_ in enumerate(thr):
                                a[8 + j] += int(np.sum(ae > k_))
                            if big.any():
                                a[-1] = max(a[-1], int(np.round(LSB * theta[sel][big].max())))
                            absv[key].append(ae.astype(np.float32))
                            if split == 0 and seed == 0 and cond == "all":
                                rk = (did, cs, cm.types[kd], tlab, node)
                                r = robust.setdefault(rk, [0.0, 0, 0, -1, 0])
                                r[0] += float(np.sum(ek[~big] ** 2))
                                r[1] += int((~big).sum())
                                r[2] += int(big.sum())
                                if big.any():
                                    r[3] = max(r[3], int(np.round(LSB * theta[sel][big].max())))
                                r[4] += int(np.sum(gk[big] == 0))
                                robust_abs[rk].append(ae[~big])
                        if (ds, split, seed, did, cs, cond, tlab) == (sp["dataset"], sp["split"], sp["seed"], sp["design"],
                                                                     sp["cell_set"], sp["condition"], sp["time"]) \
                                and node in sp["nodes"]:
                            spot[node] = e.copy()
                            spot["kind"], spot["theta"], spot["idx"] = kind.copy(), theta.copy(), idx.copy()
    absv_c = {k_: np.concatenate(v) for k_, v in absv.items()}
    robust_c = {k_: np.concatenate(v) for k_, v in robust_abs.items()}
    return {"order": order, "ds": ds, "split": split, "acc": acc, "absv": absv_c, "clip": clip, "robust": robust,
            "robust_abs": robust_c, "spot": spot}


def post_e51(results: List[dict], run: Path, smoke: bool, log) -> dict:
    cfg5 = G["cfg5"]
    thr = cfg5["e51"]["exceed_lsb"]
    pct = cfg5["e51"]["percentiles"]
    acc: Dict[tuple, list] = {}
    clip: Dict[tuple, list] = {}
    robust: Dict[tuple, list] = {}
    absv: Dict[tuple, list] = collections.defaultdict(list)
    robust_abs: Dict[tuple, list] = collections.defaultdict(list)
    for r in results:
        add_into(acc, r["acc"], max_idx=(8 + len(thr),))
        add_into(clip, r["clip"])
        add_into(robust, r["robust"], max_idx=(3,))
        for k_, v in r["absv"].items():
            absv[k_].append(v)
        for k_, v in r["robust_abs"].items():
            robust_abs[k_].append(v)
    rows = []
    stats = {}
    for key in sorted(acc):
        a = acc[key]
        n, n_ext, n_zero, s2, s1, s2o, s1o, n_ok = a[:8]
        q = np.percentile(np.concatenate(absv[key]).astype(float), pct)
        c = clip[key[:3]]
        row = {"design": key[0], "cell_set": key[1], "type": key[2], "condition": key[3], "time": key[4],
               "node": key[5], "n": n, "n_extreme": n_ext, "n_extreme_zero_g": n_zero,
               "extreme_max_theta_x15": a[-1] if n_ext else "",
               "rms_all_lsb": math.sqrt(s2 / n), "mean_all_lsb": s1 / n,
               "rms_noext_lsb": math.sqrt(s2o / n_ok), "mean_noext_lsb": s1o / n_ok,
               **{f"p_exc_{k_:g}": a[8 + j] / n for j, k_ in enumerate(thr)},
               **{f"q{p_:g}_abs_lsb": float(v) for p_, v in zip(pct, q)},
               "clipped": c[0], "clipped_fraction": c[0] / c[1]}
        rows.append(row)
        stats[key] = row
    E4.write_csv(run / "e51_stats.csv", rows)
    E4.write_csv(run / "e51_clipping.csv", [{"design": k_[0], "cell_set": k_[1], "type": k_[2], "clipped": v[0],
                                             "constraints": v[1], "fraction": v[0] / v[1]} for k_, v in sorted(clip.items())])
    checks = {}
    # spot (independent re-check 1)
    sp = cfg5["e51"]["spot"]
    spot = next((r["spot"] for r in results if r["spot"]), None)
    if spot:
        kd = spot["kind"]
        E4.write_csv(run / "e51_spot_bounds.csv",
                     [{"k": int(i), "kind": int(kd[i]), "theta": float(spot["theta"][i]), "instance": int(spot["idx"][i]),
                       **{f"e_{nd}_lsb": float(spot[nd][i]) for nd in sp["nodes"]}} for i in range(kd.size)])
        sel = kd == (0 if sp["type"] in ("L6T", "LBUF") else 1)
        write_json(run / "e51_spot.json", {"case": sp, "n_bounds_of_type": int(sel.sum()),
                                           "counts": {nd: {"abs_gt_0.5": int(np.sum(np.abs(spot[nd][sel]) > 0.5)),
                                                           "abs_gt_1": int(np.sum(np.abs(spot[nd][sel]) > 1.0))}
                                                      for nd in sp["nodes"]}})
    if smoke:
        checks["C1"] = {"pass": None, "note": "not evaluated in a smoke run (needs the full set)"}
        checks["C2"] = {"pass": None, "note": "not evaluated in a smoke run (needs the full set)"}
        return checks
    e4run = ROOT / cfg5["reuse"]["e4_forest_run"]
    worst_n, worst_rel = 0, 0.0
    missing = []
    with open(e4run / "cell_errors.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            key = (r["design"], r["cell_set"], r["type"], r["condition"], r["time"], r["node"])
            if key not in stats:
                missing.append("/".join(key))
                continue
            worst_n = max(worst_n, abs(int(r["n"]) - stats[key]["n"]))
            worst_rel = max(worst_rel, abs(stats[key]["rms_all_lsb"] / float(r["rms_lsb"]) - 1))
    checks["C1"] = {"max_abs_n_diff": worst_n, "max_rel_rms_diff": worst_rel, "missing": missing,
                    "pass": bool(worst_n == 0 and worst_rel <= 1e-9 and not missing)}
    worst_cnt, worst_rel2 = 0, 0.0
    missing = []
    with open(e4run / "analysis_cell_errors_robust.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            key = (r["design"], r["cell_set"], r["type"], r["time"], r["node"])
            if key not in robust:
                missing.append("/".join(key))
                continue
            s2, n_ok, n_big, _, n_zero = robust[key]
            med = float(np.median(np.concatenate(robust_abs[key])))
            worst_cnt = max(worst_cnt, abs(int(r["n"]) - (n_ok + n_big)), abs(int(r["n_extreme"]) - n_big),
                            abs(int(r["extreme_with_zero_conductance"]) - n_zero))
            worst_rel2 = max(worst_rel2, abs(math.sqrt(s2 / n_ok) / float(r["rms_lsb_without_extreme"]) - 1),
                             abs(med / float(r["median_abs_lsb"]) - 1))
    checks["C2"] = {"max_abs_count_diff": worst_cnt, "max_rel_value_diff": worst_rel2, "missing": missing,
                    "pass": bool(worst_cnt == 0 and worst_rel2 <= 1e-9 and not missing)}
    log(f"C1 max |dn| {worst_n}, max rel rms diff {worst_rel:.2e}; C2 max count diff {worst_cnt}, "
        f"max rel diff {worst_rel2:.2e}")
    return checks


# --------------------------------------------------------------------------------------------- E5.2
def job_e52(args) -> dict:
    order, ds, ds_idx, split, seeds = args
    cfg4, cfg5, inst = G["cfg4"], G["cfg5"], G["inst"]
    designs, cell_sets, conditions = loops(G)
    c52 = cfg5["e52"]
    p, rep = pcm_setup(cfg4)
    pts = E4.eval_points(cfg4)
    theta, kind = forest_bounds(ds, split, cfg4["bits"])
    n_pool = cfg4["pools"]["n_instances"]
    ext = cfg5["extreme_lsb"]
    times = cfg4["times_inference_s"]
    tl = list(times)
    acc: Dict[tuple, list] = {}
    ahat: Dict[tuple, list] = {}
    c4 = 0.0
    for seed in seeds:
        draws, idx = E4.forest_state(cfg4, ds_idx, split, seed, theta.size, n_pool)
        big = replica_draws(cfg4, cfg5, ds_idx, split, seed, c52["n_limit"])
        small = {n: replica_draws(cfg4, cfg5, ds_idx, split, seed, n) for n in c52["replica_counts"]}
        a_rep = {}
        for tlab in c52["times"]:
            t_idx, t_inf = tl.index(tlab), float(times[tlab])
            a_inf = replica_factor(rep, big, "all", t_idx, t_inf, p)
            a_rep[tlab] = (a_inf, {n: replica_factor(rep, d, "all", t_idx, t_inf, p) for n, d in small.items()})
        for di, did in enumerate(designs):
            for si, cs in enumerate(cell_sets):
                cm = E4.CellModel(inst, cfg4, did, cs)
                for cond in conditions:
                    ref, g_t, _ = E4.theta_effs(cm, theta, kind, idx, draws, cond, pts, p, rep)
                    ref_e = {(r_[0], r_[1]): LSB * (r_[3] - theta) for r_ in ref}
                    for tlab in c52["times"]:
                        t_idx, t_inf = tl.index(tlab), float(times[tlab])
                        g_now, a_e4 = E4.pcm_states(g_t, draws, cond, t_idx, t_inf, p, rep)
                        if cond == "unif":
                            a_inf = a_e4
                            variants = [("c", g, a_inf * (1 + g)) for g in c52["gains"]]
                        else:
                            a_inf, a_n = a_rep[tlab]
                            variants = ([("a", 64, a_e4)] + [("c", g, a_inf * (1 + g)) for g in c52["gains"]]
                                        + [("d", n, a_n[n]) for n in c52["replica_counts"]])
                            if di == 0 and si == 0:
                                for var, par, a_v in variants:
                                    h = ahat.setdefault((tlab, var, par), [0, 0.0, 0.0])
                                    h[0] += 1
                                    h[1] += a_v / a_inf - 1
                                    h[2] += (a_v / a_inf - 1) ** 2
                        for node in c52["nodes"]:
                            e_b = LSB * (cm.theta_eff(g_now, kind, idx, node, a_inf) - theta)
                            if cond == "unif":                       # g = 0 is the E4 bound (C4)
                                c4 = max(c4, float(np.max(np.abs(e_b - ref_e[(tlab, node)]))))
                            for var, par, a_v in [("b", c52["n_limit"] if cond == "all" else 0, a_inf)] + variants:
                                e_v = e_b if var == "b" else LSB * (cm.theta_eff(g_now, kind, idx, node, a_v) - theta)
                                if var == "a":
                                    c4 = max(c4, float(np.max(np.abs(e_v - ref_e[(tlab, node)]))))
                                de = e_v - e_b
                                for kd in (0, 1):
                                    sel = kind == kd
                                    ok_b = np.abs(e_b[sel]) <= ext
                                    ok_v = np.abs(e_v[sel]) <= ext
                                    ok = ok_b & ok_v
                                    d_ = de[sel][ok]
                                    key = (did, cs, cm.types[kd], cond, tlab, node, var, par)
                                    a = acc.setdefault(key, [0, 0.0, 0.0, 0.0, 0.0, 0, 0.0, 0])
                                    a[0] += int(ok.sum())
                                    a[1] += float(np.sum(d_ ** 2))
                                    a[2] += float(np.sum(d_))
                                    a[3] = max(a[3], float(np.max(np.abs(d_)))) if d_.size else a[3]
                                    a[4] += float(np.sum(e_v[sel][ok_v] ** 2))
                                    a[5] += int(ok_v.sum())
                                    a[6] += float(np.sum(e_b[sel][ok_b] ** 2))
                                    a[7] += int(ok_b.sum())
    return {"order": order, "ds": ds, "split": split, "acc": acc, "ahat": ahat, "c4": c4}


def gain_limit(gs: List[float], vals: List[float], budget: float) -> float:
    """|g| at which the added rms reaches the budget (linear interpolation from (0, 0)); inf beyond the tested range."""
    pts = sorted(zip([0.0] + list(gs), [0.0] + list(vals)))
    for (g0, v0), (g1, v1) in zip(pts[:-1], pts[1:]):
        if v1 > budget >= v0:
            return g0 + (budget - v0) / (v1 - v0) * (g1 - g0)
    return math.inf


def post_e52(results: List[dict], run: Path, smoke: bool, log) -> dict:
    cfg5 = G["cfg5"]
    c52 = cfg5["e52"]
    budget = c52["budget_added_rms_lsb"]
    acc: Dict[tuple, list] = {}
    ahat: Dict[tuple, list] = {}
    c4 = 0.0
    for r in results:
        add_into(acc, r["acc"], max_idx=(3,))
        add_into(ahat, r["ahat"])
        c4 = max(c4, r["c4"])
    rows = []
    added = {}
    for key in sorted(acc, key=lambda k_: tuple(str(x) for x in k_)):
        n, s2, s1, mx, s2v, nv, s2b, nb = acc[key]
        rv, rb = math.sqrt(s2v / nv), math.sqrt(s2b / nb)
        added[key] = math.sqrt(s2 / n)
        rows.append({"design": key[0], "cell_set": key[1], "type": key[2], "condition": key[3], "time": key[4],
                     "node": key[5], "variant": key[6], "param": key[7], "n": n, "added_rms_lsb": added[key],
                     "added_mean_lsb": s1 / n, "added_max_abs_lsb": mx, "rms_noext_lsb": rv,
                     "rms_noext_reference_lsb": rb, "rms_change_lsb": rv - rb})
    E4.write_csv(run / "e52_added.csv", rows)
    E4.write_csv(run / "e52_ahat.csv", [{"time": k_[0], "variant": k_[1], "param": k_[2], "draws": v[0],
                                         "mean_rel_to_limit": v[1] / v[0],
                                         "rms_rel_to_limit": math.sqrt(v[2] / v[0])} for k_, v in sorted(
                                             ahat.items(), key=lambda kv: tuple(str(x) for x in kv[0]))])
    a1y = E4.a_uniform(YEAR_S, G["cfg4"]["pcm_model"]["t0"])
    lim_rows = []
    groups = sorted({k_[:6] for k_ in acc}, key=lambda k_: tuple(str(x) for x in k_))
    for grp in groups:
        pos = [g for g in c52["gains"] if g > 0]
        neg = [g for g in c52["gains"] if g < 0]
        gp = gain_limit(pos, [added[grp + ("c", g)] for g in pos], budget)
        gn = gain_limit([-g for g in neg], [added[grp + ("c", g)] for g in neg], budget)
        r_ = gp / (1 + gp) if np.isfinite(gp) else math.nan
        upd = math.ceil(abs(math.log(a1y)) / -math.log(1 - r_)) if np.isfinite(r_) else ""
        row = {"design": grp[0], "cell_set": grp[1], "type": grp[2], "condition": grp[3], "time": grp[4],
               "node": grp[5], "g_max_pos": gp, "g_max_neg": gn, "g_max": min(gp, gn), "update_fraction_r": r_,
               "updates_first_year": upd}
        if grp[3] == "all":
            ns = c52["replica_counts"]
            vals = {n: added[grp + ("d", n)] for n in ns}
            ok = [n for n in ns if vals[n] <= budget]
            row.update({f"added_rms_N{n}": vals[n] for n in ns})
            row["added_rms_E4_64"] = added[grp + ("a", 64)]
            row["min_N_within_budget"] = min(ok) if ok else f"> {max(ns)}"
        lim_rows.append(row)
    E4.write_csv(run / "e52_limits.csv", lim_rows)
    log(f"C4 max |de| {c4:.3e} LSB")
    return {"C4": {"max_abs_de_lsb": c4, "pass": bool(c4 <= 1e-12)}}


# --------------------------------------------------------------------------------------------- E5.3
def job_e53(args) -> dict:
    order, ds, ds_idx, split, seeds = args
    cfg4, cfg5, inst = G["cfg4"], G["cfg5"], G["inst"]
    designs, cell_sets, conditions = loops(G)
    c53 = cfg5["e53"]
    p, rep = pcm_setup(cfg4)
    pts = E4.eval_points(cfg4)
    theta, kind = forest_bounds(ds, split, cfg4["bits"])
    n_pool = cfg4["pools"]["n_instances"]
    ext = cfg5["extreme_lsb"]
    times = cfg4["times_inference_s"]
    tl = list(times)
    vfs = c53["full_scale_v"]
    sp = c53["spot"]
    acc: Dict[tuple, list] = {}
    c3 = 0.0
    spot: dict = {}
    for seed in seeds:
        draws, idx = E4.forest_state(cfg4, ds_idx, split, seed, theta.size, n_pool)
        for did in designs:
            for cs in cell_sets:
                cm = E4.CellModel(inst, cfg4, did, cs)
                for cond in conditions:
                    ref, g_t, _ = E4.theta_effs(cm, theta, kind, idx, draws, cond, pts, p, rep)
                    ref_x = {(r_[0], r_[1]): r_[3] for r_ in ref}
                    in_spot = (ds, split, seed, did, cs, cond) == (sp["dataset"], sp["split"], sp["seed"], sp["design"],
                                                                  sp["cell_set"], sp["condition"])
                    for tlab, node in c53["points"]:
                        t_idx, t_inf = tl.index(tlab), float(times[tlab])
                        g_now, a_hat = E4.pcm_states(g_t, draws, cond, t_idx, t_inf, p, rep)
                        v, m = bound_voltage(cm, g_now, kind, idx, node, a_hat)
                        x0 = x_of_v(cm, v, m)
                        c3 = max(c3, float(np.max(np.abs(LSB * (x0 - ref_x[(tlab, node)])))))
                        ok = np.abs(LSB * (x0 - theta)) <= ext
                        m01 = map_voltage(cm.map_curve, np.array([0.0, 1.0]), m, cm.d_v, cm.g_lo, cm.g_hi)
                        cases = []
                        for b in c53["bits"]:
                            cases.append(("DL", "full", b, 0.0, vfs / (2 ** b - 1)))
                            cases.append(("DL", "span", b, float(m01[0]), float(m01[1] - m01[0]) / (2 ** b - 1)))
                            if node == "supply":
                                cases.append(("SL", "+half", b, None, vfs / (2 ** b - 1)))
                                cases.append(("SL", "-half", b, None, vfs / (2 ** b - 1)))
                        for qty, grid, b, off, step in cases:
                            if qty == "DL":
                                vq = quantized_switch(v, kind, off, step)
                                d_eff = None
                            else:     # one V_SL per branch type: its rounding error is common to all bounds
                                d_eff = cm.d_v / a_hat + (0.5 if grid == "+half" else -0.5) * step
                                vq, _ = bound_voltage(cm, g_now, kind, idx, node, a_hat, d_eff=d_eff)
                            xq = x_of_v(cm, vq, m)
                            de = LSB * (xq - x0)
                            lost = ok & ~np.isfinite(vq)
                            use = ok & np.isfinite(vq)
                            for kd in (0, 1):
                                sel = kind == kd
                                d_ = de[sel & use]
                                key = (did, cs, cm.types[kd], cond, tlab, node, qty, grid, b)
                                a = acc.setdefault(key, [0, 0.0, 0.0, 0])
                                a[0] += int(d_.size)
                                a[1] += float(np.sum(d_ ** 2))
                                a[2] = max(a[2], float(np.max(np.abs(d_)))) if d_.size else a[2]
                                a[3] += int(np.sum(lost[sel]))
                            if in_spot and b == sp["bits"] and tlab == "1y":
                                spot[(node, qty, grid)] = {"v": v, "vq": vq, "x0": x0, "xq": xq, "off": off,
                                                           "step": step, "m": m, "a_hat": a_hat, "g_now": g_now,
                                                           "idx": idx, "kind": kind, "theta": theta, "d_eff": d_eff,
                                                           "ok": ok}
    spot_rows = []
    if spot:
        ok_all = np.logical_and.reduce([s["ok"] for s in spot.values()])
        cand = np.flatnonzero(ok_all)
        pick = np.sort(np.random.default_rng([cfg4["base_seed"], sp["rng_stream"]]).choice(cand, sp["n_bounds"],
                                                                                            replace=False))
        for (node, qty, grid), s in sorted(spot.items()):
            for i in pick:
                kd = int(s["kind"][i])
                spot_rows.append({"config": f"{node}/{qty}/{grid}/{sp['bits']} bit", "k": int(i), "kind": kd,
                                  "instance": int(s["idx"][i]), "theta": float(s["theta"][i]),
                                  "g_now_us": float(s["g_now"][i]), "a_hat": s["a_hat"], "map_factor": s["m"],
                                  "grid_offset_v": s["off"] if s["off"] is not None else "", "grid_step_v": s["step"],
                                  "d_eff_v": s["d_eff"] if s["d_eff"] is not None else "",
                                  "v_bound_v": float(s["v"][i]), "v_switch_quantized_v": float(s["vq"][i]),
                                  "x_exact": float(s["x0"][i]), "x_quantized": float(s["xq"][i]),
                                  "added_error_lsb": float(LSB * (s["xq"][i] - s["x0"][i]))})
    return {"order": order, "ds": ds, "split": split, "acc": acc, "c3": c3, "spot_rows": spot_rows}


def post_e53(results: List[dict], run: Path, smoke: bool, log) -> dict:
    cfg5 = G["cfg5"]
    c53 = cfg5["e53"]
    budget = c53["budget_added_rms_lsb"]
    acc: Dict[tuple, list] = {}
    c3 = 0.0
    spot_rows = []
    for r in results:
        add_into(acc, r["acc"], max_idx=(2,))
        c3 = max(c3, r["c3"])
        spot_rows += r["spot_rows"]
    rows = []
    rms = {}
    for key in sorted(acc, key=lambda k_: tuple(str(x) for x in k_)):
        n, s2, mx, lost = acc[key]
        rms[key] = math.sqrt(s2 / n) if n else math.nan
        rows.append({"design": key[0], "cell_set": key[1], "type": key[2], "condition": key[3], "time": key[4],
                     "node": key[5], "quantity": key[6], "grid": key[7], "bits": key[8], "n": n,
                     "added_rms_lsb": rms[key], "added_max_abs_lsb": mx, "lost": lost})
    E4.write_csv(run / "e53_added.csv", rows)
    for key in [k_ for k_ in rms if k_[6] == "SL" and k_[7] == "+half"]:      # worst case of the two signs
        rms[key[:7] + ("worst", key[8])] = max(rms[key], rms[key[:7] + ("-half", key[8])])
    min_rows = []
    for grp in sorted({k_[:8] for k_ in rms}, key=lambda k_: tuple(str(x) for x in k_)):
        ok = [b for b in c53["bits"] if rms[grp + (b,)] <= budget]
        min_rows.append({"design": grp[0], "cell_set": grp[1], "type": grp[2], "condition": grp[3], "time": grp[4],
                         "node": grp[5], "quantity": grp[6], "grid": grp[7],
                         "min_bits": min(ok) if ok else f"> {max(c53['bits'])}",
                         **{f"added_rms_{b}bit": rms[grp + (b,)] for b in c53["bits"]}})
    E4.write_csv(run / "e53_min_bits.csv", min_rows)
    E4.write_csv(run / "e53_spot.csv", spot_rows)
    log(f"C3 max |de| {c3:.3e} LSB")
    return {"C3": {"max_abs_de_lsb": c3, "pass": bool(c3 <= 1e-12)}}


# --------------------------------------------------------------------------------------------- E5.4
def job_e54(args) -> dict:
    order, ds, ds_idx, split, seeds = args
    cfg4, cfg5, inst = G["cfg4"], G["cfg5"], G["inst"]
    designs, cell_sets, conditions = loops(G)
    c54 = cfg5["e54"]
    p, rep = pcm_setup(cfg4)
    pts = E4.eval_points(cfg4)
    theta, kind = forest_bounds(ds, split, cfg4["bits"])
    k = theta.size
    n_pool = cfg4["pools"]["n_instances"]
    ext = cfg5["extreme_lsb"]
    thr = cfg5["e51"]["exceed_lsb"]
    tgrid = list(c54["times_s"].items())
    overlap = [t for t in cfg4["times_evaluated"] if t in c54["times_s"]]
    acc: Dict[tuple, list] = {}
    ref_acc: Dict[tuple, list] = {}
    c5u = 0.0

    def accumulate(store, key, e, kd_sel):
        ek = e[kd_sel]
        ae = np.abs(ek)
        big = ae > ext
        a = store.setdefault(key, [0, 0, 0.0, 0] + [0] * len(thr))
        a[0] += int(kd_sel.sum())
        a[1] += int(big.sum())
        a[2] += float(np.sum(ek[~big] ** 2))
        a[3] += int((~big).sum())
        for j, k_ in enumerate(thr):
            a[4 + j] += int(np.sum(ae > k_))

    for seed in seeds:
        draws, idx = E4.forest_state(cfg4, ds_idx, split, seed, k, n_pool)
        rng = np.random.default_rng([cfg4["base_seed"], ds_idx, split, seed, c54["rng_stream"]])
        z_cell = rng.standard_normal((len(tgrid), k))
        z_rep = rng.standard_normal((len(tgrid), cfg4["replica_cells"]))
        d54 = {"z_prog": draws["z_prog"], "z_nu": draws["z_nu"], "z_read": z_cell[:, :, None],
               "z_rep_prog": draws["z_rep_prog"], "z_rep_nu": draws["z_rep_nu"], "z_rep_read": z_rep}
        for did in designs:
            for cs in cell_sets:
                cm = E4.CellModel(inst, cfg4, did, cs)
                for cond in conditions:
                    ref, g_t, _ = E4.theta_effs(cm, theta, kind, idx, draws, cond, pts, p, rep)
                    ref_e = {(r_[0], r_[1]): LSB * (r_[3] - theta) for r_ in ref}
                    g0, _ = E4.pcm_states(g_t, d54, cond, 0, 0.0, p, rep)
                    e0 = LSB * (cm.theta_eff(g0, kind, idx, "t0", 1.0) - theta)
                    for kd in (0, 1):
                        accumulate(acc, (did, cs, cm.types[kd], cond, "t0", "t0"), e0, kind == kd)
                    for j, (tlab, ts) in enumerate(tgrid):
                        g_now, a_hat = E4.pcm_states(g_t, d54, cond, j, float(ts), p, rep)
                        for node in c54["nodes"]:
                            nd = "t0" if node == "none" else node
                            e = LSB * (cm.theta_eff(g_now, kind, idx, nd, a_hat) - theta)
                            for kd in (0, 1):
                                accumulate(acc, (did, cs, cm.types[kd], cond, tlab, node), e, kind == kd)
                            if tlab in overlap and node != "none":
                                er = ref_e[(tlab, node)]
                                if cond == "unif":
                                    c5u = max(c5u, float(np.max(np.abs(e - er))))
                                for kd in (0, 1):
                                    accumulate(ref_acc, (did, cs, cm.types[kd], cond, tlab, node), er, kind == kd)
    return {"order": order, "ds": ds, "split": split, "acc": acc, "ref_acc": ref_acc, "c5u": c5u}


def time_to_budget(t_s: List[float], vals: List[float], v_t0: float, budget: float) -> Tuple[str, float]:
    if v_t0 > budget:
        return "exceeded at t0", 0.0
    for i, v in enumerate(vals):
        if v > budget:
            if i == 0:
                return f"< {fmt_duration(t_s[0])}", math.nan
            lt0, lt1 = math.log(t_s[i - 1]), math.log(t_s[i])
            return "interpolated", math.exp(lt0 + (budget - vals[i - 1]) / (v - vals[i - 1]) * (lt1 - lt0))
    return f"> {fmt_duration(t_s[-1])}", math.inf


def post_e54(results: List[dict], run: Path, smoke: bool, log) -> dict:
    cfg5 = G["cfg5"]
    c54 = cfg5["e54"]
    thr = cfg5["e51"]["exceed_lsb"]
    acc: Dict[tuple, list] = {}
    ref_acc: Dict[tuple, list] = {}
    c5u = 0.0
    for r in results:
        add_into(acc, r["acc"])
        add_into(ref_acc, r["ref_acc"])
        c5u = max(c5u, r["c5u"])

    def metrics(a):
        return {"n": a[0], "n_extreme": a[1], "rms_noext_lsb": math.sqrt(a[2] / a[3]),
                **{f"p_exc_{k_:g}": a[4 + j] / a[0] for j, k_ in enumerate(thr)}}

    curves = {key: metrics(a) for key, a in acc.items()}
    tgrid = list(c54["times_s"].items())
    order_t = {lab: i for i, (lab, _) in enumerate([("t0", 0)] + tgrid)}
    E4.write_csv(run / "e54_curves.csv",
                 [{"design": k_[0], "cell_set": k_[1], "type": k_[2], "condition": k_[3], "node": k_[5],
                   "time": k_[4], "t_s": dict(tgrid).get(k_[4], 0), **v}
                  for k_, v in sorted(curves.items(), key=lambda kv: (kv[0][:4], kv[0][5], order_t[kv[0][4]]))])
    budget_rows = []
    groups = sorted({(k_[0], k_[1], k_[2], k_[3]) for k_ in acc})
    t_s = [float(t) for _, t in tgrid]
    for grp in groups:
        v_t0 = curves[grp + ("t0", "t0")]
        for node in c54["nodes"]:
            series = [curves[grp + (lab, node)] for lab, _ in tgrid]
            for metric, col, budgets in (("rms_noext", "rms_noext_lsb", c54["budgets"]["rms_noext_lsb"]),
                                         ("p_exc_1", "p_exc_1", c54["budgets"]["p_exc_1"])):
                for bud in budgets:
                    status, tb = time_to_budget(t_s, [s[col] for s in series], v_t0[col], bud)
                    budget_rows.append({"design": grp[0], "cell_set": grp[1], "type": grp[2], "condition": grp[3],
                                        "node": node, "metric": metric, "budget": bud, "value_t0": v_t0[col],
                                        "value_1y": curves[grp + ("1y", node)][col], "status": status,
                                        "t_budget_s": tb if status == "interpolated" else "",
                                        "t_budget": fmt_duration(tb) if status == "interpolated" else status,
                                        "refresh_per_year": (YEAR_S / tb) if status == "interpolated" else ""})
    E4.write_csv(run / "e54_budget.csv", budget_rows)
    worst_rel = 0.0
    for key, a in ref_acc.items():
        if key[3] != "all":
            continue
        worst_rel = max(worst_rel, abs(math.sqrt(acc[key][2] / acc[key][3]) / math.sqrt(a[2] / a[3]) - 1))
    log(f"C5 unif max |de| {c5u:.3e} LSB; all max rel rms diff {worst_rel:.4f}")
    if smoke:     # the read-noise comparison needs the full set (a few hundred bounds per key in a smoke run)
        return {"C5": {"unif_max_abs_de_lsb": c5u, "all_max_rel_rms_diff": worst_rel,
                       "note": "condition all not evaluated in a smoke run (sample too small)", "pass": bool(c5u <= 1e-12)}}
    return {"C5": {"unif_max_abs_de_lsb": c5u, "all_max_rel_rms_diff": worst_rel,
                   "pass": bool(c5u <= 1e-12 and worst_rel <= 0.03)}}


# --------------------------------------------------------------------------------------------- driver
JOBS = {"e51": job_e51, "e52": job_e52, "e53": job_e53, "e54": job_e54}
POSTS = {"e51": post_e51, "e52": post_e52, "e53": post_e53, "e54": post_e54}


def hashes(cfg_path: Path, cfg5: dict) -> Dict[str, str]:
    files = [cfg_path, Path(__file__).resolve(), ROOT / cfg5["reuse"]["e4_config"], ROOT / cfg5["reuse"]["e4_script"]]
    files += [ROOT / f"src/pcm_acam/{m}.py" for m in ("row_model", "acam", "pcm_model", "trees", "io_utils")]
    return {str(p_.relative_to(ROOT)): sha256_file(p_) for p_ in files}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", choices=PARTS + ("all",), default="all")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--config", default=str(ROOT / "experiments" / "E5_config.json"))
    ap.add_argument("--workers", type=int, default=2, help="processes (results do not depend on it)")
    args = ap.parse_args()
    cfg_path = Path(args.config).resolve()
    cfg5 = json.loads(cfg_path.read_text(encoding="utf-8"))
    cfg4 = json.loads((ROOT / cfg5["reuse"]["e4_config"]).read_text(encoding="utf-8"))
    run = make_run_dir(ROOT / "results", cfg5["run_prefix"], args.smoke)
    log = make_logger(run / "run.log")
    inst_path = ROOT / cfg5["reuse"]["instrument"]
    info = {"experiment": cfg5["run_prefix"], "parts": args.part, "smoke": args.smoke,
            "start_jst": now_jst().isoformat(timespec="seconds"), "environment": environment_info(),
            "sha256": hashes(cfg_path, cfg5), "instrument": cfg5["reuse"]["instrument"],
            "instrument_sha256": sha256_file(inst_path), "workers": args.workers}
    try:
        import sklearn
        info["environment"]["sklearn"] = sklearn.__version__
    except ImportError:
        pass
    write_json(run / "run_info.json", info)
    write_json(run / "config_used.json", cfg5)
    if info["instrument_sha256"] != cfg5["reuse"]["instrument_sha256"]:
        log("instrument SHA-256 differs from the plan; stop")
        return 2
    G["cfg4"], G["cfg5"] = cfg4, cfg5
    G["inst"] = E4.Instrument.load(inst_path)
    names = list(cfg4["datasets"])
    parts = list(PARTS) if args.part == "all" else [args.part]
    log(f"E5 parts {parts}; run dir {run.name}; instrument {cfg5['reuse']['instrument']}")
    checks: Dict[str, dict] = {}
    t_all = time.time()
    for part in parts:
        if args.smoke:
            datasets, splits, n_seed = cfg5["smoke"]["datasets"], cfg5["smoke"]["split_seeds"], cfg5["smoke"]["device_seeds"]
        else:
            s = cfg5["full_set"]                       # every part runs on the full E4 set (plan v1.0, D3)
            datasets, splits, n_seed = names, s["split_seeds"], s["device_seeds"]
        jobs = [(i, ds, names.index(ds), sp_, list(range(n_seed)))
                for i, (ds, sp_) in enumerate((ds, sp_) for ds in datasets for sp_ in splits)]
        size = {"synthetic": 0, "digits": 1, "breast_cancer": 2, "wine": 3, "iris": 4}
        run_order = sorted(jobs, key=lambda j: size.get(j[1], 9))
        t0 = time.time()
        results = []
        if args.workers > 1:
            import multiprocessing as mp
            with mp.get_context("fork").Pool(args.workers) as pool:
                for r in pool.imap_unordered(JOBS[part], run_order):
                    results.append(r)
                    log(f"{part}: {r['ds']} split {r['split']} done ({time.time() - t0:.0f} s)")
        else:
            for j in run_order:
                r = JOBS[part](j)
                results.append(r)
                log(f"{part}: {r['ds']} split {r['split']} done ({time.time() - t0:.0f} s)")
        results.sort(key=lambda r: r["order"])
        checks.update(POSTS[part](results, run, args.smoke, log))
        info[f"elapsed_{part}_s"] = round(time.time() - t0, 1)
        log(f"{part} finished in {info[f'elapsed_{part}_s']} s")
    failed = [k_ for k_, v in checks.items() if v.get("pass") is False]
    checks["failed"] = failed
    checks["all_pass"] = not failed
    write_json(run / "checks.json", checks)
    info["end_jst"] = now_jst().isoformat(timespec="seconds")
    info["elapsed_s"] = round(time.time() - t_all, 1)
    info["status"] = "complete" if not failed else "checks_failed"
    write_json(run / "run_info.json", info)
    log(f"E5 {'complete' if not failed else 'CHECK FAILURE: ' + ', '.join(failed)}; {info['elapsed_s']} s")
    return 0 if not failed else 2


if __name__ == "__main__":
    sys.exit(main())
