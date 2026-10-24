"""E5 independent re-check (docs/E5_protocol.md, section 10).

Written from the plan text (sections 2, 3, 5, 6, 8, 10), experiments/E5_config.json and the run outputs of
results/E5_20261024_0210/ only; the E5 scripts (E5_reliability_cost.py, E5_tables.py) were not read. The E4
machinery, independently checked in E4, is used as a library: experiments/E4_forest_system.py (Instrument,
CellModel, forest_state, pcm_states, a_uniform) and src/pcm_acam (trees.dataset_bundle, trees.quantize,
row_model.Pool.v_at / Curve1D, pcm_model.PCMParams, acam.Scheme). The bound voltage V*, the input map M_m and its
inverse, the compensation-node settings, the DAC quantization, the statistics and the time to budget are own code
(the E4 effective bound CellModel.theta_eff is evaluated next to the own one as a cross-check only).

Items
  1  iris / split 0 / seed 0, P1, 6T2M, lower bounds (L6T), condition unif, 1 year, nodes dac_remap and supply:
     number of bounds with |e| > 0.5 LSB and |e| > 1 LSB versus e51_spot.json (pass: identical counts); every
     bound error is also compared with e51_spot_bounds.csv.
  2  e53_spot.csv (20 bounds; dac_remap and supply x DL full and span grid, and the supply V_SL DAC at +-step/2;
     8 bit, condition all, 1 year): quantized effective bound recomputed from plan section 5, starting from the
     row's instance, g_now_us and a_hat; V* from the instance's divider curve (pass: added error, x_exact and
     x_quantized within 1e-6 LSB). The row inputs are also traced back to the E4 draws.
  3  E5.4, condition unif, 6T2M, L6T: rms without bound failures over the full set (5 datasets x 3 splits x 20
     device seeds) at t0 and the 13 times of e54.times_s; time to the 0.5 LSB budget for P1 none, P1 dac_remap,
     P1 supply and P3 supply (plan section 6) versus e54_budget.csv (pass: relative difference <= 1 % for
     interpolated values, identical status otherwise); curves versus e54_curves.csv.

Usage (from anywhere):  python experiments/E5_independent/run_check.py [--items 1,2,3] [--workers 2]
Output: a new folder results/E5_independent_check_<YYYYMMDD_HHMM, JST>/ (suffix _b, _c, ... if the name is
taken); nothing existing is modified.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True  # never write .pyc files into the project's existing __pycache__ folders

import argparse  # noqa: E402
import csv  # noqa: E402
import hashlib  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import platform  # noqa: E402
import time  # noqa: E402
from datetime import datetime, timedelta, timezone  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "results" / "E5_20261024_0210"
CFG4_PATH = ROOT / "experiments" / "E4_config.json"
CFG5_PATH = ROOT / "experiments" / "E5_config.json"
JST = timezone(timedelta(hours=9))
LSB = 15.0                 # 1 LSB = 1/15 of the normalized input
YEAR = 31536000.0          # 1 year in s (refresh rate = 1 year / T_B)
NODES = ("none", "dac_remap", "supply", "ideal")


def _load_e4():
    spec = importlib.util.spec_from_file_location("E4_forest_system", ROOT / "experiments" / "E4_forest_system.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)          # inserts src/ into sys.path
    return mod


E4 = _load_e4()
from pcm_acam.acam import Scheme  # noqa: E402
from pcm_acam.pcm_model import PCMParams  # noqa: E402
from pcm_acam.trees import dataset_bundle, quantize  # noqa: E402

_INST = None


# --------------------------------------------------------------------------------------------- small helpers
def jst_now() -> datetime:
    return datetime.now(JST)


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def new_out_dir() -> Path:
    base = ROOT / "results" / f"E5_independent_check_{jst_now().strftime('%Y%m%d_%H%M')}"
    d, i = base, 0
    while d.exists():
        i += 1
        d = Path(f"{base}_{chr(ord('a') + i)}")
    d.mkdir(parents=True, exist_ok=False)
    return d


def write_csv(path: Path, rows: list) -> None:
    keys: list = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def read_csv(path: Path) -> list:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_json(path: Path, obj) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.write("\n")


def instrument():
    global _INST
    if _INST is None:
        _INST = E4.Instrument.load(ROOT / json.loads(CFG5_PATH.read_text())["reuse"]["instrument"])
    return _INST


def a_unif(t_s: float, t0: float = 20.0, nu: float = 0.049) -> float:
    """Uniform drift factor a(t) = ((t + t0) / t0)^-nu (plan section 2, condition unif)."""
    return float(((t_s + t0) / t0) ** (-nu))


def fmt_t(t: float) -> str:
    for lim, div, unit in ((60, 1, "s"), (3600, 60, "min"), (86400, 3600, "h"), (np.inf, 86400, "d")):
        if t < lim:
            return f"{t / div:.4g} {unit}"
    return f"{t:.4g} s"


# --------------------------------------------------------------------------------------------- own bound model
def node_setting(node: str, a_hat: float, d_v: float):
    """(PCM conductance divisor, V_SL - V_sp,nom, input-map factor m) of a compensation node (plan section 2)."""
    if node in ("none", "t0"):           # programming-time setting used at the later time
        return 1.0, d_v, 1.0
    if node == "dac_remap":              # input map M_a, supplies nominal
        return 1.0, d_v, a_hat
    if node == "supply":                 # V_SL = V_sp,nom + D / a_hat, map M0
        return 1.0, d_v / a_hat, 1.0
    if node == "ideal":                  # G / a_hat inside the bound, V_SL nominal, map M0
        return a_hat, d_v, 1.0
    raise ValueError(node)


class Bounds:
    """Input-referred bound of one design and cell set.

    V*: data-line voltage at which the instance's divider (pool of the branch type of the bound) conducts
    g (V_SL - V_sp,instance), V_SL = V_sp,nom(branch type) + supply offset. Input map M_m(x) = I_0^-1(m G(x) D),
    I_0 = nominal curve of the cell set's lower branch type, G(x) = g_lo + x (g_hi - g_lo) uS.
    """

    def __init__(self, inst, cfg4: dict, design: str, cell_set: str):
        cs = cfg4["cell_sets"][cell_set]
        c = cfg4["circuit"]
        self.types = {0: cs["lower"], 1: cs["upper"]}
        self.pool = {k: inst.pools[(design, t)] for k, t in self.types.items()}
        self.v_sp_nom = {k: float(self.pool[k].v_sp[0]) for k in (0, 1)}
        self.curve = self.pool[0].nominal
        self.d_v, self.g_lo, self.g_hi = float(c["d_v"]), float(c["g_lo_us"]), float(c["g_hi_us"])

    def v_star(self, kind, idx, g_us, d_sl):
        kind = np.asarray(kind)
        idx = np.asarray(idx)
        g_us = np.broadcast_to(np.asarray(g_us, float), kind.shape)
        d_sl = np.broadcast_to(np.asarray(d_sl, float), kind.shape)
        v = np.full(kind.shape, np.nan)
        for k in (0, 1):
            s = kind == k
            if s.any():
                pool = self.pool[k]
                cur = g_us[s] * 1e-6 * (self.v_sp_nom[k] + d_sl[s] - pool.v_sp[idx[s]])
                v[s] = pool.v_at(idx[s], cur)
        return v

    def v_star_brent(self, kind, idx, g_us, d_sl):
        """Second inversion, for the sensitivity check of items 1 and 2: Brent root of
        ln I_k(V) = ln(g (V_SL - V_sp,k)) on the stored instance curve (Pool.ln_i: nominal fine curve plus the
        Catmull-Rom deviation), xtol 1e-15 V; -inf / +inf below / above the stored window (as Pool.v_at)."""
        from scipy.optimize import brentq
        kind, idx = np.atleast_1d(kind), np.atleast_1d(idx)
        g_us = np.broadcast_to(np.asarray(g_us, float), kind.shape)
        d_sl = np.broadcast_to(np.asarray(d_sl, float), kind.shape)
        out = np.empty(kind.shape)
        for j in range(kind.size):
            k, i = int(kind[j]), int(idx[j])
            pool = self.pool[k]
            cur = g_us[j] * 1e-6 * (self.v_sp_nom[k] + d_sl[j] - pool.v_sp[i])
            y = math.log(max(cur, 1e-300))

            def f(v, pool=pool, i=i, y=y):
                return float(pool.ln_i(np.array([i]), np.array([v]))[0]) - y

            lo, hi = float(pool.vg_c[0]), float(pool.vg_c[-1])
            if f(lo) > 0:
                out[j] = -np.inf
            elif f(hi) < 0:
                out[j] = np.inf
            else:
                out[j] = brentq(f, lo, hi, xtol=1e-15, rtol=1e-15, maxiter=500)
        return out

    def m_of(self, x, m: float):
        """M_m(x) (data-line voltage of input x)."""
        cur = (self.g_lo + np.asarray(x, float) * (self.g_hi - self.g_lo)) * 1e-6 * m * self.d_v
        return np.interp(np.log(cur), self.curve.ln_i, self.curve.vg)

    def m_inv(self, v, m: float):
        """x = M_m^-1(V) = (I_0(V) / (m D) - g_lo) / (g_hi - g_lo)."""
        i0 = np.exp(np.interp(np.asarray(v, float), self.curve.vg, self.curve.ln_i))
        return (i0 / (m * self.d_v) * 1e6 - self.g_lo) / (self.g_hi - self.g_lo)

    def x_eff(self, kind, idx, g_us, d_sl, m: float, x_cap: float = 10.0):
        """Effective bound; outside the stored divider window x = +-10 (E4 convention for bound failures)."""
        v = self.v_star(kind, idx, g_us, d_sl)
        fin = np.isfinite(v)
        x = self.m_inv(np.where(fin, v, 0.0), m)
        x = np.where(fin, x, np.where(v > 0, x_cap, -x_cap))
        return np.clip(x, -x_cap, x_cap), v


def dl_switch(v_star, kind, offset: float, step: float):
    """Data-line DAC (plan section 5): read voltages rounded to the nearest point of offset + n step (all integers
    n). Lower bounds switch at V_g - step/2 (V_g the lowest grid point >= V*), upper bounds at V_g' + step/2 (V_g'
    the highest grid point <= V*). Returns the switching voltage and (V* - offset) / step."""
    n = (np.asarray(v_star, float) - offset) / step
    lower = np.asarray(kind) == 0
    v_g = np.where(lower, offset + np.ceil(n) * step, offset + np.floor(n) * step)
    return np.where(lower, v_g - 0.5 * step, v_g + 0.5 * step), n


def forest(cfg4: dict, ds: str, split: int, seed: int):
    names = list(cfg4["datasets"])
    b = dataset_bundle(ds, split)
    rows = b["rows"]
    theta = quantize(rows.cons_thr, cfg4["bits"])
    kind = rows.cons_kind
    draws, idx = E4.forest_state(cfg4, names.index(ds), split, seed, theta.size, cfg4["pools"]["n_instances"])
    return theta, kind, draws, idx


# --------------------------------------------------------------------------------------------- item 1
def item1(cfg4: dict, cfg5: dict, out: Path, log) -> dict:
    sp = cfg5["e51"]["spot"]
    inst = instrument()
    theta, kind, _, idx = forest(cfg4, sp["dataset"], sp["split"], sp["seed"])
    cm = E4.CellModel(inst, cfg4, sp["design"], sp["cell_set"])
    bd = Bounds(inst, cfg4, sp["design"], sp["cell_set"])
    g_t, clipped = cm.targets(theta, kind, idx)
    t_s = float(cfg5["e54"]["times_s"][sp["time"]])
    a = a_unif(t_s)
    lib_a = E4.a_uniform(t_s, cfg4["pcm_model"]["t0"])
    g_now = a * g_t                                  # condition unif: no PCM noise, a_hat = a(t)
    lo = kind == 0
    assert bd.types[0] == sp["type"]
    e, e_brent, lib_diff = {}, {}, {}
    for node in sp["nodes"]:
        gdiv, d_sl, m = node_setting(node, a, bd.d_v)
        x, _ = bd.x_eff(kind, idx, g_now / gdiv, d_sl, m)
        e[node] = LSB * (x - theta)
        x_lib = cm.theta_eff(g_now, kind, idx, node, a)
        lib_diff[node] = float(np.max(np.abs(x - x_lib)) * LSB)
        vb = bd.v_star_brent(kind, idx, g_now / gdiv, d_sl)
        assert np.all(np.isfinite(vb))
        e_brent[node] = LSB * (bd.m_inv(vb, m) - theta)
    run = json.loads((RUN / "e51_spot.json").read_text())
    counts, counts_brent, margins = {}, {}, {}
    for node in sp["nodes"]:
        ae = np.abs(e[node][lo])
        counts[node] = {"abs_gt_0.5": int(np.sum(ae > 0.5)), "abs_gt_1": int(np.sum(ae > 1.0))}
        ab = np.abs(e_brent[node][lo])
        counts_brent[node] = {"abs_gt_0.5": int(np.sum(ab > 0.5)), "abs_gt_1": int(np.sum(ab > 1.0))}
        margins[node] = {"min_abs_distance_to_0.5_lsb": float(np.min(np.abs(ae - 0.5))),
                         "min_abs_distance_to_1_lsb": float(np.min(np.abs(ae - 1.0)))}
    same = (int(lo.sum()) == run["n_bounds_of_type"]) and all(counts[nd] == run["counts"][nd] for nd in sp["nodes"])
    brent_diff = {nd: float(np.max(np.abs(e_brent[nd] - e[nd]))) for nd in sp["nodes"]}
    # per-bound comparison with e51_spot_bounds.csv (all bounds of the forest, both kinds)
    rb = read_csv(RUN / "e51_spot_bounds.csv")
    per, worst, meta_ok = [], {nd: 0.0 for nd in sp["nodes"]}, True
    for r in rb:
        k = int(r["k"])
        ok = (int(r["kind"]) == int(kind[k]) and int(r["instance"]) == int(idx[k]) and float(r["theta"]) == float(theta[k]))
        meta_ok &= ok
        row = {"k": k, "kind": int(kind[k]), "instance": int(idx[k]), "theta": float(theta[k]), "meta_match": ok}
        for nd in sp["nodes"]:
            mine, theirs = float(e[nd][k]), float(r[f"e_{nd}_lsb"])
            worst[nd] = max(worst[nd], abs(mine - theirs))
            row[f"e_{nd}_lsb_own"], row[f"e_{nd}_lsb_run"], row[f"diff_{nd}_lsb"] = mine, theirs, mine - theirs
        per.append(row)
    write_csv(out / "item1_bounds.csv", per)
    res = {"case": sp, "t_s": t_s, "a_t": a, "a_t_minus_E4_a_uniform": a - lib_a,
           "n_bounds_of_type_own": int(lo.sum()), "n_bounds_of_type_run": run["n_bounds_of_type"],
           "counts_own": counts, "counts_run": run["counts"], "threshold_margins": margins,
           "clipped_lower_bounds": int(np.sum(clipped[lo])),
           "own_vs_E4_CellModel_theta_eff_max_abs_lsb": lib_diff,
           "brent_inversion": {"counts": counts_brent, "max_abs_diff_to_bisection_lsb": brent_diff,
                               "counts_identical_to_run": all(counts_brent[nd] == run["counts"][nd] for nd in sp["nodes"])},
           "per_bound_vs_e51_spot_bounds": {"rows": len(rb), "rows_expected": int(theta.size),
                                            "kind_instance_theta_identical": bool(meta_ok),
                                            "max_abs_diff_lsb": worst},
           "pass": bool(same)}
    write_json(out / "item1_counts.json", res)
    log(f"item 1: own {counts} vs run {run['counts']} (n {int(lo.sum())} / {run['n_bounds_of_type']}); "
        f"per-bound max |diff| {worst}; Brent inversion counts {counts_brent}, max |de| {brent_diff} "
        f"-> {'PASS' if same else 'FAIL'}")
    return res


# --------------------------------------------------------------------------------------------- item 2
def item2(cfg4: dict, cfg5: dict, out: Path, log) -> dict:
    sp = cfg5["e53"]["spot"]
    e53 = cfg5["e53"]
    inst = instrument()
    bd = Bounds(inst, cfg4, sp["design"], sp["cell_set"])
    cm = E4.CellModel(inst, cfg4, sp["design"], sp["cell_set"])
    nlev = 2 ** int(sp["bits"]) - 1
    fs = float(e53["full_scale_v"])
    rows = read_csv(RUN / "e53_spot.csv")

    # provenance: the row inputs against the E4 draws of the same forest (condition all, 1 year)
    theta, kind, draws, idx = forest(cfg4, sp["dataset"], sp["split"], sp["seed"])
    g_t, _ = cm.targets(theta, kind, idx)
    p = PCMParams.from_dict(cfg4["pcm_model"])
    rep = Scheme(name="replica", kind="absolute", devices=1, g_lo=cfg4["circuit"]["g_lo_us"],
                 g_hi=cfg4["circuit"]["g_hi_us"])
    labels = list(cfg4["times_inference_s"])
    t1y = float(cfg4["times_inference_s"]["1y"])
    g_now_all, a_hat_all = E4.pcm_states(g_t, draws, sp["condition"], labels.index("1y"), t1y, p, rep)
    x_e4 = {nd: cm.theta_eff(g_now_all, kind, idx, nd, a_hat_all) for nd in ("dac_remap", "supply")}

    out_rows = []
    worst = {"added_error_lsb": 0.0, "x_exact_lsb": 0.0, "x_quantized_lsb": 0.0}
    worst_v = {"v_bound_v": 0.0, "v_switch_quantized_v": 0.0, "grid_offset_v": 0.0, "grid_step_v": 0.0,
               "d_eff_v": 0.0, "map_factor": 0.0}
    prov = {"kind_instance_theta_identical": True, "g_now_max_rel_diff": 0.0, "a_hat_max_rel_diff": 0.0,
            "x_exact_vs_E4_theta_eff_max_abs_lsb": 0.0}
    worst_b = {"added_error_lsb": 0.0, "x_exact_lsb": 0.0, "x_quantized_lsb": 0.0}
    min_tie = np.inf
    for r in rows:
        node, qty, variant, _bits = r["config"].split("/")
        assert _bits == f"{sp['bits']} bit"
        k, kd, ii = int(r["k"]), int(r["kind"]), int(r["instance"])
        g, a_hat = float(r["g_now_us"]), float(r["a_hat"])
        _, d_sl, m = node_setting(node, a_hat, bd.d_v)
        args_ = (np.array([kd]), np.array([ii]), np.array([g]))
        v_star = float(bd.v_star(*args_, d_sl)[0])
        v_star_b = float(bd.v_star_brent(*args_, d_sl)[0])          # second inversion (sensitivity)
        x_exact = float(bd.m_inv(v_star, m))
        x_exact_b = float(bd.m_inv(v_star_b, m))
        rec = {"config": r["config"], "k": k, "kind": kd, "instance": ii, "map_factor": m}
        if qty == "DL":
            if variant == "full":
                off, step = 0.0, fs / nlev
            elif variant == "span":
                v0, v1 = bd.m_of(np.array([0.0, 1.0]), m)
                off, step = float(v0), float(v1 - v0) / nlev
            else:
                raise ValueError(variant)
            v_sw, n = dl_switch(np.array([v_star]), np.array([kd]), off, step)
            v_sw = float(v_sw[0])
            v_sw_b = float(dl_switch(np.array([v_star_b]), np.array([kd]), off, step)[0][0])
            frac = float(n[0] - np.floor(n[0]))
            min_tie = min(min_tie, frac, 1.0 - frac)
            rec.update({"grid_offset_v": off, "grid_step_v": step, "d_eff_v": None, "v_star_over_step_frac": frac})
        elif qty == "SL":
            step = fs / nlev
            sign = {"+half": 1.0, "-half": -1.0}[variant]
            d_q = bd.d_v / a_hat + sign * 0.5 * step          # worst case D_eff = D / a_hat +- step / 2
            v_sw = float(bd.v_star(*args_, d_q)[0])
            v_sw_b = float(bd.v_star_brent(*args_, d_q)[0])
            rec.update({"grid_offset_v": None, "grid_step_v": step, "d_eff_v": d_q})
        else:
            raise ValueError(qty)
        x_q = float(bd.m_inv(v_sw, m))
        added = LSB * (x_q - x_exact)
        d_add = added - float(r["added_error_lsb"])
        d_xe = LSB * (x_exact - float(r["x_exact"]))
        d_xq = LSB * (x_q - float(r["x_quantized"]))
        worst["added_error_lsb"] = max(worst["added_error_lsb"], abs(d_add))
        worst["x_exact_lsb"] = max(worst["x_exact_lsb"], abs(d_xe))
        worst["x_quantized_lsb"] = max(worst["x_quantized_lsb"], abs(d_xq))
        x_q_b = float(bd.m_inv(v_sw_b, m))
        worst_b["added_error_lsb"] = max(worst_b["added_error_lsb"],
                                         abs(LSB * (x_q_b - x_exact_b) - float(r["added_error_lsb"])))
        worst_b["x_exact_lsb"] = max(worst_b["x_exact_lsb"], abs(LSB * (x_exact_b - float(r["x_exact"]))))
        worst_b["x_quantized_lsb"] = max(worst_b["x_quantized_lsb"], abs(LSB * (x_q_b - float(r["x_quantized"]))))
        for key, mine in (("v_bound_v", v_star), ("v_switch_quantized_v", v_sw), ("grid_offset_v", rec["grid_offset_v"]),
                          ("grid_step_v", rec["grid_step_v"]), ("d_eff_v", rec["d_eff_v"]), ("map_factor", m)):
            theirs = r[key]
            if mine is None or theirs == "":
                assert (mine is None) == (theirs == ""), (key, r["config"])
                continue
            worst_v[key] = max(worst_v[key], abs(mine - float(theirs)))
        # provenance of the inputs
        ok = (kd == int(kind[k]) and ii == int(idx[k]) and float(r["theta"]) == float(theta[k]))
        prov["kind_instance_theta_identical"] &= ok
        prov["g_now_max_rel_diff"] = max(prov["g_now_max_rel_diff"], abs(g / g_now_all[k] - 1.0))
        prov["a_hat_max_rel_diff"] = max(prov["a_hat_max_rel_diff"], abs(a_hat / a_hat_all - 1.0))
        prov["x_exact_vs_E4_theta_eff_max_abs_lsb"] = max(prov["x_exact_vs_E4_theta_eff_max_abs_lsb"],
                                                          abs(LSB * (float(r["x_exact"]) - x_e4[node][k])))
        rec.update({"v_bound_v_own": v_star, "v_bound_v_run": float(r["v_bound_v"]),
                    "v_switch_v_own": v_sw, "v_switch_v_run": float(r["v_switch_quantized_v"]),
                    "x_exact_own": x_exact, "x_exact_run": float(r["x_exact"]),
                    "x_quantized_own": x_q, "x_quantized_run": float(r["x_quantized"]),
                    "added_error_lsb_own": added, "added_error_lsb_run": float(r["added_error_lsb"]),
                    "diff_added_lsb": d_add, "diff_x_exact_lsb": d_xe, "diff_x_quantized_lsb": d_xq})
        out_rows.append(rec)
    write_csv(out / "item2_rows.csv", out_rows)
    configs = sorted({r["config"] for r in rows})
    sel = selection_check(cfg4, sp, bd, theta, kind, idx, g_now_all, a_hat_all, nlev, fs,
                          sorted({int(r["k"]) for r in rows}))
    passed = bool(len(rows) == 20 * len(configs) and max(worst.values()) <= 1e-6)
    res = {"rows": len(rows), "configs": configs, "bounds": len({int(r['k']) for r in rows}),
           "max_abs_diff_lsb": worst, "max_abs_diff_other_columns": worst_v,
           "brent_inversion_max_abs_diff_to_run_lsb": worst_b,
           "min_distance_of_V*_to_a_DL_grid_point_in_steps": float(min_tie),
           "provenance_vs_E4_draws": prov, "selection_of_the_20_bounds": sel,
           "pass": passed}
    write_json(out / "item2_summary.json", res)
    log(f"item 2: {len(rows)} rows ({len(configs)} configurations); max |diff| added {worst['added_error_lsb']:.3e}, "
        f"x_exact {worst['x_exact_lsb']:.3e}, x_quantized {worst['x_quantized_lsb']:.3e} LSB (Brent inversion: "
        f"{max(worst_b.values()):.3e} LSB); provenance {prov}; "
        f"selection {sel.get('matching_variant')} -> {'PASS' if passed else 'FAIL'}")
    return res


def selection_check(cfg4, sp, bd, theta, kind, idx, g_now, a_hat, nlev, fs, run_ks) -> dict:
    """Informational: plan section 8 draws the 20 bounds with default_rng([base_seed, 62]) without replacement
    from the bounds that are not bound failures under the 5 settings; the exact call is not fixed by the plan, so
    two natural forms are tried."""
    fail = np.zeros(theta.size, dtype=bool)
    for node in ("dac_remap", "supply"):
        _, d_sl, m = node_setting(node, a_hat, bd.d_v)
        x, v = bd.x_eff(kind, idx, g_now, d_sl, m)
        fail |= np.abs(LSB * (x - theta)) > 15.0
        fin = np.isfinite(v)
        grids = [(0.0, fs / nlev)]
        v0, v1 = bd.m_of(np.array([0.0, 1.0]), m)
        grids.append((float(v0), float(v1 - v0) / nlev))
        for off, step in grids:
            v_sw, _ = dl_switch(np.where(fin, v, 0.0), kind, off, step)
            xq = np.where(fin, bd.m_inv(v_sw, m), x)
            fail |= np.abs(LSB * (xq - theta)) > 15.0
        if node == "supply":
            for sign in (1.0, -1.0):
                xq, _ = bd.x_eff(kind, idx, g_now, bd.d_v / a_hat + sign * 0.5 * fs / nlev, 1.0)
                fail |= np.abs(LSB * (xq - theta)) > 15.0
    cand = np.where(~fail)[0]
    seq = [cfg4["base_seed"], int(sp["rng_stream"])]
    variants = {
        "sort(rng.choice(candidates, 20, replace=False))":
            np.sort(np.random.default_rng(seq).choice(cand, int(sp["n_bounds"]), replace=False)),
        "sort(rng.permutation(candidates)[:20])":
            np.sort(np.random.default_rng(seq).permutation(cand)[:int(sp["n_bounds"])]),
    }
    match = [name for name, ks in variants.items() if list(map(int, ks)) == list(run_ks)]
    return {"bounds": int(theta.size), "bound_failures": int(fail.sum()), "candidates": int(cand.size),
            "run_ks": list(run_ks), "matching_variant": match[0] if match else None,
            "note": "informational (the plan fixes the stream and the candidate set, not the exact numpy call)"}


# --------------------------------------------------------------------------------------------- item 3
def item3_job(args):
    cfg4, ds, split, n_seed, times = args
    inst = instrument()
    t_start = time.time()
    b = dataset_bundle(ds, split)
    rows = b["rows"]
    theta = quantize(rows.cons_thr, cfg4["bits"])
    kind = rows.cons_kind
    k_all = theta.size
    lo = np.where(kind == 0)[0]
    names = list(cfg4["datasets"])
    th, ix = [], []
    for seed in range(n_seed):
        _, idx = E4.forest_state(cfg4, names.index(ds), split, seed, k_all, cfg4["pools"]["n_instances"])
        th.append(theta[lo])
        ix.append(idx[lo])
    th = np.concatenate(th)
    ix = np.concatenate(ix)
    kd = np.zeros(th.size, dtype=int)
    acc, lib_diff = {}, 0.0
    pts = [("t0", 0.0, "t0")] + [(lab, float(ts), nd) for lab, ts in times.items() for nd in NODES]
    for design in ("P1", "P3"):
        cm = E4.CellModel(inst, cfg4, design, "6T2M")
        bd = Bounds(inst, cfg4, design, "6T2M")
        assert bd.types[0] == "L6T"
        g_t, _ = cm.targets(th, kd, ix)
        for lab, ts, nd in pts:
            a = 1.0 if ts == 0.0 else a_unif(ts)
            g_now = a * g_t
            gdiv, d_sl, m = node_setting(nd, a, bd.d_v)
            x, _ = bd.x_eff(kd, ix, g_now / gdiv, d_sl, m)
            e = LSB * (x - th)
            x_lib = cm.theta_eff(g_now, kd, ix, "t0" if nd == "none" else nd, a)
            lib_diff = max(lib_diff, float(np.max(np.abs(x - x_lib))) * LSB)
            ae = np.abs(e)
            ext = ae > 15.0
            acc[(design, nd, lab)] = [float(np.sum(e[~ext] ** 2)), int((~ext).sum()), int(ext.sum()),
                                      int(np.sum(ae > 0.5)), int(np.sum(ae > 1.0)), int(np.sum(ae > 2.0)), int(th.size)]
    return ds, split, int(th.size), acc, lib_diff, time.time() - t_start


def time_to_budget(v_t0: float, ts: list, vals: list, budget: float):
    """Plan section 6: first evaluated time whose value exceeds the budget, interpolated linearly in log t from the
    previous time; special statuses for t0, 10 s and beyond 10 y."""
    if v_t0 > budget:
        return "exceeded at t0", None
    for i, (t, v) in enumerate(zip(ts, vals)):
        if v > budget:
            if i == 0:
                return "< 10 s", None
            t1, v1 = ts[i - 1], vals[i - 1]
            f = (budget - v1) / (v - v1)
            return "interpolated", float(math.exp(math.log(t1) + f * (math.log(t) - math.log(t1))))
    return "> 10 y", None


def item3(cfg4: dict, cfg5: dict, out: Path, log, workers: int) -> dict:
    times = cfg5["e54"]["times_s"]
    labs = list(times)
    ts = [float(times[t]) for t in labs]
    splits = cfg5["full_set"]["split_seeds"]
    n_seed = int(cfg5["full_set"]["device_seeds"])
    datasets = list(cfg4["datasets"])
    jobs = [(cfg4, ds, sp, n_seed, times) for ds in datasets for sp in splits]
    order = {"synthetic": 0, "digits": 1, "breast_cancer": 2, "wine": 3, "iris": 4}
    jobs.sort(key=lambda j: order.get(j[1], 9))
    t0 = time.time()
    results = []
    if workers > 1:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(workers) as pool:
            for r in pool.imap_unordered(item3_job, jobs):
                results.append(r)
                log(f"  item 3: {r[0]} split {r[1]}: {r[2]} lower bounds (20 seeds), {r[5]:.0f} s "
                    f"({time.time() - t0:.0f} s since start)")
    else:
        for j in jobs:
            r = item3_job(j)
            results.append(r)
            log(f"  item 3: {r[0]} split {r[1]}: {r[2]} lower bounds, {r[5]:.0f} s")
    tot: dict = {}
    lib_diff = 0.0
    for r in results:
        lib_diff = max(lib_diff, r[4])
        for key, v in r[3].items():
            a = tot.setdefault(key, [0.0, 0, 0, 0, 0, 0, 0])
            for i in range(7):
                a[i] += v[i]

    def stats(key):
        ss, n_ok, n_ext, g05, g1, g2, n = tot[key]
        return {"n": n, "n_extreme": n_ext, "rms_noext_lsb": math.sqrt(ss / n_ok),
                "p_exc_0.5": g05 / n, "p_exc_1": g1 / n, "p_exc_2": g2 / n}

    # curves versus e54_curves.csv
    run_c = {(r["design"], r["node"], r["time"]): r for r in read_csv(RUN / "e54_curves.csv")
             if r["condition"] == "unif" and r["cell_set"] == "6T2M" and r["type"] == "L6T"}
    curve_rows = []
    worst_rel_all, worst_rel_req = 0.0, 0.0
    counts_ok = True
    required = [("P1", "none"), ("P1", "dac_remap"), ("P1", "supply"), ("P3", "supply")]
    for design in ("P1", "P3"):
        for nd, lab in [("t0", "t0")] + [(nd, lab) for nd in NODES for lab in labs]:
            s = stats((design, nd, lab))
            rr = run_c[(design, nd, lab)]
            rel = s["rms_noext_lsb"] / float(rr["rms_noext_lsb"]) - 1.0
            same_n = (s["n"] == int(rr["n"]) and s["n_extreme"] == int(rr["n_extreme"]))
            counts_ok &= same_n
            worst_rel_all = max(worst_rel_all, abs(rel))
            if (design, nd) in required or nd == "t0":
                worst_rel_req = max(worst_rel_req, abs(rel))
            row = {"design": design, "node": nd, "time": lab, "t_s": 0.0 if lab == "t0" else float(times[lab]),
                   "n_own": s["n"], "n_run": int(rr["n"]), "n_extreme_own": s["n_extreme"],
                   "n_extreme_run": int(rr["n_extreme"]), "rms_noext_lsb_own": s["rms_noext_lsb"],
                   "rms_noext_lsb_run": float(rr["rms_noext_lsb"]), "rms_rel_diff": rel}
            for q in ("p_exc_0.5", "p_exc_1", "p_exc_2"):
                row[f"{q}_own"], row[f"{q}_run"] = s[q], float(rr[q])
                row[f"{q}_abs_diff"] = s[q] - float(rr[q])
            curve_rows.append(row)
    write_csv(out / "item3_curves.csv", curve_rows)
    worst_pexc = max(abs(r[f"{q}_abs_diff"]) for r in curve_rows for q in ("p_exc_0.5", "p_exc_1", "p_exc_2"))

    # time to budget versus e54_budget.csv
    run_b = {(r["design"], r["node"], r["metric"], float(r["budget"])): r for r in read_csv(RUN / "e54_budget.csv")
             if r["condition"] == "unif" and r["cell_set"] == "6T2M" and r["type"] == "L6T"}
    budget_rows = []
    req_res = []
    all_ok = True
    for design in ("P1", "P3"):
        for nd in NODES:
            for metric, col, budgets in (("rms_noext", "rms_noext_lsb", (0.5, 1.0, 1.5)), ("p_exc_1", "p_exc_1", (0.05, 0.2))):
                v_t0 = stats((design, "t0", "t0"))[col]
                vals = [stats((design, nd, lab))[col] for lab in labs]
                for bgt in budgets:
                    st, tb = time_to_budget(v_t0, ts, vals, bgt)
                    rr = run_b[(design, nd, metric, bgt)]
                    tb_run = float(rr["t_budget_s"]) if rr["t_budget_s"] != "" else None
                    rel = (tb / tb_run - 1.0) if (tb is not None and tb_run is not None) else None
                    ok = (st == rr["status"]) and (rel is None or abs(rel) <= 0.01) and ((tb is None) == (tb_run is None))
                    is_req = (metric == "rms_noext" and bgt == 0.5 and (design, nd) in required)
                    if not ok:
                        all_ok = False
                    row = {"design": design, "node": nd, "metric": metric, "budget": bgt, "required": is_req,
                           "value_t0_own": v_t0, "value_t0_run": float(rr["value_t0"]),
                           "value_1y_own": vals[labs.index("1y")], "value_1y_run": float(rr["value_1y"]),
                           "status_own": st, "status_run": rr["status"], "t_budget_s_own": tb,
                           "t_budget_s_run": tb_run, "t_budget_rel_diff": rel,
                           "refresh_per_year_own": (YEAR / tb if tb else None),
                           "refresh_per_year_run": (float(rr["refresh_per_year"]) if rr["refresh_per_year"] else None),
                           "agree": ok}
                    budget_rows.append(row)
                    if is_req:
                        req_res.append(row)
    write_csv(out / "item3_budget.csv", budget_rows)
    req_pass = all(r["agree"] for r in req_res) and len(req_res) == 4
    res = {"set": {"datasets": datasets, "splits": splits, "device_seeds": n_seed},
           "times_s": times, "required_cases": [
               {k: r[k] for k in ("design", "node", "status_own", "status_run", "t_budget_s_own", "t_budget_s_run",
                                  "t_budget_rel_diff")} for r in req_res],
           "curves_max_rel_diff_rms_required_cases_and_t0": worst_rel_req,
           "curves_max_rel_diff_rms_all_unif_6T2M_L6T": worst_rel_all,
           "curves_max_abs_diff_p_exc": worst_pexc,
           "curves_n_and_n_extreme_identical": bool(counts_ok),
           "all_unif_6T2M_L6T_budget_rows_agree": bool(all_ok), "budget_rows_compared": len(budget_rows),
           "own_vs_E4_CellModel_theta_eff_max_abs_lsb": lib_diff,
           "elapsed_s": round(time.time() - t0, 1),
           "pass": bool(req_pass)}
    write_json(out / "item3_summary.json", res)
    for r in req_res:
        own = fmt_t(r["t_budget_s_own"]) if r["t_budget_s_own"] else "-"
        theirs = fmt_t(r["t_budget_s_run"]) if r["t_budget_s_run"] else "-"
        log(f"item 3: {r['design']} {r['node']}: own {r['status_own']} {r['t_budget_s_own']} s ({own}) vs run "
            f"{r['status_run']} {r['t_budget_s_run']} s ({theirs}); rel diff {r['t_budget_rel_diff']}")
    log(f"item 3: curves max rel diff (required + t0) {worst_rel_req:.3e}, all {worst_rel_all:.3e}; p_exc max abs "
        f"diff {worst_pexc:.3e}; n identical {counts_ok}; all {len(budget_rows)} budget rows agree {all_ok}; "
        f"own vs E4 theta_eff {lib_diff:.3e} LSB -> {'PASS' if req_pass else 'FAIL'}")
    return res


# --------------------------------------------------------------------------------------------- driver
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--items", default="1,2,3")
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    items = [int(s) for s in args.items.split(",") if s.strip()]
    out = new_out_dir()
    log_path = out / "run.log"

    def log(msg: str) -> None:
        line = f"[{jst_now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
        print(line, flush=True)

    cfg4 = json.loads(CFG4_PATH.read_text(encoding="utf-8"))
    cfg5 = json.loads(CFG5_PATH.read_text(encoding="utf-8"))
    run_cfg = json.loads((RUN / "config_used.json").read_text(encoding="utf-8"))
    inst_path = ROOT / cfg5["reuse"]["instrument"]
    inputs = ["experiments/E4_forest_system.py", "experiments/E4_config.json", "experiments/E5_config.json",
              "docs/E5_protocol.md", cfg5["reuse"]["instrument"]] + \
             [f"src/pcm_acam/{m}.py" for m in ("row_model", "acam", "pcm_model", "trees", "io_utils")] + \
             [f"results/E5_20261024_0210/{f}" for f in ("e51_spot.json", "e51_spot_bounds.csv", "e53_spot.csv",
                                                        "e54_budget.csv", "e54_curves.csv", "config_used.json")]
    import sklearn
    info = {"check": "E5 independent re-check (plan section 10)", "start_jst": jst_now().isoformat(timespec="seconds"),
            "items": items, "workers": args.workers, "run_checked": str(RUN.relative_to(ROOT)),
            "environment": {"python": sys.version.split()[0], "numpy": np.__version__, "sklearn": sklearn.__version__,
                            "platform": platform.platform()},
            "sha256_inputs": {f: sha256(ROOT / f) for f in inputs},
            "sha256_script": {str(Path(__file__).resolve().relative_to(ROOT)): sha256(Path(__file__).resolve())},
            "config_used_equals_E5_config": run_cfg == cfg5}
    inst_ok = info["sha256_inputs"][cfg5["reuse"]["instrument"]] == cfg5["reuse"]["instrument_sha256"]
    info["instrument_sha256_matches_plan"] = inst_ok
    write_json(out / "run_info.json", info)
    log(f"E5 independent check -> {out.relative_to(ROOT)}; instrument sha256 ok {inst_ok}; "
        f"config_used == E5_config {info['config_used_equals_E5_config']}")
    if not inst_ok:
        log("instrument SHA-256 differs from the plan; stop")
        return 2
    instrument()
    compare = {}
    t0 = time.time()
    if 1 in items:
        compare["item1"] = item1(cfg4, cfg5, out, log)
    if 2 in items:
        compare["item2"] = item2(cfg4, cfg5, out, log)
    if 3 in items:
        compare["item3"] = item3(cfg4, cfg5, out, log, args.workers)
    compare["all_pass"] = all(v["pass"] for k, v in compare.items() if k.startswith("item"))
    write_json(out / "compare.json", compare)
    info["end_jst"] = jst_now().isoformat(timespec="seconds")
    info["elapsed_s"] = round(time.time() - t0, 1)
    info["result"] = {k: v["pass"] for k, v in compare.items() if k.startswith("item")}
    write_json(out / "run_info.json", info)
    log(f"done in {info['elapsed_s']} s: {info['result']} -> all pass {compare['all_pass']}")
    return 0 if compare["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
