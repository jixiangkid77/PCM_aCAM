#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PCM_aCAM E0.1 -- INDEPENDENT recompute of the primary (4-bit) numbers.

Written only from the written specification:
  * /home/claude/PCM_aCAM/docs/E0_1_protocol.md   (frozen protocol, sections 2-6, 9)
  * /home/claude/PCM_aCAM/experiments/E0_1_config.json (frozen configuration)
  * the verifier brief (summary + clarifications)
No project source code, project script or other result folder was read or imported.

Usage
  python3 E0_1_independent_recompute.py
        full run, writes results/E0_1_independent_check_<YYYYMMDD_HHMM JST>/
  python3 E0_1_independent_recompute.py --quick --out-root <dir>
        small smoke run (iris, wine; split 0; 2 device seeds) into <dir>
"""
import argparse
import datetime as _dt
import hashlib
import json
import math
import os
import platform
import sys
import time

import numpy as np
import scipy
import sklearn
from sklearn.datasets import (load_breast_cancer, load_digits, load_iris,
                              load_wine, make_classification)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Delivered copy: the two absolute paths of the run copy (sha256 46158c19...) are made
# relative to the project folder; nothing else differs.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(_ROOT, "experiments", "E0_1_config.json")
RESULTS_ROOT = os.path.join(_ROOT, "results")
JST = _dt.timezone(_dt.timedelta(hours=9), "JST")      # Japan has no DST

# ----------------------------------------------------------------------------
# Constants of the frozen plan (cross-checked against the config at start-up)
# ----------------------------------------------------------------------------
DATASETS = ["iris", "wine", "breast_cancer", "digits", "synthetic"]
SPLITS = [0, 1, 2]
N_DEV = 5
BASE_SEED = 20261012
BITS = 4
T_LABELS = ["t0", "1h", "1d", "30d", "365d"]
T_INF = [0.0, 3600.0, 86400.0, 2592000.0, 31536000.0]

G_MAX = 25.0                       # uS
PROG_C = (0.26348, 1.9650, -1.1731)
T0 = 20.0                          # s
T_READ = 2.5e-7                    # s
N_REPLICA = 64

ABS_SCHEMES = {"V1_abs_full": (0.0, 25.0), "V2_abs_high": (8.22, 25.0)}
RATIO_GSUM = 25.0

PAIRS = [("V1_abs_full", "none"), ("V1_abs_full", "replica"),
         ("V2_abs_high", "none"), ("V2_abs_high", "replica"),
         ("V3_ratio", "none")]
CONDITIONS = [("prog", "t0")] + [("all", lab) for lab in T_LABELS]

SYNTH_STR = ("sklearn make_classification(n_samples=5000, n_features=20, n_informative=10, "
             "n_redundant=4, n_classes=4, n_clusters_per_class=2, class_sep=1.0, "
             "random_state=20261012)")


def load_config():
    with open(CONFIG_PATH, "rb") as fh:
        raw = fh.read()
    cfg = json.loads(raw.decode("utf-8"))
    p = cfg["pcm_model"]
    assert p["g_max"] == G_MAX and p["g_min"] == 0.0
    assert tuple(p["prog_coeff"]) == PROG_C and p["prog_coeff_g_max_reference"] == G_MAX
    assert p["t0"] == T0 and p["t_read"] == T_READ
    assert (p["nu_mean_a"], p["nu_mean_b"], p["nu_mean_min"], p["nu_mean_max"]) == (-0.0155, 0.0244, 0.049, 0.1)
    assert (p["nu_std_a"], p["nu_std_b"], p["nu_std_min"], p["nu_std_max"]) == (-0.0125, -0.0059, 0.008, 0.045)
    assert (p["read_q0"], p["read_exp"], p["read_floor"], p["read_qmax"]) == (0.0088, 0.65, 0.001, 0.2)
    assert (p["prog_noise_scale"], p["read_noise_scale"], p["drift_scale"]) == (1.0, 1.0, 1.0)
    s = cfg["schemes"]
    for name, (lo, hi) in ABS_SCHEMES.items():
        assert (s[name]["g_lo"], s[name]["g_hi"]) == (lo, hi), name
    assert s["V3_ratio"]["g_sum"] == RATIO_GSUM
    assert abs(s["V3_ratio"]["r_lo"] - 1.0 / 3.0) < 1e-15 and abs(s["V3_ratio"]["r_hi"] - 2.0 / 3.0) < 1e-15
    assert cfg["replica_cells"] == N_REPLICA
    assert list(cfg["datasets"].keys()) == DATASETS
    assert cfg["datasets"]["synthetic"] == SYNTH_STR
    assert cfg["split_seeds"] == SPLITS and cfg["device_seeds"] == N_DEV and cfg["base_seed"] == BASE_SEED
    assert cfg["bits_primary"] == BITS
    assert list(cfg["times_inference_s"].keys()) == T_LABELS
    assert [float(v) for v in cfg["times_inference_s"].values()] == T_INF
    return cfg, hashlib.sha256(raw).hexdigest()


# ----------------------------------------------------------------------------
# PCM device model (protocol 2.1)
# ----------------------------------------------------------------------------
def prog_sigma(G):
    g = G / G_MAX
    return PROG_C[0] + PROG_C[1] * g + PROG_C[2] * g * g


def program(G_target, z):
    return np.maximum(G_target + prog_sigma(G_target) * z, 0.0)


def drift_params(G_target):
    g_rel = np.maximum(G_target / G_MAX, 1e-7)
    lg = np.log(g_rel)
    mu = np.clip(-0.0155 * lg + 0.0244, 0.049, 0.1)
    sd = np.clip(-0.0125 * lg - 0.0059, 0.008, 0.045)
    return mu, sd


def drift_exponent(G_target, z):
    mu, sd = drift_params(G_target)
    return np.abs(mu + sd * z)


def drift_and_read(G_prog, nu, z_read, t_inf):
    t = t_inf + T0
    if t > T0:
        G_drift = G_prog * (t / T0) ** (-nu)
    else:
        G_drift = np.array(G_prog, dtype=np.float64, copy=True)
    Q = np.minimum(0.0088 / np.maximum((G_prog / G_MAX) ** 0.65, 1e-3), 0.2)
    sig = Q * math.sqrt(math.log((t + T_READ) / (2.0 * T_READ)))
    return np.maximum(G_drift + np.abs(G_drift) * sig * z_read, 0.0)


def replica_ahat(g_lo, g_hi, z_rep_prog, z_rep_nu, z_rep_read_t, t_inf):
    G_t = np.full(N_REPLICA, (g_lo + g_hi) / 2.0)
    G_p = program(G_t, z_rep_prog)
    nu = drift_exponent(G_t, z_rep_nu)
    G_f = drift_and_read(G_p, nu, z_rep_read_t, t_inf)
    return float(np.mean(G_f) / np.mean(G_p))


def abs_theta(G_eff, a_hat, g_lo, g_hi):
    return (G_eff / a_hat - g_lo) / (g_hi - g_lo)


def ratio_theta(G1, G2):
    s = G1 + G2
    zero = s == 0.0
    r = G1 / np.where(zero, 1.0, s)
    r = np.where(zero, 0.5, r)
    return (r - 1.0 / 3.0) / (1.0 / 3.0)


def quantize(thr, bits):
    lv = float(2 ** bits - 1)
    return np.round(np.clip(thr, 0.0, 1.0) * lv) / lv


def draw_noise(ds_idx, split, dev, K):
    rng = np.random.default_rng([BASE_SEED, ds_idx, split, dev])
    Z = {}
    Z["prog"] = rng.standard_normal((K, 2))
    Z["nu"] = rng.standard_normal((K, 2))
    Z["read"] = rng.standard_normal((5, K, 2))
    Z["rep_prog"] = rng.standard_normal(N_REPLICA)
    Z["rep_nu"] = rng.standard_normal(N_REPLICA)
    Z["rep_read"] = rng.standard_normal((5, N_REPLICA))
    return Z


def zero_noise(K):
    return {"prog": np.zeros((K, 2)), "nu": np.zeros((K, 2)), "read": np.zeros((5, K, 2)),
            "rep_prog": np.zeros(N_REPLICA), "rep_nu": np.zeros(N_REPLICA),
            "rep_read": np.zeros((5, N_REPLICA))}


def all_thresholds(theta_q, Z):
    """List of (scheme, track, condition, t_label, a_hat, theta_eff) in PAIRS x CONDITIONS order."""
    out = []
    for scheme in ("V1_abs_full", "V2_abs_high"):
        g_lo, g_hi = ABS_SCHEMES[scheme]
        G = g_lo + theta_q * (g_hi - g_lo)
        Gp = program(G, Z["prog"][:, 0])
        nu = drift_exponent(G, Z["nu"][:, 0])
        th_prog = abs_theta(Gp, 1.0, g_lo, g_hi)
        Gf = [drift_and_read(Gp, nu, Z["read"][ti, :, 0], T_INF[ti]) for ti in range(5)]
        ah = [replica_ahat(g_lo, g_hi, Z["rep_prog"], Z["rep_nu"], Z["rep_read"][ti], T_INF[ti])
              for ti in range(5)]
        for track in ("none", "replica"):
            out.append((scheme, track, "prog", "t0", 1.0, th_prog))     # prog: a_hat = 1
            for ti, lab in enumerate(T_LABELS):
                a = 1.0 if track == "none" else ah[ti]
                out.append((scheme, track, "all", lab, a, abs_theta(Gf[ti], a, g_lo, g_hi)))
    r = 1.0 / 3.0 + theta_q / 3.0
    G1 = RATIO_GSUM * r
    G2 = RATIO_GSUM * (1.0 - r)
    G1p = program(G1, Z["prog"][:, 0])
    G2p = program(G2, Z["prog"][:, 1])
    nu1 = drift_exponent(G1, Z["nu"][:, 0])
    nu2 = drift_exponent(G2, Z["nu"][:, 1])
    out.append(("V3_ratio", "none", "prog", "t0", 1.0, ratio_theta(G1p, G2p)))
    for ti, lab in enumerate(T_LABELS):
        G1f = drift_and_read(G1p, nu1, Z["read"][ti, :, 0], T_INF[ti])
        G2f = drift_and_read(G2p, nu2, Z["read"][ti, :, 1], T_INF[ti])
        out.append(("V3_ratio", "none", "all", lab, 1.0, ratio_theta(G1f, G2f)))
    return out


# ----------------------------------------------------------------------------
# Data and forest (protocol 3)
# ----------------------------------------------------------------------------
def load_xy(name):
    if name == "synthetic":
        X, y = make_classification(n_samples=5000, n_features=20, n_informative=10, n_redundant=4,
                                   n_classes=4, n_clusters_per_class=2, class_sep=1.0,
                                   random_state=20261012)
        return np.asarray(X, dtype=np.float64), np.asarray(y)
    loader = {"iris": load_iris, "wine": load_wine, "breast_cancer": load_breast_cancer,
              "digits": load_digits}[name]
    d = loader()
    return np.asarray(d.data, dtype=np.float64), np.asarray(d.target)


def split_raw(X, y, split):
    return train_test_split(X, y, test_size=0.3, stratify=y, random_state=split)


def minmax(Xtr, Xte):
    mn = Xtr.min(axis=0)
    span = Xtr.max(axis=0) - mn
    span = np.where(span == 0.0, 1.0, span)
    return (Xtr - mn) / span, np.clip((Xte - mn) / span, 0.0, 1.0)


def fit_forest(Xtr, ytr, split):
    rf = RandomForestClassifier(n_estimators=16, max_depth=6, random_state=split, n_jobs=1)
    rf.fit(Xtr, ytr)
    return rf


def leaf_vector(tree_, node):
    v = np.array(tree_.value[node, 0, :], dtype=np.float64)
    return v / v.sum()


# ----------------------------------------------------------------------------
# Tree -> CAM rows (protocol 9)
# ----------------------------------------------------------------------------
class Cam:
    pass


def build_cam(rf):
    feat, kind, thr = [], [], []          # kind 0 = lower (x > L), 1 = upper (x <= U)
    row_start, row_tree, row_leaf, row_vec, row_raw = [0], [], [], [], []
    tree_row_start = [0]
    n_single_leaf = 0
    for ti, est in enumerate(rf.estimators_):
        tr = est.tree_
        cl, cr, ft, th = tr.children_left, tr.children_right, tr.feature, tr.threshold
        if cl[0] == -1:
            n_single_leaf += 1
        stack = [(0, ())]
        while stack:                       # DFS pre-order, left child first
            node, path = stack.pop()
            if cl[node] == -1:             # leaf -> one row
                lo_order, lo_val, up_order, up_val = [], {}, [], {}
                for f, k, t in path:       # path is root -> leaf
                    if k == 0:
                        if f in lo_val:
                            if t > lo_val[f]:
                                lo_val[f] = t      # tightest lower bound = largest
                        else:
                            lo_val[f] = t
                            lo_order.append(f)
                    else:
                        if f in up_val:
                            if t < up_val[f]:
                                up_val[f] = t      # tightest upper bound = smallest
                        else:
                            up_val[f] = t
                            up_order.append(f)
                for f in lo_order:
                    feat.append(f); kind.append(0); thr.append(lo_val[f])
                for f in up_order:
                    feat.append(f); kind.append(1); thr.append(up_val[f])
                row_start.append(len(feat))
                row_vec.append(leaf_vector(tr, node))
                row_raw.append(np.array(tr.value[node, 0, :], dtype=np.float64))
                row_tree.append(ti)
                row_leaf.append(int(node))
            else:
                f = int(ft[node])
                t = float(th[node])
                stack.append((int(cr[node]), path + ((f, 0, t),)))   # right: x > t
                stack.append((int(cl[node]), path + ((f, 1, t),)))   # left: x <= t (popped first)
        tree_row_start.append(len(row_tree))
    cam = Cam()
    cam.feat = np.array(feat, dtype=np.int64)
    cam.kind = np.array(kind, dtype=np.int8)
    cam.thr = np.array(thr, dtype=np.float64)
    cam.row_start = np.array(row_start, dtype=np.int64)
    cam.row_tree = np.array(row_tree, dtype=np.int64)
    cam.row_leaf = np.array(row_leaf, dtype=np.int64)
    cam.V = np.vstack(row_vec)
    cam.V_raw = np.vstack(row_raw)
    cam.tree_row_start = np.array(tree_row_start, dtype=np.int64)
    cam.n_single_leaf = n_single_leaf
    cam.K = len(feat)
    cam.n_rows = len(row_tree)
    cam.n_trees = len(rf.estimators_)
    return cam


class CamEvaluator:
    def __init__(self, cam, X, y, classes):
        self.cam = cam
        self.n = X.shape[0]
        self.XK = np.ascontiguousarray(X[:, cam.feat])
        self.is_upper = (cam.kind == 1)[None, :]
        rs = cam.row_start
        self.nonempty = rs[:-1] < rs[1:]
        self.starts = rs[:-1][self.nonempty]
        self.all_nonempty = bool(self.nonempty.all())
        self.trs = cam.tree_row_start
        self.y = np.asarray(y)
        self.classes = np.asarray(classes)
        self._sat = np.empty(self.XK.shape, dtype=bool)

    def match(self, theta):
        theta = np.asarray(theta, dtype=np.float64)
        assert theta.shape == (self.cam.K,) and np.all(np.isfinite(theta))
        np.greater(self.XK, theta[None, :], out=self._sat)            # x > theta
        np.not_equal(self._sat, self.is_upper, out=self._sat)        # upper: not(x > theta) == x <= theta
        if self.all_nonempty:
            return np.logical_and.reduceat(self._sat, self.starts, axis=1)
        m = np.ones((self.n, self.cam.n_rows), dtype=bool)          # unconstrained row always matches
        if self.starts.size:
            m[:, self.nonempty] = np.logical_and.reduceat(self._sat, self.starts, axis=1)
        return m

    def evaluate(self, theta, keep_match=False):
        m = self.match(theta)
        mf = m.astype(np.float64)
        counts = np.add.reduceat(m.astype(np.int32), self.trs[:-1], axis=1)   # (n, trees)
        integrity = float(np.mean(counts == 1))
        score = np.zeros((self.n, self.cam.V.shape[1]))
        score_raw = np.zeros_like(score)
        for t in range(self.cam.n_trees):                          # trees in forest order
            a, b = self.trs[t], self.trs[t + 1]
            score += mf[:, a:b] @ self.cam.V[a:b]
            score_raw += mf[:, a:b] @ self.cam.V_raw[a:b]           # diagnostic only
        any_match = counts.sum(axis=1) > 0
        pidx = np.argmax(score, axis=1)                             # lowest index on ties
        correct = any_match & (self.classes[pidx] == self.y)
        ss = np.sort(score, axis=1)
        gap = ss[:, -1] - ss[:, -2]
        # diagnostics: other plausible summation orders / leaf-vector forms
        pidx_one = np.argmax(mf @ self.cam.V, axis=1)              # one matmul over all rows
        pidx_raw = np.argmax(score_raw, axis=1)                     # tree_.value without renormalising
        acc_one = float(np.mean(any_match & (self.classes[pidx_one] == self.y)))
        acc_raw = float(np.mean(any_match & (self.classes[pidx_raw] == self.y)))
        res = {"acc": float(np.mean(correct)), "integrity": integrity,
               "pred": np.where(any_match, self.classes[pidx], -1),
               "n_nomatch": int(np.sum(~any_match)),
               "n_exact_tie": int(np.sum(any_match & (gap == 0.0))),
               "n_near_tie": int(np.sum(any_match & (gap > 0.0) & (gap <= 1e-9))),
               "n_pred_diff_onematmul": int(np.sum(any_match & (pidx_one != pidx))),
               "n_pred_diff_rawvalue": int(np.sum(any_match & (pidx_raw != pidx))),
               "acc_diff_onematmul": acc_one != float(np.mean(correct)),
               "acc_diff_rawvalue": acc_raw != float(np.mean(correct))}
        if keep_match:
            res["match"] = m
            res["counts"] = counts
        return res


# ----------------------------------------------------------------------------
# Self-check helpers
# ----------------------------------------------------------------------------
def traverse_quantized(rf, X, bits):
    """Direct traversal of every tree: go left iff x <= theta_q(node threshold)."""
    n = X.shape[0]
    leaves = np.empty((n, len(rf.estimators_)), dtype=np.int64)
    score = np.zeros((n, len(rf.classes_)))
    for ti, est in enumerate(rf.estimators_):
        tr = est.tree_
        cl, cr, ft = tr.children_left, tr.children_right, tr.feature
        thq = quantize(tr.threshold, bits)
        node = np.zeros(n, dtype=np.int64)
        for _ in range(tr.max_depth + 1):
            internal = cl[node] != -1
            if not internal.any():
                break
            idx = np.nonzero(internal)[0]
            nd = node[idx]
            go_left = X[idx, ft[nd]] <= thq[nd]
            node[idx] = np.where(go_left, cl[nd], cr[nd])
        assert np.all(cl[node] == -1)
        leaves[:, ti] = node
        vec = {int(u): leaf_vector(tr, int(u)) for u in np.unique(node)}
        score += np.vstack([vec[int(u)] for u in node])
    pred = rf.classes_[np.argmax(score, axis=1)]
    return leaves, pred


def hand_check():
    """Check 3: PCM formulas vs. hand values computed independently with math."""
    errs = []
    for G in (0.0, 12.5, 25.0):
        g = G / 25.0
        hand = 0.26348 + 1.9650 * g - 1.1731 * g ** 2
        errs.append(abs(float(prog_sigma(np.array([G]))[0]) - hand))
    for G in (0.025, 5.0, 25.0):
        L = math.log(max(G / 25.0, 1e-7))
        mu_h = min(max(-0.0155 * L + 0.0244, 0.049), 0.1)
        sd_h = min(max(-0.0125 * L - 0.0059, 0.008), 0.045)
        mu, sd = drift_params(np.array([G]))
        errs.append(abs(float(mu[0]) - mu_h))
        errs.append(abs(float(sd[0]) - sd_h))
    # literal hand values (rounded, 1e-6 tolerance) as a typo guard
    lit = {"sigma0": 0.26348, "sigma12.5": 0.952705, "sigma25": 1.05538,
           "mu0.025": 0.1, "sd0.025": 0.045, "mu5": 0.0493463, "sd5": 0.0142180,
           "mu25": 0.049, "sd25": 0.008}
    got = {"sigma0": float(prog_sigma(np.array([0.0]))[0]),
           "sigma12.5": float(prog_sigma(np.array([12.5]))[0]),
           "sigma25": float(prog_sigma(np.array([25.0]))[0])}
    for G, key in ((0.025, "0.025"), (5.0, "5"), (25.0, "25")):
        mu, sd = drift_params(np.array([G]))
        got["mu" + key] = float(mu[0])
        got["sd" + key] = float(sd[0])
    lit_err = max(abs(got[k] - lit[k]) for k in lit)
    return {"max_abs_err_formula": max(errs), "pass_1e-12": max(errs) <= 1e-12,
            "max_abs_err_vs_rounded_literals": lit_err, "literal_pass_1e-6": lit_err <= 1e-6,
            "values": got}


def zero_noise_check(theta_q):
    """Check 4: no programming/read noise, zero drift spread -> V2+replica and V3 exact."""
    out = all_thresholds(theta_q, zero_noise(theta_q.size))
    err = {}
    for scheme, track, cond, lab, a, th in out:
        if (scheme, track) in (("V2_abs_high", "replica"), ("V3_ratio", "none")):
            key = f"{scheme}+{track}@{cond}@{lab}"
            err[key] = float(np.max(np.abs(th - theta_q)))
    return err


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def fmt(v):
    if isinstance(v, (float, np.floating)):
        return repr(float(v))
    return str(v)


def write_csv(path, header, rows):
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(",".join(header) + "\n")
        for r in rows:
            fh.write(",".join(fmt(v) for v in r) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out-root", default=RESULTS_ROOT)
    args = ap.parse_args()

    t_start = time.time()
    now = _dt.datetime.now(JST)
    stamp = now.strftime("%Y%m%d_%H%M")
    cfg, cfg_sha = load_config()
    with open(os.path.abspath(__file__), "rb") as fh:
        script_sha = hashlib.sha256(fh.read()).hexdigest()

    datasets = ["iris", "wine"] if args.quick else DATASETS
    splits = [0] if args.quick else SPLITS
    n_dev = 2 if args.quick else N_DEV

    per_eval = []                 # rows for per_eval.csv
    refs = []                     # rows for refs.csv
    acc_ref = {}                  # (ds, split) -> acc_ref
    selfcheck = {"cam_vs_quantized_traversal_split0": {}, "ideal_cam_one_match_per_tree_all_splits": {},
                 "float_threshold_cam_vs_sklearn_split0": {}, "zero_noise_tracking_check_split0": {}}
    diag = {"interior_exact_ties_ideal": {}, "boundary_exact_ties_ideal": {},
            "near_ties_1e-12_not_exact_ideal": {}, "alt_minmaxscaler": {},
            "half_rounding_cases": 0, "thresholds_outside_01": 0, "single_leaf_trees": 0,
            "rows_without_constraints": 0, "infeasible_rows_after_quantization": 0,
            "score_exact_ties_ideal": 0, "score_near_ties_ideal": 0,
            "score_exact_ties_noisy": 0, "score_near_ties_noisy": 0, "nomatch_samples_noisy": 0,
            "rows_renormalisation_changed_value": 0,
            "pred_diff_onematmul_ideal": 0, "pred_diff_rawvalue_ideal": 0,
            "pred_diff_onematmul_noisy": 0, "pred_diff_rawvalue_noisy": 0,
            "evals_acc_diff_onematmul": 0, "evals_acc_diff_rawvalue": 0,
            "n_evaluations": 0}

    for ds in datasets:
        ds_idx = DATASETS.index(ds)
        X, y = load_xy(ds)
        for split in splits:
            Xtr_raw, Xte_raw, ytr, yte = split_raw(X, y, split)
            Xtr, Xte = minmax(Xtr_raw, Xte_raw)
            rf = fit_forest(Xtr, ytr, split)
            cam = build_cam(rf)
            theta_q = quantize(cam.thr, BITS)
            ev = CamEvaluator(cam, Xte, yte, rf.classes_)

            # ---- ideal 4-bit CAM (reference) -------------------------------------
            ideal = ev.evaluate(theta_q, keep_match=True)
            acc_ref[(ds, split)] = ideal["acc"]
            refs.append([ds, split, Xte.shape[0], cam.n_rows, cam.K, ideal["acc"]])
            diag["score_exact_ties_ideal"] += ideal["n_exact_tie"]
            diag["score_near_ties_ideal"] += ideal["n_near_tie"]
            diag["pred_diff_onematmul_ideal"] += ideal["n_pred_diff_onematmul"]
            diag["pred_diff_rawvalue_ideal"] += ideal["n_pred_diff_rawvalue"]
            diag["rows_renormalisation_changed_value"] += int(np.sum(np.any(cam.V != cam.V_raw, axis=1)))
            n_bad = int(np.sum(ideal["counts"] != 1))
            selfcheck["ideal_cam_one_match_per_tree_all_splits"][f"{ds}/split{split}"] = n_bad

            # ---- diagnostics on the specification ---------------------------------
            lv = float(2 ** BITS - 1)
            scaled = np.clip(cam.thr, 0.0, 1.0) * lv
            diag["half_rounding_cases"] += int(np.sum(scaled - np.floor(scaled) == 0.5))
            diag["thresholds_outside_01"] += int(np.sum((cam.thr < 0.0) | (cam.thr > 1.0)))
            diag["single_leaf_trees"] += cam.n_single_leaf
            diag["rows_without_constraints"] += int(np.sum(~ev.nonempty))
            diag["infeasible_rows_after_quantization"] += int(np.sum(_infeasible_rows(cam, theta_q)))
            eq = ev.XK == theta_q[None, :]
            interior = (theta_q > 0.0) & (theta_q < 1.0)
            key = f"{ds}/split{split}"
            diag["interior_exact_ties_ideal"][key] = int(np.sum(eq[:, interior]))
            diag["boundary_exact_ties_ideal"][key] = int(np.sum(eq[:, ~interior]))
            near = (np.abs(ev.XK - theta_q[None, :]) <= 1e-12) & ~eq
            diag["near_ties_1e-12_not_exact_ideal"][key] = int(np.sum(near))
            # alternative normalisation arithmetic (sklearn MinMaxScaler) -> sensitivity of acc_ref
            sc = MinMaxScaler().fit(Xtr_raw)
            Xtr_a = sc.transform(Xtr_raw)
            Xte_a = np.clip(sc.transform(Xte_raw), 0.0, 1.0)
            rf_a = fit_forest(Xtr_a, ytr, split)
            same = all(np.array_equal(a.tree_.threshold, b.tree_.threshold)
                       and np.array_equal(a.tree_.feature, b.tree_.feature)
                       and np.array_equal(a.tree_.children_left, b.tree_.children_left)
                       and np.array_equal(a.tree_.value, b.tree_.value)
                       for a, b in zip(rf.estimators_, rf_a.estimators_))
            cam_a = build_cam(rf_a)
            acc_a = CamEvaluator(cam_a, Xte_a, yte, rf_a.classes_).evaluate(quantize(cam_a.thr, BITS))["acc"]
            diag["alt_minmaxscaler"][key] = {
                "forest_identical": bool(same),
                "n_train_values_differ": int(np.sum(Xtr_a != Xtr)),
                "n_test_values_differ": int(np.sum(Xte_a != Xte)),
                "acc_ref": ideal["acc"], "acc_ref_alt": acc_a}

            # ---- self-checks on split 0 ---------------------------------------------
            if split == 0:
                leaves, pred_trav = traverse_quantized(rf, Xte, BITS)
                m = ideal["match"]
                leaf_mismatch = 0
                for t in range(cam.n_trees):
                    a, b = cam.tree_row_start[t], cam.tree_row_start[t + 1]
                    mt = m[:, a:b]
                    one = mt.sum(axis=1) == 1
                    row = a + np.argmax(mt, axis=1)
                    leaf_mismatch += int(np.sum(~one | (cam.row_leaf[row] != leaves[:, t])))
                pred_mismatch = int(np.sum(ideal["pred"] != pred_trav))
                selfcheck["cam_vs_quantized_traversal_split0"][ds] = {
                    "n_test": int(Xte.shape[0]), "prediction_mismatches": pred_mismatch,
                    "leaf_mismatches_or_not_exactly_one_match": leaf_mismatch}
                fl = ev.evaluate(cam.thr)["pred"]
                skl = rf.predict(Xte)
                selfcheck["float_threshold_cam_vs_sklearn_split0"][ds] = {
                    "agreement": float(np.mean(fl == skl)), "n_disagree": int(np.sum(fl != skl))}
                selfcheck["zero_noise_tracking_check_split0"][ds] = max(zero_noise_check(theta_q).values())

            # ---- device seeds ----------------------------------------------------------
            for dev in range(n_dev):
                Z = draw_noise(ds_idx, split, dev, cam.K)
                for scheme, track, cond, lab, a_hat, theta in all_thresholds(theta_q, Z):
                    r = ev.evaluate(theta)
                    per_eval.append([ds, split, dev, scheme, track, cond, lab, a_hat, r["acc"], r["integrity"]])
                    diag["score_exact_ties_noisy"] += r["n_exact_tie"]
                    diag["score_near_ties_noisy"] += r["n_near_tie"]
                    diag["nomatch_samples_noisy"] += r["n_nomatch"]
                    diag["pred_diff_onematmul_noisy"] += r["n_pred_diff_onematmul"]
                    diag["pred_diff_rawvalue_noisy"] += r["n_pred_diff_rawvalue"]
                    diag["evals_acc_diff_onematmul"] += int(r["acc_diff_onematmul"])
                    diag["evals_acc_diff_rawvalue"] += int(r["acc_diff_rawvalue"])
                    diag["n_evaluations"] += 1
            print(f"[{_dt.datetime.now(JST):%H:%M:%S}] {ds} split {split}: n_test={Xte.shape[0]} "
                  f"rows={cam.n_rows} K={cam.K} acc_ref={ideal['acc']:.4f}", flush=True)

    # ---- summary ---------------------------------------------------------------------
    losses = {}
    for ds, split, dev, scheme, track, cond, lab, a_hat, acc, integ in per_eval:
        losses.setdefault((scheme, track, cond, lab, ds), []).append(100.0 * (acc_ref[(ds, split)] - acc))
    summary = []
    S = {}
    for scheme, track in PAIRS:
        for cond, lab in CONDITIONS:
            per_ds = [float(np.mean(losses[(scheme, track, cond, lab, ds)])) for ds in datasets]
            for ds in datasets:
                assert len(losses[(scheme, track, cond, lab, ds)]) == len(splits) * n_dev
            mean_l = float(np.mean(per_ds))
            imax = int(np.argmax(per_ds))
            row = [scheme, track, cond, lab, mean_l, per_ds[imax], datasets[imax]] + per_ds
            summary.append(row)
            S[(scheme, track, cond, lab)] = {"mean": mean_l, "max": per_ds[imax], "worst": datasets[imax],
                                             "per_dataset": dict(zip(datasets, per_ds))}

    # ---- gates (config "decision", 4 bits) ------------------------------------------------
    def meets_a(scheme, track):
        r = S[(scheme, track, "prog", "t0")]
        return r["mean"] <= 2.0 and r["max"] <= 5.0

    gA_detail = {s: {"mean_loss_prog": S[(s, "none", "prog", "t0")]["mean"],
                     "max_loss_prog": S[(s, "none", "prog", "t0")]["max"],
                     "meets_A": meets_a(s, "none")}
                 for s in ("V1_abs_full", "V2_abs_high", "V3_ratio")}
    gate_A = any(d["meets_A"] for d in gA_detail.values())
    m1 = S[("V1_abs_full", "none", "prog", "t0")]["mean"]
    m2 = S[("V2_abs_high", "none", "prog", "t0")]["mean"]
    chosen = "V1_abs_full" if m1 <= m2 else "V2_abs_high"
    diag["gateB_tie_between_V1_V2"] = bool(m1 == m2)
    b_val = S[(chosen, "none", "all", "1d")]["mean"]
    gate_B = b_val >= 10.0
    gC_detail = {}
    for s, t in (("V1_abs_full", "replica"), ("V2_abs_high", "replica"), ("V3_ratio", "none")):
        v30 = S[(s, t, "all", "30d")]["mean"]
        gC_detail[f"{s}+{t}"] = {"meets_A": meets_a(s, t),
                                 "mean_loss_prog": S[(s, t, "prog", "t0")]["mean"],
                                 "max_loss_prog": S[(s, t, "prog", "t0")]["max"],
                                 "mean_loss_all@30d": v30,
                                 "passes_C": bool(meets_a(s, t) and v30 <= 5.0)}
    gate_C = any(d["passes_C"] for d in gC_detail.values())
    failing = [g for g, ok in (("A", gate_A), ("B", gate_B), ("C", gate_C)) if not ok]
    gates = {
        "gate_A": "PASS" if gate_A else "FAIL",
        "gate_B": "PASS" if gate_B else "FAIL",
        "gate_C": "PASS" if gate_C else "FAIL",
        "verdict": "GO" if not failing else "NO_GO",
        "failing_gates": failing,
        "details": {
            "gate_A": {"rule": cfg["decision"]["gate_A"], "per_scheme": gA_detail},
            "gate_B": {"rule": cfg["decision"]["gate_B"], "mean_loss_prog": {"V1_abs_full": m1, "V2_abs_high": m2},
                       "chosen_scheme": chosen, "untracked_mean_loss_all@1d": b_val},
            "gate_C": {"rule": cfg["decision"]["gate_C"], "candidates": gC_detail},
        },
        "scope": {"datasets": datasets, "splits": splits, "device_seeds": n_dev, "bits": BITS},
    }

    # ---- write outputs (new folder, never overwrite) ----------------------------------------
    name = f"E0_1_independent_check_{stamp}" + ("_quick" if args.quick else "")
    out = os.path.join(args.out_root, name)
    os.makedirs(out, exist_ok=False)
    write_csv(os.path.join(out, "per_eval.csv"),
              ["dataset", "split", "device_seed", "scheme", "track", "condition", "t", "a_hat", "acc", "integrity"],
              per_eval)
    write_csv(os.path.join(out, "refs.csv"),
              ["dataset", "split", "n_test", "n_rows", "n_constraints", "acc_ref"], refs)
    write_csv(os.path.join(out, "summary.csv"),
              ["scheme", "track", "condition", "t", "mean_loss_pp", "max_loss_pp", "worst_dataset"]
              + [f"loss_pp_{ds}" for ds in datasets], summary)
    with open(os.path.join(out, "gates.json"), "w", encoding="utf-8") as fh:
        json.dump(gates, fh, indent=2, ensure_ascii=False)
    selfcheck["hand_values_check3"] = hand_check()
    tot_pred = sum(v["prediction_mismatches"] for v in selfcheck["cam_vs_quantized_traversal_split0"].values())
    tot_leaf = sum(v["leaf_mismatches_or_not_exactly_one_match"]
                   for v in selfcheck["cam_vs_quantized_traversal_split0"].values())
    selfcheck["SELF_CHECK_total_prediction_mismatches"] = tot_pred
    selfcheck["SELF_CHECK_total_leaf_mismatches"] = tot_leaf
    selfcheck["SELF_CHECK_pass"] = (tot_pred == 0 and tot_leaf == 0)
    with open(os.path.join(out, "self_check.json"), "w", encoding="utf-8") as fh:
        json.dump(selfcheck, fh, indent=2, ensure_ascii=False)
    with open(os.path.join(out, "diagnostics.json"), "w", encoding="utf-8") as fh:
        json.dump(diag, fh, indent=2, ensure_ascii=False)
    elapsed = time.time() - t_start
    run_info = {"runstamp_jst": now.strftime("%Y-%m-%d %H:%M:%S %Z"), "elapsed_s": elapsed,
                "script": os.path.abspath(__file__), "script_sha256": script_sha,
                "config": CONFIG_PATH, "config_sha256": cfg_sha,
                "config_plan_version": cfg.get("plan_version"),
                "python": sys.version, "platform": platform.platform(),
                "numpy": np.__version__, "scikit_learn": sklearn.__version__, "scipy": scipy.__version__,
                "quick": bool(args.quick)}
    with open(os.path.join(out, "run_info.json"), "w", encoding="utf-8") as fh:
        json.dump(run_info, fh, indent=2, ensure_ascii=False)
    with open(os.path.join(out, "spec_notes.md"), "w", encoding="utf-8") as fh:
        fh.write(spec_notes(now, cfg_sha, script_sha, diag, selfcheck, gates))

    # ---- console summary ---------------------------------------------------------------
    print("\nOUTPUT:", out)
    print("self-check: prediction mismatches =", tot_pred, "| leaf mismatches =", tot_leaf)
    print(f"{'scheme':12s} {'track':8s} " + " ".join(f"{c+'@'+l:>16s}" for c, l in CONDITIONS))
    for scheme, track in PAIRS:
        cells = [f"{S[(scheme, track, c, l)]['mean']:7.3f}/{S[(scheme, track, c, l)]['max']:7.3f}" for c, l in CONDITIONS]
        print(f"{scheme:12s} {track:8s} " + " ".join(f"{x:>16s}" for x in cells))
    print("gates:", {k: gates[k] for k in ("gate_A", "gate_B", "gate_C", "verdict")})
    print(f"elapsed {elapsed:.1f} s")


def _infeasible_rows(cam, theta_q):
    """Rows whose quantized constraints can never be met (some feature has L_q >= U_q)."""
    bad = np.zeros(cam.n_rows, dtype=bool)
    for r in range(cam.n_rows):
        a, b = cam.row_start[r], cam.row_start[r + 1]
        lo = {}
        up = {}
        for k in range(a, b):
            (lo if cam.kind[k] == 0 else up)[int(cam.feat[k])] = theta_q[k]
        for f in lo:
            if f in up and lo[f] >= up[f]:
                bad[r] = True
        if any(v >= 1.0 for v in lo.values()) or any(v < 0.0 for v in up.values()):
            bad[r] = True
    return bad


def spec_notes(now, cfg_sha, script_sha, diag, selfcheck, gates):
    alt = diag["alt_minmaxscaler"]
    alt_same = all(v["forest_identical"] for v in alt.values())
    alt_diff = {k: (v["acc_ref"], v["acc_ref_alt"]) for k, v in alt.items() if v["acc_ref"] != v["acc_ref_alt"]}
    n_test_diff = sum(v["n_test_values_differ"] for v in alt.values())
    int_ties = sum(diag["interior_exact_ties_ideal"].values())
    bnd_ties = sum(diag["boundary_exact_ties_ideal"].values())
    near_ties = sum(diag["near_ties_1e-12_not_exact_ideal"].values())
    tie_by = {k: v for k, v in diag["interior_exact_ties_ideal"].items() if v}
    L = []
    L.append("# E0.1 独立复算：规格歧义与处理记录（spec notes）\n")
    L.append(f"- 生成时间：[{now:%Y-%m-%d %H:%M}] JST")
    L.append("- 依据：`docs/E0_1_protocol.md`（冻结版）、`experiments/E0_1_config.json`"
             f"（plan 1.1，SHA-256 `{cfg_sha}`）、复核任务说明。未读取项目源码、项目脚本或其他结果目录。")
    L.append(f"- 复算脚本 SHA-256：`{script_sha}`；配置常数与任务说明逐项断言一致（全部一致）。\n")
    L.append("## 1 数据与归一化\n")
    L.append("1. **归一化的浮点算式未规定**。本实现：span = max − min（训练集），span = 0 → 1；"
             "x_n = (x − min)/span；训练集不截断（构造上已在 [0,1]）；测试集 `np.clip` 到 [0,1]。"
             "sklearn `MinMaxScaler` 的算式 x·scale + min_ 与之可差 1 ulp，只在测试值恰好等于量化门限时改变比较结果。"
             f"诊断：用 MinMaxScaler 重跑，森林{'全部逐位相同' if alt_same else '存在不同'}，"
             f"测试特征有 {n_test_diff} 个值差 ≥1 ulp；理想 4-bit CAM 中测试值与内部门限（0<θ_q<1）恰好相等的"
             f"（样本, 约束）对共 {int_ties} 个{('（' + ', '.join(f'{k}: {v}' for k, v in tie_by.items()) + '）') if tie_by else ''}，"
             f"相差在 1e-12 以内但不相等的对数 = {near_ties}；"
             + ("acc_ref 在两种算式下完全相同。" if not alt_diff else
                "acc_ref 有差异：" + "; ".join(f"{k}: {a!r} vs {b!r}" for k, (a, b) in alt_diff.items()) + "。"))
    L.append(f"   θ_q ∈ {{0, 1}} 处的恰好相等共 {bnd_ties} 个（主要是 digits 的 0 像素与 θ_q = 0），"
             "这类相等在任何算式下都确定（0 与 1 在两种算式下都精确），按 x ≤ θ 匹配上界、x > θ 不匹配下界处理。")
    L.append("2. 数据以 float64 读入；标签直接用 sklearn 原值（classes_ = 0..C−1）；森林在归一化后的训练集上拟合"
             "（sklearn 内部转 float32，门限为两个 float32 值的中点，以 float64 保存）。CAM 与自检遍历都用 float64 特征"
             "与 float64 量化门限比较，不做 float32 转换。\n")
    L.append("## 2 树 → CAM 行\n")
    L.append("3. 最紧界在原始门限上取（下界取最大、上界取最小）后再量化；因量化单调不减，与先量化后取最紧界等价。")
    L.append(f"4. 无约束行（整棵树只有一个叶）：永远匹配。本次出现 {diag['single_leaf_trees']} 棵单叶树。")
    L.append(f"5. 量化后不可满足的行（同一特征 L_q ≥ U_q，或 L_q ≥ 1、U_q < 0）照常保留在行序与随机数下标中，"
             f"理想 CAM 下永不匹配；本次共 {diag['infeasible_rows_after_quantization']} 行。")
    L.append("6. 类别向量 = `tree_.value[leaf,0,:]` 除以其和（sklearn 1.9 已存比例，归一化只在 ulp 级别起作用）。\n")
    L.append("## 3 量化与随机数\n")
    L.append(f"7. round 用 `np.round`（半数取偶）。thr·15 小数部分恰为 0.5 的门限数 = {diag['half_rounding_cases']}，"
             "因此“半数取偶/四舍五入”之争对结果无影响。门限超出 [0,1] 的个数 = "
             f"{diag['thresholds_outside_01']}。")
    L.append("8. 随机数严格按 z_prog (K,2) → z_nu (K,2) → z_read (5,K,2) → z_rep_prog (64) → z_rep_nu (64) → "
             "z_rep_read (5,64) 的顺序以 `standard_normal` 抽取（C 序），prog 条件也照样全部抽取后只用 z_prog。"
             "V1 与 V2 共用同一组 z（第 0 列）与同一组复制单元 z（目标分别 12.5、16.61 µS），即成对比较。")
    L.append("9. 5 个时刻的读噪声是 5 组独立抽样（z_read[t_idx]），不是同一器件随时间的相关序列；"
             "编程噪声与漂移指数在各时刻共用。\n")
    L.append("## 4 器件与方案\n")
    L.append("10. t = t_inference + 20 s；t > 20 s 才乘漂移因子，t0 时因子 = 1；读噪声 sig 用这个 t（t0 时 t = 20 s），"
             "因此 t0 已有读噪声。Q 用 G_prog 计算，乘的是 |G_drift|（照协议原文）。")
    L.append("11. μ、σν 用编程目标 G 计算，g_rel = max(G/25, 1e-7)（下限只影响 G = 0，且被 clip 吸收，不改变数值）。")
    L.append("12. V3：r = 1/3 + θ_q/3（任务说明写法）；与配置的 r_lo + θ_q (r_hi − r_lo) 只差 ≤1 ulp，在连续噪声下无影响。"
             "G1+G2 = 0 时比值取 0.5（按任务说明）。")
    L.append("13. replica：â(t) = mean(64 个复制单元在 t 的 G_final) / mean(64 个复制单元的 G_prog)；"
             "prog 条件 â = 1，所以 (V1/V2, replica, prog) 与 (V1/V2, none, prog) 完全相同，仍各自输出一行。"
             "V3 的 â 记为 1.0。\n")
    L.append("## 5 匹配、投票与指标\n")
    L.append("14. 得分求和顺序：每棵树内用 one-hot 匹配向量 × 类别矩阵（通常只有一行匹配，结果精确），"
             "再按森林顺序逐棵累加；argmax 并列取最小下标。诊断：理想 CAM 中得分恰好并列的样本 "
             f"{diag['score_exact_ties_ideal']} 个、0 < 差 ≤ 1e-9 的近并列 {diag['score_near_ties_ideal']} 个；"
             f"全部噪声评估中恰好并列 {diag['score_exact_ties_noisy']} 个、近并列 {diag['score_near_ties_noisy']} 个"
             "（近并列的样本其结果可能依赖浮点求和顺序）。敏感度诊断：改成“全部行一次矩阵乘”求和时，"
             f"理想 CAM 预测变化 {diag['pred_diff_onematmul_ideal']} 个、噪声评估预测变化 "
             f"{diag['pred_diff_onematmul_noisy']} 个（精度改变的评估 {diag['evals_acc_diff_onematmul']} 次）；"
             "改用未再归一化的 `tree_.value` 时（再归一化改变了 "
             f"{diag['rows_renormalisation_changed_value']} 行的值），理想 CAM 预测变化 {diag['pred_diff_rawvalue_ideal']} 个、"
             f"噪声评估预测变化 {diag['pred_diff_rawvalue_noisy']} 个（精度改变的评估 {diag['evals_acc_diff_rawvalue']} 次）。")
    L.append(f"15. 所有树都没有匹配行的样本记为错误；噪声评估中这样的样本共 {diag['nomatch_samples_noisy']} 个"
             f"（{diag['n_evaluations']} 次评估合计）。路径完整度在全部测试样本上计算（含无匹配样本）。")
    L.append("16. loss 先逐次评估算 100·(acc_ref − acc) 再在 3 分割 × 5 器件种子上平均（与先平均再相减数学上相同）；"
             "max_loss 并列时取数据集顺序中的第一个。per_eval.csv 的 t 列写时刻标签（t0/1h/1d/30d/365d），"
             "condition 列写 prog/all。\n")
    L.append("## 6 判定\n")
    L.append("17. 关口 A 用 track = none 的 prog 行（与 replica 相同）。关口 C 的候选各自用自己的 prog 行判定 A 标准。")
    L.append("18. 关口 B：V1、V2 的 mean_loss(prog) 若相等，规格未说取哪一个；本实现取 V1。本次"
             + ("出现并列。" if diag.get("gateB_tie_between_V1_V2") else "未出现并列，规则不起作用。"))
    L.append("19. 判定阈值按“≤ / ≥”含等号处理（照配置原文）。\n")
    L.append("## 7 自检\n")
    sc = selfcheck["cam_vs_quantized_traversal_split0"]
    L.append("- 理想 4-bit CAM vs 量化门限直接遍历（split 0，全部数据集）：预测不一致 "
             f"{selfcheck['SELF_CHECK_total_prediction_mismatches']} 个，叶节点不一致或非恰好一行匹配 "
             f"{selfcheck['SELF_CHECK_total_leaf_mismatches']} 个（" + ", ".join(
                 f"{k}: {v['prediction_mismatches']}/{v['n_test']}" for k, v in sc.items()) + "）。")
    L.append("- 理想 CAM 每棵树恰好一行匹配（全部分割）：非 1 的（样本, 树）对数 = "
             + ", ".join(f"{k}: {v}" for k, v in selfcheck["ideal_cam_one_match_per_tree_all_splits"].items()) + "。")
    L.append("- float64 门限 CAM vs sklearn predict（split 0）：" + ", ".join(
        f"{k}: {v['agreement']:.4f}" for k, v in selfcheck["float_threshold_cam_vs_sklearn_split0"].items()) + "。")
    L.append("- 无噪声、σν = 0 时 V2+replica 与 V3 复原 θ_q 的最大误差（split 0）：" + ", ".join(
        f"{k}: {v:.1e}" for k, v in selfcheck["zero_noise_tracking_check_split0"].items()) + "。")
    L.append(f"- PCM 公式手算值最大误差：{selfcheck['hand_values_check3']['max_abs_err_formula']:.1e}。\n")
    L.append("## 8 判定结果（本复算）\n")
    L.append(f"- gate_A = {gates['gate_A']}，gate_B = {gates['gate_B']}，gate_C = {gates['gate_C']}，"
             f"verdict = {gates['verdict']}" + (f"（失败关口：{', '.join(gates['failing_gates'])}）" if gates['failing_gates'] else "") + "。")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
