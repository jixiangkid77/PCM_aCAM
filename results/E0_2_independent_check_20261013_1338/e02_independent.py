#!/usr/bin/env python3
"""Independent re-implementation of the PCM_aCAM E0.2 primary analysis.

Written only from docs/E0_1_protocol.md, experiments/E0_1_config.json,
docs/E0_2_protocol.md and experiments/E0_2_config.json (no project code read).

Primary analysis: mismatch m20, threshold programming, 4 bits, nodes
none/dac/supply/ideal, conditions prog@t0 and all@{t0,1h,1d,30d,365d},
5 datasets x 3 splits x 5 device seeds (1,800 evaluations).

Usage:
    python3 e02_independent.py                 # final run -> <project>/results/E0_2_independent_check_<JST stamp>/
    python3 e02_independent.py --out DIR       # development run into DIR (must not exist)
"""
import argparse
import csv
import datetime
import hashlib
import json
import os
import platform
import re
import sys
import time
from zoneinfo import ZoneInfo

import numpy as np
import scipy
import sklearn
from sklearn.datasets import (load_breast_cancer, load_digits, load_iris,
                              load_wine, make_classification)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# ----------------------------------------------------------------------------
# Paths (config paths derived relative to the project folder)
# ----------------------------------------------------------------------------
PROJECT_DIR = "/home/claude/PCM_aCAM"
CFG1_PATH = os.path.join(PROJECT_DIR, "experiments", "E0_1_config.json")
CFG2_PATH = os.path.join(PROJECT_DIR, "experiments", "E0_2_config.json")
PROTO1_PATH = os.path.join(PROJECT_DIR, "docs", "E0_1_protocol.md")
PROTO2_PATH = os.path.join(PROJECT_DIR, "docs", "E0_2_protocol.md")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")

DATASETS = ["iris", "wine", "breast_cancer", "digits", "synthetic"]   # ds_idx 0..4
NODES = ["none", "dac", "supply", "ideal"]
TIE_TOL = 1e-9          # only for the diagnostic "tolerant argmax" (not the primary result)
CHECK_TOL = 1e-12


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def jst_now():
    return datetime.datetime.now(ZoneInfo("Asia/Tokyo"))


# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
def load_config():
    with open(CFG1_PATH) as f:
        cfg1 = json.load(f)
    with open(CFG2_PATH) as f:
        cfg2 = json.load(f)

    pcm = dict(cfg2["pcm_model"])
    # E0.2 says the PCM model is identical to E0.1: verify
    assert pcm == cfg1["pcm_model"], "E0.1 and E0.2 PCM model blocks differ"

    cell = cfg2["cell_model"]
    m20 = cfg2["mismatch"]["settings"]["m20"]
    assert m20["role"] == "primary"

    # synthetic dataset parameters parsed from the config string
    s = cfg2["datasets"]["synthetic"]
    inner = s[s.index("(") + 1: s.rindex(")")]
    synth = {}
    for part in inner.split(","):
        k, v = part.split("=")
        v = v.strip()
        synth[k.strip()] = int(v) if re.fullmatch(r"-?\d+", v) else float(v)
    assert s == cfg1["datasets"]["synthetic"]

    # forest parameters parsed from the config string
    model = cfg2["model"]
    n_est = int(re.search(r"n_estimators=(\d+)", model).group(1))
    max_depth = int(re.search(r"max_depth=(\d+)", model).group(1))

    times = list(cfg2["times_inference_s"].items())   # JSON order: t0, 1h, 1d, 30d, 365d
    assert [k for k, _ in times] == ["t0", "1h", "1d", "30d", "365d"]
    assert [v for _, v in times] == [0, 3600, 86400, 2592000, 31536000]

    c = dict(
        cfg1=cfg1, cfg2=cfg2, pcm=pcm, cell=cell, m20=m20, synth=synth,
        n_estimators=n_est, max_depth=max_depth,
        split_seeds=list(cfg2["split_seeds"]), n_dev=int(cfg2["device_seeds"]),
        base_seed=int(cfg2["base_seed"]), bits=int(cfg2["bits"]),
        n_rep=int(cfg2["replica_cells"]), times=times,
    )
    # sanity against the task summary
    assert (n_est, max_depth) == (16, 6)
    assert c["split_seeds"] == [0, 1, 2] and c["n_dev"] == 5 and c["base_seed"] == 20261012
    assert c["bits"] == 4 and c["n_rep"] == 64
    assert (cell["v_sl"], cell["v_th"], cell["v_tx"], cell["g_lo"], cell["g_hi"], cell["g_max"]) == \
        (0.5, 0.2, 0.35, 2.5, 22.5, 25.0)
    assert cell["replica_target_uS"] == 0.5 * (cell["g_lo"] + cell["g_hi"]) == 12.5
    assert (m20["sigma_vth_V"], m20["sigma_vtx_V"], m20["sigma_beta"], m20["r_in_V"]) == (0.02, 0.02, 0.02, 0.4)
    assert synth == dict(n_samples=5000, n_features=20, n_informative=10, n_redundant=4,
                         n_classes=4, n_clusters_per_class=2, class_sep=1.0, random_state=20261012)
    return c


# ----------------------------------------------------------------------------
# Data and forest
# ----------------------------------------------------------------------------
def load_dataset(name, synth):
    if name == "iris":
        d = load_iris()
    elif name == "wine":
        d = load_wine()
    elif name == "breast_cancer":
        d = load_breast_cancer()
    elif name == "digits":
        d = load_digits()
    elif name == "synthetic":
        X, y = make_classification(**synth)
        return np.asarray(X, dtype=np.float64), np.asarray(y)
    else:
        raise ValueError(name)
    return np.asarray(d.data, dtype=np.float64), np.asarray(d.target)


def split_and_normalize(X, y, split_seed):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=split_seed)
    lo = Xtr.min(axis=0)
    hi = Xtr.max(axis=0)
    span = hi - lo
    span[span == 0] = 1.0
    Xtr_n = (Xtr - lo) / span
    Xte_n = np.clip((Xte - lo) / span, 0.0, 1.0)
    return Xtr_n, Xte_n, ytr, yte


def quantize(thr, bits):
    L = float(2 ** bits - 1)
    return np.round(np.clip(thr, 0.0, 1.0) * L) / L


def extract_rows(est):
    """Leaves in depth-first pre-order (left child first). For each leaf: list of
    constraints (feature, is_upper, tightest float threshold): lower bounds first,
    then upper bounds, each group ordered by first appearance of the feature as
    that kind of bound on the root-to-leaf path."""
    tr = est.tree_
    cl, cr, feat, thr, val = (tr.children_left, tr.children_right, tr.feature,
                              tr.threshold, tr.value)
    rows = []

    def rec(node, lo_ord, lo_val, up_ord, up_val):
        if cl[node] == cr[node]:            # leaf: both children == -1
            cons = [(f, False, lo_val[f]) for f in lo_ord] + [(f, True, up_val[f]) for f in up_ord]
            v = np.asarray(val[node, 0, :], dtype=np.float64)
            rows.append((int(node), cons, v / v.sum()))
            return
        f = int(feat[node])
        th = float(thr[node])
        # left child: x[f] <= th  -> upper bound (tightest = min)
        new_up_val = dict(up_val)
        if f in up_val:
            new_up_ord = up_ord
            new_up_val[f] = min(up_val[f], th)
        else:
            new_up_ord = up_ord + [f]
            new_up_val[f] = th
        rec(int(cl[node]), lo_ord, lo_val, new_up_ord, new_up_val)
        # right child: x[f] > th -> lower bound (tightest = max)
        new_lo_val = dict(lo_val)
        if f in lo_val:
            new_lo_ord = lo_ord
            new_lo_val[f] = max(lo_val[f], th)
        else:
            new_lo_ord = lo_ord + [f]
            new_lo_val[f] = th
        rec(int(cr[node]), new_lo_ord, new_lo_val, up_ord, up_val)

    rec(0, [], {}, [], {})
    return rows


class CAM:
    def __init__(self, rf):
        feats, is_up, thrs, row_starts, n_cons, V, leaf_nodes, row_tree, tree_row_starts = \
            [], [], [], [], [], [], [], [], []
        k = 0
        r = 0
        for t, est in enumerate(rf.estimators_):
            tree_row_starts.append(r)
            for node, cons, v in extract_rows(est):
                row_starts.append(k)
                n_cons.append(len(cons))
                for (f, up, th) in cons:
                    feats.append(f)
                    is_up.append(up)
                    thrs.append(th)
                    k += 1
                V.append(v)
                leaf_nodes.append(node)
                row_tree.append(t)
                r += 1
        self.feat = np.asarray(feats, dtype=np.int64)
        self.is_upper = np.asarray(is_up, dtype=bool)
        self.thr = np.asarray(thrs, dtype=np.float64)
        self.row_starts = np.asarray(row_starts, dtype=np.int64)
        self.n_cons = np.asarray(n_cons, dtype=np.int64)
        self.V = np.vstack(V)
        self.leaf_nodes = np.asarray(leaf_nodes, dtype=np.int64)
        self.row_tree = np.asarray(row_tree, dtype=np.int64)
        self.tree_row_starts = np.asarray(tree_row_starts, dtype=np.int64)
        self.classes = rf.classes_
        self.K = k
        self.R = r
        self.T = len(rf.estimators_)
        self.nonempty = self.n_cons > 0
        self.starts_ne = self.row_starts[self.nonempty]

    def row_match(self, M):
        """M: (N, K) constraint matches -> (N, R) row matches (a row with no
        constraint would always match)."""
        N = M.shape[0]
        if self.nonempty.all():
            return np.logical_and.reduceat(M, self.row_starts, axis=1)
        RM = np.ones((N, self.R), dtype=bool)
        RM[:, self.nonempty] = np.logical_and.reduceat(M, self.starts_ne, axis=1)
        return RM

    def evaluate(self, Xf, y, theta):
        """Xf = X_test[:, feat] (N, K). Lower bound matches if x > theta,
        upper bound matches if x <= theta."""
        M = (Xf > theta) != self.is_upper          # upper: not(x > theta) == (x <= theta)
        RM = self.row_match(M)
        cnt = np.add.reduceat(RM.view(np.int8), self.tree_row_starts, axis=1, dtype=np.int32)  # (N, T)
        integrity = float(np.mean(cnt == 1))
        any_match = cnt.sum(axis=1) > 0
        score = RM.astype(np.float64) @ self.V                                                    # (N, C)
        idx = np.argmax(score, axis=1)                                                            # lowest index on ties
        correct = any_match & (self.classes[idx] == y)
        # diagnostic: tolerant tie-breaking (scores within TIE_TOL of the max are treated as tied)
        smax = score.max(axis=1, keepdims=True)
        idx_tol = np.argmax(score >= smax - TIE_TOL, axis=1)
        correct_tol = any_match & (self.classes[idx_tol] == y)
        return dict(acc=float(np.mean(correct)), integrity=integrity,
                    n_correct=int(correct.sum()), n_nomatch=int((~any_match).sum()),
                    n_correct_tol=int(correct_tol.sum()), RM=RM, cnt=cnt, idx=idx)


def traverse_quantized(est, X, bits):
    """Reference traversal: follow x <= quantize(thr) -> left, else right."""
    tr = est.tree_
    N = X.shape[0]
    node = np.zeros(N, dtype=np.int64)
    ar = np.arange(N)
    for _ in range(tr.max_depth + 2):
        leaf = tr.children_left[node] == -1
        if leaf.all():
            break
        f = np.where(leaf, 0, tr.feature[node])
        th = quantize(tr.threshold[node], bits)
        go_left = X[ar, f] <= th
        nxt = np.where(go_left, tr.children_left[node], tr.children_right[node])
        node = np.where(leaf, node, nxt)
    assert (tr.children_left[node] == -1).all()
    return node


# ----------------------------------------------------------------------------
# PCM model (AIHWKit PCMLikeNoiseModel as specified)
# ----------------------------------------------------------------------------
def prog_sigma(G, p):
    x = G / p["g_max"]
    c0, c1, c2 = p["prog_coeff"]
    sig = c0 + c1 * x + c2 * x * x
    return sig * (p["g_max"] / p["prog_coeff_g_max_reference"])


def pcm_program(G_target, z, p):
    return np.maximum(G_target + p["prog_noise_scale"] * prog_sigma(G_target, p) * z, 0.0)


def pcm_nu(G_target, z, p):
    g_rel = np.maximum(np.abs(G_target / p["g_max"]), 1e-7)
    lg = np.log(g_rel)
    mu = np.clip(p["nu_mean_a"] * lg + p["nu_mean_b"], p["nu_mean_min"], p["nu_mean_max"])
    sd = np.clip(p["nu_std_a"] * lg + p["nu_std_b"], p["nu_std_min"], p["nu_std_max"])
    return np.abs(mu + sd * z) * p["drift_scale"]


def pcm_at(G_prog, nu, z_read, t_inf, p):
    t = t_inf + p["t0"]
    if t > p["t0"]:
        G_drift = G_prog * (t / p["t0"]) ** (-nu)
    else:
        G_drift = G_prog.copy()
    q = np.minimum(p["read_q0"] / np.maximum((np.abs(G_prog) / p["g_max"]) ** p["read_exp"], p["read_floor"]),
                   p["read_qmax"])
    sig = q * np.sqrt(np.log((t + p["t_read"]) / (2.0 * p["t_read"])))
    return np.maximum(G_drift + np.abs(G_drift) * p["read_noise_scale"] * sig * z_read, 0.0)


def make_draws(base_seed, ds_idx, split, dev, K, n_t, n_rep):
    rng = np.random.default_rng([base_seed, ds_idx, split, dev])
    d = {}
    d["z_prog"] = rng.standard_normal((K, 2))
    d["z_nu"] = rng.standard_normal((K, 2))
    d["z_read"] = rng.standard_normal((n_t, K, 2))
    d["z_rep_prog"] = rng.standard_normal(n_rep)
    d["z_rep_nu"] = rng.standard_normal(n_rep)
    d["z_rep_read"] = rng.standard_normal((n_t, n_rep))
    rng2 = np.random.default_rng([base_seed, ds_idx, split, dev, 2])
    d["z_vth"] = rng2.standard_normal(K)
    d["z_vtx"] = rng2.standard_normal(K)
    d["z_beta"] = rng2.standard_normal(K)
    return d


# ----------------------------------------------------------------------------
# Cell model (Li 2020 Eq. 1): V_bound = V_TH + G (V_SL/V_TX - 1)/beta
# ----------------------------------------------------------------------------
class Cell:
    def __init__(self, cell, r_in):
        self.v_sl = float(cell["v_sl"])
        self.v_th = float(cell["v_th"])
        self.v_tx = float(cell["v_tx"])
        self.g_lo = float(cell["g_lo"])
        self.g_hi = float(cell["g_hi"])
        self.g_max = float(cell["g_max"])
        self.r_in = float(r_in)
        self.c_nom = self.v_sl / self.v_tx - 1.0
        self.beta_nom = self.c_nom * (self.g_hi - self.g_lo) / self.r_in
        self.k_nom = self.c_nom / self.beta_nom

    def k_i(self, v, dvtx, dbeta):
        return (v / (self.v_tx + dvtx) - 1.0) / (self.beta_nom * (1.0 + dbeta))

    def g_in(self, theta_q):
        return self.g_lo + theta_q * (self.g_hi - self.g_lo)

    def target_threshold(self, theta_q, dvth, dvtx, dbeta):
        raw = (self.g_in(theta_q) * self.k_nom - dvth) / self.k_i(self.v_sl, dvtx, dbeta)
        return np.clip(raw, 0.0, self.g_max), raw

    def target_conductance(self, theta_q):
        g = self.g_in(theta_q)
        return np.clip(g, 0.0, self.g_max), g

    def theta_eff(self, node, G_t, a_hat, dvth, dvtx, dbeta):
        if node == "none":
            s, v, G = 1.0, self.v_sl, G_t
        elif node == "dac":
            s, v, G = a_hat, self.v_sl, G_t
        elif node == "supply":
            s, v, G = 1.0, self.v_tx * (1.0 + self.c_nom / a_hat), G_t
        elif node == "ideal":
            s, v, G = 1.0, self.v_sl, G_t / a_hat
        else:
            raise ValueError(node)
        v_rel = dvth + G * self.k_i(v, dvtx, dbeta)
        return (v_rel / (s * self.k_nom) - self.g_lo) / (self.g_hi - self.g_lo)


def simulate_conditions(theta_q, d, cellm, mism, p, times, n_rep, programming="threshold"):
    """Returns (list of (condition, t_label, t_s, G_t, a_hat), mismatch tuple, raw/clipped targets)."""
    dvth = mism["sigma_vth_V"] * d["z_vth"]
    dvtx = mism["sigma_vtx_V"] * d["z_vtx"]
    dbeta = mism["sigma_beta"] * d["z_beta"]
    if programming == "threshold":
        G_target, raw = cellm.target_threshold(theta_q, dvth, dvtx, dbeta)
    else:
        G_target, raw = cellm.target_conductance(theta_q)
    G_prog = pcm_program(G_target, d["z_prog"][:, 0], p)
    nu = pcm_nu(G_target, d["z_nu"][:, 0], p)
    rep_target = np.full(n_rep, 0.5 * (cellm.g_lo + cellm.g_hi))
    rep_prog = pcm_program(rep_target, d["z_rep_prog"], p)
    rep_nu = pcm_nu(rep_target, d["z_rep_nu"], p)
    rep_prog_mean = rep_prog.mean()
    conds = [("prog", "t0", 0, G_prog, 1.0)]
    for slot, (lab, t_inf) in enumerate(times):
        G_t = pcm_at(G_prog, nu, d["z_read"][slot, :, 0], float(t_inf), p)
        rep_t = pcm_at(rep_prog, rep_nu, d["z_rep_read"][slot], float(t_inf), p)
        a_hat = float(rep_t.mean() / rep_prog_mean)
        conds.append(("all", lab, t_inf, G_t, a_hat))
    return conds, (dvth, dvtx, dbeta), G_target, raw


# ----------------------------------------------------------------------------
# Self-checks
# ----------------------------------------------------------------------------
def self_check_1(prepared, c, cellm):
    """Zero mismatch, PCM noise and drift off: theta_eff == theta_q for every node,
    every condition, both programming modes."""
    p0 = dict(c["pcm"], prog_noise_scale=0.0, read_noise_scale=0.0, drift_scale=0.0)
    zero = dict(sigma_vth_V=0.0, sigma_vtx_V=0.0, sigma_beta=0.0)
    worst = 0.0
    a_hats = []
    for (ds_idx, ds, split), P in prepared.items():
        for dev in range(c["n_dev"]):
            d = make_draws(c["base_seed"], ds_idx, split, dev, P["cam"].K, len(c["times"]), c["n_rep"])
            for prog_mode in ("threshold", "conductance"):
                conds, (dvth, dvtx, dbeta), _, _ = simulate_conditions(
                    P["theta_q"], d, cellm, zero, p0, c["times"], c["n_rep"], prog_mode)
                for (cond, lab, t_s, G_t, a_hat) in conds:
                    a_hats.append(a_hat)
                    for node in NODES:
                        th = cellm.theta_eff(node, G_t, a_hat, dvth, dvtx, dbeta)
                        worst = max(worst, float(np.max(np.abs(th - P["theta_q"]))))
    return dict(max_abs_err=worst, a_hat_range=[min(a_hats), max(a_hats)],
                passed=bool(worst <= CHECK_TOL))


def self_check_2(prepared, c, cellm):
    """Uniform drift (mu = 0.049 for every state, sigma_nu = 0), no programming or
    read noise, m20 mismatch, threshold programming: compare with closed forms on
    unclipped cells (dac and ideal required; none and supply additionally)."""
    pu = dict(c["pcm"], nu_mean_a=0.0, nu_std_a=0.0, nu_std_b=0.0, nu_std_min=0.0, nu_std_max=0.0,
              prog_noise_scale=0.0, read_noise_scale=0.0)
    K0 = cellm.k_nom * (cellm.g_hi - cellm.g_lo)
    worst = {n: 0.0 for n in NODES}
    worst_ahat = 0.0
    n_unclipped = 0
    n_cells = 0
    for (ds_idx, ds, split), P in prepared.items():
        thq = P["theta_q"]
        for dev in range(c["n_dev"]):
            d = make_draws(c["base_seed"], ds_idx, split, dev, P["cam"].K, len(c["times"]), c["n_rep"])
            conds, (dvth, dvtx, dbeta), G_target, raw = simulate_conditions(
                thq, d, cellm, c["m20"], pu, c["times"], c["n_rep"], "threshold")
            ok = (raw >= 0.0) & (raw <= cellm.g_max)
            n_unclipped += int(ok.sum())
            n_cells += ok.size
            g_in = cellm.g_in(thq)
            eps = dvtx / (cellm.v_tx * cellm.c_nom)
            for (cond, lab, t_s, G_t, a_hat) in conds:
                if cond == "prog":
                    a = 1.0
                else:
                    a = ((t_s + 20.0) / 20.0) ** (-0.049)
                worst_ahat = max(worst_ahat, abs(a_hat - a))
                closed = {
                    "none": a * thq - (1.0 - a) * cellm.g_lo / (cellm.g_hi - cellm.g_lo) + (1.0 - a) * dvth / K0,
                    "dac": thq + (1.0 / a - 1.0) * dvth / K0,
                    "supply": thq + (g_in - dvth / cellm.k_nom) * eps * (1.0 - a) / ((1.0 - eps) * (cellm.g_hi - cellm.g_lo)),
                    "ideal": thq,
                }
                for node in NODES:
                    th = cellm.theta_eff(node, G_t, a_hat, dvth, dvtx, dbeta)
                    err = float(np.max(np.abs(th[ok] - closed[node][ok])))
                    worst[node] = max(worst[node], err)
    passed_required = worst["dac"] <= CHECK_TOL and worst["ideal"] <= CHECK_TOL
    passed_all = all(v <= CHECK_TOL for v in worst.values())
    return dict(max_abs_err=worst, max_abs_ahat_minus_a=worst_ahat,
                unclipped_cells=n_unclipped, cells=n_cells,
                passed_required_dac_ideal=bool(passed_required), passed_all_four=bool(passed_all))


def self_check_3(prepared, c, cellm):
    """Extra (protocol check 3): zero mismatch, full PCM model: dac == supply == ideal."""
    zero = dict(sigma_vth_V=0.0, sigma_vtx_V=0.0, sigma_beta=0.0)
    worst = 0.0
    for (ds_idx, ds, split), P in prepared.items():
        for dev in range(c["n_dev"]):
            d = make_draws(c["base_seed"], ds_idx, split, dev, P["cam"].K, len(c["times"]), c["n_rep"])
            conds, (dvth, dvtx, dbeta), _, _ = simulate_conditions(
                P["theta_q"], d, cellm, zero, c["pcm"], c["times"], c["n_rep"], "threshold")
            for (cond, lab, t_s, G_t, a_hat) in conds:
                th = {n: cellm.theta_eff(n, G_t, a_hat, dvth, dvtx, dbeta) for n in ("dac", "supply", "ideal")}
                worst = max(worst, float(np.max(np.abs(th["dac"] - th["ideal"]))),
                            float(np.max(np.abs(th["supply"] - th["ideal"]))))
    return dict(max_abs_err=worst, passed=bool(worst <= CHECK_TOL))


def structure_check(prepared, c):
    """Extra (protocol check 1 for all splits, check 2 for info): ideal quantized CAM ==
    traversal with quantized thresholds; float64-threshold CAM vs sklearn predict."""
    out = {}
    all_ok = True
    for (ds_idx, ds, split), P in prepared.items():
        cam, rf, Xte, yte, Xf = P["cam"], P["rf"], P["Xte"], P["yte"], P["Xf"]
        ev = cam.evaluate(Xf, yte, P["theta_q"])
        one_per_tree = bool((ev["cnt"] == 1).all())
        leaves_ok = True
        for t, est in enumerate(rf.estimators_):
            leaf_trav = traverse_quantized(est, Xte, c["bits"])
            r0, r1 = cam.tree_row_starts[t], (cam.tree_row_starts[t + 1] if t + 1 < cam.T else cam.R)
            rm = ev["RM"][:, r0:r1]
            leaf_cam = cam.leaf_nodes[r0:r1][np.argmax(rm, axis=1)]
            leaves_ok &= bool(np.array_equal(leaf_cam, leaf_trav))
        evf = cam.evaluate(Xf, yte, cam.thr)
        sk = rf.predict(Xte)
        agree_float = float(np.mean(cam.classes[evf["idx"]] == sk))
        exact_ties = int(np.sum(Xf == P["theta_q"]))
        ok = one_per_tree and leaves_ok
        all_ok &= ok
        out[f"{ds}/split{split}"] = dict(one_match_per_tree=one_per_tree, leaves_equal_traversal=leaves_ok,
                                         float64_cam_vs_sklearn_agreement=agree_float,
                                         exact_value_threshold_ties=exact_ties)
    return dict(passed=bool(all_ok), detail=out)


def normalization_variant_diagnostic(prepared, X_by_ds, c):
    """Diagnostic (not a gate): refit with sklearn MinMaxScaler arithmetic (X*scale + min)
    instead of (X - min)/span; report whether the forest and acc_ref change."""
    from sklearn.preprocessing import MinMaxScaler
    out = {}
    all_same = True
    for (ds_idx, ds, split), P in prepared.items():
        X, y = X_by_ds[ds]
        A_tr, A_te, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=split)
        sc = MinMaxScaler().fit(A_tr)
        B_tr = sc.transform(A_tr)
        B_te = np.clip(sc.transform(A_te), 0.0, 1.0)
        rfB = RandomForestClassifier(n_estimators=c["n_estimators"], max_depth=c["max_depth"],
                                     random_state=split, n_jobs=1).fit(B_tr, ytr)
        same = True
        for ea, eb in zip(P["rf"].estimators_, rfB.estimators_):
            ta, tb = ea.tree_, eb.tree_
            same &= (ta.node_count == tb.node_count and np.array_equal(ta.feature, tb.feature)
                     and np.array_equal(ta.threshold, tb.threshold) and np.array_equal(ta.value, tb.value))
        camB = CAM(rfB)
        accB = camB.evaluate(np.ascontiguousarray(B_te[:, camB.feat]), yte, quantize(camB.thr, c["bits"]))["acc"]
        same_acc = accB == P["acc_ref"]
        all_same &= bool(same) and bool(same_acc)
        out[f"{ds}/split{split}"] = dict(forest_identical=bool(same), acc_ref_minmaxscaler=accB,
                                         acc_ref_identical=bool(same_acc))
    return dict(all_identical=bool(all_same), detail=out)


SPEC_NOTES = """# E0.2 independent check: specification notes

Independent re-implementation written only from `docs/E0_1_protocol.md`, `experiments/E0_1_config.json`,
`docs/E0_2_protocol.md` and `experiments/E0_2_config.json` (no project code, no project results read).
Run: {start_jst}. Script `e02_independent.py` (copy in this folder), SHA-256 `{script_sha}`.

## Ambiguities resolved (and how much they matter)

1. **Feature space of the forest.** The forest is fitted on the min-max-normalized training split, so split
   thresholds are in normalized units and `theta_q = round(clip(thr,0,1)*15)/15` applies directly.
2. **Normalization arithmetic.** float64 `(X - min)/span`, `span = max - min` (0 -> 1), test values scaled
   then clipped to [0, 1]. sklearn `MinMaxScaler` arithmetic (`X*scale + min`) differs by ULPs. Diagnostic:
   forests identical and acc_ref identical for all 15 (dataset, split): **{norm_same}** (checks.json,
   `diag_normalization_variant`).
3. **Quantizer rounding.** `np.round` (half to even). The only exactly representable half-way case is
   thr = 0.5 (7.5 -> 8 under half-even and half-up alike), so the rounding rule cannot matter.
4. **Tightest bound vs quantization order.** The tightest float bound is taken first and then quantized; this is
   equivalent to quantize-then-tightest because the quantizer is monotone.
5. **Constraint order when a bound is tightened deeper in the path.** The feature keeps the position of its
   first appearance as that kind of bound; only its value is replaced by the tighter one (lower: max, upper: min).
   A feature with both a lower and an upper bound appears once in each group.
6. **Leaf order.** Depth-first pre-order with the left child first; only leaves emit rows, so rows follow the
   left-to-right leaf order; trees in `estimators_` order. Rows without any constraint (stumps) would always
   match; none occurred (refs.csv `rows_without_constraints`).
7. **Leaf class vector.** `tree_.value[leaf, 0, :] / sum` (sklearn 1.9.1 already stores fractions).
8. **Score summation and ties.** Score = float64 product (row-match indicator) x (class vectors); prediction =
   `np.argmax` (first index among exactly equal floats). Diagnostic: a tolerant argmax (scores within 1e-9 of
   the maximum treated as tied, lowest index wins) changes the number of correct samples in **{tol_diff}** of
   the {n_eval} evaluations (per_eval.csv `n_correct_tol`).
9. **No match in any tree** counts as wrong even if argmax of the zero vector equals the label.
10. **Path integrity** per evaluation = share of (test sample, tree) pairs with exactly one matching row
    (0 or >= 2 matches count as not intact).
11. **Exact value-threshold ties** (`x == theta`) follow the stated comparisons (lower `x > theta`, upper
    `x <= theta`). They only occur for acc_ref (theta_eff = theta_q); counts in refs.csv
    (`exact_value_threshold_ties`, dominated by digits pixels at exactly 0 or 1 meeting theta_q = 0 or 1). Under
    device noise theta_eff is continuous, so exact ties have probability zero.
12. **Which conductance sets the noise parameters.** Programming-noise sigma and the drift exponent use the
    clipped threshold-programming target (the value actually programmed); the read-noise Q uses G_prog after
    `max(.,0)` (AIHWKit semantics). For high-side clipped targets (raw > 25 uS) mu/sigma_nu are unchanged by
    this choice (both already at their floor at 25 uS) but sigma_prog is evaluated at 25 uS; for low-side
    clipped targets (raw < 0) the target is 0, so g_rel = 1e-7, mu = 0.1, sigma_nu = 0.045 and
    G_prog = max(0.26348 z, 0).
13. **Clipping count** = number of bounds whose raw target `(g_in k_nom - dV_TH)/k_i(v_sl)` is < 0 or > 25 uS
    (clipping.csv gives low/high separately). Total: **{clip_total}** of {cons_total} programmed bounds
    ({clip_low} low, {clip_high} high).
14. **a_hat.** Ratio of means: mean of the 64 replica `G_final(t)` / mean of the 64 replica `G_prog` (after
    the `max(.,0)` clamp), replica read noise from `z_rep_read[slot]`. At all@t0 a_hat differs from 1 because
    the accumulated read noise is already present (no drift at t0: t = 20 s is not > t0).
15. **prog condition.** `G_t = G_prog`, a_hat = 1 for every node; supply then uses
    `V_TX,nom (1 + c_nom) = v_sl` up to rounding, so the four nodes coincide. Accuracy identical across the four
    nodes in every prog evaluation: **{prog_same}**.
16. **k_nom** computed as `c_nom/beta_nom` with `beta_nom = c_nom (g_hi - g_lo)/r_in` (= r_in/(g_hi - g_lo) =
    0.02 V/uS up to rounding); c_nom = {c_nom!r}, beta_nom = {beta_nom!r} uS/V, k_nom = {k_nom!r} V/uS.
17. **theta_eff is not clipped**; a bound outside [0, 1] simply always or never matches.
18. **Pairing.** All nodes of one (dataset, split, device seed, condition) share the same G_t and a_hat; the five
    time slots of one device seed share z_prog and z_nu (one device trajectory) and use z_read[slot] /
    z_rep_read[slot].
19. **Loss averaging.** Per evaluation `loss = 100 (acc_ref(split) - acc)`; per dataset the mean over the 15
    (split, device) evaluations; mean_loss = mean over the 5 datasets; max_loss = largest per-dataset loss.
    Gates compare the float64 mean_loss differences with 1.0 pp (no rounding before comparison).
20. **Self-check 1** runs the same pipeline with zero mismatch and prog/read noise and drift scales set to 0,
    for both programming modes, all nodes, all six conditions, all datasets/splits/device seeds.
21. **Self-check 2** sets nu_mean_a = 0 (so mu = clip(0.0244, 0.049, 0.1) = 0.049), all nu_std parameters to 0,
    prog/read noise scales to 0, m20 mismatch, threshold programming; compares with hand-derived closed forms
    on unclipped cells, a = ((t_inference + 20)/20)^-0.049 (a_hat from the replica equals a to
    {ahat_err:.1e}). Closed forms (K0 = k_nom (g_hi - g_lo), eps = dV_TX/(V_TX,nom c_nom)):
    - dac: `theta_q + (1/a - 1) dV_TH/K0`
    - ideal: `theta_q`
    - none: `a theta_q - (1 - a) g_lo/(g_hi - g_lo) + (1 - a) dV_TH/K0`
    - supply: `theta_q + (g_in - dV_TH/k_nom) eps (1 - a)/((1 - eps)(g_hi - g_lo))`
22. **Output folder timestamp** is JST at run start.

## Output files

- `per_eval.csv`: the requested columns `dataset, split, device_seed, node, condition, t, a_hat, acc, integrity`
  (floats via `repr`), then extra columns `n_correct, n_test, n_nomatch, n_correct_tol, t_s, loss_pp`.
  `condition` is `prog` or `all`; `t` is `t0, 1h, 1d, 30d, 365d` (prog rows have `t = t0`); `t_s` = t_inference.
- `refs.csv`, `clipping.csv`, `summary.csv` (requested columns first, diagnostics after), `gates.json`,
  `checks.json`, `run_info.json` (versions, spec-file and script SHA-256), `e02_independent.py` (script copy).

## Diagnostics recorded beyond the request

- `checks.json`: structure check (ideal quantized CAM = quantized tree traversal, exactly one match per tree,
  matching leaf = traversal leaf, all datasets and splits), float64-threshold CAM vs sklearn predict agreement,
  extra check 3 (zero mismatch with the full PCM model: dac = supply = ideal), the normalization variant.
- `summary.csv`: mean path integrity, mean a_hat, mean and RMS effective-threshold error in LSB over unclipped
  cells (to compare with the protocol's section 5 estimates).
"""


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None, help="development output folder (must not exist)")
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--devs", type=int, default=None)
    args = ap.parse_args()

    t_start = time.time()
    start_jst = jst_now()
    c = load_config()
    if args.out is None:
        out_dir = os.path.join(RESULTS_DIR, "E0_2_independent_check_" + start_jst.strftime("%Y%m%d_%H%M"))
    else:
        out_dir = args.out
    os.makedirs(out_dir, exist_ok=False)
    print("output:", out_dir, flush=True)

    ds_list = [d for d in args.datasets.split(",") if d]
    n_dev = c["n_dev"] if args.devs is None else args.devs
    cellm = Cell(c["cell"], c["m20"]["r_in_V"])

    # ---- data, forests, CAMs, references -------------------------------------------------
    prepared = {}
    refs = []
    X_by_ds = {}
    for ds_idx, ds in enumerate(DATASETS):
        if ds not in ds_list:
            continue
        X, y = load_dataset(ds, c["synth"])
        X_by_ds[ds] = (X, y)
        for split in c["split_seeds"]:
            Xtr, Xte, ytr, yte = split_and_normalize(X, y, split)
            rf = RandomForestClassifier(n_estimators=c["n_estimators"], max_depth=c["max_depth"],
                                        random_state=split, n_jobs=1).fit(Xtr, ytr)
            cam = CAM(rf)
            theta_q = quantize(cam.thr, c["bits"])
            Xf = np.ascontiguousarray(Xte[:, cam.feat])
            ev = cam.evaluate(Xf, yte, theta_q)
            prepared[(ds_idx, ds, split)] = dict(rf=rf, cam=cam, theta_q=theta_q, Xte=Xte, yte=yte, Xf=Xf,
                                                 acc_ref=ev["acc"], integ_ref=ev["integrity"])
            refs.append(dict(dataset=ds, split=split, n_test=len(yte), n_constraints=cam.K, acc_ref=ev["acc"],
                             n_rows=cam.R, n_classes=len(cam.classes), integrity_ref=ev["integrity"],
                             n_correct_ref=ev["n_correct"], n_correct_ref_tol=ev["n_correct_tol"],
                             rows_without_constraints=int((cam.n_cons == 0).sum()),
                             n_theta_q_0=int((theta_q == 0).sum()), n_theta_q_1=int((theta_q == 1).sum()),
                             exact_value_threshold_ties=int(np.sum(Xf == theta_q))))
            print(f"  {ds} split {split}: n_test {len(yte)} K {cam.K} R {cam.R} acc_ref {ev['acc']:.6f}", flush=True)

    # ---- self-checks -----------------------------------------------------------------------
    checks = {}
    checks["structure_cam_equals_quantized_traversal"] = structure_check(prepared, c)
    checks["self_check_1_zero_mismatch_no_pcm_noise"] = self_check_1(prepared, c, cellm)
    checks["self_check_2_uniform_drift_closed_forms"] = self_check_2(prepared, c, cellm)
    checks["extra_check_3_zero_mismatch_nodes_equal"] = self_check_3(prepared, c, cellm)
    checks["diag_normalization_variant"] = normalization_variant_diagnostic(prepared, X_by_ds, c)
    for k, v in checks.items():
        vv = {kk: vv for kk, vv in v.items() if kk != "detail"}
        print("check", k, vv, flush=True)
    required_ok = (checks["structure_cam_equals_quantized_traversal"]["passed"]
                   and checks["self_check_1_zero_mismatch_no_pcm_noise"]["passed"]
                   and checks["self_check_2_uniform_drift_closed_forms"]["passed_required_dac_ideal"])
    with open(os.path.join(out_dir, "checks.json"), "w") as f:
        json.dump(checks, f, indent=2)
    if not required_ok:
        print("SELF-CHECK FAILED: stopping before the primary analysis", flush=True)
        sys.exit(2)

    # ---- primary analysis ------------------------------------------------------------------
    per_eval = []
    clipping = []
    theta_stats = {}      # (node, cond, t) -> [sum err, sum err^2, n] in LSB over unclipped cells
    LSB = 1.0 / (2 ** c["bits"] - 1)
    for (ds_idx, ds, split), P in prepared.items():
        cam, yte, Xf, thq = P["cam"], P["yte"], P["Xf"], P["theta_q"]
        for dev in range(n_dev):
            d = make_draws(c["base_seed"], ds_idx, split, dev, cam.K, len(c["times"]), c["n_rep"])
            conds, (dvth, dvtx, dbeta), G_target, raw = simulate_conditions(
                thq, d, cellm, c["m20"], c["pcm"], c["times"], c["n_rep"], "threshold")
            lowc = int((raw < 0.0).sum())
            highc = int((raw > cellm.g_max).sum())
            clipping.append(dict(dataset=ds, split=split, device_seed=dev, n_constraints=cam.K,
                                 clipped=lowc + highc, clipped_low=lowc, clipped_high=highc))
            unclipped = (raw >= 0.0) & (raw <= cellm.g_max)
            for (cond, lab, t_s, G_t, a_hat) in conds:
                for node in NODES:
                    th = cellm.theta_eff(node, G_t, a_hat, dvth, dvtx, dbeta)
                    ev = cam.evaluate(Xf, yte, th)
                    per_eval.append(dict(dataset=ds, split=split, device_seed=dev, node=node, condition=cond,
                                         t=lab, a_hat=a_hat, acc=ev["acc"], integrity=ev["integrity"],
                                         n_correct=ev["n_correct"], n_test=len(yte), n_nomatch=ev["n_nomatch"],
                                         n_correct_tol=ev["n_correct_tol"], t_s=t_s,
                                         loss_pp=100.0 * (P["acc_ref"] - ev["acc"])))
                    e = (th[unclipped] - thq[unclipped]) / LSB
                    s = theta_stats.setdefault((node, cond, lab), [0.0, 0.0, 0])
                    s[0] += float(e.sum())
                    s[1] += float((e * e).sum())
                    s[2] += int(e.size)
        print(f"  done {ds} split {split}  ({time.time() - t_start:.1f} s)", flush=True)

    # ---- summary ---------------------------------------------------------------------------
    cond_list = [("prog", "t0")] + [("all", lab) for lab, _ in c["times"]]
    ds_present = [d for d in DATASETS if d in ds_list]
    summary = []
    table = {}
    for node in NODES:
        for (cond, lab) in cond_list:
            per_ds = {}
            integ = []
            ahat = []
            for ds in ds_present:
                vals = [r["loss_pp"] for r in per_eval
                        if r["dataset"] == ds and r["node"] == node and r["condition"] == cond and r["t"] == lab]
                assert len(vals) == len(c["split_seeds"]) * n_dev
                per_ds[ds] = float(np.mean(vals))
                integ += [r["integrity"] for r in per_eval
                          if r["dataset"] == ds and r["node"] == node and r["condition"] == cond and r["t"] == lab]
                ahat += [r["a_hat"] for r in per_eval
                         if r["dataset"] == ds and r["node"] == node and r["condition"] == cond and r["t"] == lab]
            mean_loss = float(np.mean([per_ds[ds] for ds in ds_present]))
            worst_ds = max(ds_present, key=lambda k: per_ds[k])
            row = dict(node=node, condition=cond, t=lab, mean_loss_pp=mean_loss, max_loss_pp=per_ds[worst_ds],
                       worst_dataset=worst_ds)
            for ds in ds_present:
                row["loss_" + ds] = per_ds[ds]
            row["mean_integrity"] = float(np.mean(integ))
            row["mean_a_hat"] = float(np.mean(ahat))
            st = theta_stats[(node, cond, lab)]
            row["theta_err_mean_lsb_unclipped"] = st[0] / st[2]
            row["theta_err_rms_lsb_unclipped"] = float(np.sqrt(st[1] / st[2]))
            summary.append(row)
            table[(node, cond, lab)] = row

    def ml(node, cond, lab):
        return table[(node, cond, lab)]["mean_loss_pp"]

    d365_dac = ml("dac", "all", "365d") - ml("ideal", "all", "365d")
    d365_sup = ml("supply", "all", "365d") - ml("ideal", "all", "365d")
    d30_dac = ml("dac", "all", "30d") - ml("ideal", "all", "30d")
    d30_sup = ml("supply", "all", "30d") - ml("ideal", "all", "30d")
    gate_A = d365_dac >= 1.0
    if gate_A:
        classification = "GO_STRONG" if d365_sup >= 1.0 else "GO_WEAK"
    else:
        classification = "NO_GO"

    # tie-breaking sensitivity diagnostic: losses with tolerant argmax
    tol_diff = sum(1 for r in per_eval if r["n_correct_tol"] != r["n_correct"])
    # prog condition: the four nodes must give identical accuracy and integrity
    prog_groups = {}
    for r in per_eval:
        if r["condition"] == "prog":
            prog_groups.setdefault((r["dataset"], r["split"], r["device_seed"]), set()).add(
                (r["n_correct"], r["integrity"]))
    prog_same = all(len(v) == 1 for v in prog_groups.values())
    gates = dict(
        primary="mismatch m20, threshold programming, 4 bits, r_in 0.4 V",
        complete_design=(ds_present == DATASETS and n_dev == c["n_dev"]),
        n_evaluations=len(per_eval),
        mean_loss_pp={f"{n}@{cd}@{lb}": ml(n, cd, lb) for n in NODES for (cd, lb) in cond_list},
        max_loss_pp={f"{n}@{cd}@{lb}": table[(n, cd, lb)]["max_loss_pp"] for n in NODES for (cd, lb) in cond_list},
        diff_365d_dac_minus_ideal_pp=d365_dac,
        diff_365d_supply_minus_ideal_pp=d365_sup,
        diff_30d_dac_minus_ideal_pp=d30_dac,
        diff_30d_supply_minus_ideal_pp=d30_sup,
        diff_365d_none_minus_ideal_pp=ml("none", "all", "365d") - ml("ideal", "all", "365d"),
        diff_30d_none_minus_ideal_pp=ml("none", "all", "30d") - ml("ideal", "all", "30d"),
        gate_A=dict(rule="PASS if mean_loss(dac, all@365d) - mean_loss(ideal, all@365d) >= 1.0 pp",
                    value_pp=d365_dac, result="PASS" if gate_A else "FAIL"),
        classification=dict(rule="if gate A passes: GO_STRONG if mean_loss(supply, all@365d) - mean_loss(ideal, all@365d) >= 1.0 pp, otherwise GO_WEAK; NO_GO if gate A fails",
                            value_pp=d365_sup, result=classification),
        verdict=classification,
        tie_break_sensitivity=dict(evaluations_where_tolerant_argmax_changes_correct_count=tol_diff,
                                   tolerance=TIE_TOL),
        prog_condition_nodes_identical=prog_same,
    )

    # ---- write outputs ---------------------------------------------------------------------
    pe_cols = ["dataset", "split", "device_seed", "node", "condition", "t", "a_hat", "acc", "integrity",
               "n_correct", "n_test", "n_nomatch", "n_correct_tol", "t_s", "loss_pp"]
    with open(os.path.join(out_dir, "per_eval.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(pe_cols)
        for r in per_eval:
            w.writerow([repr(float(r[k])) if isinstance(r[k], float) else r[k] for k in pe_cols])
    ref_cols = list(refs[0].keys())
    with open(os.path.join(out_dir, "refs.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(ref_cols)
        for r in refs:
            w.writerow([repr(float(r[k])) if isinstance(r[k], float) else r[k] for k in ref_cols])
    clip_cols = ["dataset", "split", "device_seed", "n_constraints", "clipped", "clipped_low", "clipped_high"]
    with open(os.path.join(out_dir, "clipping.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(clip_cols)
        for r in clipping:
            w.writerow([r[k] for k in clip_cols])
    sum_cols = list(summary[0].keys())
    with open(os.path.join(out_dir, "summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(sum_cols)
        for r in summary:
            w.writerow([repr(float(r[k])) if isinstance(r[k], float) else r[k] for k in sum_cols])
    with open(os.path.join(out_dir, "gates.json"), "w") as f:
        json.dump(gates, f, indent=2)

    script_path = os.path.abspath(__file__)
    run_info = dict(
        start_jst=start_jst.strftime("%Y-%m-%d %H:%M:%S %Z"),
        end_jst=jst_now().strftime("%Y-%m-%d %H:%M:%S %Z"),
        runtime_s=time.time() - t_start,
        script=script_path, script_sha256=sha256_of(script_path),
        spec_files={p: sha256_of(p) for p in (PROTO1_PATH, CFG1_PATH, PROTO2_PATH, CFG2_PATH)},
        python=sys.version, numpy=np.__version__, sklearn=sklearn.__version__, scipy=scipy.__version__,
        platform=platform.platform(),
        cell=dict(c_nom=cellm.c_nom, beta_nom_uS_per_V=cellm.beta_nom, k_nom_V_per_uS=cellm.k_nom),
        datasets=ds_present, n_dev=n_dev,
        clipped_total=sum(r["clipped"] for r in clipping),
        constraints_total=sum(r["n_constraints"] for r in clipping),
    )
    with open(os.path.join(out_dir, "run_info.json"), "w") as f:
        json.dump(run_info, f, indent=2)
    with open(script_path) as f_in, open(os.path.join(out_dir, "e02_independent.py"), "w") as f_out:
        f_out.write(f_in.read())
    notes = SPEC_NOTES.format(
        start_jst=run_info["start_jst"], script_sha=run_info["script_sha256"],
        norm_same="yes" if checks["diag_normalization_variant"]["all_identical"] else "NO (see checks.json)",
        tol_diff=tol_diff, n_eval=len(per_eval),
        clip_total=run_info["clipped_total"], cons_total=run_info["constraints_total"],
        clip_low=sum(r["clipped_low"] for r in clipping), clip_high=sum(r["clipped_high"] for r in clipping),
        prog_same="yes" if prog_same else "NO",
        c_nom=cellm.c_nom, beta_nom=cellm.beta_nom, k_nom=cellm.k_nom,
        ahat_err=checks["self_check_2_uniform_drift_closed_forms"]["max_abs_ahat_minus_a"])
    with open(os.path.join(out_dir, "spec_notes.md"), "w") as f:
        f.write(notes)

    print("\nnode    cond  t     mean_loss  max_loss  worst")
    for r in summary:
        print(f"{r['node']:7s} {r['condition']:5s} {r['t']:5s} {r['mean_loss_pp']:9.4f} {r['max_loss_pp']:9.4f}  {r['worst_dataset']}")
    print(f"\nd365 dac-ideal {d365_dac:.4f}  supply-ideal {d365_sup:.4f}")
    print(f"d30  dac-ideal {d30_dac:.4f}  supply-ideal {d30_sup:.4f}")
    print("gate A", gates["gate_A"]["result"], " classification", classification)
    print("tie-break-sensitive evaluations:", tol_diff)
    print(f"runtime {time.time() - t_start:.1f} s")


if __name__ == "__main__":
    main()
