"""Datasets, random forests, and their conversion into analog-CAM rows.

A decision path in a tree is a conjunction of interval constraints, one
interval per feature. Each leaf becomes one CAM row: for every feature used
on the path the row stores a lower bound (match if x > L) and/or an upper
bound (match if x <= U), which are the semantics of the sklearn split
``x <= threshold``. Features not used on the path are wildcards.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from sklearn.datasets import (
    load_breast_cancer,
    load_digits,
    load_iris,
    load_wine,
    make_classification,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def load_dataset(name: str) -> Tuple[np.ndarray, np.ndarray]:
    if name == "iris":
        d = load_iris()
    elif name == "wine":
        d = load_wine()
    elif name == "breast_cancer":
        d = load_breast_cancer()
    elif name == "digits":
        d = load_digits()
    elif name == "synthetic":
        x, y = make_classification(
            n_samples=5000, n_features=20, n_informative=10, n_redundant=4, n_classes=4,
            n_clusters_per_class=2, class_sep=1.0, random_state=20261012,
        )
        return x.astype(float), y.astype(int)
    else:
        raise ValueError(name)
    return d.data.astype(float), d.target.astype(int)


def split_and_normalize(x: np.ndarray, y: np.ndarray, split_seed: int):
    """Stratified 70/30 split; min-max on the training split; test clipped to [0, 1]."""
    x_tr, x_te, y_tr, y_te = train_test_split(
        x, y, test_size=0.3, stratify=y, random_state=split_seed
    )
    lo = x_tr.min(axis=0)
    span = x_tr.max(axis=0) - lo
    span[span == 0] = 1.0
    x_tr = (x_tr - lo) / span
    x_te = np.clip((x_te - lo) / span, 0.0, 1.0)
    return x_tr, x_te, y_tr, y_te


def train_forest(x_tr: np.ndarray, y_tr: np.ndarray, split_seed: int) -> RandomForestClassifier:
    rf = RandomForestClassifier(n_estimators=16, max_depth=6, random_state=split_seed, n_jobs=1)
    rf.fit(x_tr, y_tr)
    return rf


@dataclass
class CamRows:
    """Flat constraint list of all rows of a forest.

    cons_row[k], cons_feat[k], cons_kind[k] (0 = lower, 1 = upper),
    cons_thr[k] (float threshold in normalized units) describe constraint k.
    row_tree[r] and row_value[r, c] describe row r. Constraints are sorted by
    row; row_start[r] is the index of the first constraint of row r and
    row_ncons[r] how many it has (0 = the row always matches).
    """

    cons_row: np.ndarray
    cons_feat: np.ndarray
    cons_kind: np.ndarray
    cons_thr: np.ndarray
    row_tree: np.ndarray
    row_value: np.ndarray
    row_start: np.ndarray
    row_ncons: np.ndarray
    n_trees: int


def forest_to_rows(rf: RandomForestClassifier) -> CamRows:
    cons_row: List[int] = []
    cons_feat: List[int] = []
    cons_kind: List[int] = []
    cons_thr: List[float] = []
    row_tree: List[int] = []
    row_value: List[np.ndarray] = []
    n_classes = len(rf.classes_)
    for t_idx, est in enumerate(rf.estimators_):
        tree = est.tree_
        stack = [(0, {}, {})]  # node, lower bounds {feat: thr}, upper bounds {feat: thr}
        while stack:
            node, lows, ups = stack.pop()
            left, right = tree.children_left[node], tree.children_right[node]
            if left == -1:  # leaf
                r = len(row_tree)
                row_tree.append(t_idx)
                v = tree.value[node, 0, :].astype(float)
                # forest trees are fit on the encoded labels of all classes, so the
                # value columns follow rf.classes_
                assert v.shape[0] == n_classes
                v = v / v.sum() if v.sum() > 0 else np.full(n_classes, 1.0 / n_classes)
                row_value.append(v)
                for f, thr in lows.items():
                    cons_row.append(r); cons_feat.append(f); cons_kind.append(0); cons_thr.append(thr)
                for f, thr in ups.items():
                    cons_row.append(r); cons_feat.append(f); cons_kind.append(1); cons_thr.append(thr)
                continue
            f = int(tree.feature[node])
            thr = float(tree.threshold[node])
            ups_l = dict(ups)
            ups_l[f] = min(ups.get(f, np.inf), thr)        # left: x <= thr
            lows_r = dict(lows)
            lows_r[f] = max(lows.get(f, -np.inf), thr)     # right: x > thr
            stack.append((right, lows_r, dict(ups)))
            stack.append((left, dict(lows), ups_l))
    cons_row_a = np.asarray(cons_row, dtype=int)
    order = np.argsort(cons_row_a, kind="stable")
    n_rows = len(row_tree)
    ncons = np.bincount(cons_row_a, minlength=n_rows)
    start = np.concatenate([[0], np.cumsum(ncons)[:-1]])
    return CamRows(
        cons_row=cons_row_a[order],
        cons_feat=np.asarray(cons_feat, dtype=int)[order],
        cons_kind=np.asarray(cons_kind, dtype=int)[order],
        cons_thr=np.asarray(cons_thr, dtype=float)[order],
        row_tree=np.asarray(row_tree, dtype=int),
        row_value=np.vstack(row_value),
        row_start=start.astype(int),
        row_ncons=ncons.astype(int),
        n_trees=len(rf.estimators_),
    )


def quantize(thr: np.ndarray, bits: int) -> np.ndarray:
    """theta = round(thr * (2^B - 1)) / (2^B - 1), thresholds clipped to [0, 1] first."""
    q = (1 << bits) - 1
    return np.round(np.clip(thr, 0.0, 1.0) * q) / q


def traverse_quantized(rf: RandomForestClassifier, x: np.ndarray, bits: int) -> np.ndarray:
    """Reference: walk every tree with quantized split thresholds; returns summed leaf values."""
    n_classes = len(rf.classes_)
    score = np.zeros((x.shape[0], n_classes))
    for est in rf.estimators_:
        tree = est.tree_
        thr_q = quantize(tree.threshold.astype(float), bits)
        for n in range(x.shape[0]):
            node = 0
            while tree.children_left[node] != -1:
                f = tree.feature[node]
                node = tree.children_left[node] if x[n, f] <= thr_q[node] else tree.children_right[node]
            v = tree.value[node, 0, :].astype(float)
            score[n] += v / v.sum()
    return score


def dataset_bundle(name: str, split_seed: int) -> Dict[str, object]:
    x, y = load_dataset(name)
    x_tr, x_te, y_tr, y_te = split_and_normalize(x, y, split_seed)
    rf = train_forest(x_tr, y_tr, split_seed)
    return {"x_te": x_te, "y_te": y_te, "rf": rf, "rows": forest_to_rows(rf),
            "acc_float": float(np.mean(rf.predict(x_te) == y_te))}
