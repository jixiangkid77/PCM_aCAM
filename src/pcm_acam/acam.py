"""PCM analog CAM model: threshold-to-device mapping, device states, drift
tracking, effective thresholds, matching and voting.

Threshold mapping (first-order model of the 6T2M-type cell): each bound is
set by a memristor-transistor voltage divider, so the threshold voltage is
affine in the device conductance and increases with it for both bounds. The
input DAC maps the normalized feature x in [0, 1] onto the same affine
range, so in normalized units:

  absolute scheme:  theta = (G / a_hat - g_lo) / (g_hi - g_lo)
  ratio scheme:     theta = (G1 / (G1 + G2) - r_lo) / (r_hi - r_lo)

a_hat is the global drift factor applied by the tracking bias (1 without
tracking). In the ratio scheme the bound compares the input with a
PCM-PCM divider, so the mean drift cancels without any tracking.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from .pcm_model import (
    PCMParams,
    apply_drift_and_read_noise,
    apply_programming_noise,
    drift_coefficients,
)
from .trees import CamRows


@dataclass(frozen=True)
class Scheme:
    name: str
    kind: str  # "absolute" or "ratio"
    devices: int
    g_lo: float = 0.0
    g_hi: float = 25.0
    g_sum: float = 25.0
    r_lo: float = 1.0 / 3.0
    r_hi: float = 2.0 / 3.0

    @classmethod
    def from_cfg(cls, name: str, d: dict) -> "Scheme":
        return cls(
            name=name, kind=d["kind"], devices=int(d["devices_per_bound"]),
            g_lo=float(d.get("g_lo", 0.0)), g_hi=float(d.get("g_hi", 25.0)),
            g_sum=float(d.get("g_sum", 25.0)), r_lo=float(d.get("r_lo", 1.0 / 3.0)),
            r_hi=float(d.get("r_hi", 2.0 / 3.0)),
        )

    def replica_target(self) -> float:
        return 0.5 * (self.g_lo + self.g_hi)


def device_targets(theta: np.ndarray, sch: Scheme) -> np.ndarray:
    """Programmed targets (uS), shape (K, devices)."""
    if sch.kind == "absolute":
        g = sch.g_lo + theta * (sch.g_hi - sch.g_lo)
        return np.repeat(g[:, None], sch.devices, axis=1)
    if sch.kind == "ratio":
        r = sch.r_lo + theta * (sch.r_hi - sch.r_lo)
        return np.stack([r * sch.g_sum, (1.0 - r) * sch.g_sum], axis=1)
    raise ValueError(sch.kind)


def device_states(
    targets: np.ndarray, draws: dict, condition: str, t_idx: int, t_inf: float, p: PCMParams
) -> np.ndarray:
    """Effective conductances (uS) of all devices, shape like targets.

    condition: "ideal" (targets), "prog" (programming noise only, t0) or
    "all" (programming noise, drift and accumulated read noise at t_inf).
    draws["z_prog"], draws["z_nu"]: (K, 2); draws["z_read"]: (n_times, K, 2).
    """
    n_dev = targets.shape[1]
    if condition == "ideal":
        return targets.copy()
    g_prog = apply_programming_noise(targets, draws["z_prog"][:, :n_dev], p)
    if condition == "prog":
        return g_prog
    if condition == "all":
        nu = drift_coefficients(targets, draws["z_nu"][:, :n_dev], p)
        return apply_drift_and_read_noise(g_prog, nu, t_inf, draws["z_read"][t_idx][:, :n_dev], p)
    raise ValueError(condition)


def replica_factor(
    sch: Scheme, draws: dict, condition: str, t_idx: int, t_inf: float, p: PCMParams
) -> float:
    """a_hat: mean replica conductance at t over its mean right after programming."""
    if condition in ("ideal", "prog"):
        return 1.0
    g_t = np.full(draws["z_rep_prog"].shape[0], sch.replica_target())
    g_prog = apply_programming_noise(g_t, draws["z_rep_prog"], p)
    nu = drift_coefficients(g_t, draws["z_rep_nu"], p)
    g_now = apply_drift_and_read_noise(g_prog, nu, t_inf, draws["z_rep_read"][t_idx], p)
    return float(np.mean(g_now) / np.mean(g_prog))


def effective_theta(g_eff: np.ndarray, sch: Scheme, a_hat: float = 1.0) -> np.ndarray:
    if sch.kind == "absolute":
        g = g_eff.mean(axis=1)
        return (g / a_hat - sch.g_lo) / (sch.g_hi - sch.g_lo)
    if sch.kind == "ratio":
        tot = g_eff[:, 0] + g_eff[:, 1]
        r = np.where(tot > 0, g_eff[:, 0] / np.where(tot > 0, tot, 1.0), 0.5)
        return (r - sch.r_lo) / (sch.r_hi - sch.r_lo)
    raise ValueError(sch.kind)


def cam_match(x: np.ndarray, rows: CamRows, theta_eff: np.ndarray) -> np.ndarray:
    """Boolean match matrix (N samples x R rows)."""
    xv = x[:, rows.cons_feat]                                   # (N, K)
    sat = np.where(rows.cons_kind[None, :] == 0, xv > theta_eff[None, :], xv <= theta_eff[None, :])
    viol = (~sat).astype(np.int32)
    n_rows = rows.row_tree.shape[0]
    match = np.ones((x.shape[0], n_rows), dtype=bool)
    has = rows.row_ncons > 0
    if has.any():
        sums = np.add.reduceat(viol, rows.row_start[has], axis=1)
        match[:, has] = sums == 0
    return match


def cam_predict(
    x: np.ndarray, rows: CamRows, theta_eff: np.ndarray
) -> Tuple[np.ndarray, float]:
    """Predicted class index (-1 when no row of any tree matches) and path integrity."""
    match = cam_match(x, rows, theta_eff)
    score = match.astype(float) @ rows.row_value
    pred = np.where(score.max(axis=1) > 0, score.argmax(axis=1), -1)
    per_tree = np.zeros((x.shape[0], rows.n_trees), dtype=np.int32)
    for t in range(rows.n_trees):
        per_tree[:, t] = match[:, rows.row_tree == t].sum(axis=1)
    integrity = float(np.mean(per_tree == 1))
    return pred, integrity


def make_draws(seed_seq, n_cons: int, n_times: int, n_rep: int) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed_seq)
    return {
        "z_prog": rng.standard_normal((n_cons, 2)),
        "z_nu": rng.standard_normal((n_cons, 2)),
        "z_read": rng.standard_normal((n_times, n_cons, 2)),
        "z_rep_prog": rng.standard_normal(n_rep),
        "z_rep_nu": rng.standard_normal(n_rep),
        "z_rep_read": rng.standard_normal((n_times, n_rep)),
    }
