"""Row-level behavioural model of a PCM analog-CAM match line, for forest-level simulation (instrument layer).

Branch types (one bound of a cell; PCM conductance G from the branch supply V_SL to the divider node, divider
nMOS from the node to ground with its gate on the shared data line DL):

  L6T   lower bound of the 6T2M cell: the divider node X drives the match-line (ML) pull-down directly
  UINV  upper bound of the 6T2M cell: node Y -> CMOS inverter -> pull-down
  LBUF  lower bound with a gain stage: node X -> inverter -> inverter -> pull-down

Switching point V_sp of a branch instance: the node voltage at which its pull-down gate reaches the pull-down's
switching voltage (L6T: V_sw itself; UINV: V_Y*; LBUF: V_X*). Quasi-static bound (E1-E3b):
G (V_SL - V_sp) = I_div(V_DL, V_DS = V_sp).

Match-line model (E3c): the ML is precharged to V_pre and judged "mismatch" if it falls through V_ref before
t_s(N). Inverse-time additivity: if the pull-down current factorises as f(V_G) h(V_DS), a row whose pull-downs
have gate voltages V_j crosses V_ref at t_c = t_s(N) (1 + (2N - 1) w0) / sum_j w_j(V_j), with
w_j(V) = I_D,j(V, V_DS = v_mid) / I_D,nom(V_sw, V_DS = v_mid) and w0 = w_nom(0). A row of N cells therefore
mismatches if a bound is violated or if
    sum_k w_k + (2N - n_c) w0 >= 1 + (2N - 1) w0        (n_c constrained bounds; wildcards at gate 0).

Leakage of a matched bound, per instance: the divider node follows from the load line
    G_k (V_SL - V) = I_k(V_DL, V_sp,k) * I_nom(V_G', V) / I_nom(V_G', V_sp,k),
where I_k(., V_sp,k) is the instance's divider curve (Monte-Carlo pool) and V_G' the nominal gate voltage that
conducts the same current at V_DS = V_sp,k (the V_DS dependence is borrowed from the nominal I-V surface at the
same operating point). The node then passes through the instance's own inverters (transfer stored around its
switching point) and its own pull-down discharge curve.

Monte-Carlo storage: divider and pull-down curves as the deviation of ln I from the nominal curve on a coarse
gate grid (smooth; Catmull-Rom interpolation), inverter transfers on a grid of input offsets around the
instance's switching point.

No experiment numbers in this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

from .trees import CamRows


# --------------------------------------------------------------------------------------------- interpolation
def _catmull_rom(table: np.ndarray, idx: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """Cubic (Catmull-Rom) interpolation of table[idx, :] at fractional column positions pos (clamped)."""
    n = table.shape[1]
    pos = np.clip(pos, 0.0, n - 1.000001)
    i1 = pos.astype(np.int64)
    t = pos - i1
    i0 = np.maximum(i1 - 1, 0)
    i2 = np.minimum(i1 + 1, n - 1)
    i3 = np.minimum(i1 + 2, n - 1)
    p0, p1, p2, p3 = (table[idx, i].astype(float) for i in (i0, i1, i2, i3))
    return 0.5 * (2 * p1 + (p2 - p0) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t + (3 * p1 - p0 - 3 * p2 + p3) * t ** 3)


def _linear_rows(table: np.ndarray, idx: np.ndarray, pos: np.ndarray) -> np.ndarray:
    n = table.shape[1]
    pos = np.clip(pos, 0.0, n - 1.000001)
    i0 = pos.astype(np.int64)
    f = pos - i0
    return table[idx, i0] * (1.0 - f) + table[idx, i0 + 1] * f


# --------------------------------------------------------------------------------------------- I-V surface
class Surface:
    """Nominal divider drain current I(V_G, V_DS) on a grid; ln I interpolated along V_G, I along V_DS."""

    def __init__(self, vg: np.ndarray, vds: np.ndarray, ln_i: np.ndarray):
        self.vg = np.asarray(vg, float)
        self.vds = np.asarray(vds, float)
        self.ln_i = np.asarray(ln_i, float)      # (n_vds, n_vg)
        self.dg = float(self.vg[1] - self.vg[0])
        self.dd = float(self.vds[1] - self.vds[0])

    @classmethod
    def from_current(cls, vg, vds, cur) -> "Surface":
        return cls(vg, vds, np.log(np.maximum(np.asarray(cur, float), 1e-30)))

    def current(self, vgate, vds) -> np.ndarray:
        vgate, vds = np.broadcast_arrays(np.asarray(vgate, float), np.asarray(vds, float))
        gi = np.clip((vgate - self.vg[0]) / self.dg, 0.0, self.vg.size - 1.000001)
        g0 = gi.astype(np.int64)
        fg = gi - g0
        di = np.clip((vds - self.vds[0]) / self.dd, 0.0, self.vds.size - 1.000001)
        d0 = di.astype(np.int64)
        fd = di - d0
        l00, l01 = self.ln_i[d0, g0], self.ln_i[d0, g0 + 1]
        l10, l11 = self.ln_i[d0 + 1, g0], self.ln_i[d0 + 1, g0 + 1]
        i0 = np.exp(l00 + fg * (l01 - l00))
        i1 = np.exp(l10 + fg * (l11 - l10))
        return (1.0 - fd) * i0 + fd * i1

    def gate_for_current(self, current, vds, iters: int = 32) -> np.ndarray:
        current, vds = np.broadcast_arrays(np.asarray(current, float), np.asarray(vds, float))
        lo = np.full(current.shape, self.vg[0])
        hi = np.full(current.shape, self.vg[-1])
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            up = self.current(mid, vds) < current
            lo = np.where(up, mid, lo)
            hi = np.where(up, hi, mid)
        return 0.5 * (lo + hi)

    def node(self, g_us, v_sl, v_dl, iters: int = 40) -> np.ndarray:
        """Nominal divider-node voltage: G (V_SL - V) = I(V_DL, V) (unique root on [0, V_SL])."""
        g_us, v_sl, v_dl = np.broadcast_arrays(np.asarray(g_us, float), np.asarray(v_sl, float),
                                               np.asarray(v_dl, float))
        g = g_us * 1e-6
        lo = np.zeros(g.shape)
        hi = np.array(v_sl, float)
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            pos = g * (v_sl - mid) - self.current(v_dl, mid) > 0
            lo = np.where(pos, mid, lo)
            hi = np.where(pos, hi, mid)
        return 0.5 * (lo + hi)


# --------------------------------------------------------------------------------------------- 1-D curves
class Curve1D:
    """Monotone ln I(V_G) on a uniform grid (nominal divider at a fixed V_DS)."""

    def __init__(self, vg: np.ndarray, ln_i: np.ndarray):
        self.vg = np.asarray(vg, float)
        self.ln_i = np.maximum.accumulate(np.asarray(ln_i, float))

    def i_at(self, v) -> np.ndarray:
        return np.exp(np.interp(np.asarray(v, float), self.vg, self.ln_i))

    def v_at(self, current) -> np.ndarray:
        """Gate voltage for the drain current; NaN below/above the curve."""
        lc = np.log(np.maximum(np.asarray(current, float), 1e-300))
        v = np.interp(lc, self.ln_i, self.vg)
        return np.where((lc < self.ln_i[0]) | (lc > self.ln_i[-1]), np.nan, v)


@dataclass
class Pool:
    """Divider instances of one branch type and design.

    v_sp[k]: switching point of instance k (k = 0 nominal); the instance's divider curve is taken at V_DS = v_sp[k].
    dln[k, j]: ln I_k - ln I_0 on the coarse gate grid vg_c (I_0: nominal divider at V_DS = v_sp[0], given on the
    fine grid by `nominal`).
    """

    v_sp: np.ndarray
    vg_c: np.ndarray
    dln: np.ndarray
    nominal: Curve1D

    def ln_i(self, idx: np.ndarray, v: np.ndarray) -> np.ndarray:
        v = np.asarray(v, float)
        base = np.interp(v, self.nominal.vg, self.nominal.ln_i)
        pos = (v - self.vg_c[0]) / (self.vg_c[1] - self.vg_c[0])
        return base + _catmull_rom(self.dln, np.asarray(idx), pos)

    def i_at(self, idx, v) -> np.ndarray:
        return np.exp(self.ln_i(np.asarray(idx), np.asarray(v, float)))

    def v_at(self, idx, current, iters: int = 40) -> np.ndarray:
        """Gate voltage at which instance idx conducts `current`; -inf below and +inf above the window."""
        idx = np.asarray(idx)
        y = np.log(np.maximum(np.asarray(current, float), 1e-300))
        lo = np.full(y.shape, self.vg_c[0])
        hi = np.full(y.shape, self.vg_c[-1])
        below = y < self.ln_i(idx, lo)
        above = y > self.ln_i(idx, hi)
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            up = self.ln_i(idx, mid) < y
            lo = np.where(up, mid, lo)
            hi = np.where(up, hi, mid)
        v = 0.5 * (lo + hi)
        v = np.where(below, -np.inf, v)
        return np.where(above, np.inf, v)


def compress_curves(vg: np.ndarray, cur: np.ndarray, window: Tuple[float, float],
                    coarse_step: float) -> Tuple[Curve1D, np.ndarray, np.ndarray]:
    """Nominal fine curve (row 0) and the per-instance deviations of ln I on a coarse grid inside `window`."""
    ln = np.maximum.accumulate(np.log(np.maximum(cur, 1e-30)), axis=1)
    nominal = Curve1D(vg, ln[0])
    vg_c = np.arange(window[0], window[1] + 1e-12, coarse_step)
    dln = np.vstack([np.interp(vg_c, vg, ln[k]) - np.interp(vg_c, vg, ln[0]) for k in range(ln.shape[0])])
    return nominal, vg_c, dln.astype(np.float32)


# --------------------------------------------------------------------------------------------- switching chain
@dataclass
class Chain:
    """Node -> (inverters) -> pull-down of one branch type, per instance (k = 0 nominal).

    pd: discharge curves of the instance pull-downs, ln I_D(V, V_DS = v_mid) = pd_nom(V) + pd_dln[k](V);
    w = I_D,k / I_D,nom(V_sw,nom). Inverters: output at (switching input + offset) for offsets rel_grid.
    """

    kind: str
    v_sp: np.ndarray
    pd_vg: np.ndarray
    pd_ln_nom: np.ndarray
    pd_vg_c: np.ndarray
    pd_dln: np.ndarray
    ln_ref: float
    rel_grid: Optional[np.ndarray] = None
    t_first: Optional[np.ndarray] = None
    v_mid_in: Optional[np.ndarray] = None      # LBUF: input of the second inverter at the switching point
    t_second: Optional[np.ndarray] = None

    def _inv(self, table: np.ndarray, idx: np.ndarray, offset: np.ndarray) -> np.ndarray:
        pos = (offset - self.rel_grid[0]) / (self.rel_grid[1] - self.rel_grid[0])
        return _linear_rows(table, idx, pos)

    def gate(self, idx: np.ndarray, node_v: np.ndarray) -> np.ndarray:
        if self.kind == "L6T":
            return node_v
        if self.kind == "UINV":
            return self._inv(self.t_first, idx, node_v - self.v_sp[idx])
        if self.kind == "LBUF":
            w = self._inv(self.t_first, idx, node_v - self.v_sp[idx])
            return self._inv(self.t_second, idx, w - self.v_mid_in[idx])
        raise ValueError(self.kind)

    def w(self, idx: np.ndarray, gate_v: np.ndarray) -> np.ndarray:
        base = np.interp(gate_v, self.pd_vg, self.pd_ln_nom)
        pos = (gate_v - self.pd_vg_c[0]) / (self.pd_vg_c[1] - self.pd_vg_c[0])
        return np.exp(base + _catmull_rom(self.pd_dln, idx, pos) - self.ln_ref)

    def w0(self) -> float:
        return float(np.exp(np.interp(0.0, self.pd_vg, self.pd_ln_nom) - self.ln_ref))


def solve_nodes(surf: Surface, pool: Pool, idx: np.ndarray, g_us: np.ndarray, v_sl: np.ndarray,
                v_dl: np.ndarray, iters: int = 32) -> np.ndarray:
    """Per-instance divider-node voltage (load line with the instance curve and the nominal V_DS dependence)."""
    v_sp = pool.v_sp[idx]
    i1 = pool.i_at(idx, v_dl)
    vge = surf.gate_for_current(i1, v_sp, iters)
    i_ref = surf.current(vge, v_sp)
    g = np.asarray(g_us, float) * 1e-6
    v_sl = np.broadcast_to(np.asarray(v_sl, float), g.shape)
    lo = np.zeros(g.shape)
    hi = np.array(v_sl, float)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        pos = g * (v_sl - mid) - i1 * surf.current(vge, mid) / i_ref > 0
        lo = np.where(pos, mid, lo)
        hi = np.where(pos, hi, mid)
    return 0.5 * (lo + hi)


# --------------------------------------------------------------------------------------------- maps and bounds
def g_of_x(x, g_lo: float, g_hi: float) -> np.ndarray:
    return g_lo + np.asarray(x, float) * (g_hi - g_lo)


def map_voltage(map_curve: Curve1D, x, map_factor: float, d_v: float, g_lo: float, g_hi: float) -> np.ndarray:
    """Input map: V_DL(x) = I_0^{-1}(m G(x) D) (M0 for m = 1, Ma for m = a); 0 V where G(x) <= 0; clamped to the
    swept gate range above."""
    g = g_of_x(x, g_lo, g_hi)
    cur = np.maximum(g, 1e-9) * 1e-6 * map_factor * d_v
    lc = np.log(cur)
    v = np.interp(lc, map_curve.ln_i, map_curve.vg)
    v = np.where(lc < map_curve.ln_i[0], 0.0, v)
    return np.where(g <= 0, 0.0, v)


def program_targets(pool: Pool, idx: np.ndarray, v_target: np.ndarray, v_sl_nom: float, g_max: float):
    """Threshold verify: G_k = I_div,k(V_target) / (V_SL,nom - V_sp,k) in uS, clipped to [0, g_max]."""
    g = pool.i_at(idx, v_target) / (v_sl_nom - pool.v_sp[idx]) * 1e6
    clipped = (g > g_max) | ~np.isfinite(g)
    return np.clip(np.where(np.isfinite(g), g, g_max), 0.0, g_max), clipped


def effective_x(pool: Pool, idx: np.ndarray, g_us: np.ndarray, v_sl: float, map_curve: Curve1D,
                map_factor: float, d_v: float, g_lo: float, g_hi: float, x_cap: float = 10.0) -> np.ndarray:
    """Input-referred bound: V_DL* = I_div,k^{-1}(G (V_SL - V_sp,k)); x_eff = G^{-1}(I_0(V_DL*) / (m D))."""
    cur = np.asarray(g_us, float) * 1e-6 * (v_sl - pool.v_sp[idx])
    v_b = pool.v_at(idx, cur)
    finite = np.isfinite(v_b)
    i0 = map_curve.i_at(np.where(finite, v_b, 0.0))
    x = (i0 / (map_factor * d_v) * 1e6 - g_lo) / (g_hi - g_lo)
    x = np.where(finite, x, np.where(v_b > 0, x_cap, -x_cap))
    return np.clip(x, -x_cap, x_cap)


# --------------------------------------------------------------------------------------------- row decision
def violation_match(x: np.ndarray, rows: CamRows, theta_eff: np.ndarray) -> np.ndarray:
    """Match matrix (N x R) from bound violations only (lower: x > theta; upper: x <= theta)."""
    xv = x[:, rows.cons_feat]
    sat = np.where(rows.cons_kind[None, :] == 0, xv > theta_eff[None, :], xv <= theta_eff[None, :])
    viol = (~sat).astype(np.int32)
    match = np.ones((x.shape[0], rows.row_tree.shape[0]), dtype=bool)
    has = rows.row_ncons > 0
    if has.any():
        sums = np.add.reduceat(viol, rows.row_start[has], axis=1)
        match[:, has] = sums == 0
    return match


@dataclass
class BranchState:
    """Everything needed to evaluate the leakage of the bounds of one kind (lower or upper) of a forest."""

    surf: Surface
    pool: Pool
    chain: Chain
    idx: np.ndarray          # per constraint (full length; only entries of this kind are used)
    g_us: np.ndarray         # PCM conductance now, per constraint
    v_sl: float
    map_curve: Curve1D
    map_factor: float
    d_v: float
    g_lo: float
    g_hi: float
    u_cut: float             # beyond this distance inside the bound the leakage is taken as w(0)
    extra: dict = field(default_factory=dict)

    def leak(self, k: np.ndarray, xv: np.ndarray, u: np.ndarray) -> np.ndarray:
        w = np.full(k.shape, np.nan)
        near = u < self.u_cut
        if near.any():
            kk = k[near]
            idx = self.idx[kk]
            v_dl = map_voltage(self.map_curve, xv[near], self.map_factor, self.d_v, self.g_lo, self.g_hi)
            node = solve_nodes(self.surf, self.pool, idx, self.g_us[kk], np.full(kk.shape, self.v_sl), v_dl)
            w[near] = self.chain.w(idx, self.chain.gate(idx, node))
        if (~near).any():
            w[~near] = self.chain.w(self.idx[k[~near]], np.zeros(int((~near).sum())))
        return w


def crosstalk_match(x: np.ndarray, rows: CamRows, theta_eff: np.ndarray, match_v: np.ndarray,
                    lower: BranchState, upper: BranchState, n_feat: int, w0: float):
    """Remove the matches whose leakage sum reaches the sensing threshold. Returns (match, sums of the
    violation-matched pairs, (sample, row) indices of those pairs, per-pair contributions (k, w))."""
    n_idx, r_idx = np.nonzero(match_v)
    nc = rows.row_ncons[r_idx]
    pair = np.repeat(np.arange(r_idx.size), nc)
    offs = np.arange(int(nc.sum())) - np.repeat(np.cumsum(nc) - nc, nc)
    k = rows.row_start[r_idx][pair] + offs
    xv = x[n_idx[pair], rows.cons_feat[k]]
    th = theta_eff[k]
    is_lo = rows.cons_kind[k] == 0
    u = np.where(is_lo, xv - th, th - xv)
    w = np.empty(u.shape)
    if is_lo.any():
        w[is_lo] = lower.leak(k[is_lo], xv[is_lo], u[is_lo])
    if (~is_lo).any():
        w[~is_lo] = upper.leak(k[~is_lo], xv[~is_lo], u[~is_lo])
    s = np.bincount(pair, weights=w, minlength=r_idx.size) + (2 * n_feat - nc) * w0
    fail = s >= 1.0 + (2 * n_feat - 1) * w0
    match = match_v.copy()
    match[n_idx[fail], r_idx[fail]] = False
    return match, s, (n_idx, r_idx), (pair, k, w)


def vote(match: np.ndarray, rows: CamRows, classes: np.ndarray):
    """Predicted labels (-1 when no row matches), leaf-loss rate and multi-match rate per (sample, tree)."""
    score = match.astype(float) @ rows.row_value
    pred = np.where(score.max(axis=1) > 0, classes[score.argmax(axis=1)], -1)
    per_tree = np.zeros((match.shape[0], rows.n_trees), dtype=np.int32)
    for t in range(rows.n_trees):
        per_tree[:, t] = match[:, rows.row_tree == t].sum(axis=1)
    return pred, float(np.mean(per_tree == 0)), float(np.mean(per_tree > 1))
