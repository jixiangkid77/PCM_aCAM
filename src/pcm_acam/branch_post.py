"""Quasi-static analysis of one aCAM bound branch from transistor I-V curves (instrument layer).

Branch (lower bound of a 6T2M cell, Li et al. 2020): PCM conductance G from the divider supply
V_SL to node X; divider nMOS from X to ground with its gate on the search input V_DL; X drives
the match-line pull-down. The pull-down is "on" when V_X >= V_sw, where V_sw is the gate voltage
at which the pull-down conducts I_sw with its drain at the precharged match line.

At the switching point V_X = V_sw, so Kirchhoff's current law gives
    G * (V_SL - V_sw) = I_div(V_DL, V_DS = V_sw)
and the bound is  B(G; V_SL) = I_div^{-1}( G (V_SL - V_sw) ),
evaluated on the divider's I-V curve taken at V_DS = V_sw (exact for a linear PCM conductance;
the gate of the pull-down draws no DC current).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass
class IVCurve:
    """Monotone drain current versus gate voltage (log-current interpolation)."""

    vg: np.ndarray
    ln_i: np.ndarray

    @classmethod
    def from_arrays(cls, vg: np.ndarray, current: np.ndarray, i_floor: float) -> "IVCurve":
        cur = np.asarray(current, dtype=float)
        if np.any(cur <= 0):
            # leakage sign flips can only occur deep below threshold; keep the part above i_floor
            pass
        ln = np.log(np.maximum(cur, 1e-30))
        # keep the strictly increasing tail that starts below i_floor
        d = np.diff(ln)
        bad = np.where(d <= 0)[0]
        start = 0 if bad.size == 0 else int(bad[-1]) + 1
        if cur[start] > i_floor:
            raise ValueError(f"I-V curve not monotone above the floor (start current {cur[start]:.3e} A)")
        return cls(vg=np.asarray(vg, dtype=float)[start:], ln_i=ln[start:])

    def v_at(self, current: np.ndarray) -> np.ndarray:
        """Gate voltage at which the drain current equals `current` (A); NaN outside the swept range."""
        lc = np.log(np.asarray(current, dtype=float))
        v = np.interp(lc, self.ln_i, self.vg, left=np.nan, right=np.nan)
        return v

    def i_at(self, v: np.ndarray) -> np.ndarray:
        return np.exp(np.interp(np.asarray(v, dtype=float), self.vg, self.ln_i, left=np.nan, right=np.nan))


def switching_voltage(vg: np.ndarray, current: np.ndarray, i_sw: float) -> float:
    c = IVCurve.from_arrays(vg, current, i_floor=i_sw / 10)
    v = float(c.v_at(np.array([i_sw]))[0])
    if not np.isfinite(v):
        raise ValueError("I_sw not reached in the swept range")
    return v


def g_of_x(x: np.ndarray, g_lo: float, g_hi: float) -> np.ndarray:
    return g_lo + np.asarray(x) * (g_hi - g_lo)


class Mapping:
    """Monotone input map x -> V_DL on a dense grid, with inverse by interpolation."""

    def __init__(self, x: np.ndarray, v: np.ndarray):
        # Points whose bound lies beyond the swept gate range (V_DL > supply) are not available to a
        # DAC: keep the finite part, which must be one contiguous block covering x in [0, 1].
        keep = np.isfinite(v)
        if not keep.any():
            raise ValueError("mapping has no finite point")
        idx = np.where(keep)[0]
        if idx[-1] - idx[0] + 1 != idx.size:
            raise ValueError("finite part of the mapping is not contiguous")
        x, v = x[keep], v[keep]
        if x[0] > 0 or x[-1] < 1:
            raise ValueError(f"mapping covers only x in [{x[0]:.4f}, {x[-1]:.4f}]")
        if not np.all(np.diff(v) > 0):
            raise ValueError("mapping is not strictly increasing")
        self.x, self.v = x, v

    def inverse(self, vdl: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        vdl = np.asarray(vdl, dtype=float)
        x = np.interp(vdl, self.v, self.x)
        oor = (vdl < self.v[0]) | (vdl > self.v[-1]) | ~np.isfinite(vdl)
        return x, oor


def bound(curve: IVCurve, g_us: np.ndarray, v_sl: float, v_sw: float) -> np.ndarray:
    """Bound voltage of a branch for conductances g_us (uS)."""
    return curve.v_at(np.asarray(g_us) * 1e-6 * (v_sl - v_sw))


def summarize(err_lsb: np.ndarray, valid: np.ndarray) -> Dict[str, float]:
    e = err_lsb[valid]
    if e.size == 0:
        return {"n": 0, "rms": np.nan, "mean": np.nan, "p95_abs": np.nan, "max_abs": np.nan}
    return {"n": int(e.size), "rms": float(np.sqrt(np.mean(e ** 2))), "mean": float(np.mean(e)),
            "p95_abs": float(np.percentile(np.abs(e), 95)), "max_abs": float(np.max(np.abs(e)))}
