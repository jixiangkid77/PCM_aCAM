"""6T2M analog-CAM cell model with transistor mismatch and global drift-compensation nodes.

Bound equation (C. Li et al., "Analog content-addressable memories with memristors,"
Nat. Commun. 11, 1638 (2020), Eq. 1; divider transistor in the triode regime):

    V_bound = V_TH + G * (V_SL / V_TX - 1) / beta

    V_TH   threshold of the divider transistor driven by the data line (additive, does not drift)
    V_TX   threshold of the match-line pull-down (lower bound) or of the inverter (upper bound)
    V_SL   divider supply (SL_hi)
    beta   dG_T / dV_DL of the divider transistor
    G      memristor conductance (here PCM; drifts)

Nominal design: c_nom = V_SL / V_TX,nom - 1, beta_nom = c_nom * (g_hi - g_lo) / r_in and
k_nom = c_nom / beta_nom = r_in / (g_hi - g_lo) (volts per uS). The input DAC maps x in [0, 1]
onto V_DL = V_TH,nom + s * k_nom * (g_lo + x * (g_hi - g_lo)), so the nominal cell maps a
conductance G onto the normalized threshold (G - g_lo) / (g_hi - g_lo).

Mismatch (per bound device, i.e. per CAM constraint):
    V_TH,i = V_TH,nom + dV_TH,i ;  V_TX,i = V_TX,nom + dV_TX,i ;  beta_i = beta_nom * (1 + d_beta,i)

Programming:
    "threshold"    verify through the search path: G is set so that the cell's own bound equals
                   the target at t0 (absorbs the mismatch); clipped to [0, g_max]
    "conductance"  verify the conductance: G is set to the nominal value (mismatch stays)

Global drift-compensation nodes (a_hat from the replica cells, 1 when there is no drift):
    "none"    s = 1, V_SL nominal
    "dac"     s = a_hat: the DAC gain is scaled around V_TH,nom
    "supply"  V_SL(t) = V_TX,nom * (1 + c_nom / a_hat): the divider supply restores the nominal cell
    "ideal"   G -> G / a_hat inside the bound: a mismatch-invariant global scaling (reference only)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

NODES = ("none", "dac", "supply", "ideal")


@dataclass(frozen=True)
class CellParams:
    v_sl: float     # V
    v_th: float     # V, nominal divider-transistor threshold
    v_tx: float     # V, nominal pull-down / inverter threshold
    g_lo: float     # uS, conductance mapped onto x = 0
    g_hi: float     # uS, conductance mapped onto x = 1
    g_max: float    # uS, largest programmable conductance
    r_in: float     # V, input voltage span for x in [0, 1]

    @property
    def dg(self) -> float:
        return self.g_hi - self.g_lo

    def nominal(self) -> Tuple[float, float, float]:
        """(c_nom, beta_nom [uS/V], k_nom [V/uS])."""
        c = self.v_sl / self.v_tx - 1.0
        beta = c * self.dg / self.r_in
        return c, beta, c / beta


def mismatch(z: Dict[str, np.ndarray], sigma_vth: float, sigma_vtx: float, sigma_beta: float) -> Dict[str, np.ndarray]:
    """Scale shared standard-normal draws into per-constraint mismatch (volts, relative)."""
    return {"d_vth": sigma_vth * z["z_vth"], "d_vtx": sigma_vtx * z["z_vtx"], "d_beta": sigma_beta * z["z_beta"]}


def cell_k(cp: CellParams, v_sl: float, mm: Dict[str, np.ndarray]) -> np.ndarray:
    """Per-cell k_i = (V_SL / V_TX,i - 1) / beta_i in V per uS."""
    _, beta_nom, _ = cp.nominal()
    return (v_sl / (cp.v_tx + mm["d_vtx"]) - 1.0) / (beta_nom * (1.0 + mm["d_beta"]))


def program_targets(theta: np.ndarray, cp: CellParams, mm: Dict[str, np.ndarray], mode: str) -> Tuple[np.ndarray, int]:
    """Target conductances (uS) and the number of cells whose target had to be clipped."""
    g_in = cp.g_lo + theta * cp.dg
    if mode == "conductance":
        return g_in.copy(), 0
    if mode != "threshold":
        raise ValueError(mode)
    _, _, k_nom = cp.nominal()
    g = (g_in * k_nom - mm["d_vth"]) / cell_k(cp, cp.v_sl, mm)
    clipped = int(np.sum((g < 0.0) | (g > cp.g_max)))
    return np.clip(g, 0.0, cp.g_max), clipped


def effective_theta(g_t: np.ndarray, cp: CellParams, mm: Dict[str, np.ndarray], node: str, a_hat: float) -> np.ndarray:
    """Normalized threshold seen by the input x for device conductances g_t (uS) at time t."""
    c_nom, _, k_nom = cp.nominal()
    s, v_sl, g = 1.0, cp.v_sl, g_t
    if node == "dac":
        s = a_hat
    elif node == "supply":
        v_sl = cp.v_tx * (1.0 + c_nom / a_hat)
    elif node == "ideal":
        g = g_t / a_hat
    elif node != "none":
        raise ValueError(node)
    v_rel = mm["d_vth"] + g * cell_k(cp, v_sl, mm)          # bound voltage minus V_TH,nom
    return (v_rel / (s * k_nom) - cp.g_lo) / cp.dg


def closed_form_uniform_drift(theta: np.ndarray, cp: CellParams, mm: Dict[str, np.ndarray], node: str, a: float) -> np.ndarray:
    """Effective threshold for threshold-verify programming under a uniform drift factor a
    (no programming or read noise, no drift spread, unclipped cells), derived by hand:

        none   : ((dV_TH + a (g_in k_nom - dV_TH)) / k_nom - g_lo) / dg
        dac    : theta + (1/a - 1) dV_TH / (k_nom dg)
        supply : ((dV_TH + a (g_in k_nom - dV_TH) rho_i) / k_nom - g_lo) / dg,
                 rho_i = (V_SL(t)/V_TX,i - 1) / (V_SL/V_TX,i - 1), V_SL(t) = V_TX,nom (1 + c_nom / a)
        ideal  : theta
    """
    c_nom, _, k_nom = cp.nominal()
    g_in = cp.g_lo + theta * cp.dg
    dv = mm["d_vth"]
    if node == "none":
        return ((dv + a * (g_in * k_nom - dv)) / k_nom - cp.g_lo) / cp.dg
    if node == "dac":
        return theta + (1.0 / a - 1.0) * dv / (k_nom * cp.dg)
    if node == "supply":
        vtx = cp.v_tx + mm["d_vtx"]
        v_sl_t = cp.v_tx * (1.0 + c_nom / a)
        rho = (v_sl_t / vtx - 1.0) / (cp.v_sl / vtx - 1.0)
        return ((dv + a * (g_in * k_nom - dv) * rho) / k_nom - cp.g_lo) / cp.dg
    if node == "ideal":
        return theta.copy()
    raise ValueError(node)
