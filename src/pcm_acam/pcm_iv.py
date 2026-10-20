"""Nonlinear PCM read characteristic for the aCAM bound branch (instrument layer, used from E3a on).

Subthreshold conduction in amorphous phase-change material is commonly written with a sinh field
dependence (trap-limited / Poole-Frenkel type, e.g. Ielmini & Zhang, J. Appl. Phys. 2007):

    I(V) = A * s(V; V0),   s(V; V0) = V0 * sinh(V / V0)   (s -> V for V0 -> infinity, a linear resistor)

A is set by the programmed state; V0 is the characteristic voltage of the nonlinearity (smaller V0 =
more nonlinear). A state is identified by its chord conductance at the programming voltage D,
G = I(D)/D, and its V0 may depend on the state:

    V0(G) = V0_ref * (G / G_ref)**beta

Drift is modelled as a uniform factor on the whole characteristic, I(V, t) = a * I(V, 0) (the sinh
shape is kept; in the trap-limited model drift changes the prefactor, not V0).

Compensation nodes (lower-bound branch, PCM between V_SL and node X, switching point V_X = V_sw):
  * input remap (DAC): V_SL unchanged, so the PCM stays at the voltage it was programmed at -> the
    nonlinearity never enters (first-order law of E2 unchanged).
  * supply, linear rule: V_SL(t) = V_sw,nom + D / a_hat (correct only for a linear PCM).
  * supply, replica rule: V_SL(t) = V_sw,nom + V*, with a * s(V*; V0_rep) = s(D; V0_rep): the voltage
    that restores the current of a replica cell of conductance G_rep (a feedback loop on the replica).

For a cell programmed by threshold verify at PCM voltage D - delta and read after drift at V_t - delta,
the switching current changes by the factor
    rho(delta) = a * s(V_t - delta; V0) / s(D - delta; V0).
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from .law_predict import eps_sensitivities
from .spice_iv import MM_KEYS


def s_of_v(v, v0):
    """V0 * sinh(V / V0), elementwise with broadcasting; the linear limit V where V0 = inf."""
    v, v0 = np.broadcast_arrays(np.asarray(v, dtype=float), np.asarray(v0, dtype=float))
    out = np.array(v, dtype=float, copy=True)
    fin = np.isfinite(v0)
    out[fin] = v0[fin] * np.sinh(v[fin] / v0[fin])
    return out


def v0_of_g(g_us, v0_ref: float, beta: float, g_ref_us: float):
    g = np.asarray(g_us, dtype=float)
    if not np.isfinite(v0_ref):
        return np.full_like(g, np.inf)
    return v0_ref * (g / g_ref_us) ** beta


def replica_vt(d_v: float, a: float, v0_rep: float) -> float:
    """PCM voltage V* with a * s(V*) = s(D) (exact restoration of the replica's switching current)."""
    if not np.isfinite(v0_rep):
        return d_v / a
    return float(v0_rep * np.arcsinh(np.sinh(d_v / v0_rep) / a))


def rho(delta, vt: float, d_v: float, a: float, v0):
    """Switching-current factor after drift and supply compensation (see module docstring)."""
    delta = np.asarray(delta, dtype=float)
    return a * s_of_v(vt - delta, v0) / s_of_v(d_v - delta, v0)


def _gauss_grid(sigma: float, limit: float, n: int = 4001):
    delta = np.linspace(-6.0 * sigma, 6.0 * sigma, n)
    w = np.exp(-0.5 * (delta / sigma) ** 2)
    keep = delta < 0.95 * limit
    return delta[keep], w[keep]


def predict_supply(curves, theta: np.ndarray, g_lo: float, g_hi: float, d_v: float, a: float, sigma_delta: float,
                   vt: float, v0_theta: np.ndarray) -> Dict[str, object]:
    """Predicted mean-square input-referred error (x units^2) per theta for a supply node.

    dx = (G/dG) [ (rho(delta) - 1) + rho0 * (eps(rho0 I) - eps(I)) ],  eps(I) = sum_p s_p(I) z_p + lambda(I) delta
    E[dx^2] = (G/dG)^2 { E_delta[(rho(delta) - 1 + rho0 dlambda delta)^2] + rho0^2 sum_p ds_p^2 }
    with rho0 = rho(0), ds_p = s_p(rho0 I) - s_p(I), dlambda = lambda(rho0 I) - lambda(I); the expectation over
    delta ~ N(0, sigma_delta^2) by the trapezoid rule on [-6 sigma, 6 sigma] (points with delta >= 0.95 min(D, V_t)
    dropped and the weights renormalized). For a linear PCM and V_t = D/a this is the (r - 1) form of the E2 law.
    """
    dg = g_hi - g_lo
    g = g_lo + theta * dg
    i_t = g * 1e-6 * d_v
    rho0 = np.array([float(rho(0.0, vt, d_v, a, v0)) for v0 in v0_theta])
    s1 = eps_sensitivities(curves, i_t)
    s2 = eps_sensitivities(curves, rho0 * i_t)
    ds2 = sum((np.asarray(s2[k]) - np.asarray(s1[k])) ** 2 for k in MM_KEYS)
    dlam = np.asarray(s2["lambda"]) - np.asarray(s1["lambda"])
    delta, w = _gauss_grid(sigma_delta, min(d_v, vt))
    ms_delta = np.empty_like(theta)
    for j, v0 in enumerate(v0_theta):
        val = (rho(delta, vt, d_v, a, v0) - 1.0 + rho0[j] * dlam[j] * delta) ** 2
        ms_delta[j] = np.trapezoid(w * val, delta) / np.trapezoid(w, delta)
    w2 = (g / dg) ** 2
    ms = w2 * (ms_delta + rho0 ** 2 * ds2)
    ms_sys = w2 * (rho0 - 1.0) ** 2
    return {"ms_x": ms, "ms_systematic_x": ms_sys, "rho0": rho0,
            "rms_lsb": float(15.0 * np.sqrt(np.mean(ms))),
            "rms_systematic_lsb": float(15.0 * np.sqrt(np.mean(ms_sys)))}


def predict_dac(curves, theta: np.ndarray, g_lo: float, g_hi: float, d_v: float, a: float,
                sigma_delta: float) -> Dict[str, object]:
    """E2 law for the input-remap node (independent of the PCM nonlinearity)."""
    dg = g_hi - g_lo
    g = g_lo + theta * dg
    i_t = g * 1e-6 * d_v
    s1 = eps_sensitivities(curves, i_t)
    s2 = eps_sensitivities(curves, a * i_t)
    var = sum((np.asarray(s2[k]) - np.asarray(s1[k])) ** 2 for k in MM_KEYS)
    var = var + (np.asarray(s2["lambda"]) - np.asarray(s1["lambda"])) ** 2 * sigma_delta ** 2
    ms = (g / dg) ** 2 * var
    return {"ms_x": ms, "rms_lsb": float(15.0 * np.sqrt(np.mean(ms)))}


def grid_label(v0_ref: float, beta: Optional[float]) -> str:
    if not np.isfinite(v0_ref):
        return "linear"
    return f"V0={v0_ref:g}|beta={beta:+g}"
