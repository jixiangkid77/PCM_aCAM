"""Design-time predictor of the drift-compensation errors of an aCAM bound branch (instrument layer).

First-order laws (proposed after E1; see docs/E1_results.md section 6). Input-referred error of a cell
programmed by threshold verify at threshold theta, after a uniform drift factor a with perfect tracking:

  supply node:      dx = (G/dG) * ln[(D - a delta) / (D - delta)]
  input remap node: dx = (G/dG) * [eps(a I) - eps(I)]

  G = G(theta), dG = g_hi - g_lo, D = PCM voltage at the switching point, I = G * D,
  delta = V_sw,k - V_sw,nom (pull-down), eps(I) = relative current deficit of the cell's divider at the
  nominal gate voltage for current I:  eps(I) = sum_p s_p(I) z_p + lambda(I) delta,
  s_p(I) = -d ln I_D / d mm_p (mm_p = standard-normal PDK mismatch parameter),
  lambda(I) = -d ln I_D / d V_DS.

With independent standard-normal mm_p and delta ~ N(0, sigma_delta^2), sigma_delta^2 = sum_p (dV_sw/dmm_p)^2:
  E[dx_supply^2] = (G/dG)^2 E[ln^2 r(delta)]                    (trapezoid rule, see e_ln2_r)
  E[dx_remap^2]  = (G/dG)^2 { sum_p [s_p(aI) - s_p(I)]^2 + [lambda(aI) - lambda(I)]^2 sigma_delta^2 }
  predicted rms (LSB) = 15 * sqrt( mean over theta of E[dx^2] )

Only nominal and +-1 sigma perturbed device simulations are needed (no Monte Carlo).
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from .branch_post import IVCurve, switching_voltage
from .spice_iv import MM_KEYS, iv_sweep

DVDS = 0.005  # V, step for d ln I / d V_DS


def _perturbed_mm():
    rows = [None]
    for k in MM_KEYS:
        rows.append({k: +1.0})
        rows.append({k: -1.0})
    return rows


def pulldown_sensitivity(lib, dev, w, l, vdd, i_sw, sweep, temp=None) -> Dict[str, float]:
    mm = _perturbed_mm()
    vg, cur = iv_sweep(lib, dev, w, l, [vdd] * len(mm), mm, sweep["vg_start"], sweep["vg_stop"], sweep["vg_step"], temp=temp)
    vsw = [switching_voltage(vg, cur[i], i_sw) for i in range(len(mm))]
    out = {"vsw_nom": vsw[0]}
    var = 0.0
    for j, k in enumerate(MM_KEYS):
        s = (vsw[1 + 2 * j] - vsw[2 + 2 * j]) / 2.0
        out[f"dvsw_d{k}"] = s
        var += s * s
    out["sigma_delta"] = float(np.sqrt(var))
    return out


def divider_sensitivity(lib, dev, w, l, vsw_nom, sweep, temp=None, i_floor=1e-9):
    """Nominal divider curve at V_DS = V_sw,nom plus curves for mm_p = +-1 and V_DS +- DVDS."""
    mm = _perturbed_mm() + [None, None]
    vds = [vsw_nom] * (len(mm) - 2) + [vsw_nom + DVDS, vsw_nom - DVDS]
    vg, cur = iv_sweep(lib, dev, w, l, vds, mm, sweep["vg_start"], sweep["vg_stop"], sweep["vg_step"], temp=temp)
    curves = [IVCurve.from_arrays(vg, cur[i], i_floor) for i in range(len(mm))]
    return vg, cur, curves


def eps_sensitivities(curves, currents: np.ndarray) -> Dict[str, np.ndarray]:
    """s_p(I) for each mismatch parameter and lambda(I), at the nominal gate voltage for each current."""
    nom = curves[0]
    v0 = nom.v_at(currents)
    out = {}
    for j, k in enumerate(MM_KEYS):
        lp = np.log(curves[1 + 2 * j].i_at(v0))
        lm = np.log(curves[2 + 2 * j].i_at(v0))
        out[k] = -(lp - lm) / 2.0
    lhi = np.log(curves[-2].i_at(v0))
    llo = np.log(curves[-1].i_at(v0))
    out["lambda"] = -(lhi - llo) / (2.0 * DVDS)
    return out


def e_ln2_r(d_v: float, a: float, sigma: float, n: int = 4001) -> float:
    """E[ ln^2((D - a delta)/(D - delta)) ] for delta ~ N(0, sigma^2).

    Trapezoid rule on delta in [-6 sigma, +6 sigma]; points with delta >= 0.95 D (a pull-down that switches
    above the divider supply, i.e. a cell that cannot be programmed) are dropped and the Gaussian weights
    renormalized.
    """
    delta = np.linspace(-6.0 * sigma, 6.0 * sigma, n)
    w = np.exp(-0.5 * (delta / sigma) ** 2)
    keep = delta < 0.95 * d_v
    delta, w = delta[keep], w[keep]
    val = np.log((d_v - a * delta) / (d_v - delta)) ** 2
    return float(np.trapezoid(w * val, delta) / np.trapezoid(w, delta))


def predict(curves, theta: np.ndarray, g_lo: float, g_hi: float, d_v: float, a: float,
            sigma_delta: float) -> Dict[str, object]:
    """Predicted per-theta mean-square errors (in x units^2) and rms in LSB for each node/setting."""
    dg = g_hi - g_lo
    g = g_lo + theta * dg
    i_t = g * 1e-6 * d_v
    s1 = eps_sensitivities(curves, i_t)
    s2 = eps_sensitivities(curves, a * i_t)
    w2 = (g / dg) ** 2
    var_div = sum((s2[k] - s1[k]) ** 2 for k in MM_KEYS)
    var_lam = (s2["lambda"] - s1["lambda"]) ** 2 * sigma_delta ** 2
    var_sup = np.full_like(theta, e_ln2_r(d_v, a, sigma_delta))
    per_theta = {
        "pdk|dac_remap": w2 * (var_div + var_lam),
        "pdk|supply": w2 * var_sup,
        "div_only|dac_remap": w2 * var_div,
        "div_only|supply": np.zeros_like(theta),
        "pd_only|dac_remap": w2 * var_lam,
        "pd_only|supply": w2 * var_sup,
    }
    rms = {k: float(15.0 * np.sqrt(np.mean(v))) for k, v in per_theta.items()}
    return {"per_theta_ms_x": {k: v.tolist() for k, v in per_theta.items()}, "rms_lsb": rms,
            "s_at_I": {k: v.tolist() for k, v in s1.items()}, "s_at_aI": {k: v.tolist() for k, v in s2.items()}}
