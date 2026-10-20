"""PCM device model for inference, transcribed from AIHWKit.

Source of every formula and constant in this module:
  AIHWKit (IBM/aihwkit, master branch, read 2026-10-12)
    src/aihwkit/inference/noise/pcm.py            PCMLikeNoiseModel
    src/aihwkit/inference/converter/conductance.py SinglePairConductanceConverter
  The statistical model is the one fitted on measured PCM devices in
  S. R. Nandakumar et al., "Phase-change memory models for deep learning
  training and inference," ICECS 2019 (IEEE Xplore 8964852), as cited in
  the AIHWKit docstring.

Units: conductances in microsiemens (uS), times in seconds.
The functions take pre-drawn standard-normal arrays (``z_*``) so that
different experimental conditions can share the same random draws.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

ZERO_CLIP = 1e-7  # AIHWKit _ZERO_CLIP


@dataclass(frozen=True)
class PCMParams:
    """Constants of AIHWKit PCMLikeNoiseModel / SinglePairConductanceConverter."""

    g_max: float = 25.0
    g_min: float = 0.0
    prog_coeff: Tuple[float, float, float] = (0.26348, 1.9650, -1.1731)
    prog_coeff_g_max_reference: float = 25.0
    t0: float = 20.0
    t_read: float = 250.0e-9
    nu_mean_a: float = -0.0155
    nu_mean_b: float = 0.0244
    nu_mean_min: float = 0.049
    nu_mean_max: float = 0.1
    nu_std_a: float = -0.0125
    nu_std_b: float = -0.0059
    nu_std_min: float = 0.008
    nu_std_max: float = 0.045
    read_q0: float = 0.0088
    read_exp: float = 0.65
    read_floor: float = 1e-3
    read_qmax: float = 0.2
    prog_noise_scale: float = 1.0
    read_noise_scale: float = 1.0
    drift_scale: float = 1.0

    @classmethod
    def from_dict(cls, d: dict) -> "PCMParams":
        keys = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        kw = {k: (tuple(v) if k == "prog_coeff" else v) for k, v in d.items() if k in keys}
        return cls(**kw)


def map_weights(w: np.ndarray, p: PCMParams) -> Tuple[np.ndarray, np.ndarray, float]:
    """SinglePairConductanceConverter.convert_to_conductances.

    abs max over the whole matrix; one device of each pair is at g_min.
    Returns (G_plus, G_minus, scale_ratio) with scale_ratio in uS per weight unit.
    """
    abs_max = max(float(np.max(np.abs(w))), ZERO_CLIP)
    scale_ratio = (p.g_max - p.g_min) / abs_max
    sw = w * scale_ratio
    gp = np.clip(sw, 0.0, p.g_max) + p.g_min
    gn = np.clip(-sw, 0.0, p.g_max) + p.g_min
    return gp, gn, scale_ratio


def prog_sigma(g_target: np.ndarray, p: PCMParams) -> np.ndarray:
    """Programming-noise standard deviation (uS), PCMLikeNoiseModel."""
    x = g_target / p.g_max
    c0, c1, c2 = p.prog_coeff
    sig = c0 + c1 * x + c2 * x * x
    return sig * (p.g_max / p.prog_coeff_g_max_reference)


def apply_programming_noise(g_target: np.ndarray, z: np.ndarray, p: PCMParams) -> np.ndarray:
    """g_prog = clamp(g_target + scale * sigma(g_target) * z, min=0)."""
    g = g_target + p.prog_noise_scale * prog_sigma(g_target, p) * z
    return np.maximum(g, 0.0)


def nu_mean_std(g_target: np.ndarray, p: PCMParams) -> Tuple[np.ndarray, np.ndarray]:
    """Mean and standard deviation of the drift exponent at a target conductance."""
    g_rel = np.maximum(np.abs(g_target / p.g_max), ZERO_CLIP)
    lg = np.log(g_rel)
    mu = np.clip(p.nu_mean_a * lg + p.nu_mean_b, p.nu_mean_min, p.nu_mean_max)
    sd = np.clip(p.nu_std_a * lg + p.nu_std_b, p.nu_std_min, p.nu_std_max)
    return mu, sd


def drift_coefficients(g_target: np.ndarray, z: np.ndarray, p: PCMParams) -> np.ndarray:
    """nu = |mu(g) + sigma(g) * z| * drift_scale (PCMLikeNoiseModel)."""
    mu, sd = nu_mean_std(g_target, p)
    nu = np.abs(mu + sd * z)
    return nu * p.drift_scale


def drift_factor(nu: np.ndarray, t_inference: float, p: PCMParams) -> np.ndarray:
    """(t / t0) ** (-nu) with t = t_inference + t0; 1 if t <= t0."""
    t = t_inference + p.t0
    if t > p.t0:
        return (t / p.t0) ** (-nu)
    return np.ones_like(nu)


def apply_drift(g_prog: np.ndarray, nu: np.ndarray, t_inference: float, p: PCMParams) -> np.ndarray:
    """Drift only (no read noise)."""
    return g_prog * drift_factor(nu, t_inference, p)


def apply_drift_and_read_noise(
    g_prog: np.ndarray, nu: np.ndarray, t_inference: float, z_read: np.ndarray, p: PCMParams
) -> np.ndarray:
    """PCMLikeNoiseModel.apply_drift_noise_to_conductance (drift + accumulated 1/f noise)."""
    t = t_inference + p.t0
    g_drift = apply_drift(g_prog, nu, t_inference, p)
    if t > 0:
        base = np.maximum((np.abs(g_prog) / p.g_max) ** p.read_exp, p.read_floor)
        q_s = np.minimum(p.read_q0 / base, p.read_qmax)
        sig_noise = q_s * np.sqrt(np.log((t + p.t_read) / (2.0 * p.t_read)))
        g_final = g_drift + np.abs(g_drift) * p.read_noise_scale * sig_noise * z_read
    else:
        g_final = g_prog
    return np.maximum(g_final, 0.0)
