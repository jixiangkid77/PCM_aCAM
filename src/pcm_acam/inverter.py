"""Upper-bound branch of the 6T2M cell: inverter between the divider node and the match-line pull-down (E3b).

Branch (Li et al. 2020, 6T2M): PCM from SL_hi to node Y, divider nMOS from Y to ground with its gate on the
search input DL, Y drives a CMOS inverter whose output Z drives the match-line pull-down. When the input
exceeds the upper bound, Y falls, Z rises above the pull-down switching voltage V_sw,pd and the row is
discharged (mismatch).

The switching point of the branch is the inverter input voltage V_Y* at which the inverter output equals
V_sw,pd (the inverter gate draws no DC current, so the quasi-static analysis of the lower branch carries over
with V_sw -> V_Y*):   G (V_SL - V_Y*) = I_div(V_DL, V_DS = V_Y*).

Mismatch of the inverter (nMOS, pMOS) shifts V_Y* directly; mismatch of the pull-down shifts it divided by
the inverter gain.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from .spice_iv import MM_KEYS, mm_string, run_netlist


def transfer_netlist(lib: Path, inv: dict, mm_n: Sequence[Optional[Dict[str, float]]],
                     mm_p: Sequence[Optional[Dict[str, float]]], vdd: float, sweep: dict,
                     temp: Optional[float] = None) -> str:
    """N unloaded inverters with a common input Y; inv = {n_dev, wn, ln, p_dev, wp, lp}."""
    if len(mm_n) != len(mm_p):
        raise ValueError("mm_n and mm_p must have the same length")
    lines = ["* inverter transfer", f'.include "{Path(lib).resolve()}"', f"vdd vdd 0 {vdd:.9g}", "vy y 0 0"]
    for k, (mn, mp) in enumerate(zip(mm_n, mm_p)):
        lines.append(f"xp{k} z{k} y vdd vdd sky130_fd_pr__{inv['p_dev']} w={inv['wp']:.6g} l={inv['lp']:.6g}{mm_string(mp)}")
        lines.append(f"xn{k} z{k} y 0 0 sky130_fd_pr__{inv['n_dev']} w={inv['wn']:.6g} l={inv['ln']:.6g}{mm_string(mn)}")
    if temp is not None:
        lines.append(f".temp {float(temp):.6g}")
    lines += [f".dc vy {sweep['vg_start']:.6g} {sweep['vg_stop']:.6g} {sweep['vg_step']:.6g}", ".end"]
    return "\n".join(lines) + "\n"


def transfer(lib: Path, inv: dict, mm_n, mm_p, vdd: float, sweep: dict, temp=None, chunk: int = 400):
    """Return (y [P], z [N, P]) for N inverters (chunked netlists)."""
    out, y = [], None
    for s in range(0, len(mm_n), chunk):
        sl = slice(s, min(len(mm_n), s + chunk))
        vec = run_netlist(transfer_netlist(lib, inv, list(mm_n[sl]), list(mm_p[sl]), vdd, sweep, temp))
        yy = vec["v(y)"]
        if y is None:
            y = yy
        elif not np.array_equal(y, yy):
            raise RuntimeError("inconsistent sweep vectors")
        out.append(np.vstack([vec[f"v(z{k})"] for k in range(sl.stop - sl.start)]))
    return y, np.vstack(out)


def input_at_output(y: np.ndarray, z: np.ndarray, v_out: float) -> float:
    """Inverter input at which the (decreasing) output equals v_out; linear interpolation."""
    z = np.asarray(z, dtype=float)
    if not np.all(np.diff(z) <= 1e-9):
        # allow tiny numerical non-monotonicity only far from the crossing
        i = int(np.argmin(np.abs(z - v_out)))
        lo, hi = max(i - 40, 0), min(i + 40, z.size - 1)
        if not np.all(np.diff(z[lo:hi + 1]) < 0):
            raise ValueError("inverter output not monotone near the crossing")
    if not (z.min() < v_out < z.max()):
        raise ValueError("output level not reached in the swept range")
    return float(np.interp(-v_out, -z, y))


def gain_at(y: np.ndarray, z: np.ndarray, y0: float) -> float:
    g = np.gradient(z, y)
    return float(-np.interp(y0, y, g))


def vy_star_sensitivity(lib: Path, inv: dict, pd: dict, vdd: float, i_sw: float, sweep: dict,
                        temp=None) -> Dict[str, float]:
    """Nominal V_Y* and its central-difference sensitivities to the 12 mismatch parameters
    (inverter nMOS, inverter pMOS, pull-down), plus sigma = sqrt(sum s^2) and the inverter gain."""
    from .branch_post import switching_voltage
    from .spice_iv import iv_sweep

    rows = [None]
    for k in MM_KEYS:
        rows += [{k: +1.0}, {k: -1.0}]
    vg, cur = iv_sweep(lib, pd["device"], pd["w_um"], pd["l_um"], [vdd] * len(rows), rows,
                       sweep["vg_start"], sweep["vg_stop"], sweep["vg_step"], temp=temp)
    vsw = [switching_voltage(vg, cur[i], i_sw) for i in range(len(rows))]
    none = [None] * len(rows)
    y, z_n = transfer(lib, inv, rows, none, vdd, sweep, temp)          # nMOS perturbed
    _, z_p = transfer(lib, inv, none, rows, vdd, sweep, temp)          # pMOS perturbed
    out = {"vsw_pd_nom": vsw[0]}
    v0 = input_at_output(y, z_n[0], vsw[0])
    out["vy_star_nom"] = v0
    out["inverter_gain"] = gain_at(y, z_n[0], v0)
    var = 0.0
    for j, k in enumerate(MM_KEYS):
        sn = (input_at_output(y, z_n[1 + 2 * j], vsw[0]) - input_at_output(y, z_n[2 + 2 * j], vsw[0])) / 2.0
        sp = (input_at_output(y, z_p[1 + 2 * j], vsw[0]) - input_at_output(y, z_p[2 + 2 * j], vsw[0])) / 2.0
        sd = (input_at_output(y, z_n[0], vsw[1 + 2 * j]) - input_at_output(y, z_n[0], vsw[2 + 2 * j])) / 2.0
        out[f"dvy_dinvn_{k}"], out[f"dvy_dinvp_{k}"], out[f"dvy_dpd_{k}"] = sn, sp, sd
        var += sn * sn + sp * sp + sd * sd
    out["sigma_delta"] = float(np.sqrt(var))
    return out


def upper_branch_netlist(lib: Path, div: dict, inv: dict, g_us: List[float], v_sl: float, vdd: float,
                         sweep: dict, temp=None) -> str:
    """Nominal full upper branches (PCM resistor, divider, inverter) versus the input voltage."""
    lines = ["* upper branches", f'.include "{Path(lib).resolve()}"', f"vsl sl 0 {v_sl:.12g}", f"vdd vdd 0 {vdd:.9g}",
             "vdl dl 0 0"]
    for k, g in enumerate(g_us):
        lines.append(f"rp{k} sl y{k} {1.0 / (float(g) * 1e-6):.12g}")
        lines.append(f"xd{k} y{k} dl 0 0 sky130_fd_pr__{div['device']} w={div['w_um']:.6g} l={div['l_um']:.6g}")
        lines.append(f"xp{k} z{k} y{k} vdd vdd sky130_fd_pr__{inv['p_dev']} w={inv['wp']:.6g} l={inv['lp']:.6g}")
        lines.append(f"xn{k} z{k} y{k} 0 0 sky130_fd_pr__{inv['n_dev']} w={inv['wn']:.6g} l={inv['ln']:.6g}")
    if temp is not None:
        lines.append(f".temp {float(temp):.6g}")
    lines += [f".dc vdl {sweep['vg_start']:.6g} {sweep['vg_stop']:.6g} {sweep['vg_step']:.6g}", ".end"]
    return "\n".join(lines) + "\n"


def upper_branch_bounds(lib: Path, div: dict, inv: dict, g_us: List[float], v_sl: float, vdd: float, v_sw_pd: float,
                        sweep: dict, temp=None) -> np.ndarray:
    """Upper bound = input voltage at which the inverter output reaches the pull-down switching voltage."""
    vec = run_netlist(upper_branch_netlist(lib, div, inv, g_us, v_sl, vdd, sweep, temp))
    vdl = vec["v(dl)"]
    return np.array([float(np.interp(v_sw_pd, vec[f"v(z{k})"], vdl)) for k in range(len(g_us))])
