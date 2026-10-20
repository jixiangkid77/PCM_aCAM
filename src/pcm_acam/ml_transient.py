"""Match-line (ML) transient sensing for the aCAM lower-bound branch (instrument layer, E3c).

Sensing model: the ML is held at V_pre during the DC operating point (``.ic``, no UIC, so every branch node
starts at its DC value) and released at t = 0; it discharges through all pull-downs of the row into a lumped
capacitance C_SA + N * C_wire plus the device capacitances of the 2N pull-downs (lower and upper branch of
each cell). A cell is judged "mismatch" if V_ML falls through V_ref before the sense time t_s.

The transient switching voltage V_sw,tr of a pull-down is the DC gate voltage at which V_ML reaches V_ref
exactly at t_s; the transient bound of a full branch is the input V_DL at which the same happens.
Both are found by vectorised bisection: K independent copies of the circuit in one netlist, one ngspice run
per bisection step, crossing times from ``.meas``.

Instrument layer only (no experiment numbers).
"""

from __future__ import annotations

import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from .spice_iv import SPICEINIT, mm_string

PD_DEV = "sky130_fd_pr__nfet_01v8_lvt"
DIV_DEV = "sky130_fd_pr__nfet_01v8"


@dataclass
class MLSpec:
    v_pre: float          # precharge voltage (V)
    v_ref: float          # sense threshold (V)
    c_sa: float           # sense-amplifier + precharge load (F)
    c_wire: float         # wire capacitance per cell (F)
    pd_w: float
    pd_l: float
    t_step: float = 0.1e-9
    t_max: float = 0.2e-9

    def c_lumped(self, n_cells: int) -> float:
        return self.c_sa + n_cells * self.c_wire


def run_meas(lines: List[str], names: Sequence[str], keep_dir: Optional[Path] = None,
             timeout: int = 7200) -> np.ndarray:
    """Run a transient netlist (lines without .end) and return the .meas values; NaN where the measure failed."""
    text = "\n".join(lines + [".options numdgt=10", ".end"]) + "\n"
    with tempfile.TemporaryDirectory(prefix="pcm_acam_ml_") as td:
        wd = Path(keep_dir) if keep_dir is not None else Path(td)
        wd.mkdir(parents=True, exist_ok=True)
        (wd / ".spiceinit").write_text(SPICEINIT)
        (wd / "net.sp").write_text(text)
        res = subprocess.run(["ngspice", "-b", "net.sp"], cwd=wd, capture_output=True, text=True, timeout=timeout)
        out = res.stdout + "\n" + res.stderr
        if keep_dir is not None:
            (wd / "ngspice.log").write_text(out)
    low = out.lower()
    for bad in ("singular matrix", "timestep too small", "unknown subckt", "could not find", "no convergence"):
        if bad in low:
            raise RuntimeError(f"ngspice reported '{bad}':\n{res.stdout[-2000:]}")
    vals = np.full(len(names), np.nan)
    for i, nm in enumerate(names):
        m = re.search(rf"^{re.escape(nm)}\s*=\s*([0-9.eE+-]+)", out, re.M)
        if m:
            vals[i] = float(m.group(1))
    return vals


def row_devices(i: int, spec: MLSpec, n_cells: int, cut_gate: str, leak_gates: Sequence[str],
                mm_cut: Optional[Dict[str, float]] = None) -> List[str]:
    """ML node ml{i}: lumped capacitor, the cell-under-test pull-down (gate node cut_gate), leaker lower-branch
    pull-downs (one per entry of leak_gates), the remaining lower-branch pull-downs at gate 0 and n_cells
    upper-branch pull-downs at gate 0 (matched cells driven by an inverter output near 0 V)."""
    if len(leak_gates) > n_cells - 1:
        raise ValueError("more leakers than other cells")
    pd = f"{PD_DEV} w={spec.pd_w:.6g} l={spec.pd_l:.6g}"
    lines = [f"cml{i} ml{i} 0 {spec.c_lumped(n_cells):.9g}", f"xpc{i} ml{i} {cut_gate} 0 0 {pd}{mm_string(mm_cut)}"]
    for j, g in enumerate(leak_gates):
        lines.append(f"xpl{i}_{j} ml{i} {g} 0 0 {pd}")
    for j in range(n_cells - 1 - len(leak_gates)):
        lines.append(f"xpo{i}_{j} ml{i} 0 0 0 {pd}")
    for j in range(n_cells):
        lines.append(f"xpu{i}_{j} ml{i} 0 0 0 {pd}")
    lines += [f".ic v(ml{i})={spec.v_pre:.9g}", f".meas tran tc{i} when v(ml{i})={spec.v_ref:.9g} fall=1"]
    return lines


def header(lib: Path, title: str, temp: Optional[float] = None) -> List[str]:
    lines = [f"* {title}", f'.include "{Path(lib).resolve()}"']
    if temp is not None:
        lines.append(f".temp {float(temp):.6g}")
    return lines


def tran_card(spec: MLSpec, t_stop: float) -> str:
    return f".tran {spec.t_step:.6g} {t_stop:.9g} 0 {spec.t_max:.6g}"


def bisect(build: Callable[[np.ndarray], List[str]], k: int, lo: np.ndarray, hi: np.ndarray, t_s: np.ndarray,
           iters: int, increasing_means_faster: bool) -> np.ndarray:
    """Vectorised bisection on one parameter per circuit so that the crossing time equals t_s[i].

    build(values) returns the netlist lines (header, all K circuits with .meas tc{i}, .tran card).
    increasing_means_faster: True for a pull-down gate voltage (higher -> faster discharge), False for the
    lower-bound input V_DL (higher -> divider pulls X down -> slower discharge). A failed measure (no
    crossing within the transient) counts as "slower than t_s".
    """
    lo, hi = np.array(lo, float), np.array(hi, float)
    names = [f"tc{i}" for i in range(k)]
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        tc = run_meas(build(mid), names)
        faster = np.isfinite(tc) & (tc <= t_s)
        if increasing_means_faster:
            hi = np.where(faster, mid, hi)
            lo = np.where(faster, lo, mid)
        else:
            lo = np.where(faster, mid, lo)
            hi = np.where(faster, hi, mid)
    return 0.5 * (lo + hi)


def crossing_times(lines_fn: Callable[[], List[str]], k: int) -> np.ndarray:
    return run_meas(lines_fn(), [f"tc{i}" for i in range(k)])


def branch_lines(i: int, b: int, div_w: float, div_l: float, sl_node: str, g_us: float, v_dl: float,
                 x_node: str, mm_div: Optional[Dict[str, float]] = None) -> List[str]:
    """Lower-bound branch b of circuit i: PCM resistor 1/G from the SL node to x_node, divider from x_node to
    ground with its gate on its own DL source."""
    return [f"rp{i}_{b} {sl_node} {x_node} {1.0 / (float(g_us) * 1e-6):.12g}",
            f"vdl{i}_{b} dl{i}_{b} 0 {float(v_dl):.12g}",
            f"xd{i}_{b} {x_node} dl{i}_{b} 0 0 {DIV_DEV} w={div_w:.6g} l={div_l:.6g}{mm_string(mm_div)}"]
