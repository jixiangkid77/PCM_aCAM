"""ngspice helpers: transistor I-V sweeps with per-instance sky130 mismatch, binary raw files.

Instrument layer only (no experiment numbers). Requires ngspice on PATH and the model subset
built by pcm_acam.sky130_models (third_party/sky130_fd_pr_subset/sky130_subset_tt.lib.spice).
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

MM_KEYS = ("vth0", "toxe", "voff", "nfactor")
SPICEINIT = "set ngbehavior=hsa\nset ng_nomodcheck\n"


def ngspice_version() -> str:
    out = subprocess.run(["ngspice", "-v"], capture_output=True, text=True).stdout
    for ln in out.splitlines():
        if "ngspice-" in ln:
            return ln.strip(" *")
    return out.strip()


def read_raw_binary(path: Path) -> Dict[str, np.ndarray]:
    """Parse an ngspice binary raw file (real data) into {vector name: array}."""
    data = Path(path).read_bytes()
    marker = b"Binary:\n"
    pos = data.find(marker)
    if pos < 0:
        raise RuntimeError(f"{path}: not a binary raw file")
    header = data[:pos].decode("latin-1").splitlines()
    nvar = npts = None
    names: List[str] = []
    in_vars = False
    for ln in header:
        if ln.startswith("No. Variables:"):
            nvar = int(ln.split(":")[1])
        elif ln.startswith("No. Points:"):
            npts = int(ln.split(":")[1])
        elif ln.startswith("Flags:") and "complex" in ln:
            raise RuntimeError("complex raw data not supported")
        elif ln.startswith("Variables:"):
            in_vars = True
        elif in_vars and ln.strip():
            parts = ln.split()
            names.append(parts[1].lower())
    if nvar is None or npts is None or len(names) != nvar:
        raise RuntimeError(f"{path}: bad header ({nvar}, {npts}, {len(names)})")
    arr = np.frombuffer(data[pos + len(marker):], dtype="<f8", count=nvar * npts).reshape(npts, nvar)
    return {n: arr[:, k].copy() for k, n in enumerate(names)}


def run_netlist(netlist: str, keep_dir: Optional[Path] = None, timeout: int = 1800) -> Dict[str, np.ndarray]:
    """Run ngspice in batch mode on `netlist` (must contain an analysis card) and return the raw vectors."""
    with tempfile.TemporaryDirectory(prefix="pcm_acam_ng_") as td:
        wd = Path(keep_dir) if keep_dir is not None else Path(td)
        wd.mkdir(parents=True, exist_ok=True)
        (wd / ".spiceinit").write_text(SPICEINIT)
        (wd / "net.sp").write_text(netlist)
        res = subprocess.run(["ngspice", "-b", "-r", "out.raw", "net.sp"], cwd=wd,
                             capture_output=True, text=True, timeout=timeout)
        (wd / "ngspice.log").write_text(res.stdout + "\n--- stderr ---\n" + res.stderr)
        if res.returncode != 0 or not (wd / "out.raw").exists():
            raise RuntimeError(f"ngspice failed (rc={res.returncode}):\n{res.stdout[-3000:]}\n{res.stderr[-2000:]}")
        low = (res.stdout + res.stderr).lower()
        for bad in ("singular matrix", "timestep too small", "no convergence", "unknown subckt", "could not find"):
            if bad in low:
                raise RuntimeError(f"ngspice reported '{bad}':\n{res.stdout[-3000:]}")
        vec = read_raw_binary(wd / "out.raw")
        for k, v in vec.items():
            if not np.all(np.isfinite(v)):
                raise RuntimeError(f"non-finite values in vector {k}")
        return vec


def mm_string(mm: Optional[Dict[str, float]]) -> str:
    if not mm:
        return ""
    return " " + " ".join(f"mm_{k}={float(mm.get(k, 0.0)):.9g}" for k in MM_KEYS)


def iv_sweep_netlist(lib: Path, dev: str, w: float, l: float, vds: Sequence[float],
                     mm: Sequence[Optional[Dict[str, float]]], vg_start: float, vg_stop: float,
                     vg_step: float, title: str = "iv sweep", temp: Optional[float] = None) -> str:
    """N independent devices, common gate sweep, each drain held by its own source at vds[k]."""
    if len(vds) != len(mm):
        raise ValueError("vds and mm must have the same length")
    lines = [f"* {title}", f'.include "{Path(lib).resolve()}"', "vg g 0 0"]
    for k, (vd, m) in enumerate(zip(vds, mm)):
        lines.append(f"vd{k} d{k} 0 {float(vd):.9g}")
        lines.append(f"x{k} d{k} g 0 0 sky130_fd_pr__{dev} w={w:.6g} l={l:.6g}{mm_string(m)}")
    if temp is not None:
        lines.append(f".temp {float(temp):.6g}")
    lines += [f".dc vg {vg_start:.6g} {vg_stop:.6g} {vg_step:.6g}", ".options savecurrents=0", ".end"]
    return "\n".join(lines) + "\n"


def iv_sweep(lib: Path, dev: str, w: float, l: float, vds: Sequence[float],
             mm: Sequence[Optional[Dict[str, float]]], vg_start: float, vg_stop: float,
             vg_step: float, chunk: int = 400, keep_dir: Optional[Path] = None,
             temp: Optional[float] = None) -> tuple[np.ndarray, np.ndarray]:
    """Drain current (A, positive into the drain) of each device versus gate voltage.

    Returns (vg [P], I [N, P]).
    """
    n = len(vds)
    out = []
    vg = None
    for s in range(0, n, chunk):
        sl = slice(s, min(n, s + chunk))
        net = iv_sweep_netlist(lib, dev, w, l, list(vds[sl]), list(mm[sl]), vg_start, vg_stop, vg_step, temp=temp)
        kd = None if keep_dir is None else Path(keep_dir) / f"chunk_{s:05d}"
        vec = run_netlist(net, keep_dir=kd)
        vsweep = vec.get("v(g)")
        if vsweep is None:
            vsweep = vec[[k for k in vec if "sweep" in k][0]]
        if vg is None:
            vg = vsweep
        elif not np.array_equal(vg, vsweep):
            raise RuntimeError("inconsistent sweep vectors between chunks")
        cur = np.vstack([-vec[f"i(vd{k})"] for k in range(sl.stop - sl.start)])  # source current is negative
        out.append(cur)
    return vg, np.vstack(out)


def op_param_sweep(lib: Path, dev: str, w: float, l: float, vds: float, vg_start: float, vg_stop: float,
                   vg_step: float, params: Sequence[str] = ("vth", "vdsat", "gm", "gds"),
                   temp: Optional[float] = None) -> Dict[str, np.ndarray]:
    """Nominal device: BSIM4 operating-point quantities versus gate voltage at a fixed drain voltage."""
    inst = f"@m.x0.msky130_fd_pr__{dev}"
    saves = " ".join(f"{inst}[{p}]" for p in params)
    net = "\n".join([
        "* op-parameter sweep", f'.include "{Path(lib).resolve()}"', "vg g 0 0", f"vd0 d0 0 {vds:.9g}",
        f"x0 d0 g 0 0 sky130_fd_pr__{dev} w={w:.6g} l={l:.6g}",
        f".save v(g) i(vd0) {saves}"] + ([f".temp {float(temp):.6g}"] if temp is not None else []) + [
        f".dc vg {vg_start:.6g} {vg_stop:.6g} {vg_step:.6g}", ".end"]) + "\n"
    vec = run_netlist(net)
    out = {"vg": vec["v(g)"], "id": -vec["i(vd0)"]}
    for p in params:
        key = f"{inst}[{p}]".lower()
        for cand in (key, f"v({key})", f"i({key})"):
            if cand in vec:
                out[p] = vec[cand]
                break
        else:
            raise RuntimeError(f"missing {key}; have {list(vec)[:8]}")
    return out


def branch_dc_netlist(lib: Path, dev: str, w: float, l: float, g_us: Sequence[float], v_sl: float,
                      vdl_start: float, vdl_stop: float, vdl_step: float,
                      mm: Optional[Sequence[Optional[Dict[str, float]]]] = None,
                      temp: Optional[float] = None) -> str:
    """Full divider branches (PCM as a linear resistor 1/G from SL to node xk, transistor xk -> 0, gate = DL)."""
    mm = list(mm) if mm is not None else [None] * len(g_us)
    lines = ["* divider branches", f'.include "{Path(lib).resolve()}"', f"vsl sl 0 {v_sl:.9g}", "vdl dl 0 0"]
    for k, (g, m) in enumerate(zip(g_us, mm)):
        lines.append(f"rp{k} sl n{k} {1.0 / (float(g) * 1e-6):.12g}")
        lines.append(f"x{k} n{k} dl 0 0 sky130_fd_pr__{dev} w={w:.6g} l={l:.6g}{mm_string(m)}")
    if temp is not None:
        lines.append(f".temp {float(temp):.6g}")
    lines += [f".dc vdl {vdl_start:.6g} {vdl_stop:.6g} {vdl_step:.6g}", ".end"]
    return "\n".join(lines) + "\n"


def branch_dc(lib: Path, dev: str, w: float, l: float, g_us: Sequence[float], v_sl: float,
              vdl_start: float, vdl_stop: float, vdl_step: float,
              mm: Optional[Sequence[Optional[Dict[str, float]]]] = None,
              temp: Optional[float] = None) -> tuple[np.ndarray, np.ndarray]:
    """Node voltage of each branch versus DL voltage. Returns (vdl [P], vx [N, P])."""
    vec = run_netlist(branch_dc_netlist(lib, dev, w, l, g_us, v_sl, vdl_start, vdl_stop, vdl_step, mm, temp=temp))
    vdl = vec["v(dl)"]
    return vdl, np.vstack([vec[f"v(n{k})"] for k in range(len(g_us))])
