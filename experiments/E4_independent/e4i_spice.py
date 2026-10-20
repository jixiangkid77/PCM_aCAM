"""E4 independent check: ngspice helpers (netlist snippets, sequential runner, binary raw reader,
mismatch draws). One ngspice process at a time: run_ngspice waits for the process to finish."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import numpy as np

import e4i_core as C

GROUPS = ["div_lo", "pd_lo", "b1n", "b1p", "b2n", "b2p", "div_hi", "un", "up", "pd_hi"]
NFET = "sky130_fd_pr__nfet_01v8"
NFET_LVT = "sky130_fd_pr__nfet_01v8_lvt"
PFET = "sky130_fd_pr__pfet_01v8"
PD_W, PD_L = 0.42, 0.15
INV_WN, INV_LN, INV_WP, INV_LP = 2.0, 0.15, 0.42, 4.0
VDD = 1.8


def pool_draws(n: int = 2000) -> dict:
    """numpy default_rng([20261017, 5]); standard_normal((n, 4)) per group, in GROUPS order;
    columns (vth0, toxe, voff, nfactor); instance k >= 1 uses row k - 1."""
    rng = np.random.default_rng(list(C.POOL_SEED))
    return {g: rng.standard_normal((n, 4)) for g in GROUPS}


def mm(z) -> str:
    if z is None:
        return ""
    return (f"mm_vth0={z[0]:.17g} mm_toxe={z[1]:.17g} mm_voff={z[2]:.17g} "
            f"mm_nfactor={z[3]:.17g}")


def inst_z(draws: dict, group: str, k: int):
    """Mismatch vector of instance k (k = 0: nominal)."""
    return None if k == 0 else draws[group][k - 1]


def header(title: str, options: str = "") -> str:
    lines = [f"* {title}", f'.include "{C.LIB_PATH}"']
    if options:
        lines.append(f".options {options}")
    return "\n".join(lines) + "\n"


def write_spiceinit(work: Path):
    work.mkdir(parents=True, exist_ok=True)
    (work / ".spiceinit").write_text("set ngbehavior=hsa\nset ng_nomodcheck\n")


def run_ngspice(cir: Path, timeout: float = 7200) -> str:
    """Run one netlist in its own directory (which holds .spiceinit) and return the log."""
    work = cir.parent
    if not (work / ".spiceinit").exists():
        write_spiceinit(work)
    proc = subprocess.run(["ngspice", "-b", cir.name], cwd=work, capture_output=True,
                          text=True, timeout=timeout)
    log = proc.stdout + proc.stderr
    (work / (cir.stem + ".log")).write_text(log)
    if proc.returncode != 0 or re.search(r"(?im)^\s*error", log):
        raise RuntimeError(f"ngspice failed on {cir.name} (code {proc.returncode}); see log")
    return log


def read_raw(path: Path) -> dict:
    """Binary ngspice raw file (real data) -> {variable name: array}."""
    data = path.read_bytes()
    pos = data.index(b"Binary:\n")
    head = data[:pos].decode("latin-1").splitlines()
    n_var = n_pts = None
    names = []
    in_vars = False
    for ln in head:
        if ln.startswith("No. Variables:"):
            n_var = int(ln.split(":")[1])
        elif ln.startswith("No. Points:"):
            n_pts = int(ln.split(":")[1])
        elif ln.startswith("Flags:"):
            assert "complex" not in ln
        elif ln.startswith("Variables:"):
            in_vars = True
        elif in_vars and ln.strip():
            parts = ln.split()
            names.append(parts[1].lower())
    assert n_var == len(names)
    arr = np.frombuffer(data[pos + len(b"Binary:\n"):], dtype="<f8")
    arr = arr[: n_var * n_pts].reshape(n_pts, n_var)
    return {nm: arr[:, i].copy() for i, nm in enumerate(names)}


def meas_values(log: str) -> dict:
    """Values printed by 'meas' commands (name = value)."""
    out = {}
    for m in re.finditer(r"(?m)^\s*([A-Za-z_][\w]*)\s*=\s*([-+0-9.eE]+)", log):
        try:
            out[m.group(1).lower()] = float(m.group(2))
        except ValueError:
            pass
    return out
