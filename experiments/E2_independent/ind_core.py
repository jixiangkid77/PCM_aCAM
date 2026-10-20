"""Core simulation primitives for the E2 independent check.

All circuits are simulated with ngspice (one job at a time).  Two families:

1. Single-device circuits (every terminal driven by a source): drain current I_D(V_G, V_DS).
   - iv_sweep: shared gate sweep 0..1.8 V (0.5 mV) for many devices
   - solve_vg: gate voltage giving a target current, coarse sweep + local cubic,
     then refined with a per-device offset sweep (+-w around the coarse root, fine step)
   - op_currents: exact-voltage operating points (no interpolation)

2. Full-branch circuits: PCM = linear resistor 1/G from SL to node X, divider nMOS from X
   to ground (body grounded), gate on DL.  The bound is the V_DL at which V_X equals a
   threshold (the instance's pull-down switching voltage).  No I-V inversion is used:
   - branch_bounds: coarse shared DL sweep 0..1.8 V (0.5 mV), crossing V_X = thr,
     then a per-branch offset sweep (gate = s + v0_j, +-w, fine step) and crossing again.
"""
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq

import ind_spice as S

VG_START, VG_STOP, VG_STEP = 0.0, 1.8, 0.0005


def _cross(x, y, target, increasing=True, nloc=4):
    """Single crossing of y(x) = target, local cubic spline + brentq.  Requires strict
    monotonicity of y over the whole array (checked)."""
    d = np.diff(y)
    if increasing:
        if not np.all(d > 0):
            raise ValueError("not strictly increasing")
        k = np.searchsorted(y, target)
    else:
        if not np.all(d < 0):
            raise ValueError("not strictly decreasing")
        k = np.searchsorted(-y, -target)
    if k <= 0 or k >= len(y):
        raise ValueError(f"target {target} outside range [{y.min()}, {y.max()}]")
    lo, hi = max(0, k - nloc), min(len(y), k + nloc)
    cs = CubicSpline(x[lo:hi], y[lo:hi])
    root = brentq(lambda v: float(cs(v)) - target, x[k - 1], x[k], xtol=1e-16, rtol=1e-15)
    # linear-interpolation estimate for diagnostics
    lin = x[k - 1] + (target - y[k - 1]) / (y[k] - y[k - 1]) * (x[k] - x[k - 1])
    slope = (y[k] - y[k - 1]) / (x[k] - x[k - 1])
    return root, lin, slope


# ---------------------------------------------------------------- single device

def iv_sweep(corner, temp_c, devs, workdir, name):
    """devs: list of dict(model, w, l, mm, vds).  Shared gate sweep 0..1.8 V, 0.5 mV.
    Returns (vg, I) with I shape (n_dev, n_pts), I = drain current (A, positive)."""
    L = S.header(corner, temp_c, f"iv sweep {name}")
    L.append("vg g 0 0")
    for j, d in enumerate(devs):
        L.append(f"vd{j} d{j} 0 {S.fmt(d['vds'])}")
        L.append(S.dev(f"m{j}", f"d{j}", "g", "0", "0", d["model"], d["w"], d["l"], d.get("mm")))
    L.append(".save " + " ".join(f"i(vd{j})" for j in range(len(devs))))
    L.append(f".dc vg {VG_START} {VG_STOP} {VG_STEP}")
    p = S.run_ngspice(L, workdir, name)[0]
    vg = p["v(v-sweep)"]
    I = np.array([-p[f"i(vd{j})"] for j in range(len(devs))])
    return vg, I


def op_currents(corner, temp_c, devs, workdir, name):
    """devs: list of dict(model, w, l, mm, vg, vds).  Operating point, exact voltages.
    Returns drain currents (A, positive)."""
    L = S.header(corner, temp_c, f"op {name}")
    for j, d in enumerate(devs):
        L.append(f"vg{j} g{j} 0 {S.fmt(d['vg'])}")
        L.append(f"vd{j} d{j} 0 {S.fmt(d['vds'])}")
        L.append(S.dev(f"m{j}", f"d{j}", f"g{j}", "0", "0", d["model"], d["w"], d["l"], d.get("mm")))
    L.append(".save " + " ".join(f"i(vd{j})" for j in range(len(devs))))
    L.append(".op")
    p = S.run_ngspice(L, workdir, name)[0]
    return np.array([-p[f"i(vd{j})"][0] for j in range(len(devs))])


def solve_vg(corner, temp_c, devs, targets, workdir, name, w=40e-6, step=0.1e-6):
    """Gate voltage at which each device (dict model,w,l,mm,vds) conducts targets[j] (A).
    Stage 1: shared sweep + local cubic on ln I.  Stage 2: offset sweep +-w around the
    stage-1 root with a fine step, local cubic on ln I against the saved gate voltage.
    Returns dict(root, coarse_lin, coarse_cubic, stage2_minus_stage1, resid_lnI)."""
    targets = np.asarray(targets, float)
    vg, I = iv_sweep(corner, temp_c, devs, workdir, name + "_coarse")
    c_cub = np.empty(len(devs)); c_lin = np.empty(len(devs))
    for j in range(len(devs)):
        r, lin, _ = _cross(vg, np.log(I[j]), np.log(targets[j]), increasing=True)
        c_cub[j], c_lin[j] = r, lin
    # stage 2
    L = S.header(corner, temp_c, f"refine {name}")
    L.append("vs s 0 0")
    for j, d in enumerate(devs):
        L.append(f"vo{j} g{j} s {S.fmt(c_cub[j])}")
        L.append(f"vd{j} d{j} 0 {S.fmt(d['vds'])}")
        L.append(S.dev(f"m{j}", f"d{j}", f"g{j}", "0", "0", d["model"], d["w"], d["l"], d.get("mm")))
    L.append(".save " + " ".join(f"v(g{j}) i(vd{j})" for j in range(len(devs))))
    L.append(f".dc vs {S.fmt(-w)} {S.fmt(w)} {S.fmt(step)}")
    p = S.run_ngspice(L, workdir, name + "_refine")[0]
    root = np.empty(len(devs))
    for j in range(len(devs)):
        x = p[f"v(g{j})"]
        y = np.log(-p[f"i(vd{j})"])
        root[j], _, _ = _cross(x, y, np.log(targets[j]), increasing=True)
    # residual check at the refined root (exact-voltage operating point)
    odevs = [dict(d, vg=root[j]) for j, d in enumerate(devs)]
    Iop = op_currents(corner, temp_c, odevs, workdir, name + "_check")
    resid = np.log(Iop) - np.log(targets)
    return dict(root=root, coarse_lin=c_lin, coarse_cubic=c_cub,
                stage2_minus_stage1=root - c_cub, resid_lnI=resid)


# ---------------------------------------------------------------- full branch

def _branch_lines(j, br, gate_node):
    """PCM resistor 1/G from SL_j to X_j, divider from X_j to ground, gate on gate_node."""
    return [f"vsl{j} sl{j} 0 {S.fmt(br['vsl'])}",
            f"r{j} sl{j} x{j} {S.fmt(1.0 / br['G'])}",
            S.dev(f"m{j}", f"x{j}", gate_node, "0", "0", br["model"], br["w"], br["l"], br.get("mm"))]


def branch_bounds(corner, temp_c, branches, workdir, name, w=100e-6, step=0.2e-6):
    """branches: list of dict(G [S], vsl [V], thr [V], model, w, l, mm).
    Returns dict(bound, coarse, dvx_dvdl, stage2_minus_stage1, kcl_resid)."""
    n = len(branches)
    # stage 1: shared DL sweep
    L = S.header(corner, temp_c, f"branch coarse {name}")
    L.append("vdl dl 0 0")
    for j, br in enumerate(branches):
        L += _branch_lines(j, br, "dl")
    L.append(".save " + " ".join(f"v(x{j})" for j in range(n)))
    L.append(f".dc vdl {VG_START} {VG_STOP} {VG_STEP}")
    p = S.run_ngspice(L, workdir, name + "_coarse")[0]
    vdl = p["v(v-sweep)"]
    coarse = np.empty(n)
    for j, br in enumerate(branches):
        vx = p[f"v(x{j})"]
        coarse[j], _, _ = _cross(vdl, vx, br["thr"], increasing=False)
    # stage 2: per-branch offset sweep around the coarse bound
    L = S.header(corner, temp_c, f"branch refine {name}")
    L.append("vs s 0 0")
    for j, br in enumerate(branches):
        L.append(f"vo{j} g{j} s {S.fmt(coarse[j])}")
        L += _branch_lines(j, br, f"g{j}")
    L.append(".save " + " ".join(f"v(g{j}) v(x{j})" for j in range(n)))
    L.append(f".dc vs {S.fmt(-w)} {S.fmt(w)} {S.fmt(step)}")
    p = S.run_ngspice(L, workdir, name + "_refine")[0]
    bound = np.empty(n); slope = np.empty(n)
    for j, br in enumerate(branches):
        x = p[f"v(g{j})"]; vx = p[f"v(x{j})"]
        bound[j], _, slope[j] = _cross(x, vx, br["thr"], increasing=False)
    # Diagnostic only (not used for the bound): the single device at (V_G = bound,
    # V_DS = thr) must carry the PCM current G (V_SL - thr).
    odevs = [dict(model=br["model"], w=br["w"], l=br["l"], mm=br.get("mm"),
                  vg=bound[j], vds=br["thr"]) for j, br in enumerate(branches)]
    Iop = op_currents(corner, temp_c, odevs, workdir, name + "_opcheck")
    Ipcm = np.array([br["G"] * (br["vsl"] - br["thr"]) for br in branches])
    resid = np.log(Iop) - np.log(Ipcm)
    return dict(bound=bound, coarse=coarse, dvx_dvdl=slope,
                stage2_minus_stage1=bound - coarse, op_resid_lnI=resid)
