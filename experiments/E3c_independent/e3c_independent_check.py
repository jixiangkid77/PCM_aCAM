#!/usr/bin/env python3
"""E3c independent check (design P1 only).

Re-implemented from the frozen protocol docs/E3c_protocol.md (sections 2, 3, 4, 8),
experiments/E3c_config.json and the lambda definition of docs/E2_protocol.md section 2.
No code from src/ or from any other experiment script is used.  See README.md.

Stages (all ngspice jobs are run strictly one after another):
  1. V_sw of the LVT pull-down (100 nA * W/L at V_DS = 1.8 V).
  2. t_s(1), t_s(16): ML released from 1.8 V at t = 0, time of the 0.9 V fall crossing.
  3. Divider map / lambda; leaker V_X,j (single-branch DC operating point, polished by Newton
     with fixed-bias device evaluations); pull-down-only transient switching voltages with and
     without leakers -> delta_l, false-mismatch flags; predicted shifts.
  4. Full-row transient (CUT and leakers as full branches): bound on the CUT input by bisection,
     exact inverse of the nominal map -> input-referred error, shift = error(with) - error(isolated).
  5. Comparison with the main run (predictions.json, t3_crosstalk.csv, t2_isolated.csv).
"""
import argparse
import csv
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CFG_PATH = ROOT / "experiments" / "E3c_config.json"
LIB = ROOT / "third_party" / "sky130_fd_pr_subset" / "sky130_subset_tt.lib.spice"
WORK_BASE = ROOT / "logs" / "E3c_independent_work"
RESULTS_BASE = ROOT / "results"
MAIN_PRED = ROOT / "results" / "E3c_predict_20261014_1543" / "predictions.json"
MAIN_SIM = ROOT / "results" / "E3c_20261014_1549"
JST = timezone(timedelta(hours=9))

# ----------------------------------------------------------------------------------------------
# constants from the frozen configuration
# ----------------------------------------------------------------------------------------------
CFG = json.loads(CFG_PATH.read_text())
CIRC = CFG["circuit"]
MLC = CFG["ml"]
VDD = float(CIRC["vdd"])
D = float(CIRC["d_v"])
G_LO = CIRC["g_lo_us"] * 1e-6
G_HI = CIRC["g_hi_us"] * 1e-6
DG = G_HI - G_LO
PD_DEV = "sky130_fd_pr__" + CIRC["pulldown_device"]
DIV_DEV = "sky130_fd_pr__" + CIRC["divider_device"]
PD_W = CIRC["pulldown"]["w_um"]
PD_L = CIRC["pulldown"]["l_um"]
P1 = next(d for d in CFG["designs"] if d["id"] == "P1")
DIV_W = P1["w_div_um"]
DIV_L = P1["l_div_um"]
I_SW = 100e-9 * PD_W / PD_L
V_PRE = float(MLC["v_pre"])
V_REF = float(MLC["v_ref"])
C_SA = float(MLC["c_sa_f"])
C_WIRE = float(MLC["c_wire_per_cell_f"])
T_STEP = float(MLC["t_step_s"])
T_MAX = float(MLC["t_max_s"])
T_STOP_FACTOR = float(MLC["t_stop_factor"])
NU = CFG["drift"]["nu"]
A = (CFG["drift"]["times_s"]["1y"] / CFG["drift"]["t0_s"]) ** (-NU)
N_CELLS = int(CFG["crosstalk"]["n_cells"])
THETA_LEAK = float(CFG["crosstalk"]["leaker_theta"])
CONFIGS = {c["id"]: c for c in CFG["crosstalk"]["configs"]}
CHECK_CONFIGS = ["k4d6", "k15d7.5"]
NODES = list(CFG["nodes"])
THETAS = [float(t) for t in CFG["thetas"]]
FULL_THETAS = [0.0, 1.0]
LAMBDA_DV = 0.005           # E2 protocol section 2: V_DS +- 5 mV central difference
BISECT_TOL_V = 1e-7         # final bracket width (< 1 uV requested)
BISECT_NPTS = 3             # interior points per bracket and run (bracket shrinks 4x per run)

# thresholds of the independent check (task / protocol section 8)
TOL_TS_REL = 1e-4
TOL_VX_V = 1e-5
TOL_DL_V = 1e-5
TOL_SHIFT_LSB = 0.01


def G_of(x):
    return G_LO + DG * x


def node_params(node):
    """gf: PCM conductance factor; vsl_off: V_SL - V_sw; mapf: input-map current factor; kappa."""
    if node == "t0":
        return dict(gf=1.0, vsl_off=D, mapf=1.0, kappa=1.0)
    if node == "dac_1y":
        return dict(gf=A, vsl_off=D, mapf=A, kappa=1.0)
    if node == "supply_1y":
        return dict(gf=A, vsl_off=D / A, mapf=1.0, kappa=A)
    raise ValueError(node)


def fmt(v):
    return repr(float(v))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ----------------------------------------------------------------------------------------------
# ngspice plumbing
# ----------------------------------------------------------------------------------------------
class Spice:
    BAD = ("error", "timestep too small", "singular", "aborted", "no convergence", "fatal")

    def __init__(self, workdir):
        self.work = Path(workdir)
        self.work.mkdir(parents=True, exist_ok=False)
        self.n = 0
        self.runs = []

    def run(self, tag, body, control):
        self.n += 1
        d = self.work / f"{self.n:03d}_{tag}"
        d.mkdir(parents=True)
        (d / ".spiceinit").write_text("set ngbehavior=hsa\nset ng_nomodcheck\n")
        net = [f"* E3c independent check: {tag}", f'.include "{LIB}"', ".temp 27"] + body
        net += [".control", "set filetype=ascii"] + control + [".endc", ".end"]
        (d / "run.cir").write_text("\n".join(net) + "\n")
        t0 = time.time()
        p = subprocess.run(["ngspice", "-b", "run.cir"], cwd=d, capture_output=True, text=True)
        dt = time.time() - t0
        (d / "ngspice.log").write_text(p.stdout + "\n----- stderr -----\n" + p.stderr)
        low = (p.stdout + p.stderr).lower()
        bad = [b for b in self.BAD if b in low]
        self.runs.append(dict(n=self.n, tag=tag, seconds=round(dt, 3), returncode=p.returncode, flags=bad))
        if p.returncode != 0 or bad:
            raise RuntimeError(f"ngspice run {d} failed: rc={p.returncode} flags={bad}")
        print(f"  [ngspice {self.n:03d}] {tag}: {dt:.1f} s", flush=True)
        return d


def parse_raw(path):
    txt = Path(path).read_text()
    hdr, vals = txt.split("Values:\n")
    var_block = hdr.split("Variables:\n")[1].strip("\n").split("\n")
    names = [line.split("\t")[2] for line in var_block]
    arr = np.array(vals.split(), dtype=float).reshape(-1, len(names) + 1)[:, 1:]
    return names, arr


def first_fall_crossing(t, v, vref):
    """first falling crossing through vref with linear interpolation (as .meas WHEN ... FALL=1)."""
    idx = np.nonzero((v[1:] <= vref) & (v[:-1] > vref))[0]
    if len(idx) == 0:
        return math.inf
    i = idx[0] + 1
    return float(t[i - 1] + (vref - v[i - 1]) * (t[i] - t[i - 1]) / (v[i] - v[i - 1]))


# ----------------------------------------------------------------------------------------------
# DC helpers (fixed-bias device evaluations)
# ----------------------------------------------------------------------------------------------
def dev_currents(sp, pts, tag):
    """pts: list of (dev, w, l, vg, vd); source and body at 0 V. Returns drain currents (A)."""
    body = []
    for i, (dev, w, l, vg, vd) in enumerate(pts):
        body += [f"Vd{i} d{i} 0 {fmt(vd)}", f"Vg{i} g{i} 0 {fmt(vg)}",
                 f"X{i} d{i} g{i} 0 0 {dev} w={w} l={l}"]
    d = sp.run(tag, body, ["op", "write op.raw"])
    names, arr = parse_raw(d / "op.raw")
    return np.array([-arr[0, names.index(f"i(vd{i})")] for i in range(len(pts))])


def dev_sweep(sp, dev, w, l, vd, tag, v0=0.0, v1=None, step=0.0005):
    v1 = VDD if v1 is None else v1
    body = [f"Vd d 0 {fmt(vd)}", "Vg g 0 0", f"X1 d g 0 0 {dev} w={w} l={l}"]
    d = sp.run(tag, body, [f"dc Vg {fmt(v0)} {fmt(v1)} {fmt(step)}", "write dc.raw"])
    names, arr = parse_raw(d / "dc.raw")
    return arr[:, names.index("v(g)")], -arr[:, names.index("i(vd)")]


def gate_for_currents(sp, dev, w, l, vd, targets, tag):
    """Gate voltages at which the nominal device at V_DS = vd conducts each target current.
    Initial value: log-linear interpolation of a 0.5 mV sweep; then Newton on ln I with
    fixed-bias operating points (h = 1 uV) until the update is < 1e-12 V."""
    targets = np.asarray(targets, dtype=float)
    vg, cur = dev_sweep(sp, dev, w, l, vd, tag + "_sweep")
    lnc = np.log(np.abs(cur) + 1e-300)
    est = []
    for It in targets:
        lt = math.log(It)
        idx = np.nonzero((lnc[:-1] < lt) & (lnc[1:] >= lt))[0]
        if len(idx) != 1:
            raise RuntimeError(f"non-monotonic or out-of-range sweep for target {It}")
        k = idx[0]
        est.append(vg[k] + (lt - lnc[k]) * (vg[k + 1] - vg[k]) / (lnc[k + 1] - lnc[k]))
    est = np.array(est)
    h = 1e-6
    n = len(est)
    for it in range(8):
        pts = [(dev, w, l, v, vd) for v in est] + [(dev, w, l, v + h, vd) for v in est]
        c = dev_currents(sp, pts, f"{tag}_newton{it}")
        f0 = np.log(c[:n])
        slope = (np.log(c[n:]) - f0) / h
        step = (np.log(targets) - f0) / slope
        est = est + step
        if np.max(np.abs(step)) < 1e-12:
            break
    c = dev_currents(sp, [(dev, w, l, v, vd) for v in est], f"{tag}_verify")
    return est, c / targets - 1.0


def branch_vx(sp, branches, tag):
    """Divider node voltage of single branches (V_SL source - PCM 1/G - X - divider - ground).
    Returns (ngspice DC operating point, Newton-polished root of G (V_SL - X) = I_div(V_DL, X))."""
    body = []
    for i, (vsl, g, vdl) in enumerate(branches):
        body += [f"Vsl{i} sl{i} 0 {fmt(vsl)}", f"R{i} sl{i} x{i} {fmt(1.0 / g)}",
                 f"Xd{i} x{i} dl{i} 0 0 {DIV_DEV} w={DIV_W} l={DIV_L}", f"Vdl{i} dl{i} 0 {fmt(vdl)}"]
    d = sp.run(tag + "_op", body, ["op", "write op.raw"])
    names, arr = parse_raw(d / "op.raw")
    vx_op = np.array([arr[0, names.index(f"v(x{i})")] for i in range(len(branches))])
    vsl = np.array([b[0] for b in branches])
    g = np.array([b[1] for b in branches])
    vdl = np.array([b[2] for b in branches])
    vx = vx_op.copy()
    h = 1e-6
    n = len(vx)
    resid = None
    for it in range(8):
        pts = [(DIV_DEV, DIV_W, DIV_L, a, b) for a, b in zip(vdl, vx)]
        pts += [(DIV_DEV, DIV_W, DIV_L, a, b + h) for a, b in zip(vdl, vx)]
        c = dev_currents(sp, pts, f"{tag}_polish{it}")
        f = g * (vsl - vx) - c[:n]
        fp = -g - (c[n:] - c[:n]) / h
        step = -f / fp
        vx = vx + step
        resid = f
        if np.max(np.abs(step)) < 1e-12:
            break
    return vx_op, vx, resid


# ----------------------------------------------------------------------------------------------
# match-line transients
# ----------------------------------------------------------------------------------------------
def row_spec(n_cells, cut, leakers=()):
    """Gate drivers of the 2N pull-downs on one ML: CUT lower, k leaker lowers, remaining lowers
    at gate 0, N upper-branch pull-downs at gate 0.  Driver: ('v', V) ideal gate source,
    ('br', V_SL, G_pcm, V_DL) full branch (X node drives the gate), ('z',) gate at 0 V."""
    leakers = list(leakers)
    gates = [cut] + leakers + [("z",)] * (n_cells - 1 - len(leakers)) + [("z",)] * n_cells
    return dict(C=C_SA + n_cells * C_WIRE, gates=gates)


def copy_lines(c, spec):
    lines = [f"C{c} ml{c} 0 {fmt(spec['C'])}"]
    for p, gd in enumerate(spec["gates"]):
        if gd[0] == "z":
            gnode = "0"
        elif gd[0] == "v":
            gnode = f"g{c}_{p}"
            lines.append(f"Vg{c}_{p} {gnode} 0 {fmt(gd[1])}")
        elif gd[0] == "br":
            _, vsl, g, vdl = gd
            gnode = f"x{c}_{p}"
            lines += [f"Vsl{c}_{p} sl{c}_{p} 0 {fmt(vsl)}", f"R{c}_{p} sl{c}_{p} {gnode} {fmt(1.0 / g)}",
                      f"Xd{c}_{p} {gnode} dl{c}_{p} 0 0 {DIV_DEV} w={DIV_W} l={DIV_L}",
                      f"Vdl{c}_{p} dl{c}_{p} 0 {fmt(vdl)}"]
        else:
            raise ValueError(gd)
        lines.append(f"Xp{c}_{p} ml{c} {gnode} 0 0 {PD_DEV} w={PD_W} l={PD_L}")
    lines += [f".ic v(ml{c})={fmt(V_PRE)}", f".save v(ml{c})"]
    return lines


def crossing_times(sp, specs, tstop, tag):
    body = []
    for c, s in enumerate(specs):
        body += copy_lines(c, s)
    ctrl = [f"tran {fmt(T_STEP)} {fmt(tstop)} 0 {fmt(T_MAX)}", "write tran.raw"]
    d = sp.run(tag, body, ctrl)
    names, arr = parse_raw(d / "tran.raw")
    t = arr[:, names.index("time")]
    out = [first_fall_crossing(t, arr[:, names.index(f"v(ml{c})")], V_REF) for c in range(len(specs))]
    (d / "tran.raw").unlink()  # keep netlist and log only
    return np.array(out), len(t)


def bisect_cases(sp, cases, tstop, tag, tol=BISECT_TOL_V, npts=BISECT_NPTS):
    """Solve t_cross(v) = ts per case by bracketing (t_cross monotone in v; either direction).
    cases: dicts with name, lo, hi, ts, build(v) -> row spec. All cases share each ngspice run."""
    specs = []
    for cs in cases:
        specs += [cs["build"](cs["lo"]), cs["build"](cs["hi"])]
    tc, _ = crossing_times(sp, specs, tstop, f"{tag}_it00")
    st = {}
    for i, cs in enumerate(cases):
        t_lo, t_hi = tc[2 * i], tc[2 * i + 1]
        s_lo, s_hi = np.sign(t_lo - cs["ts"]), np.sign(t_hi - cs["ts"])
        if s_lo == s_hi or s_lo == 0 or s_hi == 0:
            raise RuntimeError(f"initial bracket does not contain the root: {cs['name']} {t_lo} {t_hi}")
        st[cs["name"]] = dict(lo=cs["lo"], hi=cs["hi"], t_lo=t_lo, t_hi=t_hi, s_lo=s_lo, runs=1)
    it = 0
    while True:
        active = [cs for cs in cases if st[cs["name"]]["hi"] - st[cs["name"]]["lo"] > tol]
        if not active:
            break
        it += 1
        specs, plist = [], []
        for cs in active:
            s = st[cs["name"]]
            pts = [s["lo"] + (s["hi"] - s["lo"]) * (j + 1) / (npts + 1) for j in range(npts)]
            plist.append(pts)
            specs += [cs["build"](v) for v in pts]
        tc, _ = crossing_times(sp, specs, tstop, f"{tag}_it{it:02d}")
        k = 0
        for cs, pts in zip(active, plist):
            s = st[cs["name"]]
            tcs = list(tc[k:k + npts])
            k += npts
            xs = [s["lo"]] + pts + [s["hi"]]
            tv = [s["t_lo"]] + tcs + [s["t_hi"]]
            for j in range(len(xs) - 1):
                sg = np.sign(tv[j + 1] - cs["ts"])
                if sg == 0:
                    s["lo"] = s["hi"] = xs[j + 1]
                    s["t_lo"] = s["t_hi"] = tv[j + 1]
                    break
                if sg != s["s_lo"]:
                    s["lo"], s["hi"], s["t_lo"], s["t_hi"] = xs[j], xs[j + 1], tv[j], tv[j + 1]
                    break
            else:
                raise RuntimeError(f"lost the bracket for {cs['name']}")
            s["runs"] += 1
    out = {}
    for cs in cases:
        s = st[cs["name"]]
        mid = 0.5 * (s["lo"] + s["hi"])
        interp = mid
        if math.isfinite(s["t_lo"]) and math.isfinite(s["t_hi"]) and s["t_hi"] != s["t_lo"]:
            interp = s["lo"] + (cs["ts"] - s["t_lo"]) * (s["hi"] - s["lo"]) / (s["t_hi"] - s["t_lo"])
        out[cs["name"]] = dict(v=mid, v_interp=interp, lo=s["lo"], hi=s["hi"], width=s["hi"] - s["lo"],
                               t_lo=s["t_lo"], t_hi=s["t_hi"], runs=s["runs"])
    return out


# ----------------------------------------------------------------------------------------------
# main computation
# ----------------------------------------------------------------------------------------------
def compute(sp, log):
    R = dict(constants=dict(VDD=VDD, D=D, G_lo=G_LO, G_hi=G_HI, I_sw=I_SW, a=A, C_SA=C_SA, C_wire=C_WIRE,
                            t_step=T_STEP, t_max=T_MAX, t_stop_factor=T_STOP_FACTOR, n_cells=N_CELLS,
                            pulldown=f"{PD_DEV} {PD_W}/{PD_L}", divider=f"{DIV_DEV} {DIV_W}/{DIV_L}",
                            bisect_tol_V=BISECT_TOL_V, bisect_npts=BISECT_NPTS))
    # 1. V_sw ------------------------------------------------------------------------------
    log("stage 1: V_sw")
    v, rel = gate_for_currents(sp, PD_DEV, PD_W, PD_L, VDD, [I_SW], "vsw")
    vsw = float(v[0])
    R["vsw_V"] = vsw
    R["vsw_current_relerr"] = float(rel[0])
    log(f"  V_sw = {vsw:.12f} V (I rel err {rel[0]:.2e})")

    # 2. t_s ---------------------------------------------------------------------------------
    log("stage 2: t_s(1), t_s(16)")
    ts = {}
    for n in (1, N_CELLS):
        spec = row_spec(n, ("v", vsw))
        t_first = crossing_times(sp, [spec], 2e-6, f"ts{n}_pre")[0][0]
        t_fin, npt = crossing_times(sp, [spec], T_STOP_FACTOR * t_first, f"ts{n}")
        ts[n] = float(t_fin[0])
        R.setdefault("t_s", {})[str(n)] = dict(t_s_s=ts[n], t_s_pre_s=float(t_first), n_time_points=npt,
                                                 C_F=spec["C"], n_pulldowns=len(spec["gates"]))
        log(f"  t_s({n}) = {ts[n]:.10e} s (pre-run with 2 us stop: {t_first:.10e})")
    tstop = T_STOP_FACTOR * ts[N_CELLS]
    R["tstop_s"] = tstop

    # 3a. divider map and lambda ---------------------------------------------------------------
    log("stage 3: divider map, lambda, leaker V_X")
    targets = {}
    for th in THETAS:
        targets[("I", th)] = G_of(th) * D
        targets[("aI", th)] = A * G_of(th) * D
    for cid in CHECK_CONFIGS:
        xj = THETA_LEAK + CONFIGS[cid]["d_lsb"] / 15.0
        targets[("leak_I", cid)] = G_of(xj) * D
        targets[("leak_aI", cid)] = A * G_of(xj) * D
    keys = list(targets)
    vg_map, rel = gate_for_currents(sp, DIV_DEV, DIV_W, DIV_L, vsw, [targets[k] for k in keys], "divmap")
    gate = {k: float(v) for k, v in zip(keys, vg_map)}
    R["divider_map"] = [dict(key=f"{k[0]}|{k[1]}", current_A=targets[k], vdl_V=gate[k], rel_err=float(e))
                        for k, e in zip(keys, rel)]
    lam_keys = [k for k in keys if k[0] in ("I", "aI")]
    pts = [(DIV_DEV, DIV_W, DIV_L, gate[k], vsw + LAMBDA_DV) for k in lam_keys]
    pts += [(DIV_DEV, DIV_W, DIV_L, gate[k], vsw - LAMBDA_DV) for k in lam_keys]
    c = dev_currents(sp, pts, "lambda")
    nl = len(lam_keys)
    lam = {k: float(-(math.log(c[i]) - math.log(c[nl + i])) / (2 * LAMBDA_DV)) for i, k in enumerate(lam_keys)}
    R["lambda_per_V"] = {"I": [lam[("I", th)] for th in THETAS], "aI": [lam[("aI", th)] for th in THETAS]}
    log(f"  lambda(I)  = {R['lambda_per_V']['I']}")
    log(f"  lambda(aI) = {R['lambda_per_V']['aI']}")

    # 3b. leaker V_X ---------------------------------------------------------------------------
    combos = [(cid, node) for cid in CHECK_CONFIGS for node in NODES]
    branches, leak = [], {}
    for cid, node in combos:
        p = node_params(node)
        vsl = vsw + p["vsl_off"]
        gj = p["gf"] * G_of(THETA_LEAK)
        vdl = gate[("leak_aI", cid)] if node == "dac_1y" else gate[("leak_I", cid)]
        leak[(cid, node)] = dict(vsl=vsl, g=gj, vdl=vdl, k=int(CONFIGS[cid]["k"]),
                                 x_j=THETA_LEAK + CONFIGS[cid]["d_lsb"] / 15.0)
        branches.append((vsl, gj, vdl))
    vx_op, vx, resid = branch_vx(sp, branches, "leaker_vx")
    for i, cn in enumerate(combos):
        leak[cn].update(vx=float(vx[i]), vx_op=float(vx_op[i]), resid_A=float(resid[i]))
        log(f"  {cn}: V_X = {vx[i]:.9f} V (op {vx_op[i]:.9f}; V_X - V_sw = {1e3 * (vx[i] - vsw):.4f} mV)")

    # 3c. false mismatch (pull-down only, CUT gate 0) -------------------------------------------
    specs = [row_spec(N_CELLS, ("z",), [("v", leak[cn]["vx"])] * leak[cn]["k"]) for cn in combos]
    # same with leakers as full branches (protocol check C2 style)
    specs += [row_spec(N_CELLS, ("z",), [("br", leak[cn]["vsl"], leak[cn]["g"], leak[cn]["vdl"])] * leak[cn]["k"])
              for cn in combos]
    tco, _ = crossing_times(sp, specs, tstop, "cut_off")
    for i, cn in enumerate(combos):
        leak[cn]["t_cross_cut_off_s"] = float(tco[i])
        leak[cn]["t_cross_cut_off_fullbranch_s"] = float(tco[len(combos) + i])
        leak[cn]["false_mismatch"] = bool(tco[i] < ts[N_CELLS])
        leak[cn]["false_mismatch_fullbranch"] = bool(tco[len(combos) + i] < ts[N_CELLS])
        log(f"  {cn}: CUT off crossing {tco[i]:.6e} s (full-branch leakers {tco[len(combos) + i]:.6e}); "
            f"false mismatch = {leak[cn]['false_mismatch']}")

    # 3d. pull-down-only transient switching voltages ---------------------------------------
    log("stage 3d: pull-down-only bisection")
    cases = [dict(name="iso16", lo=vsw - 0.05, hi=vsw + 0.05, ts=ts[N_CELLS],
                  build=lambda v: row_spec(N_CELLS, ("v", v))),
             dict(name="iso1", lo=vsw - 0.05, hi=vsw + 0.05, ts=ts[1],
                  build=lambda v: row_spec(1, ("v", v)))]
    for cn in combos:
        if leak[cn]["false_mismatch"]:
            continue
        lk = [("v", leak[cn]["vx"])] * leak[cn]["k"]
        cases.append(dict(name=f"{cn[0]}|{cn[1]}", lo=vsw - 0.3, hi=vsw + 0.02, ts=ts[N_CELLS],
                          build=(lambda lk_: (lambda v: row_spec(N_CELLS, ("v", v), lk_)))(lk)))
    bres = bisect_cases(sp, cases, tstop, "pd")
    R["pulldown_bisection"] = bres
    vtr16 = bres["iso16"]["v"]
    R["vsw_tr_nominal_V"] = {"1": bres["iso1"]["v"], "16": vtr16}
    log(f"  V_sw,tr(16) = {vtr16:.10f} (V_sw {vsw:.10f}, diff {1e6 * (vtr16 - vsw):.3f} uV); "
        f"V_sw,tr(1) = {bres['iso1']['v']:.10f}")
    for cn in combos:
        L = leak[cn]
        p = node_params(cn[1])
        L["kappa"] = p["kappa"]
        if L["false_mismatch"]:
            L["vsw_tr_with_leakers_V"] = None
            L["delta_l_V"] = None
            L["pred_shift_lsb"] = None
            continue
        b = bres[f"{cn[0]}|{cn[1]}"]
        L["vsw_tr_with_leakers_V"] = b["v"]
        L["delta_l_V"] = b["v"] - vtr16
        lam_list = R["lambda_per_V"]["aI" if cn[1] == "dac_1y" else "I"]
        L["lambda_b_per_V"] = lam_list
        L["pred_shift_lsb"] = [15.0 * (G_of(th) / DG) * ((1.0 - p["kappa"] * L["delta_l_V"] / D)
                                                          * math.exp(lb * L["delta_l_V"]) - 1.0)
                               for th, lb in zip(THETAS, lam_list)]
        log(f"  {cn}: delta_l = {1e3 * L['delta_l_V']:.5f} mV; pred shift = "
            + ", ".join(f"{s:.5f}" for s in L["pred_shift_lsb"]))

    # 4. full-row transient -----------------------------------------------------------------
    log("stage 4: full-row bisection")
    vdl0 = {}
    for node in NODES:
        for th in FULL_THETAS:
            vdl0[(node, th)] = gate[("aI", th)] if node == "dac_1y" else gate[("I", th)]
    cases = []
    for node in NODES:
        p = node_params(node)
        vsl = vsw + p["vsl_off"]
        for th in FULL_THETAS:
            gc = p["gf"] * G_of(th)
            v0 = vdl0[(node, th)]
            cases.append(dict(name=f"iso|{node}|{th}", lo=v0 - 0.05, hi=v0 + 0.05, ts=ts[N_CELLS],
                              build=(lambda vsl_, gc_: (lambda v: row_spec(N_CELLS, ("br", vsl_, gc_, v))))(vsl, gc)))
            for cid in CHECK_CONFIGS:
                L = leak[(cid, node)]
                if L["false_mismatch"]:
                    continue
                lk = [("br", L["vsl"], L["g"], L["vdl"])] * L["k"]
                cases.append(dict(name=f"{cid}|{node}|{th}", lo=v0 - 0.05, hi=v0 + 0.4, ts=ts[N_CELLS],
                                  build=(lambda vsl_, gc_, lk_: (lambda v: row_spec(N_CELLS, ("br", vsl_, gc_, v), lk_)))(vsl, gc, lk)))
    fres = bisect_cases(sp, cases, tstop, "full")
    R["fullrow_bisection"] = fres
    names = [cs["name"] for cs in cases]
    pts = [(DIV_DEV, DIV_W, DIV_L, fres[nm]["v"], vsw) for nm in names]
    pts += [(DIV_DEV, DIV_W, DIV_L, fres[nm]["v_interp"], vsw) for nm in names]
    i0 = dev_currents(sp, pts, "fullrow_i0")
    full = {}
    for i, nm in enumerate(names):
        tag, node, th = nm.split("|")
        th = float(th)
        p = node_params(node)
        x_eff = (i0[i] / (p["mapf"] * D) - G_LO) / DG
        x_eff_i = (i0[len(names) + i] / (p["mapf"] * D) - G_LO) / DG
        full[nm] = dict(case=tag, node=node, theta=th, bound_V=fres[nm]["v"], bound_interp_V=fres[nm]["v_interp"],
                        bracket_width_V=fres[nm]["width"], vdl_nominal_V=vdl0[(node, th)], I0_A=float(i0[i]),
                        x_eff=float(x_eff), error_lsb=float(15.0 * (x_eff - th)),
                        error_interp_lsb=float(15.0 * (x_eff_i - th)))
    for nm, f in full.items():
        if f["case"] == "iso":
            continue
        iso = full[f"iso|{f['node']}|{f['theta']}"]
        f["isolated_error_lsb"] = iso["error_lsb"]
        f["sim_shift_lsb"] = f["error_lsb"] - iso["error_lsb"]
        f["sim_shift_interp_lsb"] = f["error_interp_lsb"] - iso["error_interp_lsb"]
        L = leak[(f["case"], f["node"])]
        f["pred_shift_lsb"] = L["pred_shift_lsb"][THETAS.index(f["theta"])]
        log(f"  {nm}: bound {f['bound_V']:.9f} V, error {f['error_lsb']:.5f}, iso {iso['error_lsb']:.5f}, "
            f"shift {f['sim_shift_lsb']:.5f} LSB (pred {f['pred_shift_lsb']:.5f})")
    R["leakers"] = {f"{k[0]}|{k[1]}": v for k, v in leak.items()}
    R["fullrow"] = full
    return R


# ----------------------------------------------------------------------------------------------
# comparison with the main run
# ----------------------------------------------------------------------------------------------
def fnum(x):
    return "" if x is None else repr(float(x))


def write_csv(path, rows):
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def compare(R, outdir):
    pred = json.loads(MAIN_PRED.read_text())
    t3rows = list(csv.DictReader(open(MAIN_SIM / "t3_crosstalk.csv")))
    t2rows = list(csv.DictReader(open(MAIN_SIM / "t2_isolated.csv")))
    items = {}

    # informational: V_sw, V_sw,tr, lambda
    info = [dict(quantity="vsw_nom_V", main=pred["vsw_nom_V"], independent=R["vsw_V"],
                 diff=R["vsw_V"] - pred["vsw_nom_V"])]
    for n in ("1", "16"):
        m = pred["t1"][n]["vsw_tr_nominal_V"]
        info.append(dict(quantity=f"vsw_tr_nominal_V[N={n}]", main=m, independent=R["vsw_tr_nominal_V"][n],
                         diff=R["vsw_tr_nominal_V"][n] - m))
    info.append(dict(quantity="a_1y", main=pred["a"]["1y"], independent=A, diff=A - pred["a"]["1y"]))
    for kind in ("I", "aI"):
        for j, th in enumerate(THETAS):
            m = pred["divider_lambda_per_V"]["P1"][kind][j]
            me = R["lambda_per_V"][kind][j]
            info.append(dict(quantity=f"lambda_{kind}[theta={th:.4f}]", main=m, independent=me, diff=me - m))
    write_csv(outdir / "compare_info.csv", info)

    # t_s
    rows = []
    for n in ("1", "16"):
        m = pred["t_s_s"][n]
        me = R["t_s"][n]["t_s_s"]
        rel = me / m - 1.0
        rows.append(dict(n_cells=n, main_s=m, independent_s=me, rel_diff=rel, threshold_rel=TOL_TS_REL,
                         gated=(n == "16"), pass_=abs(rel) <= TOL_TS_REL))
    write_csv(outdir / "compare_ts.csv", rows)
    r16 = [r for r in rows if r["n_cells"] == "16"][0]
    items["t_s16"] = dict(pass_=r16["pass_"], max_abs_rel_diff=abs(r16["rel_diff"]), threshold=TOL_TS_REL)

    # leakers: V_X, delta_l, false mismatch, predicted shift
    lrows, prow = [], []
    for key, L in R["leakers"].items():
        cid, node = key.split("|")
        m = pred["t3"][f"P1|{cid}|{node}"]
        dvx = L["vx"] - m["leaker_vx_V"]
        row = dict(config=cid, node=node, k=L["k"], x_j=L["x_j"], vsl_V=L["vsl"], g_pcm_S=L["g"], vdl_V=L["vdl"],
                   vx_main_V=m["leaker_vx_V"], vx_indep_V=L["vx"], vx_indep_op_V=L["vx_op"], vx_diff_mV=1e3 * dvx,
                   vx_pass=abs(dvx) <= TOL_VX_V,
                   fm_main=m["false_mismatch"], fm_indep=L["false_mismatch"],
                   fm_indep_fullbranch=L["false_mismatch_fullbranch"], fm_pass=(m["false_mismatch"] == L["false_mismatch"]),
                   t_cut_off_main_s=m.get("t_cross_cut_off_s"), t_cut_off_indep_s=L["t_cross_cut_off_s"],
                   t_cut_off_fullbranch_indep_s=L["t_cross_cut_off_fullbranch_s"],
                   vtr_main_V=m.get("vsw_tr_with_leakers_V"), vtr_indep_V=L["vsw_tr_with_leakers_V"],
                   delta_l_main_mV=m.get("delta_l_mV"),
                   delta_l_indep_mV=None if L["delta_l_V"] is None else 1e3 * L["delta_l_V"],
                   kappa_main=m.get("kappa"), kappa_indep=L["kappa"])
        if L["delta_l_V"] is not None and m.get("delta_l_mV") is not None:
            row["delta_l_diff_mV"] = row["delta_l_indep_mV"] - m["delta_l_mV"]
            row["delta_l_pass"] = abs(row["delta_l_diff_mV"]) <= 1e3 * TOL_DL_V
        else:
            row["delta_l_diff_mV"] = None
            row["delta_l_pass"] = (L["delta_l_V"] is None) == (m.get("delta_l_mV") is None)
        lrows.append(row)
        if L["pred_shift_lsb"] is not None:
            for j, th in enumerate(THETAS):
                mp = m["pred_shift_lsb"][j] if m.get("pred_shift_lsb") else None
                ip = L["pred_shift_lsb"][j]
                d = None if mp is None else ip - mp
                gated = th in FULL_THETAS
                prow.append(dict(config=cid, node=node, theta=th, kappa=L["kappa"], lambda_b=L["lambda_b_per_V"][j],
                                 delta_l_mV=1e3 * L["delta_l_V"], pred_main_lsb=mp, pred_indep_lsb=ip, diff_lsb=d,
                                 gated=gated, pass_=(d is not None and abs(d) <= TOL_SHIFT_LSB)))
    write_csv(outdir / "compare_leakers.csv", lrows)
    write_csv(outdir / "compare_pred_shift.csv", prow)
    items["vx"] = dict(pass_=all(r["vx_pass"] for r in lrows),
                       max_abs_diff_mV=max(abs(r["vx_diff_mV"]) for r in lrows), threshold_mV=1e3 * TOL_VX_V)
    dl = [r for r in lrows if r["delta_l_diff_mV"] is not None]
    items["delta_l"] = dict(pass_=all(r["delta_l_pass"] for r in lrows),
                            max_abs_diff_mV=max((abs(r["delta_l_diff_mV"]) for r in dl), default=None),
                            threshold_mV=1e3 * TOL_DL_V, n=len(dl))
    items["false_mismatch"] = dict(pass_=all(r["fm_pass"] for r in lrows),
                                   flags={f"{r['config']}|{r['node']}": [r["fm_main"], r["fm_indep"]] for r in lrows})
    g = [r for r in prow if r["gated"]]
    items["pred_shift"] = dict(pass_=all(r["pass_"] for r in g) and len(g) > 0,
                               max_abs_diff_lsb_theta01=max((abs(r["diff_lsb"]) for r in g if r["diff_lsb"] is not None), default=None),
                               max_abs_diff_lsb_all4=max((abs(r["diff_lsb"]) for r in prow if r["diff_lsb"] is not None), default=None),
                               threshold_lsb=TOL_SHIFT_LSB, n_gated=len(g))

    # full-row shifts
    def t3_lookup(cid, node, th):
        for r in t3rows:
            if r["design"] == "P1" and r["config"] == cid and r["node"] == node and abs(float(r["theta"]) - th) < 1e-9:
                return r
        return None

    def t2_lookup(node, th):
        for r in t2rows:
            if (r["design"] == "P1" and r["n_cells"] == "16" and r["node"] == node
                    and abs(float(r["theta"]) - th) < 1e-9):
                return r
        return None

    frows = []
    for nm, f in R["fullrow"].items():
        if f["case"] == "iso":
            continue
        m = t3_lookup(f["case"], f["node"], f["theta"])
        row = dict(config=f["case"], node=f["node"], theta=f["theta"], bound_indep_V=f["bound_V"],
                   bound_main_V=None if m is None else float(m["bound_tr_V"]),
                   error_indep_lsb=f["error_lsb"], error_main_lsb=None if m is None else float(m["error_lsb"]),
                   iso_error_indep_lsb=f["isolated_error_lsb"],
                   iso_error_main_lsb=None if m is None else float(m["isolated_error_lsb"]),
                   shift_indep_lsb=f["sim_shift_lsb"], shift_indep_interp_lsb=f["sim_shift_interp_lsb"],
                   shift_main_lsb=None if m is None else float(m["sim_shift_lsb"]),
                   pred_indep_lsb=f["pred_shift_lsb"], pred_main_lsb=None if m is None else float(m["pred_shift_lsb"]))
        if m is not None:
            row["bound_diff_uV"] = 1e6 * (row["bound_indep_V"] - row["bound_main_V"])
            row["shift_diff_lsb"] = row["shift_indep_lsb"] - row["shift_main_lsb"]
            row["iso_error_diff_lsb"] = row["iso_error_indep_lsb"] - row["iso_error_main_lsb"]
            row["error_diff_lsb"] = row["error_indep_lsb"] - row["error_main_lsb"]
            row["shift_pass"] = abs(row["shift_diff_lsb"]) <= TOL_SHIFT_LSB
            row["iso_pass"] = abs(row["iso_error_diff_lsb"]) <= TOL_SHIFT_LSB
            # protocol T3 rule applied to the independent numbers (informational)
            row["t3_rule_indep_pass"] = abs(f["pred_shift_lsb"] - f["sim_shift_lsb"]) <= max(
                0.1 * abs(f["sim_shift_lsb"]), 0.02)
        else:
            row.update(bound_diff_uV=None, shift_diff_lsb=None, iso_error_diff_lsb=None, error_diff_lsb=None,
                       shift_pass=False, iso_pass=False, t3_rule_indep_pass=None)
        frows.append(row)
    write_csv(outdir / "compare_fullrow_shift.csv", frows)
    irows = []
    for nm, f in R["fullrow"].items():
        if f["case"] != "iso":
            continue
        m = t2_lookup(f["node"], f["theta"])
        row = dict(node=f["node"], theta=f["theta"], vdl_nominal_indep_V=f["vdl_nominal_V"],
                   bound_qs_main_V=None if m is None else float(m["bound_qs_V"]),
                   bound_indep_V=f["bound_V"], bound_main_V=None if m is None else float(m["bound_tr_V"]),
                   error_indep_lsb=f["error_lsb"], error_main_lsb=None if m is None else float(m["error_lsb"]))
        if m is not None:
            row["bound_qs_diff_uV"] = 1e6 * (row["vdl_nominal_indep_V"] - row["bound_qs_main_V"])
            row["bound_diff_uV"] = 1e6 * (row["bound_indep_V"] - row["bound_main_V"])
            row["error_diff_lsb"] = row["error_indep_lsb"] - row["error_main_lsb"]
            row["pass_"] = abs(row["error_diff_lsb"]) <= TOL_SHIFT_LSB
        else:
            row.update(bound_qs_diff_uV=None, bound_diff_uV=None, error_diff_lsb=None, pass_=False)
        irows.append(row)
    write_csv(outdir / "compare_isolated_t2.csv", irows)
    items["fullrow_shift"] = dict(pass_=all(r["shift_pass"] for r in frows) and len(frows) > 0,
                                  max_abs_diff_lsb=max((abs(r["shift_diff_lsb"]) for r in frows
                                                        if r["shift_diff_lsb"] is not None), default=None),
                                  threshold_lsb=TOL_SHIFT_LSB, n=len(frows))
    items["fullrow_isolated_error"] = dict(
        pass_=all(r["iso_pass"] for r in frows) and all(r["pass_"] for r in irows),
        max_abs_diff_lsb_t3=max((abs(r["iso_error_diff_lsb"]) for r in frows if r["iso_error_diff_lsb"] is not None), default=None),
        max_abs_diff_lsb_t2=max((abs(r["error_diff_lsb"]) for r in irows if r["error_diff_lsb"] is not None), default=None),
        threshold_lsb=TOL_SHIFT_LSB, n_t2=len(irows))
    return items, dict(info=info, ts=rows, leakers=lrows, pred_shift=prow, fullrow=frows, isolated=irows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stamp", default=None, help="JST stamp YYYYMMDD_HHMM (default: now)")
    args = ap.parse_args()
    stamp = args.stamp or datetime.now(JST).strftime("%Y%m%d_%H%M")
    outdir = RESULTS_BASE / f"E3c_independent_check_{stamp}"
    if outdir.exists():
        sys.exit(f"output folder exists, refusing to overwrite: {outdir}")
    work = WORK_BASE / f"run_{stamp}"
    sp = Spice(work)
    outdir.mkdir(parents=True)
    logf = open(outdir / "run.log", "w")

    def log(msg):
        line = f"[{datetime.now(JST).strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    t_start = time.time()
    log(f"E3c independent check, root {ROOT}, work {work}, out {outdir}")
    ver = subprocess.run(["ngspice", "--version"], capture_output=True, text=True).stdout.strip().splitlines()
    R = compute(sp, log)
    (work / "state.json").write_text(json.dumps(R, indent=1, default=str))
    items, tables = compare(R, outdir)
    overall = all(v["pass_"] for v in items.values())
    summary = dict(
        experiment="E3c independent check (P1)",
        created_jst=datetime.now(JST).strftime("%Y-%m-%d %H:%M:%S"),
        overall_pass=overall,
        items=items,
        computed=R,
        main_run=dict(predictions=str(MAIN_PRED.relative_to(ROOT)), predictions_sha256=sha256(MAIN_PRED),
                      t3_csv=str((MAIN_SIM / "t3_crosstalk.csv").relative_to(ROOT)),
                      t3_csv_sha256=sha256(MAIN_SIM / "t3_crosstalk.csv"),
                      t2_csv=str((MAIN_SIM / "t2_isolated.csv").relative_to(ROOT)),
                      t2_csv_sha256=sha256(MAIN_SIM / "t2_isolated.csv")),
        inputs=dict(config_sha256=sha256(CFG_PATH), lib_sha256=sha256(LIB),
                    script_sha256=sha256(Path(__file__).resolve())),
        environment=dict(ngspice=" ".join(l.strip("* ") for l in ver if "ngspice" in l.lower())[:200],
                         python=platform.python_version(), numpy=np.__version__),
        runtime=dict(seconds=round(time.time() - t_start, 1), ngspice_runs=sp.n,
                     ngspice_seconds=round(sum(r["seconds"] for r in sp.runs), 1), work_dir=str(work.relative_to(ROOT))),
    )
    (outdir / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    (outdir / "ngspice_runs.json").write_text(json.dumps(sp.runs, indent=1))
    log("ITEMS: " + json.dumps(items, default=str))
    log(f"overall pass = {overall}; {sp.n} ngspice runs; {time.time() - t_start:.0f} s")
    logf.close()


if __name__ == "__main__":
    main()
