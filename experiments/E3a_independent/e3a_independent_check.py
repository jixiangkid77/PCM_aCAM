#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
E3a independent check (docs/E3a_protocol.md, section 9).

Written only from docs/E3a_protocol.md (sections 2, 3, 4, 9), docs/E2_protocol.md (section 2),
experiments/E3a_config.json, experiments/E2_config.json and the SPICE model files.  It does not
import or read anything under src/ or any other experiment script.

Task 1  design-time predictor (protocol section 3) for P1 and S1 from own single-device ngspice
        simulations (nominal and +-1 sigma per mismatch parameter, V_DS +- 5 mV), 1 year,
        grid points linear, (0.3, 0), (0.3, +0.5), (0.3, -0.5); nodes dac_remap, supply_lin,
        supply_rep.  Compared with results/E3a_predict_20261014_0442/predictions.json.
Task 2  full-branch DC spot check: PCM = ngspice B-source between SL and X
        (I = a*A*V0*sinh(V(sl,x)/V0), or a*A*V(sl,x) for the linear PCM), divider nMOS X -> 0 with
        its gate on DL; V_DL is swept and the bound is the V_DL at which V_X = V_sw,k.
        P1 and S1, pdk setting, instances 0..19, theta in {1, 5, 10, 14}/15, 1 year, the same four
        grid points, nodes dac_remap, supply_lin, supply_rep.  Compared with
        results/E3a_20261014_0442/arrays_<did>.npz.

Usage:  python experiments/E3a_independent/e3a_independent_check.py
        (options: --n-inst N for a reduced development run; --tag TEXT appended to the folder name)
"""
from __future__ import annotations

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

# --------------------------------------------------------------------------------------------
# locations (repository root derived from this file: experiments/E3a_independent/<this file>)
# --------------------------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
CFG_PATH = ROOT / "experiments" / "E3a_config.json"
LIB_DIR = ROOT / "third_party" / "sky130_fd_pr_subset_corners"
RESULTS = ROOT / "results"
WORK_BASE = ROOT / "logs" / "E3a_independent_work"
MAIN_PRED = RESULTS / "E3a_predict_20261014_0442" / "predictions.json"
MAIN_MC_DIR = RESULTS / "E3a_20261014_0442"
JST = timezone(timedelta(hours=9))

# numerical settings of this checker (own choices, see README.md)
SPICE_OPTIONS = ".options reltol=1e-7 vntol=1e-10 abstol=1e-16"
COARSE_STEP = 0.0005      # V, coarse gate sweep 0 .. 1.8 V (config sweep step)
FINE_STEP = 1e-6          # V, fine sweep step around each crossing
FINE_SPAN = 0.0015        # V, fine window = [lo - 0.5 mV, lo + 1.0 mV] around the coarse bracket [lo, lo + 0.5 mV]
FINE_PRE = 0.0005         # V, start of the fine window below the coarse lower bracket point
VDS_STEP = 0.005          # V, lambda from V_DS +- 5 mV (E2 config predictor.vds_step_v)
I_SW_PER_SQ = 100e-9      # A, switching criterion 100 nA * W/L (config circuit.criterion)

# thresholds of the check (task statement / protocol section 9)
THR_PRED_REL = 0.01
THR_PRED_ABS = 0.002      # LSB
THR_VSW_NOM = 0.05e-3     # V
THR_SIGMA_REL = 0.01
THR_BOUND = 0.05e-3       # V (V_sw,k and M0(theta))
THR_ERR = 0.01            # LSB

CHECK_GRID = [  # (predictions.json label, npz label, V0_ref, beta)
    ("linear", "linear", math.inf, 0.0),
    ("V0=0.3|beta=+0", "V00.3_beta+0", 0.3, 0.0),
    ("V0=0.3|beta=+0.5", "V00.3_beta+0.5", 0.3, 0.5),
    ("V0=0.3|beta=-0.5", "V00.3_beta-0.5", 0.3, -0.5),
]
CHECK_NODES = ["dac_remap", "supply_lin", "supply_rep"]
CHECK_DESIGNS = ["P1", "S1"]
CHECK_THETA_IDX = [1, 5, 10, 14]
N_SPOT = 20
PARAMS = ["vth0", "toxe", "voff", "nfactor"]  # column order of the draws (config mismatch.rng)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------------------------
# ngspice helpers
# --------------------------------------------------------------------------------------------
def read_raw(path: Path):
    """Parse an ngspice binary rawfile (real data). Returns a list of (plotname, {vector: array})."""
    data = Path(path).read_bytes()
    pos, plots = 0, []
    while pos < len(data):
        hdr, names = {}, []
        while True:
            end = data.index(b"\n", pos)
            line = data[pos:end].decode("ascii", "replace")
            pos = end + 1
            if line.startswith("Binary:"):
                break
            if line.startswith("Variables:"):
                for _ in range(hdr["No. Variables"]):
                    end = data.index(b"\n", pos)
                    names.append(data[pos:end].decode().split()[1])
                    pos = end + 1
                continue
            if ":" in line:
                k, v = line.split(":", 1)
                k, v = k.strip(), v.strip()
                hdr[k] = int(v) if k in ("No. Variables", "No. Points") else v
        if "complex" in hdr.get("Flags", "real"):
            raise RuntimeError("complex rawfile not expected")
        nv, npt = hdr["No. Variables"], hdr["No. Points"]
        arr = np.frombuffer(data, dtype="<f8", count=nv * npt, offset=pos).reshape(npt, nv)
        pos += 8 * nv * npt
        plots.append((hdr.get("Plotname", ""), {n: arr[:, i].copy() for i, n in enumerate(names)}))
        while pos < len(data) and data[pos:pos + 1] in (b"\n", b"\r", b" "):
            pos += 1
    return plots


class Spice:
    """Runs ngspice jobs strictly one at a time in their own working directories."""

    def __init__(self, workdir: Path):
        self.workdir = workdir
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.n_runs = 0
        self.t_total = 0.0
        self.log = []

    def header(self, title: str) -> list[str]:
        lib = (LIB_DIR / "sky130_subset_tt.lib.spice").as_posix()
        return [f"* {title}", f'.include "{lib}"', ".temp 27", SPICE_OPTIONS]

    def run(self, tag: str, lines: list[str]):
        d = self.workdir / f"{self.n_runs:03d}_{tag}"
        d.mkdir(parents=True, exist_ok=False)
        (d / ".spiceinit").write_text("set ngbehavior=hsa\nset ng_nomodcheck\n")
        (d / "net.sp").write_text("\n".join(lines) + "\n.end\n")
        t0 = time.time()
        r = subprocess.run(["ngspice", "-b", "-r", "out.raw", "net.sp"], cwd=d,
                           capture_output=True, text=True)
        dt = time.time() - t0
        self.n_runs += 1
        self.t_total += dt
        (d / "ngspice.log").write_text(r.stdout + "\n----- stderr -----\n" + r.stderr)
        text = (r.stdout + r.stderr).lower()
        bad = [w for w in ("error", "singular", "iteration limit", "failed", "no convergence",
                           "gmin stepping", "source stepping") if w in text]
        if r.returncode != 0 or not (d / "out.raw").exists() or bad:
            raise RuntimeError(f"ngspice problem in {d}: rc={r.returncode} flags={bad}")
        self.log.append({"tag": tag, "dir": d.name, "seconds": round(dt, 2)})
        print(f"    ngspice {d.name}: {dt:.1f} s", flush=True)
        return read_raw(d / "out.raw")


def fmt(x: float) -> str:
    return f"{float(x):.17g}"


def device_line(inst: str, d: str, g: str, s: str, model: str, w: float, l: float, z) -> str:
    z = [float(v) for v in z]
    return (f"{inst} {d} {g} {s} 0 {model} w={fmt(w)} l={fmt(l)} mm_vth0={fmt(z[0])} "
            f"mm_toxe={fmt(z[1])} mm_voff={fmt(z[2])} mm_nfactor={fmt(z[3])}")


def save_lines(vectors: list[str], per_line: int = 25) -> list[str]:
    return [".save " + " ".join(vectors[i:i + per_line]) for i in range(0, len(vectors), per_line)]


def interp_cross(x0, x1, y0, y1, target):
    return x0 + (target - y0) * (x1 - x0) / (y1 - y0)


def invert_iv(sp: Spice, tag: str, entries: list[dict]) -> np.ndarray:
    """Gate voltage at which each device conducts its target drain current at its V_DS.

    entries: dict(model, w, l, z, vds, target).  Coarse shared-gate sweep 0..1.8 V (0.5 mV), then a
    fine sweep (1 uV) of each device in its own window; ln I is interpolated linearly between the
    two fine points that bracket the target (single-device simulations only)."""
    n = len(entries)
    lines = sp.header(f"{tag} coarse")
    lines.append("vg g 0 0")
    for i, e in enumerate(entries):
        lines.append(f"vd{i} d{i} 0 {fmt(e['vds'])}")
        lines.append(device_line(f"xm{i}", f"d{i}", "g", "0", e["model"], e["w"], e["l"], e["z"]))
    lines += save_lines([f"i(vd{i})" for i in range(n)])
    lines.append(f".dc vg 0 1.8 {fmt(COARSE_STEP)}")
    vec = sp.run(tag + "_coarse", lines)[0][1]
    vg = vec["v(v-sweep)"]
    lo = np.empty(n)
    for i, e in enumerate(entries):
        cur = -vec[f"i(vd{i})"]
        j = int(np.argmax(cur >= e["target"]))
        if cur[j] < e["target"] or j == 0:
            raise RuntimeError(f"{tag}: target {e['target']} not bracketed for entry {i}")
        if not np.all(np.diff(cur[max(j - 3, 0):j + 3]) > 0):
            raise RuntimeError(f"{tag}: non-monotone current near crossing, entry {i}")
        lo[i] = vg[j - 1]
    # fine sweep: gate_i = V(dl) + off_i
    off = lo - FINE_PRE
    lines = sp.header(f"{tag} fine")
    lines.append("vdl dl 0 0")
    for i, e in enumerate(entries):
        lines.append(f"voff{i} g{i} dl {fmt(off[i])}")
        lines.append(f"vd{i} d{i} 0 {fmt(e['vds'])}")
        lines.append(device_line(f"xm{i}", f"d{i}", f"g{i}", "0", e["model"], e["w"], e["l"], e["z"]))
    lines += save_lines([f"i(vd{i})" for i in range(n)])
    lines.append(f".dc vdl 0 {fmt(FINE_SPAN)} {fmt(FINE_STEP)}")
    vec = sp.run(tag + "_fine", lines)[0][1]
    sw = vec["v(v-sweep)"]
    out = np.empty(n)
    for i, e in enumerate(entries):
        cur = -vec[f"i(vd{i})"]
        if not np.all(np.diff(cur) > 0):
            raise RuntimeError(f"{tag}: fine curve not monotone, entry {i}")
        j = int(np.argmax(cur >= e["target"]))
        if cur[j] < e["target"] or j == 0:
            raise RuntimeError(f"{tag}: fine window misses the target, entry {i}")
        out[i] = off[i] + interp_cross(sw[j - 1], sw[j], math.log(cur[j - 1]), math.log(cur[j]),
                                       math.log(e["target"]))
    return out


def op_currents(sp: Spice, tag: str, entries: list[dict]) -> np.ndarray:
    """Drain currents at exact bias points (one .op).  entries: dict(model, w, l, z, vgs, vds)."""
    lines = sp.header(tag)
    for i, e in enumerate(entries):
        lines.append(f"vg{i} g{i} 0 {fmt(e['vgs'])}")
        lines.append(f"vd{i} d{i} 0 {fmt(e['vds'])}")
        lines.append(device_line(f"xm{i}", f"d{i}", f"g{i}", "0", e["model"], e["w"], e["l"], e["z"]))
    lines += save_lines([f"i(vd{i})" for i in range(len(entries))])
    lines.append(".op")
    vec = sp.run(tag, lines)[0][1]
    return np.array([-vec[f"i(vd{i})"][0] for i in range(len(entries))])


def fullbranch_bounds(sp: Spice, tag: str, branches: list[dict]) -> np.ndarray:
    """Bound of each lower-bound branch from full-branch DC sweeps.

    branch: dict(model, w, l, z (divider), vsl, k_amp (= a*A), v0 (inf = linear), vsw).
    Circuit per branch: V_SL source -> B-source PCM (SL -> X) -> divider nMOS (drain X, gate DL,
    source/bulk 0).  V_DL is swept; the bound is the V_DL at which V_X = V_sw,k (linear
    interpolation of V_X between adjacent fine-sweep points, 1 uV apart)."""
    n = len(branches)
    vsl_vals = sorted({float(b["vsl"]) for b in branches})
    sl_name = {v: f"sl{i}" for i, v in enumerate(vsl_vals)}

    def branch_lines(gate_of):
        ls = [f"v{sl_name[v]} {sl_name[v]} 0 {fmt(v)}" for v in vsl_vals]
        for i, b in enumerate(branches):
            sl = sl_name[float(b["vsl"])]
            if math.isinf(b["v0"]):
                expr = f"{fmt(b['k_amp'])}*v({sl},nx{i})"
            else:
                expr = f"{fmt(b['k_amp'])}*{fmt(b['v0'])}*sinh(v({sl},nx{i})/{fmt(b['v0'])})"
            ls.append(f"bpcm{i} {sl} nx{i} i={expr}")
            ls.append(device_line(f"xdiv{i}", f"nx{i}", gate_of(i), "0", b["model"], b["w"], b["l"], b["z"]))
        return ls

    # coarse sweep, shared gate
    lines = sp.header(f"{tag} coarse") + ["vdl dl 0 0"] + branch_lines(lambda i: "dl")
    lines += save_lines([f"v(nx{i})" for i in range(n)])
    lines.append(f".dc vdl 0 1.8 {fmt(COARSE_STEP)}")
    vec = sp.run(tag + "_coarse", lines)[0][1]
    vdl = vec["v(v-sweep)"]
    lo = np.full(n, np.nan)
    valid = np.ones(n, dtype=bool)
    for i, b in enumerate(branches):
        vx = vec[f"v(nx{i})"]
        idx = np.nonzero(vx <= b["vsw"])[0]
        if idx.size == 0 or idx[0] == 0:
            valid[i] = False
            continue
        j = int(idx[0])
        seg = vx[max(j - 4, 0):j + 4]
        if not np.all(np.diff(seg) < 0):
            raise RuntimeError(f"{tag}: V_X not monotone near the crossing, branch {i}")
        lo[i] = vdl[j - 1]
    if not valid.any():
        return np.full(n, np.nan)
    off = np.where(valid, lo - FINE_PRE, 0.0)
    # fine sweep, per-branch gate offsets
    lines = sp.header(f"{tag} fine") + ["vdl dl 0 0"]
    lines += [f"voff{i} g{i} dl {fmt(off[i])}" for i in range(n)]
    lines += branch_lines(lambda i: f"g{i}")
    lines += save_lines([f"v(nx{i})" for i in range(n)])
    lines.append(f".dc vdl 0 {fmt(FINE_SPAN)} {fmt(FINE_STEP)}")
    vec = sp.run(tag + "_fine", lines)[0][1]
    sw = vec["v(v-sweep)"]
    out = np.full(n, np.nan)
    for i, b in enumerate(branches):
        if not valid[i]:
            continue
        vx = vec[f"v(nx{i})"]
        if not np.all(np.diff(vx) < 0):
            raise RuntimeError(f"{tag}: fine V_X not monotone, branch {i}")
        idx = np.nonzero(vx <= b["vsw"])[0]
        if idx.size == 0 or idx[0] == 0:
            raise RuntimeError(f"{tag}: fine window misses the crossing, branch {i}")
        j = int(idx[0])
        out[i] = off[i] + interp_cross(sw[j - 1], sw[j], vx[j - 1], vx[j], b["vsw"])
    return out


# --------------------------------------------------------------------------------------------
# PCM model (protocol section 2)
# --------------------------------------------------------------------------------------------
def s_fun(v, v0):
    """s(V; V0) = V0 sinh(V / V0); V0 = inf is the linear PCM (s = V)."""
    if math.isinf(v0):
        return np.asarray(v, dtype=float)
    return v0 * np.sinh(np.asarray(v, dtype=float) / v0)


def v0_of_g(v0_ref: float, beta: float, g: float, g_ref: float) -> float:
    if math.isinf(v0_ref):
        return math.inf
    return v0_ref * (g / g_ref) ** beta


def v_star(d: float, a: float, v0_rep: float) -> float:
    """a s(V*; V0_rep) = s(D; V0_rep)."""
    if math.isinf(v0_rep):
        return d / a
    return v0_rep * math.asinh(math.sinh(d / v0_rep) / a)


# --------------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-inst", type=int, default=N_SPOT, help="instances in the spot check (20)")
    ap.add_argument("--tag", default="", help="suffix of the output folder name")
    args = ap.parse_args()

    t_start = time.time()
    stamp = datetime.now(JST).strftime("%Y%m%d_%H%M")
    outdir = RESULTS / (f"E3a_independent_check_{stamp}" + (f"_{args.tag}" if args.tag else ""))
    outdir.mkdir(parents=False, exist_ok=False)          # never overwrite
    workdir = WORK_BASE / outdir.name
    sp = Spice(workdir)
    print(f"output folder {outdir}\nwork folder   {workdir}", flush=True)

    cfg = json.loads(CFG_PATH.read_text())
    circ = cfg["circuit"]
    pd_spec = circ["pulldowns"]["pd_min"]
    model_of = {"nfet_01v8": "sky130_fd_pr__nfet_01v8", "nfet_01v8_lvt": "sky130_fd_pr__nfet_01v8_lvt"}
    div_model = model_of[circ["divider_device"]]
    pd_model = model_of[pd_spec["device"]]
    pd_w, pd_l = float(pd_spec["w_um"]), float(pd_spec["l_um"])
    vdd = float(circ["vdd"])
    g_lo, g_hi = circ["g_lo_us"] * 1e-6, circ["g_hi_us"] * 1e-6
    dG = g_hi - g_lo
    g_ref = cfg["pcm_model"]["g_ref_us"] * 1e-6
    nu, t0 = cfg["drift"]["nu"], cfg["drift"]["t0_s"]
    t_1y = cfg["drift"]["times_s"]["1y"]
    a = (t_1y / t0) ** (-nu)
    theta = np.arange(16) / 15.0
    G = g_lo + dG * theta
    designs = {d["id"]: d for d in cfg["designs"]}
    mm = cfg["mismatch"]
    rng = np.random.default_rng([mm["base_seed"], mm["rng_stream"]])
    z_div = rng.standard_normal((mm["n_instances"], 4))
    z_pd = rng.standard_normal((mm["n_instances"], 4))
    i_sw = I_SW_PER_SQ * pd_w / pd_l
    print(f"a(1y) = {a:.10f}; I_sw = {i_sw:.6e} A", flush=True)

    # ---------------- pull-down: V_sw nominal, +-1 sigma, 300 instances ----------------
    print("[1] pull-down switching voltages", flush=True)
    zero = np.zeros(4)
    pd_entries = [dict(model=pd_model, w=pd_w, l=pd_l, z=zero, vds=vdd, target=i_sw)]
    for p in range(4):
        for sgn in (+1.0, -1.0):
            z = zero.copy()
            z[p] = sgn
            pd_entries.append(dict(model=pd_model, w=pd_w, l=pd_l, z=z, vds=vdd, target=i_sw))
    for k in range(mm["n_instances"]):
        pd_entries.append(dict(model=pd_model, w=pd_w, l=pd_l, z=z_pd[k], vds=vdd, target=i_sw))
    vsw_all = invert_iv(sp, "pulldown_vsw", pd_entries)
    vsw_nom = float(vsw_all[0])
    dvsw = np.array([(vsw_all[1 + 2 * p] - vsw_all[2 + 2 * p]) / 2.0 for p in range(4)])
    sigma_delta = float(np.sqrt(np.sum(dvsw ** 2)))
    vsw_k = vsw_all[9:]
    delta_k = vsw_k - vsw_nom
    print(f"    V_sw,nom = {vsw_nom:.9f} V; dVsw/dz = {dvsw}; sigma_delta = {sigma_delta:.9f} V", flush=True)

    main_pred = json.loads(MAIN_PRED.read_text())
    summary = {
        "experiment": "E3a independent check (protocol section 9)",
        "created_jst": datetime.now(JST).strftime("%Y-%m-%d %H:%M:%S"),
        "script": str(Path(__file__).resolve().relative_to(ROOT)),
        "script_sha256": sha256(Path(__file__).resolve()),
        "config_sha256": sha256(CFG_PATH),
        "main_predictions_sha256": sha256(MAIN_PRED),
        "main_arrays_sha256": {did: sha256(MAIN_MC_DIR / f"arrays_{did}.npz") for did in CHECK_DESIGNS},
        "ngspice": subprocess.run(["ngspice", "--version"], capture_output=True, text=True).stdout.split("\n")[1].strip(),
        "python": platform.python_version(), "numpy": np.__version__,
        "spice_options": SPICE_OPTIONS, "coarse_step_V": COARSE_STEP, "fine_step_V": FINE_STEP,
        "a_1y": a, "n_spot_instances": args.n_inst,
        "thresholds": {"pred_rel": THR_PRED_REL, "pred_abs_lsb": THR_PRED_ABS, "vsw_nom_V": THR_VSW_NOM,
                       "sigma_delta_rel": THR_SIGMA_REL, "bound_V": THR_BOUND, "err_lsb": THR_ERR},
        "pulldown": {"vsw_nom_V": vsw_nom, "dvsw_dz_V": dict(zip(PARAMS, dvsw.tolist())),
                     "sigma_delta_V": sigma_delta},
        "task1": {}, "task2": {},
    }
    t1_rows, t2_rows, b_rows, vsw_rows = [], [], [], []

    for did in CHECK_DESIGNS:
        des = designs[did]
        D = float(des["d_v"])
        w_div, l_div = float(des["w_div_um"]), float(des["l_div_um"])
        vsl_nom = vsw_nom + D
        print(f"[2] design {did}: D = {D} V, divider {w_div}/{l_div} um", flush=True)
        div_nom = dict(model=div_model, w=w_div, l=l_div, z=zero)

        # ---------- nominal map M0(theta_j) = B_nom(G_j): nominal divider at V_DS = V_sw,nom ----------
        def vg_of_current(tag, currents):
            ents = [dict(div_nom, vds=vsw_nom, target=float(c)) for c in currents]
            return invert_iv(sp, tag, ents)

        I = G * D
        m0 = vg_of_current(f"{did}_M0", I)

        # ---------- Task 1: predictor (protocol section 3; E2 section 2.3 sensitivities) ----------
        # currents at which s_p and lambda are needed: I, a I, and rho0 I per (grid point, supply node)
        rho0 = {}
        vt = {}
        for lab, _, v0r, beta in CHECK_GRID:
            v0_rep = v0_of_g(v0r, beta, g_ref, g_ref)
            vt[(lab, "supply_lin")] = D / a
            vt[(lab, "supply_rep")] = v_star(D, a, v0_rep)
            for node in ("supply_lin", "supply_rep"):
                r = []
                for j in range(16):
                    v0 = v0_of_g(v0r, beta, G[j], g_ref)
                    r.append(float(a * s_fun(vt[(lab, node)], v0) / s_fun(D, v0)))
                rho0[(lab, node)] = np.array(r)
        cur_list = [I, a * I] + [rho0[key] * I for key in sorted(rho0)]
        cur_all = np.concatenate(cur_list)
        vg_extra = vg_of_current(f"{did}_vg_of_I", cur_all[16:])
        vg_all = np.concatenate([m0, vg_extra])
        # single-device evaluations at the exact gate voltages: nominal, +-1 sigma per parameter, V_DS +- 5 mV
        variants = [("nom", zero, vsw_nom)]
        for p in range(4):
            for sgn in (+1.0, -1.0):
                z = zero.copy()
                z[p] = sgn
                variants.append((f"{PARAMS[p]}{'+' if sgn > 0 else '-'}", z, vsw_nom))
        variants.append(("vds+", zero, vsw_nom + VDS_STEP))
        variants.append(("vds-", zero, vsw_nom - VDS_STEP))
        ents = []
        for _, z, vds in variants:
            for vg in vg_all:
                ents.append(dict(div_nom, z=z, vgs=float(vg), vds=float(vds)))
        cur = op_currents(sp, f"{did}_sens_op", ents).reshape(len(variants), vg_all.size)
        lnI = np.log(cur)
        nom_check = float(np.max(np.abs(lnI[0] - np.log(cur_all))))
        s_p = np.stack([-(lnI[1 + 2 * p] - lnI[2 + 2 * p]) / 2.0 for p in range(4)])   # (4, ncur)
        lam = -(lnI[9] - lnI[10]) / (2 * VDS_STEP)

        def sens_at(block):  # block index into cur_list
            s0 = sum(c.size for c in cur_list[:block])
            return s_p[:, s0:s0 + 16], lam[s0:s0 + 16]

        sI, lI = sens_at(0)
        saI, laI = sens_at(1)
        keys_sorted = sorted(rho0)

        # expectation over delta ~ N(0, sigma^2): trapezoid on [-6s, 6s], 4001 points, drop delta >= 0.95 D, renormalize
        dgrid = np.linspace(-6 * sigma_delta, 6 * sigma_delta, 4001)
        wts = np.full(dgrid.size, dgrid[1] - dgrid[0])
        wts[0] *= 0.5
        wts[-1] *= 0.5
        wts *= np.exp(-0.5 * (dgrid / sigma_delta) ** 2)
        keep = dgrid < 0.95 * D
        dg, wk = dgrid[keep], wts[keep] / np.sum(wts[keep])

        gfac = G / dG
        # dac_remap (E2 law; independent of the PCM nonlinearity)
        e_dac = gfac ** 2 * (np.sum((saI - sI) ** 2, axis=0) + (laI - lI) ** 2 * sigma_delta ** 2)
        rms_dac = 15 * math.sqrt(float(np.mean(e_dac)))
        my_pred = {}
        for lab, _, v0r, beta in CHECK_GRID:
            my_pred[(lab, "dac_remap")] = {"rms_lsb": rms_dac}
            for node in ("supply_lin", "supply_rep"):
                key = (lab, node)
                srho, lrho = sens_at(2 + keys_sorted.index(key))
                Vt = vt[key]
                e_sup = np.empty(16)
                for j in range(16):
                    v0 = v0_of_g(v0r, beta, G[j], g_ref)
                    r0 = rho0[key][j]
                    rho_d = a * s_fun(Vt - dg, v0) / s_fun(D - dg, v0)
                    dlam = lrho[j] - lI[j]
                    dsp = srho[:, j] - sI[:, j]
                    e_sup[j] = gfac[j] ** 2 * (np.sum(wk * (rho_d - 1 + r0 * dlam * dg) ** 2)
                                               + r0 ** 2 * np.sum(dsp ** 2))
                my_pred[key] = {
                    "rms_lsb": 15 * math.sqrt(float(np.mean(e_sup))),
                    "rms_systematic_lsb": 15 * math.sqrt(float(np.mean((gfac * (rho0[key] - 1)) ** 2))),
                    "vt_V": Vt, "v_sl_V": vsw_nom + Vt, "rho0": rho0[key].tolist(),
                }
        # comparison with predictions.json
        mp = main_pred[did]
        ps = mp["pulldown_sensitivity"]
        t1 = {"vsw_nom": {}, "sigma_delta": {}, "rms": [], "extra": {}}
        dv = abs(vsw_nom - mp["vsw_nom_V"])
        t1["vsw_nom"] = {"mine_V": vsw_nom, "main_V": mp["vsw_nom_V"], "abs_diff_V": dv, "pass": dv <= THR_VSW_NOM}
        rs = abs(sigma_delta - ps["sigma_delta"]) / ps["sigma_delta"]
        t1["sigma_delta"] = {"mine_V": sigma_delta, "main_V": ps["sigma_delta"], "rel_diff": rs,
                             "pass": rs <= THR_SIGMA_REL}
        t1_rows.append([did, "vsw_nom_V", "", "", vsw_nom, mp["vsw_nom_V"], vsw_nom - mp["vsw_nom_V"], "",
                        dv <= THR_VSW_NOM])
        t1_rows.append([did, "sigma_delta_V", "", "", sigma_delta, ps["sigma_delta"],
                        sigma_delta - ps["sigma_delta"], rs, rs <= THR_SIGMA_REL])
        for p in range(4):
            mv = ps[f"dvsw_d{PARAMS[p]}"]
            t1_rows.append([did, f"dvsw_d{PARAMS[p]}_V", "", "", dvsw[p], mv, dvsw[p] - mv,
                            "" if mv == 0 else abs(dvsw[p] - mv) / abs(mv), "info"])
        max_rho0_diff, max_vt_diff, max_sys_rel = 0.0, 0.0, 0.0
        for lab, _, _, _ in CHECK_GRID:
            for node in CHECK_NODES:
                mine = my_pred[(lab, node)]["rms_lsb"]
                main = mp["pred"]["1y"][lab][node]["rms_lsb"]
                ad = abs(mine - main)
                rd = ad / abs(main) if main != 0 else math.inf
                ok = (rd <= THR_PRED_REL) or (ad <= THR_PRED_ABS)
                t1["rms"].append({"grid": lab, "node": node, "mine_lsb": mine, "main_lsb": main,
                                  "abs_diff_lsb": ad, "rel_diff": rd, "pass": ok})
                t1_rows.append([did, "rms_1y_lsb", lab, node, mine, main, mine - main, rd, ok])
                if node != "dac_remap":
                    mm_ = mp["pred"]["1y"][lab][node]
                    max_rho0_diff = max(max_rho0_diff, float(np.max(np.abs(np.array(mm_["rho0"]) - rho0[(lab, node)]))))
                    max_vt_diff = max(max_vt_diff, abs(mm_["vt_V"] - vt[(lab, node)]))
                    sm, sx = my_pred[(lab, node)]["rms_systematic_lsb"], mm_["rms_systematic_lsb"]
                    if sx != 0:
                        max_sys_rel = max(max_sys_rel, abs(sm - sx) / abs(sx))
                    else:
                        max_sys_rel = max(max_sys_rel, abs(sm))
                    t1_rows.append([did, "rms_systematic_1y_lsb", lab, node, sm, sx, sm - sx,
                                    abs(sm - sx) / abs(sx) if sx else "", "info"])
        t1["extra"] = {"max_abs_diff_rho0": max_rho0_diff, "max_abs_diff_vt_V": max_vt_diff,
                       "max_rel_diff_rms_systematic": max_sys_rel,
                       "nominal_current_consistency_max_abs_dlnI": nom_check}
        t1["pass"] = bool(t1["vsw_nom"]["pass"] and t1["sigma_delta"]["pass"] and all(r["pass"] for r in t1["rms"]))
        summary["task1"][did] = t1
        print(f"    Task 1 {did}: pass = {t1['pass']}", flush=True)

        # ---------- Task 2: full-branch spot check ----------
        z_arr = np.load(MAIN_MC_DIR / f"arrays_{did}.npz")
        n_inst = args.n_inst
        jj = CHECK_THETA_IDX
        # instances 0..n-1 of the pdk draws, plus k = -1: the zero-mismatch cell (own self-check, like C2)
        inst = [(k, z_div[k], float(vsw_k[k])) for k in range(n_inst)] + [(-1, zero, vsw_nom)]
        # programming currents: instance divider at gate M0(theta), V_DS = V_sw,k (threshold verify at V_SL,nom)
        ents = []
        for k, zk, vswk in inst:
            for j in jj:
                ents.append(dict(model=div_model, w=w_div, l=l_div, z=zk, vgs=float(m0[j]), vds=vswk))
        i_prog = op_currents(sp, f"{did}_prog_op", ents).reshape(len(inst), len(jj))
        branches, meta = [], []
        for ki, (k, zk, vswk) in enumerate(inst):
            dk = vswk - vsw_nom
            for jx, j in enumerate(jj):
                for lab, npz_lab, v0r, beta in CHECK_GRID:
                    v0 = v0_of_g(v0r, beta, G[j], g_ref)
                    amp = i_prog[ki, jx] / float(s_fun(D - dk, v0))   # A_k: A s(D - delta_k) = I_prog
                    v0_rep = v0_of_g(v0r, beta, g_ref, g_ref)
                    for node in CHECK_NODES:
                        if node == "dac_remap":
                            vsl = vsl_nom
                        elif node == "supply_lin":
                            vsl = vsw_nom + D / a
                        else:
                            vsl = vsw_nom + v_star(D, a, v0_rep)
                        branches.append(dict(model=div_model, w=w_div, l=l_div, z=zk, vsl=vsl,
                                             k_amp=a * amp, v0=v0, vsw=vswk))
                        meta.append((k, j, lab, npz_lab, node))
        bounds = fullbranch_bounds(sp, f"{did}_fullbranch", branches)
        ok_b = np.isfinite(bounds)
        i0 = np.full(bounds.size, np.nan)
        if ok_b.any():
            i0[ok_b] = op_currents(sp, f"{did}_I0_op", [dict(div_nom, vgs=float(b), vds=vsw_nom) for b in bounds[ok_b]])
        # errors
        err = np.full(bounds.size, np.nan)
        for i, (k, j, lab, npz_lab, node) in enumerate(meta):
            if not ok_b[i]:
                continue
            g_eff = i0[i] / (a * D) if node == "dac_remap" else i0[i] / D
            x_eff = (g_eff - g_lo) / dG
            err[i] = 15 * (x_eff - theta[j])
        # own zero-mismatch self-check: dac_remap error 0; supply error = 15 (G/dG)(rho0 - 1)
        self_dev = {}
        for i, (k, j, lab, npz_lab, node) in enumerate(meta):
            if k != -1:
                continue
            expect = 0.0 if node == "dac_remap" else 15 * (G[j] / dG) * (rho0[(lab, node)][j] - 1)
            self_dev[f"{node}|{lab}|j{j}"] = {"err_lsb": float(err[i]), "expected_lsb": float(expect),
                                              "abs_dev_lsb": float(abs(err[i] - expect))}
        # comparison
        t2 = {"per_node_grid": [], "vsw_k": {}, "m0": {}, "nan_mismatch": 0,
              "self_check_zero_mismatch_max_abs_dev_lsb": max(v["abs_dev_lsb"] for v in self_dev.values()),
              "self_check_zero_mismatch": self_dev}
        maxdiff = {}
        for i, (k, j, lab, npz_lab, node) in enumerate(meta):
            if k == -1:
                continue
            main_e = float(z_arr[f"err_{node}_1y_{npz_lab}"][k, j])
            mine_e = float(err[i])
            both_nan = (not np.isfinite(main_e)) and (not np.isfinite(mine_e))
            if both_nan:
                diff, ok = 0.0, True
            elif np.isfinite(main_e) != np.isfinite(mine_e):
                diff, ok = math.inf, False
                t2["nan_mismatch"] += 1
            else:
                diff = abs(mine_e - main_e)
                ok = diff <= THR_ERR
            t2_rows.append([did, node, lab, k, j, f"{theta[j]:.6f}", fmt(bounds[i]), mine_e, main_e,
                            mine_e - main_e if np.isfinite(diff) else "", ok])
            key = (node, lab)
            cur_ = maxdiff.get(key, (-1.0, None))
            if diff > cur_[0]:
                maxdiff[key] = (diff, (k, j, mine_e, main_e))
        for (node, lab), (dmax, where) in sorted(maxdiff.items()):
            sel = [r for r in t2_rows if r[0] == did and r[1] == node and r[2] == lab]
            t2["per_node_grid"].append({
                "node": node, "grid": lab, "n_pairs": len(sel), "n_fail": sum(1 for r in sel if not r[-1]),
                "max_abs_diff_lsb": dmax, "at_instance": where[0], "at_theta_idx": where[1],
                "mine_lsb": where[2], "main_lsb": where[3],
                "rms_mine_lsb": float(np.sqrt(np.nanmean(np.array([r[7] for r in sel]) ** 2))),
                "rms_main_lsb": float(np.sqrt(np.nanmean(np.array([r[8] for r in sel]) ** 2))),
            })
        # V_sw,k and M0 comparisons
        dvk = np.abs(vsw_k[:n_inst] - z_arr["vsw_mc"][:n_inst])
        dvk_all = np.abs(vsw_k - z_arr["vsw_mc"])
        t2["vsw_k"] = {"max_abs_diff_V_first_n": float(dvk.max()), "at_instance": int(dvk.argmax()),
                       "pass": bool(dvk.max() <= THR_BOUND),
                       "max_abs_diff_V_all300_info": float(dvk_all.max()),
                       "mc_std_vsw_mine_V": float(np.std(vsw_k, ddof=1)),
                       "mc_std_vsw_main_V": float(np.std(z_arr["vsw_mc"], ddof=1))}
        for k in range(mm["n_instances"]):
            vsw_rows.append([did, k, fmt(vsw_k[k]), fmt(z_arr["vsw_mc"][k]), vsw_k[k] - z_arr["vsw_mc"][k],
                             k < n_inst, abs(vsw_k[k] - z_arr["vsw_mc"][k]) <= THR_BOUND])
        dm0 = np.abs(m0 - z_arr["b_target"])
        t2["m0"] = {"max_abs_diff_V": float(dm0.max()), "at_theta_idx": int(dm0.argmax()),
                    "pass": bool(dm0.max() <= THR_BOUND)}
        for j in range(16):
            b_rows.append([did, j, f"{theta[j]:.6f}", fmt(m0[j]), fmt(z_arr["b_target"][j]), m0[j] - z_arr["b_target"][j],
                           abs(m0[j] - z_arr["b_target"][j]) <= THR_BOUND])
        all_err_ok = all(r[-1] for r in t2_rows if r[0] == did)
        t2["n_pairs_total"] = sum(1 for r in t2_rows if r[0] == did)
        t2["n_fail_total"] = sum(1 for r in t2_rows if r[0] == did and not r[-1])
        t2["max_abs_err_diff_lsb"] = max(p["max_abs_diff_lsb"] for p in t2["per_node_grid"])
        t2["n_invalid_bounds_mine"] = int((~ok_b).sum())
        # B1-type information: own dac_remap errors, nonlinear minus linear, and main b1_fullbranch arrays
        dac = {}
        for i, (k, j, lab, npz_lab, node) in enumerate(meta):
            if node == "dac_remap":
                dac[(k, j, lab)] = err[i]
        b1 = {}
        for lab, npz_lab, _, _ in CHECK_GRID[1:]:
            dd = [abs(dac[(k, j, lab)] - dac[(k, j, "linear")]) for k in range(n_inst) for j in jj]
            b1[lab] = float(np.max(dd))
        b1_main = {}
        for lab, npz_lab, _, _ in CHECK_GRID[:3]:
            keyz = f"b1_fullbranch_{npz_lab}"
            if keyz in z_arr.files:
                arr = z_arr[keyz]
                mine_arr = np.array([[dac[(k, j, lab)] for j in jj] for k in range(n_inst)])
                b1_main[lab] = float(np.max(np.abs(mine_arr - arr[:n_inst].astype(float))))
        t2["info_dac_remap_nonlinear_minus_linear_max_lsb"] = b1
        t2["info_vs_main_b1_fullbranch_arrays_max_abs_diff_lsb"] = b1_main
        t2["pass"] = bool(all_err_ok and t2["vsw_k"]["pass"] and t2["m0"]["pass"])
        summary["task2"][did] = t2
        print(f"    Task 2 {did}: pass = {t2['pass']}, max |err diff| = {t2['max_abs_err_diff_lsb']:.3g} LSB", flush=True)
        summary["task1"][did]["my_predictions_1y"] = {f"{lab}|{node}": v for (lab, node), v in my_pred.items()}

    summary["task1_pass"] = all(summary["task1"][d]["pass"] for d in CHECK_DESIGNS)
    summary["task2_pass"] = all(summary["task2"][d]["pass"] for d in CHECK_DESIGNS)
    summary["spice_runs"] = sp.log
    summary["runtime_s"] = round(time.time() - t_start, 1)

    def write_csv(name, header, rows):
        with open(outdir / name, "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(header)
            wr.writerows(rows)

    write_csv("task1_predictor_compare.csv",
              ["design", "quantity", "grid", "node", "mine", "main", "mine_minus_main", "rel_diff", "pass"], t1_rows)
    write_csv("task2_error_compare.csv",
              ["design", "node", "grid", "instance", "theta_idx", "theta", "bound_V_mine", "err_lsb_mine",
               "err_lsb_main", "mine_minus_main", "pass"], t2_rows)
    write_csv("task2_m0_compare.csv",
              ["design", "theta_idx", "theta", "m0_V_mine", "b_target_V_main", "mine_minus_main", "pass"], b_rows)
    write_csv("task2_vsw_compare.csv",
              ["design", "instance", "vsw_V_mine", "vsw_V_main", "mine_minus_main", "in_spot_check", "pass"], vsw_rows)
    (outdir / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    print(f"Task 1 pass: {summary['task1_pass']}; Task 2 pass: {summary['task2_pass']}; "
          f"{summary['runtime_s']} s", flush=True)


if __name__ == "__main__":
    main()
