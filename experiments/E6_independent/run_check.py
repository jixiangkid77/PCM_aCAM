#!/usr/bin/env python3
"""PCM_aCAM E6 independent check, part 1: own values, computed before any comparison.

Sources used: docs/E6_protocol.md (frozen plan, sections 2-4 and 10), experiments/E6_config.json,
docs/E4_protocol.md section 2 (cell model), the sky130 tt model subset
(third_party/sky130_fd_pr_subset_e3b) and src/pcm_acam/trees.py (dataset_bundle only: the E4
forest construction, not under test).  Everything else -- netlists, ngspice runner, binary
raw-file parser, curve inversion, settling-time extraction, power accounting -- is written here.

Items
  0  nominal switching points V_sw, V_Y*, V_X*
  1  settling times, transient SPICE, C_par = 0.5 fF on the divider node
       1a  plan section 10: P1, P3 x L6T x {t0, dac_remap 1 y} x x_prev {0, 1}, theta = 0
       1b  additional: L6T, theta = 1 (level 15), x_prev = 0
       1c  additional: UINV (6T2M map), theta = 0 (level 0), x_prev = 1
  2  static power of one programmed bound (DC SPICE), theta = 8/15 (level 8), x in {0.25, 0.75}
  3  array counts of the E4 forest, iris, splits 0, 1, 2

Run (see README.md):  python experiments/E6_independent/run_check.py
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import importlib.util
import json
import math
import os
import re
import resource
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True              # importing trees.py must not write into src/

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]                      # experiments/E6_independent -> project root
CONFIG = ROOT / "experiments" / "E6_config.json"
TREES = ROOT / "src" / "pcm_acam" / "trees.py"
RESULTS = ROOT / "results"
JST = dt.timezone(dt.timedelta(hours=9))

NFET = "sky130_fd_pr__nfet_01v8"
NFET_LVT = "sky130_fd_pr__nfet_01v8_lvt"
PFET = "sky130_fd_pr__pfet_01v8"

# SPICE options: the plan's (E6 protocol section 3) and a tighter set used only as a DC check
OPTS_PLAN = ".options reltol=1e-6 vntol=1e-9 abstol=1e-15"
OPTS_TIGHT = ".options reltol=1e-9 vntol=1e-12 abstol=1e-18"

COARSE_STEP = 1e-3      # V, first sweep for brackets
FINE_STEP = 1e-6        # V, second sweep inside the bracket (linear interpolation on it)
FINE_MARGIN = 2e-6      # V, margin of the fine sweep around the coarse bracket


# ----------------------------------------------------------------------------- utilities
def now_jst() -> dt.datetime:
    return dt.datetime.now(JST)


def stamp(t: dt.datetime | None = None) -> str:
    return (t or now_jst()).strftime("[%Y-%m-%d %H:%M:%S]")


class Log:
    def __init__(self, path: Path):
        self.f = open(path, "a", encoding="utf-8")

    def __call__(self, msg: str) -> None:
        for line in str(msg).splitlines() or [""]:
            s = f"{stamp()} {line}"
            print(s, flush=True)
            self.f.write(s + "\n")
        self.f.flush()

    def close(self) -> None:
        self.f.close()


def fmt(v: float) -> str:
    """Full-precision number for a netlist (repr of a double, read back exactly by ngspice)."""
    v = float(v)
    if not math.isfinite(v):
        raise ValueError(v)
    return repr(v)


def new_out_dir(base: Path) -> Path:
    t = now_jst().strftime("%Y%m%d_%H%M")
    name = f"E6_independent_check_{t}"
    cand = base / name
    for suf in ["", "_b", "_c", "_d", "_e", "_f", "_g", "_h"]:
        cand = base / (name + suf)
        if not cand.exists():
            cand.mkdir(parents=True)
            return cand
    raise RuntimeError("no free output folder name")


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return [jsonable(v) for v in o.tolist()]
    if isinstance(o, float) and not math.isfinite(o):
        return None
    return o


# ----------------------------------------------------------------------------- ngspice runner
def read_raw(path: Path) -> dict:
    """Parse an ngspice binary raw file (real data).  Returns {name: array}; '_names' keeps order."""
    b = path.read_bytes()
    k = b.find(b"Binary:\n")
    if k < 0:
        raise RuntimeError(f"{path.name}: not a binary raw file")
    hdr = b[:k].decode("latin-1").splitlines()
    nvar = npt = None
    names: list[str] = []
    flags = ""
    i = 0
    while i < len(hdr):
        line = hdr[i]
        if line.startswith("Flags:"):
            flags = line.split(":", 1)[1].strip()
        elif line.startswith("No. Variables:"):
            nvar = int(line.split(":", 1)[1])
        elif line.startswith("No. Points:"):
            npt = int(line.split(":", 1)[1])
        elif line.startswith("Variables:"):
            for j in range(nvar):
                names.append(hdr[i + 1 + j].split()[1].lower())
            i += nvar
        i += 1
    if "complex" in flags:
        raise RuntimeError("complex raw data not supported")
    data = np.frombuffer(b, dtype="<f8", count=nvar * npt, offset=k + 8).reshape(npt, nvar)
    out = {n: data[:, j].copy() for j, n in enumerate(names)}
    out["_names"] = names
    return out


class Spice:
    """Runs one ngspice process at a time in a private temporary directory."""

    def __init__(self, lib: Path, log: Log, netlist_dir: Path | None, tmp_base: str | None):
        self.tmp = Path(tempfile.mkdtemp(prefix="E6ind_spice_", dir=tmp_base))
        (self.tmp / ".spiceinit").write_text("set ngbehavior=hsa\nset ng_nomodcheck\n")
        self.lib = lib
        self.log = log
        self.netlist_dir = netlist_dir
        # libgomp is linked into ngspice: with both cores busy, spinning OpenMP waits make a run
        # up to ~50x slower (results bit-identical); one thread with passive waits avoids that.
        self.env = dict(os.environ, OMP_NUM_THREADS="1", OMP_WAIT_POLICY="passive")
        self.n = 0
        self.wall = 0.0
        self.cpu = 0.0
        self.warnings: list[str] = []

    def version(self) -> str:
        r = subprocess.run(["ngspice", "--version"], capture_output=True, text=True, env=self.env)
        m = re.search(r"ngspice-[0-9A-Za-z.+-]+", r.stdout)
        return m.group(0) if m else r.stdout.strip().splitlines()[0]

    def run(self, tag: str, body: str, control: str, options: str = OPTS_PLAN) -> dict:
        self.n += 1
        stem = f"{self.n:03d}_{tag}"
        keys = re.findall(r"@RAW\((\w+)\)@", control)
        for key in keys:
            control = control.replace(f"@RAW({key})@", f"{stem}_{key}.raw")
        net = (f"* E6 independent check: {tag}\n"
               f".include \"{self.lib}\"\n"
               f".temp 27\n"
               f"{options}\n"
               f"{body.strip()}\n"
               f".control\nset filetype=binary\n{control.strip()}\nquit\n.endc\n.end\n")
        cir = self.tmp / f"{stem}.cir"
        cir.write_text(net)
        if self.netlist_dir is not None:
            shutil.copy(cir, self.netlist_dir / cir.name)
        logf = self.tmp / f"{stem}.log"
        r0 = resource.getrusage(resource.RUSAGE_CHILDREN)
        t0 = time.time()
        with open(logf, "w") as lf:
            rc = subprocess.run(["ngspice", "-b", cir.name], cwd=self.tmp, env=self.env,
                                stdout=lf, stderr=subprocess.STDOUT).returncode
        wall = time.time() - t0
        r1 = resource.getrusage(resource.RUSAGE_CHILDREN)
        cpu = (r1.ru_utime - r0.ru_utime) + (r1.ru_stime - r0.ru_stime)
        self.wall += wall
        self.cpu += cpu
        text = logf.read_text(errors="replace")
        bad = [ln for ln in text.splitlines()
               if re.search(r"error|warning|singular|too small|abort|fail", ln, re.I)
               and "Reference value" not in ln]
        if bad:
            self.warnings.extend(f"{stem}: {ln.strip()}" for ln in bad[:10])
            self.log(f"  ngspice {stem}: {len(bad)} warning/error line(s), first: {bad[0].strip()}")
        if rc != 0:
            raise RuntimeError(f"ngspice failed ({rc}) for {stem}; see {logf}\n{text[-2000:]}")
        out = {}
        for key in keys:
            p = self.tmp / f"{stem}_{key}.raw"
            if not p.exists():
                raise RuntimeError(f"{stem}: missing raw file {p.name}\n{text[-2000:]}")
            out[key] = read_raw(p)
        self.log(f"  ngspice {stem}: {wall:.2f} s wall, {cpu:.2f} s cpu")
        return out

    def cleanup(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)


# ----------------------------------------------------------------------------- numerics helpers
def bracket_increasing(y: np.ndarray, target: float) -> int:
    """Index k with y[k-1] < target <= y[k]; the upward crossing must be unique on the grid."""
    up = np.nonzero((y[:-1] < target) & (y[1:] >= target))[0]
    down = np.nonzero((y[:-1] >= target) & (y[1:] < target))[0]
    if len(up) != 1 or len(down) != 0:
        raise RuntimeError(f"target {target}: {len(up)} upward / {len(down)} downward crossings "
                           f"in [{y.min()}, {y.max()}]")
    return int(up[0]) + 1


def interp_root(x: np.ndarray, y: np.ndarray, target: float) -> tuple[float, float]:
    """Linear interpolation of x at y = target on an increasing grid; also the local slope dy/dx."""
    k = bracket_increasing(y, target)
    seg = slice(max(k - 3, 0), min(k + 3, len(y)))
    if not np.all(np.diff(y[seg]) > 0):
        raise RuntimeError("curve not strictly increasing near the root")
    x0, x1, y0, y1 = x[k - 1], x[k], y[k - 1], y[k]
    slope = (y1 - y0) / (x1 - x0)
    return float(x0 + (target - y0) / slope), float(slope)


def roots_by_sweeps(sp: Spice, tag: str, body: str, src: str, yfun, targets: dict,
                    lo: float, hi: float) -> dict:
    """Roots of yfun(raw) = target (yfun increasing in the swept source) for several targets:
    coarse sweep (1 mV) for the brackets, then one fine sweep (1 uV) per distinct bracket in a
    single ngspice run, linear interpolation on the fine grid."""
    r = sp.run(f"{tag}_coarse", body, f"dc {src} {fmt(lo)} {fmt(hi)} {fmt(COARSE_STEP)}\n"
                                      f"write @RAW(s)@ all")["s"]
    xs = r[r["_names"][0]]
    ys = yfun(r)
    brackets = {}
    for key, tg in targets.items():
        k = bracket_increasing(ys, tg)
        brackets[key] = (round(float(xs[k - 1]), 9), round(float(xs[k]), 9))
    uniq = sorted(set(brackets.values()))
    ctl = []
    for j, (a, b) in enumerate(uniq):
        ctl.append(f"dc {src} {fmt(a - FINE_MARGIN)} {fmt(b + FINE_MARGIN)} {fmt(FINE_STEP)}")
        ctl.append(f"write @RAW(f{j})@ all")
    rf = sp.run(f"{tag}_fine", body, "\n".join(ctl))
    out = {}
    for key, tg in targets.items():
        j = uniq.index(brackets[key])
        raw = rf[f"f{j}"]
        x = raw[raw["_names"][0]]
        root, slope = interp_root(x, yfun(raw), tg)
        out[key] = {"root": root, "slope": slope, "coarse_bracket": list(brackets[key]),
                    "fine_points": int(len(x))}
    return out


# ----------------------------------------------------------------------------- circuit pieces
class Model:
    """Constants of the plan/config (checked against the task specification)."""

    def __init__(self, cfg: dict):
        c = cfg["circuit"]
        self.vdd = float(c["vdd"])
        self.d = float(c["d_v"])
        self.g_lo = float(c["g_lo_us"]) * 1e-6
        self.g_span = (float(c["g_hi_us"]) - float(c["g_lo_us"])) * 1e-6
        self.pd_w, self.pd_l = float(c["pulldown"]["w_um"]), float(c["pulldown"]["l_um"])
        inv = c["inverter"]
        self.wn, self.ln, self.wp, self.lp = float(inv["wn"]), float(inv["ln"]), float(inv["wp"]), float(inv["lp"])
        self.cpar = float(c["c_par_f"])
        self.designs = {d["id"]: (float(d["w_div_um"]), float(d["l_div_um"])) for d in cfg["designs"]}
        self.nu = float(cfg["drift"]["nu"])
        self.t0_s = float(cfg["drift"]["t0_s"])
        self.t_1y = float(cfg["drift"]["times_s"]["1y"])
        s = cfg["settle"]
        self.t_edge = float(s["t_edge_s"])
        self.t_rise = float(s["t_rise_s"])
        self.t_stop = {k: float(v) for k, v in s["t_stop_s"].items()}
        self.t_step = float(s["t_step_s"])
        self.t_max = float(s["t_max_step_s"])
        self.tol_lsb = float(s["tolerance_lsb"])
        self.levels = int(cfg["levels"])
        self.lsb = 1.0 / (self.levels - 1)
        self.i_sw = 100e-9 * self.pd_w / self.pd_l
        # the task specification, restated
        spec = dict(vdd=1.8, d=0.2, g_lo=2.5e-6, g_span=20e-6, pd_w=0.42, pd_l=0.15, wn=2.0, ln=0.15,
                    wp=0.42, lp=4.0, cpar=0.5e-15, nu=0.049, t0_s=20.0, t_1y=31_536_000.0,
                    t_edge=1e-9, t_rise=1e-10, t_max=2e-11, tol_lsb=0.01, levels=16)
        for k, v in spec.items():
            if not math.isclose(getattr(self, k), v, rel_tol=1e-12, abs_tol=1e-30):
                raise RuntimeError(f"config value {k} = {getattr(self, k)} differs from the specification {v}")
        if self.designs != {"P1": (0.42, 0.15), "P3": (0.42, 4.0)}:
            raise RuntimeError(f"designs differ from the specification: {self.designs}")
        if self.t_stop.get("L6T") != 1.5e-7 or self.t_stop.get("UINV") != 6e-7:
            raise RuntimeError(f"stop times differ from the specification: {self.t_stop}")
        if cfg["settle"]["spice_options"].split() != OPTS_PLAN.split():
            raise RuntimeError("SPICE options of the config differ from the plan text")

    def G(self, x: float) -> float:
        return self.g_lo + self.g_span * x

    def a(self, t: float) -> float:
        return (t / self.t0_s) ** (-self.nu)


def dev(name: str, d: str, g: str, s: str, b: str, model: str, w: float, l: float) -> str:
    return f"x{name} {d} {g} {s} {b} {model} w={fmt(w)} l={fmt(l)}"


def inv_lines(m: Model, tag: str, vin: str, vout: str, vsup: str) -> list[str]:
    """INV_S: pMOS 0.42/4 (source and body at its own supply node vsup), nMOS 2.0/0.15."""
    return [dev(f"p_{tag}", vout, vin, vsup, vsup, PFET, m.wp, m.lp),
            dev(f"n_{tag}", vout, vin, "0", "0", NFET, m.wn, m.ln)]


def branch_lines(m: Model, tag: str, btype: str, l_div: float, g: float, vsl: float, vdl: str,
                 cpar: float = 0.0) -> tuple[list[str], str, str]:
    """One programmed bound with its own sources.  Returns (lines, divider node, decision node)."""
    n = f"n_{tag}"
    lines = [f"vsl_{tag} sl_{tag} 0 {fmt(vsl)}",
             f"r_{tag} sl_{tag} {n} {fmt(1.0 / g)}",                       # PCM: linear 1/G
             dev(f"div_{tag}", n, f"dl_{tag}", "0", "0", NFET, 0.42, l_div),  # divider
             f"vdl_{tag} dl_{tag} 0 {vdl}",
             f"vml_{tag} ml_{tag} 0 {fmt(m.vdd)}"]                            # precharged ML
    if btype == "L6T":
        dec = n
    elif btype == "UINV":
        dec = f"o_{tag}"
        lines.append(f"vinv_{tag} vi_{tag} 0 {fmt(m.vdd)}")
        lines += inv_lines(m, tag, n, dec, f"vi_{tag}")
    else:
        raise ValueError(btype)
    lines.append(dev(f"pd_{tag}", f"ml_{tag}", dec, "0", "0", NFET_LVT, m.pd_w, m.pd_l))  # pull-down
    if cpar > 0:
        lines.append(f"cpar_{tag} {n} 0 {fmt(cpar)}")
    return lines, n, dec


# ----------------------------------------------------------------------------- item 0
def switching_points(sp: Spice, m: Model, log: Log) -> dict:
    res = {}
    # V_sw: pull-down at V_DS = 1.8 V conducts 100 nA x W/L
    body = "\n".join([f"vd d 0 {fmt(m.vdd)}", "vg g 0 0",
                      dev("pd", "d", "g", "0", "0", NFET_LVT, m.pd_w, m.pd_l)])
    r = roots_by_sweeps(sp, "vsw", body, "vg", lambda raw: -raw["i(vd)"], {"vsw": m.i_sw}, 0.0, m.vdd)
    vsw = r["vsw"]["root"]
    ver = sp.run("vsw_verify", body.replace("vg g 0 0", f"vg g 0 {fmt(vsw)}"), "op\nwrite @RAW(o)@ all")["o"]
    i_at = float(-ver["i(vd)"][0])
    res["V_sw"] = {"value_V": vsw, "target_current_A": m.i_sw, "current_at_root_A": i_at,
                   "rel_residual": i_at / m.i_sw - 1.0,
                   "voltage_residual_V": (m.i_sw - i_at) / r["vsw"]["slope"],
                   "gm_S": r["vsw"]["slope"], "method": "1 mV sweep -> 1 uV sweep in the bracket -> linear "
                                                       "interpolation of I_D(V_GS); .op at the root"}
    log(f"V_sw = {vsw:.9f} V (I at root {i_at:.9e} A, rel. residual {i_at / m.i_sw - 1:.2e})")

    # V_Y*: INV_S input at which its output equals V_sw (output held at V_sw, root of its current)
    def inv_forced_body(vout: float) -> str:
        return "\n".join([f"vsup vs 0 {fmt(m.vdd)}", "vin in 0 0", f"vo o 0 {fmt(vout)}",
                          *inv_lines(m, "i1", "in", "o", "vs"),
                          f"vml ml 0 {fmt(m.vdd)}", dev("pd", "ml", "o", "0", "0", NFET_LVT, m.pd_w, m.pd_l)])
    # i(vo) = current into the + node of vo; pMOS sources it (V_in low) -> decreasing in V_in
    r = roots_by_sweeps(sp, "vy_forced", inv_forced_body(vsw), "vin", lambda raw: -raw["i(vo)"],
                        {"vy": 0.0}, 0.0, m.vdd)
    vy = r["vy"]["root"]
    # free-output check: one INV_S driving the pull-down, .op at V_in = V_Y*
    free1 = "\n".join([f"vsup vs 0 {fmt(m.vdd)}", f"vin in 0 {fmt(vy)}", *inv_lines(m, "i1", "in", "o", "vs"),
                       f"vml ml 0 {fmt(m.vdd)}", dev("pd", "ml", "o", "0", "0", NFET_LVT, m.pd_w, m.pd_l)])
    o1 = sp.run("vy_free_op", free1, "op\nwrite @RAW(o)@ all")["o"]
    vout_y = float(o1["v(o)"][0])
    # free sweep around the root for the inverter gain
    g1 = sp.run("vy_free_sweep", free1.replace(f"vin in 0 {fmt(vy)}", "vin in 0 0"),
                f"dc vin {fmt(vy - 1e-4)} {fmt(vy + 1e-4)} {fmt(1e-5)}\nwrite @RAW(s)@ all")["s"]
    gain_y = float(np.polyfit(g1["v(v-sweep)"], g1["v(o)"], 1)[0])
    res["V_Y_star"] = {"value_V": vy, "method": "INV_S output held at V_sw by a source, 1 mV then 1 uV sweep of "
                                                "the input, linear interpolation of the zero of the output-source "
                                                "current; free-output .op at the root",
                       "free_output_at_root_V": vout_y, "free_output_minus_V_sw_V": vout_y - vsw,
                       "inverter_gain_dVout_dVin": gain_y}
    log(f"V_Y* = {vy:.9f} V (free output at root - V_sw = {vout_y - vsw:.2e} V, gain {gain_y:.2f})")

    # V_X*: two INV_S in series, second output = V_sw.  Primary: the full cascade, free sweep.
    casc = "\n".join([f"vsup vs 0 {fmt(m.vdd)}", "vin in 0 0", *inv_lines(m, "i1", "in", "mid", "vs"),
                      *inv_lines(m, "i2", "mid", "o", "vs"),
                      f"vml ml 0 {fmt(m.vdd)}", dev("pd", "ml", "o", "0", "0", NFET_LVT, m.pd_w, m.pd_l)])
    r = roots_by_sweeps(sp, "vx_cascade", casc, "vin", lambda raw: raw["v(o)"], {"vx": vsw}, 0.0, m.vdd)
    vx = r["vx"]["root"]
    # cross-check: first INV_S with its output held at V_Y*
    r2 = roots_by_sweeps(sp, "vx_forced", inv_forced_body(vy), "vin", lambda raw: -raw["i(vo)"],
                         {"vx": 0.0}, 0.0, m.vdd)
    vx2 = r2["vx"]["root"]
    o2 = sp.run("vx_free_op", casc.replace("vin in 0 0", f"vin in 0 {fmt(vx)}"), "op\nwrite @RAW(o)@ all")["o"]
    res["V_X_star"] = {"value_V": vx, "method": "two INV_S in series driving the pull-down, 1 mV then 1 uV sweep of "
                                                "the input, linear interpolation of V_out = V_sw",
                       "cascade_gain_dVout_dVin": r["vx"]["slope"],
                       "cross_check_first_inverter_output_held_at_V_Y_star_V": vx2,
                       "cross_check_difference_V": vx2 - vx,
                       "op_at_root_output_minus_V_sw_V": float(o2["v(o)"][0]) - vsw,
                       "op_at_root_mid_node_V": float(o2["v(mid)"][0])}
    log(f"V_X* = {vx:.9f} V (cascade gain {r['vx']['slope']:.1f}; forced-output cross-check {vx2:.9f} V, "
        f"diff {vx2 - vx:.2e} V)")
    return res


# ----------------------------------------------------------------------------- maps and programming
def divider_maps(sp: Spice, m: Model, design: str, vsw: float, vy: float, a1y: float, xs: dict, log: Log) -> dict:
    """M_f(x) = I_0^-1(f G(x) D) on the nominal divider of the design at V_DS = V_sw, f in {1, a(1 y)};
    plus I_div(M0(theta), V_DS = V_Y*) for the UINV programming."""
    w, l = m.designs[design]
    body = "\n".join([f"vd d 0 {fmt(vsw)}", "vg g 0 0", dev("dv", "d", "g", "0", "0", NFET, w, l)])
    targets = {}
    for fkey, f in (("M0", 1.0), ("Ma", a1y)):
        for xkey, x in xs.items():
            targets[(fkey, xkey)] = f * m.G(x) * m.d
    tkeys = list(targets)
    r = roots_by_sweeps(sp, f"map_{design}", body, "vg", lambda raw: -raw["i(vd)"],
                        {f"{k[0]}|{k[1]}": targets[k] for k in tkeys}, 0.0, m.vdd)
    maps = {"M0": {}, "Ma": {}}
    for k in tkeys:
        e = r[f"{k[0]}|{k[1]}"]
        maps[k[0]][k[1]] = {"x": xs[k[1]], "V_DL": e["root"], "I_target_A": targets[k], "gm_S": e["slope"]}
    # verification .op at every root (drain at V_sw), and the UINV programming currents (drain at V_Y*)
    lines = []
    for j, k in enumerate(tkeys):
        vg = maps[k[0]][k[1]]["V_DL"]
        lines += [f"vd{j} d{j} 0 {fmt(vsw)}", f"vg{j} g{j} 0 {fmt(vg)}",
                  dev(f"dv{j}", f"d{j}", f"g{j}", "0", "0", NFET, w, l)]
    prog_x = {"0": 0.0, "8/15": 8.0 / 15.0, "1": 1.0}
    for j, (pk, x) in enumerate(prog_x.items()):
        vg = maps["M0"][pk]["V_DL"]
        lines += [f"vu{j} du{j} 0 {fmt(vy)}", f"vgu{j} gu{j} 0 {fmt(vg)}",
                  dev(f"du{j}", f"du{j}", f"gu{j}", "0", "0", NFET, w, l)]
    o = sp.run(f"map_{design}_verify", "\n".join(lines), "op\nwrite @RAW(o)@ all")["o"]
    worst = 0.0
    for j, k in enumerate(tkeys):
        e = maps[k[0]][k[1]]
        i_at = float(-o[f"i(vd{j})"][0])
        e["I_at_root_A"] = i_at
        e["rel_residual"] = i_at / e["I_target_A"] - 1.0
        e["voltage_residual_V"] = (e["I_target_A"] - i_at) / e["gm_S"]
        worst = max(worst, abs(e["voltage_residual_V"]))
    prog = {}
    for j, (pk, x) in enumerate(prog_x.items()):
        i_u = float(-o[f"i(vu{j})"][0])
        prog[pk] = {"theta": x, "G_L6T_S": m.G(x), "I_div_at_V_Y_star_A": i_u, "G_UINV_S": i_u / m.d,
                    "G_UINV_over_G": i_u / m.d / m.G(x)}
    log(f"{design}: map roots verified, worst |voltage residual| = {worst:.2e} V; "
        f"G_UINV(0) = {prog['0']['G_UINV_S'] * 1e6:.6f} uS, G_UINV(8/15) = {prog['8/15']['G_UINV_S'] * 1e6:.6f} uS")
    return {"maps": maps, "programming": prog, "worst_map_voltage_residual_V": worst}


def node_setting(m: Model, sw: dict, design_data: dict, btype: str, node: str, theta_key: str, a1y: float) -> dict:
    vsp = sw["V_sw"]["value_V"] if btype == "L6T" else sw["V_Y_star"]["value_V"]
    p = design_data["programming"][theta_key]
    gk = p["G_L6T_S"] if btype == "L6T" else p["G_UINV_S"]
    if node == "t0":
        return {"G_S": gk, "V_SL": vsp + m.d, "map": "M0", "a": 1.0}
    if node == "dac_remap_1y":
        return {"G_S": a1y * gk, "V_SL": vsp + m.d, "map": "Ma", "a": a1y}
    if node == "supply_1y":
        return {"G_S": a1y * gk, "V_SL": vsp + m.d / a1y, "map": "M0", "a": a1y}
    raise ValueError(node)


# ----------------------------------------------------------------------------- item 1
def settle_time(t: np.ndarray, v: np.ndarray, lo: float, hi: float, t_edge: float) -> tuple[float | None, str]:
    """Time of the last exit from 'outside the band' (= final entry into [lo, hi]) after t_edge, minus
    t_edge, by linear interpolation between the last outside sample and the next one."""
    sel = t >= t_edge - 1e-18
    tt, vv = t[sel], v[sel]
    out = (vv < lo) | (vv > hi)
    if not out.any():
        return 0.0, "never outside after t_edge"
    k = int(np.nonzero(out)[0][-1])
    if k == len(tt) - 1:
        return None, "unsettled"
    b = hi if vv[k] > hi else lo
    tc = tt[k] + (b - vv[k]) * (tt[k + 1] - tt[k]) / (vv[k + 1] - vv[k])
    return float(tc - t_edge), "ok"


def band_crossings(t: np.ndarray, v: np.ndarray, lo: float, hi: float, t_edge: float) -> int:
    """Number of inside/outside changes of the band after t_edge (1 = one final entry, no re-exit)."""
    sel = t >= t_edge - 1e-18
    out = ((v[sel] < lo) | (v[sel] > hi)).astype(int)
    return int(np.sum(np.abs(np.diff(out))))


def tail_tau(t: np.ndarray, v: np.ndarray, v_final: float, half_band: float, t_edge: float) -> float | None:
    """Time constant of the approach to v_final: straight-line fit of ln|v - v_final| on the samples
    after the edge whose distance lies between 2 and 20 half-bands (diagnostic only)."""
    sel = t > t_edge + 0.1e-9
    dev_ = np.abs(v[sel] - v_final)
    tt = t[sel]
    win = (dev_ > 2 * half_band) & (dev_ < 20 * half_band)
    if win.sum() < 5:
        return None
    p = np.polyfit(tt[win], np.log(dev_[win]), 1)
    return float(-1.0 / p[0]) if p[0] < 0 else None


def settling_cases() -> list[dict]:
    cases = []
    for design in ("P1", "P3"):
        for node in ("t0", "dac_remap_1y"):
            for xp in (0.0, 1.0):
                cases.append(dict(group="1a", design=design, btype="L6T", node=node, theta_key="0", x_prev=xp))
    for design in ("P1", "P3"):
        for node in ("t0", "dac_remap_1y"):
            cases.append(dict(group="1b", design=design, btype="L6T", node=node, theta_key="1", x_prev=0.0))
    for design in ("P1", "P3"):
        for node in ("t0", "dac_remap_1y"):
            cases.append(dict(group="1c", design=design, btype="UINV", node=node, theta_key="0", x_prev=1.0))
    for j, c in enumerate(cases):
        c["id"] = f"s{j:02d}"
    return cases


def xkey_of(theta_key: str, which: str) -> str:
    return {"m": f"{theta_key}-d", "0": theta_key, "p": f"{theta_key}+d"}[which]


def item1(sp: Spice, m: Model, sw: dict, dd: dict, a1y: float, log: Log, conv: bool) -> dict:
    cases = settling_cases()
    # DC band limits, DC at M(theta) and at M(x_prev): one netlist, plan options and tight options
    dc = {}
    for opts, okey in ((OPTS_PLAN, "plan"), (OPTS_TIGHT, "tight")):
        lines, nodes = [], {}
        for c in cases:
            ns = node_setting(m, sw, dd[c["design"]], c["btype"], c["node"], c["theta_key"], a1y)
            mp = dd[c["design"]]["maps"][ns["map"]]
            l_div = m.designs[c["design"]][1]
            pts = {"m": xkey_of(c["theta_key"], "m"), "0": c["theta_key"], "p": xkey_of(c["theta_key"], "p"),
                   "prev": "0" if c["x_prev"] == 0.0 else "1"}
            for w, xk in pts.items():
                tag = f"{c['id']}{w}"
                ls, n, dec = branch_lines(m, tag, c["btype"], l_div, ns["G_S"], ns["V_SL"], fmt(mp[xk]["V_DL"]))
                lines += ls
                nodes[(c["id"], w)] = (n, dec)
        o = sp.run(f"settle_dc_{okey}", "\n".join(lines), "op\nwrite @RAW(o)@ all", options=opts)["o"]
        dc[okey] = {k: (float(o[f"v({n})"][0]), float(o[f"v({dec})"][0])) for k, (n, dec) in nodes.items()}
    dc_diff = max(abs(dc["plan"][k][i] - dc["tight"][k][i]) for k in dc["plan"] for i in (0, 1))
    log(f"settling DC points: max |plan - tight options| = {dc_diff:.2e} V over {len(dc['plan'])} op points x 2 nodes")

    results = []
    for c in cases:
        ns = node_setting(m, sw, dd[c["design"]], c["btype"], c["node"], c["theta_key"], a1y)
        mp = dd[c["design"]]["maps"][ns["map"]]
        l_div = m.designs[c["design"]][1]
        th = float(c["theta_key"]) if c["theta_key"] in ("0", "1") else 8 / 15
        v_prev = mp["0" if c["x_prev"] == 0.0 else "1"]["V_DL"]
        v_th = mp[c["theta_key"]]["V_DL"]
        d_m, d_0, d_p = (dc["plan"][(c["id"], w)][1] for w in ("m", "0", "p"))
        lo, hi = min(d_m, d_p), max(d_m, d_p)
        slope_lsb = (d_p - d_m) / (2 * m.tol_lsb)          # decision-node volts per LSB at theta
        t_stop = m.t_stop[c["btype"]]
        pwl = (f"pwl(0 {fmt(v_prev)} {fmt(m.t_edge)} {fmt(v_prev)} {fmt(m.t_edge + m.t_rise)} {fmt(v_th)} "
               f"{fmt(t_stop * 1.01)} {fmt(v_th)})")
        runs = [("plan", m.t_step, m.t_max, OPTS_PLAN)]
        if conv:
            runs += [("half_step", m.t_step / 2, m.t_max / 2, OPTS_PLAN),
                     ("quarter_step", m.t_step / 4, m.t_max / 4, OPTS_PLAN),
                     ("gear", m.t_step, m.t_max, OPTS_PLAN + "\n.options method=gear")]
        rec = dict(c)
        rec.update({"theta": th, "level": int(round(th * (m.levels - 1))), "G_uS": ns["G_S"] * 1e6,
                    "V_SL_V": ns["V_SL"], "map": ns["map"], "a": ns["a"], "V_DL_prev_V": v_prev, "V_DL_theta_V": v_th,
                    "V_DL_theta_minus_V": mp[xkey_of(c["theta_key"], "m")]["V_DL"],
                    "V_DL_theta_plus_V": mp[xkey_of(c["theta_key"], "p")]["V_DL"],
                    "band_lo_V": lo, "band_hi_V": hi, "band_width_mV": (hi - lo) * 1e3,
                    "V_dec_DC_theta_V": d_0, "V_dec_DC_theta_minus_V": d_m, "V_dec_DC_theta_plus_V": d_p,
                    "V_div_DC_theta_V": dc["plan"][(c["id"], "0")][0],
                    "V_dec_DC_prev_V": dc["plan"][(c["id"], "prev")][1],
                    "slope_V_per_LSB": slope_lsb, "t_stop_s": t_stop, "C_par_F": m.cpar,
                    "DC_theta_minus_V_sw_LSB": (d_0 - sw["V_sw"]["value_V"]) / slope_lsb})
        for rkey, tstep, tmax, topts in runs:
            ls, n, dec = branch_lines(m, "t", c["btype"], l_div, ns["G_S"], ns["V_SL"], pwl, cpar=m.cpar)
            vecs = " ".join(sorted({f"v({n})", f"v({dec})", "v(dl_t)"}))
            raw = sp.run(f"tran_{c['id']}_{rkey}", "\n".join(ls),
                         f"tran {fmt(tstep)} {fmt(t_stop)} 0 {fmt(tmax)}\nwrite @RAW(tr)@ {vecs}",
                         options=topts)["tr"]
            t = raw["time"]
            v = raw[f"v({dec})"]
            ts, status = settle_time(t, v, lo, hi, m.t_edge)
            after = t >= m.t_edge
            dts = np.diff(t[after])
            rec[f"{rkey}_t_settle_s"] = ts
            rec[f"{rkey}_status"] = status
            rec[f"{rkey}_n_points"] = int(len(t))
            rec[f"{rkey}_max_dt_s"] = float(dts.max()) if len(dts) else None
            rec[f"{rkey}_tstep_s"] = tstep
            rec[f"{rkey}_tmax_s"] = tmax
            rec[f"{rkey}_V_dec_t0_V"] = float(v[0])
            rec[f"{rkey}_V_dec_end_V"] = float(v[-1])
            rec[f"{rkey}_end_minus_DC_LSB"] = float((v[-1] - d_0) / slope_lsb)
            rec[f"{rkey}_end_inside_band"] = bool(lo <= v[-1] <= hi)
            rec[f"{rkey}_V_dec_max_after_edge_V"] = float(v[after].max())
            rec[f"{rkey}_V_dec_min_after_edge_V"] = float(v[after].min())
            rec[f"{rkey}_band_crossings"] = band_crossings(t, v, lo, hi, m.t_edge)
            rec[f"{rkey}_tail_tau_s"] = tail_tau(t, v, d_0, (hi - lo) / 2, m.t_edge)
            if rkey == "plan":
                rec["t0_op_minus_DC_prev_V"] = float(v[0] - rec["V_dec_DC_prev_V"])
        if conv:
            a_, b_, q_ = rec["plan_t_settle_s"], rec["half_step_t_settle_s"], rec["quarter_step_t_settle_s"]
            ok = None not in (a_, b_, q_)
            rec["conv_abs_diff_s"] = abs(a_ - b_) if ok else None
            rec["conv_rel_diff"] = (abs(a_ - b_) / max(a_, b_) if ok and max(a_, b_) > 0 else (0.0 if ok else None))
            rec["conv2_abs_diff_s"] = abs(b_ - q_) if ok else None
            rec["conv2_rel_diff"] = (abs(b_ - q_) / max(b_, q_) if ok and max(b_, q_) > 0 else (0.0 if ok else None))
            # trapezoidal rule: error ~ h^2 -> Richardson estimate from the half and quarter steps
            rec["richardson_t_settle_s"] = (4 * q_ - b_) / 3 if ok else None
            g_ = rec["gear_t_settle_s"]
            rec["gear_minus_plan_s"] = (g_ - a_) if None not in (a_, g_) else None
        ts = rec["plan_t_settle_s"]
        msg = (f"{c['id']} {c['group']} {c['design']} {c['btype']} {c['node']:<13} theta={c['theta_key']:<2} "
               f"x_prev={c['x_prev']:.0f}: t_settle = {'unsettled' if ts is None else f'{ts * 1e9:.4f} ns'}")
        if conv and rec["richardson_t_settle_s"] is not None:
            msg += (f" (half {rec['half_step_t_settle_s'] * 1e9:.4f}, quarter {rec['quarter_step_t_settle_s'] * 1e9:.4f}, "
                    f"extrapolated {rec['richardson_t_settle_s'] * 1e9:.4f}, gear {rec['gear_t_settle_s'] * 1e9:.4f} ns)")
        log(msg + f"; band {lo:.6f}-{hi:.6f} V; end-DC {rec['plan_end_minus_DC_LSB']:.1e} LSB")
        results.append(rec)
    return {"cases": results, "dc_plan_vs_tight_max_abs_V": dc_diff}


# ----------------------------------------------------------------------------- item 2
def item2(sp: Spice, m: Model, sw: dict, dd: dict, a1y: float, log: Log) -> dict:
    cases = []
    for design in ("P1", "P3"):
        for btype in ("L6T", "UINV"):
            for node in ("t0", "dac_remap_1y", "supply_1y"):
                for xk in ("0.25", "0.75", "8/15"):
                    cases.append(dict(design=design, btype=btype, node=node, xkey=xk))
    out = {}
    for opts, okey in ((OPTS_PLAN, "plan"), (OPTS_TIGHT, "tight")):
        lines, tags = [], []
        for j, c in enumerate(cases):
            ns = node_setting(m, sw, dd[c["design"]], c["btype"], c["node"], "8/15", a1y)
            vdl = dd[c["design"]]["maps"][ns["map"]][c["xkey"]]["V_DL"]
            tag = f"p{j:02d}"
            ls, n, dec = branch_lines(m, tag, c["btype"], m.designs[c["design"]][1], ns["G_S"], ns["V_SL"], fmt(vdl))
            lines += ls
            tags.append((tag, n, dec, ns, vdl))
        o = sp.run(f"power_dc_{okey}", "\n".join(lines), "op\nwrite @RAW(o)@ all", options=opts)["o"]
        res = []
        for c, (tag, n, dec, ns, vdl) in zip(cases, tags):
            i_sl = float(-o[f"i(vsl_{tag})"][0])
            i_inv = float(-o[f"i(vinv_{tag})"][0]) if c["btype"] == "UINV" else 0.0
            i_ml = float(-o[f"i(vml_{tag})"][0])
            i_dl = float(-o[f"i(vdl_{tag})"][0])
            v_n = float(o[f"v({n})"][0])
            p_sl = ns["V_SL"] * i_sl
            p_inv = m.vdd * i_inv
            r = dict(c)
            r.update({"theta": 8 / 15, "x": dd[c["design"]]["maps"][ns["map"]][c["xkey"]]["x"], "map": ns["map"],
                      "G_uS": ns["G_S"] * 1e6, "V_SL_V": ns["V_SL"], "V_DL_V": vdl, "V_div_node_V": v_n,
                      "V_dec_V": float(o[f"v({dec})"][0]), "I_SL_A": i_sl,
                      "I_SL_from_PCM_A": ns["G_S"] * (ns["V_SL"] - v_n), "I_inv_A": i_inv,
                      "I_pulldown_A_not_counted": i_ml, "I_DL_A": i_dl,
                      "P_SL_uW": p_sl * 1e6, "P_inv_uW": p_inv * 1e6, "P_uW": (p_sl + p_inv) * 1e6})
            res.append(r)
        out[okey] = res
    diffs = [abs(a["P_uW"] - b["P_uW"]) / abs(b["P_uW"]) for a, b in zip(out["plan"], out["tight"])]
    pcm = [abs(r["I_SL_A"] - r["I_SL_from_PCM_A"]) / r["I_SL_A"] for r in out["plan"]]
    for r, d in zip(out["plan"], diffs):
        r["rel_diff_plan_vs_tight_options"] = d
    log(f"power: max rel. diff plan vs tight options = {max(diffs):.2e}; "
        f"max rel. |I(SL source) - G (V_SL - V_node)| / I = {max(pcm):.2e}")
    for r in out["plan"]:
        if r["xkey"] != "8/15":
            log(f"  {r['design']} {r['btype']:<4} {r['node']:<13} x={r['xkey']:<4}: P = {r['P_uW']:.7f} uW "
                f"(SL {r['P_SL_uW']:.7f}, inv {r['P_inv_uW']:.7f}); V_div {r['V_div_node_V']:.6f} V, "
                f"V_dec {r['V_dec_V']:.6f} V")
    return {"cases": out["plan"], "tight_options_cases": out["tight"],
            "max_rel_diff_plan_vs_tight": max(diffs), "max_rel_sl_vs_pcm_current": max(pcm)}


# ----------------------------------------------------------------------------- item 3
def item3(log: Log) -> dict:
    spec = importlib.util.spec_from_file_location("e6ind_trees", TREES)
    trees = importlib.util.module_from_spec(spec)
    sys.modules["e6ind_trees"] = trees
    spec.loader.exec_module(trees)
    rows_out = []
    for split in (0, 1, 2):
        b = trees.dataset_bundle("iris", split)
        rows = b["rows"]
        n_rows = int(rows.row_tree.shape[0])
        n_feat = int(b["x_te"].shape[1])
        cells = n_rows * n_feat
        kinds = np.asarray(rows.cons_kind)
        # own 4-bit quantization, only to report how many bounds sit at level 0 or 15 (information)
        lev = np.round(np.clip(np.asarray(rows.cons_thr, dtype=float), 0.0, 1.0) * 15).astype(int)
        r = {"dataset": "iris", "split": split, "rows": n_rows, "features": n_feat, "cells": cells,
             "pcm_devices": 2 * cells, "replica_share_256": 256 / (2 * cells),
             "replica_share_64": 64 / (2 * cells),
             "programmed_bounds": int(len(kinds)), "lower_bounds": int(np.sum(kinds == 0)),
             "upper_bounds": int(np.sum(kinds == 1)), "n_test": int(b["x_te"].shape[0]),
             "n_trees": int(rows.n_trees), "rows_without_bounds": int(np.sum(np.asarray(rows.row_ncons) == 0)),
             "lower_at_level_0": int(np.sum((kinds == 0) & (lev == 0))),
             "lower_at_level_15": int(np.sum((kinds == 0) & (lev == 15))),
             "upper_at_level_0": int(np.sum((kinds == 1) & (lev == 0))),
             "upper_at_level_15": int(np.sum((kinds == 1) & (lev == 15)))}
        rows_out.append(r)
        log(f"iris split {split}: rows {n_rows}, features {n_feat}, cells {cells}, PCM {2 * cells}, "
            f"256/(2 cells) = {256 / (2 * cells):.6f}, programmed bounds {len(kinds)} "
            f"(lower {r['lower_bounds']}, upper {r['upper_bounds']})")
    mean = {k: float(np.mean([r[k] for r in rows_out])) for k in
            ("rows", "features", "cells", "pcm_devices", "replica_share_256", "replica_share_64",
             "programmed_bounds", "lower_bounds", "upper_bounds")}
    mean["replica_share_256_of_mean_devices"] = 256 / mean["pcm_devices"]
    return {"splits": rows_out, "mean_over_splits": mean}


# ----------------------------------------------------------------------------- outputs
def write_csv(path: Path, rows: list[dict], cols: list[str]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            w.writerow(["" if r.get(c) is None else (repr(r[c]) if isinstance(r[c], float) else r[c]) for c in cols])


def g6(v, nd=6):
    return "unsettled" if v is None else f"{v:.{nd}g}"


def write_report(path: Path, v: dict) -> None:
    sw = v["item0_switching_points"]
    L = []
    L.append("# E6 independent check, part 1: own values (before any comparison)")
    L.append("")
    L.append(f"- Run: {v['meta']['started_jst']} to {v['meta']['finished_jst']} (JST); wall {v['meta']['wall_s']:.0f} s; "
             f"ngspice: {v['meta']['ngspice_runs']} runs, {v['meta']['ngspice_cpu_s']:.0f} s CPU, one process at a time")
    L.append(f"- Code: `experiments/E6_independent/run_check.py` (SHA-256 `{v['meta']['script_sha256']}`)")
    L.append(f"- Inputs: plan `docs/E6_protocol.md`, config `experiments/E6_config.json` (SHA-256 `{v['meta']['config_sha256']}`), "
             f"models `{v['meta']['lib']}`, `src/pcm_acam/trees.py` (dataset_bundle only). No main-run output was read.")
    L.append(f"- {v['meta']['ngspice_version']}; .spiceinit `set ngbehavior=hsa`, `set ng_nomodcheck`; .temp 27; "
             f"environment OMP_NUM_THREADS=1, OMP_WAIT_POLICY=passive (speed only; results bit-identical)")
    L.append(f"- Drift factor a(1 y) = (31,536,000 s / 20 s)^-0.049 = {v['drift']['a_1y']:.9f}")
    L.append("")
    L.append("## Item 0: nominal switching points")
    L.append("")
    L.append("| Quantity | Value (V) | How found | Check |")
    L.append("|---|---|---|---|")
    L.append(f"| V_sw | {sw['V_sw']['value_V']:.9f} | 1 mV, then 1 µV DC sweep of the pull-down gate (V_DS = 1.8 V), linear interpolation of I_D = 280 nA | .op at root: I/280 nA - 1 = {sw['V_sw']['rel_residual']:.1e} |")
    L.append(f"| V_Y* | {sw['V_Y_star']['value_V']:.9f} | INV_S output held at V_sw; 1 mV, then 1 µV sweep of the input; zero of the output-source current | free-output .op at root: V_out - V_sw = {sw['V_Y_star']['free_output_minus_V_sw_V']:.1e} V; gain {sw['V_Y_star']['inverter_gain_dVout_dVin']:.2f} |")
    L.append(f"| V_X* | {sw['V_X_star']['value_V']:.9f} | two INV_S in series (pull-down on the output); 1 mV, then 1 µV sweep; V_out = V_sw | first INV_S held at V_Y*: {sw['V_X_star']['cross_check_first_inverter_output_held_at_V_Y_star_V']:.9f} V (diff {sw['V_X_star']['cross_check_difference_V']:.1e} V); cascade gain {sw['V_X_star']['cascade_gain_dVout_dVin']:.0f} |")
    L.append("")
    L.append("Resolution: the sweeps fix every node voltage of the device under test (except the free-output checks), so each "
             "point is an exact model evaluation; linear interpolation on the 1 µV grid; the stated residuals are the check.")
    L.append("")
    L.append("## Item 1: settling times (transient SPICE, C_par = 0.5 fF)")
    L.append("")
    s1 = v["item1_settling"]
    has_conv = any("quarter_step_t_settle_s" in r for r in s1["cases"])

    def ns_(x):
        return "unsettled" if x is None else f"{x * 1e9:.4f}"

    L.append("Settling times (ns). `plan` = the plan's numerics (max step 20 ps), the value to compare with the main run; "
             "`half`, `quarter` = max step 10 ps, 5 ps; `extrap.` = (4·quarter − half)/3 (trapezoidal error ∝ h²).")
    L.append("")
    L.append("| Case | Group | Design | Type | Node | θ (level) | x_prev | t_settle plan | half | quarter | extrap. | tail τ (ns) |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in s1["cases"]:
        tau = r.get("plan_tail_tau_s")
        L.append(f"| {r['id']} | {r['group']} | {r['design']} | {r['btype']} | {r['node']} | {r['theta_key']} ({r['level']}) | "
                 f"{r['x_prev']:.0f} | **{ns_(r['plan_t_settle_s'])}** | "
                 + (f"{ns_(r['half_step_t_settle_s'])} | {ns_(r['quarter_step_t_settle_s'])} | "
                    f"{ns_(r['richardson_t_settle_s'])} | " if has_conv else " |  |  | ")
                 + f"{'' if tau is None else f'{tau * 1e9:.4f}'} |")
    L.append("")
    L.append("Settings, data-line voltages, band and DC decision-node voltage of each case:")
    L.append("")
    L.append("| Case | G (µS) | V_SL (V) | Map | V_DL(x_prev) (V) | V_DL(θ) (V) | V_DL(θ−0.01 LSB) (V) | V_DL(θ+0.01 LSB) (V) | Band low (V) | Band high (V) | Band (mV) | V_dec,DC at M(θ) (V) | (V_dec,DC(θ) − V_sw)/slope (LSB) | V_dec,DC at M(x_prev) (V) |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in s1["cases"]:
        L.append(f"| {r['id']} | {r['G_uS']:.6f} | {r['V_SL_V']:.6f} | {r['map']} | {r['V_DL_prev_V']:.6f} | {r['V_DL_theta_V']:.6f} | "
                 f"{r['V_DL_theta_minus_V']:.6f} | {r['V_DL_theta_plus_V']:.6f} | {r['band_lo_V']:.6f} | {r['band_hi_V']:.6f} | "
                 f"{r['band_width_mV']:.4f} | {r['V_dec_DC_theta_V']:.6f} | {r['DC_theta_minus_V_sw_LSB']:.2e} | "
                 f"{r['V_dec_DC_prev_V']:.6f} |")
    L.append("")
    L.append(f"Numerics: `{OPTS_PLAN}` (the plan's), default integration (trapezoidal), `tran 10p t_stop 0 20p`, t_stop 150 ns (L6T) "
             f"and 600 ns (UINV); DL PWL M(x_prev) until 1 ns, linear to M(θ) by 1.1 ns; transient starts from the DC operating "
             f"point at M(x_prev). Band = DC decision-node voltages at M(θ ± 0.01 LSB), same branch and node setting (.op, plan "
             f"options; tight options `{OPTS_TIGHT}` change any DC point by at most {s1['dc_plan_vs_tight_max_abs_V']:.1e} V). "
             f"t_settle = time of the final entry into the band after 1 ns (linear interpolation between the last outside sample "
             f"and the next), minus 1 ns; 0 if never outside after 1 ns.")
    L.append("")
    if has_conv:
        cv = [r for r in s1["cases"] if r.get("conv_abs_diff_s") is not None]
        by = {}
        for g in ("1a", "1b", "1c"):
            rr = [r for r in cv if r["group"] == g]
            by[g] = (max(r["conv_abs_diff_s"] for r in rr), max(r["conv_rel_diff"] for r in rr),
                     max(r["conv2_abs_diff_s"] for r in rr), max(r["conv2_rel_diff"] for r in rr),
                     max(abs(r["gear_minus_plan_s"]) for r in rr),
                     max(abs(r["gear_minus_plan_s"]) / r["plan_t_settle_s"] if r["plan_t_settle_s"] else 0.0 for r in rr))
        L.append("Convergence (largest change per group):")
        L.append("")
        L.append("| Group | plan → half step | half → quarter step | Gear (plan step) vs plan |")
        L.append("|---|---|---|---|")
        for g, (a1, r1, a2, r2, ag, rg) in by.items():
            L.append(f"| {g} | {a1 * 1e12:.2f} ps ({r1 * 100:.3f} %) | {a2 * 1e12:.2f} ps ({r2 * 100:.3f} %) | "
                     f"{ag * 1e12:.2f} ps ({rg * 100:.3f} %) |")
        L.append("")
    worst_end = max(abs(r["plan_end_minus_DC_LSB"]) for r in s1["cases"])
    L.append(f"All cases inside the band at t_stop: {all(r['plan_end_inside_band'] for r in s1['cases'])}; "
             f"largest |V(t_stop) − V_DC(θ)| = {worst_end:.1e} LSB (input-referred); band crossings after 1 ns per case: "
             f"{sorted(set(r['plan_band_crossings'] for r in s1['cases']))} (0 = never outside, 1 = one final entry, no re-exit); "
             f"the transient's initial operating point equals the DC point at M(x_prev) within "
             f"{max(abs(r['t0_op_minus_DC_prev_V']) for r in s1['cases']):.1e} V.")
    L.append("")
    L.append("## Item 2: static power of one programmed bound (DC SPICE)")
    L.append("")
    L.append("θ = 8/15 (level 8; the plan text says θ = 0.5, but levels are j/15). P = V_SL·I(SL source) + V_DD·I(inverter supply) "
             "(UINV only); pull-down (ML) current not counted; DL source draws no DC current.")
    L.append("")
    L.append("| Design | Type | Node | x | G (µS) | V_SL (V) | V_DL (V) | V_div (V) | V_dec (V) | P_SL (µW) | P_inv (µW) | P (µW) |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in v["item2_power"]["cases"]:
        if r["xkey"] == "8/15":
            continue
        L.append(f"| {r['design']} | {r['btype']} | {r['node']} | {r['xkey']} | {r['G_uS']:.6f} | {r['V_SL_V']:.6f} | "
                 f"{r['V_DL_V']:.6f} | {r['V_div_node_V']:.6f} | {r['V_dec_V']:.6f} | {r['P_SL_uW']:.7f} | "
                 f"{r['P_inv_uW']:.7f} | {r['P_uW']:.7f} |")
    L.append("")
    p2 = v["item2_power"]
    L.append(f"Numerics: .op with `{OPTS_PLAN}`; rerun with `{OPTS_TIGHT}`: largest relative change of P "
             f"{p2['max_rel_diff_plan_vs_tight']:.1e}; I(SL source) vs G·(V_SL - V_div): largest relative difference "
             f"{p2['max_rel_sl_vs_pcm_current']:.1e}. The x = θ points are in values.json and power.csv (not part of the check).")
    L.append("")
    L.append("## Item 3: counts (E4 forest, iris)")
    L.append("")
    L.append("| Split | Rows | Features | Cells | PCM devices | 256 / (2 cells) | Programmed bounds (lower / upper) |")
    L.append("|---|---|---|---|---|---|---|")
    for r in v["item3_counts"]["splits"]:
        L.append(f"| {r['split']} | {r['rows']} | {r['features']} | {r['cells']} | {r['pcm_devices']} | "
                 f"{r['replica_share_256']:.6f} ({r['replica_share_256'] * 100:.3f} %) | {r['programmed_bounds']} "
                 f"({r['lower_bounds']} / {r['upper_bounds']}) |")
    mn = v["item3_counts"]["mean_over_splits"]
    L.append(f"| mean | {mn['rows']:.4f} | {mn['features']:.0f} | {mn['cells']:.4f} | {mn['pcm_devices']:.4f} | "
             f"{mn['replica_share_256']:.6f} (mean of shares; 256 / mean devices = {mn['replica_share_256_of_mean_devices']:.6f}) | "
             f"{mn['programmed_bounds']:.4f} |")
    L.append("")
    sp0 = v["item3_counts"]["splits"]
    L.append("Programmed bounds = entries of rows.cons_kind (one per programmed bound). For information (own 4-bit rounding of "
             "cons_thr): bounds at level 0 / 15 per split, lower: "
             + ", ".join(f"{r['lower_at_level_0']}/{r['lower_at_level_15']}" for r in sp0)
             + "; upper: " + ", ".join(f"{r['upper_at_level_0']}/{r['upper_at_level_15']}" for r in sp0)
             + "; rows without any bound: " + ", ".join(str(r["rows_without_bounds"]) for r in sp0) + ".")
    L.append("")
    L.append("## Assumptions and interpretations")
    L.append("")
    for note in v["meta"]["notes"]:
        L.append(f"- {note}")
    L.append("")
    if v["meta"]["ngspice_warnings"]:
        L.append("## ngspice warnings")
        L.append("")
        for wl in v["meta"]["ngspice_warnings"][:20]:
            L.append(f"- {wl}")
        L.append("")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def sha256(p: Path) -> str:
    import hashlib
    return hashlib.sha256(p.read_bytes()).hexdigest()


# ----------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", help="output folder (default: results/E6_independent_check_<YYYYMMDD_HHMM> in JST, "
                                  "_b, _c ... if taken); must not exist")
    ap.add_argument("--tmp", default=None, help="base directory for the ngspice temporary folder (default: system temp)")
    ap.add_argument("--no-conv", action="store_true", help="skip the halved-step transient reruns")
    ap.add_argument("--keep-tmp", action="store_true", help="keep the ngspice temporary folder")
    args = ap.parse_args()

    t_start = now_jst()
    if args.out:
        out = Path(args.out).resolve()
        if out.exists():
            print(f"refusing to write into existing {out}", file=sys.stderr)
            return 2
        out.mkdir(parents=True)
    else:
        out = new_out_dir(RESULTS)
    (out / "netlists").mkdir()
    log = Log(out / "run.log")
    log(f"E6 independent check part 1; output {out}")
    cfg = json.loads(CONFIG.read_text())
    m = Model(cfg)
    lib = (ROOT / cfg["models"]["lib"]).resolve()
    sp = Spice(lib, log, out / "netlists", args.tmp)
    log(f"ngspice temporary folder {sp.tmp}; {sp.version()}; model {lib.relative_to(ROOT)}")
    a1y = m.a(m.t_1y)
    log(f"a(1 y) = {a1y:.12f}; I_sw target = {m.i_sw:.6e} A; LSB = 1/{m.levels - 1}; band = +-{m.tol_lsb} LSB")

    values = {"meta": {}, "drift": {"a_1y": a1y, "formula": "(t / 20 s)^-0.049, t = 31,536,000 s"}}
    try:
        sw = switching_points(sp, m, log)
        values["item0_switching_points"] = sw
        vsw, vy = sw["V_sw"]["value_V"], sw["V_Y_star"]["value_V"]
        dlt = m.tol_lsb * m.lsb
        xs = {"0-d": -dlt, "0": 0.0, "0+d": dlt, "0.25": 0.25, "8/15": 8 / 15, "0.75": 0.75,
              "1-d": 1 - dlt, "1": 1.0, "1+d": 1 + dlt}
        dd = {d: divider_maps(sp, m, d, vsw, vy, a1y, xs, log) for d in ("P1", "P3")}
        values["maps_and_programming"] = dd
        values["item1_settling"] = item1(sp, m, sw, dd, a1y, log, conv=not args.no_conv)
        values["item2_power"] = item2(sp, m, sw, dd, a1y, log)
        values["item3_counts"] = item3(log)
    finally:
        if not args.keep_tmp:
            sp.cleanup()
    t_end = now_jst()
    values["meta"] = {
        "started_jst": t_start.strftime("%Y-%m-%d %H:%M:%S"), "finished_jst": t_end.strftime("%Y-%m-%d %H:%M:%S"),
        "wall_s": (t_end - t_start).total_seconds(), "ngspice_runs": sp.n, "ngspice_wall_s": sp.wall,
        "ngspice_cpu_s": sp.cpu, "ngspice_version": sp.version(), "ngspice_warnings": sp.warnings,
        "script_sha256": sha256(Path(__file__).resolve()), "config_sha256": sha256(CONFIG),
        "lib": str(lib.relative_to(ROOT)), "spice_options_plan": OPTS_PLAN, "spice_options_tight_check": OPTS_TIGHT,
        "env": {"OMP_NUM_THREADS": "1", "OMP_WAIT_POLICY": "passive"},
        "sweep_steps_V": {"coarse": COARSE_STEP, "fine": FINE_STEP},
        "python": sys.version.split()[0], "numpy": np.__version__,
        "notes": ["Item 2 uses theta = 8/15 (level 8): the plan text says theta = 0.5, but the run only has levels j/15.",
                  "'Last exit from the band' is read as the end of the last excursion outside the band, i.e. the final "
                  "entry into it: crossing between the last outside sample after 1 ns and the next sample.",
                  "MOSFET ad, as, pd, ps, nrd, nrs, sa, sb, sd left at the subcircuit defaults (0), nf = 1, mult = 1; "
                  "mismatch parameters at their default 0.",
                  "L6T programming uses G(theta) exactly; I_div(M0(theta), V_sw)/D equals it up to the map residual "
                  "(below 1e-11 relative, see dl_maps.csv). UINV: G_UINV(theta) = I_div(M0(theta), V_DS = V_Y*)/D from "
                  "an .op; no clipping (all values are far below 25 uS).",
                  "a(t) = (t / 20 s)^-0.049 with t = 31,536,000 s (as specified for E6; E4 used ((t + 20 s)/20 s), "
                  "which changes a by 3e-8 relative).",
                  "C_par = 0.5 fF from the divider node to ground in the transients only (no effect on DC).",
                  "Each branch has its own sources: V_SL, the DL, the pull-down drain (1.8 V, precharged ML) and, for "
                  "UINV, the INV_S supply (1.8 V, pMOS source and body). Divider and pull-down sources and bodies at 0.",
                  "DC sweeps that define V_sw, V_Y*, V_X* and the DL map fix every node of the device under test, so each "
                  "point is an exact model evaluation; roots by linear interpolation on a 1 uV grid, checked by an .op "
                  "at the root."],
    }
    (out / "values.json").write_text(json.dumps(jsonable(values), indent=1, ensure_ascii=False) + "\n")
    s_cols = ["id", "group", "design", "btype", "node", "theta_key", "level", "x_prev", "G_uS", "V_SL_V", "map", "a",
              "V_DL_prev_V", "V_DL_theta_V", "V_DL_theta_minus_V", "V_DL_theta_plus_V", "band_lo_V", "band_hi_V",
              "band_width_mV", "V_dec_DC_theta_V", "V_div_DC_theta_V", "V_dec_DC_prev_V", "slope_V_per_LSB",
              "DC_theta_minus_V_sw_LSB",
              "plan_t_settle_s", "plan_status", "half_step_t_settle_s", "half_step_status", "conv_abs_diff_s",
              "conv_rel_diff", "quarter_step_t_settle_s", "conv2_abs_diff_s", "conv2_rel_diff",
              "richardson_t_settle_s", "gear_t_settle_s", "gear_minus_plan_s",
              "plan_n_points", "plan_max_dt_s", "plan_V_dec_end_V", "plan_end_minus_DC_LSB",
              "plan_end_inside_band", "plan_band_crossings", "plan_tail_tau_s", "plan_V_dec_min_after_edge_V",
              "plan_V_dec_max_after_edge_V", "t0_op_minus_DC_prev_V", "t_stop_s"]
    write_csv(out / "settling.csv", values["item1_settling"]["cases"], s_cols)
    p_cols = ["design", "btype", "node", "xkey", "x", "theta", "map", "G_uS", "V_SL_V", "V_DL_V", "V_div_node_V",
              "V_dec_V", "I_SL_A", "I_SL_from_PCM_A", "I_inv_A", "I_pulldown_A_not_counted", "I_DL_A", "P_SL_uW",
              "P_inv_uW", "P_uW", "rel_diff_plan_vs_tight_options"]
    write_csv(out / "power.csv", values["item2_power"]["cases"], p_cols)
    c_cols = ["dataset", "split", "rows", "features", "cells", "pcm_devices", "replica_share_256", "replica_share_64",
              "programmed_bounds", "lower_bounds", "upper_bounds", "n_test", "n_trees", "rows_without_bounds",
              "lower_at_level_0", "lower_at_level_15", "upper_at_level_0", "upper_at_level_15"]
    write_csv(out / "counts.csv", values["item3_counts"]["splits"] +
              [dict(dataset="iris", split="mean", **values["item3_counts"]["mean_over_splits"])], c_cols)
    mrows = []
    for d, e in dd.items():
        for mk, mp in e["maps"].items():
            for xk, r in mp.items():
                mrows.append(dict(design=d, map=mk, xkey=xk, **r))
    write_csv(out / "dl_maps.csv", mrows, ["design", "map", "xkey", "x", "V_DL", "I_target_A", "I_at_root_A",
                                           "rel_residual", "voltage_residual_V", "gm_S"])
    write_report(out / "report_part1.md", values)
    log(f"done: {sp.n} ngspice runs, {sp.cpu:.1f} s ngspice CPU, wall {(t_end - t_start).total_seconds():.0f} s")
    log.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
