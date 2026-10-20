#!/usr/bin/env python3
"""E3b independent check (docs/E3b_protocol.md section 8).

Task 1  Re-derive the first-order predictor for U1 and U4: V_sw,pd,nom, V_Y*,nom, inverter gain,
        the 12 mismatch sensitivities of V_Y* (+-1 sigma central differences), sigma_delta_Y and
        the 1-year predicted rms (pdk|dac_remap, pdk|supply; the other four keys for information),
        and compare with the frozen predictions.json.
Task 2  Full upper-branch DC simulations (PCM resistor SL_hi -> Y, divider Y -> 0 with gate DL,
        inverter Y -> Z; bound = V_DL at which Z = V_sw,pd,k) for the first 20 pdk instances,
        theta in {1, 5, 10, 14}/15, 1 year, nodes dac_remap and supply; compare with the main
        run's arrays (errors, V_Y*,k, nominal bounds).

Written only from docs/E3b_protocol.md (sections 2, 3, 4, 8), experiments/E3b_config.json,
docs/E2_protocol.md section 2 and the SPICE model files.  Nothing under src/ and no
experiments/*.py file was read or imported.  Method and choices: README.md in this folder.

Usage:  python experiments/E3b_independent/e3b_independent_check.py
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]                      # experiments/E3b_independent -> repository root
CONFIG_PATH = ROOT / "experiments" / "E3b_config.json"
PROTOCOL_PATH = ROOT / "docs" / "E3b_protocol.md"
WORK_BASE = ROOT / "logs" / "E3b_independent_work"
RESULTS_BASE = ROOT / "results"
MAIN_PRED_PATH = RESULTS_BASE / "E3b_predict_20261014_0532" / "predictions.json"
MAIN_MC_DIR = RESULTS_BASE / "E3b_20261014_0532"

PARAMS = ("vth0", "toxe", "voff", "nfactor")   # mismatch columns, in the order of the config
JST = timezone(timedelta(hours=9))
SHA_QUOTED_IN_PROTOCOL = {                      # header and section 10 of docs/E3b_protocol.md
    "experiments/E3b_config.json": "233dc92b009c62aea36163bbc821d7694b37d53cdefce4009b393e78641a13c2",
    "results/E3b_predict_20261014_0532/predictions.json":
        "1b6014495d8e385012a4cfd4c1d215cbe1a8866d5e24e69bd12a0107a18337ea",
}

# Root finding (all crossings): stage 1 sweeps 0..VDD in H1 steps with every circuit on one swept
# node; stage 2 gives every circuit its own offset source (input = sweep + stage-1 estimate) and
# sweeps +-W2 in H2 steps; the crossing is the linear interpolation between the two stage-2 points.
H1 = 1e-3
W2 = 1.5e-3
H2 = 1e-6
DVDS = 5e-3          # +-V_DS step for lambda (E2 protocol section 2.3)
N_THETA = 16         # theta_j = j/15, j = 0..15 (E2 protocol section 2.1)

TH_RMS_REL, TH_RMS_ABS = 0.01, 0.002     # Task 1 rms: relative 1 % or absolute 0.002 LSB
TH_VY_NOM = 0.05e-3                       # Task 1 V_Y*,nom (V)
TH_SIGMA_REL = 0.01                       # Task 1 sigma_delta_Y relative
TH_ERR = 0.01                             # Task 2 error (LSB)
TH_V = 0.05e-3                            # Task 2 V_Y*,k and M0 (V)


def fmt(x: float) -> str:
    x = float(x)
    if not math.isfinite(x):
        raise ValueError(f"non-finite netlist value {x}")
    return repr(x)


def rel(path: Path) -> str:
    try:
        return str(Path(path).relative_to(ROOT))
    except ValueError:
        return str(path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------------------------
# ngspice
# --------------------------------------------------------------------------------------------
class Spice:
    """ngspice in batch mode, strictly one job at a time, in a scratch directory."""

    BAD = ("error", "fail", "singular", "abort", "too small", "not converge", "no convergence")

    def __init__(self, workdir: Path, lib: Path):
        self.dir = workdir
        self.dir.mkdir(parents=True, exist_ok=False)
        (self.dir / ".spiceinit").write_text("set ngbehavior=hsa\nset ng_nomodcheck\n")
        self.lib = lib
        self.n = 0
        self.records: list[dict] = []

    def dc(self, tag, lines, src, start, stop, step, vectors, chunk=100):
        self.n += 1
        name = f"{self.n:03d}_{tag}"
        ctrl = [".control", "option numdgt=16", "set wr_singlescale", "set wr_vecnames",
                f"dc {src} {fmt(start)} {fmt(stop)} {fmt(step)}"]
        files = []
        for j in range(0, len(vectors), chunk):
            fn = f"{name}_{j // chunk:02d}.txt"
            files.append(fn)
            ctrl.append(f"wrdata {fn} " + " ".join(vectors[j:j + chunk]))
        ctrl.append(".endc")
        net = [f"* E3b independent check: {name}", f'.include "{self.lib}"', ".temp 27",
               *lines, *ctrl, ".end"]
        (self.dir / f"{name}.sp").write_text("\n".join(net) + "\n")
        t0 = time.time()
        p = subprocess.run(["ngspice", "-b", f"{name}.sp"], cwd=self.dir,
                           capture_output=True, text=True)
        dt = time.time() - t0
        log = p.stdout + p.stderr
        (self.dir / f"{name}.log").write_text(log)
        bad = [ln for ln in log.splitlines() if any(k in ln.lower() for k in self.BAD)]
        if p.returncode != 0 or bad:
            raise RuntimeError(f"ngspice problem in {name}: rc={p.returncode} {bad[:5]}")
        scale, cols = None, []
        for fn in files:
            d = np.loadtxt(self.dir / fn, skiprows=1, ndmin=2)
            if scale is None:
                scale = d[:, 0]
            elif not np.array_equal(scale, d[:, 0]):
                raise RuntimeError(f"{name}: scale differs between output files")
            cols.append(d[:, 1:])
        data = np.hstack(cols)
        n_exp = int(round((stop - start) / step)) + 1
        if data.shape[1] != len(vectors) or abs(data.shape[0] - n_exp) > 1:
            raise RuntimeError(f"{name}: data {data.shape}, expected ({n_exp}, {len(vectors)})")
        self.records.append({"run": name, "seconds": round(dt, 2), "points": int(data.shape[0]),
                             "vectors": len(vectors)})
        print(f"  ngspice {name}: {data.shape[0]} points x {len(vectors)} vectors, {dt:.1f} s",
              flush=True)
        return scale, data


def mos(name, d, g, s, b, model, w, l, z=None):
    """sky130 subcircuit instance; W, L in um (.option scale=1.0u in the library)."""
    z = (0.0,) * 4 if z is None else tuple(float(v) for v in z)
    mm = " ".join(f"mm_{p}={fmt(v)}" for p, v in zip(PARAMS, z))
    return f"X{name} {d} {g} {s} {b} sky130_fd_pr__{model} w={fmt(w)} l={fmt(l)} {mm}"


class Circ:
    """Circuit with a swept input node; output vector 'vec' is compared with 'target'."""

    def __init__(self, build, vec, target, log=False, aux=None):
        self.build, self.vec, self.target, self.log, self.aux = build, vec, float(target), log, aux

    def f(self, col):
        if self.log:
            return np.log(np.maximum(np.abs(col), 1e-300)) - math.log(abs(self.target))
        return col - self.target


def _crossings(x, f):
    neg = f < 0.0
    idx = np.nonzero(neg[:-1] != neg[1:])[0]
    return [x[i] - f[i] * (x[i + 1] - x[i]) / (f[i + 1] - f[i]) for i in idx], idx


def solve(sp: Spice, tag, circs, common, lo, hi):
    """Input value of every circuit at which its output equals its target (two-stage DC sweep)."""
    lines = list(common) + ["VSW nsw 0 0"]
    for c in circs:
        lines += c.build("nsw")
    x, d = sp.dc(tag + "_s1", lines, "VSW", lo, hi, H1, [c.vec for c in circs])
    est = []
    for j, c in enumerate(circs):
        r, _ = _crossings(x, c.f(d[:, j]))
        if len(r) != 1:
            raise RuntimeError(f"{tag}: circuit {j} has {len(r)} crossings in stage 1")
        est.append(r[0])
    lines = list(common) + ["VSW nsw 0 0"]
    vecs = []
    for j, c in enumerate(circs):
        lines.append(f"VOF{j} nin{j} nsw {fmt(est[j])}")
        lines += c.build(f"nin{j}")
        vecs.append(c.vec)
    aux_col = {}
    for j, c in enumerate(circs):
        if c.aux:
            aux_col[j] = len(vecs)
            vecs.append(c.aux)
    s, d = sp.dc(tag + "_s2", lines, "VSW", -W2, W2, H2, vecs)
    out = []
    for j, c in enumerate(circs):
        f = c.f(d[:, j])
        r, idx = _crossings(s, f)
        if len(r) != 1:
            raise RuntimeError(f"{tag}: circuit {j} has {len(r)} crossings in stage 2")
        i = int(idx[0])
        sl = slice(max(i - 10, 0), min(i + 12, len(s)))
        slope = float(np.polyfit(s[sl] - r[0], f[sl], 2)[1])      # d(f)/d(input) at the root
        aux = None
        if j in aux_col:
            ya = d[:, aux_col[j]]
            aux = float(ya[i] + (r[0] - s[i]) * (ya[i + 1] - ya[i]) / (s[i + 1] - s[i]))
        out.append({"root": est[j] + r[0], "slope": slope, "stage1": est[j], "aux": aux,
                    "margin": float(min(r[0] - s[0], s[-1] - r[0]))})
    return out


def eval_ids(sp: Spice, tag, devs):
    """|I_D| at fixed bias (source = body = 0). devs: (model, w, l, V_G, V_D, z)."""
    lines = ["VDUM ndum 0 0", "RDUM ndum 0 1k"]
    vecs = []
    for j, (model, w, l, vg, vd, z) in enumerate(devs):
        lines += [f"VEG{j} eg{j} 0 {fmt(vg)}", f"VED{j} ed{j} 0 {fmt(vd)}",
                  mos(f"E{j}", f"ed{j}", f"eg{j}", "0", "0", model, w, l, z)]
        vecs.append(f"i(ved{j})")
    _, d = sp.dc(tag, lines, "VDUM", 0.0, 0.0, 1.0, vecs)
    if d.shape[0] != 1:
        raise RuntimeError(f"{tag}: expected one operating point")
    return np.abs(d[0])


# circuit builders (j = index inside one netlist)
def c_pulldown(j, pd, z, i_sw, vdd):
    def build(nin):
        return [f"VPD{j} pdd{j} 0 {fmt(vdd)}",
                mos(f"PD{j}", f"pdd{j}", nin, "0", "0", pd["device"], pd["w_um"], pd["l_um"], z)]
    return Circ(build, f"i(vpd{j})", i_sw, log=True)


def c_inverter(j, inv, zn, zp, target):
    def build(nin):
        return [mos(f"IP{j}", f"z{j}", nin, "vdd", "vdd", inv["p_dev"], inv["wp"], inv["lp"], zp),
                mos(f"IN{j}", f"z{j}", nin, "0", "0", inv["n_dev"], inv["wn"], inv["ln"], zn)]
    return Circ(build, f"v(z{j})", target)


def c_divider(j, model, w, l, vds, z, i_target):
    def build(nin):
        return [f"VDV{j} dv{j} 0 {fmt(vds)}", mos(f"DV{j}", f"dv{j}", nin, "0", "0", model, w, l, z)]
    return Circ(build, f"i(vdv{j})", i_target, log=True)


def c_branch(j, model, w, l, inv, vsl, g, zd, zn, zp, target):
    """Full upper branch; input node = DL; output Z compared with V_sw,pd; Y recorded."""
    def build(nin):
        return [f"VSL{j} sl{j} 0 {fmt(vsl)}",
                f"RP{j} sl{j} y{j} {fmt(1.0 / g)}",
                mos(f"DV{j}", f"y{j}", nin, "0", "0", model, w, l, zd),
                mos(f"IP{j}", f"z{j}", f"y{j}", "vdd", "vdd", inv["p_dev"], inv["wp"], inv["lp"], zp),
                mos(f"IN{j}", f"z{j}", f"y{j}", "0", "0", inv["n_dev"], inv["wn"], inv["ln"], zn)]
    return Circ(build, f"v(z{j})", target, aux=f"v(y{j})")


def unit(p, s):
    z = [0.0] * 4
    z[PARAMS.index(p)] = float(s)
    return z


def sname(p, s):
    return f"{p}{'+' if s > 0 else '-'}"


def e_r1sq(sigma, d_v, a, n=4001, span=6.0, drop=0.95):
    """E[(r-1)^2], r = (D - a delta)/(D - delta), delta ~ N(0, sigma^2); trapezoid on +-6 sigma,
    4001 points, points with delta >= 0.95 D dropped and the density renormalised."""
    d = np.linspace(-span * sigma, span * sigma, n)
    d = d[d < drop * d_v]
    w = np.exp(-0.5 * (d / sigma) ** 2)
    r = (d_v - a * d) / (d_v - d)
    return float(np.trapezoid(w * (r - 1.0) ** 2, d) / np.trapezoid(w, d))


# --------------------------------------------------------------------------------------------
# Task 1: predictor
# --------------------------------------------------------------------------------------------
def task1(sp, cfg, conds, a):
    circ = cfg["circuit"]
    vdd, pd = circ["vdd"], circ["pulldown"]
    i_sw = 100e-9 * pd["w_um"] / pd["l_um"]
    var = [("nom", None)] + [(sname(p, s), unit(p, s)) for p in PARAMS for s in (1, -1)]
    res = solve(sp, "t1_pulldown", [c_pulldown(j, pd, z, i_sw, vdd) for j, (_, z) in enumerate(var)],
                [], 0.0, vdd)
    vsw = {n: r["root"] for (n, _), r in zip(var, res)}
    shared = {"i_sw_A": i_sw, "vsw_pd": vsw,
              "dvsw_pd": {p: (vsw[sname(p, 1)] - vsw[sname(p, -1)]) / 2 for p in PARAMS}}

    theta = np.arange(N_THETA) / (N_THETA - 1)
    g_lo, g_hi = circ["g_lo_us"] * 1e-6, circ["g_hi_us"] * 1e-6
    dg = g_hi - g_lo
    gt = g_lo + dg * theta
    dm = circ["divider_device"]
    per = {}
    for cond in conds:
        cid, inv = cond["id"], circ["inverters"][cond["inverter"]]
        # V_Y*: nominal inverter at V_sw,pd,nom; perturbed inverter nMOS / pMOS at V_sw,pd,nom;
        # nominal inverter at the perturbed pull-down's V_sw,pd
        keys, circs = [("nom",)], [c_inverter(0, inv, None, None, vsw["nom"])]
        for dev in ("invn", "invp"):
            for p in PARAMS:
                for s in (1, -1):
                    z = unit(p, s)
                    circs.append(c_inverter(len(circs), inv, z if dev == "invn" else None,
                                            z if dev == "invp" else None, vsw["nom"]))
                    keys.append((dev, p, s))
        for p in PARAMS:
            for s in (1, -1):
                circs.append(c_inverter(len(circs), inv, None, None, vsw[sname(p, s)]))
                keys.append(("pd", p, s))
        res = solve(sp, f"t1_{cid}_inverter", circs, [f"VDD vdd 0 {fmt(vdd)}"], 0.0, vdd)
        vy = {k: r["root"] for k, r in zip(keys, res)}
        vy_nom, gain = vy[("nom",)], -res[0]["slope"]
        sens = {f"dvy_d{dev}_{p}": (vy[(dev, p, 1)] - vy[(dev, p, -1)]) / 2
                for dev in ("invn", "invp", "pd") for p in PARAMS}
        sigma = math.sqrt(sum(v * v for v in sens.values()))

        # divider s_p and lambda at V_DS = V_Y*,nom, at the nominal gate voltages of I and aI
        d_v = cond["d_v"]
        cur = np.concatenate([gt * d_v, a * gt * d_v])
        w, l = cond["w_div_um"], cond["l_div_um"]
        res = solve(sp, f"t1_{cid}_divider_vg",
                    [c_divider(j, dm, w, l, vy_nom, None, i) for j, i in enumerate(cur)], [], 0.0, vdd)
        vg = np.array([r["root"] for r in res])
        cols = ([("nom", None, vy_nom)] + [(sname(p, s), unit(p, s), vy_nom) for p in PARAMS for s in (1, -1)]
                + [("vds+", None, vy_nom + DVDS), ("vds-", None, vy_nom - DVDS)])
        ids = eval_ids(sp, f"t1_{cid}_divider_eval",
                       [(dm, w, l, g, vd, z) for g in vg for (_, z, vd) in cols]).reshape(len(vg), len(cols))
        ln_i = np.log(ids)
        ci = {n: k for k, (n, _, _) in enumerate(cols)}
        s_p = {p: -(ln_i[:, ci[sname(p, 1)]] - ln_i[:, ci[sname(p, -1)]]) / 2 for p in PARAMS}
        lam = -(ln_i[:, ci["vds+"]] - ln_i[:, ci["vds-"]]) / (2 * DVDS)
        nom_resid = float(np.max(np.abs(ln_i[:, ci["nom"]] - np.log(cur))))

        n = N_THETA
        k2 = (gt / dg) ** 2
        div_term = sum((s_p[p][n:] - s_p[p][:n]) ** 2 for p in PARAMS)
        lam_term = (lam[n:] - lam[:n]) ** 2 * sigma ** 2
        er = e_r1sq(sigma, d_v, a)

        def rms(v):
            return 15.0 * math.sqrt(float(np.mean(v)))

        pred = {"pdk|dac_remap": rms(k2 * (div_term + lam_term)), "pdk|supply": rms(k2 * er),
                "div_only|dac_remap": rms(k2 * div_term), "div_only|supply": 0.0,
                "y_only|dac_remap": rms(k2 * lam_term), "y_only|supply": rms(k2 * er)}
        per[cid] = {"vy_nom": vy_nom, "v_sl_nom": vy_nom + d_v, "gain": gain, "sens": sens,
                    "sigma_delta": sigma, "E_r1sq": er, "pred_1y": pred, "vg": vg, "currents": cur,
                    "s_p": s_p, "lam": lam, "nominal_lnI_residual": nom_resid,
                    "vy_raw": {"|".join(map(str, k)): v for k, v in vy.items()}}
        print(f"  {cid}: V_Y*,nom = {vy_nom:.9f} V, gain {gain:.4f}, sigma_delta_Y = {sigma * 1e3:.4f} mV, "
              f"pdk|dac_remap {pred['pdk|dac_remap']:.5f}, pdk|supply {pred['pdk|supply']:.5f}", flush=True)
    return shared, per


# --------------------------------------------------------------------------------------------
# Task 2: full-branch Monte Carlo spot check
# --------------------------------------------------------------------------------------------
def task2(sp, cfg, conds, t1s, t1p, a, n_inst, theta_idx):
    circ = cfg["circuit"]
    vdd, pd, mm = circ["vdd"], circ["pulldown"], cfg["mismatch"]
    rng = np.random.default_rng([mm["base_seed"], mm["rng_stream"]])
    n = mm["n_instances"]
    z_div = rng.standard_normal((n, 4))
    z_pd = rng.standard_normal((n, 4))
    z_invn = rng.standard_normal((n, 4))
    z_invp = rng.standard_normal((n, 4))
    ks = list(range(n_inst))
    res = solve(sp, "t2_pulldown", [c_pulldown(j, pd, z_pd[k], t1s["i_sw_A"], vdd) for j, k in enumerate(ks)],
                [], 0.0, vdd)
    vsw_k = np.array([r["root"] for r in res])

    theta = np.arange(N_THETA) / (N_THETA - 1)
    g_lo, g_hi = circ["g_lo_us"] * 1e-6, circ["g_hi_us"] * 1e-6
    dg = g_hi - g_lo
    gt = g_lo + dg * theta
    dm = circ["divider_device"]
    vdd_line = [f"VDD vdd 0 {fmt(vdd)}"]
    out = {"vsw_pd_k": vsw_k, "conds": {}}
    for cond in conds:
        cid, inv = cond["id"], circ["inverters"][cond["inverter"]]
        w, l, d_v = cond["w_div_um"], cond["l_div_um"], cond["d_v"]
        vy_nom = t1p[cid]["vy_nom"]
        vsl_nom, vsl_sup = vy_nom + d_v, vy_nom + d_v / a
        vsw_nom = t1s["vsw_pd"]["nom"]
        # V_Y*,k of the instance inverters
        res = solve(sp, f"t2_{cid}_inverter",
                    [c_inverter(j, inv, z_invn[k], z_invp[k], vsw_k[j]) for j, k in enumerate(ks)],
                    vdd_line, 0.0, vdd)
        vy_k = np.array([r["root"] for r in res])
        # nominal bounds M0(theta) (all 16 theta) and the bound at 30 uS: nominal full branch
        gl = list(gt) + [30e-6]
        res = solve(sp, f"t2_{cid}_nominal_branch",
                    [c_branch(j, dm, w, l, inv, vsl_nom, g, None, None, None, vsw_nom) for j, g in enumerate(gl)],
                    vdd_line, 0.0, vdd)
        m0 = np.array([r["root"] for r in res[:N_THETA]])
        b30 = res[N_THETA]["root"]
        y_nom_branch = np.array([r["aux"] for r in res])
        # threshold-verify programming: G_k = I_div,k(M0(theta), V_DS = V_Y*,k) / (V_SL,nom - V_Y*,k)
        i_prog = eval_ids(sp, f"t2_{cid}_program",
                          [(dm, w, l, m0[t], vy_k[j], z_div[k]) for j, k in enumerate(ks) for t in theta_idx])
        g_k = i_prog.reshape(n_inst, len(theta_idx)) / (vsl_nom - vy_k[:, None])
        # full branch: t0 (self-check: bound must equal M0), dac_remap and supply after drift a
        bounds, y_at = {}, {}
        for node, vsl, gf in (("t0", vsl_nom, 1.0), ("dac_remap", vsl_nom, a), ("supply", vsl_sup, a)):
            circs = []
            for j, k in enumerate(ks):
                for q in range(len(theta_idx)):
                    circs.append(c_branch(len(circs), dm, w, l, inv, vsl, gf * g_k[j, q],
                                          z_div[k], z_invn[k], z_invp[k], vsw_k[j]))
            res = solve(sp, f"t2_{cid}_branch_{node}", circs, vdd_line, 0.0, vdd)
            bounds[node] = np.array([r["root"] for r in res]).reshape(n_inst, len(theta_idx))
            y_at[node] = np.array([r["aux"] for r in res]).reshape(n_inst, len(theta_idx))
        # exact inverse of the nominal map: I_0(b) = nominal divider current, V_DS = V_Y*,nom
        devs = [(dm, w, l, b, vy_nom, None) for node in ("dac_remap", "supply") for b in bounds[node].ravel()]
        i0 = eval_ids(sp, f"t2_{cid}_map_inverse", devs).reshape(2, n_inst, len(theta_idx))
        th = theta[list(theta_idx)]
        x_dac = (i0[0] / (a * d_v) - g_lo) / dg           # G(x_eff) a D = I_0(b)   (map Ma)
        x_sup = (i0[1] / d_v - g_lo) / dg                 # G(x_eff) D = I_0(b)     (map M0)
        out["conds"][cid] = {
            "vy_k": vy_k, "m0": m0, "b30": b30, "y_nom_branch": y_nom_branch, "g_k": g_k,
            "bounds": bounds, "y_at": y_at, "i0": i0, "vsl_nom": vsl_nom, "vsl_supply": vsl_sup,
            "err": {"dac_remap": 15.0 * (x_dac - th), "supply": 15.0 * (x_sup - th)}}
        print(f"  {cid}: Task 2 done", flush=True)
    return out


# --------------------------------------------------------------------------------------------
# comparison and output
# --------------------------------------------------------------------------------------------
def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)


def compare(cfg, conds, t1s, t1p, t2, main_pred, main_arr, theta_idx, n_inst, outdir):
    # ---------------- Task 1 ----------------
    rows1 = []

    def add(cid, q, mine, ref, kind):
        ad = mine - ref
        rd = ad / ref if ref != 0 else float("nan")
        if kind == "vy":
            ok, th = abs(ad) <= TH_VY_NOM, "|diff| <= 0.05 mV"
        elif kind == "sigma":
            ok, th = abs(rd) <= TH_SIGMA_REL, "|rel| <= 1 %"
        elif kind == "rms":
            ok, th = (abs(rd) <= TH_RMS_REL) or (abs(ad) <= TH_RMS_ABS), "|rel| <= 1 % or |diff| <= 0.002 LSB"
        else:
            ok, th = None, "info"
        rows1.append({"condition": cid, "quantity": q, "independent": repr(float(mine)), "main": repr(float(ref)),
                      "abs_diff": repr(float(ad)), "rel_diff": repr(float(rd)), "threshold": th,
                      "pass": "" if ok is None else ("PASS" if ok else "FAIL")})
        return ok

    t1_ok = True
    for cond in conds:
        cid = cond["id"]
        mp, mine = main_pred[cid], t1p[cid]
        ys = mp["y_sensitivity"]
        add(cid, "vsw_pd_nom_V", t1s["vsw_pd"]["nom"], ys["vsw_pd_nom"], "info")
        t1_ok &= add(cid, "vy_star_nom_V", mine["vy_nom"], mp["vy_star_nom_V"], "vy")
        add(cid, "v_sl_nom_V", mine["v_sl_nom"], mp["v_sl_nom_V"], "info")
        add(cid, "inverter_gain", mine["gain"], ys["inverter_gain"], "info")
        for dev in ("invn", "invp", "pd"):
            for p in PARAMS:
                key = f"dvy_d{dev}_{p}"
                add(cid, key + "_V", mine["sens"][key], ys[key], "info")
        t1_ok &= add(cid, "sigma_delta_V", mine["sigma_delta"], ys["sigma_delta"], "sigma")
        for key, val in mine["pred_1y"].items():
            req = key in ("pdk|dac_remap", "pdk|supply")
            ok = add(cid, f"pred_1y {key}_LSB", val, mp["predictions"]["1y"][key], "rms" if req else "info")
            if req:
                t1_ok &= ok
        add(cid, "bound_at_30uS_V (full nominal branch)", t2["conds"][cid]["b30"], mp["bound_at_30uS_V"], "info")
    write_csv(outdir / "task1_compare.csv", rows1)

    rows_s = []
    for cond in conds:
        cid = cond["id"]
        m = t1p[cid]
        for j in range(len(m["vg"])):
            r = {"condition": cid, "theta_index": j % N_THETA, "current": "I" if j < N_THETA else "aI",
                 "I_A": repr(float(m["currents"][j])), "V_G_nominal_V": repr(float(m["vg"][j]))}
            for p in PARAMS:
                r[f"s_{p}"] = repr(float(m["s_p"][p][j]))
            r["lambda_per_V"] = repr(float(m["lam"][j]))
            rows_s.append(r)
    write_csv(outdir / "task1_divider_sensitivities.csv", rows_s)

    # ---------------- Task 2 ----------------
    rows_i, rows_m, rows_p = [], [], []
    worst = {"err": (0.0, None), "vy": (0.0, None), "m0_req": (0.0, None), "m0_all": (0.0, None),
             "vsw": (0.0, None), "t0": (0.0, None), "y_at": (0.0, None)}

    def upd(key, val, where):
        if abs(val) > abs(worst[key][0]):
            worst[key] = (float(val), where)

    t2_ok = True
    for cond in conds:
        cid = cond["id"]
        c2, arr = t2["conds"][cid], main_arr[cid]
        if not np.allclose(arr["theta"], np.arange(N_THETA) / (N_THETA - 1), rtol=0, atol=1e-12):
            raise RuntimeError("theta grid of the main arrays differs from j/15")
        for j in range(n_inst):
            dvy = c2["vy_k"][j] - arr["vy_mc"][j]
            dsw = t2["vsw_pd_k"][j] - arr["vsw_pd_mc"][j]
            ok = abs(dvy) <= TH_V
            t2_ok &= ok
            upd("vy", dvy, f"{cid} k={j}")
            upd("vsw", dsw, f"{cid} k={j}")
            rows_i.append({"condition": cid, "instance": j,
                           "vsw_pd_independent_V": repr(float(t2["vsw_pd_k"][j])), "vsw_pd_main_V": repr(float(arr["vsw_pd_mc"][j])),
                           "vsw_pd_diff_V": repr(float(dsw)),
                           "vy_independent_V": repr(float(c2["vy_k"][j])), "vy_main_V": repr(float(arr["vy_mc"][j])),
                           "vy_diff_V": repr(float(dvy)), "pass_vy_0.05mV": "PASS" if ok else "FAIL"})
        for t in range(N_THETA):
            dm0 = c2["m0"][t] - arr["b_target"][t]
            req = t in theta_idx
            ok = abs(dm0) <= TH_V
            if req:
                t2_ok &= ok
                upd("m0_req", dm0, f"{cid} theta={t}/15")
            upd("m0_all", dm0, f"{cid} theta={t}/15")
            rows_m.append({"condition": cid, "theta_index": t, "theta": repr(t / 15),
                           "M0_independent_V": repr(float(c2["m0"][t])), "b_target_main_V": repr(float(arr["b_target"][t])),
                           "diff_V": repr(float(dm0)), "Y_at_nominal_bound_V": repr(float(c2["y_nom_branch"][t])),
                           "required": "yes" if req else "no",
                           "pass_0.05mV": "PASS" if ok else "FAIL"})
        for j in range(n_inst):
            for q, t in enumerate(theta_idx):
                upd("t0", c2["bounds"]["t0"][j, q] - c2["m0"][t], f"{cid} k={j} theta={t}/15")
                for node in ("dac_remap", "supply"):
                    upd("y_at", c2["y_at"][node][j, q] - c2["vy_k"][j], f"{cid} k={j} theta={t}/15 {node}")
                    e_m = float(arr[f"err_pdk_{node}_1y"][j, t])
                    e_i = float(c2["err"][node][j, q])
                    ok = abs(e_i - e_m) <= TH_ERR
                    t2_ok &= ok
                    upd("err", e_i - e_m, f"{cid} k={j} theta={t}/15 {node}")
                    rows_p.append({"condition": cid, "instance": j, "theta_index": t, "node": node,
                                   "G_k_uS": repr(float(c2["g_k"][j, q] * 1e6)),
                                   "bound_independent_V": repr(float(c2["bounds"][node][j, q])),
                                   "Y_at_bound_V": repr(float(c2["y_at"][node][j, q])),
                                   "err_independent_LSB": repr(e_i), "err_main_LSB": repr(e_m),
                                   "diff_LSB": repr(e_i - e_m), "pass_0.01LSB": "PASS" if ok else "FAIL"})
    write_csv(outdir / "task2_instances.csv", rows_i)
    write_csv(outdir / "task2_nominal_bounds.csv", rows_m)
    write_csv(outdir / "task2_pairs.csv", rows_p)

    # rms over the checked pairs (both runs), per condition and node
    rms_rows = {}
    for cond in conds:
        cid = cond["id"]
        for node in ("dac_remap", "supply"):
            e_i = t2["conds"][cid]["err"][node]
            e_m = main_arr[cid][f"err_pdk_{node}_1y"][:n_inst][:, list(theta_idx)].astype(float)
            rms_rows[f"{cid}|{node}"] = {"independent": float(np.sqrt(np.mean(e_i ** 2))),
                                         "main": float(np.sqrt(np.mean(e_m ** 2)))}
    return t1_ok, t2_ok, worst, rows1, rms_rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--conditions", default="U1,U4")
    ap.add_argument("--n-inst", type=int, default=20)
    ap.add_argument("--test-out", default=None,
                    help="program test only: write outputs to this new folder instead of results/")
    args = ap.parse_args()

    stamp = datetime.now(JST).strftime("%Y%m%d_%H%M")
    if args.test_out:
        outdir = Path(args.test_out).resolve()
        workdir = outdir / "work"
    else:
        outdir = RESULTS_BASE / f"E3b_independent_check_{stamp}"
        workdir = WORK_BASE / f"run_{stamp}"
    outdir.mkdir(parents=False, exist_ok=False)          # never overwrite
    t_start = time.time()

    cfg = json.loads(CONFIG_PATH.read_text())
    lib = ROOT / cfg["models"]["subset_dir"] / cfg["models"]["lib"]
    main_pred = json.loads(MAIN_PRED_PATH.read_text())
    want = args.conditions.split(",")
    conds = [c for c in cfg["conditions"] if c["id"] in want]
    if [c["id"] for c in conds] != want:
        raise SystemExit(f"unknown condition in {want}")
    main_arr = {c["id"]: dict(np.load(MAIN_MC_DIR / f"arrays_{c['id']}.npz")) for c in conds}

    dr = cfg["drift"]
    a = (dr["times_s"]["1y"] / dr["t0_s"]) ** (-dr["nu"])
    theta_idx = (1, 5, 10, 14)

    inputs = {str(p.relative_to(ROOT)): sha256(p) for p in
              [CONFIG_PATH, PROTOCOL_PATH, MAIN_PRED_PATH]
              + [MAIN_MC_DIR / f"arrays_{c['id']}.npz" for c in conds]
              + sorted((ROOT / cfg["models"]["subset_dir"]).glob("*.spice"))}
    sha_ok = {k: inputs.get(k) == v for k, v in SHA_QUOTED_IN_PROTOCOL.items()}
    ngv = subprocess.run(["ngspice", "--version"], capture_output=True, text=True).stdout.strip().splitlines()

    print(f"E3b independent check -> {rel(outdir)}  (a_1y = {a:.6f})", flush=True)
    sp = Spice(workdir, lib)
    print("Task 1", flush=True)
    t1s, t1p = task1(sp, cfg, conds, a)
    print("Task 2", flush=True)
    t2 = task2(sp, cfg, conds, t1s, t1p, a, args.n_inst, theta_idx)
    t1_ok, t2_ok, worst, rows1, rms_rows = compare(cfg, conds, t1s, t1p, t2, main_pred, main_arr,
                                                   theta_idx, args.n_inst, outdir)

    def devs_t1(cid):
        m, mp = t1p[cid], main_pred[cid]
        d = {"vy_star_nom_diff_V": m["vy_nom"] - mp["vy_star_nom_V"],
             "sigma_delta_rel_diff": m["sigma_delta"] / mp["y_sensitivity"]["sigma_delta"] - 1.0,
             "inverter_gain_rel_diff (info)": m["gain"] / mp["y_sensitivity"]["inverter_gain"] - 1.0}
        for key in ("pdk|dac_remap", "pdk|supply"):
            ref = mp["predictions"]["1y"][key]
            d[f"{key}_abs_diff_LSB"] = m["pred_1y"][key] - ref
            d[f"{key}_rel_diff"] = m["pred_1y"][key] / ref - 1.0
        sens_abs = {k: m["sens"][k] - mp["y_sensitivity"][k] for k in m["sens"]}
        kmax = max(sens_abs, key=lambda k: abs(sens_abs[k]))
        d["max_abs_sensitivity_diff_V (info)"] = {"value": sens_abs[kmax], "where": kmax}
        return d

    summary = {
        "check": "E3b independent check (protocol section 8)",
        "run": outdir.name,
        "time_jst": datetime.now(JST).isoformat(timespec="seconds"),
        "elapsed_s": round(time.time() - t_start, 1),
        "conditions": [c["id"] for c in conds],
        "TASK1_PASS": bool(t1_ok),
        "TASK2_PASS": bool(t2_ok),
        "a_1y": a,
        "task1": {
            "vsw_pd_nom_V": t1s["vsw_pd"]["nom"],
            "vsw_pd_variants_V": t1s["vsw_pd"],
            "i_sw_A": t1s["i_sw_A"],
            "dvsw_pd_per_sigma_V": t1s["dvsw_pd"],
            "per_condition": {cid: {
                "vy_star_nom_V": m["vy_nom"], "v_sl_nom_V": m["v_sl_nom"], "inverter_gain": m["gain"],
                "sensitivities_V": m["sens"], "sigma_delta_V": m["sigma_delta"], "E_r_minus_1_sq": m["E_r1sq"],
                "predictions_1y_LSB": m["pred_1y"], "nominal_lnI_residual_at_VG": m["nominal_lnI_residual"],
                "main_vy_star_nom_V": main_pred[cid]["vy_star_nom_V"],
                "main_sigma_delta_V": main_pred[cid]["y_sensitivity"]["sigma_delta"],
                "main_predictions_1y_LSB": {k: main_pred[cid]["predictions"]["1y"][k] for k in m["pred_1y"]},
                "deviations": devs_t1(cid),
                "vy_star_perturbed_V": m["vy_raw"],
            } for cid, m in t1p.items()},
            "thresholds": {"rms": "rel <= 1 % or abs <= 0.002 LSB", "vy_star_nom": "0.05 mV", "sigma_delta": "rel <= 1 %"},
        },
        "task2": {
            "n_instances": args.n_inst, "theta_indices": list(theta_idx), "nodes": ["dac_remap", "supply"],
            "n_pairs_compared": 2 * args.n_inst * len(theta_idx) * len(conds),
            "max_abs_err_diff_LSB": {"value": worst["err"][0], "where": worst["err"][1]},
            "max_abs_vy_diff_V": {"value": worst["vy"][0], "where": worst["vy"][1]},
            "max_abs_M0_diff_required_theta_V": {"value": worst["m0_req"][0], "where": worst["m0_req"][1]},
            "max_abs_M0_diff_all16_theta_V": {"value": worst["m0_all"][0], "where": worst["m0_all"][1]},
            "max_abs_vsw_pd_diff_V (info)": {"value": worst["vsw"][0], "where": worst["vsw"][1]},
            "self_check_t0_bound_minus_M0_V": {"value": worst["t0"][0], "where": worst["t0"][1]},
            "self_check_Y_at_bound_minus_VYk_V": {"value": worst["y_at"][0], "where": worst["y_at"][1]},
            "rms_over_checked_pairs_LSB": rms_rows,
            "per_condition": {cid: {"v_sl_nom_V": c["vsl_nom"], "v_sl_supply_1y_V": c["vsl_supply"]}
                              for cid, c in t2["conds"].items()},
            "thresholds": {"err": "0.01 LSB", "vy_k": "0.05 mV", "M0": "0.05 mV"},
        },
        "method": {
            "root_finding": f"two-stage DC sweep: stage 1 {H1} V steps over 0..VDD, stage 2 +-{W2} V around the "
                            f"stage-1 estimate in {H2} V steps (one offset source per circuit), linear interpolation",
            "currents": "fixed-bias operating points (no I-V interpolation) for s_p, lambda, G_k and I_0(b)",
            "lambda_dvds_V": DVDS, "theta_grid": "j/15, j = 0..15", "ngspice_options": "defaults; option numdgt=16 for output",
            "draws": "default_rng([20261014, 4]); standard_normal((300, 4)) x 4: divider, pull-down, inverter nMOS, inverter pMOS",
        },
        "inputs_sha256": inputs,
        "sha256_matches_protocol": sha_ok,
        "ngspice": ngv[1] if len(ngv) > 1 else ngv,
        "workdir": rel(workdir),
        "ngspice_runs": sp.records,
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=1, default=float) + "\n")
    print(json.dumps({k: summary[k] for k in ("TASK1_PASS", "TASK2_PASS")}), flush=True)
    print(json.dumps(summary["task2"]["max_abs_err_diff_LSB"]), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
