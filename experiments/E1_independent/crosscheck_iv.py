#!/usr/bin/env python3
"""Diagnostic only (not used for the reported numbers): consistency of the full-branch results
with single-transistor operating points.

At a bound B of a branch with conductance G, supply V_SL and switching voltage V_sw, the divider
(drain at V_sw, gate at B) must carry I = G (V_SL - V_sw).  For every programmed case and node we
simulate the divider alone (.op, drain at V_sw, gate at B - d, B, B + d) and solve for the gate
voltage that gives the required current (quadratic interpolation of ln I).  The difference to B
measures the consistency of the branch sweeps.  Also checks I_D(V_sw) = I_sw for the pull-downs.

Usage: python3 crosscheck_iv.py <output_dir>
"""
import csv
import json
import os
import sys

import numpy as np

import e1lib as L

OUT = sys.argv[1]
A = (31536000.0 / 20.0) ** (-0.049)
LDIV = {"P1": 0.15, "P4": 8.0}
rng = np.random.default_rng([20261014, 0])
z_div = rng.standard_normal((300, 4))
z_pd = rng.standard_normal((300, 4))

nom = list(csv.DictReader(open(os.path.join(OUT, "nominal.csv"))))
VSW_NOM = float(nom[0]["vsw_nom_V"]); VSL_NOM = float(nom[0]["vsl_nom_V"]); VSL_1Y = float(nom[0]["vsl_1y_V"])
TARGET = {(r["design"], int(r["j"])): float(r["M0_V"]) for r in nom}
rows = list(csv.DictReader(open(os.path.join(OUT, "independent.csv"))))


def op_currents(devs, tag, chunk=300):
    """devs: list of (model, L, W, mm, vd, vg) -> drain current delivered by the drain source."""
    out = np.empty(len(devs))
    for s in range(0, len(devs), chunk):
        part = devs[s:s + chunk]
        lines = L.header("op cross-check")
        for k, (model, Ld, Wd, mm, vd, vg) in enumerate(part):
            lines += ["VD%d d%d 0 %s" % (k, k, L.fmt(vd)), "VG%d g%d 0 %s" % (k, k, L.fmt(vg)),
                      "XM%d d%d g%d 0 0 %s w=%s l=%s %s" % (k, k, k, model, Wd, L.fmt(Ld), L.mmstr(mm))]
        lines.append(".save " + " ".join("i(vd%d)" % k for k in range(len(part))))
        lines.append(".op")
        res = L.run_spice(lines, "%s_%d" % (tag, s // chunk))
        for k in range(len(part)):
            out[s + k] = -L._branch_current(res, "vd%d" % k)[0]
    return out


rep = {}
# ---- 1. pull-down: I_D at the reported V_sw
vsw_inst = {}
for r in rows:
    vsw_inst[int(r["instance"])] = float(r["vsw_V"])
pd = [("sky130_fd_pr__nfet_01v8_lvt", 0.15, 0.42, np.zeros(4), 1.8, VSW_NOM)]
pd += [("sky130_fd_pr__nfet_01v8_lvt", 0.15, 0.42, z_pd[i], 1.8, vsw_inst[i]) for i in range(40)]
ipd = op_currents(pd, "xc_pd")
rep["pulldown_max_rel_dev_I_at_Vsw"] = float(np.max(np.abs(ipd / L.I_SW - 1)))

# ---- 2./3. divider operating points
D_DELTA = 0.05e-3
checks = []     # (label, design, i, required current, bound)
for r in rows:
    d, i, j, node = r["design"], int(r["instance"]), int(r["j"]), r["node"]
    vsw = float(r["vsw_V"]); gp = float(r["g_prog_uS"]) * 1e-6
    gc = min(gp, 25e-6)
    b = float(r["bound_V"])
    if node == "ideal":
        checks.append(("prog_unclipped_root", d, i, gp * (VSL_NOM - vsw), TARGET[(d, j)], vsw))
        checks.append(("ideal", d, i, gc * (VSL_NOM - vsw), b, vsw))
    elif node == "none":
        checks.append(("none", d, i, A * gc * (VSL_NOM - vsw), b, vsw))
    elif node == "supply":
        checks.append(("supply", d, i, A * gc * (VSL_1Y - vsw), b, vsw))
devs = []
for lab, d, i, ireq, b, vsw in checks:
    for off in (-D_DELTA, 0.0, D_DELTA):
        devs.append(("sky130_fd_pr__nfet_01v8", LDIV[d], 0.42, z_div[i], vsw, b + off))
cur = op_currents(devs, "xc_div").reshape(-1, 3)
dv = np.empty(len(checks))
for k, (lab, d, i, ireq, b, vsw) in enumerate(checks):
    y = np.log(cur[k]); x = np.array([-D_DELTA, 0.0, D_DELTA])
    c2, c1, c0 = np.polyfit(x, y, 2)
    roots = np.roots([c2, c1, c0 - np.log(ireq)])
    roots = roots[np.isreal(roots)].real
    dv[k] = roots[np.argmin(np.abs(roots))]          # gate(I = I_req) - bound
labels = np.array([c[0] for c in checks])
for lab in ("prog_unclipped_root", "ideal", "none", "supply"):
    m = labels == lab
    rep["max_abs_gate_iv_minus_branch_bound_uV_" + lab] = float(np.max(np.abs(dv[m])) * 1e6)

# ---- 4. nominal map points at the thresholds
devs = []; meta = []
for r in nom:
    d = r["design"]; g = float(r["g_theta_uS"]) * 1e-6
    for nm, gg in (("M0", g), ("Ma", A * g)):
        b = float(r[nm + "_V"])
        meta.append((gg * (VSL_NOM - VSW_NOM)))
        for off in (-D_DELTA, 0.0, D_DELTA):
            devs.append(("sky130_fd_pr__nfet_01v8", LDIV[d], 0.42, np.zeros(4), VSW_NOM, b + off))
cur = op_currents(devs, "xc_nom").reshape(-1, 3)
dvn = []
for k, ireq in enumerate(meta):
    y = np.log(cur[k]); x = np.array([-D_DELTA, 0.0, D_DELTA])
    c2, c1, c0 = np.polyfit(x, y, 2)
    roots = np.roots([c2, c1, c0 - np.log(ireq)])
    roots = roots[np.isreal(roots)].real
    dvn.append(roots[np.argmin(np.abs(roots))])
rep["max_abs_gate_iv_minus_branch_bound_uV_nominal_M0_Ma"] = float(np.max(np.abs(dvn)) * 1e6)
print(json.dumps(rep, indent=1))
with open(os.path.join(OUT, "crosscheck_iv.json"), "w") as f:
    json.dump(rep, f, indent=1)
