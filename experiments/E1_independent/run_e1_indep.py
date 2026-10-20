#!/usr/bin/env python3
"""Independent re-computation of part of E1 (PCM_aCAM), from the written specification only.

Scope: designs P1 (L_div = 0.15 um) and P4 (L_div = 8 um), D = 0.2 V, mismatch setting "pdk",
instances 0..39, thresholds theta_j = j/15 for j in {1, 5, 10, 14}, drift 1 year,
compensation nodes none / dac_remap / supply / ideal.

All bounds come from full-branch ngspice DC sweeps (resistor + divider transistor, sweep V_DL,
crossing V_X = V_sw), V_sw from a DC sweep of the pull-down itself (see e1lib.py).

Usage: python3 run_e1_indep.py <output_dir>
"""
import csv
import json
import os
import subprocess
import sys
import time

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq

import e1lib as L

T_START = time.time()
OUT = sys.argv[1]
assert os.path.isdir(OUT), OUT

# ---------------------------------------------------------------- specification constants
BASE_SEED = 20261014
N_DRAW = 300
N_INST = 40
DESIGNS = [("P1", 0.15), ("P4", 8.0)]
D_V = 0.2
J_LIST = [1, 5, 10, 14]
THETA = {j: j / 15.0 for j in J_LIST}
G_LO_US, G_SPAN_US = 2.5, 20.0          # G(x) = 2.5 + 20 x uS
G_MAX_US = 25.0
T_1Y, T0, NU = 31536000.0, 20.0, 0.049
A = (T_1Y / T0) ** (-NU)                # drift factor, perfect tracking a_hat = a
X_GRID = np.round(np.arange(-0.12, 1.12 + 1e-9, 0.001), 6)   # 1241 points
TOL_B = 1e-9                             # root-finding tolerance on the bound (V)
NODES = ["none", "dac_remap", "supply", "ideal"]


def G_of_x(x):
    return (G_LO_US + G_SPAN_US * np.asarray(x, dtype=float)) * 1e-6


def log(msg):
    print("[%7.1fs] %s" % (time.time() - T_START, msg), flush=True)


# ---------------------------------------------------------------- mismatch draws
rng = np.random.default_rng([BASE_SEED, 0])
z_div = rng.standard_normal((N_DRAW, 4))
z_pd = rng.standard_normal((N_DRAW, 4))
ZERO = np.zeros(4)

# ---------------------------------------------------------------- pull-down V_sw
mm_pd = [ZERO] + [z_pd[i] for i in range(N_INST)]
vsw_all, vsw_coarse = L.vsw_batch(mm_pd, tag="vsw", return_diag=True)
VSW_NOM = float(vsw_all[0])
VSW = vsw_all[1:].copy()
VSL_NOM = VSW_NOM + D_V
VSL_1Y = VSW_NOM + D_V / A
log("V_sw,nom = %.9f V, V_SL,nom = %.9f V, a = %.9f, V_SL(1y) = %.9f V" % (VSW_NOM, VSL_NOM, A, VSL_1Y))
log("V_sw instances: mean %.6f V, sd %.6f V (n=%d); fine-coarse max %.3e V"
    % (VSW.mean(), VSW.std(ddof=1), N_INST, np.max(np.abs(vsw_all - vsw_coarse))))


def nominal_branch(Ld, G, vsl=None):
    return dict(L=Ld, mm=ZERO, G=float(G), vsl=VSL_NOM if vsl is None else vsl, vsw=VSW_NOM)


# ---------------------------------------------------------------- nominal maps M0, Ma
maps = {}
nominal_rows = []
for dname, Ld in DESIGNS:
    brs = [nominal_branch(Ld, g) for g in G_of_x(X_GRID)]
    m0 = L.bounds_batch(brs, tag="map_M0_" + dname)
    brs = [nominal_branch(Ld, A * g) for g in G_of_x(X_GRID)]
    ma = L.bounds_batch(brs, tag="map_Ma_" + dname)
    for nm, arr in (("M0", m0), ("Ma", ma)):
        if not np.all(np.isfinite(arr)):
            raise RuntimeError("%s %s has non-finite grid values" % (dname, nm))
        if np.any(np.diff(arr) <= 0):
            raise RuntimeError("%s %s not strictly increasing" % (dname, nm))
    sp0, spa = CubicSpline(X_GRID, m0), CubicSpline(X_GRID, ma)
    # direct values at the thresholds (and theta = 0, 1 for the LSB size)
    xs = [THETA[j] for j in J_LIST] + [0.0, 1.0]
    d0 = L.bounds_batch([nominal_branch(Ld, g) for g in G_of_x(xs)], tag="M0_theta_" + dname)
    da = L.bounds_batch([nominal_branch(Ld, A * g) for g in G_of_x(xs)], tag="Ma_theta_" + dname)
    maps[dname] = dict(L=Ld, m0=m0, ma=ma, sp0=sp0, spa=spa,
                       M0=dict(zip(J_LIST, d0[:4])), Ma=dict(zip(J_LIST, da[:4])),
                       lsb=(d0[5] - d0[4]) / 15.0)
    # spline vs direct (interpolation accuracy of the grid)
    e0 = np.max(np.abs(sp0(xs) - d0)); ea = np.max(np.abs(spa(xs) - da))
    maps[dname]["spline_err_V"] = float(max(e0, ea))
    log("%s: M0(theta)=%s  Ma(theta)=%s  LSB=%.4f mV  spline-vs-direct max %.2e V"
        % (dname, np.round(d0[:4], 6), np.round(da[:4], 6), maps[dname]["lsb"] * 1e3, max(e0, ea)))
    for j in J_LIST:
        nominal_rows.append(dict(design=dname, l_div_um=Ld, d_v=D_V, j=j, theta=THETA[j],
                                 vsw_nom_V=VSW_NOM, vsl_nom_V=VSL_NOM, a_1y=A, vsl_1y_V=VSL_1Y,
                                 g_theta_uS=float(G_of_x(THETA[j]) * 1e6),
                                 M0_V=float(maps[dname]["M0"][j]), Ma_V=float(maps[dname]["Ma"][j]),
                                 lsb_mV=float(maps[dname]["lsb"] * 1e3)))
np.savez(os.path.join(L.WORK, "nominal_maps.npz"), x=X_GRID,
         **{"%s_%s" % (d, k): maps[d][k] for d, _ in DESIGNS for k in ("m0", "ma")})

# ---------------------------------------------------------------- threshold-verify programming
# cases: (design, instance index or -1 for the nominal sanity instance, j)
cases = []
for dname, Ld in DESIGNS:
    for i in list(range(N_INST)) + [-1]:
        for j in J_LIST:
            cases.append((dname, i, j))
NC = len(cases)


def case_branch(c, G, vsl):
    dname, i, j = c
    mm = ZERO if i < 0 else z_div[i]
    vsw = VSW_NOM if i < 0 else VSW[i]
    return dict(L=maps[dname]["L"], mm=mm, G=float(G), vsl=vsl, vsw=vsw)


target = np.array([maps[d]["M0"][j] for d, i, j in cases])
g0 = np.array([G_of_x(THETA[j]) for d, i, j in cases])
slope = np.array([maps[d]["sp0"](THETA[j], 1) * G_of_x(THETA[j]) / (G_SPAN_US * 1e-6)
                  for d, i, j in cases])            # dB/d(ln G) of the nominal branch

# Root finding in u = ln G: first step uses the nominal slope dB/du, then secant steps,
# safeguarded by bisection inside the bracket [u_lo, u_hi] (B(u_lo) < target < B(u_hi)).
u_cur = np.log(g0)                       # point to evaluate in the coming iteration
u_last = np.full(NC, np.nan)             # previously evaluated point
B_last = np.full(NC, np.nan)             # bound at u_last
B_final = np.full(NC, np.nan)
u_lo = np.full(NC, -np.inf); u_hi = np.full(NC, np.inf)
done = np.zeros(NC, bool)
n_iter = np.zeros(NC, int)
for it in range(60):
    todo = np.nonzero(~done)[0]
    if len(todo) == 0:
        break
    brs = [case_branch(cases[k], np.exp(u_cur[k]), VSL_NOM) for k in todo]
    bb = L.bounds_batch(brs, tag="prog_it%02d" % it)
    for m, k in enumerate(todo):
        n_iter[k] += 1
        b = bb[m]
        if b < target[k]:
            u_lo[k] = max(u_lo[k], u_cur[k])
        else:                                        # includes b = +inf (no crossing below vmax)
            u_hi[k] = min(u_hi[k], u_cur[k])
        if np.isfinite(b) and abs(b - target[k]) <= TOL_B:
            B_final[k] = b
            done[k] = True
            continue
        if not np.isfinite(b):
            u_new = np.nan                           # force bisection / step back below
        elif np.isfinite(B_last[k]) and b != B_last[k]:
            u_new = u_cur[k] + (target[k] - b) * (u_cur[k] - u_last[k]) / (b - B_last[k])
        else:
            u_new = u_cur[k] + (target[k] - b) / slope[k]
        if not np.isfinite(u_new) or not (u_lo[k] < u_new < u_hi[k]):
            if np.isfinite(u_lo[k]) and np.isfinite(u_hi[k]):
                u_new = 0.5 * (u_lo[k] + u_hi[k])
            elif np.isfinite(u_hi[k]):
                u_new = u_hi[k] - 0.3
            else:
                u_new = u_lo[k] + 0.3
        if np.isfinite(b):
            u_last[k], B_last[k] = u_cur[k], b
        u_cur[k] = u_new
    log("programming iteration %d: %d of %d cases converged" % (it, done.sum(), NC))
if not np.all(done):
    raise RuntimeError("root finding did not converge for %d cases" % (~done).sum())
G_prog = np.exp(u_cur)                       # before clipping (S); u_cur = converged point
clipped = (G_prog > G_MAX_US * 1e-6) | (G_prog < 0)
G_c = np.clip(G_prog, 0.0, G_MAX_US * 1e-6)
log("programming done: iterations max %d, clipped %d of %d" % (n_iter.max(), clipped.sum(), NC))

# ---------------------------------------------------------------- node bounds
brs = []
for k, c in enumerate(cases):
    brs.append(case_branch(c, A * G_c[k], VSL_NOM))     # drifted, V_SL,nom   -> none, dac_remap
    brs.append(case_branch(c, A * G_c[k], VSL_1Y))      # drifted, V_SL(1y)   -> supply
    brs.append(case_branch(c, G_c[k], VSL_NOM))         # undrifted           -> ideal
bb = L.bounds_batch(brs, tag="nodes").reshape(NC, 3)
bound = {"none": bb[:, 0], "dac_remap": bb[:, 0], "supply": bb[:, 1], "ideal": bb[:, 2]}
if not np.all(np.isfinite(bb)):
    raise RuntimeError("non-finite node bound")
log("node bounds done; max bound %.4f V" % bb.max())


# ---------------------------------------------------------------- inverse maps: x_eff
def invert(sp, b):
    lo, hi = X_GRID[0], X_GRID[-1]
    if not (sp(lo) <= b <= sp(hi)):
        return np.nan
    return brentq(lambda x: float(sp(x)) - b, lo, hi, xtol=1e-14, rtol=1e-15, maxiter=200)


x_eff = {}
x_spl = {}
for nd in NODES:
    use_a = nd == "dac_remap"
    xs = np.array([invert(maps[d]["spa" if use_a else "sp0"], bound[nd][k])
                   for k, (d, i, j) in enumerate(cases)])
    x_spl[nd] = xs
# polish: one Newton step with a direct full-branch simulation of the nominal branch at x_eff,
# then a final direct check of the residual
polish_info = {}
for nd in NODES:
    use_a = nd == "dac_remap"
    xs = x_spl[nd].copy()
    if not np.all(np.isfinite(xs)):
        raise RuntimeError("node %s: %d bounds outside the map range x in [%g, %g]"
                           % (nd, np.sum(~np.isfinite(xs)), X_GRID[0], X_GRID[-1]))
    for step in range(2):
        brs = [nominal_branch(maps[d]["L"], (A if use_a else 1.0) * G_of_x(xs[k]))
               for k, (d, i, j) in enumerate(cases)]
        bn = L.bounds_batch(brs, tag="xeff_%s_%d" % (nd, step))
        der = np.array([maps[d]["spa" if use_a else "sp0"](xs[k], 1) for k, (d, i, j) in enumerate(cases)])
        resid = bound[nd] - bn
        if step == 0:
            polish_info[nd] = dict(max_abs_resid_spline_V=float(np.max(np.abs(resid))),
                                   max_abs_newton_step_lsb=float(np.max(np.abs(15 * resid / der))))
            xs = xs + resid / der
        else:
            polish_info[nd]["max_abs_resid_after_polish_V"] = float(np.max(np.abs(resid)))
            polish_info[nd]["max_abs_resid_after_polish_lsb"] = float(np.max(np.abs(15 * resid / der)))
    x_eff[nd] = xs
log("x_eff polish: %s" % json.dumps(polish_info))

# ---------------------------------------------------------------- outputs
rows = []
sanity = []
for k, (d, i, j) in enumerate(cases):
    for nd in NODES:
        r = dict(design=d, instance=i, j=j, theta=THETA[j], node=nd,
                 vsw_V=VSW_NOM if i < 0 else VSW[i], g_prog_uS=G_prog[k] * 1e6,
                 clipped=int(clipped[k]), bound_V=bound[nd][k], x_eff=x_eff[nd][k],
                 err_lsb=15.0 * (x_eff[nd][k] - THETA[j]))
        (sanity if i < 0 else rows).append(r)

cols = ["design", "instance", "j", "theta", "node", "vsw_V", "g_prog_uS", "clipped", "bound_V", "x_eff", "err_lsb"]
with open(os.path.join(OUT, "independent.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(cols)
    for r in rows:
        w.writerow([r["design"], r["instance"], r["j"], "%.15g" % r["theta"], r["node"],
                    "%.12f" % r["vsw_V"], "%.12g" % r["g_prog_uS"], r["clipped"],
                    "%.12f" % r["bound_V"], "%.12f" % r["x_eff"], "%.9f" % r["err_lsb"]])
ncols = ["design", "l_div_um", "d_v", "j", "theta", "vsw_nom_V", "vsl_nom_V", "a_1y", "vsl_1y_V",
         "g_theta_uS", "M0_V", "Ma_V", "lsb_mV"]
with open(os.path.join(OUT, "nominal.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(ncols)
    for r in nominal_rows:
        w.writerow([("%.12g" % r[c]) if isinstance(r[c], float) else r[c] for c in ncols])

# summary statistics (unclipped rows only)
summary = {}
for d, _ in DESIGNS:
    for nd in NODES:
        e = np.array([r["err_lsb"] for r in rows if r["design"] == d and r["node"] == nd and not r["clipped"]])
        n_cl = sum(1 for r in rows if r["design"] == d and r["node"] == nd and r["clipped"])
        summary["%s/%s" % (d, nd)] = dict(n_unclipped=int(len(e)), n_clipped=int(n_cl),
                                          rms_lsb=float(np.sqrt(np.mean(e ** 2))),
                                          mean_lsb=float(np.mean(e)), max_abs_lsb=float(np.max(np.abs(e))),
                                          n_nan=int(np.sum(~np.isfinite(e))))
        per_theta = {}
        for j in J_LIST:
            ej = np.array([r["err_lsb"] for r in rows if r["design"] == d and r["node"] == nd
                           and r["j"] == j and not r["clipped"]])
            per_theta[str(j)] = float(np.sqrt(np.mean(ej ** 2))) if len(ej) else None
        summary["%s/%s" % (d, nd)]["rms_lsb_per_j"] = per_theta
san = {"%s/%s/j%d" % (r["design"], r["node"], r["j"]): dict(err_lsb=r["err_lsb"],
       g_prog_rel_dev=float(r["g_prog_uS"] * 1e-6 / G_of_x(r["theta"]) - 1)) for r in sanity}

ver = subprocess.run(["ngspice", "-v"], capture_output=True, text=True).stdout
ver = [ln.strip("* ").strip() for ln in ver.splitlines() if "ngspice-" in ln]
diag = dict(
    ngspice_version=ver[0] if ver else "unknown",
    options=L.OPTIONS, a_1y=A, vsw_nom_V=VSW_NOM, vsl_nom_V=VSL_NOM, vsl_1y_V=VSL_1Y,
    vsw_inst_mean_V=float(VSW.mean()), vsw_inst_sd_V=float(VSW.std(ddof=1)),
    vsw_fine_minus_coarse_max_V=float(np.max(np.abs(vsw_all - vsw_coarse))),
    spline_vs_direct_max_V={d: maps[d]["spline_err_V"] for d, _ in DESIGNS},
    lsb_mV={d: maps[d]["lsb"] * 1e3 for d, _ in DESIGNS},
    prog_iterations_max=int(n_iter.max()), prog_iterations_mean=float(n_iter.mean()),
    prog_final_abs_dev_max_V=float(np.max(np.abs(B_final - target))),
    ideal_bound_minus_target_max_V_unclipped=float(np.max(np.abs((bound["ideal"] - target)[~clipped]))),
    n_clipped_cases={d: int(sum(clipped[k] for k, c in enumerate(cases) if c[0] == d and c[1] >= 0)) for d, _ in DESIGNS},
    clipped_cases=[dict(design=c[0], instance=c[1], j=c[2], g_prog_uS=float(G_prog[k] * 1e6))
                   for k, c in enumerate(cases) if clipped[k]],
    x_eff_polish=polish_info,
    nominal_instance_sanity=san,
    summary=summary,
    n_rows=len(rows),
    n_spice_runs=len(L.RUN_LOG),
    spice_seconds=float(sum(r[3] for r in L.RUN_LOG)),
    wall_seconds=float(time.time() - T_START),
)
with open(os.path.join(OUT, "diagnostics.json"), "w") as f:
    json.dump(diag, f, indent=1)
with open(os.path.join(L.WORK, "run_log.json"), "w") as f:
    json.dump(L.RUN_LOG, f)
log("rows written: %d" % len(rows))
for key, s in summary.items():
    log("%-14s n=%3d clipped=%d rms=%.6f mean=%+.6f max|e|=%.6f LSB"
        % (key, s["n_unclipped"], s["n_clipped"], s["rms_lsb"], s["mean_lsb"], s["max_abs_lsb"]))
log("sanity (nominal instance): " + json.dumps({k: round(v["err_lsb"], 7) for k, v in san.items()}))
