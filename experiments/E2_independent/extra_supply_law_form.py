"""Supplementary diagnostic (outside the protocol's pass/fail): form of the supply law.

With the protocol's definitions (G(x) = 2.5 + 20 x uS, map M0 unchanged, threshold-verify
programming), the instance bound under the supply node satisfies I_inst(B) = r I_inst(M0(theta)),
r = (D - a delta)/(D - delta), and x_eff solves G(x_eff) D = I_nom(B).  If the relative current
deficit eps of the instance does not change between I and rI, G(x_eff) = r G exactly, i.e.
dx = (G/dG)(r - 1); the frozen law uses (G/dG) ln r.  The two differ at second order in delta/D.

This script compares both forms with
  (1) the main run's Monte Carlo arrays (all 300 instances x 16 thresholds, err_pdk_supply_1y,
      delta_k from vsw_mc and vsw_nom from predictions.json), and
  (2) the independent full-branch results of this check (40 instances x 4 thresholds),
and evaluates E[(r - 1)^2] / E[ln^2 r] with the protocol's quadrature.
usage: python extra_supply_law_form.py <results dir of run_check.py>
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))).replace(os.sep, "/")  # repository root (was an absolute path in the original run)
PRED = f"{ROOT}/results/E2_predict_20261014_0315/predictions.json"
MCDIR = f"{ROOT}/results/E2_20261014_0316"


def expect(D, a, s, f):
    d = np.linspace(-6 * s, 6 * s, 4001)
    d = d[d < 0.95 * D]
    w = np.exp(-0.5 * (d / s) ** 2)
    trap = getattr(np, "trapezoid", None) or np.trapz
    return trap(w * f(d), d) / trap(w, d)


def fit(y, x):
    slope = float(np.sum(x * y) / np.sum(x * x))     # through the origin
    resid = y - x
    return dict(rms_ratio_sim_over_law=float(np.sqrt(np.mean(y ** 2)) / np.sqrt(np.mean(x ** 2))),
                slope_through_origin=slope,
                rms_residual_lsb=float(15 * np.sqrt(np.mean(resid ** 2))),
                max_abs_residual_lsb=float(15 * np.max(np.abs(resid))))


def main(outdir):
    cfg = json.load(open(f"{ROOT}/experiments/E2_config.json"))
    conds = {c["id"]: c for c in cfg["conditions"]}
    pred = json.load(open(PRED))
    a = (cfg["drift"]["times_s"]["1y"] / 20.0) ** (-0.049)
    out = dict(note=__doc__.strip().splitlines()[0], a_1y=a, conditions={})
    for cid in ("G2", "C2", "T3", "G6"):
        D = conds[cid]["d_v"]
        theta = np.arange(16) / 15.0
        kG = (2.5 + 20 * theta) / 20.0                 # G/dG
        arr = np.load(f"{MCDIR}/arrays_{cid}.npz")
        delta = arr["vsw_mc"] - pred[cid]["vsw_nom_V"]
        r = (D - a * delta) / (D - delta)
        dx_sim = arr["err_pdk_supply_1y"].astype(float) / 15.0          # (300, 16)
        dx_ln = kG[None, :] * np.log(r)[:, None]
        dx_lin = kG[None, :] * (r - 1.0)[:, None]
        res = dict(main_mc_300x16=dict(
            rms_sim_lsb=float(15 * np.sqrt(np.mean(dx_sim ** 2))),
            rms_law_ln_r_lsb=float(15 * np.sqrt(np.mean(dx_ln ** 2))),
            rms_law_r_minus_1_lsb=float(15 * np.sqrt(np.mean(dx_lin ** 2))),
            vs_ln_r=fit(dx_sim.ravel(), dx_ln.ravel()),
            vs_r_minus_1=fit(dx_sim.ravel(), dx_lin.ravel())))
        ind = np.load(os.path.join(outdir, f"independent_{cid}.npz"))
        nodes = list(ind["nodes"])
        e = ind["err"][nodes.index("supply")] / 15.0                      # (40, 4)
        tj = ind["theta_idx"]
        d_ind = ind["vsw"] - pred[cid]["vsw_nom_V"]
        r_i = (D - a * d_ind) / (D - d_ind)
        k_i = kG[tj]
        res["independent_full_branch_40x4"] = dict(
            vs_ln_r=fit(e.ravel(), (k_i[None, :] * np.log(r_i)[:, None]).ravel()),
            vs_r_minus_1=fit(e.ravel(), (k_i[None, :] * (r_i - 1)[:, None]).ravel()))
        s = pred[cid]["pulldown_sensitivity"]["sigma_delta"]
        e_ln = expect(D, a, s, lambda d: np.log((D - a * d) / (D - d)) ** 2)
        e_lin = expect(D, a, s, lambda d: ((D - a * d) / (D - d) - 1) ** 2)
        res["predictor_with_r_minus_1"] = dict(
            E_ln2_r=float(e_ln), E_r_minus_1_sq=float(e_lin),
            rms_ratio=float(np.sqrt(e_lin / e_ln)),
            pred_rms_ln_r_main=pred[cid]["predictions"]["1y"]["rms_lsb"]["pdk|supply"],
            pred_rms_r_minus_1=float(pred[cid]["predictions"]["1y"]["rms_lsb"]["pdk|supply"] * np.sqrt(e_lin / e_ln)))
        out["conditions"][cid] = res
    with open(os.path.join(outdir, "supplementary_supply_law_form.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
