"""E1 exploratory analysis (NOT pre-registered): first-order laws for the two compensation nodes.

Derived after the main run, checked here against its per-instance results (1 year):

  supply node, pull-down mismatch delta = V_sw,k - V_sw,nom:
      dx_supply = (G(theta) / dG) * ln[(D - a delta) / (D - delta)]       (independent of the divider's region)

  dac_remap node, divider gate-referred shift dV_k at the programmed current:
      dx_dac = dV_k * [ (gm/Id)(h(B)) - (gm/Id)(B) ] * G(theta) / dG,
      h(B) = I0^{-1}(a I0(B)) (drifted operating point),  dV_k = -ln(G_k / G(theta)) / (gm/Id)(B)
      (vanishes where gm/Id is constant, i.e. weak inversion)

Usage: python experiments/E1_analysis.py results/E1_<stamp>
Writes law_check.csv, gmid_profile.csv and law_points.npz into the run folder (new files only).
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

NODES = ("none", "dac_remap", "dac_gain", "supply", "ideal")


def main() -> int:
    run = Path(sys.argv[1])
    for name in ("law_check.csv", "gmid_profile.csv", "law_points.npz"):
        if (run / name).exists():
            sys.exit(f"{name} already exists in {run}; refusing to overwrite")
    cfg = json.loads((run / "config_used.json").read_text())
    times = list(cfg["drift"]["times_s"])
    ti = times.index("1y")
    a = (cfg["drift"]["times_s"]["1y"] / cfg["drift"]["t0_s"]) ** (-cfg["drift"]["nu"])
    g_lo, g_hi = cfg["circuit"]["g_lo_us"], cfg["circuit"]["g_hi_us"]
    dg = g_hi - g_lo
    info = {r["design"]: r for r in csv.DictReader(open(run / "design_info.csv"))}
    rows, prof, pts = [], [], {}
    for dsg in cfg["designs"]:
        d = dsg["id"]
        ar = np.load(run / f"arrays_{d}.npz")
        vg, ln_i = ar["vg_div"], ar["nom_ln_i"]
        gmid = np.gradient(ln_i, vg)                       # d ln I / dV = gm / Id (1/V)
        theta = ar["theta"]
        g_t = g_lo + theta * dg
        d_v = float(info[d]["d_v"])
        vsw_nom = float(info[d]["vsw_nom_V"])
        b_t = ar["b_target"]
        b_h = np.interp(np.log(a * g_t * 1e-6 * d_v), ln_i, vg)
        gm_t, gm_h = np.interp(b_t, vg, gmid), np.interp(b_h, vg, gmid)
        for j in range(len(theta)):
            prof.append({"design": d, "theta": round(float(theta[j]), 6), "bound_V": float(b_t[j]),
                         "gmid_at_bound_per_V": float(gm_t[j]), "bound_drifted_V": float(b_h[j]),
                         "gmid_at_drifted_per_V": float(gm_h[j]), "delta_gmid_per_V": float(gm_h[j] - gm_t[j])})
        # dac_remap law on the div_only setting
        g_k = ar["gprog_div_only"].astype(float)
        dv_k = -np.log(g_k / g_t[None, :]) / gm_t[None, :]
        pred_dac = 15.0 * dv_k * (gm_h - gm_t)[None, :] * (g_t / dg)[None, :]
        sim_dac = ar["err_div_only"][NODES.index("dac_remap"), ti].astype(float)
        ok_dac = ar["valid_div_only"][NODES.index("dac_remap"), ti] & np.isfinite(pred_dac)
        # supply law on the pd_only setting
        delta = ar["vsw_pd_only"] - vsw_nom
        r = (d_v - a * delta) / (d_v - delta)
        pred_sup = 15.0 * (g_t / dg)[None, :] * np.log(r)[:, None]
        sim_sup = ar["err_pd_only"][NODES.index("supply"), ti].astype(float)
        ok_sup = ar["valid_pd_only"][NODES.index("supply"), ti]
        for law, pred, sim, ok in (("dac_remap|div_only", pred_dac, sim_dac, ok_dac),
                                   ("supply|pd_only", pred_sup, sim_sup, ok_sup)):
            p, s = pred[ok], sim[ok]
            slope = float(np.sum(p * s) / np.sum(p * p)) if np.sum(p * p) > 0 else np.nan
            ss_res = float(np.sum((s - p) ** 2))
            ss_tot = float(np.sum((s - s.mean()) ** 2))
            rows.append({"design": d, "law": law, "n": int(ok.sum()), "rms_sim_lsb": float(np.sqrt(np.mean(s ** 2))),
                         "rms_pred_lsb": float(np.sqrt(np.mean(p ** 2))), "slope_sim_on_pred": slope,
                         "r2_identity": 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan,
                         "rms_residual_lsb": float(np.sqrt(np.mean((s - p) ** 2)))})
            pts[f"{d}|{law}|pred"] = p.astype(np.float32)
            pts[f"{d}|{law}|sim"] = s.astype(np.float32)
    for path, data in (("law_check.csv", rows), ("gmid_profile.csv", prof)):
        with open(run / path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0].keys()))
            w.writeheader()
            w.writerows(data)
    np.savez_compressed(run / "law_points.npz", **pts)
    for r_ in rows:
        print(f"{r_['design']:3} {r_['law']:20} n={r_['n']:5d} rms sim {r_['rms_sim_lsb']:.3f} pred {r_['rms_pred_lsb']:.3f} "
              f"slope {r_['slope_sim_on_pred']:.3f} R2 {r_['r2_identity']:.4f} resid {r_['rms_residual_lsb']:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
