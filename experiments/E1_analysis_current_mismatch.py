"""E1 exploratory analysis (NOT pre-registered): DAC-node law with the divider's current mismatch.

Both nodes reduce to the same form, the change of a log-current ratio times G(theta)/dG:

  supply:    dx = (G/dG) * ln[(D - a delta) / (D - delta)]                       (pull-down mismatch delta)
  dac_remap: dx = (G/dG) * [eps_k(a I) - eps_k(I)],  eps_k(I) = gm/Id_0 * dV_k(I)
             dV_k(I) = V_k(I) - V_0(I): gate-voltage difference of instance k at current I (divider)
             I = programmed current G_k (V_SL - V_sw), a I = drifted current

eps_k is (to first order) the instance's relative drain-current mismatch ln(I_0 / I_k) at the operating
point. A mismatch whose relative current error is the same at both operating points (beta mismatch, or a
threshold shift in weak inversion) is not exposed by the DAC remap.

The divider curves of the div_only setting are re-simulated with the frozen draws (identical netlists to
the main run). Usage: python experiments/E1_analysis_current_mismatch.py results/E1_<stamp>
Writes law_check_current_mismatch.csv and law_points_current_mismatch.npz (new files only).
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pcm_acam.branch_post import IVCurve  # noqa: E402
from pcm_acam.spice_iv import MM_KEYS, iv_sweep  # noqa: E402

NODES = ("none", "dac_remap", "dac_gain", "supply", "ideal")


def main() -> int:
    run = Path(sys.argv[1])
    out_csv, out_npz = run / "law_check_current_mismatch.csv", run / "law_points_current_mismatch.npz"
    if out_csv.exists() or out_npz.exists():
        sys.exit("outputs exist; refusing to overwrite")
    cfg = json.loads((run / "config_used.json").read_text())
    lib = ROOT / cfg["models"]["subset"]
    a = (cfg["drift"]["times_s"]["1y"] / cfg["drift"]["t0_s"]) ** (-cfg["drift"]["nu"])
    ti = list(cfg["drift"]["times_s"]).index("1y")
    g_lo, g_hi = cfg["circuit"]["g_lo_us"], cfg["circuit"]["g_hi_us"]
    dg = g_hi - g_lo
    mm = cfg["mismatch"]
    rng = np.random.default_rng([mm["base_seed"], 0])
    z_div = rng.standard_normal((mm["n_instances"], 4))
    n = mm["n_instances"]
    info = {r["design"]: r for r in csv.DictReader(open(run / "design_info.csv"))}
    sw = cfg["sweep"]
    rows, pts = [], {}
    for dsg in cfg["designs"]:
        d = dsg["id"]
        ar = np.load(run / f"arrays_{d}.npz")
        vsw_nom, d_v = float(info[d]["vsw_nom_V"]), float(info[d]["d_v"])
        vg, cur = iv_sweep(lib, cfg["circuit"]["divider"]["device"], cfg["circuit"]["divider"]["w_um"], dsg["l_div_um"],
                           [vsw_nom] * (n + 1), [None] + [{k: float(v) for k, v in zip(MM_KEYS, r)} for r in z_div],
                           sw["vg_start"], sw["vg_stop"], sw["vg_step"])
        curves = [IVCurve.from_arrays(vg, cur[k], 1e-9) for k in range(n + 1)]
        c0 = curves[0]
        if not np.allclose(np.log(np.maximum(cur[0], 1e-30)), ar["nom_ln_i"], atol=1e-9, rtol=0):
            sys.exit(f"{d}: re-simulated nominal curve differs from the main run")
        ln0 = c0.ln_i
        gmid0 = np.gradient(ln0, c0.vg)
        theta = ar["theta"]
        g_t = g_lo + theta * dg
        b_t = ar["b_target"]
        pred = np.full((n, len(theta)), np.nan)
        consistency = 0.0
        for k in range(n):
            ck = curves[k + 1]
            i_prog = ck.i_at(b_t)                                   # programmed current (A)
            g_k = i_prog / d_v * 1e6
            consistency = max(consistency, float(np.nanmax(np.abs(g_k / ar["gprog_div_only"][k] - 1))))
            i_drift = a * i_prog
            v0_p, v0_d = c0.v_at(i_prog), c0.v_at(i_drift)
            eps_p = np.interp(v0_p, c0.vg, gmid0) * (ck.v_at(i_prog) - v0_p)
            eps_d = np.interp(v0_d, c0.vg, gmid0) * (ck.v_at(i_drift) - v0_d)
            pred[k] = 15.0 * (g_t / dg) * (eps_d - eps_p)
        sim = ar["err_div_only"][NODES.index("dac_remap"), ti].astype(float)
        ok = ar["valid_div_only"][NODES.index("dac_remap"), ti] & np.isfinite(pred)
        p, s = pred[ok], sim[ok]
        slope = float(np.sum(p * s) / np.sum(p * p))
        r2 = 1.0 - float(np.sum((s - p) ** 2)) / float(np.sum((s - s.mean()) ** 2))
        rows.append({"design": d, "law": "dac_remap|div_only (current mismatch)", "n": int(ok.sum()),
                     "rms_sim_lsb": float(np.sqrt(np.mean(s ** 2))), "rms_pred_lsb": float(np.sqrt(np.mean(p ** 2))),
                     "slope_sim_on_pred": slope, "r2_identity": r2,
                     "rms_residual_lsb": float(np.sqrt(np.mean((s - p) ** 2))), "g_prog_consistency_rel": consistency})
        pts[f"{d}|pred"], pts[f"{d}|sim"] = p.astype(np.float32), s.astype(np.float32)
        print(f"{d}: n={ok.sum()} rms sim {rows[-1]['rms_sim_lsb']:.3f} pred {rows[-1]['rms_pred_lsb']:.3f} "
              f"slope {slope:.3f} R2 {r2:.4f} resid {rows[-1]['rms_residual_lsb']:.3f} (G consistency {consistency:.1e})")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    np.savez_compressed(out_npz, **pts)
    return 0


if __name__ == "__main__":
    sys.exit(main())
