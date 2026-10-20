"""E2 post-run analysis (exploratory, NOT part of the frozen decision).

Splits each prediction-versus-simulation difference into
  sampling : the same first-order law evaluated with the realized random draws of this run, divided by the
             frozen prediction (which is an expectation over the Gaussian mismatch distribution)
  model    : the Monte Carlo result divided by the realized-draw law (what the law misses for these very draws)
so that  sim / pred = sampling x model.

Realized-draw laws (per instance k and threshold theta, 1 year unless stated):
  dac_remap : dx = (G/dG) * ( sum_p [s_p(aI) - s_p(I)] z_div[k,p] + [lambda(aI) - lambda(I)] delta_k )
  supply    : dx = (G/dG) * ln r_k   and the variant (G/dG) * (r_k - 1),  r_k = (D - a delta_k) / (D - delta_k)
  delta_k   = V_sw,k(Monte Carlo) - V_sw,nom; the linear estimate sum_p dV_sw/dmm_p z_pd[k,p] is reported too.
The sensitivities s_p, lambda and dV_sw/dmm_p are the frozen ones in predictions.json; the draws are regenerated
from the frozen seed.

Usage: python experiments/E2_analysis.py results/E2_<stamp>
Writes analysis_decomposition.csv, analysis_supply_form.csv, analysis_times_clip.csv into the run folder
(new files only).
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

MM_KEYS = ("vth0", "toxe", "voff", "nfactor")


def rms_lsb(dx: np.ndarray) -> float:
    return float(15.0 * np.sqrt(np.mean(dx ** 2)))


def main() -> int:
    run = Path(sys.argv[1])
    outs = [run / n for n in ("analysis_decomposition.csv", "analysis_supply_form.csv", "analysis_times_clip.csv")]
    for p in outs:
        if p.exists():
            sys.exit(f"{p} exists; refusing to overwrite")
    info = json.loads((run / "run_info.json").read_text())
    root = Path(__file__).resolve().parents[1]
    pred_path = root / info["predictions_file"]
    preds = json.loads(pred_path.read_text())
    cfg = json.loads((run / "config_used.json").read_text())
    mmc = cfg["mismatch"]
    rng = np.random.default_rng([mmc["base_seed"], mmc["rng_stream"]])
    z_div = rng.standard_normal((mmc["n_instances"], 4))
    z_pd = rng.standard_normal((mmc["n_instances"], 4))
    g_lo, g_hi = cfg["circuit"]["g_lo_us"], cfg["circuit"]["g_hi_us"]
    dg = g_hi - g_lo
    sim = {(r["condition"], r["setting"], r["node"], r["time"], r["variant"]): float(r["rms"])
           for r in csv.DictReader(open(run / "summary.csv"))}
    comp = {(r["condition"], r["setting"], r["node"], r["time"]): float(r["pred_rms_lsb"])
            for r in csv.DictReader(open(run / "compare.csv"))}
    a_of = {n: float((t / cfg["drift"]["t0_s"]) ** (-cfg["drift"]["nu"])) for n, t in cfg["drift"]["times_s"].items()}

    dec, form, tc = [], [], []
    for cid, e in preds.items():
        c = e["condition"]
        ar = np.load(run / f"arrays_{cid}.npz")
        theta = ar["theta"]
        w = ((g_lo + theta * dg) / dg)[None, :]
        d_v = c["d_v"]
        pds = e["pulldown_sensitivity"]
        slope = np.array([pds[f"dvsw_d{k}"] for k in MM_KEYS])
        delta_mc = ar["vsw_mc"] - e["vsw_nom_V"]
        delta_lin = z_pd @ slope
        row_d = {"condition": cid, "set": c["set"], "sigma_delta_pred_mV": 1e3 * pds["sigma_delta"],
                 "sigma_delta_mc_mV": 1e3 * float(np.std(delta_mc, ddof=1)),
                 "sigma_delta_lin_draws_mV": 1e3 * float(np.std(delta_lin, ddof=1)),
                 "rms_delta_mc_minus_lin_mV": 1e3 * float(np.sqrt(np.mean((delta_mc - delta_lin) ** 2)))}
        for tn, a in a_of.items():
            pr = e["predictions"][tn]
            s1, s2 = pr["s_at_I"], pr["s_at_aI"]
            ds = np.array([np.array(s2[k]) - np.array(s1[k]) for k in MM_KEYS])          # (4, 16)
            dlam = np.array(s2["lambda"]) - np.array(s1["lambda"])                     # (16,)
            div_part = z_div @ ds                                                      # (n, 16)
            laws = {
                ("pdk", "dac_remap"): w * (div_part + dlam[None, :] * delta_mc[:, None]),
                ("div_only", "dac_remap"): w * div_part,
                ("pd_only", "dac_remap"): w * (dlam[None, :] * delta_mc[:, None]),
            }
            r_mc = (d_v - a * delta_mc) / (d_v - delta_mc)
            r_lin = (d_v - a * delta_lin) / (d_v - delta_lin)
            for s in ("pdk", "pd_only"):
                laws[(s, "supply")] = w * np.log(r_mc)[:, None]
            for (s, node), dx in laws.items():
                p = comp[(cid, s, node, tn)]
                lr = rms_lsb(dx)
                sv = sim[(cid, s, node, tn, "noclip")]
                dec.append({"condition": cid, "set": c["set"], "time": tn, "setting": s, "node": node,
                            "pred_rms_lsb": p, "law_realized_rms_lsb": lr, "sim_rms_lsb": sv,
                            "sampling_factor": lr / p if p > 0 else np.nan,
                            "model_factor": sv / lr if lr > 0 else np.nan,
                            "sim_over_pred": sv / p if p > 0 else np.nan})
                if node == "supply" and s == "pdk":
                    err = ar[f"err_pdk_supply_{tn}"].astype(float)
                    for name, dx2 in (("ln_r_mc_delta", w * np.log(r_mc)[:, None]),
                                      ("r_minus_1_mc_delta", w * (r_mc - 1.0)[:, None]),
                                      ("ln_r_linear_delta", w * np.log(r_lin)[:, None])):
                        form.append({"condition": cid, "time": tn, "law_form": name,
                                     "law_rms_lsb": rms_lsb(dx2), "sim_rms_lsb": sv,
                                     "sim_over_law": sv / rms_lsb(dx2),
                                     "residual_rms_lsb": float(np.sqrt(np.mean((err - 15.0 * dx2) ** 2)))})
            for node in ("dac_remap", "supply"):
                tc.append({"condition": cid, "set": c["set"], "time": tn, "node": node,
                           "pred_rms_lsb": comp[(cid, "pdk", node, tn)],
                           "sim_noclip_rms_lsb": sim[(cid, "pdk", node, tn, "noclip")],
                           "sim_clip25_rms_lsb": sim[(cid, "pdk", node, tn, "clip25")]})
        for r in dec:
            if r["condition"] == cid and "sigma_delta_pred_mV" not in r:
                r.update(row_d)
    for path, rows in zip(outs, (dec, form, tc)):
        with open(path, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            wr.writeheader()
            wr.writerows(rows)
    # console summary (1 year, pdk)
    for r in dec:
        if r["time"] == "1y" and r["setting"] == "pdk":
            print(f"{r['condition']:3} {r['node']:9} pred {r['pred_rms_lsb']:.3f} law(draws) {r['law_realized_rms_lsb']:.3f} "
                  f"sim {r['sim_rms_lsb']:.3f}  sampling {r['sampling_factor']:.3f} model {r['model_factor']:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
