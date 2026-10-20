"""E3b post-run analysis (secondary quantities named in the frozen plan; no new simulation).

Usage: python experiments/E3b_analysis.py results/E3b_<stamp>
Writes into the run folder (new files only):
  analysis_sigma_breakdown.csv   sigma_delta_Y split by device group (inverter nMOS, inverter pMOS, pull-down),
                                 from the frozen +-1 sigma sensitivities, with the Monte Carlo sigma of V_Y*
  analysis_branch_comparison.csv upper branch (this run) next to the lower branch with the same divider and D
                                 (Monte Carlo, no clipping, 1 year, both devices mismatched): E1 designs P1, P3
                                 from the pre-freeze E2 calibration run, E2 condition G1 from the E2 run. The
                                 lower-branch runs used other random draws (sampling about 4 %).
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOWER = {  # divider geometry -> (run folder, condition id) of the lower branch, pd_min, D = 0.2 V, tt, 27 C
    (0.42, 0.15): ("results/E2cal_E1_20261014_0303", "P1"),
    (1.0, 0.5): ("results/E2_20261014_0316", "G1"),
    (0.42, 4.0): ("results/E2cal_E1_20261014_0303", "P3"),
}


def rcsv(p: Path):
    with open(p, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> int:
    run = Path(sys.argv[1])
    outs = [run / "analysis_sigma_breakdown.csv", run / "analysis_branch_comparison.csv"]
    for p in outs:
        if p.exists():
            sys.exit(f"{p} exists; refusing to overwrite")
    info = json.loads((run / "run_info.json").read_text())
    preds = json.loads((ROOT / info["predictions_file"]).read_text())
    checks = json.loads((run / "checks.json").read_text())
    summ = rcsv(run / "summary.csv")

    def rms(rows, cid, setting, node, time="1y", key="condition", variant=None):
        for r in rows:
            if r[key] == cid and r["setting"] == setting and r["node"] == node and r["time"] == time and \
                    (variant is None or r.get("variant") == variant):
                return float(r["rms"])
        raise KeyError((cid, setting, node, time, variant))

    sig_rows, cmp_rows = [], []
    seen_inv = set()
    for cid, e in preds.items():
        c = e["condition"]
        ys = e["y_sensitivity"]
        grp = {"inv_nmos": 0.0, "inv_pmos": 0.0, "pulldown": 0.0}
        names = {"invn": "inv_nmos", "invp": "inv_pmos", "pd": "pulldown"}
        for k, v in ys.items():
            if k.startswith("dvy_d"):
                grp[names[k[5:].split("_")[0]]] += v * v
        if c["inverter"] not in seen_inv:
            seen_inv.add(c["inverter"])
            sig_rows.append({"inverter": c["inverter"], "condition": cid, "vy_star_nom_V": ys["vy_star_nom"],
                             "inverter_gain": ys["inverter_gain"],
                             **{f"sigma_{g}_mV": 1e3 * math.sqrt(v) for g, v in grp.items()},
                             **{f"share_{g}": v / sum(grp.values()) for g, v in grp.items()},
                             "sigma_pred_mV": 1e3 * ys["sigma_delta"],
                             "sigma_mc_mV": checks["C4"][cid]["sigma_mc_mV"]})
        key = (c["w_div_um"], c["l_div_um"])
        up_dac, up_sup = rms(summ, cid, "pdk", "dac_remap"), rms(summ, cid, "pdk", "supply")
        row = {"divider_w_um": c["w_div_um"], "divider_l_um": c["l_div_um"], "d_v": c["d_v"],
               "upper_condition": cid, "inverter": c["inverter"], "upper_dac_remap": up_dac, "upper_supply": up_sup,
               "upper_better": "dac_remap" if up_dac < up_sup else "supply"}
        if key in LOWER and abs(c["d_v"] - 0.2) < 1e-12:
            folder, lid = LOWER[key]
            lrows = rcsv(ROOT / folder / "summary.csv")
            lo_dac = rms(lrows, lid, "pdk", "dac_remap", variant="noclip")
            lo_sup = rms(lrows, lid, "pdk", "supply", variant="noclip")
            row.update({"lower_source": f"{folder}/{lid}", "lower_dac_remap": lo_dac, "lower_supply": lo_sup,
                        "lower_better": "dac_remap" if lo_dac < lo_sup else "supply"})
        cmp_rows.append(row)
    for path, rows in zip(outs, (sig_rows, cmp_rows)):
        keys = []
        for r in rows:
            for k in r:
                if k not in keys:
                    keys.append(k)
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
    for r in sig_rows:
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
    for r in cmp_rows:
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
