"""Compare the E1 main run with the independent full-branch recomputation, row by row.

Usage: python experiments/E1_compare_independent.py results/E1_<stamp> results/E1_independent_check_<stamp>
Writes results/E1_compare_independent_<stamp>/comparison.json and rows.csv (never overwritten).
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pcm_acam.io_utils import make_run_dir, write_json  # noqa: E402

NODES = ("none", "dac_remap", "dac_gain", "supply", "ideal")
TOL_ERR_LSB = 0.01
TOL_VSW_MV = 0.05
TOL_BOUND_MV = 0.05


def main() -> int:
    main_dir, ind_dir = Path(sys.argv[1]), Path(sys.argv[2])
    cfg = json.loads((main_dir / "config_used.json").read_text())
    times = list(cfg["drift"]["times_s"])
    ti = times.index("1y")
    rows = list(csv.DictReader(open(ind_dir / "independent.csv")))
    nominal = list(csv.DictReader(open(ind_dir / "nominal.csv")))
    out_dir = make_run_dir(ROOT / "results", "E1_compare_independent", smoke=False)
    arrays = {d: np.load(main_dir / f"arrays_{d}.npz") for d in sorted({r["design"] for r in rows})}
    a = (cfg["drift"]["times_s"]["1y"] / cfg["drift"]["t0_s"]) ** (-cfg["drift"]["nu"])

    out_rows = []
    worst = {"err_lsb": 0.0, "vsw_mV": 0.0, "g_rel": 0.0}
    clip_mismatch = 0
    for r in rows:
        d, i, j, node = r["design"], int(r["instance"]), int(r["j"]), r["node"]
        ar = arrays[d]
        vsw_main = float(ar["vsw_pdk"][i])
        g_main = float(ar["gprog_pdk"][i, j])
        err_main = float(ar["err_pdk"][NODES.index(node), ti, i, j])
        valid_main = bool(ar["valid_pdk"][NODES.index(node), ti, i, j])
        clipped_ind = int(r["clipped"]) == 1
        clipped_main = not (0.0 <= g_main <= cfg["circuit"]["g_max_us"])
        if clipped_ind != clipped_main:
            clip_mismatch += 1
        d_vsw = 1e3 * abs(vsw_main - float(r["vsw_V"]))
        d_g = abs(g_main - float(r["g_prog_uS"])) / max(abs(g_main), 1e-12)
        d_err = abs(err_main - float(r["err_lsb"])) if (valid_main and not clipped_ind) else np.nan
        worst["vsw_mV"] = max(worst["vsw_mV"], d_vsw)
        worst["g_rel"] = max(worst["g_rel"], d_g)
        if np.isfinite(d_err):
            worst["err_lsb"] = max(worst["err_lsb"], d_err)
        out_rows.append({"design": d, "instance": i, "j": j, "node": node, "vsw_main_V": vsw_main,
                         "vsw_ind_V": float(r["vsw_V"]), "g_main_uS": g_main, "g_ind_uS": float(r["g_prog_uS"]),
                         "clipped_main": int(clipped_main), "clipped_ind": int(clipped_ind),
                         "err_main_lsb": err_main, "err_ind_lsb": float(r["err_lsb"]), "abs_diff_lsb": d_err})

    # nominal maps: M0(theta_j) and Ma(theta_j) from the main run's stored nominal curve
    nom_cmp = []
    worst_bound = 0.0
    for nrow in nominal:
        d = nrow["design"]
        ar = arrays[d]
        th = float(nrow["theta"])
        j = int(round(th * 15))
        m0_main = float(ar["b_target"][j])
        vg, ln_i = ar["vg_div"], ar["nom_ln_i"]
        info = {k: float(v) for k, v in nrow.items() if k not in ("design",) and v not in ("",)}
        g_th = 2.5 + 20 * th
        dsg = next(x for x in cfg["designs"] if x["id"] == d)
        d_v = dsg["d_v"]
        ma_main = float(np.interp(np.log(a * g_th * 1e-6 * d_v), ln_i, vg))
        m0_ind = info.get("M0_V", info.get("m0_V", np.nan))
        ma_ind = info.get("Ma_V", info.get("ma_V", np.nan))
        db = max(abs(m0_main - m0_ind), abs(ma_main - ma_ind)) * 1e3
        worst_bound = max(worst_bound, db)
        nom_cmp.append({"design": d, "theta": th, "M0_main": m0_main, "M0_ind": m0_ind, "Ma_main": ma_main,
                        "Ma_ind": ma_ind, "max_abs_diff_mV": db})

    with open(out_dir / "rows.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    res = {"main_run": main_dir.name, "independent": ind_dir.name, "n_rows": len(out_rows),
           "n_compared_err": int(sum(np.isfinite(r["abs_diff_lsb"]) for r in out_rows)),
           "max_abs_diff": {"err_lsb": worst["err_lsb"], "vsw_mV": worst["vsw_mV"], "g_prog_rel": worst["g_rel"],
                            "nominal_maps_mV": worst_bound},
           "clip_flag_mismatches": clip_mismatch,
           "tolerances": {"err_lsb": TOL_ERR_LSB, "vsw_mV": TOL_VSW_MV, "bound_mV": TOL_BOUND_MV},
           "nominal": nom_cmp}
    res["pass"] = bool(worst["err_lsb"] <= TOL_ERR_LSB and worst["vsw_mV"] <= TOL_VSW_MV
                       and worst_bound <= TOL_BOUND_MV and clip_mismatch == 0)
    write_json(out_dir / "comparison.json", res)
    print(json.dumps({k: v for k, v in res.items() if k != "nominal"}, indent=1))
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
