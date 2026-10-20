"""Supplementary robustness check: rerun Task 2 for one condition with ngspice's default
Newton tolerances (no .options line) and compare with the tight-tolerance results of run_check.py.
usage: python extra_tolerance_robustness.py <results dir of run_check.py> [condition, default C2]
"""
import json
import os
import sys

import numpy as np

import ind_spice as S

_orig_header = S.header
S.header = lambda corner, temp_c, title="robustness", options=None: _orig_header(corner, temp_c, title, options="")

import task2_branch_mc as T2  # noqa: E402  (uses the patched S.header at call time)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))).replace(os.sep, "/")  # repository root (was an absolute path in the original run)


def main(outdir, cid="C2"):
    cfg = json.load(open(f"{ROOT}/experiments/E2_config.json"))
    conds = {c["id"]: c for c in cfg["conditions"]}
    stamp = os.path.basename(outdir.rstrip("/")).replace("E2_independent_check_", "")
    wd = f"{ROOT}/logs/E2_independent_work/run_{stamp}_default_tol_{cid}"
    r = T2.run_condition(conds[cid], cfg["circuit"]["pulldowns"], cfg["drift"]["times_s"]["1y"], wd, n_inst=40)
    ind = np.load(os.path.join(outdir, f"independent_{cid}.npz"))
    res = dict(condition=cid, note="Task 2 rerun with ngspice default tolerances vs tight tolerances",
               max_abs_diff_err_lsb={n: float(np.max(np.abs(r["err"][i] - ind["err"][i])))
                                     for i, n in enumerate(r["nodes"])},
               max_abs_diff_bound_V=float(np.max(np.abs(r["bound"] - ind["bound"]))),
               max_abs_diff_M0_V=float(np.max(np.abs(r["M0"] - ind["M0"]))),
               max_abs_diff_vsw_V=float(np.max(np.abs(r["vsw"] - ind["vsw"]))))
    with open(os.path.join(outdir, f"robustness_default_tolerances_{cid}.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], *(sys.argv[2:3]))
