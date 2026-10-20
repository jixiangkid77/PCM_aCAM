"""E4 independent check: run all three tasks into a new results folder
results/E4_independent_check_<YYYYMMDD_HHMM>/ (JST; an existing folder is never overwritten).

Usage (from the project root):  python experiments/E4_independent/run_check.py
Options: --skip-sensitivity (no task-1 variants), --only 1|2|3 (development).
ngspice runs strictly one process at a time (tasks 2 and 3 run after each other).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np

import e4i_core as C
import make_report
import task1_forest
import task2_v1
import task3_switching

INPUTS = [
    "experiments/E4_config.json",
    "experiments/E0_2_config.json",
    "results/E4_spice_20261017_0602/instrument.npz",
    "results/E4_spice_20261017_0602/v1_cases.json",
    "results/E4_spice_20261017_0602/v1_rows.csv",
    "results/E4_spice_20261017_0602/checks.json",
    "results/E4_spice_20261017_0602/config_used.json",
    "results/E4_20261017_0618/per_eval.csv",
    "results/E4_20261017_0618/refs.csv",
    "results/E4_20261017_0618/run_info.json",
    "src/pcm_acam/trees.py",
    "src/pcm_acam/acam.py",
    "third_party/sky130_fd_pr_subset_e3b/sky130_subset_tt.lib.spice",
]
SCRIPTS = ["e4i_core.py", "e4i_spice.py", "task1_forest.py", "task2_v1.py", "task3_switching.py",
           "run_check.py", "make_report.py"]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def new_run_dir() -> Path:
    stamp = C.jst_now().strftime("%Y%m%d_%H%M")
    base = C.ROOT / "results" / f"E4_independent_check_{stamp}"
    d = base
    i = 0
    while d.exists():
        i += 1
        d = base.with_name(base.name + "_" + "bcdefghij"[i - 1])
    d.mkdir(parents=True)
    return d


class Log:
    def __init__(self, path: Path):
        self.f = open(path, "a")

    def __call__(self, msg: str):
        line = f"[{C.jst_now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        self.f.write(line + "\n")
        self.f.flush()


def environment() -> dict:
    import scipy
    import sklearn
    try:
        ng = subprocess.run(["ngspice", "--version"], capture_output=True, text=True).stdout
        ng = [ln.strip("* ").strip() for ln in ng.splitlines() if "ngspice-" in ln][0]
    except Exception as e:                       # noqa: BLE001
        ng = f"unavailable ({e})"
    return {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
            "sklearn": sklearn.__version__, "ngspice": ng, "platform": platform.platform()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-sensitivity", action="store_true")
    ap.add_argument("--only", choices=["1", "2", "3"], default=None)
    ap.add_argument("--out", default=None, help="development only: a new folder outside results/")
    a = ap.parse_args()
    if a.out:
        out = Path(a.out).resolve()
        out.mkdir(parents=True, exist_ok=False)
    else:
        out = new_run_dir()
    log = Log(out / "run.log")
    start = C.jst_now()
    t0 = time.time()
    log(f"E4 independent check -> {C.rel(out)}")
    info = {"check": "E4 independent check", "run_dir": C.rel(out), "start_jst": start.isoformat(timespec="seconds"),
            "environment": environment(),
            "inputs_sha256": {p: sha256(C.ROOT / p) for p in INPUTS},
            "scripts_sha256": {f"experiments/E4_independent/{s}": sha256(C.HERE / s) for s in SCRIPTS},
            "instrument_sha256_protocol_section_10": "22699dcc0a52c4c4135b47dae3dbae741aadcedced6eb6f3c919f4068168057f"}
    info["instrument_sha256_matches_protocol"] = (
        info["inputs_sha256"]["results/E4_spice_20261017_0602/instrument.npz"]
        == info["instrument_sha256_protocol_section_10"])
    with open(out / "run_info.json", "w") as f:
        json.dump(info, f, indent=1)

    cmp = {"check": info["check"], "run_dir": info["run_dir"], "start_jst": info["start_jst"],
           "instrument_sha256_matches_protocol": info["instrument_sha256_matches_protocol"]}
    if a.only in (None, "1"):
        log("task 1: forest accuracies (640 evaluations)")
        t1 = task1_forest.run(out, log=log)
        c2 = {str(n): task1_forest.c2_reproduction(out, n_inv=n, log=log) for n in (40, 64)}
        t1["c2_reproduction"] = {n: {"values_bit_identical": s["values_bit_identical"],
                                     "values_total": s["values_total"]} for n, s in c2.items()}
        if not a.skip_sensitivity:
            log("task 1 sensitivity: 64 bisection steps; bilinear ln I surface")
            s64 = task1_forest.run(out, log=log, n_inv=64, tag="_n64")
            sbl = task1_forest.run(out, log=log, surface_scheme="bilinear_ln", tag="_bilinear")
            keys = ["acc_differing", "max_sample_difference", "leaf_loss_differing",
                    "max_abs_leaf_loss_difference", "max_sample_tree_pairs_difference", "multi_match_differing",
                    "removed_by_crosstalk_differing", "differences_by_point"]
            t1["sensitivity"] = {"bisection_64_steps": {k: s64[k] for k in keys},
                                 "surface_bilinear_ln": {k: sbl[k] for k in keys}}
        cmp["task1"] = t1
    if a.only in (None, "2"):
        log("task 2: V1 row transients (SPICE)")
        cmp["task2"] = task2_v1.run(out, log=log)
    if a.only in (None, "3"):
        log("task 3: switching points (SPICE)")
        cmp["task3"] = task3_switching.run(out, log=log)
    cmp["pass"] = {f"task{i}": cmp[f"task{i}"]["pass"] for i in (1, 2, 3) if f"task{i}" in cmp}
    cmp["all_pass"] = all(cmp["pass"].values())
    cmp["end_jst"] = C.jst_now().isoformat(timespec="seconds")
    cmp["elapsed_s"] = round(time.time() - t0, 1)
    with open(out / "compare.json", "w") as f:
        json.dump(cmp, f, indent=1, default=float)
    info["end_jst"] = cmp["end_jst"]
    info["elapsed_s"] = cmp["elapsed_s"]
    with open(out / "run_info.json", "w") as f:
        json.dump(info, f, indent=1)
    if a.only is None:
        make_report.write(out, cmp, info)
    log(f"done: {cmp['pass']} all_pass={cmp['all_pass']} ({cmp['elapsed_s']} s)")


if __name__ == "__main__":
    main()
