"""E4.1 independent check: run the three tasks into a new results folder
results/E4_1_independent_check_<YYYYMMDD_HHMM>/ (start time in Japan Standard Time; an existing
folder is never overwritten, a suffix _b, _c, ... is added).

Usage (from the repository root):  python experiments/E4_1_independent/run_check.py
Options: --skip-sensitivity (no numerical variants in task 1); --out <new folder> (development
only, a folder outside results/).
ngspice runs strictly one process at a time, with OMP_NUM_THREADS=1.
"""

from __future__ import annotations

import os
import sys

os.environ["OMP_NUM_THREADS"] = "1"      # before numpy and before any ngspice process
sys.dont_write_bytecode = True           # no __pycache__ folders in the repository

import argparse            # noqa: E402
import json                # noqa: E402
import platform            # noqa: E402
import re                  # noqa: E402
import subprocess          # noqa: E402
import time                # noqa: E402
from pathlib import Path   # noqa: E402

import numpy as np         # noqa: E402

import e41i_common as K    # noqa: E402
import make_report         # noqa: E402
import task1_spice         # noqa: E402
import task2_model         # noqa: E402
import task3_decision      # noqa: E402

INPUTS = [
    "experiments/E4_1_config.json",
    "experiments/E4_config.json",
    "logs/E4_1_freeze_sha256.txt",
    "results/E4_1_20261113_1612/cases.json",
    "results/E4_1_20261113_1612/rows.csv",
    "results/E4_1_20261113_1612/bounds.csv",
    "results/E4_1_20261113_1612/checks.json",
    "results/E4_1_20261113_1612/decision.json",
    "results/E4_1_20261113_1612/run.log",
    "results/E4_1_20261113_1612/run_info.json",
    "results/E4_1_20261113_1612/config_used.json",
    "results/E4_spice_20261017_0602/instrument.npz",
    "results/E4_independent_check_20261017_0741/compare.json",
    "results/E4_independent_check_20261017_0741/task2_model_bounds.csv",
    "src/pcm_acam/trees.py",
    "src/pcm_acam/acam.py",
]
SCRIPTS = ["e41i_common.py", "task1_spice.py", "task2_model.py", "task3_decision.py", "run_check.py",
           "make_report.py"]
IMPORTED = ["experiments/E4_independent/e4i_core.py", "experiments/E4_independent/e4i_spice.py"]


def new_run_dir() -> Path:
    stamp = K.jst_now().strftime("%Y%m%d_%H%M")
    base = K.ROOT / "results" / f"E4_1_independent_check_{stamp}"
    d, i = base, 0
    while d.exists():
        i += 1
        d = base.with_name(base.name + "_" + "bcdefghijklmnopqrstuvwxyz"[i - 1])
    d.mkdir(parents=True)
    return d


class Log:
    def __init__(self, path: Path):
        self.f = open(path, "a")

    def __call__(self, msg: str):
        line = f"[{K.jst_now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        self.f.write(line + "\n")
        self.f.flush()


def environment() -> dict:
    import scipy
    import sklearn
    try:
        txt = subprocess.run(["ngspice", "--version"], capture_output=True, text=True).stdout
        ng = [ln.strip("* ").strip() for ln in txt.splitlines() if "ngspice-" in ln][0]
        solver = [ln.strip("* ").strip() for ln in txt.splitlines() if "Solver" in ln]
    except Exception as e:                       # noqa: BLE001
        ng, solver = f"unavailable ({e})", []
    return {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
            "sklearn": sklearn.__version__, "ngspice": ng, "ngspice_build": solver,
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"), "platform": platform.platform()}


def provenance(hashes: dict) -> dict:
    """The inputs against the hashes recorded by the frozen plan, the freeze log and the run of record
    (only files that this check may read are hashed here)."""
    plan = K.load_plan()
    freeze = {}
    for ln in K.FREEZE_LOG.read_text().splitlines():
        m = re.match(r"^([0-9a-f]{64})\s+(\S+)$", ln.strip())
        if m:
            freeze[m.group(2)] = m.group(1)
    with open(K.MAIN_DIR / "run_info.json") as f:
        main_sha = json.load(f)["sha256"]
    with open(K.MAIN_DIR / "config_used.json") as f:
        used = json.load(f)
    out = {
        "plan_sha256_equals_freeze_log": hashes["experiments/E4_1_config.json"] == freeze.get("experiments/E4_1_config.json"),
        "plan_sha256_equals_run_of_record": hashes["experiments/E4_1_config.json"] == main_sha.get("experiments/E4_1_config.json"),
        # config_used.json is the plan re-serialized (indent 2), so I compare the parsed content
        "config_used_content_equals_plan": used == plan,
        "e4_plan_sha256_equals_frozen_plan": hashes["experiments/E4_config.json"] == plan["inputs"]["e4_plan"]["sha256"],
        "instrument_sha256_equals_frozen_plan": (hashes["results/E4_spice_20261017_0602/instrument.npz"]
                                                 == plan["inputs"]["instrument"]["sha256"]),
    }
    shared = ["experiments/E4_config.json", "src/pcm_acam/trees.py", "src/pcm_acam/acam.py",
              "third_party/sky130_fd_pr_subset_e3b/sky130_subset_tt.lib.spice"]
    out["shared_inputs_equal_run_of_record"] = {p: hashes[p] == main_sha.get(p) for p in shared}
    out["all"] = bool(all(v for k, v in out.items() if k != "shared_inputs_equal_run_of_record")
                      and all(out["shared_inputs_equal_run_of_record"].values()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-sensitivity", action="store_true")
    ap.add_argument("--out", default=None, help="development only: a new folder outside results/")
    a = ap.parse_args()
    if a.out:
        out = Path(a.out).resolve()
        out.mkdir(parents=True, exist_ok=False)
    else:
        out = new_run_dir()
    log = Log(out / "run.log")
    start = K.jst_now()
    t0 = time.time()
    log(f"E4.1 independent check -> {K.rel(out)}")
    hashes = {p: K.sha256(K.ROOT / p) for p in INPUTS}
    for p in K.model_files():
        hashes[K.rel(p)] = K.sha256(p)
    info = {"check": "E4.1 independent check", "run_dir": K.rel(out),
            "start_jst": start.isoformat(timespec="seconds"),
            "command": "python experiments/E4_1_independent/run_check.py"
                       + (" --skip-sensitivity" if a.skip_sensitivity else ""),
            "environment": environment(),
            "inputs_sha256": hashes,
            "scripts_sha256": {f"experiments/E4_1_independent/{s}": K.sha256(K.HERE / s) for s in SCRIPTS},
            "imported_scripts_sha256": {p: K.sha256(K.ROOT / p) for p in IMPORTED}}
    info["provenance"] = provenance(hashes)
    with open(out / "run_info.json", "w") as f:
        json.dump(info, f, indent=1)
    log(f"inputs: plan, E4 plan and instrument match the recorded SHA-256: {info['provenance']['all']}")

    cmp = {"check": info["check"], "run_dir": info["run_dir"], "start_jst": info["start_jst"],
           "provenance": info["provenance"], "tolerances": K.TOL,
           "gate": {"ln_ratio": K.GATE_LN, "fraction": K.GATE_FRACTION}}
    log("task 1: SPICE side (own row netlists, t_s(N))")
    cmp["task1"] = task1_spice.run(out, log=log, sensitivity=not a.skip_sensitivity)
    log("task 2: model side (48 cases)")
    cmp["task2"] = task2_model.run(out, log=log)
    log("task 3: decision of the frozen plan (V1GS)")
    cmp["task3"] = task3_decision.run(out, log=log)
    cmp["pass"] = {f"task{i}": cmp[f"task{i}"]["pass"] for i in (1, 2, 3)}
    cmp["all_pass"] = bool(all(cmp["pass"].values()) and info["provenance"]["all"])
    cmp["end_jst"] = K.jst_now().isoformat(timespec="seconds")
    cmp["elapsed_s"] = round(time.time() - t0, 1)
    with open(out / "compare.json", "w") as f:
        json.dump(cmp, f, indent=1, default=float)
    info["end_jst"] = cmp["end_jst"]
    info["elapsed_s"] = cmp["elapsed_s"]
    with open(out / "run_info.json", "w") as f:
        json.dump(info, f, indent=1)
    make_report.write(out, cmp, info)
    log(f"done: {cmp['pass']} all_pass={cmp['all_pass']} ({cmp['elapsed_s']} s)")


if __name__ == "__main__":
    main()
