"""E4.1 independent check: paths, tolerances and helpers shared by the three tasks.

I build on the E4 independent check (experiments/E4_independent/): its instrument, PCM and
leakage code (e4i_core.py) and its ngspice helpers (e4i_spice.py) are imported from that folder
unchanged. Neither file reads or imports the main code of E4 or E4.1. Through e4i_core the only
project code that runs is src/pcm_acam/trees.py (loaded on its own) and make_draws of
src/pcm_acam/acam.py (compiled from that function's source only).

Not read and not imported here: experiments/E4_1_gainstage_rows.py, experiments/E4_forest_system.py,
src/pcm_acam/row_model.py, ml_transient.py, spice_iv.py, inverter.py, branch_post.py.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]                                   # repository root
E4I_DIR = ROOT / "experiments" / "E4_independent"


def _load_e4i(name: str):
    """Module of the E4 check loaded from its file. I do not put that folder on sys.path: it
    also holds a run_check.py and a make_report.py, which would shadow mine."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, E4I_DIR / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod             # e4i_spice imports e4i_core by this name
    spec.loader.exec_module(mod)
    return mod


C = _load_e4i("e4i_core")      # E4 independent check, unchanged
S = _load_e4i("e4i_spice")     # E4 independent check, unchanged

PLAN_PATH = ROOT / "experiments" / "E4_1_config.json"
MAIN_DIR = ROOT / "results" / "E4_1_20261113_1612"               # E4.1 run of record
SMOKE_DIR = ROOT / "results" / "E4_1_20261113_1611_smoke"        # not used for any number
E4_CHECK_DIR = ROOT / "results" / "E4_independent_check_20261017_0741"
FREEZE_LOG = ROOT / "logs" / "E4_1_freeze_sha256.txt"

DATASET, SPLIT, SEED, CONDITION, CELL_SET = "breast_cancer", 0, 0, "all", "GS"
POINTS = [("t0", "t0"), ("1y", "dac_remap"), ("1y", "supply")]
GATE_LN = 0.05            # plan: |ln(t_c,sim / t_c,pred)| <= 0.05 ...
GATE_FRACTION = 0.90      # ... for at least 90 % of the 48 cases

# Tolerances (README, "Implementation choices"). The E4 check compared its own crossing times
# with the main run at 0.5 % and found every model input that needs no root solve bit-identical
# and w, S within 4e-8 (relative) for the 48 6T2M cases. I keep both standards; the solved
# quantities get 1e-5, i.e. the 6T2M maximum times the gain of the extra inverter in the LBUF
# chain (about 20) times a margin of more than 10.
TOL = {
    "t_sim_rel": 0.005,         # my SPICE crossing time versus t_sim_ns of rows.csv
    "t_s_rel": 0.005,           # my SPICE t_s(N) versus t_s_ns
    "unsolved_rel": 1e-12,      # g_us (relative); x, theta_eff, V_DL, V_SL, a_hat (absolute)
    "solved_rel": 1e-5,         # w of every constrained bound, S and its two parts, t_c,pred
}


# ------------------------------------------------------------------------------------------ time
def jst_now() -> datetime:
    try:
        from zoneinfo import ZoneInfo
        return datetime.now(ZoneInfo("Asia/Tokyo"))
    except Exception:                                    # noqa: BLE001  (no tz database)
        return datetime.now(timezone(timedelta(hours=9)))


def rel(p: Path) -> str:
    """Path relative to the repository root (for logs and reports)."""
    try:
        return Path(p).resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(p)


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


# ------------------------------------------------------------------------------------- run inputs
def load_plan() -> dict:
    with open(PLAN_PATH) as f:
        return json.load(f)


def load_main() -> dict:
    """The E4.1 run of record: cases.json, rows.csv, bounds.csv, checks.json, decision.json."""
    with open(MAIN_DIR / "cases.json") as f:
        cases = json.load(f)
    with open(MAIN_DIR / "rows.csv") as f:
        rows = list(csv.DictReader(f))
    with open(MAIN_DIR / "bounds.csv") as f:
        bounds = list(csv.DictReader(f))
    with open(MAIN_DIR / "checks.json") as f:
        checks = json.load(f)
    with open(MAIN_DIR / "decision.json") as f:
        decision = json.load(f)
    assert len(rows) == len(cases["cases"])
    for i, r in enumerate(rows):
        assert int(r["case"]) == i
    # bounds.csv lists the bounds of each case in the order of cases.json
    per_case = {}
    for b in bounds:
        per_case.setdefault(int(b["case"]), []).append(b)
    for i, c in enumerate(cases["cases"]):
        assert [int(b["kind"]) for b in per_case.get(i, [])] == [bd["kind"] for bd in c["bounds"]]
    return {"cases": cases, "rows": rows, "bounds": bounds, "bounds_by_case": per_case,
            "checks": checks, "decision": decision}


def group_key(c: dict) -> str:
    return f"{c['design']}|{c['time']}|{c['node']}"


def first_of_groups(cases: list) -> list:
    """Index of the first case of each (design, time, node) group, in file order."""
    seen, sel = set(), []
    for i, c in enumerate(cases):
        k = group_key(c)
        if k not in seen:
            seen.add(k)
            sel.append(i)
    return sel


def lib_rel_from(work: Path) -> str:
    """Model library path relative to a netlist folder (so that no absolute path enters the
    stored netlists and logs)."""
    lib = ROOT / C.load_config()["models"]["lib"]
    return Path(os.path.relpath(lib, work)).as_posix()


def model_files() -> list:
    """Every file of the sky130 subset that ngspice reads (for the SHA-256 record)."""
    lib = ROOT / C.load_config()["models"]["lib"]
    return sorted(p for p in lib.parent.iterdir() if p.suffix == ".spice")


def write_csv(path: Path, rows: list):
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)


def max_abs(vals, default=float("nan")) -> float:
    vals = [abs(float(v)) for v in vals]
    return max(vals) if vals else default
