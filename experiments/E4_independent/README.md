# E4 independent check

Independent re-implementation check of experiment E4 (forest system, PCM analog CAM with
transistor-level bound errors and match-line crosstalk). Written from `docs/E4_protocol.md`,
`experiments/E4_config.json`, `docs/E0_2_protocol.md`, `experiments/E0_2_config.json`, the run
outputs and the checker brief. It does not read or import `src/pcm_acam/row_model.py` or
`experiments/E4_forest_system.py`. The only project code used:

- `src/pcm_acam/trees.py` (`dataset_bundle`, `quantize`), loaded as a stand-alone module so
  that no other file of the package is executed;
- `make_draws` from `src/pcm_acam/acam.py`, compiled from that function's source only (the rest
  of `acam.py` and its imports are not executed).

All paths are resolved relative to the project root from the location of the scripts.

## Command

Run from the project root (the folder that contains `experiments/`, `results/`, `src/` and
`third_party/`):

```
python experiments/E4_independent/run_check.py
```

This creates a new folder `results/E4_independent_check_<YYYYMMDD_HHMM>/` (JST time stamp of the
start; if that folder exists, a suffix `_b`, `_c`, ... is added, nothing is overwritten) and runs:

| Task | Script | What it does | Time (2 cores) |
|---|---|---|---|
| 1 | `task1_forest.py` | 640 evaluations (wine, breast_cancer; split 0; seeds 0-1; P1, P3; arms xt, is, gs, gi; conditions all, unif; t0 and 1d/30d/1y x dac_remap/supply/ideal) from `instrument.npz`, compared with `per_eval.csv`; recomputation of check C2 of `checks.json`; two sensitivity variants (64-step bisection, bilinear ln I surface) | about 2.5 min |
| 2 | `task2_v1.py` | own full-row transient netlists for the first V1 case of each (design, node) group; crossing time of 0.9 V versus `v1_rows.csv`; t_s(N); all 48 V1 cases recomputed with the task-1 model | about 40 s |
| 3 | `task3_switching.py` | switching points of instances 0-20 of L6T, UINV, LBUF (and W*) by DC sweeps; extra checks of the stored pull-down, inverter and divider curves | about 30 s |

`run_check.py --skip-sensitivity` omits the two task-1 variants. ngspice (tested with ngspice-42)
must be on the PATH; it runs one process at a time. Python packages: numpy, scikit-learn (for
`trees.py`), scipy (only to record its version).

Outputs in the results folder: `compare.json` (all numbers and pass flags), `report.md`
(Chinese summary), `run_info.json` (environment, SHA-256 of all inputs and of these scripts),
`run.log`, `task1_*.csv/json`, `task2_*.csv/json`, `task3_*.csv/json`, and the SPICE netlists,
logs and raw files in `spice_task2/` and `spice_task3/`. `make_report.py` rewrites `report.md`
from `compare.json` and `run_info.json` of a results folder given as its argument; for the run of
record:

```
python experiments/E4_independent/make_report.py results/E4_independent_check_20261017_0741
```

Run of record: `results/E4_independent_check_20261017_0741/` (all three tasks PASS).
`results/E4_independent_check_20261017_0732/` is an earlier run with identical numbers, superseded
(see its `NOTE_superseded.txt`).

## Files

| File | Content |
|---|---|
| `e4i_core.py` | paths; loaders of `trees.py` and `make_draws`; instrument reconstruction (Catmull-Rom deviations, bisection inversion, I-V surface, chains); PCM model and replica a_hat; programming, bound voltages, effective thresholds, per-instance leakage |
| `e4i_spice.py` | mismatch draws, netlist helpers, sequential ngspice runner, binary raw reader |
| `task1_forest.py`, `task2_v1.py`, `task3_switching.py` | the three tasks |
| `run_check.py` | driver (results folder, logging, hashes, compare.json, report) |
| `make_report.py` | report.md from compare.json |

## Implementation choices not fixed by the brief

- Bound-voltage inversion: bisection on `[vg_c[0], vg_c[-1]]`, 40 steps, midpoint returned. With
  these settings all 32 values of check C2 in `checks.json` are reproduced bit for bit (64 steps
  reproduce none of them bit for bit). The step count matters only where an input equals a
  threshold exactly and the effective threshold equals theta up to rounding (unif at t0 and at
  the ideal node); the 64-step variant is reported as a sensitivity run.
- Surface inversion (V_G') and node solve: bisection, 64 steps; I-V surface interpolated as the
  brief describes (ln I linear along V_G, then I linear along V_DS); a bilinear-in-ln-I variant
  is reported.
- Bounds beyond the u-cut (UINV 3 LSB, LBUF 0.45 LSB): w of the instance's own pull-down at 0 V.
- `multi_match` = share of (sample, tree) pairs with more than one matching row;
  `removed_by_crosstalk` = number of (sample, row) pairs with no violated bound that the
  crosstalk rule removes (both compared with `per_eval.csv` as extra information).
- Switching points: pull-downs swept at 0.1 mV with linear interpolation of ln I; inverters swept
  at 20 uV with linear interpolation; tight DC tolerances (`reltol=1e-7 abstol=1e-16 vntol=1e-10`;
  default tolerances give the same values).
- V1 transients: `.ic v(ml)=1.8` without UIC, `reltol=1e-4 abstol=1e-15 vntol=1e-8`, maximum
  step 0.1 ns, ML capacitor C_SA + N C_wire; mismatch parameters passed as `mm_vth0`, `mm_toxe`,
  `mm_voff`, `mm_nfactor` (default tolerances and a 0.4 ns step give the same crossing time).
