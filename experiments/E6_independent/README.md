# E6 independent check (part 1: own values)

Written by an assistant that did not take part in writing `experiments/E6_cost.py`. Inputs: the frozen plan
`docs/E6_protocol.md`, `experiments/E6_config.json`, `docs/E4_protocol.md` section 2, the sky130 tt model subset
`third_party/sky130_fd_pr_subset_e3b/` and `src/pcm_acam/trees.py` (only `dataset_bundle`, the E4 forest
construction). Netlists, the ngspice runner, the binary raw-file parser, curve inversion, settling-time extraction
and power accounting are written in `run_check.py`; no other project code is imported.

## Commands

From the project root:

```
python experiments/E6_independent/run_check.py
```

writes `results/E6_independent_check_<YYYYMMDD_HHMM>/` (JST; `_b`, `_c` ... if the name is taken; an existing
folder is never written into). Options:

- `--tmp DIR`: base folder for the private ngspice temporary folder (default: the system temp folder; `TMPDIR` is
  honoured); it is deleted at the end unless `--keep-tmp`
- `--no-conv`: skip the transient convergence reruns (half step, quarter step, Gear)
- `--out DIR`: write to DIR instead (must not exist)

Requirements: Python 3 with numpy, scikit-learn (for `trees.py`), and `ngspice` (ngspice-42) on the PATH. One ngspice
process at a time; about 90 short runs, about 1 min CPU in total. ngspice is started with `OMP_NUM_THREADS=1` and
`OMP_WAIT_POLICY=passive`: this ngspice links libgomp, and with both cores busy the default spinning OpenMP waits made
identical runs up to about 50 times slower at random (outputs bit-identical).

## Phase 2

```
python experiments/E6_independent/run_extra_lbuf.py
python experiments/E6_independent/compare.py results/E6_20261024_1032
```

- `run_extra_lbuf.py` (step A, run before any main-run file was opened): additional group 1d, the LBUF lower
  branch of the GS cell set (P1, P3; t0 and input remap at 1 year; level 15; x_prev = 0; GS map and programming at
  V_DS = V_X*). Settling to ±0.01 LSB and ±0.1 LSB, tail time constants of X, the first INV_S output and the
  decision node. Writes only new files into `results/E6_independent_check_20261024_0936/`: `extra_lbuf.json`,
  `extra_lbuf.csv`, `extra_lbuf_run.log`, `netlists_extra_lbuf/`.
- `compare.py` (step B): reads from the main-run folder only `settle.csv`, `power_curves.npz`, `energy_dl_ml.csv`,
  `overhead.csv`, `checks.json`, `run.log`; writes `compare.json` and `report.md` into the independent results folder
  (refuses to overwrite). As an explanation aid for item 2 it also evaluates my DC power at the two 1 mV data-line
  grid points around V_DL (one short ngspice run; `--tmp` sets its temporary folder).

## Outputs

| File | Content |
|---|---|
| `values.json` | all values, settings and checks (full double precision) |
| `settling.csv` | item 1: 16 settling cases (1a: plan section 10; 1b, 1c: additional) |
| `power.csv` | item 2: static power, 24 cases at x = 0.25, 0.75 plus 12 at x = θ (information) |
| `counts.csv` | item 3: iris, splits 0-2 and their mean |
| `dl_maps.csv` | data-line map M0 and M_a(1 y) at the x values used, with the inversion residuals |
| `netlists/` | every netlist that was run |
| `run.log` | log with JST time stamps |
| `report_part1.md` | tables and numerics |
