# E4.1 independent check

Independent re-implementation check of experiment E4.1 (full-row transient check of the row model
for the gain-stage cell set GS, i.e. check V1 of E4 repeated for GS). Written from the frozen plan
`experiments/E4_1_config.json`, the E4 plan `experiments/E4_config.json`, the outputs of the E4.1
run of record and a short task description. It builds on the E4 independent check
(`experiments/E4_independent/`): `e4i_core.py` (instrument reconstruction, PCM model, programming,
bound voltages, effective thresholds, per-instance leakage, including the two-inverter LBUF chain)
and `e4i_spice.py` (mismatch draws of the E4 pools, ngspice runner, `meas` parser) are imported from
there unchanged. They are loaded by file path, so that the `run_check.py` and `make_report.py` of
that folder cannot shadow the ones here.

## What it reads, and what it does not

Reads:

- `experiments/E4_1_config.json`, `experiments/E4_config.json`, `logs/E4_1_freeze_sha256.txt`;
- the E4.1 run of record `results/E4_1_20261113_1612/`: `cases.json`, `rows.csv`, `bounds.csv`,
  `checks.json`, `decision.json`, `run.log`, `run_info.json`, `config_used.json`;
- `results/E4_spice_20261017_0602/instrument.npz` (through `e4i_core.py`);
- `results/E4_independent_check_20261017_0741/compare.json` and `task2_model_bounds.csv` (the
  6T2M numbers that set the model-side tolerance, see below);
- the sky130 subset `third_party/sky130_fd_pr_subset_e3b/` (through ngspice);
- project code only through `e4i_core.py`, in the same restricted way as the E4 check:
  `src/pcm_acam/trees.py` (`dataset_bundle`, `quantize`; loaded as a stand-alone module) and
  `make_draws` of `src/pcm_acam/acam.py` (compiled from that function's source only).

Does not read or import: `experiments/E4_1_gainstage_rows.py`, `experiments/E4_forest_system.py`,
`src/pcm_acam/row_model.py`, `ml_transient.py`, `spice_iv.py`, `inverter.py`, `branch_post.py`.
The `run_info.json` of the run of record lists their SHA-256; I compare only the recorded hashes of
files that this check reads. The numbers of the smoke run `results/E4_1_20261113_1611_smoke/` are
not used.

## Command

Run from the repository root (the folder that contains `experiments/`, `results/`, `src/` and
`third_party/`):

```
python experiments/E4_1_independent/run_check.py
```

This creates a new folder `results/E4_1_independent_check_<YYYYMMDD_HHMM>/` (start time in Japan
Standard Time; if that folder exists, a suffix `_b`, `_c`, ... is added, nothing is overwritten)
and runs:

| Task | Script | What it does | Time (2 cores) |
|---|---|---|---|
| 1 | `task1_spice.py` | own GS row netlists for the first case of each (design, time, node) group of `cases.json` (cases 0, 8, 16, 24, 32, 40); crossing time of 0.9 V versus `t_sim_ns` of `rows.csv` (0.5 %) and versus `t_pred_ns`; t_s(N) by SPICE versus `t_s_ns`; gate voltages at 2 ns versus the SPICE gates of `bounds.csv`. Extra, outside the pass flag: the 4 cases whose LBUF bounds carry at least half of S (11, 15, 21, 25); three numerical variants | about 60 s |
| 2 | `task2_model.py` | all 48 cases recomputed with the E4-check model, cell set GS: instance index, x, conductance, effective threshold, data-line voltage, u, divider node, gate and w of every constrained bound; a_hat, supplies, S, its LBUF and UINV parts and t_c,pred of every case; compared with `cases.json`, `rows.csv`, `bounds.csv`. Extra: the case selection of the plan redone from scratch | about 2 s |
| 3 | `task3_decision.py` | gate V1GS recomputed (A) from `rows.csv`, (B) with my t_c,pred, (C) with my t_c,pred and my t_s(N), (D) for my SPICE rows; row decisions (model versus simulation, near ties); gate voltages at 2 ns, model versus simulation | under 1 s |

`run_check.py --skip-sensitivity` omits the numerical variants of task 1. ngspice (tested with
ngspice-42) must be on the PATH; `run_check.py` sets `OMP_NUM_THREADS=1` and runs one ngspice
process at a time. Python packages: numpy, scikit-learn (for `trees.py`), scipy (only to record its
version). The tasks also run on their own (`--out <new folder>`; task 3 needs the outputs of tasks 1
and 2 in that folder).

Outputs in the results folder: `compare.json` (all numbers and pass flags), `report.md` (Chinese
summary), `run_info.json` (environment and ngspice version, SHA-256 of every input file, of these
scripts and of the two imported E4-check scripts, provenance checks of the inputs against the
frozen plan, the freeze log and the run of record), `run.log`, `task1_*.csv/json`,
`task2_*.csv/json`, `task3_*.csv/json`, and `spice_task1/` with the netlists and ngspice logs of
task 1 (the netlists use `meas` only, so no raw files are written). `make_report.py` rewrites
`report.md` from `compare.json` and `run_info.json` of a results folder given as its argument:

```
python experiments/E4_1_independent/make_report.py results/E4_1_independent_check_<YYYYMMDD_HHMM>
```

Run of record: `results/E4_1_independent_check_20261113_1643/` (all three tasks PASS, 58 s).

## Files

| File | Content |
|---|---|
| `e41i_common.py` | paths; loading of `e4i_core.py` and `e4i_spice.py` by file path; tolerances; loaders of the E4.1 outputs; helpers |
| `task1_spice.py`, `task2_model.py`, `task3_decision.py` | the three tasks |
| `run_check.py` | driver (results folder, logging, hashes, provenance, `compare.json`, report) |
| `make_report.py` | `report.md` from `compare.json` |

## Implementation choices not fixed by the plan

- Rows simulated: the first case of each (design, time, node) group in the order of `cases.json`
  (6 cases), as the E4 check did for V1. In these six the LBUF bounds carry at most 1.1e-4 of S
  (they are deep inside their bounds or past the u-cut), so they test the LBUF chain only in its
  off state. I therefore also simulate, outside the pass flag, every case whose LBUF bounds carry
  at least half of S (`S_lower_LBUF / S >= 0.5` in `rows.csv`: cases 11, 15, 21, 25), with the same
  netlist, settings and threshold.
- Netlists: one row per netlist (the main run put 12 rows into each); element and node names my
  own; INV_S = pfet_01v8 0.42/4 from VDD (bulk at VDD) and nfet_01v8 2.0/0.15 to ground (E4 plan);
  per-device mismatch parameters `mm_vth0`, `mm_toxe`, `mm_voff`, `mm_nfactor` from the pool draws
  (`e4i_spice.pool_draws`; instance k uses row k - 1 of its group); wildcard pull-downs and the
  pull-downs of the t_s(N) netlist nominal (no mismatch parameters); PCM as a linear resistor 1/G;
  the model library included through a path relative to the netlist folder; `.spiceinit` as in the
  E4 check (`ngbehavior=hsa`).
- Transient: `.options reltol=1e-4 abstol=1e-15 vntol=1e-8`, maximum step 0.1 ns, stop time
  2 t_pred + 100 ns (the settings of the E4 check; the main run used the plan's 0.2 ns maximum step
  and 6 t_s(N)); crossing by `meas tran ... when v(ml)=0.9 fall=1` (7 significant digits). Variants
  in `task1_sensitivity.csv`: default tolerances and a 0.4 ns step for case 0; the six rows in one
  netlist with the plan's 0.2 ns step and 6 t_s(N). All three print the same crossing times.
- t_s(N): nominal pull-down at V_sw,nom of `instrument.npz`, the other 59 nominal pull-downs at
  0 V, same ML capacitor and settings, stop time 1.2 us.
- Tolerances. SPICE: 0.5 % relative for the crossing times and for t_s(N), as the E4 check.
  Model: the E4 check set no numeric model-side threshold; for the 48 6T2M cases it found every
  input that needs no root solve bit-identical and w, S within 4.0e-8 and 3.9e-8 (relative). I
  require (a) instance indices and bound sets identical, and x, conductance (relative), effective
  threshold, data-line voltage, u, a_hat and supplies within 1e-12; (b) w, S, the LBUF and UINV
  parts of S (divided by S) and t_c,pred within 1e-5 relative. (b) is the 6T2M maximum times the
  gain of the second inverter of the LBUF chain (about 20), about 1e-6, with a margin of 10; it is
  still 5000 times below the 0.05 in ln t of the gate.
- Model side: `e4i_core.py` as it is (bound inversion by 40 bisection steps, surface inversion
  and node solve by 64; I-V surface ln I linear along V_G, then I linear along V_DS; LBUF gate
  t_second(t_first(node)) by linear interpolation, clamped at +-60 mV; LBUF bounds beyond 0.45 LSB
  and UINV bounds beyond 3 LSB inside the bound take w of the instance's own pull-down at 0 V);
  SL_lo from the LBUF pool (V_X*,nom + D_eff), SL_hi from the UINV pool. t_c,pred uses t_s(N) of
  the run of record (413.384 ns as stored); variant C of task 3 uses my SPICE t_s(N).
- Diagnostics: d ln w / d node by a central difference of +-0.1 uV and the implied node-voltage
  difference (delta ln w) / slope where |slope| > 1 /V. Window classes of the gate comparison:
  u-cut; node outside the stored transfer window (|X - X*_k| > 60 mV or |W - W*_k| > 60 mV for
  LBUF, |Y - Y*_k| > 60 mV for UINV); otherwise in window. `bounds.csv` reports as `gate_model_V`
  the chain output before the u-cut (reproduced within 0.04 uV), while its `w_model` uses the
  0 V gate; task 3 compares both conventions with SPICE.
- Case selection (extra): S of every (sample, row) pair without a violated bound, candidates with
  0.3 <= S <= 3, stable sort by ln S, indices `np.round(np.linspace(0, n - 1, 8))`, duplicates
  removed; the pair counts are compared with the lines of the main run's `run.log`.
- Row decisions: model S >= 1 + (2N - 1) w0, simulation t_c,sim <= t_s(N); near tie |ln S| < 0.05.
