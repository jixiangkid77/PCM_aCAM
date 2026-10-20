# PCM_aCAM

Code, simulation records and results of my study of drift compensation in analog content-addressable memories (aCAMs) that use phase change memory (PCM) for tree-based inference.

An aCAM evaluates a decision tree by storing each split threshold as a conductance in a voltage divider of a cell, and threshold-verify programming absorbs each cell's transistor mismatch into that conductance. PCM conductance drifts with time. An aCAM has no digital output that could be rescaled, so a global drift compensation has to act on an analog node: the data-line (input) voltages, called *input remap* here, or the supply of the dividers, called *divider supply*. The study measures which mismatch each node re-exposes as the PCM drifts, formulates first-order laws for the resulting bound errors, tests them as a design-time predictor against transistor-level Monte Carlo simulations (SkyWater sky130 device models in ngspice), and evaluates random-forest accuracy at system level. All results are simulations.

## Layout

```
src/pcm_acam/   simulation library
experiments/    frozen plans (E*_config.json), run, analysis and figure scripts,
                independent checks (E*_independent/, E0_*_independent_recompute.py)
results/        one folder per run, never overwritten
logs/           console logs of the main runs and of two independent-check reruns;
                freeze records (SHA-256) of E2-E4
figures/        figures of each step (PDF and PNG), made by experiments/E*_figures.py
third_party/    sky130 device-model subsets for ngspice (see THIRD_PARTY_NOTICES.md)
```

The scripts find the library, the model files and earlier runs through paths relative to the repository root, so the folders must stay together. Run every command from the repository root.

| module (`src/pcm_acam/`) | content |
|---|---|
| `pcm_model.py` | PCM programming noise, drift and read noise: NumPy transcription of the PCM model of AIHWKit |
| `trees.py` | datasets, random forests, conversion of root-to-leaf paths into CAM rows, quantization |
| `acam.py` | threshold mapping, device states, replica tracking, matching and voting |
| `cell.py` | behavioral 6T2M bound equation (Li et al. 2020, Eq. 1) with transistor mismatch, threshold or conductance verification, compensation nodes |
| `io_utils.py` | run folders, logging, hashes |
| `sky130_models.py` | builds the model subsets in `third_party/` (per-instance mismatch parameters) |
| `spice_iv.py` | ngspice runner, binary raw-file reader, I-V sweeps, full-branch DC sweeps |
| `branch_post.py` | bounds of a branch from its I-V curves, input maps, error summaries |
| `law_predict.py` | first-order laws as a predictor (sensitivities from ±1σ single-device sweeps) |
| `pcm_iv.py` | sinh-type PCM read characteristic, supply rules, extended supply-node predictor |
| `inverter.py` | upper-bound branch: inverter transfer curve, trip point, sensitivities |
| `ml_transient.py` | match-line transient: row netlists, crossing times, bisection |
| `row_model.py` | row-level match-line model: Monte Carlo pools of the branch types, per-instance leakage, row decisions and votes |

## Experiments

Plans and scripts are in `experiments/`, runs in `results/`. A run of record with two entries is a two-phase run (first phase, then the phase that uses it).

| step | question | plan | main script | runs of record | independent check |
|---|---|---|---|---|---|
| E0.1 | Behavioral: can PCM hold the thresholds of an aCAM, does drift break it, and do replica tracking or ratio thresholds rescue it? | `E0_1_config.json` | `E0_1_acam_magnitude.py` | `E0_1_20261012_1652` | `E0_1_independent_check_20261012_1708`, `E0_1_compare_independent_20261012_1717` |
| E0.2 | Behavioral: does global compensation re-expose the transistor mismatch absorbed by threshold verification, and at which node? | `E0_2_config.json` | `E0_2_mismatch_coupling.py` | `E0_2_20261013_1325` | `E0_2_independent_check_20261013_1338`, `E0_2_compare_independent_20261013_1341` |
| E1 | Transistor level, lower-bound branch with the PDK mismatch model: operating region of the divider transistor and the error of each node | `E1_config.json` | `E1_spice_bound.py` | `E1_20261014_0156` | `E1_independent_check_20261014_0206`, `E1_compare_independent_20261014_0224` |
| E2 | Prospective test of the first-order laws as a design-time predictor in 14 conditions not used to formulate them (geometries, ss/ff corners, −40/85 °C); before it, a calibration of the predictor on the E1 design points | `E2_config.json`, `E2cal_E1_config.json` | `E2_law_validation.py` | `E2_predict_20261014_0315` + `E2_20261014_0316`; calibration `E2cal_E1_predict_20261014_0303` + `E2cal_E1_20261014_0303` | `E2_independent_check_20261014_0335` |
| E3a | PCM read nonlinearity (sinh-type characteristic) at the two nodes, with a linear or a replica-feedback supply rule | `E3a_config.json` | `E3a_pcm_nonlinearity.py` | `E3a_predict_20261014_0442` + `E3a_20261014_0442` | `E3a_independent_check_20261014_0458` |
| E3b | Upper-bound branch (divider, inverter, pull-down) | `E3b_config.json` | `E3b_upper_branch.py` | `E3b_predict_20261014_0532` + `E3b_20261014_0532` | `E3b_independent_check_20261014_1441` |
| E3c | Transient match-line sensing and crosstalk from the other matched cells of a row (nominal devices) | `E3c_config.json` | `E3c_ml_transient.py` | `E3c_predict_20261014_1543` + `E3c_20261014_1549` | `E3c_independent_check_20261014_1628` |
| E4 | System level: random-forest accuracy with transistor-level bound errors, PCM noise and match-line crosstalk at both nodes; the 6T2M cell as published versus a lower-bound branch with a two-inverter gain stage | `E4_config.json` | `E4_forest_system.py` | `E4_spice_20261017_0602` + `E4_20261017_0618` | `E4_independent_check_20261017_0741` |

Most steps also have analysis and figure scripts (`E*_analysis.py`, `E*_figures.py`) that take a run folder; E0.1, E0.2 and E3c have no analysis script.

### Outcomes

Classes and gates are those of the frozen plans; the full records are in each run's `decision.json`. LSB is one 4-bit threshold step (1/15 of the normalized input range); pp is percentage points. For E1-E3b, errors are rms over 300 Monte Carlo instances and the 16 thresholds, one year after programming.

| step | outcome |
|---|---|
| E0.1 | GO. Random forests (16 trees, depth 6, 4-bit thresholds; five datasets, 3 splits × 5 device seeds) lose 18.4 pp of accuracy on average one day after programming without drift tracking, and 0.42 pp after 30 days with tracking by 64 replica cells, relative to an ideal 4-bit CAM. |
| E0.2 | NO_GO. With 20 mV threshold and 2 % current-factor mismatch and threshold verification, compensation by the DAC gain costs 0.70 pp more than ideal compensation after one year, below the 1.0 pp gate (supply node, for reference: 0.40 pp). |
| E1 | Design dependent. The supply node cancels the divider mismatch (below 1e-7 LSB) and exposes the pull-down offset (0.52-0.54 LSB). The input remap exposes the divider mismatch (0.72, 0.33, 0.10 and 0.03 LSB alone for divider lengths of 0.15, 1, 4 and 8 µm) and, for the long dividers, also part of the pull-down offset (0.06 and 0.15 LSB at 4 and 8 µm); with both mismatched it gives 0.73, 0.32, 0.12 and 0.15 LSB. Largest gap between the nodes 0.42 LSB. |
| E2 | 28/28 frozen predictions within max(15 %, 0.02 LSB) of the Monte Carlo result, largest deviation −12.4 %; predicted better node correct in all 12 clear cases (and in the 2 close ones); mismatch decomposition 42/42. |
| E3a | The input remap is immune to the read nonlinearity (largest change 9.3e-5 LSB). The extended supply law held in 173/173 usable cases (3 excluded by the frozen validity rule; largest deviation −14.8 %) and predicted the better node in 79/79 clear cases. Where the supply node is better with a linear PCM (short divider), the linear supply rule loses this advantage at chord-conductance ratios G(0.4 V)/G(0.2 V) of 1.005-1.06; with replica feedback the supply node keeps it over the whole tested range if the nonlinearity is the same at all conductance levels, and loses it at ratios between 1.06 and 1.54 if the nonlinearity scales with the level. |
| E3b | 10/10 predictions within tolerance (largest +7.7 %), better node 4/4, decomposition 15/15. The inverter divides the pull-down offset by its gain (16-23), so the switching point of the upper branch is set by the inverter itself. |
| E3c | For an isolated cell, transient sensing reproduces the quasi-static bound (48/48 within 0.02 LSB). The crosstalk law held in 98/100 cases; the supply node gave the smaller crosstalk shift in 28/28 cases (9-36 % of that with the input remap). |
| E4 | H2 holds: in the 6T2M cell as published, the one-year accuracy loss with the supply node is 2.20 and 1.21 pp lower than with the input remap (short and long divider). H3 holds: crosstalk costs 4.12 and 3.15 pp at t0. H1 does not hold: with a gain stage, the node with the smaller predicted cell error wins for the short divider (+0.74 pp, 95 % CI [+0.51, +0.98]), but the difference for the long divider is not resolved (−0.06 pp [−0.21, +0.08]). |

## Frozen plans and records

- Every step has a plan, `experiments/E*_config.json`, with its settings, checks and decision rules, fixed before the main run. Every main run records the SHA-256 of its plan and of the code it ran in `run_info.json` (E0.1 and E0.2: plan and main script; from E1 on also the library modules) and keeps the plan as `config_used.json` (re-serialized up to E3c, so its bytes differ from the plan file; identical for E4). `logs/*_freeze_sha256.txt` record the files frozen for E2-E4.
- The library was extended twice without changing earlier results: before E2, `spice_iv.py` gained a temperature option (netlists without it are identical to those of E1) and `sky130_models.py` gained process corners (a tt-only build is byte-identical to the E1 subset); before E3b, `sky130_models.py` gained an optional device list (default unchanged; the earlier subsets rebuild byte for byte). The SHA-256 recorded by E1 for both files, and by the E2 calibration, E2 and E3a runs for `sky130_models.py`, therefore differ from the files here; all other hashes recorded by the runs of record match.
- E2, E3a, E3b and E3c run in two phases: the prediction phase (`--phase predict`) writes `predictions.json`, whose SHA-256 is recorded before the Monte Carlo or transient phase (`--phase mc` or `--phase sim`) reads it. E4 first stores the transistor-level instrument (`--phase spice`) and then evaluates the forests (`--phase forest`).
- Before a freeze, only smoke runs (code checks with small sizes), nominal feasibility scans and, for E2, the calibration on the in-sample E1 design points were run. Two smoke runs changed a model before the freeze: the λ_b factor of the crosstalk law (E3c) and the per-instance leakage of the row model (E4). Smoke runs also changed checks or settings before the freeze (for example the C4 tolerance of E1, the x grid of E2, check C4 of E3b and check T1 of E3c); the NOTE file of each superseded smoke run says what changed. The E4 plan also records how its row model reproduced the E3c data before the freeze and that the number of device seeds was chosen from the run time of a smoke run.
- Two main runs were repeated without a change of the decision rules: `E0_1_20261012_1650` stopped at check 2 before any result was computed (the check was corrected in plan 1.1), and `E1_20261014_0151` stopped at a secondary design point whose nominal input map left the 1.8 V sweep range (code fix).
- Some comments say "pre-registered" or "not pre-registered". This means part of the frozen plan or not; the plans were fixed and hashed before the runs, not registered on a public platform.
- Plans, scripts, freeze records and some notes refer to written protocols and result records in `docs/`. These working documents are not part of this repository; the plans carry the settings, checks and decision rules. Two independent checks hash protocol files and stop with a missing-file error without them: `E3b_independent/e3b_independent_check.py` at the start (after creating its empty results folder), and `E0_2_independent_recompute.py` at the end, after its CSV files and `gates.json` are written.
- Independent checks: for every step, a separate re-implementation written from the written protocol, the plan file and a short task description, without reading the main code, recomputed selected cases and compared them with the run of record. The E4 check reused the forest construction (`trees.py`) and the noise draws (`make_draws` in `acam.py`). The checks of E2-E4 compare with the runs of record through paths fixed in their scripts. Each `E*_independent/` folder has a README with its method.
- The scripts of the E0.1 and E0.2 checks in `experiments/` differ from the copies that ran only in their file paths (each says so in a comment), so the script SHA-256 in those checks' `run_info.json` differ; the E1 and E2 checks were rerun after the same change (`_rerun` folders, same results).

## Results folders

- Main runs and comparisons are named `<step>_<YYYYMMDD_HHMM>` after the start time in Japan Standard Time and never reuse a folder (a suffix `_b`, `_c`, ... is added). The independent checks use the same pattern; most of them stop if the folder exists (the E2 check waits for the next minute, the E4 check adds a suffix), and the E1 check writes into a folder given on its command line.
- `_predict_`, `_spice_`: first phase of a two-phase run. `_smoke`: smoke run. `_independent_check_`, `_compare_independent_`: independent checks. `_rerun`: an independent check repeated after its paths were made relative.
- Typical files: `run_info.json` (environment, SHA-256 of plan, code and inputs), `config_used.json`, `run.log`, `checks.json`, `decision.json`, `summary.csv`, `arrays_*.npz` (per-instance values), `per_eval.csv` (one row per evaluation), `predictions.json`, `instrument.npz` (E4).
- Smoke runs and the folders with a `NOTE_superseded.txt` or `NOTE_crashed.txt` (which says why) are kept as records; the decisions come only from the runs of record. `E0_1_20261012_1650` explains itself at the end of its `run.log`.
- Not included: the ngspice binary output (`*.raw`, 28 MB) of task 3 of the two E4 independent checks (`spice_task3/`), whose netlists and logs are included and whose parsed values are in the CSV and JSON files; and the scratch folders `logs/*_independent_work/` of the E1-E3c checks (netlists, logs and raw files of single solver jobs), which some included files mention. Rerunning a check writes them again (the E4 check into its new results folder).

## Reproducing

Python with numpy, scipy, scikit-learn and matplotlib (`requirements.txt`). The runs of record used Python 3.13.16, numpy 2.5.3, scipy 1.18.1 and scikit-learn 1.9.1 on Linux, and the figures were made with matplotlib 3.11.2; other versions may change the random forests and therefore the numbers. E1-E4 need ngspice on the PATH (the runs used ngspice-42); run one ngspice job at a time. Times below are those of the runs of record on a 2-core machine.

```
pip install -r requirements.txt

python experiments/E0_1_acam_magnitude.py                    # about 5 min
python experiments/E0_1_independent_recompute.py
python experiments/E0_2_mismatch_coupling.py                 # about 7 min
python experiments/E0_2_independent_recompute.py             # needs docs/E0_*_protocol.md at the end (see above)

python experiments/E1_spice_bound.py                         # about 5 min
mkdir results/E1_independent_check_repeat
python experiments/E1_independent/run_e1_indep.py results/E1_independent_check_repeat

python experiments/E2_law_validation.py --config experiments/E2cal_E1_config.json --phase predict
python experiments/E2_law_validation.py --config experiments/E2cal_E1_config.json --phase mc --predictions results/E2cal_E1_predict_20261014_0303/predictions.json
python experiments/E2_law_validation.py --phase predict
python experiments/E2_law_validation.py --phase mc --predictions results/E2_predict_20261014_0315/predictions.json   # about 7 min
python experiments/E2_independent/run_check.py

python experiments/E3a_pcm_nonlinearity.py --phase predict
python experiments/E3a_pcm_nonlinearity.py --phase mc --predictions results/E3a_predict_20261014_0442/predictions.json
python experiments/E3a_independent/e3a_independent_check.py
python experiments/E3b_upper_branch.py --phase predict
python experiments/E3b_upper_branch.py --phase mc --predictions results/E3b_predict_20261014_0532/predictions.json
python experiments/E3b_independent/e3b_independent_check.py  # needs docs/E3b_protocol.md (see above)
python experiments/E3c_ml_transient.py --phase predict                # about 6 min
python experiments/E3c_ml_transient.py --phase sim --predictions results/E3c_predict_20261014_1543/predictions.json   # about 15 min
python experiments/E3c_independent/e3c_independent_check.py           # about 5 min

python experiments/E4_forest_system.py --phase spice                  # about 15 min
python experiments/E4_forest_system.py --phase forest --instrument results/E4_spice_20261017_0602   # about 36 min with 2 processes
python experiments/E4_independent/run_check.py                        # about 3.5 min
```

The second phase can be run on the stored first phase of the run of record, as above, or on a new one. `run_e1_indep.py` writes into the existing folder given as its argument, so give it a new, empty one. Comparisons with the independent checks write new folders:

```
python experiments/E0_1_compare_independent.py results/E0_1_20261012_1652 results/E0_1_independent_check_20261012_1708
python experiments/E0_2_compare_independent.py results/E0_2_20261013_1325 results/E0_2_independent_check_20261013_1338
python experiments/E1_compare_independent.py results/E1_20261014_0156 results/E1_independent_check_20261014_0206
```

The analysis scripts (`E1_analysis.py`, `E1_analysis_current_mismatch.py`, `E2_analysis.py`, `E3a_analysis.py`, `E3b_analysis.py`, `E4_analysis.py`) take a run folder and write their tables into it. For the runs of record these tables are already stored; all but `E4_analysis.py` refuse to overwrite them. The figure scripts take the run folder, read these tables where needed (E1, E3a, E3b), and write into `figures/`, replacing the stored figures:

```
python experiments/E0_1_figures.py results/E0_1_20261012_1652
python experiments/E0_2_figures.py results/E0_2_20261013_1325
python experiments/E1_figures.py results/E1_20261014_0156
python experiments/E2_figures.py results/E2_20261014_0316
python experiments/E3a_figures.py results/E3a_20261014_0442
python experiments/E3b_figures.py results/E3b_20261014_0532
python experiments/E3c_figures.py results/E3c_20261014_1549
python experiments/E4_figures.py results/E4_20261017_0618
```

The model subsets in `third_party/` can be rebuilt from the SkyWater repository (only `LICENSE`, `models/parameters/lod.spice` and `cells/{nfet_01v8,nfet_01v8_lvt,pfet_01v8}/` are read); from the repository root, with `src/` on the Python path (on Windows, set `PYTHONPATH` to `src` instead of the prefix):

```
git clone https://github.com/google/skywater-pdk-libs-sky130_fd_pr ../skywater-pdk-libs-sky130_fd_pr
git -C ../skywater-pdk-libs-sky130_fd_pr checkout f62031a1be9aefe902d6d54cddd6f59b57627436
PYTHONPATH=src python -m pcm_acam.sky130_models ../skywater-pdk-libs-sky130_fd_pr third_party/sky130_fd_pr_subset
PYTHONPATH=src python -m pcm_acam.sky130_models ../skywater-pdk-libs-sky130_fd_pr third_party/sky130_fd_pr_subset_corners tt ss ff
PYTHONPATH=src python -m pcm_acam.sky130_models ../skywater-pdk-libs-sky130_fd_pr third_party/sky130_fd_pr_subset_e3b --devices nfet_01v8,nfet_01v8_lvt,pfet_01v8
```

This reproduces the three subsets byte for byte; `MANIFEST.json` in each lists the source commit and the SHA-256 of the original and of every written file. The 8 µm divider of E1 lies on a model bin edge (4-8 and 8-20 µm); ngspice-42 uses the 8-20 µm bin, and the other choice moves the nominal bounds by at most 0.11 mV, about 0.002 LSB (`results/E1_independent_check_20261014_0206/bin_probe.json`).

## Data

The five datasets are loaded at run time: iris, wine, breast cancer (Wisconsin, diagnostic) and digits as shipped with scikit-learn, and a synthetic set from `sklearn.datasets.make_classification` (parameters in the plans and in `src/pcm_acam/trees.py`). No data file is included.

## What is not here

- No measured device or chip data. In E0.1, E0.2 and E4 the PCM follows the AIHWKit PCM model (programming noise, drift with a state-dependent exponent, read noise); E1-E3c use a uniform drift factor (exponent 0.049) with ideal tracking. In SPICE the PCM is a linear conductance, or a sinh-type element in E3a.
- Transistors are the sky130 models, with the PDK's mismatch model in E1-E3b and E4 (E3c uses nominal devices); there is no layout and no parasitic extraction, and the match line is a lumped capacitance.
- Temperature enters only through the transistor models (E2: −40 and 85 °C); the PCM model has no temperature dependence.
- The PCM read nonlinearity is studied in E3a only; E4 uses a linear read characteristic.
- The written protocols and reports of the steps are working documents and are not included.

## License

My code and records are under the MIT License (`LICENSE`). `third_party/` holds modified subsets of the SkyWater sky130 device models under the Apache License 2.0, and `src/pcm_acam/pcm_model.py` transcribes formulas and constants of IBM AIHWKit (MIT License); see `THIRD_PARTY_NOTICES.md`.

## References

The plans name the sources of individual parameters; the main ones:

- C. Li et al., "Analog content-addressable memories with memristors," *Nat. Commun.*, vol. 11, 1638, 2020, doi: [10.1038/s41467-020-15254-4](https://doi.org/10.1038/s41467-020-15254-4).
- G. Pedretti et al., "Tree-based machine learning performed in-memory with memristive analog CAM," *Nat. Commun.*, vol. 12, 5806, 2021, doi: [10.1038/s41467-021-25873-0](https://doi.org/10.1038/s41467-021-25873-0).
- S. R. Nandakumar et al., "Phase-change memory models for deep learning training and inference," in *Proc. IEEE ICECS*, 2019, pp. 727-730, doi: [10.1109/ICECS46596.2019.8964852](https://doi.org/10.1109/ICECS46596.2019.8964852).
- M. J. Rasch et al., "A flexible and fast PyTorch toolkit for simulating training and inference on analog crossbar arrays," in *Proc. IEEE AICAS*, 2021, doi: [10.1109/AICAS51828.2021.9458494](https://doi.org/10.1109/AICAS51828.2021.9458494); AIHWKit: https://github.com/IBM/aihwkit
- SkyWater sky130 primitive device models: https://github.com/google/skywater-pdk-libs-sky130_fd_pr (commit f62031a1be9aefe902d6d54cddd6f59b57627436)
- ngspice: https://ngspice.sourceforge.io/
