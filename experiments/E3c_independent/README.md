# E3c independent check (design P1)

Independent re-implementation of part of E3c (match-line transient sensing and row crosstalk,
lower-bound branch) for the post-run check of `docs/E3c_protocol.md` section 8.

## Sources used

Only: `docs/E3c_protocol.md` (sections 2, 3, 4, 8), `experiments/E3c_config.json`
(`ml`, `circuit`, `crosstalk`, `laws`, `nodes`, `drift`), `docs/E2_protocol.md` section 2
(lambda = -d ln I_D / d V_DS, central difference at V_DS +- 5 mV), the SPICE models in
`third_party/sky130_fd_pr_subset/`, and the main-run outputs that are compared
(`results/E3c_predict_20261014_1543/predictions.json`,
`results/E3c_20261014_1549/t3_crosstalk.csv`, `results/E3c_20261014_1549/t2_isolated.csv`).
Nothing under `src/` and no other experiment script was read or imported.

## Run

```
python3 experiments/E3c_independent/e3c_independent_check.py
```

The repository root is derived from the script location. ngspice-42 is called in batch mode,
one job at a time; every job gets its own folder with `.spiceinit`
(`set ngbehavior=hsa`, `set ng_nomodcheck`), netlist and log under
`logs/E3c_independent_work/run_<JST stamp>/` (transient raw files are deleted after parsing).
Outputs go to a new folder `results/E3c_independent_check_<YYYYMMDD_HHMM>/` (JST); the script
refuses to overwrite an existing folder. Run time about 6 min (2 cores).

## Method (P1: divider nfet_01v8 0.42/0.15 um, pull-downs nfet_01v8_lvt 0.42/0.15 um, D = 0.2 V, G(x) = 2.5 + 20x uS)

All devices are nominal (mm_* = 0), tt, 27 C, source and body grounded, no ad/as/pd/ps
(subcircuit defaults), every pull-down instantiated individually (no m multiplier).

1. **V_sw**: gate voltage at which the LVT pull-down conducts 100 nA * W/L = 280 nA at V_DS = 1.8 V.
   0.5 mV DC sweep for a start value, then Newton on ln I_D with fixed-bias operating points
   (finite difference 1 uV) until the update is < 1e-12 V (current error ~1e-15 relative).
2. **t_s(N)**: one ML node with C_SA + N * 1.0 fF (66 fF for N = 16, 51 fF for N = 1), 2N
   pull-downs on the ML (CUT gate = ideal source at V_sw; all other 2N - 1 gates at 0 V).
   `.ic v(ml)=1.8` without UIC (ML forced during the operating point, released at t = 0).
   `tran 0.1n <2.5 t_s> 0 0.2n` (stop time from a first run with 2 us stop; the time grid is
   limited by the 0.2 ns maximum step, so the crossing time is independent of the stop time).
   Crossing time = first falling crossing of 0.9 V, linear interpolation between time points
   on the full-precision ASCII waveform (same rule as `.meas ... when v(ml)=0.9 fall=1`, whose
   stored value is rounded to 7 digits). Checked before the run: identical crossing time for
   one copy alone and inside a 10-copy netlist (relative 6e-14) and with reltol 1e-5 (0.0).
3. **Divider map and lambda**: gate voltage at which the nominal divider at V_DS = V_sw conducts a
   target current (same sweep + Newton procedure). lambda_b = -(ln I(V_sw + 5 mV) - ln I(V_sw - 5 mV)) / 10 mV
   at that gate voltage, for I = G(theta) D and a G(theta) D, theta = 0, 1/3, 2/3, 1.
4. **Leaker V_X,j** (k4d6: x_j = 0.9; k15d7.5: x_j = 1.0; theta_j = 0.5): single branch
   V_SL -> PCM 1/G_j -> X -> divider (gate V_DL,j) -> ground. G_j = G(0.5) (t0) or a G(0.5) (1 y);
   V_SL = V_sw + D (t0, dac_1y) or V_sw + D/a (supply_1y); V_DL,j from the nominal map for
   G(x_j) D (t0, supply) or a G(x_j) D (dac). The ngspice operating point is then polished by
   Newton on G_j (V_SL - X) = I_div(V_DL,j, X) with fixed-bias device evaluations (both values reported;
   they agree to < 1 nV). a = (31536000/20)^(-0.049).
5. **delta_l and false mismatch** (pull-down-only row, gates as ideal sources): false mismatch if the
   row with the CUT gate at 0 V and the k leaker gates at V_X,j crosses 0.9 V before t_s(16) (also
   reported with the leakers as full branches). Otherwise the CUT gate voltage giving a crossing exactly
   at t_s(16) is found by bisection, with leakers and without (isolated 16-cell row);
   delta_l = difference.
6. **Predicted shift**: 15 (G/dG) [(1 - kappa delta_l/D) exp(lambda_b delta_l) - 1], G = G(theta),
   dG = 20 uS, kappa = 1 (t0, dac_1y) or a (supply_1y), lambda_b at I (t0, supply) or a I (dac).
7. **Full-row transient**: CUT = full branch (V_SL -> 1/(G_f G(theta)) -> X -> divider, gate on its own
   DL source; X drives the CUT pull-down), k leakers = full branches as in 4 (individual branches),
   remaining 15 - k lower and 16 upper pull-downs at gate 0, 66 fF, same precharge and transient.
   Bisection on the CUT input V_DL for a crossing exactly at t_s(16). x_eff from the exact inverse
   of the nominal map: G(x_eff) * mapf * D = I_0, I_0 = nominal divider current at
   (V_DL,bound, V_DS = V_sw), mapf = a (dac_1y) else 1; error = 15 (x_eff - theta);
   shift = error(with leakers) - error(isolated 16-cell row).

**Bisection**: all cases of a stage share each ngspice run (independent copies in one netlist);
each run places 3 equally spaced interior points in every open bracket (bracket shrinks 4x per
run) until the bracket is <= 0.1 uV; the midpoint is reported (the linear interpolation inside
the final bracket is also stored). Initial brackets: CUT gate V_sw +- 50 mV (isolated),
[V_sw - 300 mV, V_sw + 20 mV] (with leakers); CUT input V_DL,nom(theta) +- 50 mV (isolated),
[V_DL,nom - 50 mV, V_DL,nom + 400 mV] (with leakers); both ends are verified to bracket the root.

## Thresholds (protocol section 8 / task)

t_s(16) relative 1e-4; V_X,j 0.01 mV; delta_l 0.01 mV; false-mismatch flags identical;
predicted shift (theta = 0, 1) 0.01 LSB; full-row shift and isolated error 0.01 LSB.
V_sw, V_sw,tr, lambda and t_s(1) are compared for information only.

## Outputs

`summary.json` (pass/fail per item, maxima, all computed values, SHA-256 of inputs and of the
compared main-run files), `compare_ts.csv`, `compare_leakers.csv` (V_X, false mismatch,
crossing time with the CUT off, delta_l), `compare_pred_shift.csv` (all four thetas; theta = 0, 1 gated),
`compare_fullrow_shift.csv`, `compare_isolated_t2.csv`, `compare_info.csv`, `ngspice_runs.json`, `run.log`.

## Result of run 20261014_1628 (`results/E3c_independent_check_20261014_1628/`)

All items pass (42 ngspice runs, 321 s).

| Item | Max deviation from the main run | Where | Threshold |
|---|---|---|---|
| t_s(16) | 3.2e-7 relative (331.0561057 vs 331.056 ns; main value stored with 6 digits) | - | 1e-4 |
| V_X,j | 0.36 uV (main lower in all 6 cases, 0.08-0.36 uV) | k4d6 / supply_1y | 0.01 mV |
| delta_l | 1.41 uV (other cases <= 0.19 uV) | k15d7.5 / t0 | 0.01 mV |
| false mismatch | identical: only k15d7.5 / dac_1y (leakers alone cross at 320.63 ns < 331.06 ns) | - | identical |
| predicted shift | 1.7e-4 LSB (theta = 0, 1 and all four thetas) | k15d7.5 / t0 / theta 1 | 0.01 LSB |
| full-row shift | 1.7e-4 LSB | k15d7.5 / t0 / theta 1 | 0.01 LSB |
| isolated error (t3 and t2, N = 16) | 8e-7 LSB | dac_1y / theta 1 | 0.01 LSB |

Informational: V_sw 0.526221473 V (main +0.034 uV); lambda differs by <= 4.2e-7 /V; main nominal-map
gate voltages (t2 `bound_qs_V`) are 0.02-0.07 uV higher than the exact inversion, which with the
divider gain (about 3) accounts for the V_X offsets. Supplementary scratch test
(`logs/E3c_independent_work/sens_driver.py`, `sens_k15t0/`): with the leaker gates at the main run's
V_X, the k15d7.5 / t0 switching voltage moves by +1.91 uV (15 leakers amplify the V_X offset about 13x),
leaving 0.56 uV to the main value; at this point dt/dV = -5.6e-7 s/V, so a +-5e-13 s (6-digit)
crossing-time resolution corresponds to +-0.9 uV.
