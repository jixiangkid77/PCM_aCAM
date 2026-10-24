# E6 independent check: comparison with the run of record

- Main run of record: `results/E6_20261024_1032`; files read: settle.csv, power_curves.npz, energy_dl_ml.csv, overhead.csv, checks.json, run.log (nothing else)
- Independent values: `results/E6_independent_check_20261024_0936/values.json` (phase 1, 09:37 JST) and `extra_lbuf.json` (1d, written 10:48 JST before any main-run file was opened)
- Created 2026-10-24 10:52:53 JST by `experiments/E6_independent/compare.py` (SHA-256 `6df0b7f5e1505504003b87ae411302238f4cb12665d5cfdd2d9d1abaffbe4804`)
- Main-run checks (checks.json): C1 pass, C1b pass, C2 pass, C3 pass, C4 pass, C5 pass

## Summary

| Item | Status | Criterion | Result | Largest difference |
|---|---|---|---|---|
| 1a settling (frozen plan) | frozen plan | ≤ 2 % or ≤ 0.1 ns | PASS (8) | 0.166 ps; rel 0.0038 %; band ≤ 3.1e-06 LSB, v_dc ≤ 2.9e-06 LSB |
| 1b settling, L6T θ = 1 (additional) | additional | ≤ 2 % or ≤ 0.1 ns | PASS (4) | 0.694 ps; rel 0.1839 %; band ≤ 1.8e-05 LSB, v_dc ≤ 1.4e-05 LSB |
| 1c settling, UINV θ = 0 (additional) | additional | ≤ 2 % or ≤ 0.1 ns | PASS (4) | 4.055 ps; rel 0.0035 %; band ≤ 6.6e-06 LSB, v_dc ≤ 7.6e-06 LSB |
| 1d settling, LBUF θ = 1 (additional) | additional | ≤ 2 % or ≤ 0.1 ns | PASS (4) | 31.491 ps; rel 0.0101 %; band ≤ 3.5e-05 LSB, v_dc ≤ 3.5e-05 LSB |
| 2 static power (frozen plan) | frozen plan | ≤ 0.5 % | PASS (24) | rel 0.00913 % |
| 3 iris split 0 counts (frozen plan) | frozen plan | identical | PASS | rows 126/126, features 4/4, cells 504/504, 256/(2 cells) 0.253968/0.253968 |
| 3 means over splits (additional) | additional | identical | PASS | see below |

## Item 1: settling times (ns; mine at the plan's numerics; main = settle.csv, c_par_f = 5e-16)

| Group | Case | Design | Type | Node | Level | x_prev | Mine | Main | Diff (ps) | Rel | Pass | Δband lo/hi (µV) | Δv_dc (µV) | Mine converged |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1a | s00 | P1 | L6T | t0 | 0 | 0 | 0.0000 | 0.0000 | 0.000 | 0.0000 % | yes | +0.180 / +0.213 | +0.107 | 0.0000 |
| 1a | s01 | P1 | L6T | t0 | 0 | 1 | 2.2416 | 2.2416 | 0.030 | 0.0013 % | yes | +0.180 / +0.213 | +0.107 | 2.2418 |
| 1a | s02 | P1 | L6T | dac_remap_1y | 0 | 0 | 0.0000 | 0.0000 | 0.000 | 0.0000 % | yes | +0.215 / +0.211 | +0.044 | 0.0000 |
| 1a | s03 | P1 | L6T | dac_remap_1y | 0 | 1 | 4.4325 | 4.4323 | 0.166 | 0.0038 % | yes | +0.215 / +0.211 | +0.044 | 4.4324 |
| 1a | s04 | P3 | L6T | t0 | 0 | 0 | 0.0000 | 0.0000 | 0.000 | 0.0000 % | yes | +0.191 / -0.002 | +0.099 | 0.0000 |
| 1a | s05 | P3 | L6T | t0 | 0 | 1 | 4.3498 | 4.3498 | -0.030 | 0.0007 % | yes | +0.191 / -0.002 | +0.099 | 4.3503 |
| 1a | s06 | P3 | L6T | dac_remap_1y | 0 | 0 | 0.0000 | 0.0000 | 0.000 | 0.0000 % | yes | +0.329 / +0.242 | +0.308 | 0.0000 |
| 1a | s07 | P3 | L6T | dac_remap_1y | 0 | 1 | 7.7829 | 7.7829 | -0.000 | 0.0000 % | yes | +0.329 / +0.242 | +0.308 | 7.7831 |
| 1b | s08 | P1 | L6T | t0 | 15 | 0 | 0.3766 | 0.3773 | -0.694 | 0.1839 % | yes | +0.011 / +0.147 | +0.088 | 0.3814 |
| 1b | s09 | P1 | L6T | dac_remap_1y | 15 | 0 | 0.6782 | 0.6783 | -0.158 | 0.0233 % | yes | +0.179 / +0.091 | +0.141 | 0.6808 |
| 1b | s10 | P3 | L6T | t0 | 15 | 0 | 0.8736 | 0.8739 | -0.280 | 0.0320 % | yes | -0.020 / -0.021 | -0.006 | 0.8762 |
| 1b | s11 | P3 | L6T | dac_remap_1y | 15 | 0 | 1.1129 | 1.1133 | -0.480 | 0.0431 % | yes | +0.021 / -0.024 | +0.009 | 1.1148 |
| 1c | s12 | P1 | UINV | t0 | 0 | 1 | 63.8261 | 63.8248 | 1.217 | 0.0019 % | yes | +10.941 / +12.220 | +13.613 | 63.8257 |
| 1c | s13 | P1 | UINV | dac_remap_1y | 0 | 1 | 114.8893 | 114.8852 | 4.055 | 0.0035 % | yes | +10.967 / +11.373 | +15.058 | 114.8889 |
| 1c | s14 | P3 | UINV | t0 | 0 | 1 | 76.5514 | 76.5499 | 1.473 | 0.0019 % | yes | +15.715 / +12.118 | +13.861 | 76.5512 |
| 1c | s15 | P3 | UINV | dac_remap_1y | 0 | 1 | 140.5996 | 140.5986 | 1.022 | 0.0007 % | yes | +10.330 / +8.889 | +9.116 | 140.5994 |
| 1d | d0 | P1 | LBUF | t0 | 15 | 0 | 305.9976 | 305.9817 | 15.914 | 0.0052 % | yes | -78.858 / -129.913 | -98.273 | 305.9981 |
| 1d | d1 | P1 | LBUF | dac_remap_1y | 15 | 0 | 310.2315 | 310.2630 | -31.491 | 0.0101 % | yes | -146.862 / -110.615 | -126.681 | 310.2320 |
| 1d | d2 | P3 | LBUF | t0 | 15 | 0 | 301.8458 | 301.8698 | -24.083 | 0.0080 % | yes | -186.004 / -211.849 | -208.932 | 301.8460 |
| 1d | d3 | P3 | LBUF | dac_remap_1y | 15 | 0 | 303.4172 | 303.4205 | -3.361 | 0.0011 % | yes | -172.629 / -220.582 | -194.048 | 303.4174 |

Δ = mine − main. 'Mine converged' = extrapolated from the 10 ps and 5 ps maximum-step runs (trapezoidal); the comparison uses the plan's 20 ps value.

## 1d additional values (LBUF, GS cell set, θ = 1, x_prev = 0)

| Case | Design | Node | G (µS) | V_SL (V) | t_settle ±0.01 LSB (ns) | t_settle ±0.1 LSB (ns) | τ mid node (ns) | τ decision node (ns) | τ X (ns) |
|---|---|---|---|---|---|---|---|---|---|
| d0 | P1 | t0 | 22.500000 | 0.768421 | 305.9976 | 171.5039 | 53.06 / 51.22 | 51.13 / 51.03 | 28.08 / 87.11 |
| d1 | P1 | dac_remap_1y | 11.181278 | 0.768421 | 310.2315 | 175.1953 | 53.47 / 51.68 | 51.57 / 51.49 | 27.62 / 95.91 |
| d2 | P3 | t0 | 22.500000 | 0.768421 | 301.8458 | 164.7560 | 53.31 / 51.28 | 51.18 / 51.08 | 28.11 / 87.13 |
| d3 | P3 | dac_remap_1y | 11.181278 | 0.768421 | 303.4172 | 163.6041 | 53.99 / 51.87 | 51.74 / 51.65 | 27.74 / 96.06 |

τ: fit of ln|v − v_DC(θ)| where the node's input-referred distance is 0.003–0.03 LSB (last part of the approach) / 0.0003–0.003 LSB (tail). The approach is set by one slow pole of about 51 ns at the first INV_S output (it drives the second INV_S input); the decision node follows it, X is fast and only dragged through the first INV_S's gate-drain capacitance.

## Item 2: static power (µW), row 8 (θ = 8/15), x = 0.25 (index 250) and 0.75 (index 750)

| Design | Type | Node | x | Mine | Main | Rel diff | Mine on the main's 1 mV DL grid | Rel diff to that |
|---|---|---|---|---|---|---|---|---|
| P1 | L6T | t0 | 0.25 | 1.1770353 | 1.1770565 | 0.00180 % | 1.1770564 | 0.000008 % |
| P1 | L6T | t0 | 0.75 | 2.4209243 | 2.4209456 | 0.00088 % | 2.4209435 | 0.000091 % |
| P1 | L6T | dac_remap_1y | 0.25 | 0.5882782 | 0.5882852 | 0.00120 % | 0.5882846 | 0.000114 % |
| P1 | L6T | dac_remap_1y | 0.75 | 1.1961775 | 1.1961888 | 0.00095 % | 1.1961873 | 0.000124 % |
| P1 | L6T | supply_1y | 0.25 | 1.5967185 | 1.5967376 | 0.00120 % | 1.5967375 | 0.000006 % |
| P1 | L6T | supply_1y | 0.75 | 2.9896643 | 2.9896772 | 0.00043 % | 2.9896750 | 0.000076 % |
| P1 | UINV | t0 | 0.25 | 1.7389532 | 1.7389881 | 0.00201 % | 1.7389883 | 0.000010 % |
| P1 | UINV | t0 | 0.75 | 3.2005730 | 3.2007584 | 0.00579 % | 3.2007458 | 0.000395 % |
| P1 | UINV | dac_remap_1y | 0.25 | 1.0931781 | 1.0931923 | 0.00130 % | 1.0931924 | 0.000009 % |
| P1 | UINV | dac_remap_1y | 0.75 | 1.8566757 | 1.8568450 | 0.00912 % | 1.8568363 | 0.000466 % |
| P1 | UINV | supply_1y | 0.25 | 2.0033584 | 2.0034055 | 0.00235 % | 2.0034048 | 0.000036 % |
| P1 | UINV | supply_1y | 0.75 | 3.4886922 | 3.4890108 | 0.00913 % | 3.4890018 | 0.000258 % |
| P3 | L6T | t0 | 0.25 | 1.0953657 | 1.0953673 | 0.00015 % | 1.0953668 | 0.000046 % |
| P3 | L6T | t0 | 0.75 | 2.4846916 | 2.4846905 | 0.00004 % | 2.4846900 | 0.000021 % |
| P3 | L6T | dac_remap_1y | 0.25 | 0.5441890 | 0.5441909 | 0.00034 % | 0.5441904 | 0.000084 % |
| P3 | L6T | dac_remap_1y | 0.75 | 1.2546731 | 1.2546740 | 0.00007 % | 1.2546735 | 0.000039 % |
| P3 | L6T | supply_1y | 0.25 | 1.4069421 | 1.4069441 | 0.00015 % | 1.4069435 | 0.000045 % |
| P3 | L6T | supply_1y | 0.75 | 3.0846757 | 3.0846725 | 0.00010 % | 3.0846720 | 0.000017 % |
| P3 | UINV | t0 | 0.25 | 1.5740756 | 1.5740771 | 0.00010 % | 1.5740776 | 0.000031 % |
| P3 | UINV | t0 | 0.75 | 3.1030181 | 3.1030376 | 0.00063 % | 3.1030276 | 0.000323 % |
| P3 | UINV | dac_remap_1y | 0.25 | 0.9880487 | 0.9880511 | 0.00024 % | 0.9880521 | 0.000098 % |
| P3 | UINV | dac_remap_1y | 0.75 | 1.7497931 | 1.7498260 | 0.00188 % | 1.7498179 | 0.000459 % |
| P3 | UINV | supply_1y | 0.25 | 1.6716125 | 1.6716166 | 0.00025 % | 1.6716161 | 0.000030 % |
| P3 | UINV | supply_1y | 0.75 | 3.4306371 | 3.4306496 | 0.00036 % | 3.4306439 | 0.000168 % |

Explanation aid (not the check): the main run sweeps the data line in 1 mV steps and interpolates linearly at V_DL = M(x). Doing the same with my own DC power at the two bracketing grid points reduces the largest difference from 0.00913 % to 0.000466 %: the differences are mainly the interpolation bias of a convex P(V_DL) (main always higher), not a model difference; the rest is at the level of the sub-µV differences in the switching points and data-line maps.

## Item 3: counts

| Quantity (iris, mean over splits 0–2) | Mine | Main (overhead.csv) | Identical |
|---|---|---|---|
| rows_mean | 126.0 | 126.0 | yes |
| features | 4.0 | 4.0 | yes |
| cells | 504.0 | 504.0 | yes |
| pcm_devices | 1008.0 | 1008.0 | yes |
| programmed_bounds_mean | 389.0 | 389.0 | yes |
| replica_64_share (64 / mean devices) | 0.06349206349206349 | 0.06349206349206349 | yes |
| replica_256_share (256 / mean devices) | 0.25396825396825395 | 0.25396825396825395 | yes |

overhead.csv's shares are 256 (64) / mean PCM devices; the mean of the per-split shares would be 0.253979 (0.063495).

## Switching points (run.log prints 6 decimals)

| Quantity | Main (printed) | Mine | Mine − printed (µV) | Implied range of mine − main (µV) | Within print resolution |
|---|---|---|---|---|---|
| V_sw | 0.526222 | 0.526221473 | -0.527 | -1.027 to -0.027 | no |
| V_Y_star | 0.570323 | 0.570321776 | -1.224 | -1.724 to -0.724 | no |
| V_X_star | 0.568421 | 0.568420612 | -0.388 | -0.888 to +0.112 | yes |

