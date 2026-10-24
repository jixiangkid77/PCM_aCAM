# E6 independent check, part 1: own values (before any comparison)

- Run: 2026-10-24 09:36:13 to 2026-10-24 09:37:04 (JST); wall 51 s; ngspice: 86 runs, 46 s CPU, one process at a time
- Code: `experiments/E6_independent/run_check.py` (SHA-256 `bf845195bd1101445846c728df2a818d58a4995dfe0a90abc367936a71ee0639`)
- Inputs: plan `docs/E6_protocol.md`, config `experiments/E6_config.json` (SHA-256 `a89efd26159a7826d38181f949352beae63b79640e9ea1660c57f6133056f9f4`), models `third_party/sky130_fd_pr_subset_e3b/sky130_subset_tt.lib.spice`, `src/pcm_acam/trees.py` (dataset_bundle only). No main-run output was read.
- ngspice-42; .spiceinit `set ngbehavior=hsa`, `set ng_nomodcheck`; .temp 27; environment OMP_NUM_THREADS=1, OMP_WAIT_POLICY=passive (speed only; results bit-identical)
- Drift factor a(1 y) = (31,536,000 s / 20 s)^-0.049 = 0.496945710

## Item 0: nominal switching points

| Quantity | Value (V) | How found | Check |
|---|---|---|---|
| V_sw | 0.526221473 | 1 mV, then 1 µV DC sweep of the pull-down gate (V_DS = 1.8 V), linear interpolation of I_D = 280 nA | .op at root: I/280 nA - 1 = -5.3e-11 |
| V_Y* | 0.570321776 | INV_S output held at V_sw; 1 mV, then 1 µV sweep of the input; zero of the output-source current | free-output .op at root: V_out - V_sw = 3.3e-11 V; gain -22.63 |
| V_X* | 0.568420612 | two INV_S in series (pull-down on the output); 1 mV, then 1 µV sweep; V_out = V_sw | first INV_S held at V_Y*: 0.568420612 V (diff 6.9e-11 V); cascade gain 538 |

Resolution: the sweeps fix every node voltage of the device under test (except the free-output checks), so each point is an exact model evaluation; linear interpolation on the 1 µV grid; the stated residuals are the check.

## Item 1: settling times (transient SPICE, C_par = 0.5 fF)

Settling times (ns). `plan` = the plan's numerics (max step 20 ps), the value to compare with the main run; `half`, `quarter` = max step 10 ps, 5 ps; `extrap.` = (4·quarter − half)/3 (trapezoidal error ∝ h²).

| Case | Group | Design | Type | Node | θ (level) | x_prev | t_settle plan | half | quarter | extrap. | tail τ (ns) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| s00 | 1a | P1 | L6T | t0 | 0 (0) | 0 | **0.0000** | 0.0000 | 0.0000 | 0.0000 |  |
| s01 | 1a | P1 | L6T | t0 | 0 (0) | 1 | **2.2416** | 2.2417 | 2.2417 | 2.2418 | 0.3492 |
| s02 | 1a | P1 | L6T | dac_remap_1y | 0 (0) | 0 | **0.0000** | 0.0000 | 0.0000 | 0.0000 |  |
| s03 | 1a | P1 | L6T | dac_remap_1y | 0 (0) | 1 | **4.4325** | 4.4324 | 4.4324 | 4.4324 | 0.6977 |
| s04 | 1a | P3 | L6T | t0 | 0 (0) | 0 | **0.0000** | 0.0000 | 0.0000 | 0.0000 |  |
| s05 | 1a | P3 | L6T | t0 | 0 (0) | 1 | **4.3498** | 4.3502 | 4.3503 | 4.3503 | 0.4449 |
| s06 | 1a | P3 | L6T | dac_remap_1y | 0 (0) | 0 | **0.0000** | 0.0000 | 0.0000 | 0.0000 |  |
| s07 | 1a | P3 | L6T | dac_remap_1y | 0 (0) | 1 | **7.7829** | 7.7831 | 7.7831 | 7.7831 | 0.8780 |
| s08 | 1b | P1 | L6T | t0 | 1 (15) | 0 | **0.3766** | 0.3804 | 0.3812 | 0.3814 | 0.0403 |
| s09 | 1b | P1 | L6T | dac_remap_1y | 1 (15) | 0 | **0.6782** | 0.6802 | 0.6806 | 0.6808 | 0.0804 |
| s10 | 1b | P3 | L6T | t0 | 1 (15) | 0 | **0.8736** | 0.8755 | 0.8761 | 0.8762 | 0.1022 |
| s11 | 1b | P3 | L6T | dac_remap_1y | 1 (15) | 0 | **1.1129** | 1.1144 | 1.1147 | 1.1148 | 0.1224 |
| s12 | 1c | P1 | UINV | t0 | 0 (0) | 1 | **63.8261** | 63.8258 | 63.8258 | 63.8257 | 9.2513 |
| s13 | 1c | P1 | UINV | dac_remap_1y | 0 (0) | 1 | **114.8893** | 114.8890 | 114.8890 | 114.8889 | 16.6410 |
| s14 | 1c | P3 | UINV | t0 | 0 (0) | 1 | **76.5514** | 76.5513 | 76.5512 | 76.5512 | 11.2543 |
| s15 | 1c | P3 | UINV | dac_remap_1y | 0 (0) | 1 | **140.5996** | 140.5995 | 140.5994 | 140.5994 | 20.7032 |

Settings, data-line voltages, band and DC decision-node voltage of each case:

| Case | G (µS) | V_SL (V) | Map | V_DL(x_prev) (V) | V_DL(θ) (V) | V_DL(θ−0.01 LSB) (V) | V_DL(θ+0.01 LSB) (V) | Band low (V) | Band high (V) | Band (mV) | V_dec,DC at M(θ) (V) | (V_dec,DC(θ) − V_sw)/slope (LSB) | V_dec,DC at M(x_prev) (V) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s00 | 2.500000 | 0.726221 | M0 | 0.679581 | 0.679581 | 0.679304 | 0.679856 | 0.525358 | 0.527086 | 1.7276 | 0.526221 | -7.28e-11 | 0.526221 |
| s01 | 2.500000 | 0.726221 | M0 | 0.817929 | 0.679581 | 0.679304 | 0.679856 | 0.525358 | 0.527086 | 1.7276 | 0.526221 | -7.28e-11 | 0.046161 |
| s02 | 1.242364 | 0.726221 | Ma | 0.644958 | 0.644958 | 0.644705 | 0.645211 | 0.525365 | 0.527080 | 1.7149 | 0.526221 | -8.34e-11 | 0.526221 |
| s03 | 1.242364 | 0.726221 | Ma | 0.767112 | 0.644958 | 0.644705 | 0.645211 | 0.525365 | 0.527080 | 1.7149 | 0.526221 | -8.34e-11 | 0.044969 |
| s04 | 2.500000 | 0.726221 | M0 | 0.755081 | 0.755081 | 0.754483 | 0.755677 | 0.525169 | 0.527274 | 2.1043 | 0.526221 | -5.22e-12 | 0.526221 |
| s05 | 2.500000 | 0.726221 | M0 | 1.242315 | 0.755081 | 0.754483 | 0.755677 | 0.525169 | 0.527274 | 2.1043 | 0.526221 | -5.22e-12 | 0.110801 |
| s06 | 1.242364 | 0.726221 | Ma | 0.688308 | 0.688308 | 0.687871 | 0.688744 | 0.525172 | 0.527271 | 2.0992 | 0.526221 | -1.35e-11 | 0.526221 |
| s07 | 1.242364 | 0.726221 | Ma | 1.013628 | 0.688308 | 0.687871 | 0.688744 | 0.525172 | 0.527271 | 2.0992 | 0.526221 | -1.35e-11 | 0.084855 |
| s08 | 22.500000 | 0.726221 | M0 | 0.679581 | 0.817929 | 0.817882 | 0.817977 | 0.526120 | 0.526323 | 0.2030 | 0.526221 | -2.33e-10 | 0.699467 |
| s09 | 11.181278 | 0.726221 | Ma | 0.644958 | 0.767112 | 0.767073 | 0.767152 | 0.526122 | 0.526321 | 0.1982 | 0.526221 | -2.57e-10 | 0.699287 |
| s10 | 22.500000 | 0.726221 | M0 | 0.755081 | 1.242315 | 1.242063 | 1.242568 | 0.526113 | 0.526330 | 0.2161 | 0.526221 | -5.04e-13 | 0.703748 |
| s11 | 11.181278 | 0.726221 | Ma | 0.688308 | 1.013628 | 1.013476 | 1.013780 | 0.526106 | 0.526337 | 0.2320 | 0.526221 | -1.12e-11 | 0.703692 |
| s12 | 2.629528 | 0.770322 | M0 | 0.817929 | 0.679581 | 0.679304 | 0.679856 | 0.506765 | 0.546087 | 39.3211 | 0.526221 | 1.66e-11 | 1.799999 |
| s13 | 1.306733 | 0.770322 | Ma | 0.767112 | 0.644958 | 0.644705 | 0.645211 | 0.513876 | 0.553344 | 39.4685 | 0.533407 | 3.64e-03 | 1.799999 |
| s14 | 2.507420 | 0.770322 | M0 | 1.242315 | 0.755081 | 0.754483 | 0.755677 | 0.502734 | 0.550359 | 47.6258 | 0.526221 | 1.37e-11 | 1.799998 |
| s15 | 1.246051 | 0.770322 | Ma | 1.013628 | 0.688308 | 0.687871 | 0.688744 | 0.505268 | 0.552913 | 47.6456 | 0.528766 | 1.07e-03 | 1.799999 |

Numerics: `.options reltol=1e-6 vntol=1e-9 abstol=1e-15` (the plan's), default integration (trapezoidal), `tran 10p t_stop 0 20p`, t_stop 150 ns (L6T) and 600 ns (UINV); DL PWL M(x_prev) until 1 ns, linear to M(θ) by 1.1 ns; transient starts from the DC operating point at M(x_prev). Band = DC decision-node voltages at M(θ ± 0.01 LSB), same branch and node setting (.op, plan options; tight options `.options reltol=1e-9 vntol=1e-12 abstol=1e-18` change any DC point by at most 5.3e-15 V). t_settle = time of the final entry into the band after 1 ns (linear interpolation between the last outside sample and the next), minus 1 ns; 0 if never outside after 1 ns.

Convergence (largest change per group):

| Group | plan → half step | half → quarter step | Gear (plan step) vs plan |
|---|---|---|---|
| 1a | 0.47 ps (0.011 %) | 0.04 ps (0.001 %) | 1.07 ps (0.047 %) |
| 1b | 3.79 ps (0.995 %) | 0.76 ps (0.198 %) | 29.36 ps (7.794 %) |
| 1c | 0.25 ps (0.000 %) | 0.08 ps (0.000 %) | 0.93 ps (0.001 %) |

All cases inside the band at t_stop: True; largest |V(t_stop) − V_DC(θ)| = 3.0e-07 LSB (input-referred); band crossings after 1 ns per case: [0, 1] (0 = never outside, 1 = one final entry, no re-exit); the transient's initial operating point equals the DC point at M(x_prev) within 1.4e-14 V.

## Item 2: static power of one programmed bound (DC SPICE)

θ = 8/15 (level 8; the plan text says θ = 0.5, but levels are j/15). P = V_SL·I(SL source) + V_DD·I(inverter supply) (UINV only); pull-down (ML) current not counted; DL source draws no DC current.

| Design | Type | Node | x | G (µS) | V_SL (V) | V_DL (V) | V_div (V) | V_dec (V) | P_SL (µW) | P_inv (µW) | P (µW) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| P1 | L6T | t0 | 0.25 | 13.166667 | 0.726221 | 0.741497 | 0.603125 | 0.603125 | 1.1770353 | 0.0000000 | 1.1770353 |
| P1 | L6T | t0 | 0.75 | 13.166667 | 0.726221 | 0.798615 | 0.473037 | 0.473037 | 2.4209243 | 0.0000000 | 2.4209243 |
| P1 | L6T | dac_remap_1y | 0.25 | 6.543119 | 0.726221 | 0.700863 | 0.602419 | 0.602419 | 0.5882782 | 0.0000000 | 0.5882782 |
| P1 | L6T | dac_remap_1y | 0.75 | 6.543119 | 0.726221 | 0.750757 | 0.474488 | 0.474488 | 1.1961775 | 0.0000000 | 1.1961775 |
| P1 | L6T | supply_1y | 0.25 | 6.543119 | 0.928680 | 0.741497 | 0.665909 | 0.665909 | 1.5967185 | 0.0000000 | 1.5967185 |
| P1 | L6T | supply_1y | 0.75 | 6.543119 | 0.928680 | 0.798615 | 0.436673 | 0.436673 | 2.9896643 | 0.0000000 | 2.9896643 |
| P1 | UINV | t0 | 0.25 | 13.717291 | 0.770322 | 0.741497 | 0.647130 | 0.009836 | 1.3017327 | 0.4372205 | 1.7389532 |
| P1 | UINV | t0 | 0.75 | 13.717291 | 0.770322 | 0.798615 | 0.517219 | 1.680943 | 2.6744690 | 0.5261040 | 3.2005730 |
| P1 | UINV | dac_remap_1y | 0.25 | 6.816749 | 0.770322 | 0.700863 | 0.646053 | 0.010161 | 0.6525462 | 0.4406319 | 1.0931781 |
| P1 | UINV | dac_remap_1y | 0.75 | 6.816749 | 0.770322 | 0.750757 | 0.517458 | 1.680016 | 1.3278079 | 0.5288678 | 1.8566757 |
| P1 | UINV | supply_1y | 0.25 | 6.816749 | 0.972780 | 0.741497 | 0.710324 | 0.001794 | 1.7403987 | 0.2629597 | 2.0033584 |
| P1 | UINV | supply_1y | 0.75 | 6.816749 | 0.972780 | 0.798615 | 0.480375 | 1.760264 | 3.2652339 | 0.2234583 | 3.4886922 |
| P3 | L6T | t0 | 0.25 | 13.166667 | 0.726221 | 0.922287 | 0.611666 | 0.611666 | 1.0953657 | 0.0000000 | 1.0953657 |
| P3 | L6T | t0 | 0.75 | 13.166667 | 0.726221 | 1.146306 | 0.466369 | 0.466369 | 2.4846916 | 0.0000000 | 2.4846916 |
| P3 | L6T | dac_remap_1y | 0.25 | 6.543119 | 0.726221 | 0.804672 | 0.611698 | 0.611698 | 0.5441890 | 0.0000000 | 0.5441890 |
| P3 | L6T | dac_remap_1y | 0.75 | 6.543119 | 0.726221 | 0.953717 | 0.462177 | 0.462177 | 1.2546731 | 0.0000000 | 1.2546731 |
| P3 | L6T | supply_1y | 0.25 | 6.543119 | 0.928680 | 0.922287 | 0.697140 | 0.697140 | 1.4069421 | 0.0000000 | 1.4069421 |
| P3 | L6T | supply_1y | 0.75 | 6.543119 | 0.928680 | 1.146306 | 0.421037 | 0.421037 | 3.0846757 | 0.0000000 | 3.0846757 |
| P3 | UINV | t0 | 0.25 | 13.235450 | 0.770322 | 0.922287 | 0.656090 | 0.007557 | 1.1646603 | 0.4094153 | 1.5740756 |
| P3 | UINV | t0 | 0.75 | 13.235450 | 0.770322 | 1.146306 | 0.507358 | 1.712531 | 2.6810605 | 0.4219575 | 3.1030181 |
| P3 | UINV | dac_remap_1y | 0.25 | 6.577300 | 0.770322 | 0.804672 | 0.656107 | 0.007553 | 0.5786858 | 0.4093629 | 0.9880487 |
| P3 | UINV | dac_remap_1y | 0.75 | 6.577300 | 0.770322 | 0.953717 | 0.506347 | 1.715175 | 1.3374653 | 0.4123278 | 1.7497931 |
| P3 | UINV | supply_1y | 0.25 | 6.577300 | 0.972780 | 0.922287 | 0.741964 | 0.000832 | 1.4768256 | 0.1947868 | 1.6716125 |
| P3 | UINV | supply_1y | 0.75 | 6.577300 | 0.972780 | 1.146306 | 0.455640 | 1.780237 | 3.3088033 | 0.1218339 | 3.4306371 |

Numerics: .op with `.options reltol=1e-6 vntol=1e-9 abstol=1e-15`; rerun with `.options reltol=1e-9 vntol=1e-12 abstol=1e-18`: largest relative change of P 0.0e+00; I(SL source) vs G·(V_SL - V_div): largest relative difference 1.2e-15. The x = θ points are in values.json and power.csv (not part of the check).

## Item 3: counts (E4 forest, iris)

| Split | Rows | Features | Cells | PCM devices | 256 / (2 cells) | Programmed bounds (lower / upper) |
|---|---|---|---|---|---|---|
| 0 | 126 | 4 | 504 | 1008 | 0.253968 (25.397 %) | 385 (207 / 178) |
| 1 | 125 | 4 | 500 | 1000 | 0.256000 (25.600 %) | 378 (208 / 170) |
| 2 | 127 | 4 | 508 | 1016 | 0.251969 (25.197 %) | 404 (220 / 184) |
| mean | 126.0000 | 4 | 504.0000 | 1008.0000 | 0.253979 (mean of shares; 256 / mean devices = 0.253968) | 389.0000 |

Programmed bounds = entries of rows.cons_kind (one per programmed bound). For information (own 4-bit rounding of cons_thr): bounds at level 0 / 15 per split, lower: 0/0, 0/0, 2/0; upper: 0/0, 0/0, 2/0; rows without any bound: 0, 0, 0.

## Assumptions and interpretations

- Item 2 uses theta = 8/15 (level 8): the plan text says theta = 0.5, but the run only has levels j/15.
- 'Last exit from the band' is read as the end of the last excursion outside the band, i.e. the final entry into it: crossing between the last outside sample after 1 ns and the next sample.
- MOSFET ad, as, pd, ps, nrd, nrs, sa, sb, sd left at the subcircuit defaults (0), nf = 1, mult = 1; mismatch parameters at their default 0.
- L6T programming uses G(theta) exactly; I_div(M0(theta), V_sw)/D equals it up to the map residual (below 1e-11 relative, see dl_maps.csv). UINV: G_UINV(theta) = I_div(M0(theta), V_DS = V_Y*)/D from an .op; no clipping (all values are far below 25 uS).
- a(t) = (t / 20 s)^-0.049 with t = 31,536,000 s (as specified for E6; E4 used ((t + 20 s)/20 s), which changes a by 3e-8 relative).
- C_par = 0.5 fF from the divider node to ground in the transients only (no effect on DC).
- Each branch has its own sources: V_SL, the DL, the pull-down drain (1.8 V, precharged ML) and, for UINV, the INV_S supply (1.8 V, pMOS source and body). Divider and pull-down sources and bodies at 0.
- DC sweeps that define V_sw, V_Y*, V_X* and the DL map fix every node of the device under test, so each point is an exact model evaluation; roots by linear interpolation on a 1 uV grid, checked by an .op at the root.

