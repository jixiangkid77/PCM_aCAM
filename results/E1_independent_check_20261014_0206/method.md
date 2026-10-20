# E1 independent check: method

- Run: [2026-10-14 02:06] to [2026-10-14 02:23] JST. Folder `results/E1_independent_check_20261014_0206/`.
- Scripts: `/home/claude/E1_indep/` (`e1lib.py`, `run_e1_indep.py`, plus the diagnostics `tol_test.py`, `crosscheck_iv.py`, `bin_probe.py`). Netlists and ngspice logs are in `/home/claude/E1_indep/work/`.
- Inputs read: `docs/E1_protocol.md`, `experiments/E1_config.json`, and the model files in `third_party/sky130_fd_pr_subset/` (MANIFEST.json, the .lib header, the mismatch/corner files, and the subckt/model headers of the two pm3 files). I did not open anything in `src/`, `experiments/*.py`, the other `results/` folders, or the Project documents.
- Tools: ngspice-42 (batch mode, `-b -r` binary raw output; the default SPARSE solver was used), Python 3, numpy 2.5.3, scipy 1.18.1.

## What was computed
- **Scope.** Designs P1 (divider L = 0.15 µm) and P4 (divider L = 8 µm). Both use W_div = 0.42 µm, D = 0.2 V and the LVT pull-down 0.42/0.15. Mismatch setting is "pdk", instances 0..39, thresholds j ∈ {1, 5, 10, 14}, drift of 1 year, nodes none / dac_remap / supply / ideal.
- **Draws.** `rng = default_rng([20261014, 0])`, then `z_div = standard_normal((300,4))`, then `z_pd = standard_normal((300,4))`. Rows 0..39 go to mm_vth0, mm_toxe, mm_voff and mm_nfactor in that column order. For traceability: z_div[0] = (-0.470931, 0.461829, -0.810164, 1.860959) and z_pd[0] = (0.674129, -0.305051, 1.950562, -0.320597).
- **Netlists.** Each one does `.include` of `sky130_subset_tt.lib.spice` (which sets `.option scale=1.0u`), with `.temp 27`. The work folder holds a `.spiceinit` containing `set ngbehavior=hsa` and `set ng_nomodcheck`. Instances are written as `X.. d g 0 0 sky130_fd_pr__nfet_01v8[_lvt] w=0.42 l=.. mm_vth0=.. mm_toxe=.. mm_voff=.. mm_nfactor=..`, so source and bulk are at ground. Options are `reltol=1e-6 abstol=1e-15 vntol=1e-9`.
- **V_sw (own pull-down simulation).** Drain is held at 1.8 V by a source and the gate is swept. The coarse sweep runs from 0 to 1.8 V in 0.5 mV steps. The fine sweep adds a per-instance gate offset, covers ±0.5 mV in 0.005 mV steps, and finds the crossing of ln I_D with ln(280 nA) by linear interpolation. I_D is the current delivered by the drain source.
- **Bound (full branch only).** The circuit is: V_SL source, then a linear resistor 1/G, then node X, then the divider to ground with its gate on V_DL. A DC sweep of V_DL follows. The coarse sweep runs from 0 to 2.5 V in 0.5 mV steps. The fine sweep covers ±1 mV around the coarse crossing in 0.01 mV steps, using a per-branch gate-offset source so that many branches share one netlist. The bound is the first crossing of V(X) through V_sw, linearly interpolated. Before interpolation I check that V(X) is strictly non-increasing in V_DL.
  - Many branches, each with its own supply source, resistor and divider instance, are simulated in parallel in one netlist, with up to 200 per run.
  - The 2.5 V coarse range only lets root-finding iterates overshoot. Every reported bound is below 1.8 V (maximum 1.6728 V).
- **Nominal maps.** M0(x) = B_nom(G(x); V_SL,nom) and Ma(x) = B_nom(a·G(x); V_SL,nom), with G(x) = 2.5 + 20x µS. Each was simulated on 1241 x points (x from −0.12 to 1.12 in steps of 0.001), all with full branches. Both maps are strictly increasing. The values in `nominal.csv` at the thresholds come from separate direct simulations; the cubic spline through the grid agrees with them to within 5e-12 V.
- **Programming.** For each pair (i, j) I solved B_i(G; V_SL,nom) = M0(θ_j) by root finding in ln G. The first step uses the nominal slope dB/dlnG, later steps are secant steps safeguarded by bisection, and every evaluation is a full-branch simulation. The stopping rule is |B − target| ≤ 1e-9 V, which took at most 7 iterations.
  - `g_prog_uS` is the root before clipping.
  - When the root exceeds 25 µS, the pair has `clipped = 1` and the clipped value of 25 µS is the one used in every node.
- **Nodes.** a = (31536000/20)^(−0.049) = 0.496945710; V_SL,nom = V_sw,nom + 0.2; V_SL(1y) = V_sw,nom + 0.2/a.

  | Node | Bound simulated | Inverse map |
  |---|---|---|
  | none | B_i(a·G_c; V_SL,nom) | M0 |
  | dac_remap | B_i(a·G_c; V_SL,nom), the same bound as none | Ma |
  | supply | B_i(a·G_c; V_SL(1y)) | M0 |
  | ideal | B_i(G_c; V_SL,nom) | M0 |

  The ideal bound is a fresh simulation, not the bound taken from the root finder.
- **x_eff.** I inverted the cubic spline of the map with brentq. I then took one Newton step using a direct full-branch simulation of the nominal branch at G(x_eff), or at a·G(x_eff) for dac_remap. The Newton step was at most 7e-10 LSB, and the final residual is below 2e-15 V. err_lsb = 15·(x_eff − θ_j).

## Numerical accuracy (diagnostics in `diagnostics.json`, `crosscheck_iv.json`, `bin_probe.json`, `work/tol_test.json`)
- Changing the ngspice tolerances (defaults versus the options above versus reltol 1e-9) changes the bounds by at most 0.008 µV and V_sw by 0.
- Changing the fine step from 0.01 mV to 1 µV changes the bounds by at most 2e-5 µV.
- The coarse result alone (0.5 mV steps plus interpolation) differs from the fine result by at most 0.37 µV. For V_sw the coarse-versus-fine difference is at most 0.035 µV.
- Consistency with single-transistor operating points (a diagnostic only, not used for any number): at each reported bound, the divider (drain at V_sw, gate at the bound) carries the branch current G·(V_SL − V_sw) to within the equivalent of 0.001 µV in gate voltage. I_D(V_sw) equals 280 nA to within 1e-11 relative.
- **Expected accuracy.** bound_V and vsw_V are good to about 0.01 µV, and err_lsb to a few 1e-6 LSB. The ideal-node error on unclipped rows is at most 1e-7 LSB (maximum 9.5e-8).
- **Pipeline check with the nominal instance** (mm = 0 on both devices; reported only in `diagnostics.json`):
  - G_prog equals G(θ).
  - dac_remap, supply and ideal give 0 error (|e| < 1e-9 LSB).
  - none gives the analytic value 15·((a·G(θ) − 2.5 µS)/20 µS − θ), for example −1.44628 LSB at j = 1.

## Results (unclipped rows; RMS over the 40 × 4 grid minus clipped pairs)

| Design | n unclipped | none | dac_remap | supply | ideal |
|---|---|---|---|---|---|
| P1 | 145 (15 clipped) | 5.0177 | 0.7078 | 0.4880 | 0.0000 |
| P4 | 157 (3 clipped) | 5.2833 | 0.1465 | 0.4931 | 0.0000 |

All entries are RMS of err_lsb.

- Clipped pairs: in P1, 11 at j = 14 and 4 at j = 10; in P4, 3 at j = 14. The largest root was 48.2 µS (P1, instance 25, j = 14).
- V_sw: nominal 0.526221473 V. Across the 40 instances the mean is 0.524610 V and the SD is 20.65 mV, against a PDK slope prediction of 21.74 mV.
- LSB size (M0(1) − M0(0))/15: 9.2233 mV for P1 and 54.4777 mV for P4.

## Ambiguities and how I resolved them
1. **Clipped pairs.** All nodes, including ideal, use the clipped conductance G_c = 25 µS, and drift gives a·G_c. `g_prog_uS` is the unclipped root. Clipped rows stay in the CSV, but `clipped = 1` and they are excluded from the RMS above.
2. **Definition of I_D for V_sw.** I used the terminal current delivered by the 1.8 V drain source, which includes the about 1e-11 relative junction/gmin share.
3. **Layout of `nominal.csv`.** It is long format, one row per design × threshold. Besides the requested columns it also carries a, V_SL(1y), G(θ) and the LSB size.
4. **Ideal node.** The bound is simulated at G_c with V_SL,nom, which is the same as the config's "G → G/â".
5. **Mismatch hooks with no effect.** mm_nfactor is passed to both devices as specified but has no effect: nfet_01v8 has no nfactor hook (MANIFEST patch count 0), and nfactor_slope is 0 in both mismatch files. mm_voff has no effect on the LVT pull-down (voff_slope = 0.00). In effect the divider varies in vth0, toxe and voff, and the pull-down varies in vth0 and toxe (slope1 values in LVT bin 87).
6. **Model bins.** ngspice-42 selects LVT bin 87 for 0.42/0.15, nfet bin 161 for 0.42/0.15 and nfet bin 154 for 0.42/8. The divider sizes lie exactly on bin edges (W = 0.42 µm; L = 8 µm is the boundary between bin 155, 4–8 µm, and bin 154, 8–20 µm). Forcing the neighbouring bin moves the nominal bound as follows:
   - W edge: at most 0.09 µV for P1 and 0.26 µV for P4.
   - L edge (L = 7.999 µm, bin 155): up to 0.114 mV for P4, which is about 0.002 LSB.
   - Not a problem here, since bin selection is deterministic for an identical netlist syntax. A different simulator version that resolved the L = 8 µm edge the other way would shift P4 bounds by about 0.1 mV.

## What failed
- Nothing in the final run: no ngspice warnings or errors in any of the 161 logs, and no out-of-range x_eff or NaN.
- One diagnostic attempt did not do what I intended: nudging L to 7.9999999 µm did not switch bins (ngspice-42 evidently applies a tolerance at the edge), so the probe uses L = 7.999 µm.
- I made no comparison with the main run.
