# E2 independent check (protocol section 9)

Independent re-implementation of the two checks in `docs/E2_protocol.md` section 9, for conditions
G2, C2, T3, G6. It was written from `docs/E2_protocol.md` (sections 2, 4, 9), `experiments/E2_config.json`
and the sky130 subset model files only. No file under `src/` and no main-run `*.py` was read. The only
main-run outputs read are `results/E2_predict_20261014_0315/predictions.json` and
`results/E2_20261014_0316/arrays_<cid>.npz`, and from the arrays only the keys `theta`, `b_target`,
`vsw_mc`, `err_pdk_dac_remap_1y` and `err_pdk_supply_1y`.

## Files

| file | role |
|---|---|
| `ind_spice.py` | ngspice batch runner (`.spiceinit` with `ngbehavior=hsa`, `ng_nomodcheck`), binary raw-file reader, netlist helpers |
| `ind_core.py` | simulation primitives: single-device I-V sweeps, exact-voltage operating points, gate-voltage root finding, full-branch bounds |
| `task1_predictor.py` | Task 1: design-time predictor of section 2 |
| `task2_branch_mc.py` | Task 2: full-branch Monte Carlo spot check, section 4 |
| `run_check.py` | runs both tasks, compares with the main run, writes `results/E2_independent_check_<JST stamp>/` |
| `extra_supply_law_form.py` | supplementary diagnostic of the supply law's functional form (outside pass/fail) |
| `extra_tolerance_robustness.py` | supplementary: Task 2 rerun with ngspice default tolerances |

Run: `python run_check.py --conditions G2,C2,T3,G6 --n-inst 40`, then the two `extra_*.py` scripts with
the results folder as their argument. ngspice jobs run one at a time.

## Method

**Common setup.**
- Models: `.include .../sky130_subset_<corner>.lib.spice` and `.temp <T>`. Each transistor gets only
  `w`, `l` (µm) and `mm_vth0/toxe/voff/nfactor`. Source and body are grounded.
- Newton tolerances are tightened (`reltol=1e-7 vntol=1e-10 abstol=1e-16`). A rerun of Task 2 for C2 with
  ngspice's defaults changed nothing beyond 1.2e-13 LSB.

**Root finding (V_sw, nominal gate voltage for a given current).**
- Stage 1: gate sweep 0–1.8 V in 0.5 mV steps, then a local cubic spline of ln I_D and `brentq`.
- Stage 2: a per-device offset sweep (gate = s + coarse root, ±40 µV, 0.1 µV step), then cubic again.
- Verification: an exact-voltage `.op` at the root reproduces the target current to |Δln I| ≤ 1e-14.
- The main run interpolates ln I linearly on the 0.5 mV grid. That accounts for its +30 to +55 nV offset in
  V_sw and its +0.00 to +0.15 µV offset in M0.

**Task 1 (section 2).**
- Pull-down at V_DS = 1.8 V, target 100 nA·W/L (280 nA for both pd_min and pd_4x). It is evaluated at
  nominal and at mm_p = ±1 for each of the four parameters (9 roots).
- dV_sw/dz_p uses central differences, and σ_δ = sqrt(Σ_p (dV_sw/dz_p)²).
- Divider at V_DS = V_sw,nom: the nominal gate voltage V_G is found for I = G(θ_j)·D and a·G(θ_j)·D,
  j = 0..15, with G = 2.5 + 20θ µS.
- At each V_G, exact-voltage operating points give:
  - s_p = −[ln I(+1) − ln I(−1)]/2
  - λ = −[ln I(V_DS+5 mV) − ln I(V_DS−5 mV)]/0.01 V

  These are evaluated directly, not interpolated from perturbed curves.
- E[ln² r] with r = (D − aδ)/(D − δ): trapezoid rule over 4001 points in ±6σ_δ. Points with δ ≥ 0.95 D
  are dropped and the result is normalised by the trapezoid integral of the kept Gaussian weights. No point
  was dropped for these four conditions. Adaptive quadrature gives the same value within 4.1e-12 relative.
- rms = 15·sqrt(mean over the 16 θ of E[Δx²]), using the per-θ expressions and the settings exactly as in
  section 2.2. a = (31536000/20)^−0.049 = 0.4969457.

**Task 2 (section 4, pdk setting, instances 0–39, θ ∈ {1,5,10,14}/15, 1 year).**
- Draws: `default_rng([20261014, 2])`, z_div = standard_normal((300,4)) first, then z_pd. Columns are
  (vth0, toxe, voff, nfactor).
- V_sw,k comes from the root finder above, with drain at 1.8 V. V_SL,nom = V_sw,nom + D.
- Full branch: PCM resistor 1/G from SL to X, divider drain at X, gate on DL. The bound is the V_DL at which
  V_X equals the threshold. It is found with a shared DL sweep (0–1.8 V, 0.5 mV, local cubic on V_X),
  then a per-branch offset sweep (±100 µV, 0.2 µV step). No I-V inversion is used for any bound.
  - Diagnostic only: a single device at (V_G = bound, V_DS = threshold) carries G·(V_SL − threshold) to
    |Δln I| ≤ 2e-14.
- M0(θ): nominal full branch with G(θ), SL = V_SL,nom and threshold V_sw,nom. It is my own value and is not
  taken from `b_target`.
- Programming: G_k(θ) = I_div,k(V_G = M0(θ), V_DS = V_sw,k)/(V_SL,nom − V_sw,k), from exact-voltage
  operating points, without clipping.
- Drifted bounds use the instance full branch with conductance a·G_k and threshold V_sw,k:
  - dac_remap: SL = V_SL,nom
  - supply: SL = V_sw,nom + D/a
- x_eff: both maps are nominal bounds at V_SL,nom, so B_nom(G') = B is equivalent to
  G'·D = I_nom(B, V_DS = V_sw,nom). I_nom is evaluated at the exact voltages, which needs no x grid.
  - dac_remap (Ma(x) = B_nom(a·G(x))): x_eff = (G'/a − 2.5 µS)/20 µS
  - supply (M0(x) = B_nom(G(x))): x_eff = (G' − 2.5 µS)/20 µS
  - error = 15·(x_eff − θ)
- Closure check: a nominal full branch at conductance G' reproduces B within 1.1e-15 V.
- A zero-mismatch instance is appended as a self-check. Its errors are ≤ 4e-14 LSB.

## Interpretation choices

1. Roots and sensitivities are computed at exact voltages rather than by linear interpolation of ln I on the
   0.5 mV grid. Both implement the protocol; the effect is ≤ 0.15 µV, which is ≤ 5e-6 LSB.
2. x_eff is the exact inverse of the monotone nominal map, not a lookup on the x ∈ [−0.12, 1.6] grid.
3. "Renormalised" in the E[ln² r] quadrature means normalising by the trapezoid integral of the kept
   weights. This makes no difference here because nothing is dropped.
4. The tolerance for V_sw,nom (0.05 mV) and for σ_δ (1 % relative) follows the caller's instructions.
   The dV_sw/dz_p values are reported as information only.

## Supplementary finding (not part of the frozen pass/fail)

With the protocol's error definition (G(x) linear in x, M0 unchanged), consider the limit where the
instance's relative current deficit ε does not change between I and rI. In that limit the supply node gives
G(x_eff) = r·G exactly, so Δx = (G/ΔG)(r − 1). The frozen law uses (G/ΔG)·ln r, which is smaller by about
(r − 1)²/2.

`extra_supply_law_form.py` quantifies this on the main run's own MC arrays and on this check's full-branch
data. See `supplementary_supply_law_form.json`:
- For G2 and T3, the (r − 1) form fits the per-pair supply errors with about 3.5× smaller residual rms.
- With Gaussian δ, the (r − 1) form would raise the predicted supply rms by 2.7 % (G2, C2, T3) and by
  1.2 % (G6).
- For G6, the λ-related change of ε (a second-order term in δ) partly offsets this, and ln r fits slightly
  better.

These effects are far inside the 15 % tolerance of the decision rule.

## Delivered copy

The only change after the original run (`results/E2_independent_check_20261014_0335/`) is that the
repository paths (`ROOT` in `run_check.py` and the two `extra_*.py` scripts, `MODEL_DIR` in
`ind_spice.py`) are now derived from the location of the scripts instead of being absolute paths.
A rerun with this copy (`results/E2_independent_check_20261014_0350_rerun/`) reproduced the four
comparison CSV files byte for byte and the `independent_<cid>.npz` arrays exactly. Run from any
directory: `python experiments/E2_independent/run_check.py`.
The SHA-256 values of the original scripts are recorded in
`results/E2_independent_check_20261014_0335/run_info.json` and `supplementary_run_info.json`; the
supplementary scripts (`extra_*.py`) were not rerun after the path change.
