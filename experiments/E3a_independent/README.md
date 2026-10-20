# E3a independent check (protocol section 9)

`e3a_independent_check.py` re-implements two parts of E3a from the written protocol only. It does the predictor (Task 1) and a full-branch Monte Carlo spot check (Task 2) for designs P1 and S1. It then compares both with the main run.

- Sources read: `docs/E3a_protocol.md` (sections 2, 3, 4, 9), `docs/E2_protocol.md` (section 2), `experiments/E3a_config.json`, `experiments/E2_config.json`, the SPICE models in `third_party/sky130_fd_pr_subset_corners/`, and the main-run outputs `results/E3a_predict_20261014_0442/predictions.json` and `results/E3a_20261014_0442/arrays_<did>.npz`.
- Not read: anything under `src/` and any other experiment script.
- Run: `python experiments/E3a_independent/e3a_independent_check.py`. The repository root is derived from the script location.
- Outputs:
  - New folder `results/E3a_independent_check_<YYYYMMDD_HHMM>/` (JST; the script refuses to overwrite). It holds `summary.json`, `task1_predictor_compare.csv`, `task2_error_compare.csv`, `task2_m0_compare.csv` and `task2_vsw_compare.csv`.
  - ngspice netlists and logs go to `logs/E3a_independent_work/<same name>/`.
- ngspice (ngspice-42) runs one job at a time. Every job runs in its own directory with a `.spiceinit` (`set ngbehavior=hsa`, `set ng_nomodcheck`). Each job uses tt and `.temp 27`.

## Method

Common elements:
- Switching voltage V_sw: the pull-down (nfet_01v8_lvt 0.42/0.15, V_DS = 1.8 V) conducts 100 nA·W/L = 280 nA.
- Mismatch draws: `default_rng([20261014, 3])`. `z_div` (300×4) is drawn first, then `z_pd` (300×4). The columns are vth0, toxe, voff, nfactor. They are passed as `mm_*` instance parameters.
- Drift factor: a = (31536000/20)^(−0.049) = 0.4969457103.
- Thresholds: θ_j = j/15 for j = 0…15. G(θ) = 2.5 + 20θ µS and ΔG = 20 µS.

Solving for a gate voltage at a target current is done in two steps. First, a coarse sweep from 0 to 1.8 V in 0.5 mV steps on a shared gate brackets the crossing. Second, a fine sweep in 1 µV steps covers a 1.5 mV window per device, with each gate offset by its own series source. ln I is interpolated linearly between the two fine points around the target. This method gives V_sw (nominal, ±1σ per parameter, and all 300 instances), and the nominal map M0(θ) = B_nom(G(θ)), which is the gate voltage at which the nominal divider carries G·D at V_DS = V_sw,nom.

**Task 1 (predictor, protocol section 3 with the E2 section 2.3 sensitivities)**

The sensitivities are computed as follows:
- ∂V_sw/∂z_p comes from central differences at ±1σ, and σ_δ² = Σ(∂V_sw/∂z_p)².
- s_p(I) = −∂ln I_D/∂z_p and λ(I) = −∂ln I_D/∂V_DS (V_DS ± 5 mV) use central differences of single-device `.op` currents. These are evaluated at the exact nominal gate voltage V_G(I), obtained by the inversion above at V_DS = V_sw,nom.
- They are evaluated for I = G·D, a·I and ρ0·I.

The laws:
- dac_remap: (G/ΔG)²{Σ_p[s_p(aI) − s_p(I)]² + [λ(aI) − λ(I)]²σ_δ²}.
- Supply nodes: (G/ΔG)²{E_δ[(ρ(δ) − 1 + ρ0Δλδ)²] + ρ0²ΣΔs_p²}, where ρ(δ) = a·s(V_t − δ; V0)/s(D − δ; V0).
  - supply_lin: V_t = D/a.
  - supply_rep: V_t = V* with a·s(V*; V0_ref) = s(D; V0_ref), because V0(G_rep = 12.5 µS) = V0_ref.
- E_δ uses the trapezoid rule on [−6σ_δ, 6σ_δ] with 4001 points. Points with δ ≥ 0.95D are dropped and the weights are renormalized by their sum. For P1 and S1 no point is dropped.
- The rms is 15·√(mean over the 16 θ).
- The systematic rms is 15·√(mean[(G/ΔG)(ρ0 − 1)]²). It is reported for information only.

**Task 2 (full-branch spot check, protocol sections 2, 4, 9)**

For each instance k = 0…19, each θ in {1, 5, 10, 14}/15, each grid point and each node:
- The divider current at gate M0(θ) and V_DS = V_sw,k comes from a single-device `.op`.
- Programming: A_k = I_prog/s(D − δ_k; V0(G(θ))), with no clipping.
- The full branch is: V_SL source → B-source `i = a*A_k*V0*sinh(v(sl,x)/V0)` (or `a*A_k*v(sl,x)` for the linear PCM) → node X → divider nMOS with drain X, gate DL and source/bulk 0.
- V_SL depends on the node:
  - dac_remap: V_sw,nom + D.
  - supply_lin: V_sw,nom + D/a.
  - supply_rep: V_sw,nom + V*.
- V_DL is swept (coarse 0.5 mV, then fine 1 µV with per-branch offsets). The bound is the V_DL at which V_X = V_sw,k, from linear interpolation of V_X between adjacent fine points. No I–V curve is inverted for a bound.
- I_0(bound) (the nominal divider at V_DS = V_sw,nom) is evaluated exactly by `.op`.
- x_eff = (I_0/(a·D) − 2.5 µS)/20 µS for dac_remap (map Ma), and (I_0/D − 2.5 µS)/20 µS for the supply nodes (map M0). The error is 15(x_eff − θ).

Self-check: a zero-mismatch cell (k = −1) runs through the same path. The dac_remap error must be 0 and the supply error must equal 15(G/ΔG)(ρ0 − 1). Both hold to below 1e-9 LSB.

## Interpretation choices

1. A cell's V0 follows from its target level, V0 = V0_ref·(G(θ)/12.5 µS)^β, as the config and task state. It does not follow from the programmed chord conductance A_k·s(D)/D. Protocol section 2 by itself ("state identified by G = I(D)/D") would also allow the second reading.
2. The supply_rep replica is nominal, with V* computed analytically (perfect tracking).
3. The B-source carries the whole drifted characteristic (a·A_k). No pull-down is placed on node X, because its gate draws no DC current (igcmod = igbmod = 0).
4. Currents are drain-terminal currents (−i(V_drain)).
5. Tolerances are `.options reltol=1e-7 vntol=1e-10 abstol=1e-16`. They were tested against the ngspice defaults: bounds changed by < 1e-9 V.
6. mm_nfactor has no effect in this model subset (nfactor_slope = 0; nfet_01v8 is not patched for it), so its sensitivities are exactly 0. The LVT voff slope is also 0.
7. s_p and λ are taken at exact gate voltages with `.op`, not by interpolating 0.5 mV curves. Agreement with the main run (~1e-7 relative) shows that this choice does not matter.
