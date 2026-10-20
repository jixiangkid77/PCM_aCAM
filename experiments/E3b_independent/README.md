# E3b independent check (protocol section 8)

Independent re-implementation of two parts of E3b (upper-bound branch), written only from
`docs/E3b_protocol.md` (sections 2, 3, 4, 8), `experiments/E3b_config.json`,
`docs/E2_protocol.md` section 2 (predictor definitions) and the SPICE model files in
`third_party/sky130_fd_pr_subset_e3b/`. Nothing under `src/` and no `experiments/*.py` file was
read or imported. The main run is used only through its output files
(`results/E3b_predict_20261014_0532/predictions.json`, `results/E3b_20261014_0532/arrays_<cid>.npz`).

Run from anywhere (the repository root is derived from the script location):

    python experiments/E3b_independent/e3b_independent_check.py            # U1, U4, 20 instances
    python experiments/E3b_independent/e3b_independent_check.py --n-inst 2 --test-out <new dir>   # program test

Outputs go to a new folder `results/E3b_independent_check_<YYYYMMDD_HHMM>/` (Japan time; the
script refuses to reuse an existing folder); ngspice netlists, logs and raw tables go to
`logs/E3b_independent_work/run_<YYYYMMDD_HHMM>/`. ngspice jobs run strictly one at a time.

## Task 1: predictor (U1, U4)

| Quantity | How it is computed here |
|---|---|
| V_sw,pd | pull-down nfet_01v8_lvt 0.42/0.15, V_DS = 1.8 V, gate voltage where I_D = 100 nA x W/L = 280 nA (crossing of ln I_D) |
| V_Y*,nom | nominal unloaded inverter (pMOS drain Z, gate Y, source/body 1.8 V; nMOS drain Z, gate Y, source/body 0): input Y where Z = V_sw,pd,nom |
| inverter gain | -dZ/dY at V_Y*,nom: quadratic fit to the 1 uV-step stage-2 sweep over about +-10 uV |
| 12 sensitivities | +-1 sigma central differences (z = +-1 in one of mm_vth0, mm_toxe, mm_voff, mm_nfactor): inverter nMOS and pMOS perturbed, output target V_sw,pd,nom; pull-down perturbed: V_Y* of the nominal inverter at the perturbed pull-down's V_sw,pd |
| sigma_delta_Y | square root of the sum of the 12 squared sensitivities |
| s_p(I), lambda(I) | divider nfet_01v8 at V_DS = V_Y*,nom. The nominal gate voltage of every current I = G D and a G D (16 theta x 2) is found by root finding on the nominal device; then the nominal device, each parameter at +-1 sigma, and V_DS +-5 mV are evaluated at exactly that gate voltage (operating points, no curve interpolation). s_p = -[ln I(+1) - ln I(-1)]/2, lambda = -[ln I(V_DS+5 mV) - ln I(V_DS-5 mV)]/0.01 V |
| dac_remap (pdk) | E[dx^2] = (G/dG)^2 {sum_p [s_p(aI) - s_p(I)]^2 + [lambda(aI) - lambda(I)]^2 sigma_delta_Y^2} |
| supply (pdk) | E[dx^2] = (G/dG)^2 E[(r-1)^2], r = (D - a delta)/(D - delta), delta ~ N(0, sigma_delta_Y^2); trapezoid rule, 4001 points on +-6 sigma, points with delta >= 0.95 D dropped, density renormalised by its own trapezoid integral |
| rms | 15 sqrt(mean over theta_j = j/15, j = 0..15, of E[dx^2]); G = 2.5 + 20 theta uS, dG = 20 uS, a = (31536000/20)^-0.049 |
| div_only, y_only | for information only: div_only dac_remap = sum_p term, supply = 0; y_only dac_remap = lambda term, supply = pdk supply |

Thresholds: rms relative <= 1 % or absolute <= 0.002 LSB; V_Y*,nom <= 0.05 mV; sigma_delta_Y <= 1 % relative.
The other quantities (V_sw,pd,nom, gain, the 12 sensitivities, the other four prediction keys,
the nominal bound at 30 uS) are reported for information.

## Task 2: full-branch Monte Carlo spot check (U1, U4; pdk; first 20 instances; theta = 1, 5, 10, 14 /15; 1 year)

1. Draws: `numpy.random.default_rng([20261014, 4])`, then `standard_normal((300, 4))` for the
   divider, the pull-down, the inverter nMOS and the inverter pMOS (columns vth0, toxe, voff,
   nfactor), the same draws for both conditions.
2. V_sw,pd,k of the instance pull-down (as above); V_Y*,k = input of the instance inverter
   (nMOS and pMOS mismatch) where its output equals V_sw,pd,k.
3. M0(theta): nominal full upper branch (PCM resistor 1/G(theta) from V_SL,nom = V_Y*,nom + D to Y,
   nominal divider, nominal inverter), V_DL where Z = V_sw,pd,nom; all 16 theta (the four checked
   theta are the pass/fail set) and G = 30 uS (information).
4. Threshold-verify programming: G_k(theta) = I_div,k(V_G = M0(theta), V_DS = V_Y*,k) /
   (V_SL,nom - V_Y*,k), evaluated as an operating point of the instance divider; no clipping.
5. Full upper branch per (instance, theta, node): PCM resistor 1/(a G_k) from SL_hi to Y, the
   instance divider from Y to ground (gate DL, body 0), the instance inverter at Y (output Z,
   no load); sweep V_DL; bound = V_DL at which Z = V_sw,pd,k. dac_remap: V_SL = V_SL,nom;
   supply: V_SL = V_Y*,nom + D/a. No I-V inversion is used for the bound. A t = 0 run (G_k,
   V_SL,nom) is a self-check: its bound must equal M0(theta).
6. Error = 15 (x_eff - theta) with the exact inverse of the nominal map:
   dac_remap G(x_eff) a D = I_0(b), supply G(x_eff) D = I_0(b), where I_0(b) is the nominal
   divider current at V_G = b, V_DS = V_Y*,nom (operating point, no interpolation).

Thresholds: error 0.01 LSB per pair; V_Y*,k vs `vy_mc[k]` and M0 vs `b_target` 0.05 mV.
V_sw,pd,k vs `vsw_pd_mc[k]` is reported for information.

## Numerical choices

- Root finding (every crossing): stage 1 sweeps the input 0..1.8 V in 1 mV steps with every
  circuit of a netlist on one swept node and requires exactly one crossing; stage 2 gives every
  circuit its own series offset source (input = sweep + stage-1 estimate) and sweeps +-1.5 mV in
  1 uV steps; the crossing is the linear interpolation between the two bracketing stage-2 points.
  Linear interpolation over 1 uV contributes far less than 1 nV.
- Currents compared with a target (pull-down, divider) are compared in ln |I|.
- ngspice-42 with `.spiceinit` (`set ngbehavior=hsa`, `set ng_nomodcheck`), `.temp 27`, default
  tolerances and gmin, `option numdgt=16` for the output tables (without it `wrdata` writes only
  9 significant digits). A pre-test on one full branch gave the same bound to 13 digits with
  default and with tight tolerances (reltol 1e-7, abstol 1e-16, vntol 1e-10); gmin = 1e-15
  moved it by 49 nV.
- Junction geometry parameters (ad, as, pd, ps) are left at the subcircuit defaults (0).
- The provenance of the inputs is recorded: SHA-256 of the config, the protocol, the main-run
  files and the model files; the config and predictions.json hashes are compared with the values
  quoted in the protocol.

## Output files

| File | Content |
|---|---|
| `summary.json` | pass/fail of both tasks, maximum deviations and where, self-checks, own values, input hashes, ngspice run list |
| `task1_compare.csv` | every Task 1 quantity: independent, main, absolute and relative difference, threshold, pass |
| `task1_divider_sensitivities.csv` | nominal gate voltages, s_p and lambda for I and aI, each theta |
| `task2_pairs.csv` | per (condition, instance, theta, node): G_k, bound, Y at the bound, both errors, difference, pass |
| `task2_instances.csv` | per (condition, instance): V_sw,pd,k and V_Y*,k, both runs, difference, pass |
| `task2_nominal_bounds.csv` | per (condition, theta): M0 independent vs `b_target`, Y at the nominal bound, pass |
