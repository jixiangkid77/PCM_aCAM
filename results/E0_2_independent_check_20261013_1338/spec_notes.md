# E0.2 independent check: specification notes

Independent re-implementation written only from `docs/E0_1_protocol.md`, `experiments/E0_1_config.json`,
`docs/E0_2_protocol.md` and `experiments/E0_2_config.json` (no project code, no project results read).
Run: 2026-10-13 13:38:08 JST. Script `e02_independent.py` (copy in this folder), SHA-256 `46ed6501d9754467065c13c9c9079073d78f9967f3fd4e679dd1e51ec6b63d73`.

## Ambiguities resolved (and how much they matter)

1. **Feature space of the forest.** The forest is fitted on the min-max-normalized training split, so split
   thresholds are in normalized units and `theta_q = round(clip(thr,0,1)*15)/15` applies directly.
2. **Normalization arithmetic.** float64 `(X - min)/span`, `span = max - min` (0 -> 1), test values scaled
   then clipped to [0, 1]. sklearn `MinMaxScaler` arithmetic (`X*scale + min`) differs by ULPs. Diagnostic:
   forests identical and acc_ref identical for all 15 (dataset, split): **yes** (checks.json,
   `diag_normalization_variant`).
3. **Quantizer rounding.** `np.round` (half to even). The only exactly representable half-way case is
   thr = 0.5 (7.5 -> 8 under half-even and half-up alike), so the rounding rule cannot matter.
4. **Tightest bound vs quantization order.** The tightest float bound is taken first and then quantized; this is
   equivalent to quantize-then-tightest because the quantizer is monotone.
5. **Constraint order when a bound is tightened deeper in the path.** The feature keeps the position of its
   first appearance as that kind of bound; only its value is replaced by the tighter one (lower: max, upper: min).
   A feature with both a lower and an upper bound appears once in each group.
6. **Leaf order.** Depth-first pre-order with the left child first; only leaves emit rows, so rows follow the
   left-to-right leaf order; trees in `estimators_` order. Rows without any constraint (stumps) would always
   match; none occurred (refs.csv `rows_without_constraints`).
7. **Leaf class vector.** `tree_.value[leaf, 0, :] / sum` (sklearn 1.9.1 already stores fractions).
8. **Score summation and ties.** Score = float64 product (row-match indicator) x (class vectors); prediction =
   `np.argmax` (first index among exactly equal floats). Diagnostic: a tolerant argmax (scores within 1e-9 of
   the maximum treated as tied, lowest index wins) changes the number of correct samples in **0** of
   the 1800 evaluations (per_eval.csv `n_correct_tol`).
9. **No match in any tree** counts as wrong even if argmax of the zero vector equals the label.
10. **Path integrity** per evaluation = share of (test sample, tree) pairs with exactly one matching row
    (0 or >= 2 matches count as not intact).
11. **Exact value-threshold ties** (`x == theta`) follow the stated comparisons (lower `x > theta`, upper
    `x <= theta`). They only occur for acc_ref (theta_eff = theta_q); counts in refs.csv
    (`exact_value_threshold_ties`, dominated by digits pixels at exactly 0 or 1 meeting theta_q = 0 or 1). Under
    device noise theta_eff is continuous, so exact ties have probability zero.
12. **Which conductance sets the noise parameters.** Programming-noise sigma and the drift exponent use the
    clipped threshold-programming target (the value actually programmed); the read-noise Q uses G_prog after
    `max(.,0)` (AIHWKit semantics). For high-side clipped targets (raw > 25 uS) mu/sigma_nu are unchanged by
    this choice (both already at their floor at 25 uS) but sigma_prog is evaluated at 25 uS; for low-side
    clipped targets (raw < 0) the target is 0, so g_rel = 1e-7, mu = 0.1, sigma_nu = 0.045 and
    G_prog = max(0.26348 z, 0).
13. **Clipping count** = number of bounds whose raw target `(g_in k_nom - dV_TH)/k_i(v_sl)` is < 0 or > 25 uS
    (clipping.csv gives low/high separately). Total: **2206** of 179200 programmed bounds
    (80 low, 2126 high).
14. **a_hat.** Ratio of means: mean of the 64 replica `G_final(t)` / mean of the 64 replica `G_prog` (after
    the `max(.,0)` clamp), replica read noise from `z_rep_read[slot]`. At all@t0 a_hat differs from 1 because
    the accumulated read noise is already present (no drift at t0: t = 20 s is not > t0).
15. **prog condition.** `G_t = G_prog`, a_hat = 1 for every node; supply then uses
    `V_TX,nom (1 + c_nom) = v_sl` up to rounding, so the four nodes coincide. Accuracy identical across the four
    nodes in every prog evaluation: **yes**.
16. **k_nom** computed as `c_nom/beta_nom` with `beta_nom = c_nom (g_hi - g_lo)/r_in` (= r_in/(g_hi - g_lo) =
    0.02 V/uS up to rounding); c_nom = 0.4285714285714286, beta_nom = 21.42857142857143 uS/V, k_nom = 0.02 V/uS.
17. **theta_eff is not clipped**; a bound outside [0, 1] simply always or never matches.
18. **Pairing.** All nodes of one (dataset, split, device seed, condition) share the same G_t and a_hat; the five
    time slots of one device seed share z_prog and z_nu (one device trajectory) and use z_read[slot] /
    z_rep_read[slot].
19. **Loss averaging.** Per evaluation `loss = 100 (acc_ref(split) - acc)`; per dataset the mean over the 15
    (split, device) evaluations; mean_loss = mean over the 5 datasets; max_loss = largest per-dataset loss.
    Gates compare the float64 mean_loss differences with 1.0 pp (no rounding before comparison).
20. **Self-check 1** runs the same pipeline with zero mismatch and prog/read noise and drift scales set to 0,
    for both programming modes, all nodes, all six conditions, all datasets/splits/device seeds.
21. **Self-check 2** sets nu_mean_a = 0 (so mu = clip(0.0244, 0.049, 0.1) = 0.049), all nu_std parameters to 0,
    prog/read noise scales to 0, m20 mismatch, threshold programming; compares with hand-derived closed forms
    on unclipped cells, a = ((t_inference + 20)/20)^-0.049 (a_hat from the replica equals a to
    1.1e-16). Closed forms (K0 = k_nom (g_hi - g_lo), eps = dV_TX/(V_TX,nom c_nom)):
    - dac: `theta_q + (1/a - 1) dV_TH/K0`
    - ideal: `theta_q`
    - none: `a theta_q - (1 - a) g_lo/(g_hi - g_lo) + (1 - a) dV_TH/K0`
    - supply: `theta_q + (g_in - dV_TH/k_nom) eps (1 - a)/((1 - eps)(g_hi - g_lo))`
22. **Output folder timestamp** is JST at run start.

## Output files

- `per_eval.csv`: the requested columns `dataset, split, device_seed, node, condition, t, a_hat, acc, integrity`
  (floats via `repr`), then extra columns `n_correct, n_test, n_nomatch, n_correct_tol, t_s, loss_pp`.
  `condition` is `prog` or `all`; `t` is `t0, 1h, 1d, 30d, 365d` (prog rows have `t = t0`); `t_s` = t_inference.
- `refs.csv`, `clipping.csv`, `summary.csv` (requested columns first, diagnostics after), `gates.json`,
  `checks.json`, `run_info.json` (versions, spec-file and script SHA-256), `e02_independent.py` (script copy).

## Diagnostics recorded beyond the request

- `checks.json`: structure check (ideal quantized CAM = quantized tree traversal, exactly one match per tree,
  matching leaf = traversal leaf, all datasets and splits), float64-threshold CAM vs sklearn predict agreement,
  extra check 3 (zero mismatch with the full PCM model: dac = supply = ideal), the normalization variant.
- `summary.csv`: mean path integrity, mean a_hat, mean and RMS effective-threshold error in LSB over unclipped
  cells (to compare with the protocol's section 5 estimates).
