# E1 independent check (full-branch re-implementation)

Written by an independent checker from docs/E1_protocol.md and experiments/E1_config.json only
(it did not read src/ or the main experiment scripts). Every bound comes from full-branch ngspice
DC sweeps (resistor + divider transistor, crossing of V_X with V_sw) and root finding on the
conductance; nothing uses the I-V inversion of the main run.

Changes in this delivered copy: the model path is relative to the project root, and the work
folder (default logs/E1_independent_work/, or $E1_INDEP_WORK) is created with its .spiceinit.
The rerun results/E1_independent_check_20261014_0233_rerun/ reproduces independent.csv and
nominal.csv of results/E1_independent_check_20261014_0206/ byte for byte.

Usage (from the project root):
    mkdir results/E1_independent_check_<stamp>
    python experiments/E1_independent/run_e1_indep.py results/E1_independent_check_<stamp>   # about 7 min
Diagnostics: tol_test.py (solver tolerances), crosscheck_iv.py (single-transistor operating points),
bin_probe.py (model bin edge at L = 8 um).
