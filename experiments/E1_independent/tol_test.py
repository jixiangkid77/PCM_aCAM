"""Sensitivity of V_sw and branch bounds to ngspice tolerances and sweep resolution.

Run: python3 tol_test.py   (writes work/tol_test.json)
"""
import json
import numpy as np
import e1lib as L

rng = np.random.default_rng([20261014, 0])
z_div = rng.standard_normal((300, 4))
z_pd = rng.standard_normal((300, 4))

opt_sets = {
    "default": {},                                              # ngspice defaults
    "tight": dict(L.OPTIONS),                                    # used in the main check
    "tighter": {"reltol": 1e-9, "abstol": 1e-18, "vntol": 1e-12},
}
mm_pd = [np.zeros(4)] + [z_pd[i] for i in range(5)]
vsw = {}
for name, o in opt_sets.items():
    vsw[name] = L.vsw_batch(mm_pd, tag="tol_pd_" + name, options=o)

vsw_nom = vsw["tight"][0]
vsl = vsw_nom + 0.2
brs = []
for Ld in (0.15, 8.0):
    for i in range(5):
        for G in (1.3e-6, 2.5e-6, 10e-6, 22.5e-6):
            brs.append(dict(L=Ld, mm=z_div[i], G=G, vsl=vsl, vsw=vsw["tight"][i + 1]))
bnd = {}
for name, o in opt_sets.items():
    bnd[name] = L.bounds_batch(brs, tag="tol_br_" + name, options=o)
# resolution: coarse-only estimate at 0.5 mV and a finer fine-step
b_fine2 = L.bounds_batch(brs, tag="tol_br_fine2", fine_half=0.2e-3, fine_step=0.001e-3)
_, b_coarse = L.bounds_batch(brs, tag="tol_br_coarse", return_diag=True)

rep = {
    "vsw_V": {k: list(map(float, v)) for k, v in vsw.items()},
    "max_abs_dvsw_default_minus_tight_uV": float(np.max(np.abs(vsw["default"] - vsw["tight"])) * 1e6),
    "max_abs_dvsw_tighter_minus_tight_uV": float(np.max(np.abs(vsw["tighter"] - vsw["tight"])) * 1e6),
    "max_abs_dbound_default_minus_tight_uV": float(np.max(np.abs(bnd["default"] - bnd["tight"])) * 1e6),
    "max_abs_dbound_tighter_minus_tight_uV": float(np.max(np.abs(bnd["tighter"] - bnd["tight"])) * 1e6),
    "max_abs_dbound_finestep1uV_minus_tight_uV": float(np.max(np.abs(b_fine2 - bnd["tight"])) * 1e6),
    "max_abs_dbound_coarseonly_minus_tight_uV": float(np.max(np.abs(b_coarse - bnd["tight"])) * 1e6),
}
print(json.dumps(rep, indent=1))
with open(L.WORK + "/tol_test.json", "w") as f:
    json.dump(rep, f, indent=1)
