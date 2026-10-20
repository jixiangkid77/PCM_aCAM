#!/usr/bin/env python3
"""Diagnostic only: sensitivity of the nominal bound to the BSIM4 model-bin choice.

The divider sizes sit exactly on bin edges of sky130_fd_pr__nfet_01v8 (W = 0.42 um is the edge
between W bins 0.39-0.42 and 0.42-0.52; L = 8 um is the edge between L bins 4-8 and 8-20).
ngspice-42 selects the bins whose lower edge equals the size (W 0.42-0.52, L 8-20).  Here the size
is nudged below the edge (W by 1e-7 um; L by 1e-3 um because ngspice-42 still picks the 8-20 um bin
for L = 7.9999999) to force the neighbouring bin; the geometric change itself is small (relative
2.4e-7 for W, 1.25e-4 for L), so a difference mainly shows the discontinuity between bins.

Usage: python3 bin_probe.py <output_dir>
"""
import json
import os
import sys

import numpy as np

import e1lib as L

OUT = sys.argv[1]
vsw = float(L.vsw_batch([np.zeros(4)], tag="probe_pd")[0])
vsl = vsw + 0.2
Gs = [2.5e-6, 7.5e-6, 12.5e-6, 17.5e-6, 22.5e-6]
variants = {
    "P1_as_used(L0.15,W0.42)": (0.15, 0.42),
    "P1_Wbelow(L0.15,W0.4199999)": (0.15, 0.4199999),
    "P4_as_used(L8,W0.42)": (8.0, 0.42),
    "P4_Lbelow(L7.999,W0.42)": (7.999, 0.42),
    "P4_Wbelow(L8,W0.4199999)": (8.0, 0.4199999),
}
res = {}
for name, (Ld, Wd) in variants.items():
    L.W_DIV = Wd
    b = L.bounds_batch([dict(L=Ld, mm=np.zeros(4), G=g, vsl=vsl, vsw=vsw) for g in Gs], tag="probe")
    res[name] = list(map(float, b))
L.W_DIV = 0.42
rep = {"G_uS": [g * 1e6 for g in Gs], "bounds_V": res}
for base, alts in (("P1_as_used(L0.15,W0.42)", ["P1_Wbelow(L0.15,W0.4199999)"]),
                   ("P4_as_used(L8,W0.42)", ["P4_Lbelow(L7.999,W0.42)", "P4_Wbelow(L8,W0.4199999)"])):
    for alt in alts:
        rep["max_abs_diff_mV_" + alt] = float(np.max(np.abs(np.array(res[alt]) - np.array(res[base]))) * 1e3)
print(json.dumps(rep, indent=1))
with open(os.path.join(OUT, "bin_probe.json"), "w") as f:
    json.dump(rep, f, indent=1)
