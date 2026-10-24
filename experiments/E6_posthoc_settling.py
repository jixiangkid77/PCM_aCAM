"""E6 post-hoc diagnostic (not pre-registered): settling of the worst cases with wider tolerance bands, and the
node that sets the settling time.

For every (design, cell set, branch type) and the node-times t0, input remap 1 y and supply 1 y, the case with the
largest settling time of the run of record (settle_summary.csv, C_par 0.5 fF) is simulated again as a single copy
with the netlist functions of experiments/E6_cost.py (same numerics), saving the divider node, the internal
inverter node (LBUF) and the decision node. Reported per case: the settling time to bands of +-0.01 (reproduces the
run), +-0.05 and +-0.1 LSB (input-referred, DC band as in the plan), and the time constant of the last decade of
each node's approach to its final value.

Usage: python experiments/E6_posthoc_settling.py results/E6_<stamp>
Writes <run>/posthoc_settling.csv and <run>/posthoc_settling.md (nothing else is changed).
"""

from __future__ import annotations

import copy
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "src"))
sys.dont_write_bytecode = True

import E6_cost as E  # noqa: E402  (sets the OpenMP environment)
from pcm_acam.spice_iv import run_netlist  # noqa: E402

BANDS = (0.01, 0.05, 0.1)
NTS = (("t0", "t0"), ("dac_remap", "1y"), ("supply", "1y"))


def tail_tau(t: np.ndarray, v: np.ndarray, t_edge: float) -> float:
    """Time constant of the last decade of |v - v_end| before it drops below 1e-6 V (log-linear fit); NaN if the
    deviation is never above 1e-5 V after the edge."""
    dev = np.abs(v - v[-1])
    sel = (t > t_edge) & (dev > 1e-6)
    if not sel.any() or dev[sel].max() < 1e-5:
        return float("nan")
    idx = np.where(sel)[0]
    hi = dev[idx].max()
    win = idx[(dev[idx] <= hi / 10) & (dev[idx] >= max(hi / 1000, 1e-6))]
    if win.size < 5:
        return float("nan")
    k, _ = np.polyfit(t[win], np.log(dev[win]), 1)
    return float(-1.0 / k) if k < 0 else float("nan")


def main() -> int:
    run = Path(sys.argv[1])
    cfg = json.loads((ROOT / "experiments" / "E6_config.json").read_text())
    nom = E.Nominal(cfg, ROOT / cfg["models"]["lib"])
    summ = list(csv.DictReader(open(run / "settle_summary.csv", encoding="utf-8")))
    a1 = E.a_of(cfg, "1y")
    rows = []
    for r in summ:
        if abs(float(r["c_par_f"]) - cfg["circuit"]["c_par_f"]) > 1e-20 or (r["node"], r["time"]) not in NTS:
            continue
        dsg, cs, typ, node, tname = r["design"], r["cell_set"], r["type"], r["node"], r["time"]
        a = 1.0 if node == "t0" else a1
        j, xp = int(r["at_level"]), float(r["at_x_prev"])
        t_stop = cfg["settle"]["t_stop_s"][typ]
        net, meta = E.settle_netlist(nom, cfg, dsg, cs, typ, [(node, tname, a, j, xp)], cfg["circuit"]["c_par_f"],
                                     t_stop)
        nodes = ["v(n0)", f"v({meta[0]['gate']})"] + (["v(w0)"] if typ == "LBUF" else [])
        lines = [(".save " + " ".join(sorted(set(nodes)))) if ln.startswith(".save") else ln for ln in net.splitlines()]
        vec = run_netlist("\n".join(lines) + "\n", timeout=3600)
        t, vg = vec["time"], vec[f"v({meta[0]['gate']})"]
        out = {"design": dsg, "cell_set": cs, "type": typ, "node": node, "time": tname, "level": j, "x_prev": xp,
               "t_settle_run_ns": float(r["t_settle_max_ns"])}
        for tol in BANDS:
            c2 = copy.deepcopy(cfg)
            c2["settle"]["tolerance_lsb"] = tol
            bnet, bmeta = E.band_netlist(nom, c2, dsg, cs, typ, [(node, tname, a)], [j])
            bvec = run_netlist(bnet)
            b = {m["side"]: float(bvec[f"v({m['gate']})"][0]) for m in bmeta}
            lo, hi = sorted((b[-1], b[1]))
            out[f"t_{tol:g}lsb_ns"] = E.settle_time(t, vg, lo, hi, cfg["settle"]["t_edge_s"]) * 1e9
        for nd in nodes:
            out[f"tau_{nd[2:-1]}_ns"] = tail_tau(t, vec[nd], cfg["settle"]["t_edge_s"]) * 1e9
        rows.append(out)
    keys = []
    for o in rows:
        keys += [k for k in o if k not in keys]
    with open(run / "posthoc_settling.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    md = ["# E6 post-hoc: settling of the worst cases with wider bands (not pre-registered)", "",
          "| design | cell set / type | node | level / x_prev | run (ns) | ±0.01 LSB | ±0.05 LSB | ±0.1 LSB | tau divider node | tau inverter node (LBUF) | tau decision node |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for o in rows:
        md.append(f"| {o['design']} | {o['cell_set']}/{o['type']} | {o['node']} {o['time']} | {o['level']} / {o['x_prev']:g} | "
                  f"{o['t_settle_run_ns']:.1f} | {o['t_0.01lsb_ns']:.1f} | {o['t_0.05lsb_ns']:.1f} | {o['t_0.1lsb_ns']:.1f} | "
                  f"{o.get('tau_n0_ns', float('nan')):.2f} | {o.get('tau_w0_ns', float('nan')):.2f} | "
                  f"{o.get('tau_z0_ns', o.get('tau_n0_ns', float('nan'))):.2f} |")
    (run / "posthoc_settling.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"wrote {run / 'posthoc_settling.csv'} and posthoc_settling.md ({len(rows)} cases)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
