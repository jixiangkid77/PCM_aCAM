"""E6: cost of the two drift-compensation nodes (input remap, divider supply).

Parts (plan: docs/E6_protocol.md; configuration: experiments/E6_config.json):
  settle    E6.1 settling of the decision node (pull-down gate) after a data-line step, transient SPICE
  power     E6.2 static power of every branch type versus the input, DC SPICE; forest averages (E4 forests)
  dlml      E6.3 divider gate charge (data-line load) and match-line charge, transient SPICE; energy per search
  overhead  E6.4 area and update counts (forests, E5 records, sky130 flip-flop size)
All devices nominal (sky130 tt, 27 C); uniform drift a(t) = (t / 20 s)^-0.049 with a_hat = a.

    python experiments/E6_cost.py --part all [--smoke] [--workers 2]
Outputs (never overwritten): results/E6_<stamp>[_smoke]/
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time

# ngspice-42 is built with OpenMP: two parallel ngspice processes with the default OpenMP settings on two cores
# ran about 30 times slower than with one thread each, with identical results (plan v1.1, section 0)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OMP_WAIT_POLICY", "passive")
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pcm_acam.acam import cam_match  # noqa: E402
from pcm_acam.branch_post import IVCurve, switching_voltage  # noqa: E402
from pcm_acam.inverter import input_at_output, transfer  # noqa: E402
from pcm_acam.io_utils import environment_info, make_logger, make_run_dir, now_jst, sha256_file, write_json  # noqa: E402
from pcm_acam.spice_iv import iv_sweep, ngspice_version, run_netlist  # noqa: E402
from pcm_acam.trees import dataset_bundle, quantize  # noqa: E402

TYPES = (("6T2M", "L6T"), ("6T2M", "UINV"), ("GS", "LBUF"), ("GS", "UINV"))
PD = "sky130_fd_pr__nfet_01v8_lvt"


# ----------------------------------------------------------------------------------------------- helpers
def write_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        return
    keys: List[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def a_of(cfg: dict, tname: str) -> float:
    d = cfg["drift"]
    return float((d["times_s"][tname] / d["t0_s"]) ** (-d["nu"]))


def node_times(cfg: dict, smoke: bool) -> List[Tuple[str, str, float]]:
    times = cfg["smoke"]["times"] if smoke else list(cfg["drift"]["times_s"])
    out = [("t0", "t0", 1.0)]
    for node in ("dac_remap", "supply"):
        out += [(node, t, a_of(cfg, t)) for t in times]
    return out


class Nominal:
    """Nominal switching points, divider curves, input maps and programmed conductances."""

    def __init__(self, cfg: dict, lib: Path):
        c = cfg["circuit"]
        self.c, self.lib = c, lib
        pd = c["pulldown"]
        self.i_sw = 100e-9 * pd["w_um"] / pd["l_um"]
        vg, cur = iv_sweep(lib, pd["device"], pd["w_um"], pd["l_um"], [c["vdd"]], [None], 0.0, c["vdd"], 0.0005)
        self.vsw = switching_voltage(vg, cur[0], self.i_sw)
        y, z = transfer(lib, c["inverter"], [None], [None], c["vdd"], {"vg_start": 0.0, "vg_stop": c["vdd"],
                                                                        "vg_step": 0.0005})
        v_y = input_at_output(y, z[0], self.vsw)
        v_x = input_at_output(y, z[0], v_y)
        self.v_sp = {"L6T": self.vsw, "UINV": v_y, "LBUF": v_x}
        self.d, self.g_lo, self.g_hi = c["d_v"], c["g_lo_us"], c["g_hi_us"]
        self.curves: Dict[Tuple[str, str], IVCurve] = {}
        for d in cfg["designs"]:
            vg, cur = iv_sweep(lib, c["divider_device"], d["w_div_um"], d["l_div_um"],
                               [self.v_sp[t] for t in ("L6T", "UINV", "LBUF")], [None] * 3, 0.0, c["vdd"], 0.0005)
            for k, t in enumerate(("L6T", "UINV", "LBUF")):
                self.curves[(d["id"], t)] = IVCurve.from_arrays(vg, cur[k], i_floor=1e-12)
        self.designs = {d["id"]: d for d in cfg["designs"]}

    def g_of_x(self, x):
        return self.g_lo + np.asarray(x, dtype=float) * (self.g_hi - self.g_lo)

    def lower(self, cell_set: str, cfg: dict) -> str:
        return cfg["cell_sets"][cell_set]["lower"]

    def map_v(self, design: str, cell_set: str, cfg: dict, x, f: float) -> np.ndarray:
        """Data-line voltage of input x under the map M_f of the cell set."""
        cur = self.curves[(design, self.lower(cell_set, cfg))]
        return cur.v_at(f * self.g_of_x(x) * 1e-6 * self.d)

    def g_prog(self, design: str, cell_set: str, typ: str, cfg: dict, theta) -> np.ndarray:
        """Nominal threshold verify: conductance (uS) that puts the bound of this branch type at M0(theta)."""
        v = self.map_v(design, cell_set, cfg, theta, 1.0)
        return self.curves[(design, typ)].i_at(v) / self.d * 1e6

    def setting(self, node: str, a: float, typ: str) -> Tuple[float, float, float]:
        """(conductance factor, V_SL, map factor) of a branch type at a node and drift factor."""
        if node == "t0":
            return 1.0, self.v_sp[typ] + self.d, 1.0
        if node == "dac_remap":
            return a, self.v_sp[typ] + self.d, a
        if node == "supply":
            return a, self.v_sp[typ] + self.d / a, 1.0
        raise ValueError(node)


def dev(name: str, w: float, l: float) -> str:
    return f"sky130_fd_pr__{name} w={w:.6g} l={l:.6g}"


def cell_lines(i: int, typ: str, g_us: float, v_sl: float, dl_node: str, c: dict, w_div: float, l_div: float,
               c_par: float) -> Tuple[List[str], str, str]:
    """One branch: own SL source vsl{i}, own inverter supply vdd{i}, divider node n{i}; returns the lines,
    the decision node (pull-down gate) and the inverter-supply source name ('' for L6T)."""
    inv, pd = c["inverter"], c["pulldown"]
    lines = [f"vsl{i} sl{i} 0 {v_sl:.12g}", f"rp{i} sl{i} n{i} {1.0 / (float(g_us) * 1e-6):.12g}",
             f"xd{i} n{i} {dl_node} 0 0 {dev(c['divider_device'], w_div, l_div)}"]
    if c_par > 0:
        lines.append(f"cp{i} n{i} 0 {c_par:.6g}")
    stages = {"L6T": 0, "UINV": 1, "LBUF": 2}[typ]
    src = ""
    node_in = f"n{i}"
    if stages:
        src = f"vdd{i}"
        lines.append(f"{src} vd{i} 0 {c['vdd']:.9g}")
    for s in range(stages):
        node_out = f"z{i}" if s == stages - 1 else f"w{i}"
        lines.append(f"xpa{s}_{i} {node_out} {node_in} vd{i} vd{i} {dev(inv['p_dev'], inv['wp'], inv['lp'])}")
        lines.append(f"xna{s}_{i} {node_out} {node_in} 0 0 {dev(inv['n_dev'], inv['wn'], inv['ln'])}")
        node_in = node_out
    lines.append(f"xpd{i} ml {node_in} 0 0 {dev(pd['device'], pd['w_um'], pd['l_um'])}")
    return lines, node_in, src


def header(lib: Path, title: str, c: dict) -> List[str]:
    return [f"* {title}", f'.include "{Path(lib).resolve()}"', f"vml ml 0 {c['vdd']:.9g}"]


def _run(netlist: str) -> Dict[str, np.ndarray]:
    return run_netlist(netlist, timeout=7200)


# ----------------------------------------------------------------------------------------------- E6.1
def band_netlist(nom: Nominal, cfg: dict, design: str, cs: str, typ: str, nts, levels) -> Tuple[str, List[dict]]:
    """DC operating points at x = theta - d, theta, theta + d (d = tolerance) for every node-time and level."""
    c = cfg["circuit"]
    d = nom.designs[design]
    tol = cfg["settle"]["tolerance_lsb"] / (cfg["levels"] - 1)
    lines = header(nom.lib, f"E6 band {design} {cs} {typ}", c)
    meta = []
    i = 0
    for node, tname, a in nts:
        gf, v_sl, mf = nom.setting(node, a, typ)
        for j in levels:
            th = j / (cfg["levels"] - 1)
            g = gf * float(nom.g_prog(design, cs, typ, cfg, th))
            for s, xv in ((-1, th - tol), (0, th), (1, th + tol)):
                v_dl = float(nom.map_v(design, cs, cfg, xv, mf))
                lines.append(f"vdl{i} dl{i} 0 {v_dl:.12g}")
                cl, gate, src = cell_lines(i, typ, g, v_sl, f"dl{i}", c, d["w_div_um"], d["l_div_um"], 0.0)
                lines += cl
                meta.append({"i": i, "node": node, "time": tname, "a": a, "level": j, "side": s, "g_us": g,
                             "v_sl": v_sl, "v_dl": v_dl, "gate": gate, "src": src})
                i += 1
    lines += [cfg["settle"]["spice_options"], ".op", ".end"]
    return "\n".join(lines) + "\n", meta


def settle_netlist(nom: Nominal, cfg: dict, design: str, cs: str, typ: str, cases, c_par: float,
                   t_stop: float) -> Tuple[str, List[dict]]:
    c, st = cfg["circuit"], cfg["settle"]
    d = nom.designs[design]
    lines = header(nom.lib, f"E6 settle {design} {cs} {typ} cpar {c_par:.3g}", c)
    meta, saves = [], []
    i = 0
    t_e, t_r = st["t_edge_s"], st["t_rise_s"]
    for node, tname, a, j, xp in cases:
        gf, v_sl, mf = nom.setting(node, a, typ)
        th = j / (cfg["levels"] - 1)
        g = gf * float(nom.g_prog(design, cs, typ, cfg, th))
        v1 = float(nom.map_v(design, cs, cfg, th, mf))
        v0 = float(nom.map_v(design, cs, cfg, xp, mf))
        lines.append(f"vdl{i} dl{i} 0 PWL(0 {v0:.12g} {t_e:.6g} {v0:.12g} {t_e + t_r:.6g} {v1:.12g} "
                     f"{t_stop:.6g} {v1:.12g})")
        cl, gate, _ = cell_lines(i, typ, g, v_sl, f"dl{i}", c, d["w_div_um"], d["l_div_um"], c_par)
        lines += cl
        saves.append(f"v({gate})")
        meta.append({"i": i, "node": node, "time": tname, "a": a, "level": j, "x_prev": xp, "gate": gate})
        i += 1
    lines += [".save " + " ".join(saves), st["spice_options"],
              f".tran {st['t_step_s']:.6g} {t_stop:.6g} 0 {st['t_max_step_s']:.6g}", ".end"]
    return "\n".join(lines) + "\n", meta


def settle_time(t: np.ndarray, v: np.ndarray, lo: float, hi: float, t_edge: float) -> float:
    """Time after the data-line edge of the last exit from [lo, hi] (linear interpolation); 0 if never outside
    after the edge; NaN if outside at the end."""
    after = t >= t_edge
    out = (v < lo) | (v > hi)
    if out[-1]:
        return float("nan")
    idx = np.where(out & after)[0]
    if idx.size == 0:
        return 0.0
    k = int(idx[-1])                    # last sample outside; k + 1 is inside
    bound = lo if v[k] < lo else hi
    t_x = t[k] + (bound - v[k]) * (t[k + 1] - t[k]) / (v[k + 1] - v[k])
    return float(t_x - t_edge)


def part_settle(cfg: dict, nom: Nominal, run_dir: Path, smoke: bool, workers: int, log) -> Tuple[List[dict], dict]:
    st = cfg["settle"]
    nts = node_times(cfg, smoke)
    levels = cfg["smoke"]["levels"] if smoke else list(range(cfg["levels"]))
    designs = cfg["smoke"]["designs"] if smoke else [d["id"] for d in cfg["designs"]]
    cp_main = cfg["circuit"]["c_par_f"]
    cpars = [cp_main] + ([] if smoke else list(cfg["circuit"]["c_par_sensitivity_f"]))
    t_stops = cfg["smoke"]["t_stop_s"] if smoke else st["t_stop_s"]
    sens = st["sensitivity"]
    nts_sens = [nt for nt in nts if nt[0] == "t0" or nt[1] in sens["times"]]
    jobs, keys = [], []
    for dsg in designs:
        for cs, typ in TYPES:
            net, meta = band_netlist(nom, cfg, dsg, cs, typ, nts, levels)
            jobs.append(net)
            keys.append(("band", dsg, cs, typ, None, meta))
            for cp in cpars:
                if cp == cp_main:
                    cases = [(n, t, a, j, xp) for n, t, a in nts for j in levels for xp in st["x_prev"]]
                else:
                    cases = [(n, t, a, j, xp) for n, t, a in nts_sens for j in sens["levels"] for xp in st["x_prev"]]
                step = st["max_copies_per_netlist"]
                for s0 in range(0, len(cases), step):
                    net, meta = settle_netlist(nom, cfg, dsg, cs, typ, cases[s0:s0 + step], cp, t_stops[typ])
                    jobs.append(net)
                    keys.append(("tran", dsg, cs, typ, cp, meta))
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(_run, jobs))
    log(f"E6.1: {len(jobs)} netlists ({time.time() - t0:.0f} s)")
    bands: Dict[tuple, dict] = {}
    c3 = {"max_rel_dev": 0.0, "n": 0}
    for (kind, dsg, cs, typ, cp, meta), vec in zip(keys, results):
        if kind != "band":
            continue
        for m in meta:
            vg = float(vec[f"v({m['gate']})"][0])
            key = (dsg, cs, typ, m["node"], m["time"], m["level"])
            bands.setdefault(key, {})[m["side"]] = vg
            if m["side"] == 0:
                if typ == "L6T" and m["node"] == "t0":
                    i_sl = -float(vec[f"i(vsl{m['i']})"][0])
                    th = m["level"] / (cfg["levels"] - 1)
                    ref = float(nom.g_of_x(th)) * 1e-6 * nom.d
                    c3["max_rel_dev"] = max(c3["max_rel_dev"], abs(i_sl / ref - 1.0))
                    c3["n"] += 1
    tol = st["tolerance_lsb"]
    slope = {k: abs(b[1] - b[-1]) / (2 * tol) for k, b in bands.items()}       # V per LSB at x = theta
    c1 = {"max_dev_lsb": 0.0, "n": 0}
    resid: Dict[tuple, float] = {}
    for k, b in bands.items():
        dev_lsb = abs(b[0] - nom.vsw) / slope[k]
        if k[2] == "UINV" and k[3] == "dac_remap":
            if k[4] == "1y":
                resid[(k[1], k[0])] = max(resid.get((k[1], k[0]), 0.0), dev_lsb)
            continue
        c1["max_dev_lsb"] = max(c1["max_dev_lsb"], dev_lsb)
        c1["n"] += 1
    ref = cfg["checks"]["C1b_reference_lsb"]
    c1b = {"residual_lsb": {f"{cs}|{d}": v for (cs, d), v in resid.items()},
           "reference_lsb": {f"{cs}|{d}": ref[cs][d] for (cs, d) in resid}}
    c1b["pass"] = all(abs(v - ref[cs][d]) <= max(0.1 * ref[cs][d], 0.002) for (cs, d), v in resid.items())
    rows: List[dict] = []
    c2 = {"max_dev_lsb": 0.0, "n_unsettled": 0, "n": 0}
    for (kind, dsg, cs, typ, cp, meta), vec in zip(keys, results):
        if kind != "tran":
            continue
        t = vec["time"]
        for m in meta:
            b = bands[(dsg, cs, typ, m["node"], m["time"], m["level"])]
            lo, hi = sorted((b[-1], b[1]))
            v = vec[f"v({m['gate']})"]
            ts = settle_time(t, v, lo, hi, st["t_edge_s"])
            k_b = (dsg, cs, typ, m["node"], m["time"], m["level"])
            c2["max_dev_lsb"] = max(c2["max_dev_lsb"], abs(float(v[-1]) - b[0]) / slope[k_b])
            c2["n_unsettled"] += int(not np.isfinite(ts))
            c2["n"] += 1
            rows.append({"design": dsg, "cell_set": cs, "type": typ, "node": m["node"], "time": m["time"],
                         "a": m["a"], "level": m["level"], "x_prev": m["x_prev"], "c_par_f": cp,
                         "t_settle_ns": ts * 1e9 if np.isfinite(ts) else float("nan"),
                         "band_lo_v": lo, "band_hi_v": hi, "v_dc_v": b[0], "v_end_v": float(v[-1]),
                         "slope_v_per_lsb": slope[(dsg, cs, typ, m["node"], m["time"], m["level"])]})
    write_csv(run_dir / "settle.csv", rows)
    checks = {"C1": {**c1, "limit_lsb": 1e-3, "pass": c1["max_dev_lsb"] <= 1e-3},
              "C1b": c1b,
              "C2": {**c2, "limit_lsb": 1e-3, "pass": c2["max_dev_lsb"] <= 1e-3 and c2["n_unsettled"] == 0},
              "C3": {**c3, "limit_rel": 5e-3, "pass": c3["max_rel_dev"] <= 5e-3}}
    # summary: largest settling time per (design, cell set, type, node, time, c_par)
    summ: Dict[tuple, dict] = {}
    for r in rows:
        k = (r["design"], r["cell_set"], r["type"], r["node"], r["time"], r["c_par_f"])
        s = summ.setdefault(k, {"max": -1.0, "arg": None, "n": 0, "nan": 0})
        s["n"] += 1
        if not np.isfinite(r["t_settle_ns"]):
            s["nan"] += 1
            continue
        if r["t_settle_ns"] > s["max"]:
            s["max"], s["arg"] = r["t_settle_ns"], (r["level"], r["x_prev"])
    srows = []
    for (dsg, cs, typ, node, tname, cp), s in sorted(summ.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2],
                                                                                   kv[0][5], kv[0][3], kv[0][4])):
        base = summ.get((dsg, cs, typ, "t0", "t0", cp), {"max": float("nan")})["max"]
        srows.append({"design": dsg, "cell_set": cs, "type": typ, "node": node, "time": tname, "c_par_f": cp,
                      "t_settle_max_ns": s["max"], "at_level": s["arg"][0] if s["arg"] else "",
                      "at_x_prev": s["arg"][1] if s["arg"] else "", "ratio_to_t0": s["max"] / base,
                      "n_cases": s["n"], "n_unsettled": s["nan"]})
    write_csv(run_dir / "settle_summary.csv", srows)
    checks["C4"] = check_c4(rows, cpars, sens)
    return srows, checks


def check_c4(rows: List[dict], cpars: List[float], sens: dict) -> dict:
    """C4: the largest settling time per (design, cell set, type, node, time) does not decrease with C_par.
    Compared on the cases run at every C_par (the sensitivity levels and times, both starting points); an unsettled
    case counts as infinitely long. Plan v1.1: v1.0 also looked up node-times without sensitivity runs (KeyError)."""
    c4 = {"pass": True, "n": 0, "violations": [], "n_cases": 0, "n_cases_nonmonotone": 0,
          "basis": "largest settling time over the sensitivity cases (levels x x_prev) at each C_par"}
    if len(cpars) < 2:
        return c4
    order = sorted(cpars)
    groups: Dict[tuple, Dict[float, float]] = {}
    cases: Dict[tuple, Dict[float, float]] = {}
    for r in rows:
        if r["level"] not in sens["levels"] or not (r["node"] == "t0" or r["time"] in sens["times"]):
            continue
        t = r["t_settle_ns"] if np.isfinite(r["t_settle_ns"]) else float("inf")
        k = (r["design"], r["cell_set"], r["type"], r["node"], r["time"])
        g = groups.setdefault(k, {})
        g[r["c_par_f"]] = max(g.get(r["c_par_f"], -1.0), t)
        cases.setdefault(k + (r["level"], r["x_prev"]), {})[r["c_par_f"]] = t
    for k, g in groups.items():
        vals = [g[cp] for cp in order]
        c4["n"] += 1
        if any(vals[q + 1] < vals[q] - 1e-6 for q in range(len(vals) - 1)):
            c4["pass"] = False
            c4["violations"].append({"key": list(k), "values_ns": vals})
    for k, g in cases.items():
        vals = [g[cp] for cp in order]
        c4["n_cases"] += 1
        c4["n_cases_nonmonotone"] += int(any(vals[q + 1] < vals[q] - 1e-6 for q in range(len(vals) - 1)))
    return c4


# ----------------------------------------------------------------------------------------------- E6.2
def power_netlist(nom: Nominal, cfg: dict, design: str, cs: str, typ: str, node: str, a: float,
                  levels) -> Tuple[str, List[dict]]:
    c = cfg["circuit"]
    d = nom.designs[design]
    gf, v_sl, _ = nom.setting(node, a, typ)
    lines = header(nom.lib, f"E6 power {design} {cs} {typ} {node} {a:.6f}", c) + ["vdl dl 0 0"]
    meta = []
    for i, j in enumerate(levels):
        th = j / (cfg["levels"] - 1)
        g = gf * float(nom.g_prog(design, cs, typ, cfg, th))
        cl, gate, src = cell_lines(i, typ, g, v_sl, "dl", c, d["w_div_um"], d["l_div_um"], 0.0)
        lines += cl
        meta.append({"i": i, "level": j, "g_us": g, "v_sl": v_sl, "src": src})
    lines += [f".dc vdl 0 {c['vdd']:.9g} {cfg['power']['vdl_step']:.9g}", ".end"]
    return "\n".join(lines) + "\n", meta


def part_power(cfg: dict, nom: Nominal, run_dir: Path, smoke: bool, workers: int, log):
    c = cfg["circuit"]
    nts = node_times(cfg, smoke)
    levels = list(range(cfg["levels"]))
    designs = cfg["smoke"]["designs"] if smoke else [d["id"] for d in cfg["designs"]]
    jobs, keys = [], []
    for dsg in designs:
        for cs, typ in TYPES:
            for node, tname, a in nts:
                net, meta = power_netlist(nom, cfg, dsg, cs, typ, node, a, levels)
                jobs.append(net)
                keys.append((dsg, cs, typ, node, tname, a, meta))
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(_run, jobs))
    log(f"E6.2: {len(jobs)} DC sweeps ({time.time() - t0:.0f} s)")
    curves: Dict[tuple, Dict[int, np.ndarray]] = {}
    xg = {n: np.linspace(0.0, 1.0, n) for n in (cfg["power"]["x_grid"], cfg["power"]["x_grid_check"])}
    store = {}
    for (dsg, cs, typ, node, tname, a, meta), vec in zip(keys, results):
        vdl = vec["v(dl)"]
        _, _, mf = nom.setting(node, a, typ)
        for nx, x in xg.items():
            v_of_x = nom.map_v(dsg, cs, cfg, x, mf)
            p = np.empty((len(meta), nx))
            for m in meta:
                p_v = m["v_sl"] * (-vec[f"i(vsl{m['i']})"])
                if m["src"]:
                    p_v = p_v + c["vdd"] * (-vec[f"i({m['src']})"])
                p[m["level"]] = np.interp(v_of_x, vdl, p_v)
            curves[(dsg, cs, typ, node, tname, nx)] = p
            if nx == cfg["power"]["x_grid"]:
                store[f"{dsg}|{cs}|{typ}|{node}|{tname}"] = p.astype(np.float64)
    np.savez_compressed(run_dir / "power_curves.npz", x=xg[cfg["power"]["x_grid"]], **store)
    return curves


def forest_bundles(cfg: dict, smoke: bool, log) -> List[dict]:
    dss = cfg["smoke"]["datasets"] if smoke else cfg["datasets"]
    splits = cfg["smoke"]["splits"] if smoke else cfg["split_seeds"]
    out = []
    t0 = time.time()
    for ds in dss:
        for sp in splits:
            b = dataset_bundle(ds, sp)
            rows = b["rows"]
            thq = quantize(rows.cons_thr, cfg["bits"])
            out.append({"dataset": ds, "split": sp, "x": b["x_te"], "rows": rows, "level":
                        np.rint(thq * (cfg["levels"] - 1)).astype(int), "theta_q": thq,
                        "n_features": int(b["x_te"].shape[1]), "n_rows": int(rows.row_tree.shape[0])})
    log(f"forests: {len(out)} ({time.time() - t0:.0f} s)")
    return out


def forest_power(curves, bundles, cfg, nts, designs, nx: int) -> List[dict]:
    xg = np.linspace(0.0, 1.0, nx)
    out = []
    for b in bundles:
        rows, x = b["rows"], b["x"]
        xv = x[:, rows.cons_feat]                                     # (N, K)
        kind, lev = rows.cons_kind, b["level"]
        sat = np.where(kind[None, :] == 0, xv > b["theta_q"][None, :], xv <= b["theta_q"][None, :])
        for dsg in designs:
            for cs in ("6T2M", "GS"):
                typ = {0: cfg["cell_sets"][cs]["lower"], 1: cfg["cell_sets"][cs]["upper"]}
                for node, tname, a in nts:
                    pk = np.empty(xv.shape)
                    for kk in (0, 1):
                        P = curves[(dsg, cs, typ[kk], node, tname, nx)]
                        for j in np.unique(lev[kind == kk]):
                            sel = np.where((kind == kk) & (lev == j))[0]
                            pk[:, sel] = np.interp(xv[:, sel], xg, P[j])
                    tot = pk.sum(axis=1)
                    out.append({"dataset": b["dataset"], "split": b["split"], "design": dsg, "cell_set": cs,
                                "node": node, "time": tname, "a": a, "n_bounds": int(xv.shape[1]),
                                "n_samples": int(xv.shape[0]), "p_search_uw": float(tot.mean() * 1e6),
                                "p_per_bound_uw": float(pk.mean() * 1e6),
                                "p_satisfied_uw": float(pk[sat].mean() * 1e6),
                                "p_violated_uw": float(pk[~sat].mean() * 1e6) if (~sat).any() else float("nan"),
                                "share_violated": float((~sat).mean())})
    return out


# ----------------------------------------------------------------------------------------------- E6.3
def gate_charge(nom: Nominal, cfg: dict, design: str, v_drain: float) -> Tuple[np.ndarray, np.ndarray]:
    """Charge drawn by the divider gate while it ramps 0 -> V_DD with the drain at v_drain (C, V)."""
    c, dl = cfg["circuit"], cfg["dl"]
    d = nom.designs[design]
    tr = dl["gate_ramp_s"]
    net = "\n".join([f"* gate charge {design}", f'.include "{Path(nom.lib).resolve()}"',
                     f"vg g 0 PWL(0 0 {tr:.6g} {c['vdd']:.9g})", f"vd dr 0 {v_drain:.12g}",
                     f"xd dr g 0 0 {dev(c['divider_device'], d['w_div_um'], d['l_div_um'])}",
                     f".tran {tr / 2000:.6g} {tr:.6g} 0 {tr / 2000:.6g}", ".end"]) + "\n"
    vec = run_netlist(net)
    t, v, i = vec["time"], vec["v(g)"], -vec["i(vg)"]
    q = np.concatenate([[0.0], np.cumsum(0.5 * (i[1:] + i[:-1]) * np.diff(t))])
    return v, q


def ml_charge(nom: Nominal, cfg: dict, n_cells: int) -> float:
    """Charge to precharge a row of n_cells (lumped C plus 2 n_cells pull-downs at gate 0) from 0 to V_pre."""
    c, m = cfg["circuit"], cfg["ml"]
    pd = c["pulldown"]
    tr = 1e-8
    lines = [f"* ML charge N={n_cells}", f'.include "{Path(nom.lib).resolve()}"',
             f"vm ml 0 PWL(0 0 {tr:.6g} {m['v_pre']:.9g})",
             f"cml ml 0 {m['c_sa_f'] + n_cells * m['c_wire_per_cell_f']:.9g}"]
    lines += [f"xp{j} ml 0 0 0 {dev(pd['device'], pd['w_um'], pd['l_um'])}" for j in range(2 * n_cells)]
    lines += [f".tran {tr / 2000:.6g} {tr:.6g} 0 {tr / 2000:.6g}", ".end"]
    vec = run_netlist("\n".join(lines) + "\n")
    t, i = vec["time"], -vec["i(vm)"]
    return float(np.sum(0.5 * (i[1:] + i[:-1]) * np.diff(t)))


def t_s_of(nom: Nominal, cfg: dict, n_cells: int) -> float:
    """Sense time as in E3c/E4: cell-under-test pull-down at V_sw, the other 2N - 1 gates at 0 V."""
    c, m = cfg["circuit"], cfg["ml"]
    pd = c["pulldown"]
    lines = [f"* t_s N={n_cells}", f'.include "{Path(nom.lib).resolve()}"', f"vgc gc 0 {nom.vsw:.12g}",
             f"cml ml 0 {m['c_sa_f'] + n_cells * m['c_wire_per_cell_f']:.9g}",
             f"xpc ml gc 0 0 {dev(pd['device'], pd['w_um'], pd['l_um'])}"]
    lines += [f"xpo{j} ml 0 0 0 {dev(pd['device'], pd['w_um'], pd['l_um'])}" for j in range(2 * n_cells - 1)]
    lines += [f".ic v(ml)={m['v_pre']:.9g}", f".meas tran tc when v(ml)={m['v_ref']:.9g} fall=1",
              ".tran 1e-10 5e-6 0 2e-10", ".options numdgt=10", ".end"]
    import subprocess
    import tempfile
    from pcm_acam.spice_iv import SPICEINIT
    with tempfile.TemporaryDirectory(prefix="pcm_acam_e6_") as td:
        (Path(td) / ".spiceinit").write_text(SPICEINIT)
        (Path(td) / "net.sp").write_text("\n".join(lines) + "\n")
        res = subprocess.run(["ngspice", "-b", "net.sp"], cwd=td, capture_output=True, text=True, timeout=600)
    mt = re.search(r"^tc\s*=\s*([0-9.eE+-]+)", res.stdout, re.M)
    if not mt:
        raise RuntimeError("t_s measure failed")
    return float(mt.group(1))


def part_dlml(cfg: dict, nom: Nominal, bundles: List[dict], nts, designs, run_dir: Path, log):
    c = cfg["circuit"]
    q_curves = {}
    cap_rows = []
    for dsg in designs:
        for typ in ("L6T", "UINV", "LBUF"):
            v, q = gate_charge(nom, cfg, dsg, nom.v_sp[typ])
            q_curves[(dsg, typ)] = (v, q)
            cap_rows.append({"design": dsg, "drain_at": typ, "v_drain_v": nom.v_sp[typ],
                             "q_full_fc": float(q[-1] * 1e15), "c_avg_ff": float(q[-1] / c["vdd"] * 1e15)})
    feats = sorted({b["n_features"] for b in bundles})
    ts = {n: t_s_of(nom, cfg, n) for n in sorted(set(feats) | set(cfg["settle"]["rows_for_t_s"]))}
    qml = {n: ml_charge(nom, cfg, n) for n in feats}
    for n in sorted(ts):
        cap_rows.append({"design": "", "drain_at": f"ML N={n}", "t_s_ns": ts[n] * 1e9,
                         "q_ml_fc": qml[n] * 1e15 if n in qml else "", "e_precharge_pj":
                         cfg["ml"]["v_pre"] * qml[n] * 1e12 if n in qml else ""})
    write_csv(run_dir / "capacitance.csv", cap_rows)
    out = []
    for b in bundles:
        x, rows = b["x"], b["rows"]
        n_rows = b["n_rows"]
        match = cam_match(x, rows, b["theta_q"])
        n_mis = (~match).sum(axis=1)                                   # mismatching rows per search
        e_ml = n_mis * cfg["ml"]["v_pre"] * qml[b["n_features"]]
        for dsg in designs:
            for cs in ("6T2M", "GS"):
                lo, up = cfg["cell_sets"][cs]["lower"], cfg["cell_sets"][cs]["upper"]
                vq_lo, vq_up = q_curves[(dsg, lo)], q_curves[(dsg, up)]
                for node, tname, a in nts:
                    _, _, mf = nom.setting(node, a, lo)
                    v = nom.map_v(dsg, cs, cfg, x, mf)                 # (N, F) data-line voltages
                    q = np.interp(v, *vq_lo) + np.interp(v, *vq_up)  # charge of the two dividers of a cell
                    dq = np.diff(q, axis=0)                            # consecutive searches
                    e_dl = c["vdd"] * np.clip(dq, 0.0, None).sum(axis=1) * n_rows
                    out.append({"dataset": b["dataset"], "split": b["split"], "design": dsg, "cell_set": cs,
                                "node": node, "time": tname, "n_rows": n_rows, "n_features": b["n_features"],
                                "t_s_ns": ts[b["n_features"]] * 1e9, "e_dl_pj": float(e_dl.mean() * 1e12),
                                "e_ml_pj": float(e_ml.mean() * 1e12),
                                "mismatching_rows": float(n_mis.mean())})
    return out, ts


# ----------------------------------------------------------------------------------------------- E6.4
def lef_area_um2(path: Path) -> float:
    txt = path.read_text()
    m = re.search(r"SIZE\s+([0-9.]+)\s+BY\s+([0-9.]+)", txt)
    return float(m.group(1)) * float(m.group(2))


def part_overhead(cfg: dict, bundles: List[dict], designs, run_dir: Path) -> List[dict]:
    ov = cfg["overhead"]
    ff = lef_area_um2(ROOT / ov["ff_lef"])
    e5 = ROOT / ov["e5_run"]
    with open(e5 / "e53_min_bits.csv", encoding="utf-8") as f:
        mb = list(csv.DictReader(f))
    with open(e5 / "e52_limits.csv", encoding="utf-8") as f:
        lim = list(csv.DictReader(f))
    bits_full = {d: max(int(r["min_bits"]) for r in mb if r["design"] == d and r["quantity"] == "DL" and
                        r["grid"] == "full") for d in designs}
    bits_vsl = {d: max(int(r["min_bits"]) for r in mb if r["design"] == d and r["quantity"] == "SL") for d in designs}
    upd = [int(r["updates_first_year"]) for r in lim if r["time"] == "1y"]
    rows = []
    by_ds: Dict[str, List[dict]] = {}
    for b in bundles:
        by_ds.setdefault(b["dataset"], []).append(b)
    for ds, bs in by_ds.items():
        n_rows = float(np.mean([b["n_rows"] for b in bs]))
        n_feat = bs[0]["n_features"]
        n_bounds = float(np.mean([b["rows"].cons_kind.size for b in bs]))
        cells = n_rows * n_feat
        r = {"dataset": ds, "rows_mean": n_rows, "features": n_feat, "cells": cells, "pcm_devices": 2 * cells,
             "programmed_bounds_mean": n_bounds}
        for n in ov["replica_cells"]:
            r[f"replica_{n}_share"] = n / (2 * cells)
        rows.append(r)
    for dsg in designs:
        bits = (1 << ov["input_bits"]) * bits_full[dsg]
        rows.append({"dataset": "", "design": dsg, "lut_bits": bits, "lut_area_um2_ff": bits * ff,
                     "ff_area_um2": ff, "dl_bits_full_scale": bits_full[dsg], "vsl_dac_bits": bits_vsl[dsg],
                     "adjustable_supplies_per_cell_set": 2,
                     "divider_gate_area_per_cell_um2": 2 * cfg_design(cfg, dsg)["w_div_um"] *
                     cfg_design(cfg, dsg)["l_div_um"],
                     "updates_first_year_min": min(upd), "updates_first_year_max": max(upd)})
    write_csv(run_dir / "overhead.csv", rows)
    return rows


def cfg_design(cfg: dict, dsg: str) -> dict:
    return next(d for d in cfg["designs"] if d["id"] == dsg)


# ----------------------------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="all", choices=["all", "settle", "power", "dlml", "overhead"])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--config", default=str(ROOT / "experiments" / "E6_config.json"))
    args = ap.parse_args()
    cfg_path = Path(args.config)
    cfg = json.loads(cfg_path.read_text())
    run_dir = make_run_dir(ROOT / "results", cfg["run_prefix"], smoke=args.smoke)
    log = make_logger(run_dir / "run.log")
    lib = ROOT / cfg["models"]["lib"]
    files = [cfg_path, Path(__file__), lib, ROOT / cfg["overhead"]["ff_lef"]] + \
        [ROOT / "src/pcm_acam" / f for f in ("spice_iv.py", "branch_post.py", "inverter.py", "trees.py", "acam.py")]
    info = {"started": now_jst().isoformat(), "argv": sys.argv, "ngspice": ngspice_version(), "env": environment_info(),
            "openmp": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OMP_WAIT_POLICY")},
            "sha256": {str(p.relative_to(ROOT)): sha256_file(p) for p in files}}
    write_json(run_dir / "run_info.json", info)
    log(f"E6 start ({'smoke' if args.smoke else 'full'}), part {args.part}, run dir {run_dir.name}")
    t0 = time.time()
    nom = Nominal(cfg, lib)
    log(f"nominal: V_sw {nom.vsw:.6f} V, V_Y* {nom.v_sp['UINV']:.6f} V, V_X* {nom.v_sp['LBUF']:.6f} V "
        f"({time.time() - t0:.0f} s)")
    nts = node_times(cfg, args.smoke)
    designs = cfg["smoke"]["designs"] if args.smoke else [d["id"] for d in cfg["designs"]]
    checks: dict = {}
    settle_rows = None
    if args.part in ("all", "settle"):
        settle_rows, ch = part_settle(cfg, nom, run_dir, args.smoke, args.workers, log)
        checks.update(ch)
    bundles = None
    if args.part in ("all", "power", "dlml", "overhead"):
        bundles = forest_bundles(cfg, args.smoke, log)
    if args.part in ("all", "power"):
        curves = part_power(cfg, nom, run_dir, args.smoke, args.workers, log)
        fp = forest_power(curves, bundles, cfg, nts, designs, cfg["power"]["x_grid"])
        fp2 = forest_power(curves, bundles, cfg, nts, designs, cfg["power"]["x_grid_check"])
        dev5 = max(abs(b["p_search_uw"] / a["p_search_uw"] - 1.0) for a, b in zip(fp, fp2))
        checks["C5"] = {"max_rel_dev": dev5, "limit_rel": 5e-3, "pass": dev5 <= 5e-3}
        write_csv(run_dir / "power_forest.csv", fp)
        log(f"E6.2: forest averages ({len(fp)} rows)")
    if args.part in ("all", "dlml"):
        dl, ts = part_dlml(cfg, nom, bundles, nts, designs, run_dir, log)
        write_csv(run_dir / "energy_dl_ml.csv", dl)
        log(f"E6.3: data-line and match-line energy ({len(dl)} rows)")
    if args.part in ("all", "overhead"):
        part_overhead(cfg, bundles, designs, run_dir)
        log("E6.4: overhead counts")
    write_json(run_dir / "checks.json", checks)
    info["finished"] = now_jst().isoformat()
    info["seconds"] = round(time.time() - t0, 1)
    info["checks_pass"] = {k: v.get("pass") for k, v in checks.items()}
    write_json(run_dir / "run_info.json", info)
    log(f"E6 done in {time.time() - t0:.0f} s; checks: " + ", ".join(f"{k} {'pass' if v.get('pass') else 'FAIL'}"
                                                                     for k, v in checks.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
