"""E4: random-forest inference on a PCM analog CAM with transistor-level bound errors, AIHWKit PCM noise and
match-line crosstalk; input remap versus divider supply as the drift-compensation node.

Frozen plan: experiments/E4_config.json (Chinese description: docs/E4_protocol.md).

Phase 1 (spice): Monte-Carlo pools of the three branch types (sky130 PDK mismatch: dividers, pull-downs and
inverters per instance), nominal divider I-V surfaces, checks C1-C6 and the row validation V1 (full-row
transient SPICE versus the behavioural row model).
    python experiments/E4_forest_system.py --phase spice [--smoke]
Phase 2 (forest): accuracy of the forests on the CAM for every design, arm, condition, time and node.
    python experiments/E4_forest_system.py --phase forest --instrument results/E4_spice_<stamp> [--smoke]

Outputs (never overwritten): results/E4_spice_<stamp>[_smoke]/ and results/E4_<stamp>[_smoke]/
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pcm_acam.acam import Scheme, cam_predict, make_draws, replica_factor  # noqa: E402
from pcm_acam.branch_post import switching_voltage  # noqa: E402
from pcm_acam.inverter import input_at_output, transfer, upper_branch_netlist  # noqa: E402
from pcm_acam.io_utils import environment_info, make_logger, make_run_dir, now_jst, sha256_file, write_json  # noqa: E402
from pcm_acam.ml_transient import MLSpec, header, run_meas, tran_card  # noqa: E402
from pcm_acam.pcm_model import (PCMParams, apply_drift_and_read_noise, apply_programming_noise,  # noqa: E402
                                drift_coefficients)
from pcm_acam.row_model import (BranchState, Chain, Curve1D, Pool, Surface, compress_curves,  # noqa: E402
                                crosstalk_match, effective_x, map_voltage, program_targets, solve_nodes,
                                violation_match, vote)
from pcm_acam.spice_iv import MM_KEYS, branch_dc, iv_sweep, mm_string, ngspice_version, run_netlist  # noqa: E402
from pcm_acam.trees import dataset_bundle, quantize, traverse_quantized  # noqa: E402

TYPES = ("L6T", "UINV", "LBUF")
NODES = ("dac_remap", "supply", "ideal")
DRAW_KEYS = ("div_lo", "pd_lo", "b1n", "b1p", "b2n", "b2p", "div_hi", "un", "up", "pd_hi")
LSB = 15.0
REL_GRID = np.round(np.arange(-0.06, 0.06 + 1e-12, 0.0005), 6)
PD_GRID_C = np.round(np.arange(0.0, 0.8 + 1e-12, 0.005), 6)


# --------------------------------------------------------------------------------------------- helpers
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


def hashes(cfg_path: Path, cfg: dict) -> Dict[str, str]:
    files = [cfg_path, Path(__file__)] + [ROOT / f"src/pcm_acam/{m}.py" for m in
                                          ("row_model", "spice_iv", "branch_post", "inverter", "ml_transient",
                                           "acam", "pcm_model", "trees", "io_utils")] + [ROOT / cfg["models"]["lib"]]
    return {str(p.relative_to(ROOT)): sha256_file(p) for p in files}


def mm_rows(z: np.ndarray) -> List[Dict[str, float]]:
    return [{k: float(v) for k, v in zip(MM_KEYS, r)} for r in z]


def mm_of(z: np.ndarray) -> Dict[str, float]:
    return {k: float(v) for k, v in zip(MM_KEYS, z)}


def a_uniform(t_inf: float, t0: float) -> float:
    return float(((t_inf + t0) / t0) ** (-0.049))


def pool_draws(cfg: dict, n: int) -> Dict[str, np.ndarray]:
    pc = cfg["pools"]
    rng = np.random.default_rng([pc["base_seed"], pc["rng_stream"]])
    full = {k: rng.standard_normal((pc["n_instances"], 4)) for k in DRAW_KEYS}
    return {k: v[:n] for k, v in full.items()}


# --------------------------------------------------------------------------------------------- instrument
class Instrument:
    """Everything the forest phase needs, built by the spice phase (or loaded from instrument.npz)."""

    def __init__(self):
        self.pools: Dict[Tuple[str, str], Pool] = {}
        self.chains: Dict[str, Chain] = {}
        self.surfaces: Dict[str, Surface] = {}
        self.vsw_nom = np.nan

    def save(self, path: Path) -> None:
        arr = {"vsw_nom": np.array([self.vsw_nom])}
        for (d, t), p in self.pools.items():
            pre = f"pool__{d}__{t}__"
            arr[pre + "v_sp"], arr[pre + "vg_c"], arr[pre + "dln"] = p.v_sp, p.vg_c, p.dln
            arr[pre + "nom_vg"], arr[pre + "nom_ln_i"] = p.nominal.vg, p.nominal.ln_i
        for t, ch in self.chains.items():
            pre = f"chain__{t}__"
            for f in ("v_sp", "pd_vg", "pd_ln_nom", "pd_vg_c", "pd_dln", "rel_grid", "t_first", "v_mid_in", "t_second"):
                v = getattr(ch, f)
                if v is not None:
                    arr[pre + f] = v
            arr[pre + "ln_ref"] = np.array([ch.ln_ref])
        for d, s in self.surfaces.items():
            pre = f"surface__{d}__"
            arr[pre + "vg"], arr[pre + "vds"], arr[pre + "ln_i"] = s.vg, s.vds, s.ln_i.astype(np.float32)
        np.savez_compressed(path, **arr)

    @classmethod
    def load(cls, path: Path) -> "Instrument":
        z = np.load(path)
        inst = cls()
        inst.vsw_nom = float(z["vsw_nom"][0])
        for key in z.files:
            parts = key.split("__")
            if parts[0] == "pool" and parts[3] == "v_sp":
                pre = f"pool__{parts[1]}__{parts[2]}__"
                inst.pools[(parts[1], parts[2])] = Pool(v_sp=z[pre + "v_sp"], vg_c=z[pre + "vg_c"], dln=z[pre + "dln"],
                                                        nominal=Curve1D(z[pre + "nom_vg"], z[pre + "nom_ln_i"]))
            elif parts[0] == "chain" and parts[2] == "v_sp":
                t = parts[1]
                pre = f"chain__{t}__"
                get = (lambda f: z[pre + f] if pre + f in z.files else None)
                inst.chains[t] = Chain(kind=t, v_sp=get("v_sp"), pd_vg=get("pd_vg"), pd_ln_nom=get("pd_ln_nom"),
                                       pd_vg_c=get("pd_vg_c"), pd_dln=get("pd_dln"), ln_ref=float(get("ln_ref")[0]),
                                       rel_grid=get("rel_grid"), t_first=get("t_first"), v_mid_in=get("v_mid_in"),
                                       t_second=get("t_second"))
            elif parts[0] == "surface" and parts[2] == "vg":
                pre = f"surface__{parts[1]}__"
                inst.surfaces[parts[1]] = Surface(z[pre + "vg"], z[pre + "vds"], z[pre + "ln_i"].astype(float))
        return inst


class CellModel:
    """Bounds of one cell set (lower type, upper type) and divider design."""

    def __init__(self, inst: Instrument, cfg: dict, design: str, cell_set: str):
        cs = cfg["cell_sets"][cell_set]
        c = cfg["circuit"]
        self.design, self.cell_set = design, cell_set
        self.types = {0: cs["lower"], 1: cs["upper"]}
        self.pool = {k: inst.pools[(design, t)] for k, t in self.types.items()}
        self.chain = {k: inst.chains[t] for k, t in self.types.items()}
        self.surf = inst.surfaces[design]
        self.map_curve = self.pool[0].nominal
        self.v_sp_nom = {k: float(p.v_sp[0]) for k, p in self.pool.items()}
        self.d_v, self.g_lo, self.g_hi, self.g_max = c["d_v"], c["g_lo_us"], c["g_hi_us"], c["g_max_us"]
        cut = cfg["chains"]["u_cut_lsb"]
        self.u_cut = {k: (np.inf if t == "L6T" else cut[t] / LSB) for k, t in self.types.items()}

    def targets(self, theta: np.ndarray, kind: np.ndarray, idx: np.ndarray):
        v_t = map_voltage(self.map_curve, theta, 1.0, self.d_v, self.g_lo, self.g_hi)
        g = np.empty(theta.shape)
        clipped = np.zeros(theta.shape, dtype=bool)
        for k in (0, 1):
            sel = kind == k
            if sel.any():
                g[sel], clipped[sel] = program_targets(self.pool[k], idx[sel], v_t[sel], self.v_sp_nom[k] + self.d_v,
                                                       self.g_max)
        return g, clipped

    def node_setting(self, node: str, a_hat: float) -> Tuple[float, float, float]:
        """(PCM conductance divisor, V_SL - V_sp,nom, input-map factor)."""
        if node == "supply":
            return 1.0, self.d_v / a_hat, 1.0
        if node == "ideal":
            return a_hat, self.d_v, 1.0
        if node == "dac_remap":
            return 1.0, self.d_v, a_hat
        if node == "t0":
            return 1.0, self.d_v, 1.0
        raise ValueError(node)

    def theta_eff(self, g_now: np.ndarray, kind: np.ndarray, idx: np.ndarray, node: str, a_hat: float) -> np.ndarray:
        gdiv, d_eff, m = self.node_setting(node, a_hat)
        x = np.empty(g_now.shape)
        for k in (0, 1):
            sel = kind == k
            if sel.any():
                x[sel] = effective_x(self.pool[k], idx[sel], g_now[sel] / gdiv, self.v_sp_nom[k] + d_eff,
                                     self.map_curve, m, self.d_v, self.g_lo, self.g_hi)
        return x

    def branch_states(self, g_now: np.ndarray, idx: np.ndarray, node: str, a_hat: float) -> Tuple[BranchState, BranchState]:
        gdiv, d_eff, m = self.node_setting(node, a_hat)
        out = []
        for k in (0, 1):
            out.append(BranchState(surf=self.surf, pool=self.pool[k], chain=self.chain[k], idx=idx, g_us=g_now / gdiv,
                                   v_sl=self.v_sp_nom[k] + d_eff, map_curve=self.map_curve, map_factor=m,
                                   d_v=self.d_v, g_lo=self.g_lo, g_hi=self.g_hi, u_cut=self.u_cut[k]))
        return out[0], out[1]


def pcm_states(g_target: np.ndarray, draws: dict, cond: str, t_idx: int, t_inf: float, p: PCMParams,
               rep: Scheme) -> Tuple[np.ndarray, float]:
    if cond == "unif":
        if t_inf == 0:
            return g_target.copy(), 1.0
        a = a_uniform(t_inf, p.t0)
        return a * g_target, a
    g_prog = apply_programming_noise(g_target, draws["z_prog"][:, 0], p)
    if t_inf == 0:
        return g_prog, 1.0
    nu = drift_coefficients(g_target, draws["z_nu"][:, 0], p)
    g = apply_drift_and_read_noise(g_prog, nu, t_inf, draws["z_read"][t_idx][:, 0], p)
    return g, replica_factor(rep, draws, "all", t_idx, t_inf, p)


def eval_points(cfg: dict) -> List[Tuple[str, int, float, str]]:
    """(time label, index into the E0.2 time list, t_inf, node)."""
    times = cfg["times_inference_s"]
    labels = list(times)
    pts = [("t0", 0, 0.0, "t0")]
    for t in cfg["times_evaluated"]:
        for node in NODES:
            pts.append((t, labels.index(t), float(times[t]), node))
    return pts


# --------------------------------------------------------------------------------------------- SPICE pieces
def pd_chain_curves(lib: Path, cfg: dict, mm: list) -> Tuple[np.ndarray, np.ndarray]:
    c = cfg["circuit"]
    pd = c["pulldown"]
    vg, cur = iv_sweep(lib, pd["device"], pd["w_um"], pd["l_um"], [cfg["ml"]["v_mid"]] * len(mm), mm, 0.0, c["vdd"], 0.0005)
    return vg, np.log(np.maximum(cur, 1e-30))


def build_instrument(cfg: dict, lib: Path, n: int, designs: List[dict], log) -> Tuple[Instrument, dict]:
    c = cfg["circuit"]
    pc = cfg["pools"]
    sw = pc["sweep"]
    pd = c["pulldown"]
    inv = c["inverter"]
    i_sw = 100e-9 * pd["w_um"] / pd["l_um"]
    inst = Instrument()
    raw: dict = {"designs": {}}
    t0 = time.time()
    z = pool_draws(cfg, n)
    mm = {k: [None] + mm_rows(v) for k, v in z.items()}
    vsw, pd_ln = {}, {}
    for side in ("lo", "hi"):
        vg, cur = iv_sweep(lib, pd["device"], pd["w_um"], pd["l_um"], [c["vdd"]] * (n + 1), mm[f"pd_{side}"],
                           sw["vg_start"], sw["vg_stop"], 0.0005)
        vsw[side] = np.array([switching_voltage(vg, cur[k], i_sw) for k in range(n + 1)])
        pd_vg, pd_ln[side] = pd_chain_curves(lib, cfg, mm[f"pd_{side}"])
    inst.vsw_nom = float(vsw["lo"][0])
    ln_ref = float(np.interp(inst.vsw_nom, pd_vg, pd_ln["lo"][0]))
    isw = {"vg_start": 0.0, "vg_stop": c["vdd"], "vg_step": pc["sweep"]["inverter_step"]}
    y, zu = transfer(lib, inv, mm["un"], mm["up"], c["vdd"], isw)
    _, zb1 = transfer(lib, inv, mm["b1n"], mm["b1p"], c["vdd"], isw)
    _, zb2 = transfer(lib, inv, mm["b2n"], mm["b2p"], c["vdd"], isw)
    v_y = np.array([input_at_output(y, zu[k], vsw["hi"][k]) for k in range(n + 1)])
    v_w = np.array([input_at_output(y, zb2[k], vsw["lo"][k]) for k in range(n + 1)])
    v_x = np.array([input_at_output(y, zb1[k], v_w[k]) for k in range(n + 1)])
    v_sp = {"L6T": vsw["lo"], "UINV": v_y, "LBUF": v_x}
    raw["v_sp"], raw["pd"], raw["pd_vg"] = v_sp, pd_ln, pd_vg

    def rel(trans: np.ndarray, centre: np.ndarray) -> np.ndarray:
        return np.vstack([np.interp(centre[k] + REL_GRID, y, trans[k]) for k in range(n + 1)]).astype(np.float32)

    def pd_delta(ln: np.ndarray) -> np.ndarray:
        return np.vstack([np.interp(PD_GRID_C, pd_vg, ln[k]) - np.interp(PD_GRID_C, pd_vg, ln[0])
                          for k in range(n + 1)]).astype(np.float32)

    base = {"pd_vg": pd_vg, "pd_ln_nom": pd_ln["lo"][0], "pd_vg_c": PD_GRID_C, "ln_ref": ln_ref}
    inst.chains["L6T"] = Chain(kind="L6T", v_sp=v_sp["L6T"], pd_dln=pd_delta(pd_ln["lo"]), **base)
    inst.chains["UINV"] = Chain(kind="UINV", v_sp=v_y, pd_dln=pd_delta(pd_ln["hi"]), rel_grid=REL_GRID,
                                t_first=rel(zu, v_y), **base)
    inst.chains["LBUF"] = Chain(kind="LBUF", v_sp=v_x, pd_dln=pd_delta(pd_ln["lo"]), rel_grid=REL_GRID,
                                t_first=rel(zb1, v_x), v_mid_in=v_w, t_second=rel(zb2, v_w), **base)
    log(f"pull-downs and inverters: V_sw,nom {inst.vsw_nom:.6f} V, V_Y*,nom {v_y[0]:.6f} V, V_X*,nom {v_x[0]:.6f} V "
        f"({time.time() - t0:.0f} s)")
    div_mm = {"L6T": mm["div_lo"], "LBUF": mm["div_lo"], "UINV": mm["div_hi"]}
    for d in designs:
        did = d["id"]
        raw["designs"][did] = {}
        for t in TYPES:
            t1 = time.time()
            vg, cur = iv_sweep(lib, c["divider_device"], d["w_div_um"], d["l_div_um"], list(v_sp[t]), div_mm[t],
                               sw["vg_start"], sw["vg_stop"], sw["vg_step"])
            nom = Curve1D(vg, np.log(np.maximum(cur[0], 1e-30)))
            lo_v = max(0.0, float(nom.v_at(np.array([1e-10]))[0]) - 0.1)
            hi_v = nom.v_at(np.array([5e-5]))[0]
            hi_v = c["vdd"] if not np.isfinite(hi_v) else min(c["vdd"], float(hi_v) + 0.1)
            lo_v = np.floor(lo_v / pc["coarse_step"]) * pc["coarse_step"]
            nominal, vg_c, dln = compress_curves(vg, cur, (lo_v, hi_v), pc["coarse_step"])
            inst.pools[(did, t)] = Pool(v_sp=v_sp[t], vg_c=vg_c, dln=dln, nominal=nominal)
            raw["designs"][did][t] = (vg, cur)
            log(f"{did} {t}: {n} instances, window {lo_v:.3f}-{hi_v:.3f} V ({time.time() - t1:.0f} s)")
        t1 = time.time()
        sf = cfg["surface"]
        vds_grid = np.round(np.arange(sf["vds_start"], sf["vds_stop"] + 1e-9, sf["vds_step"]), 6)
        vgs, curs = iv_sweep(lib, c["divider_device"], d["w_div_um"], d["l_div_um"], list(vds_grid),
                             [None] * vds_grid.size, sw["vg_start"], sw["vg_stop"], sf["vg_step"])
        ln_s = np.log(np.maximum(curs, 1e-30)).astype(np.float32).astype(float)   # stored precision
        inst.surfaces[did] = Surface(vgs, vds_grid, ln_s)
        log(f"{did}: I-V surface {curs.shape} ({time.time() - t1:.0f} s)")
    return inst, raw


def lbuf_netlist(lib: Path, div: dict, inv: dict, g_us: List[float], v_sl: float, vdd: float, sweep: dict) -> str:
    lines = ["* lower branches with a two-inverter gain stage", f'.include "{Path(lib).resolve()}"',
             f"vsl sl 0 {v_sl:.12g}", f"vdd vdd 0 {vdd:.9g}", "vdl dl 0 0"]
    for k, g in enumerate(g_us):
        lines.append(f"rp{k} sl x{k} {1.0 / (float(g) * 1e-6):.12g}")
        lines.append(f"xd{k} x{k} dl 0 0 sky130_fd_pr__{div['device']} w={div['w_um']:.6g} l={div['l_um']:.6g}")
        for s, (a, b) in enumerate(((f"x{k}", f"w{k}"), (f"w{k}", f"z{k}"))):
            lines.append(f"xp{s}_{k} {b} {a} vdd vdd sky130_fd_pr__{inv['p_dev']} w={inv['wp']:.6g} l={inv['lp']:.6g}")
            lines.append(f"xn{s}_{k} {b} {a} 0 0 sky130_fd_pr__{inv['n_dev']} w={inv['wn']:.6g} l={inv['ln']:.6g}")
    lines += [f".dc vdl {sweep['vg_start']:.9g} {sweep['vg_stop']:.9g} {sweep['vg_step']:.9g}", ".end"]
    return "\n".join(lines) + "\n"


def fine_bound(netlist_fn, coarse: np.ndarray, out_node: str, k: int, level: float, rising: bool) -> float:
    """Re-sweep +-1 mV around a coarse estimate with a 2 uV step and interpolate the crossing of `level`."""
    vec = run_netlist(netlist_fn({"vg_start": coarse - 0.001, "vg_stop": coarse + 0.001, "vg_step": 2e-6}))
    vdl, zz = vec["v(dl)"], vec[f"v({out_node}{k})"]
    return float(np.interp(level, zz, vdl)) if rising else float(np.interp(-level, -zz, vdl))


def check_c1_c3(cfg: dict, lib: Path, inst: Instrument, designs: List[dict]) -> Tuple[dict, dict]:
    c = cfg["circuit"]
    inv = c["inverter"]
    sweep = {"vg_start": 0.0, "vg_stop": c["vdd"], "vg_step": 0.0005}
    g_chk = [2.5, 12.5, 22.5]
    a1 = a_uniform(float(cfg["times_inference_s"]["1y"]), cfg["pcm_model"]["t0"])
    c1, c3 = {}, {}
    for d in designs:
        did = d["id"]
        div = {"device": c["divider_device"], "w_um": d["w_div_um"], "l_um": d["l_div_um"]}
        surf = inst.surfaces[did]
        for t in TYPES:
            pool = inst.pools[(did, t)]
            v_sp = float(pool.v_sp[0])
            worst1, worst3 = 0.0, 0.0
            for v_sl in (v_sp + c["d_v"], v_sp + c["d_v"] / a1):
                b_iv = pool.nominal.v_at(np.array(g_chk) * 1e-6 * (v_sl - v_sp))
                node = None
                if t == "L6T":
                    vdl, vx = branch_dc(lib, c["divider_device"], d["w_div_um"], d["l_div_um"], g_chk, v_sl,
                                        sweep["vg_start"], sweep["vg_stop"], sweep["vg_step"])
                    b_br = np.array([np.interp(-inst.vsw_nom, -vx[k], vdl) for k in range(len(g_chk))])
                    node = vx
                elif t == "UINV":
                    vec = run_netlist(upper_branch_netlist(lib, div, inv, g_chk, v_sl, c["vdd"], sweep))
                    vdl = vec["v(dl)"]
                    node = np.vstack([vec[f"v(y{k})"] for k in range(len(g_chk))])
                    b_br = np.array([fine_bound(lambda sw_, k=k: upper_branch_netlist(lib, div, inv, [g_chk[k]], v_sl,
                                                                                      c["vdd"], sw_),
                                                np.interp(inst.vsw_nom, vec[f"v(z{k})"], vdl), "z", 0, inst.vsw_nom, True)
                                     for k in range(len(g_chk))])
                else:
                    vec = run_netlist(lbuf_netlist(lib, div, inv, g_chk, v_sl, c["vdd"], sweep))
                    vdl = vec["v(dl)"]
                    b_br = np.array([fine_bound(lambda sw_, k=k: lbuf_netlist(lib, div, inv, [g_chk[k]], v_sl, c["vdd"], sw_),
                                                np.interp(-inst.vsw_nom, -vec[f"v(z{k})"], vdl), "z", 0, inst.vsw_nom, False)
                                     for k in range(len(g_chk))])
                worst1 = max(worst1, float(np.max(np.abs(b_br - b_iv))) * 1e3)
                if node is not None:
                    for k, g in enumerate(g_chk):
                        sel = (node[k] > v_sp - 0.25) & (node[k] < v_sp + 0.05)
                        x_s = surf.node(np.full(int(sel.sum()), g), np.full(int(sel.sum()), v_sl), vdl[sel])
                        worst3 = max(worst3, float(np.max(np.abs(x_s - node[k][sel]))) * 1e6)
            c1[f"{did}/{t}"] = {"max_abs_diff_mV": worst1, "pass": bool(worst1 <= 0.05)}
            if t != "LBUF":
                c3[f"{did}/{t}"] = {"max_abs_diff_uV": worst3, "pass": bool(worst3 <= 20.0)}
    return c1, c3


def check_c2(cfg: dict, inst: Instrument, designs: List[dict]) -> dict:
    out = {}
    theta = np.arange(16) / 15.0
    p_t0 = cfg["pcm_model"]["t0"]
    for d in designs:
        for cs_name in cfg["cell_sets"]:
            cm = CellModel(inst, cfg, d["id"], cs_name)
            for k in (0, 1):
                kind = np.full(16, k)
                idx = np.zeros(16, dtype=int)
                g_t, _ = cm.targets(theta, kind, idx)
                worst = {nd: 0.0 for nd in NODES}
                worst["t0"] = float(np.max(np.abs(LSB * (cm.theta_eff(g_t, kind, idx, "t0", 1.0) - theta))))
                for tlab in cfg["times_evaluated"]:
                    a = a_uniform(float(cfg["times_inference_s"][tlab]), p_t0)
                    for nd in NODES:
                        e = LSB * (cm.theta_eff(a * g_t, kind, idx, nd, a) - theta)
                        worst[nd] = max(worst[nd], float(np.max(np.abs(e))))
                key = f"{d['id']}/{cs_name}/{cm.types[k]}"
                checked = ["t0", "supply", "ideal"] + (["dac_remap"] if k == 0 else [])
                out[key] = {"max_abs_lsb": worst, "checked": checked,
                            "pass": bool(all(worst[nd] <= 0.001 for nd in checked))}
                if k == 1:
                    out[key]["shared_dl_residual_dac_remap_max_lsb"] = worst["dac_remap"]
    return out


def check_c6(cfg: dict, inst: Instrument, raw: dict, designs: List[dict]) -> dict:
    out = {}
    theta = np.arange(16) / 15.0
    a1 = a_uniform(float(cfg["times_inference_s"]["1y"]), cfg["pcm_model"]["t0"])
    g = 2.5 + 20 * theta
    for d in designs:
        for t in TYPES:
            pool = inst.pools[(d["id"], t)]
            vg, cur = raw["designs"][d["id"]][t]
            ln = np.maximum.accumulate(np.log(np.maximum(cur, 1e-30)), axis=1)
            worst = 0.0
            n = cur.shape[0]
            for scale in (1.0, a1):
                ic = g * 1e-6 * cfg["circuit"]["d_v"] * scale
                y = np.log(ic)
                v_raw = np.vstack([np.interp(y, ln[k], vg) for k in range(n)])
                v_rec = pool.v_at(np.repeat(np.arange(n), 16), np.tile(ic, n)).reshape(n, 16)
                worst = max(worst, float(np.max(np.abs(v_raw - v_rec))) * 1e6)
            out[f"pool {d['id']}/{t}"] = {"max_abs_diff_uV": worst, "pass": bool(worst <= 2.0)}
    vsw = inst.vsw_nom
    for t, ch in inst.chains.items():
        side = "hi" if t == "UINV" else "lo"
        ln = raw["pd"][side]
        n = ln.shape[0]
        v = vsw + np.linspace(-0.25, 0.05, 61)
        idx = np.repeat(np.arange(n), v.size)
        vv = np.tile(v, n)
        w_rec = ch.w(idx, vv)
        w_raw = np.exp(np.concatenate([np.interp(v, raw["pd_vg"], ln[k]) for k in range(n)]) - ch.ln_ref)
        worst = float(np.max(np.abs(np.log(w_rec / w_raw))))
        out[f"chain {t}"] = {"max_abs_dln_w": worst, "pass": bool(worst <= 0.002)}
    return out


def check_c5(datasets: List[str], bits: int) -> dict:
    worst_mis, worst_int = 0, 1.0
    for ds in datasets:
        b = dataset_bundle(ds, 0)
        rows, x, rf = b["rows"], b["x_te"], b["rf"]
        theta = quantize(rows.cons_thr, bits)
        pred, integ = cam_predict(x, rows, theta)
        ref = traverse_quantized(rf, x, bits).argmax(axis=1)
        worst_mis = max(worst_mis, int(np.sum(pred != ref)))
        worst_int = min(worst_int, integ)
    return {"mismatched_predictions": worst_mis, "min_integrity": worst_int,
            "pass": bool(worst_mis == 0 and worst_int == 1.0)}


# --------------------------------------------------------------------------------------------- forest pieces
def forest_state(cfg: dict, ds_idx: int, split: int, seed: int, k: int, n_pool: int):
    draws = make_draws([cfg["base_seed"], ds_idx, split, seed], k, len(cfg["times_inference_s"]), cfg["replica_cells"])
    idx = np.random.default_rng([cfg["base_seed"], ds_idx, split, seed, 4]).integers(1, n_pool + 1, k)
    return draws, idx


def theta_effs(cm: CellModel, theta: np.ndarray, kind: np.ndarray, idx: np.ndarray, draws: dict, cond: str,
               pts: List[tuple], p: PCMParams, rep: Scheme):
    g_t, clipped = cm.targets(theta, kind, idx)
    out = []
    for tlab, t_idx, t_inf, node in pts:
        g_now, a_hat = pcm_states(g_t, draws, cond, t_idx, t_inf, p, rep)
        out.append((tlab, node, a_hat, cm.theta_eff(g_now, kind, idx, node, a_hat), g_now))
    return out, g_t, clipped


# --------------------------------------------------------------------------------------------- V1
def v1_cases(cfg: dict, inst: Instrument, designs: List[dict], n_pool: int, per_group: int, log) -> List[dict]:
    names = list(cfg["datasets"])
    ds, split, seed = "breast_cancer", 0, 0
    b = dataset_bundle(ds, split)
    rows, x = b["rows"], b["x_te"]
    k = rows.cons_thr.size
    theta = quantize(rows.cons_thr, cfg["bits"])
    kind = rows.cons_kind
    p = PCMParams.from_dict(cfg["pcm_model"])
    rep = Scheme(name="replica", kind="absolute", devices=1, g_lo=cfg["circuit"]["g_lo_us"], g_hi=cfg["circuit"]["g_hi_us"])
    draws, idx = forest_state(cfg, names.index(ds), split, seed, k, n_pool)
    times = list(cfg["times_inference_s"])
    t1y = float(cfg["times_inference_s"]["1y"])
    pts = [("t0", 0, 0.0, "t0"), ("1y", times.index("1y"), t1y, "dac_remap"), ("1y", times.index("1y"), t1y, "supply")]
    n_feat = x.shape[1]
    w0 = inst.chains["L6T"].w0()
    cases = []
    for d in designs:
        cm = CellModel(inst, cfg, d["id"], "6T2M")
        res, _, _ = theta_effs(cm, theta, kind, idx, draws, "all", pts, p, rep)
        for tlab, node, a_hat, th, g_now in res:
            m_v = violation_match(x, rows, th)
            lo_st, up_st = cm.branch_states(g_now, idx, node, a_hat)
            _, s, (n_idx, r_idx), (pair, kk_all, w_all) = crosstalk_match(x, rows, th, m_v, lo_st, up_st, n_feat, w0)
            sel = np.where((s >= 0.3) & (s <= 3.0))[0]
            if sel.size == 0:
                log(f"V1 {d['id']} {node}: no candidate pairs")
                continue
            order = sel[np.argsort(np.log(s[sel]))]
            pick = order[np.unique(np.round(np.linspace(0, order.size - 1, min(per_group, order.size))).astype(int))]
            gdiv, d_eff, m = cm.node_setting(node, a_hat)
            for q in pick:
                r = r_idx[q]
                bounds = []
                for kk, ww in zip(kk_all[pair == q], w_all[pair == q]):
                    xv = float(x[n_idx[q], rows.cons_feat[kk]])
                    kd = int(kind[kk])
                    v_dl = float(map_voltage(cm.map_curve, np.array([xv]), m, cm.d_v, cm.g_lo, cm.g_hi)[0])
                    bounds.append({"kind": kd, "feat": int(rows.cons_feat[kk]), "x": xv, "theta_eff": float(th[kk]),
                                   "g_us": float(g_now[kk] / gdiv), "idx": int(idx[kk]), "v_dl": v_dl, "w_model": float(ww)})
                cases.append({"design": d["id"], "time": tlab, "node": node, "a_hat": a_hat, "sample": int(n_idx[q]),
                              "row": int(r), "S": float(s[q]), "n_feat": n_feat, "bounds": bounds,
                              "v_sl_lo": cm.v_sp_nom[0] + d_eff, "v_sl_hi": cm.v_sp_nom[1] + d_eff,
                              "w_div_um": d["w_div_um"], "l_div_um": d["l_div_um"]})
    return cases


def v1_netlist(cfg: dict, lib: Path, cases: List[dict], z: Dict[str, np.ndarray], spec: MLSpec, t_stop: float,
               n_feat: int) -> Tuple[List[str], List[str]]:
    c = cfg["circuit"]
    inv = c["inverter"]
    pd = f"sky130_fd_pr__{c['pulldown']['device']} w={c['pulldown']['w_um']:.6g} l={c['pulldown']['l_um']:.6g}"
    lines = header(lib, "E4 V1 full-row transients") + [f"vdd vdd 0 {c['vdd']:.9g}"]
    names: List[str] = []
    for i, cs in enumerate(cases):
        dv = f"sky130_fd_pr__{c['divider_device']} w={cs['w_div_um']:.6g} l={cs['l_div_um']:.6g}"
        lines += [f"vsllo{i} sllo{i} 0 {cs['v_sl_lo']:.12g}", f"vslhi{i} slhi{i} 0 {cs['v_sl_hi']:.12g}",
                  f"cml{i} ml{i} 0 {spec.c_lumped(n_feat):.9g}"]
        for j, b in enumerate(cs["bounds"]):
            q = b["idx"] - 1
            lines.append(f"vdl{i}_{j} dl{i}_{j} 0 {b['v_dl']:.12g}")
            r = 1.0 / (max(b["g_us"], 1e-6) * 1e-6)
            if b["kind"] == 0:
                gate = f"x{i}_{j}"
                lines += [f"rp{i}_{j} sllo{i} x{i}_{j} {r:.12g}",
                          f"xd{i}_{j} x{i}_{j} dl{i}_{j} 0 0 {dv}{mm_string(mm_of(z['div_lo'][q]))}",
                          f"xpd{i}_{j} ml{i} x{i}_{j} 0 0 {pd}{mm_string(mm_of(z['pd_lo'][q]))}"]
            else:
                gate = f"z{i}_{j}"
                lines += [f"rp{i}_{j} slhi{i} y{i}_{j} {r:.12g}",
                          f"xd{i}_{j} y{i}_{j} dl{i}_{j} 0 0 {dv}{mm_string(mm_of(z['div_hi'][q]))}",
                          f"xip{i}_{j} z{i}_{j} y{i}_{j} vdd vdd sky130_fd_pr__{inv['p_dev']} w={inv['wp']:.6g} "
                          f"l={inv['lp']:.6g}{mm_string(mm_of(z['up'][q]))}",
                          f"xin{i}_{j} z{i}_{j} y{i}_{j} 0 0 sky130_fd_pr__{inv['n_dev']} w={inv['wn']:.6g} "
                          f"l={inv['ln']:.6g}{mm_string(mm_of(z['un'][q]))}",
                          f"xpd{i}_{j} ml{i} z{i}_{j} 0 0 {pd}{mm_string(mm_of(z['pd_hi'][q]))}"]
            lines.append(f".meas tran g{i}_{j} find v({gate}) at=2e-9")
            names.append(f"g{i}_{j}")
        for j in range(2 * n_feat - len(cs["bounds"])):
            lines.append(f"xpo{i}_{j} ml{i} 0 0 0 {pd}")
        lines += [f".ic v(ml{i})={spec.v_pre:.9g}", f".meas tran tc{i} when v(ml{i})={spec.v_ref:.9g} fall=1"]
    return lines + [tran_card(spec, t_stop)], names


def t_s_of(cfg: dict, lib: Path, spec: MLSpec, vsw: float, n_feat: int) -> float:
    c = cfg["circuit"]
    pd = f"sky130_fd_pr__{c['pulldown']['device']} w={c['pulldown']['w_um']:.6g} l={c['pulldown']['l_um']:.6g}"
    lines = header(lib, "t_s") + [f"vgc gc 0 {vsw:.12g}", f"cml0 ml0 0 {spec.c_lumped(n_feat):.9g}",
                                  f"xpc ml0 gc 0 0 {pd}"]
    lines += [f"xpo{j} ml0 0 0 0 {pd}" for j in range(2 * n_feat - 1)]
    lines += [f".ic v(ml0)={spec.v_pre:.9g}", f".meas tran tc0 when v(ml0)={spec.v_ref:.9g} fall=1", tran_card(spec, 5e-6)]
    return float(run_meas(lines, ["tc0"])[0])


def model_gates(cfg: dict, inst: Instrument, cs: dict) -> List[float]:
    """Model gate voltage of every constrained pull-down of a V1 case (for diagnosis)."""
    cm = CellModel(inst, cfg, cs["design"], "6T2M")
    out = []
    for b in cs["bounds"]:
        k = b["kind"]
        v_sl = cs["v_sl_lo"] if k == 0 else cs["v_sl_hi"]
        idx = np.array([b["idx"]])
        node = solve_nodes(cm.surf, cm.pool[k], idx, np.array([b["g_us"]]), np.array([v_sl]), np.array([b["v_dl"]]))
        out.append(float(cm.chain[k].gate(idx, node)[0]))
    return out


# --------------------------------------------------------------------------------------------- phase spice
def phase_spice(cfg: dict, cfg_path: Path, smoke: bool) -> int:
    run = make_run_dir(ROOT / "results", f"{cfg['run_prefix']}_spice", smoke)
    log = make_logger(run / "run.log")
    info = {"experiment": cfg["run_prefix"], "phase": "spice", "smoke": smoke,
            "start_jst": now_jst().isoformat(timespec="seconds"), "environment": environment_info(),
            "ngspice": ngspice_version(), "sha256": hashes(cfg_path, cfg)}
    write_json(run / "run_info.json", info)
    write_json(run / "config_used.json", cfg)
    lib = ROOT / cfg["models"]["lib"]
    n = cfg["smoke"]["n_instances"] if smoke else cfg["pools"]["n_instances"]
    designs = [d for d in cfg["designs"] if (not smoke) or d["id"] in cfg["smoke"]["designs"]]
    t_start = time.time()
    log(f"E4 spice phase; run dir {run.name}; {n} instances per branch type; designs {[d['id'] for d in designs]}")
    inst, raw = build_instrument(cfg, lib, n, designs, log)

    checks: Dict[str, dict] = {}
    t0 = time.time()
    checks["C1"], checks["C3"] = check_c1_c3(cfg, lib, inst, designs)
    checks["C2"] = check_c2(cfg, inst, designs)
    refs = {"L6T": 22.15e-3, "UINV": 10.57e-3, "LBUF": 10.57e-3}
    checks["C4"] = {}
    stats = []
    for t in TYPES:
        sd = float(np.std(raw["v_sp"][t][1:], ddof=1))
        tol = max(0.15, 3.0 / np.sqrt(2.0 * (n - 1)))
        checks["C4"][t] = {"sigma_mV": 1e3 * sd, "reference_mV": 1e3 * refs[t], "tolerance": tol,
                           "pass": bool(abs(sd / refs[t] - 1) <= tol)}
        stats.append({"type": t, "v_sp_nom_V": float(raw["v_sp"][t][0]), "sigma_v_sp_mV": 1e3 * sd,
                      "mean_v_sp_minus_nom_mV": 1e3 * float(np.mean(raw["v_sp"][t][1:] - raw["v_sp"][t][0]))})
    checks["C5"] = check_c5(cfg["smoke"]["datasets"] if smoke else list(cfg["datasets"]), cfg["bits"])
    checks["C6"] = check_c6(cfg, inst, raw, designs)
    log(f"checks C1-C6 computed ({time.time() - t0:.0f} s)")
    for name in ("C1", "C2", "C3", "C4", "C6"):
        for key, v in checks[name].items():
            if not v["pass"]:
                log(f"  {name} {key} FAIL: {v}")
    write_csv(run / "pool_stats.csv", stats)

    # V1 row validation
    t0 = time.time()
    m = cfg["ml"]
    pdc = cfg["circuit"]["pulldown"]
    spec = MLSpec(v_pre=m["v_pre"], v_ref=m["v_ref"], c_sa=m["c_sa_f"], c_wire=m["c_wire_per_cell_f"],
                  pd_w=pdc["w_um"], pd_l=pdc["l_um"])
    per_group = cfg["smoke"]["v1_cases_per_group"] if smoke else 8
    cases = v1_cases(cfg, inst, designs, n, per_group, log)
    n_feat = cases[0]["n_feat"] if cases else 30
    t_s = t_s_of(cfg, lib, spec, inst.vsw_nom, n_feat)
    z = pool_draws(cfg, n)
    tc = np.full(len(cases), np.nan)
    gates_sim: List[List[float]] = []
    chunk = 12
    for s0 in range(0, len(cases), chunk):
        sub = cases[s0:s0 + chunk]
        lines, gnames = v1_netlist(cfg, lib, sub, z, spec, 6.0 * t_s, n_feat)
        vals = run_meas(lines, [f"tc{i}" for i in range(len(sub))] + gnames)
        tc[s0:s0 + len(sub)] = vals[:len(sub)]
        g_iter = iter(vals[len(sub):])
        for cs in sub:
            gates_sim.append([float(next(g_iter)) for _ in cs["bounds"]])
    w0 = inst.chains["L6T"].w0()
    v1_rows, v1_bounds = [], []
    for ci, (cs, t_sim, gsim) in enumerate(zip(cases, tc, gates_sim)):
        t_pred = t_s * (1 + (2 * n_feat - 1) * w0) / cs["S"]
        r = float(np.log(t_sim / t_pred)) if np.isfinite(t_sim) else np.inf
        v1_rows.append({"case": ci, "design": cs["design"], "time": cs["time"], "node": cs["node"], "sample": cs["sample"],
                        "row": cs["row"], "n_bounds": len(cs["bounds"]), "S": cs["S"], "t_s_ns": 1e9 * t_s,
                        "t_pred_ns": 1e9 * t_pred, "t_sim_ns": 1e9 * t_sim if np.isfinite(t_sim) else np.nan,
                        "ln_ratio": r, "pass": bool(abs(r) <= 0.05),
                        "decision_pred_mismatch": bool(cs["S"] >= 1 + (2 * n_feat - 1) * w0),
                        "decision_sim_mismatch": bool(np.isfinite(t_sim) and t_sim <= t_s)})
        for b, gm, gs in zip(cs["bounds"], model_gates(cfg, inst, cs), gsim):
            v1_bounds.append({"case": ci, "kind": b["kind"], "u_lsb": LSB * ((b["x"] - b["theta_eff"]) if b["kind"] == 0
                                                                             else (b["theta_eff"] - b["x"])),
                              "w_model": b["w_model"], "gate_model_V": gm, "gate_sim_2ns_V": gs,
                              "gate_diff_mV": 1e3 * (gm - gs)})
    write_csv(run / "v1_rows.csv", v1_rows)
    write_csv(run / "v1_bounds.csv", v1_bounds)
    write_json(run / "v1_cases.json", {"t_s_s": t_s, "n_feat": n_feat, "w0": w0, "cases": cases})
    frac = float(np.mean([r["pass"] for r in v1_rows])) if v1_rows else 0.0
    disagree = [r for r in v1_rows if r["decision_pred_mismatch"] != r["decision_sim_mismatch"]]
    checks["V1"] = {"cases": len(v1_rows), "pass_fraction": frac,
                    "max_abs_ln_ratio": float(np.max(np.abs([r["ln_ratio"] for r in v1_rows]))) if v1_rows else None,
                    "decision_disagreements": len(disagree),
                    "decision_disagreements_near_tie": int(sum(abs(np.log(r["S"])) < 0.05 for r in disagree)),
                    "t_s_ns": 1e9 * t_s, "pass": bool(v1_rows) and frac >= 0.9}
    log(f"V1: {len(v1_rows)} cases, pass fraction {frac:.3f}, max |ln ratio| {checks['V1']['max_abs_ln_ratio']}, "
        f"t_s({n_feat}) {1e9 * t_s:.3f} ns ({time.time() - t0:.0f} s)")

    failed = [f"{nm} {k}" for nm in ("C1", "C2", "C3", "C4", "C6") for k, v in checks[nm].items() if not v["pass"]]
    if not checks["C5"]["pass"]:
        failed.append("C5")
    if not checks["V1"]["pass"]:
        failed.append("V1")
    checks["failed"] = failed
    checks["all_pass"] = not failed
    write_json(run / "checks.json", checks)
    inst.save(run / "instrument.npz")
    info["instrument_sha256"] = sha256_file(run / "instrument.npz")
    info["end_jst"] = now_jst().isoformat(timespec="seconds")
    info["elapsed_s"] = round(time.time() - t_start, 1)
    info["status"] = "complete" if not failed else "checks_failed"
    write_json(run / "run_info.json", info)
    log(f"spice phase {'complete' if not failed else 'CHECK FAILURE: ' + '; '.join(failed)}; "
        f"instrument sha256 {info['instrument_sha256']}; {info['elapsed_s']} s")
    return 0 if not failed else 2


# --------------------------------------------------------------------------------------------- phase forest
def stats_delta(units: Dict[str, List[float]]) -> dict:
    """Mean over datasets of per-dataset means; SE = sqrt(sum var_d / n_d) / n_datasets."""
    means, var_terms = [], []
    per = {}
    for ds, v in units.items():
        a = np.asarray(v, float)
        means.append(float(a.mean()))
        var_terms.append(float(a.var(ddof=1)) / a.size if a.size > 1 else 0.0)
        per[ds] = float(a.mean())
    m = float(np.mean(means))
    se = float(np.sqrt(np.sum(var_terms)) / len(means))
    return {"mean": m, "se": se, "ci_lo": m - 1.96 * se, "ci_hi": m + 1.96 * se, "per_dataset": per}


def classify(st: dict, sign: int) -> str:
    if st["ci_lo"] > 0:
        return "CONFIRMED" if sign > 0 else "CONTRADICTED"
    if st["ci_hi"] < 0:
        return "CONFIRMED" if sign < 0 else "CONTRADICTED"
    return "UNRESOLVED"


def evaluate_split(args) -> Tuple[List[dict], dict, dict, dict]:
    """All evaluations of one (dataset, split): returns (evals, ref row, error accumulators, clipping)."""
    cfg, inst_path, ds, ds_idx, split, n_seed, design_ids, n_pool = args
    inst = Instrument.load(inst_path)
    p = PCMParams.from_dict(cfg["pcm_model"])
    rep = Scheme(name="replica", kind="absolute", devices=1, g_lo=cfg["circuit"]["g_lo_us"], g_hi=cfg["circuit"]["g_hi_us"])
    pts = eval_points(cfg)
    arms_of = collections.defaultdict(list)
    for arm, a in cfg["arms"].items():
        arms_of[a["cell_set"]].append((arm, a["crosstalk"]))
    b = dataset_bundle(ds, split)
    rows, x, y = b["rows"], b["x_te"], b["y_te"]
    classes = b["rf"].classes_
    k = rows.cons_thr.size
    theta = quantize(rows.cons_thr, cfg["bits"])
    kind = rows.cons_kind
    n_feat = x.shape[1]
    w0 = inst.chains["L6T"].w0()
    pred_ref, _ = cam_predict(x, rows, theta)
    acc_ref = float(np.mean(np.where(pred_ref >= 0, classes[np.maximum(pred_ref, 0)], -1) == y))
    ref = {"dataset": ds, "split": split, "n_test": int(y.size), "n_rows": int(rows.row_tree.size),
           "n_constraints": int(k), "n_lower": int(np.sum(kind == 0)), "n_features": n_feat,
           "acc_float": b["acc_float"], "acc_ref": acc_ref}
    evals: List[dict] = []
    err_acc: Dict[tuple, List[float]] = {}
    clip_acc: Dict[tuple, List[int]] = {}
    for seed in range(n_seed):
        draws, idx = forest_state(cfg, ds_idx, split, seed, k, n_pool)
        for did in design_ids:
            for cs_name, arm_list in arms_of.items():
                cm = CellModel(inst, cfg, did, cs_name)
                for cond in cfg["conditions"]:
                    res, g_t, clipped = theta_effs(cm, theta, kind, idx, draws, cond, pts, p, rep)
                    if cond == "all":
                        for kd in (0, 1):
                            ca = clip_acc.setdefault((did, cs_name, cm.types[kd]), [0, 0])
                            ca[0] += int(np.sum(clipped[kind == kd]))
                            ca[1] += int(np.sum(kind == kd))
                    for tlab, node, a_hat, th, g_now in res:
                        e = LSB * (th - theta)
                        for kd in (0, 1):
                            sel = kind == kd
                            acc_e = err_acc.setdefault((did, cs_name, cm.types[kd], cond, tlab, node), [0.0, 0.0, 0])
                            acc_e[0] += float(np.sum(e[sel] ** 2))
                            acc_e[1] += float(np.sum(e[sel]))
                            acc_e[2] += int(sel.sum())
                        m_v = violation_match(x, rows, th)
                        for arm, xt in arm_list:
                            if xt:
                                lo_st, up_st = cm.branch_states(g_now, idx, node, a_hat)
                                match, _, _, _ = crosstalk_match(x, rows, th, m_v, lo_st, up_st, n_feat, w0)
                            else:
                                match = m_v
                            pred, leaf_loss, multi = vote(match, rows, classes)
                            acc = float(np.mean(pred == y))
                            evals.append({"dataset": ds, "split": split, "seed": seed, "design": did, "arm": arm,
                                          "condition": cond, "time": tlab, "node": node, "a_hat": a_hat, "acc": acc,
                                          "loss_pp": 100.0 * (acc_ref - acc), "leaf_loss": leaf_loss,
                                          "multi_match": multi, "removed_by_crosstalk": int(m_v.sum() - match.sum())})
    return evals, ref, err_acc, clip_acc


def phase_forest(cfg: dict, cfg_path: Path, smoke: bool, inst_dir: Path, workers: int) -> int:
    run = make_run_dir(ROOT / "results", cfg["run_prefix"], smoke)
    log = make_logger(run / "run.log")
    sp_info = json.loads((inst_dir / "run_info.json").read_text())
    sp_checks = json.loads((inst_dir / "checks.json").read_text())
    inst_path = inst_dir / "instrument.npz"
    info = {"experiment": cfg["run_prefix"], "phase": "forest", "smoke": smoke,
            "start_jst": now_jst().isoformat(timespec="seconds"), "environment": environment_info(),
            "sha256": hashes(cfg_path, cfg), "instrument_dir": str(inst_dir.relative_to(ROOT)),
            "instrument_sha256": sha256_file(inst_path), "workers": workers}
    try:
        import sklearn
        info["environment"]["sklearn"] = sklearn.__version__
    except ImportError:
        pass
    write_json(run / "run_info.json", info)
    write_json(run / "config_used.json", cfg)
    log(f"E4 forest phase; instrument {inst_dir.name} (sha256 {info['instrument_sha256']}; recorded by the spice phase "
        f"{sp_info.get('instrument_sha256')})")
    if not sp_checks.get("all_pass", False) or sp_info.get("smoke", False) != smoke:
        log("the spice phase reported failed checks or a different run type; stop")
        return 2
    names = list(cfg["datasets"])
    if smoke:
        datasets, splits, n_seed = cfg["smoke"]["datasets"], cfg["smoke"]["split_seeds"], cfg["smoke"]["device_seeds"]
        design_ids = list(cfg["smoke"]["designs"])
        n_pool = cfg["smoke"]["n_instances"]
    else:
        datasets, splits, n_seed = names, cfg["split_seeds"], cfg["device_seeds"]
        design_ids = [d["id"] for d in cfg["designs"]]
        n_pool = cfg["pools"]["n_instances"]
    jobs = [(cfg, inst_path, ds, names.index(ds), split, n_seed, design_ids, n_pool) for ds in datasets for split in splits]
    # largest jobs first
    size = {"synthetic": 0, "digits": 1, "breast_cancer": 2, "wine": 3, "iris": 4}
    jobs.sort(key=lambda j: size.get(j[2], 9))
    t_start = time.time()
    results = []
    if workers > 1:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(workers) as pool:
            for r in pool.imap_unordered(evaluate_split, jobs):
                results.append(r)
                log(f"{r[1]['dataset']} split {r[1]['split']}: K {r[1]['n_constraints']}, N {r[1]['n_features']}, "
                    f"ref acc {r[1]['acc_ref']:.4f} ({time.time() - t_start:.0f} s since start)")
    else:
        for j in jobs:
            r = evaluate_split(j)
            results.append(r)
            log(f"{r[1]['dataset']} split {r[1]['split']}: K {r[1]['n_constraints']}, N {r[1]['n_features']}, "
                f"ref acc {r[1]['acc_ref']:.4f} ({time.time() - t_start:.0f} s since start)")
    order = {(ds, sp): i for i, (ds, sp) in enumerate((ds, sp) for ds in datasets for sp in splits)}
    results.sort(key=lambda r: order[(r[1]["dataset"], r[1]["split"])])
    evals = [e for r in results for e in r[0]]
    refs = [r[1] for r in results]
    err_acc: Dict[tuple, List[float]] = collections.defaultdict(lambda: [0.0, 0.0, 0])
    clip_acc: Dict[tuple, List[int]] = collections.defaultdict(lambda: [0, 0])
    for r in results:
        for key, v in r[2].items():
            a = err_acc[key]
            a[0] += v[0]; a[1] += v[1]; a[2] += v[2]  # noqa: E702
        for key, v in r[3].items():
            a = clip_acc[key]
            a[0] += v[0]; a[1] += v[1]  # noqa: E702
    write_csv(run / "per_eval.csv", evals)
    write_csv(run / "refs.csv", refs)
    diag = [{"design": key[0], "cell_set": key[1], "type": key[2], "condition": key[3], "time": key[4], "node": key[5],
             "n": cnt, "rms_lsb": float(np.sqrt(ss / cnt)), "mean_lsb": sm / cnt}
            for key, (ss, sm, cnt) in sorted(err_acc.items())]
    write_csv(run / "cell_errors.csv", diag)
    write_csv(run / "clipping.csv", [{"design": k_[0], "cell_set": k_[1], "type": k_[2], "clipped": v[0],
                                      "constraints": v[1], "fraction": v[0] / max(v[1], 1)}
                                     for k_, v in sorted(clip_acc.items())])

    by = collections.defaultdict(lambda: collections.defaultdict(list))
    unit = {}
    for e in evals:
        key = (e["design"], e["arm"], e["condition"], e["time"], e["node"])
        by[key][e["dataset"]].append(e)
        unit[key + (e["dataset"], e["split"], e["seed"])] = e
    summary = []
    for key, per_ds in sorted(by.items()):
        means = {ds: float(np.mean([q["loss_pp"] for q in v])) for ds, v in per_ds.items()}
        leaf = float(np.mean([np.mean([q["leaf_loss"] for q in v]) for v in per_ds.values()]))
        worst = max(means, key=means.get)
        summary.append({"design": key[0], "arm": key[1], "condition": key[2], "time": key[3], "node": key[4],
                        "n_datasets": len(means), "mean_loss_pp": float(np.mean(list(means.values()))),
                        "max_loss_pp": means[worst], "worst_dataset": worst, "leaf_loss_mean": leaf,
                        **{f"loss_{ds}": v for ds, v in means.items()}})
    write_csv(run / "summary.csv", summary)

    def paired(key_a, key_b) -> dict:
        units = collections.defaultdict(list)
        for ds in datasets:
            for split in splits:
                for seed in range(n_seed):
                    ea = unit.get(key_a + (ds, split, seed))
                    eb = unit.get(key_b + (ds, split, seed))
                    if ea is not None and eb is not None:
                        units[ds].append(ea["loss_pp"] - eb["loss_pp"])
        return stats_delta(units)

    deltas = []
    delta_index = {}
    pts = eval_points(cfg)
    for did in design_ids:
        for arm in cfg["arms"]:
            for cond in cfg["conditions"]:
                for tlab in cfg["times_evaluated"]:
                    for a_node, b_node, lab in (("dac_remap", "supply", "dac_minus_supply"),
                                                ("dac_remap", "ideal", "dac_minus_ideal"),
                                                ("supply", "ideal", "supply_minus_ideal")):
                        st = paired((did, arm, cond, tlab, a_node), (did, arm, cond, tlab, b_node))
                        deltas.append({"design": did, "arm": arm, "condition": cond, "time": tlab, "contrast": lab,
                                       **{k_: v for k_, v in st.items() if k_ != "per_dataset"},
                                       **{f"d_{ds}": v for ds, v in st["per_dataset"].items()}})
                        delta_index[(did, arm, cond, tlab, lab)] = st
        for cond in cfg["conditions"]:
            for tlab, _, _, node in pts:
                for a_arm, b_arm, lab in (("xt", "is", "xt_minus_is"), ("gs", "gi", "gs_minus_gi"),
                                          ("xt", "gs", "xt_minus_gs")):
                    st = paired((did, a_arm, cond, tlab, node), (did, b_arm, cond, tlab, node))
                    deltas.append({"design": did, "arm": lab, "condition": cond, "time": tlab, "contrast": node,
                                   **{k_: v for k_, v in st.items() if k_ != "per_dataset"},
                                   **{f"d_{ds}": v for ds, v in st["per_dataset"].items()}})
                    delta_index[(did, lab, cond, tlab, node)] = st
    write_csv(run / "deltas.csv", deltas)

    if smoke:
        decision = {"verdict": "SMOKE (not interpreted)"}
    else:
        hyp = cfg["hypotheses"]
        h1 = {did: {**delta_index[(did, "gs", "all", "1y", "dac_minus_supply")],
                    "predicted_sign": +1 if did == "P1" else -1} for did in ("P1", "P3")}
        h2 = {did: {**delta_index[(did, "xt", "all", "1y", "dac_minus_supply")], "predicted_sign": +1}
              for did in ("P1", "P3")}
        for h in (h1, h2):
            for st in h.values():
                st["class"] = classify(st, st["predicted_sign"])
                st["material"] = bool(st["class"] == "CONFIRMED" and abs(st["mean"]) >= 0.5)
        h3 = {did: delta_index[(did, "xt_minus_is", "all", "t0", "t0")] for did in ("P1", "P3")}
        decision = {
            "H1": {"rule": hyp["H1"], "by_design": h1, "holds": all(v["class"] == "CONFIRMED" for v in h1.values())},
            "H2": {"rule": hyp["H2"], "by_design": h2, "holds": all(v["class"] == "CONFIRMED" for v in h2.values())},
            "H3": {"rule": hyp["H3"], "by_design": h3, "holds": all(v["mean"] >= 1.0 for v in h3.values())},
            "classification_rule": hyp["classification"]}
    decision["instrument_sha256"] = info["instrument_sha256"]
    write_json(run / "decision.json", decision)
    info["end_jst"] = now_jst().isoformat(timespec="seconds")
    info["elapsed_s"] = round(time.time() - t_start, 1)
    write_json(run / "run_info.json", info)
    if not smoke:
        for h in ("H1", "H2"):
            log(f"{h} " + ", ".join(f"{d_}: {v['mean']:+.3f} [{v['ci_lo']:+.3f}, {v['ci_hi']:+.3f}] {v['class']}"
                                    for d_, v in decision[h]["by_design"].items()) + f" -> holds {decision[h]['holds']}")
        log("H3 " + ", ".join(f"{d_}: {v['mean']:+.3f} [{v['ci_lo']:+.3f}, {v['ci_hi']:+.3f}]"
                              for d_, v in decision["H3"]["by_design"].items()) + f" -> holds {decision['H3']['holds']}")
    log(f"forest phase finished in {info['elapsed_s']} s")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", choices=("spice", "forest"), required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--config", default=str(ROOT / "experiments" / "E4_config.json"))
    ap.add_argument("--instrument", default=None, help="results/E4_spice_<stamp> (forest phase)")
    ap.add_argument("--workers", type=int, default=2, help="processes for the forest phase (results do not depend on it)")
    args = ap.parse_args()
    cfg_path = Path(args.config).resolve()
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    if args.phase == "spice":
        return phase_spice(cfg, cfg_path, args.smoke)
    if args.instrument is None:
        ap.error("--instrument is required for the forest phase")
    return phase_forest(cfg, cfg_path, args.smoke, (ROOT / args.instrument).resolve(), args.workers)


if __name__ == "__main__":
    sys.exit(main())
