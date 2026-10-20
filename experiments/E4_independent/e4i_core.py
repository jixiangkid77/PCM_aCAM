"""E4 independent check: shared model code (written from the E4 protocol, the config and
the checker brief only; it does not import or read src/pcm_acam/row_model.py or
experiments/E4_forest_system.py).

Contents
- paths (relative to the project root, resolved from this file's location)
- loaders for the two project functions the brief allows: trees.py (dataset_bundle,
  quantize; loaded as a stand-alone module so that no other package file is executed)
  and make_draws from acam.py (extracted from the source with ast, so that the rest of
  acam.py and its imports are not executed)
- instrument reconstruction (Catmull-Rom deviations, bisection inversion, I-V surface,
  per-instance chains)
- AIHWKit-style PCM model and replica a_hat
- bound voltages, effective thresholds, per-instance leakage weights
"""

from __future__ import annotations

import ast
import importlib.util
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]                      # project root (experiments/E4_independent/..)
INSTR_DIR = ROOT / "results" / "E4_spice_20261017_0602"
FOREST_DIR = ROOT / "results" / "E4_20261017_0618"
CONFIG_PATH = ROOT / "experiments" / "E4_config.json"
LIB_PATH = ROOT / "third_party" / "sky130_fd_pr_subset_e3b" / "sky130_subset_tt.lib.spice"

DATASETS = ["iris", "wine", "breast_cancer", "digits", "synthetic"]
TIME_NAMES = ["t0", "1h", "1d", "30d", "1y"]          # index into z_read / z_rep_read
D_V = 0.2                                              # divider headroom D (V)
G_LO_US, G_SPAN_US, G_MAX_US = 2.5, 20.0, 25.0
BASE_SEED = 20261012
POOL_SEED = (20261017, 5)
# Bisection steps. The brief fixes the method (bisection on [vg_c[0], vg_c[-1]]) but not
# the step count; 40 steps with the midpoint returned reproduce every C2 value of
# checks.json bit for bit (task 1 also runs a 64-step variant as a sensitivity check).
N_INV = 40                                             # instance curve inversion (bound voltage)
N_BISECT = 64                                          # surface inversion and node solve


def jst_now() -> datetime:
    return datetime.now(timezone(timedelta(hours=9)))


def rel(p: Path) -> str:
    """Path relative to the project root, for printing / reports."""
    try:
        return str(Path(p).resolve().relative_to(ROOT))
    except ValueError:
        return str(p)


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return json.load(f)


# ----------------------------------------------------------------------------------
# allowed project code
# ----------------------------------------------------------------------------------
_TREES = None
_MAKE_DRAWS = None


def trees_module():
    """src/pcm_acam/trees.py loaded on its own (no package __init__, no other module)."""
    global _TREES
    if _TREES is None:
        path = ROOT / "src" / "pcm_acam" / "trees.py"
        spec = importlib.util.spec_from_file_location("e4i_trees_standalone", path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod            # needed by dataclasses
        spec.loader.exec_module(mod)
        _TREES = mod
    return _TREES


def make_draws_fn():
    """make_draws from src/pcm_acam/acam.py, compiled from its own source text only."""
    global _MAKE_DRAWS
    if _MAKE_DRAWS is None:
        path = ROOT / "src" / "pcm_acam" / "acam.py"
        tree = ast.parse(path.read_text(), filename=str(path))
        fns = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "make_draws"]
        assert len(fns) == 1
        module = ast.Module(body=fns, type_ignores=[])
        ns = {"np": np, "Dict": Dict, "Tuple": Tuple}
        exec(compile(module, str(path), "exec"), ns)
        _MAKE_DRAWS = ns["make_draws"]
    return _MAKE_DRAWS


# ----------------------------------------------------------------------------------
# interpolation helpers
# ----------------------------------------------------------------------------------
def catmull_rom(dln: np.ndarray, vg_c: np.ndarray, k: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Catmull-Rom interpolation of row k of dln (coarse grid vg_c) at v (brief, section 'instrument')."""
    n_c = vg_c.size
    p = (np.asarray(v, dtype=float) - vg_c[0]) / (vg_c[1] - vg_c[0])
    p = np.clip(p, 0.0, n_c - 1.000001)
    i1 = np.floor(p).astype(np.int64)
    t = p - i1
    i0 = np.maximum(i1 - 1, 0)
    i2 = np.minimum(i1 + 1, n_c - 1)
    i3 = np.minimum(i1 + 2, n_c - 1)
    p0, p1, p2, p3 = dln[k, i0], dln[k, i1], dln[k, i2], dln[k, i3]
    return 0.5 * (2.0 * p1 + (p2 - p0) * t + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t ** 2
                  + (3.0 * p1 - p0 - 3.0 * p2 + p3) * t ** 3)


def interp_rows(x: np.ndarray, xp: np.ndarray, fp_rows: np.ndarray) -> np.ndarray:
    """Element-wise np.interp with a shared grid xp and one table row per element
    (linear, clamped outside the grid)."""
    x = np.asarray(x, dtype=float)
    n = xp.size
    i = np.clip(np.searchsorted(xp, x, side="right") - 1, 0, n - 2)
    f = np.clip((x - xp[i]) / (xp[i + 1] - xp[i]), 0.0, 1.0)
    r = np.arange(x.size)
    return (1.0 - f) * fp_rows[r, i] + f * fp_rows[r, i + 1]


# ----------------------------------------------------------------------------------
# instrument
# ----------------------------------------------------------------------------------
class Pool:
    """Divider pool of one (design, branch type): instance curves at V_DS = v_sp[k]."""

    def __init__(self, z, design: str, typ: str):
        p = f"pool__{design}__{typ}__"
        self.design, self.typ = design, typ
        self.v_sp = np.asarray(z[p + "v_sp"], dtype=float)
        self.nom_vg = np.asarray(z[p + "nom_vg"], dtype=float)
        self.nom_ln = np.asarray(z[p + "nom_ln_i"], dtype=float)
        self.vg_c = np.asarray(z[p + "vg_c"], dtype=float)
        self.dln = np.asarray(z[p + "dln"], dtype=np.float64)
        self.v_sl_nom = self.v_sp[0] + D_V

    def ln_i0(self, v):
        return np.interp(v, self.nom_vg, self.nom_ln)

    def ln_ik(self, k, v):
        v = np.asarray(v, dtype=float)
        return np.interp(v, self.nom_vg, self.nom_ln) + catmull_rom(self.dln, self.vg_c, k, v)

    def inv_k(self, k, ln_target, n_iter: int | None = None):
        """Gate voltage at which instance k conducts exp(ln_target): bisection on
        [vg_c[0], vg_c[-1]]; below the window -inf, above +inf."""
        n_iter = N_INV if n_iter is None else n_iter
        ln_target = np.asarray(ln_target, dtype=float)
        k = np.broadcast_to(np.asarray(k), ln_target.shape)
        lo = np.full(ln_target.shape, self.vg_c[0])
        hi = np.full(ln_target.shape, self.vg_c[-1])
        below = ln_target < self.ln_ik(k, lo)
        above = ln_target > self.ln_ik(k, hi)
        for _ in range(n_iter):
            mid = 0.5 * (lo + hi)
            up = self.ln_ik(k, mid) < ln_target
            lo = np.where(up, mid, lo)
            hi = np.where(up, hi, mid)
        v = 0.5 * (lo + hi)
        v = np.where(below, -np.inf, v)
        v = np.where(above, np.inf, v)
        return v

    def map_v(self, x, m: float):
        """Input map of a cell set (called on its LOWER-type pool): V = I_0^-1(m G(x) D),
        0 V where G(x) <= 0, clamped at the ends of the nominal curve."""
        g = G_LO_US + G_SPAN_US * np.asarray(x, dtype=float)
        cur = m * g * 1e-6 * D_V
        with np.errstate(divide="ignore", invalid="ignore"):
            v = np.interp(np.log(np.where(cur > 0, cur, 1.0)), self.nom_ln, self.nom_vg)
        return np.where(g > 0, v, 0.0)

    def x_eff(self, v_b, m: float):
        """Effective threshold of a bound voltage (called on the cell set's LOWER-type pool)."""
        v_b = np.asarray(v_b, dtype=float)
        fin = np.isfinite(v_b)
        i0 = np.exp(self.ln_i0(np.where(fin, v_b, 0.0)))
        x = (i0 / (m * D_V) * 1e6 - G_LO_US) / G_SPAN_US
        x = np.where(v_b == -np.inf, -10.0, x)
        x = np.where(v_b == np.inf, 10.0, x)
        return x


class Chain:
    """Per-instance pull-down discharge curve and inverter transfer curves of one branch type."""

    def __init__(self, z, typ: str):
        p = f"chain__{typ}__"
        self.typ = typ
        self.v_sp = np.asarray(z[p + "v_sp"], dtype=float)
        self.pd_vg = np.asarray(z[p + "pd_vg"], dtype=float)
        self.pd_ln_nom = np.asarray(z[p + "pd_ln_nom"], dtype=float)
        self.pd_vg_c = np.asarray(z[p + "pd_vg_c"], dtype=float)
        self.pd_dln = np.asarray(z[p + "pd_dln"], dtype=np.float64)
        self.ln_ref = float(np.asarray(z[p + "ln_ref"]).ravel()[0])
        if typ in ("UINV", "LBUF"):
            self.rel_grid = np.asarray(z[p + "rel_grid"], dtype=float)
            self.t_first = np.asarray(z[p + "t_first"], dtype=np.float64)
        if typ == "LBUF":
            self.v_mid_in = np.asarray(z[p + "v_mid_in"], dtype=float)
            self.t_second = np.asarray(z[p + "t_second"], dtype=np.float64)

    def ln_w(self, k, gate):
        gate = np.asarray(gate, dtype=float)
        return (np.interp(gate, self.pd_vg, self.pd_ln_nom)
                + catmull_rom(self.pd_dln, self.pd_vg_c, k, gate) - self.ln_ref)

    def gate(self, k, node):
        """Gate voltage of the pull-down for divider node voltage 'node' (instance k)."""
        node = np.asarray(node, dtype=float)
        if self.typ == "L6T":
            return node
        w = interp_rows(node - self.v_sp[k], self.rel_grid, self.t_first[k])
        if self.typ == "UINV":
            return w
        return interp_rows(w - self.v_mid_in[k], self.rel_grid, self.t_second[k])

    def w0_nominal(self):
        return float(np.exp(np.interp(0.0, self.pd_vg, self.pd_ln_nom) - self.ln_ref))


class Surface:
    """Nominal divider ln I(V_G, V_DS). scheme 'lnvg_ivds' (main run, brief): ln I linear along
    V_G, then I linear along V_DS; scheme 'bilinear_ln' (sensitivity): ln I bilinear."""

    def __init__(self, z, design: str, scheme: str = "lnvg_ivds"):
        p = f"surface__{design}__"
        self.scheme = scheme
        self.vg = np.asarray(z[p + "vg"], dtype=float)
        self.vds = np.asarray(z[p + "vds"], dtype=float)
        self.ln_i = np.asarray(z[p + "ln_i"], dtype=np.float64)

    def _gidx(self, vg):
        n = self.vg.size
        i = np.clip(np.searchsorted(self.vg, vg, side="right") - 1, 0, n - 2)
        f = np.clip((vg - self.vg[i]) / (self.vg[i + 1] - self.vg[i]), 0.0, 1.0)
        return i, f

    def _didx(self, vds):
        n = self.vds.size
        j = np.clip(np.searchsorted(self.vds, vds, side="right") - 1, 0, n - 2)
        f = np.clip((vds - self.vds[j]) / (self.vds[j + 1] - self.vds[j]), 0.0, 1.0)
        return j, f

    def current(self, gi, gf, vds):
        """I at gate index/fraction (gi, gf) and drain voltage vds (arrays)."""
        j, fd = self._didx(np.asarray(vds, dtype=float))
        s = self.ln_i
        l0 = (1.0 - gf) * s[j, gi] + gf * s[j, gi + 1]
        l1 = (1.0 - gf) * s[j + 1, gi] + gf * s[j + 1, gi + 1]
        if self.scheme == "bilinear_ln":
            return np.exp((1.0 - fd) * l0 + fd * l1)
        return (1.0 - fd) * np.exp(l0) + fd * np.exp(l1)

    def inv_vg(self, cur, vds, n_iter: int = N_BISECT):
        """Gate voltage at which the surface conducts 'cur' at drain voltage vds (bisection
        on the full V_G grid, clamped at its ends)."""
        cur = np.asarray(cur, dtype=float)
        vds = np.asarray(vds, dtype=float)
        lo = np.full(cur.shape, self.vg[0])
        hi = np.full(cur.shape, self.vg[-1])
        for _ in range(n_iter):
            mid = 0.5 * (lo + hi)
            gi, gf = self._gidx(mid)
            up = self.current(gi, gf, vds) < cur
            lo = np.where(up, mid, lo)
            hi = np.where(up, hi, mid)
        return 0.5 * (lo + hi)


class Instrument:
    def __init__(self, path: Path = INSTR_DIR / "instrument.npz", surface_scheme: str = "lnvg_ivds"):
        z = np.load(path, allow_pickle=False)
        self.path = path
        self.pools = {(d, t): Pool(z, d, t) for d in ("P1", "P3") for t in ("L6T", "UINV", "LBUF")}
        self.chains = {t: Chain(z, t) for t in ("L6T", "UINV", "LBUF")}
        self.surfaces = {d: Surface(z, d, surface_scheme) for d in ("P1", "P3")}
        self.vsw_nom = float(np.asarray(z["vsw_nom"]).ravel()[0])
        self.w0 = self.chains["L6T"].w0_nominal()


# ----------------------------------------------------------------------------------
# PCM model (AIHWKit PCMLikeNoiseModel, constants from the config)
# ----------------------------------------------------------------------------------
class PCM:
    def __init__(self, cfg: dict):
        p = cfg["pcm_model"]
        self.p = p
        self.g_max = float(p["g_max"])
        self.t0 = float(p["t0"])
        self.t_read = float(p["t_read"])

    def sigma_prog(self, g_t):
        p = self.p
        x = g_t / self.g_max
        c0, c1, c2 = p["prog_coeff"]
        return (c0 + c1 * x + c2 * x ** 2) * (self.g_max / p["prog_coeff_g_max_reference"])

    def prog(self, g_t, z_prog):
        return np.maximum(g_t + self.p["prog_noise_scale"] * self.sigma_prog(g_t) * z_prog, 0.0)

    def nu(self, g_t, z_nu):
        p = self.p
        lg = np.log(np.maximum(g_t / self.g_max, 1e-7))
        mu = np.clip(p["nu_mean_a"] * lg + p["nu_mean_b"], p["nu_mean_min"], p["nu_mean_max"])
        s = np.clip(p["nu_std_a"] * lg + p["nu_std_b"], p["nu_std_min"], p["nu_std_max"])
        return np.abs(mu + s * z_nu)

    def at_time(self, g_t, g_prog, z_nu, z_read, t_inf: float):
        p = self.p
        t = t_inf + self.t0
        g_drift = g_prog * (t / self.t0) ** (-p["drift_scale"] * self.nu(g_t, z_nu))
        q = np.minimum(p["read_q0"] / np.maximum((g_prog / self.g_max) ** p["read_exp"],
                                                 p["read_floor"]), p["read_qmax"])
        sig = q * np.sqrt(np.log((t + self.t_read) / (2.0 * self.t_read)))
        return np.maximum(g_drift + np.abs(g_drift) * p["read_noise_scale"] * sig * z_read, 0.0)


def condition_state(pcm: PCM, cond: str, time_name: str, t_inf: float, g_t: np.ndarray,
                    draws: dict, rep_target: float = 12.5) -> Tuple[np.ndarray, float]:
    """Conductances (uS) of all bounds and the replica a_hat at one (condition, time)."""
    if cond == "unif":
        if time_name == "t0":
            return g_t.copy(), 1.0
        a = ((t_inf + 20.0) / 20.0) ** (-0.049)
        return a * g_t, a
    g_prog = pcm.prog(g_t, draws["z_prog"][:, 0])
    if time_name == "t0":
        return g_prog, 1.0
    ti = TIME_NAMES.index(time_name)
    g = pcm.at_time(g_t, g_prog, draws["z_nu"][:, 0], draws["z_read"][ti, :, 0], t_inf)
    rt = np.full(draws["z_rep_prog"].shape, rep_target)
    r_prog = pcm.prog(rt, draws["z_rep_prog"])
    r_t = pcm.at_time(rt, r_prog, draws["z_rep_nu"], draws["z_rep_read"][ti], t_inf)
    return g, float(np.mean(r_t) / np.mean(r_prog))


# ----------------------------------------------------------------------------------
# bounds and leakage
# ----------------------------------------------------------------------------------
CELL_SETS = {"6T2M": ("L6T", "UINV"), "GS": ("LBUF", "UINV")}
U_CUT = {"UINV": 3.0 / 15.0, "LBUF": 0.45 / 15.0}


class CellSetBounds:
    """All constraints of one forest realised with one cell set and one design."""

    def __init__(self, inst: Instrument, design: str, cell_set: str, kind: np.ndarray,
                 theta: np.ndarray, idx: np.ndarray, n_inv: int | None = None):
        lo_t, up_t = CELL_SETS[cell_set]
        self.n_inv = n_inv
        self.inst, self.design, self.cell_set = inst, design, cell_set
        self.lo_pool = inst.pools[(design, lo_t)]
        self.up_pool = inst.pools[(design, up_t)]
        self.types = (lo_t, up_t)
        self.kind = np.asarray(kind)
        self.theta = np.asarray(theta, dtype=float)
        self.idx = np.asarray(idx)
        self.is_lo = self.kind == 0
        self.sel = {lo_t: np.where(self.is_lo)[0], up_t: np.where(~self.is_lo)[0]}
        self.pool_of = {lo_t: self.lo_pool, up_t: self.up_pool}
        # threshold-verify programming: G_k = I_k(M0(theta)) / (V_SL,nom - v_sp[k])
        v_m0 = self.lo_pool.map_v(self.theta, 1.0)
        g = np.empty(self.theta.size)
        for t, ii in self.sel.items():
            pool = self.pool_of[t]
            k = self.idx[ii]
            cur = np.exp(pool.ln_ik(k, v_m0[ii]))
            g[ii] = cur / (pool.v_sl_nom - pool.v_sp[k]) * 1e6
        self.g_unclipped = g
        self.g_target = np.clip(g, 0.0, G_MAX_US)
        self.v_sp_k = np.empty(self.theta.size)
        for t, ii in self.sel.items():
            self.v_sp_k[ii] = self.pool_of[t].v_sp[self.idx[ii]]

    def node_params(self, node: str, g_now: np.ndarray, a_hat: float):
        """Per-bound effective conductance (uS), supply V_SL and map factor m."""
        v_sl = np.empty(self.theta.size)
        for t, ii in self.sel.items():
            pool = self.pool_of[t]
            v_sl[ii] = pool.v_sp[0] + (D_V / a_hat if node == "supply" else D_V)
        g_eff = g_now / a_hat if node == "ideal" else g_now
        m = a_hat if node == "dac_remap" else 1.0
        return g_eff, v_sl, m

    def x_eff(self, g_eff: np.ndarray, v_sl: np.ndarray, m: float):
        v_b = np.empty(self.theta.size)
        for t, ii in self.sel.items():
            pool = self.pool_of[t]
            k = self.idx[ii]
            cur = g_eff[ii] * 1e-6 * (v_sl[ii] - pool.v_sp[k])
            with np.errstate(divide="ignore", invalid="ignore"):
                ln_c = np.where(cur > 0, np.log(np.where(cur > 0, cur, 1.0)), -np.inf)
            v_b[ii] = pool.inv_k(k, ln_c, self.n_inv)
        return self.lo_pool.x_eff(v_b, m), v_b

    def leakage(self, cons: np.ndarray, x: np.ndarray, x_eff: np.ndarray, g_eff: np.ndarray,
                v_sl: np.ndarray, m: float, return_detail: bool = False):
        """Leakage weight w of matched bound 'cons' (constraint indices) at input x."""
        cons = np.asarray(cons)
        x = np.asarray(x, dtype=float)
        w = np.empty(cons.size)
        detail = {"v_dl": np.empty(cons.size), "node": np.empty(cons.size),
                  "gate": np.empty(cons.size)}
        surf = self.inst.surfaces[self.design]
        v_dl_all = self.lo_pool.map_v(x, m)
        for t in self.types:
            sel = np.where(self.kind[cons] == (0 if t == self.types[0] else 1))[0]
            if sel.size == 0:
                continue
            c = cons[sel]
            pool = self.pool_of[t]
            chain = self.inst.chains[t]
            k = self.idx[c]
            v_spk = pool.v_sp[k]
            v_dl = v_dl_all[sel]
            i_k = np.exp(pool.ln_ik(k, v_dl))
            vgp = surf.inv_vg(i_k, v_spk)
            gi, gf = surf._gidx(vgp)
            i_ref = surf.current(gi, gf, v_spk)
            g_s = g_eff[c] * 1e-6
            vs = v_sl[c]
            lo = np.zeros(c.size)
            hi = vs.copy()
            for _ in range(N_BISECT):
                mid = 0.5 * (lo + hi)
                f = g_s * (vs - mid) - i_k * surf.current(gi, gf, mid) / i_ref
                pos = f > 0
                lo = np.where(pos, mid, lo)
                hi = np.where(pos, hi, mid)
            node_v = 0.5 * (lo + hi)
            gate = chain.gate(k, node_v)
            if t in U_CUT:
                u = np.where(self.kind[c] == 0, x[sel] - x_eff[c], x_eff[c] - x[sel])
                gate = np.where(u >= U_CUT[t], 0.0, gate)
            w[sel] = np.exp(chain.ln_w(k, gate))
            detail["v_dl"][sel] = v_dl
            detail["node"][sel] = node_v
            detail["gate"][sel] = gate
        if return_detail:
            return w, detail
        return w
