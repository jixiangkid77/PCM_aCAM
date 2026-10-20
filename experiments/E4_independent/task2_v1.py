"""Task 2: full-row transient SPICE (own netlists) for 6 V1 cases (the first case of each
(design, node) group of v1_cases.json); crossing time of 0.9 V compared with t_sim_ns of
v1_rows.csv (threshold 0.5 %) and the ratio to t_pred_ns.

Also (not required by the brief):
- the sense time t_s(N) by SPICE (nominal pull-down at V_sw,nom, the other 2N - 1 at 0 V)
- model side for all 48 V1 cases: the bound conductances, effective thresholds, data-line
  voltages and leakage weights of v1_cases.json recomputed with the task-1 model, and S, t_pred
- gate voltages of the constrained pull-downs at 2 ns from SPICE versus the model gates

Row netlist (brief): lower bound: PCM resistor 1/G from SL_lo to X, divider nfet_01v8 X-DL-0-0
(div_lo), pull-down nfet_01v8_lvt 0.42/0.15 ML-X-0-0 (pd_lo); upper bound: PCM from SL_hi to Y,
divider Y-DL-0-0 (div_hi), INV_S (pfet_01v8 0.42/4 'up', nfet_01v8 2.0/0.15 'un') Y -> Z,
pull-down ML-Z-0-0 (pd_hi); the other 2N - n_c pull-downs nominal with gate 0 V; ML capacitor
C_SA + N C_wire; precharge 1.8 V by .ic (no UIC); crossing of 0.9 V.

Usage: python experiments/E4_independent/task2_v1.py --out <dir>
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

import e4i_core as C
import e4i_spice as S

TRAN_OPTS = "reltol=1e-4 abstol=1e-15 vntol=1e-8"
POINT_OF = {"t0": ("t0", "t0"), "dac_remap": ("1y", "dac_remap"), "supply": ("1y", "supply")}


def load_v1():
    with open(C.INSTR_DIR / "v1_cases.json") as f:
        v1 = json.load(f)
    with open(C.INSTR_DIR / "v1_rows.csv") as f:
        rows = list(csv.DictReader(f))
    return v1, rows


def first_of_groups(cases):
    seen, sel = set(), []
    for i, c in enumerate(cases):
        key = (c["design"], c["node"])
        if key not in seen:
            seen.add(key)
            sel.append(i)
    return sel


# ----------------------------------------------------------------------------------- model side
def model_side(v1, log):
    """Recompute every V1 bound with the task-1 model (breast_cancer, split 0, seed 0, arm xt)."""
    cfg = C.load_config()
    trees = C.trees_module()
    make_draws = C.make_draws_fn()
    inst = C.Instrument()
    pcm = C.PCM(cfg)
    ds, split, seed = "breast_cancer", 0, 0
    ds_idx = C.DATASETS.index(ds)
    b = trees.dataset_bundle(ds, split)
    rows, x_te = b["rows"], b["x_te"]
    theta = trees.quantize(rows.cons_thr, 4)
    k_cons = rows.cons_row.size
    draws = make_draws([C.BASE_SEED, ds_idx, split, seed], k_cons, 5, 64)
    idx = np.random.default_rng([C.BASE_SEED, ds_idx, split, seed, 4]).integers(1, 2001, k_cons)
    n_feat = x_te.shape[1]
    w0 = inst.w0
    t_s = v1["t_s_s"]
    cs_cache, out_cases, out_bounds = {}, [], []
    for ci, c in enumerate(v1["cases"]):
        des = c["design"]
        if des not in cs_cache:
            cs_cache[des] = C.CellSetBounds(inst, des, "6T2M", rows.cons_kind, theta, idx)
        cs = cs_cache[des]
        tname, node = POINT_OF[c["node"]]
        g_now, a_hat = C.condition_state(pcm, "all", tname, float(cfg["times_inference_s"][tname]),
                                         cs.g_target, draws)
        g_eff, v_sl, m = cs.node_params(node, g_now, a_hat)
        x_eff, _ = cs.x_eff(g_eff, v_sl, m)
        r = c["row"]
        s = c["sample"]
        cons = np.arange(rows.row_start[r], rows.row_start[r] + rows.row_ncons[r])
        x = x_te[s, rows.cons_feat[cons]]
        viol = np.where(rows.cons_kind[cons] == 0, x <= x_eff[cons], x > x_eff[cons])
        w, det = cs.leakage(cons, x, x_eff, g_eff, v_sl, m, return_detail=True)
        s_sum = float(w.sum() + (2 * n_feat - cons.size) * w0)
        t_pred = t_s * (1 + (2 * n_feat - 1) * w0) / s_sum
        v_sl_lo = inst.pools[(des, "L6T")].v_sp[0] + (C.D_V / a_hat if node == "supply" else C.D_V)
        v_sl_hi = inst.pools[(des, "UINV")].v_sp[0] + (C.D_V / a_hat if node == "supply" else C.D_V)
        out_cases.append({"case": ci, "design": des, "node": c["node"], "sample": s, "row": r,
                          "n_bounds_indep": int(cons.size), "n_bounds_main": len(c["bounds"]),
                          "any_violated_indep": bool(viol.any()),
                          "a_hat_indep": a_hat, "a_hat_main": c["a_hat"],
                          "S_indep": s_sum, "S_main": c["S"],
                          "t_pred_ns_indep": t_pred * 1e9,
                          "v_sl_lo_diff": v_sl_lo - c["v_sl_lo"], "v_sl_hi_diff": v_sl_hi - c["v_sl_hi"]})
        for bd in c["bounds"]:
            j = np.where((rows.cons_feat[cons] == bd["feat"]) & (rows.cons_kind[cons] == bd["kind"]))[0]
            assert j.size == 1
            j = int(j[0])
            cc = cons[j]
            u = (x[j] - x_eff[cc]) if bd["kind"] == 0 else (x_eff[cc] - x[j])
            typ = cs.types[bd["kind"]]
            cut = bool(typ in C.U_CUT and u >= C.U_CUT[typ])
            node_rel = float(det["node"][j] - cs.pool_of[typ].v_sp[idx[cc]])
            if cut:
                gclass = "u_cut_gate_0V"
            elif typ != "L6T" and abs(node_rel) > 0.06:
                gclass = "node_outside_transfer_window"   # stored inverter curve clamped at +-60 mV
            else:
                gclass = "in_window"
            out_bounds.append({"case": ci, "kind": bd["kind"], "feat": bd["feat"], "type": typ,
                               "u_lsb": float(u * 15), "gate_set_to_0V_by_u_cut": cut,
                               "node_minus_vsp_mV": node_rel * 1e3, "gate_class": gclass,
                               "idx_indep": int(idx[cc]), "idx_main": bd["idx"],
                               "x_diff": float(x[j] - bd["x"]),
                               "g_us_indep": float(g_eff[cc]), "g_us_main": bd["g_us"],
                               "theta_eff_indep": float(x_eff[cc]), "theta_eff_main": bd["theta_eff"],
                               "v_dl_indep": float(det["v_dl"][j]), "v_dl_main": bd["v_dl"],
                               "w_indep": float(w[j]), "w_main": bd["w_model"],
                               "node_v_model": float(det["node"][j]), "gate_v_model": float(det["gate"][j])})
    log(f"[task2] model side recomputed for {len(out_cases)} V1 cases")
    return out_cases, out_bounds


# ----------------------------------------------------------------------------------- SPICE side
def row_netlist(case: dict, n_feat: int, draws: dict, c_ml: float, tstop: float, tmax: float,
                options: str, tag: str):
    lines = [S.header(f"E4 independent V1 row {tag}", options),
             f"VDD vdd 0 {S.VDD}",
             f"VSLLO sl_lo 0 {case['v_sl_lo']:.17g}",
             f"VSLHI sl_hi 0 {case['v_sl_hi']:.17g}"]
    gate_nodes = []
    for i, bd in enumerate(case["bounds"]):
        k = int(bd["idx"])
        g = bd["g_us"] * 1e-6
        r_pcm = 1.0 / g if g > 0 else 1e18
        wd, ld = case["w_div_um"], case["l_div_um"]
        lines.append(f"VDL{i} dl{i} 0 {bd['v_dl']:.17g}")
        if bd["kind"] == 0:
            lines.append(f"RPCM{i} sl_lo x{i} {r_pcm:.17g}")
            lines.append(f"XDIV{i} x{i} dl{i} 0 0 {S.NFET} w={wd} l={ld} {S.mm(S.inst_z(draws, 'div_lo', k))}")
            lines.append(f"XPD{i} ml x{i} 0 0 {S.NFET_LVT} w={S.PD_W} l={S.PD_L} {S.mm(S.inst_z(draws, 'pd_lo', k))}")
            gate_nodes.append(f"x{i}")
        else:
            lines.append(f"RPCM{i} sl_hi y{i} {r_pcm:.17g}")
            lines.append(f"XDIV{i} y{i} dl{i} 0 0 {S.NFET} w={wd} l={ld} {S.mm(S.inst_z(draws, 'div_hi', k))}")
            lines.append(f"XIP{i} z{i} y{i} vdd vdd {S.PFET} w={S.INV_WP} l={S.INV_LP} {S.mm(S.inst_z(draws, 'up', k))}")
            lines.append(f"XIN{i} z{i} y{i} 0 0 {S.NFET} w={S.INV_WN} l={S.INV_LN} {S.mm(S.inst_z(draws, 'un', k))}")
            lines.append(f"XPD{i} ml z{i} 0 0 {S.NFET_LVT} w={S.PD_W} l={S.PD_L} {S.mm(S.inst_z(draws, 'pd_hi', k))}")
            gate_nodes.append(f"z{i}")
    n_wild = 2 * n_feat - len(case["bounds"])
    for j in range(n_wild):
        lines.append(f"XW{j} ml 0 0 0 {S.NFET_LVT} w={S.PD_W} l={S.PD_L}")
    lines.append(f"CML ml 0 {c_ml:.6g}")
    lines.append(".ic v(ml)=1.8")
    meas = [f"meas tran tcross when v(ml)=0.9 fall=1"]
    meas += [f"meas tran g{i}_2ns find v({nd}) at=2n" for i, nd in enumerate(gate_nodes)]
    lines += [".control", f"tran {tmax:.4g} {tstop:.6g} 0 {tmax:.4g}"] + meas + ["quit", ".endc", ".end"]
    return "\n".join(lines) + "\n", gate_nodes


def ts_netlist(n_feat: int, vsw: float, c_ml: float, tstop: float, tmax: float, options: str):
    lines = [S.header("E4 independent: sense time t_s(N)", options), f"VG g 0 {vsw:.17g}",
             f"XON ml g 0 0 {S.NFET_LVT} w={S.PD_W} l={S.PD_L}"]
    for j in range(2 * n_feat - 1):
        lines.append(f"XW{j} ml 0 0 0 {S.NFET_LVT} w={S.PD_W} l={S.PD_L}")
    lines += [f"CML ml 0 {c_ml:.6g}", ".ic v(ml)=1.8", ".control",
              f"tran {tmax:.4g} {tstop:.6g} 0 {tmax:.4g}", "meas tran tcross when v(ml)=0.9 fall=1",
              "quit", ".endc", ".end"]
    return "\n".join(lines) + "\n"


def run(out_dir: Path, log=print, sensitivity: bool = True):
    t0 = time.time()
    cfg = C.load_config()
    v1, v1rows = load_v1()
    cases = v1["cases"]
    n_feat = int(v1["n_feat"])
    c_ml = cfg["ml"]["c_sa_f"] + n_feat * cfg["ml"]["c_wire_per_cell_f"]
    work = out_dir / "spice_task2"
    S.write_spiceinit(work)
    draws = S.pool_draws()
    inst = C.Instrument()
    tmax = 0.1e-9

    # model side (all 48 cases)
    mc, mb = model_side(v1, log)
    with open(out_dir / "task2_model_bounds.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(mb[0].keys()))
        wr.writeheader()
        wr.writerows(mb)
    with open(out_dir / "task2_model_cases.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(mc[0].keys()))
        wr.writeheader()
        wr.writerows(mc)

    # sense time t_s(N)
    cir = work / "ts.cir"
    cir.write_text(ts_netlist(n_feat, inst.vsw_nom, c_ml, 1.2e-6, tmax, TRAN_OPTS))
    t_s_sim = S.meas_values(S.run_ngspice(cir))["tcross"]
    log(f"[task2] t_s({n_feat}) = {t_s_sim * 1e9:.3f} ns (main {v1['t_s_s'] * 1e9:.3f} ns)")

    sel = first_of_groups(cases)
    out = []
    gate_rows = []
    for ci in sel:
        case = cases[ci]
        row = v1rows[ci]
        assert int(row["case"]) == ci
        t_pred = float(row["t_pred_ns"]) * 1e-9
        tstop = 2.0 * t_pred + 100e-9
        net, gates = row_netlist(case, n_feat, draws, c_ml, tstop, tmax, TRAN_OPTS, f"case{ci}")
        cir = work / f"v1_case{ci:02d}.cir"
        cir.write_text(net)
        mv = S.meas_values(S.run_ngspice(cir))
        t_sim = mv["tcross"]
        t_sim_main = float(row["t_sim_ns"]) * 1e-9
        rec = {"case": ci, "design": case["design"], "node": case["node"], "n_bounds": len(case["bounds"]),
               "S_main": case["S"], "t_pred_ns": float(row["t_pred_ns"]), "t_sim_main_ns": t_sim_main * 1e9,
               "t_sim_indep_ns": t_sim * 1e9, "rel_diff_vs_main": t_sim / t_sim_main - 1.0,
               "ratio_indep_to_pred": t_sim / t_pred, "ratio_main_to_pred": t_sim_main / t_pred,
               "pass": abs(t_sim / t_sim_main - 1.0) <= 0.005}
        # gate voltages at 2 ns versus the model gates (diagnosis; bounds whose gate the model
        # sets to 0 V by the u-cut are reported separately)
        mbc = [b for b in mb if b["case"] == ci]
        dg = np.array([mv[f"g{i}_2ns"] - mbc[i]["gate_v_model"] for i in range(len(gates))])
        cutm = np.array([mbc[i]["gate_set_to_0V_by_u_cut"] for i in range(len(gates))])
        for i in range(len(gates)):
            gate_rows.append({"case": ci, "bound": i, "kind": mbc[i]["kind"], "type": mbc[i]["type"],
                              "u_lsb": mbc[i]["u_lsb"], "gate_set_to_0V_by_u_cut": bool(cutm[i]),
                              "node_minus_vsp_mV": mbc[i]["node_minus_vsp_mV"], "gate_class": mbc[i]["gate_class"],
                              "w_model": mbc[i]["w_main"], "gate_model_V": mbc[i]["gate_v_model"],
                              "gate_spice_2ns_V": mv[f"g{i}_2ns"], "diff_mV": dg[i] * 1e3})
        rec["max_abs_gate_diff_2ns_mV_uncut"] = float(np.max(np.abs(dg[~cutm])) * 1e3) if (~cutm).any() else np.nan
        rec["max_abs_gate_diff_2ns_mV_cut"] = float(np.max(np.abs(dg[cutm])) * 1e3) if cutm.any() else np.nan
        out.append(rec)
        log(f"[task2] case {ci} ({case['design']}, {case['node']}): t_sim {t_sim * 1e9:.3f} ns, "
            f"main {t_sim_main * 1e9:.3f} ns, diff {100 * rec['rel_diff_vs_main']:+.3f} %, "
            f"ratio to t_pred {rec['ratio_indep_to_pred']:.4f} ({time.time() - t0:.0f} s)")

    sens = []
    if sensitivity:
        # numerical sensitivity of one case: default ngspice tolerances and a 4x larger max step
        ci = sel[0]
        case = cases[ci]
        row = v1rows[ci]
        t_pred = float(row["t_pred_ns"]) * 1e-9
        for label, opts, tm in (("default_tolerances", "", tmax), ("tmax_0.4ns", TRAN_OPTS, 0.4e-9)):
            net, _ = row_netlist(case, n_feat, draws, c_ml, 2.0 * t_pred + 100e-9, tm, opts, f"case{ci}_{label}")
            cir = work / f"v1_case{ci:02d}_{label}.cir"
            cir.write_text(net)
            t_alt = S.meas_values(S.run_ngspice(cir))["tcross"]
            ref = [o for o in out if o["case"] == ci][0]["t_sim_indep_ns"] * 1e-9
            sens.append({"case": ci, "variant": label, "t_sim_ns": t_alt * 1e9, "rel_change": t_alt / ref - 1.0})

    with open(out_dir / "task2_v1_spice.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        wr.writeheader()
        wr.writerows(out)
    with open(out_dir / "task2_gates_2ns.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(gate_rows[0].keys()))
        wr.writeheader()
        wr.writerows(gate_rows)

    def maxrel(a, b):
        a, b = np.asarray(a, float), np.asarray(b, float)
        return float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-300)))

    model = {
        "n_cases": len(mc), "n_bounds": len(mb),
        "idx_all_equal": all(b["idx_indep"] == b["idx_main"] for b in mb),
        "n_bounds_equal": all(c["n_bounds_indep"] == c["n_bounds_main"] for c in mc),
        "no_violated_bound": all(not c["any_violated_indep"] for c in mc),
        "max_abs_x_diff": float(max(abs(b["x_diff"]) for b in mb)),
        "max_rel_g_us": maxrel([b["g_us_indep"] for b in mb], [b["g_us_main"] for b in mb]),
        "max_abs_theta_eff": float(max(abs(b["theta_eff_indep"] - b["theta_eff_main"]) for b in mb)),
        "max_abs_v_dl_V": float(max(abs(b["v_dl_indep"] - b["v_dl_main"]) for b in mb)),
        "max_rel_w": maxrel([b["w_indep"] for b in mb], [b["w_main"] for b in mb]),
        "max_rel_S": maxrel([c["S_indep"] for c in mc], [c["S_main"] for c in mc]),
        "max_abs_a_hat": float(max(abs(c["a_hat_indep"] - c["a_hat_main"]) for c in mc)),
        "max_abs_v_sl_diff_V": float(max(max(abs(c["v_sl_lo_diff"]), abs(c["v_sl_hi_diff"])) for c in mc)),
    }
    summary = {
        "selected_cases": sel,
        "threshold_rel": 0.005,
        "c_ml_F": c_ml,
        "w0": v1["w0"],
        "w0_indep": inst.w0,
        "tran": {"options": TRAN_OPTS, "tmax_s": tmax},
        "t_s_indep_ns": t_s_sim * 1e9, "t_s_main_ns": v1["t_s_s"] * 1e9,
        "t_s_rel_diff": t_s_sim / v1["t_s_s"] - 1.0,
        "max_abs_rel_diff_vs_main": float(max(abs(o["rel_diff_vs_main"]) for o in out)),
        "ratio_indep_to_pred_range": [float(min(o["ratio_indep_to_pred"] for o in out)),
                                      float(max(o["ratio_indep_to_pred"] for o in out))],
        "max_abs_ln_ratio_indep_to_pred": float(max(abs(np.log(o["ratio_indep_to_pred"])) for o in out)),
        "gate_2ns_by_class": {
            cl: {"n_bounds": sum(1 for g in gate_rows if g["gate_class"] == cl),
                 "max_abs_diff_mV": float(max([abs(g["diff_mV"]) for g in gate_rows if g["gate_class"] == cl] or [np.nan])),
                 "max_w_model": float(max([g["w_model"] for g in gate_rows if g["gate_class"] == cl] or [np.nan]))}
            for cl in ("in_window", "node_outside_transfer_window", "u_cut_gate_0V")},
        "cases": out,
        "numerical_sensitivity": sens,
        "model_side_all_48_cases": model,
        "pass": all(o["pass"] for o in out),
        "elapsed_s": round(time.time() - t0, 1),
    }
    with open(out_dir / "task2_summary.json", "w") as f:
        json.dump(summary, f, indent=1)
    log(f"[task2] max |rel diff| vs main {100 * summary['max_abs_rel_diff_vs_main']:.3f} %; PASS={summary['pass']}")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-sensitivity", action="store_true")
    a = ap.parse_args()
    run(Path(a.out), sensitivity=not a.no_sensitivity)
