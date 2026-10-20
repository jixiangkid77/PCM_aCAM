"""Task 3: switching points of instances 1-20 (and the nominal instance 0) of each branch type
by SPICE, compared with instrument.npz (threshold 50 uV).

- L6T : gate voltage of the lower pull-down (pd_lo draw) at 280 nA, V_DS = 1.8 V
- UINV: input of the upper inverter (un, up) at which its output equals the upper
        pull-down's V_sw (pd_hi draw)
- LBUF: W* = input of inverter 2 (b2n, b2p) at which its output equals the lower
        pull-down's V_sw (pd_lo); X* = input of inverter 1 (b1n, b1p) at which its output = W*

Extras (not required by the brief; they test the stored curves that task 1 uses):
- pull-down discharge curves at V_DS = 1.35 V versus the stored chains (C6 range)
- inverter transfer curves versus t_first / t_second
- divider curves at V_DS = v_sp[k] versus the pool reconstruction (bound-voltage error)

Usage: python experiments/E4_independent/task3_switching.py --out <dir>
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

I_SW = 100e-9 * S.PD_W / S.PD_L          # 280 nA
OPTS = "reltol=1e-7 abstol=1e-16 vntol=1e-10"


def cross(x, y, target, increasing: bool):
    """x at which the sampled curve y(x) crosses 'target': linear interpolation (result) and a
    cubic inverse interpolation through 4 points (error estimate)."""
    yy = y if increasing else -y
    tt = target if increasing else -target
    i = int(np.argmax(yy >= tt))
    if i == 0 or yy[i] < tt:
        return np.nan, np.nan
    x_lin = x[i - 1] + (tt - yy[i - 1]) * (x[i] - x[i - 1]) / (yy[i] - yy[i - 1])
    lo, hi = max(i - 2, 0), min(i + 2, x.size)
    cf = np.polyfit(yy[lo:hi] - tt, x[lo:hi], min(3, hi - lo - 1))
    return float(x_lin), float(np.polyval(cf, 0.0))


def run(out_dir: Path, n_inst: int = 20, log=print, extras: bool = True):
    t0 = time.time()
    work = out_dir / "spice_task3"
    S.write_spiceinit(work)
    draws = S.pool_draws()
    inst = C.Instrument()
    ks = list(range(0, n_inst + 1))

    # ---------------- A: pull-downs at V_DS = 1.8 V (and 1.35 V for the extra check)
    pd_devs = [("lo", k, S.inst_z(draws, "pd_lo", k)) for k in ks] + \
              [("hi", k, S.inst_z(draws, "pd_hi", k)) for k in ks if k > 0]
    res_pd = {}
    for vds, tag, (v0, v1, dv) in ((1.8, "pd18", (0.30, 0.80, 1e-4)), (1.35, "pd135", (0.0, 0.8, 5e-4))):
        lines = [S.header(f"E4 independent: pull-downs, V_DS = {vds} V", OPTS), "VG g 0 0.5"]
        names = []
        for side, k, z in pd_devs:
            nm = f"{side}{k}"
            lines.append(f"VD{nm} d{nm} 0 {vds}")
            lines.append(f"X{nm} d{nm} g 0 0 {S.NFET_LVT} w={S.PD_W} l={S.PD_L} {S.mm(z)}")
            names.append(nm)
        vec = " ".join(f"i(VD{nm})" for nm in names)
        lines += [".control", "set filetype=binary", f"dc VG {v0} {v1} {dv}",
                  f"write {tag}.raw {vec}", "quit", ".endc", ".end"]
        cir = work / f"{tag}.cir"
        cir.write_text("\n".join(lines) + "\n")
        S.run_ngspice(cir)
        raw = S.read_raw(work / f"{tag}.raw")
        vg = raw["v(v-sweep)"]
        res_pd[tag] = (vg, {nm: np.abs(raw[f"i(vd{nm})"]) for nm in names})
    log(f"[task3] pull-down sweeps done ({time.time() - t0:.1f} s)")
    vg, cur = res_pd["pd18"]
    vsw = {}
    vsw_err = 0.0
    for nm, i_d in cur.items():
        a, b = cross(vg, np.log(i_d), np.log(I_SW), True)
        vsw[nm] = a
        vsw_err = max(vsw_err, abs(a - b))

    # ---------------- B: inverters (unloaded DC transfer)
    inv_sets = {"u": ("un", "up"), "b1": ("b1n", "b1p"), "b2": ("b2n", "b2p")}
    lines = [S.header("E4 independent: INV_S transfer curves", OPTS), f"VDD vdd 0 {S.VDD}", "VIN in 0 0.57"]
    onames = []
    for pre, (gn, gp) in inv_sets.items():
        for k in ks:
            if k == 0 and pre != "u":
                continue                      # one nominal inverter is enough
            nm = f"{pre}{k}"
            lines.append(f"XP{nm} o{nm} in vdd vdd {S.PFET} w={S.INV_WP} l={S.INV_LP} {S.mm(S.inst_z(draws, gp, k))}")
            lines.append(f"XN{nm} o{nm} in 0 0 {S.NFET} w={S.INV_WN} l={S.INV_LN} {S.mm(S.inst_z(draws, gn, k))}")
            onames.append(nm)
    vec = " ".join(f"v(o{nm})" for nm in onames)
    lines += [".control", "set filetype=binary", "dc VIN 0.40 0.75 2e-5",
              f"write inv.raw {vec}", "quit", ".endc", ".end"]
    cir = work / "inv.cir"
    cir.write_text("\n".join(lines) + "\n")
    S.run_ngspice(cir)
    raw = S.read_raw(work / "inv.raw")
    vin = raw["v(v-sweep)"]
    vout = {nm: raw[f"v(o{nm})"] for nm in onames}
    vout["b10"] = vout["u0"]
    vout["b20"] = vout["u0"]
    log(f"[task3] inverter sweep done ({time.time() - t0:.1f} s)")

    rows = []
    inv_err = 0.0
    for k in ks:
        vsw_lo = vsw[f"lo{k}"]
        vsw_hi = vsw["lo0"] if k == 0 else vsw[f"hi{k}"]
        y_star, y2 = cross(vin, vout[f"u{k}"], vsw_hi, False)
        w_star, w2 = cross(vin, vout[f"b2{k}"], vsw_lo, False)
        x_star, x2 = cross(vin, vout[f"b1{k}"], w_star, False)
        inv_err = max(inv_err, abs(y_star - y2), abs(w_star - w2), abs(x_star - x2))
        main = {
            "L6T": float(inst.pools[("P1", "L6T")].v_sp[k]),
            "UINV": float(inst.pools[("P1", "UINV")].v_sp[k]),
            "LBUF": float(inst.pools[("P1", "LBUF")].v_sp[k]),
            "LBUF_Wstar": float(inst.chains["LBUF"].v_mid_in[k]),
        }
        mine = {"L6T": vsw_lo, "UINV": y_star, "LBUF": x_star, "LBUF_Wstar": w_star}
        for typ in ("L6T", "UINV", "LBUF", "LBUF_Wstar"):
            rows.append({"instance": k, "quantity": typ, "v_indep_V": mine[typ], "v_main_V": main[typ],
                         "diff_uV": (mine[typ] - main[typ]) * 1e6})
        rows.append({"instance": k, "quantity": "Vsw_upper_pulldown", "v_indep_V": vsw_hi,
                     "v_main_V": np.nan, "diff_uV": np.nan})
    with open(out_dir / "task3_switching_points.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    per_type = {}
    for typ in ("L6T", "UINV", "LBUF", "LBUF_Wstar"):
        d = np.array([r["diff_uV"] for r in rows if r["quantity"] == typ and r["instance"] >= 1])
        d0 = [r["diff_uV"] for r in rows if r["quantity"] == typ and r["instance"] == 0][0]
        per_type[typ] = {"max_abs_diff_uV_inst1_20": float(np.max(np.abs(d))),
                         "mean_diff_uV_inst1_20": float(np.mean(d)),
                         "nominal_diff_uV": float(d0),
                         "pass": bool(np.max(np.abs(d)) <= 50.0)}
    # diagnosis: the inverter crossings again, from the same SPICE data reduced to 0.5 mV knots
    # (config 'inverter_step') with linear interpolation; targets = the fine pull-down V_sw
    kn_i = np.where(np.abs(np.round(vin / 5e-4) * 5e-4 - vin) < 1e-9)[0]
    coarse = {"UINV": [], "LBUF": [], "LBUF_Wstar": []}
    for k in ks:
        hi_t = vsw["lo0"] if k == 0 else vsw[f"hi{k}"]
        y_c = cross(vin[kn_i], vout[f"u{k}"][kn_i], hi_t, False)[0]
        w_c = cross(vin[kn_i], vout[f"b2{k}"][kn_i], vsw[f"lo{k}"], False)[0]
        x_c = cross(vin[kn_i], vout[f"b1{k}"][kn_i], w_c, False)[0]
        coarse["UINV"].append((y_c - inst.pools[("P1", "UINV")].v_sp[k]) * 1e6)
        coarse["LBUF"].append((x_c - inst.pools[("P1", "LBUF")].v_sp[k]) * 1e6)
        coarse["LBUF_Wstar"].append((w_c - inst.chains["LBUF"].v_mid_in[k]) * 1e6)
    coarse_max = {t: float(np.max(np.abs(v))) for t, v in coarse.items()}
    p1p3_equal = all(np.array_equal(inst.pools[("P1", t)].v_sp, inst.pools[("P3", t)].v_sp)
                     for t in ("L6T", "UINV", "LBUF"))
    summary = {
        "threshold_uV": 50.0,
        "per_type": per_type,
        "vsw_nominal_indep_V": vsw["lo0"], "vsw_nominal_main_V": inst.vsw_nom,
        "interp_error_estimate_uV": {"pull_down": vsw_err * 1e6, "inverter": inv_err * 1e6},
        "inverter_0p5mV_grid_emulation_max_abs_diff_uV": coarse_max,
        "inverter_grid_note": ("inverter crossings from the same SPICE data reduced to 0.5 mV knots (config "
                               "'inverter_step') with linear interpolation, targets = fine pull-down V_sw; "
                               "agreement here shows that the sub-uV inverter differences are the main run's "
                               "0.5 mV grid interpolation error"),
        "pool_v_sp_identical_P1_P3_and_chain": bool(p1p3_equal and all(
            np.array_equal(inst.pools[("P1", t)].v_sp, inst.chains[t].v_sp) for t in ("L6T", "UINV", "LBUF"))),
        "spice_options": OPTS,
        "pass": bool(all(v["pass"] for v in per_type.values())),
    }

    # ---------------- extras
    if extras:
        summary["extras"] = extra_checks(out_dir, work, inst, draws, ks, res_pd, vin, vout, log, t0)
    summary["elapsed_s"] = round(time.time() - t0, 1)
    with open(out_dir / "task3_summary.json", "w") as f:
        json.dump(summary, f, indent=1)
    log(f"[task3] max |diff| (uV): " + ", ".join(f"{t} {v['max_abs_diff_uV_inst1_20']:.3f}"
                                               for t, v in per_type.items()) + f"; PASS={summary['pass']}")
    return summary


def extra_checks(out_dir, work, inst, draws, ks, res_pd, vin, vout, log, t0):
    ex = {}
    # (1) pull-down discharge curves at 1.35 V: ln w_k(V) = ln I_k(V) - ln_ref versus the chain
    vg, cur = res_pd["pd135"]
    worst = {}
    sel = (vg >= inst.vsw_nom - 0.25) & (vg <= inst.vsw_nom + 0.05)
    for side, typ in (("lo", "L6T"), ("hi", "UINV")):
        ch = inst.chains[typ]
        m = 0.0
        for k in ks:
            nm = "lo0" if k == 0 else f"{side}{k}"
            ln_sp = np.log(cur[nm][sel]) - ch.ln_ref
            ln_ch = ch.ln_w(np.full(sel.sum(), k), vg[sel])
            m = max(m, float(np.max(np.abs(ln_ch - ln_sp))))
        worst[typ] = m
    ex["pulldown_ln_w_max_abs_diff_Vsw-0.25..+0.05"] = worst
    ex["ln_ref_indep_minus_main"] = float(np.interp(inst.vsw_nom, vg, np.log(cur["lo0"]))
                                          - inst.chains["L6T"].ln_ref)
    # (2) inverter transfer curves versus t_first / t_second
    tr = {}
    for typ, pre, key, centre in (("UINV", "u", "t_first", "v_sp"), ("LBUF", "b1", "t_first", "v_sp"),
                                  ("LBUF", "b2", "t_second", "v_mid_in")):
        ch = inst.chains[typ]
        m = 0.0
        for k in ks:
            c0 = getattr(ch, centre)[k]
            xs = c0 + ch.rel_grid
            mine = np.interp(xs, vin, vout[f"{pre}{k}"])
            m = max(m, float(np.max(np.abs(mine - getattr(ch, key)[k]))))
        tr[f"{typ}_{key}"] = m * 1e3
    ex["inverter_transfer_max_abs_diff_mV"] = tr
    # the same with the curves reduced to 0.5 mV knots (linear interpolation)
    kn = np.where(np.abs(np.round(vin / 5e-4) * 5e-4 - vin) < 1e-9)[0]
    trc = {}
    for typ, pre, key, centre in (("UINV", "u", "t_first", "v_sp"), ("LBUF", "b1", "t_first", "v_sp"),
                                  ("LBUF", "b2", "t_second", "v_mid_in")):
        ch = inst.chains[typ]
        m = 0.0
        for k in ks:
            xs = getattr(ch, centre)[k] + ch.rel_grid
            mine = np.interp(xs, vin[kn], vout[f"{pre}{k}"][kn])
            m = max(m, float(np.max(np.abs(mine - getattr(ch, key)[k]))))
        trc[f"{typ}_{key}"] = m * 1e3
    ex["inverter_transfer_max_abs_diff_mV_from_0.5mV_knots"] = trc
    # (3) divider curves at V_DS = v_sp[k]: bound voltage for currents 0.2-5 uA
    lines = [S.header("E4 independent: divider curves at V_DS = v_sp[k]", OPTS), "VG g 0 0.5"]
    names = []
    for des in ("P1", "P3"):
        wd, ld = (0.42, 0.15) if des == "P1" else (0.42, 4.0)
        for typ, grp in (("L6T", "div_lo"), ("UINV", "div_hi"), ("LBUF", "div_lo")):
            pool = inst.pools[(des, typ)]
            for k in ks:
                nm = f"{des}{typ}{k}"
                lines.append(f"VD{nm} d{nm} 0 {pool.v_sp[k]:.17g}")
                lines.append(f"X{nm} d{nm} g 0 0 {S.NFET} w={wd} l={ld} {S.mm(S.inst_z(draws, grp, k))}")
                names.append((des, typ, k, nm))
    vec = " ".join(f"i(VD{n[3]})" for n in names)
    lines += [".control", "set filetype=binary", "dc VG 0.2 1.8 5e-4",
              f"write div.raw {vec}", "quit", ".endc", ".end"]
    cir = work / "div.cir"
    cir.write_text("\n".join(lines) + "\n")
    S.run_ngspice(cir)
    raw = S.read_raw(work / "div.raw")
    vgd = raw["v(v-sweep)"]
    currents = np.geomspace(0.2e-6, 5e-6, 25)
    dv = {}
    for des, typ, k, nm in names:
        pool = inst.pools[(des, typ)]
        lnc = np.log(np.abs(raw[f"i(vd{nm.lower()})"]))
        win = lnc > np.log(1e-8)                                     # monotone part of the curve
        assert np.all(np.diff(lnc[win]) > 0)
        v_sp = np.interp(np.log(currents), lnc[win], vgd[win])       # SPICE (0.5 mV grid, linear in ln I)
        v_po = pool.inv_k(np.full(currents.size, k), np.log(currents), 64)
        key = f"{des}/{typ}"
        dv[key] = max(dv.get(key, 0.0), float(np.max(np.abs(v_po - v_sp))) * 1e6)
    ex["divider_bound_voltage_max_abs_diff_uV_0.2-5uA"] = dv
    ex["note"] = ("extras use the instrument's own v_sp[k] as drain voltage; SPICE curves on a 0.5 mV "
                  "grid with linear interpolation in ln I")
    log(f"[task3] extra checks done ({time.time() - t0:.1f} s)")
    return ex


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-extras", action="store_true")
    a = ap.parse_args()
    run(Path(a.out), extras=not a.no_extras)
