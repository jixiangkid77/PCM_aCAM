"""E2 independent check (protocol section 9): Task 1 predictor + Task 2 full-branch spot check.

Reads only: docs/E2_protocol.md (by the author, not by this program), experiments/E2_config.json,
the sky130 subset models, results/E2_predict_20261014_0315/predictions.json and
results/E2_20261014_0316/arrays_<cid>.npz (keys theta, b_target, vsw_mc, err_pdk_dac_remap_1y,
err_pdk_supply_1y).  Writes results/E2_independent_check_<JST stamp>/ and
logs/E2_independent_work/run_<JST stamp>/.

usage: python run_check.py [--conditions G2,C2,T3,G6] [--n-inst 40]
"""
import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import task1_predictor as T1  # noqa: E402
import task2_branch_mc as T2  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))).replace(os.sep, "/")  # repository root (was an absolute path in the original run)
CONFIG = f"{ROOT}/experiments/E2_config.json"
PRED = f"{ROOT}/results/E2_predict_20261014_0315/predictions.json"
MCDIR = f"{ROOT}/results/E2_20261014_0316"
KEYS_T1 = ["pdk|dac_remap", "pdk|supply", "div_only|dac_remap", "pd_only|dac_remap",
           "pd_only|supply"]
TOL = dict(rms_rel=0.01, rms_abs=0.002, vsw_nom_V=0.05e-3, sigma_rel=0.01,
           vsw_k_V=0.05e-3, m0_V=0.05e-3, err_lsb=0.01)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def jst_now():
    return dt.datetime.now(dt.timezone(dt.timedelta(hours=9)))


def rel(a, b):
    return (a - b) / b if b != 0 else (0.0 if a == b else float("inf"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conditions", default="G2,C2,T3,G6")
    ap.add_argument("--n-inst", type=int, default=40)
    args = ap.parse_args()
    cids = args.conditions.split(",")

    start = jst_now()
    stamp = start.strftime("%Y%m%d_%H%M")
    outdir = f"{ROOT}/results/E2_independent_check_{stamp}"
    while os.path.exists(outdir):          # never overwrite: wait for the next minute
        time.sleep(5)
        stamp = jst_now().strftime("%Y%m%d_%H%M")
        outdir = f"{ROOT}/results/E2_independent_check_{stamp}"
    os.makedirs(outdir)
    workroot = f"{ROOT}/logs/E2_independent_work/run_{stamp}"
    os.makedirs(workroot, exist_ok=True)

    cfg = json.load(open(CONFIG))
    conds = {c["id"]: c for c in cfg["conditions"]}
    pulldowns = cfg["circuit"]["pulldowns"]
    t1y = cfg["drift"]["times_s"]["1y"]
    pred = json.load(open(PRED))

    t1_rows, vsw_rows, m0_rows, err_rows = [], [], [], []
    per_cond = {}
    for cid in cids:
        cond = conds[cid]
        wd = os.path.join(workroot, cid)
        t0 = time.time()
        # ---------------- Task 1
        p1 = T1.predict_condition(cond, pulldowns, t1y, wd)
        m = pred[cid]
        ps = m["pulldown_sensitivity"]
        rows = []
        def add(q, mine, main_, kind):
            d = mine - main_
            r = rel(mine, main_)
            if kind == "rms":
                ok = abs(r) <= TOL["rms_rel"] or abs(d) <= TOL["rms_abs"]
            elif kind == "vsw_nom":
                ok = abs(d) <= TOL["vsw_nom_V"]
            elif kind == "sigma":
                ok = abs(r) <= TOL["sigma_rel"]
            else:
                ok = None
            rows.append(dict(condition=cid, quantity=q, kind=kind, mine=float(mine),
                             main=float(main_), abs_diff=float(d), rel_diff=float(r),
                             passed=None if ok is None else bool(ok)))
        add("vsw_nom_V", p1["vsw_nom_V"], ps["vsw_nom"], "vsw_nom")
        add("sigma_delta_V", p1["pulldown"]["sigma_delta"], ps["sigma_delta"], "sigma")
        for prm in ("vth0", "toxe", "voff", "nfactor"):
            add(f"dvsw_d{prm}_V", p1["pulldown"][f"dvsw_d{prm}"], ps[f"dvsw_d{prm}"], "info")
        add("v_sl_nom_V", p1["v_sl_nom_V"], m["v_sl_nom_V"], "info")
        for k in KEYS_T1:
            add(f"rms_1y|{k}", p1["rms_lsb"][k], m["predictions"]["1y"]["rms_lsb"][k], "rms")
        add("rms_1y|div_only|supply", p1["rms_lsb"]["div_only|supply"],
            m["predictions"]["1y"]["rms_lsb"]["div_only|supply"], "info")
        t1_rows += rows
        # per-theta diagnostics against the main run's per-theta sensitivities
        sdiag = {}
        for which in ("s_at_I", "s_at_aI"):
            for prm in ("vth0", "toxe", "voff", "nfactor", "lambda"):
                mine = np.asarray(p1[which][prm]); main_ = np.asarray(m["predictions"]["1y"][which][prm])
                sdiag[f"{which}|{prm}|max_abs_diff"] = float(np.max(np.abs(mine - main_)))
        msdiag = {}
        for k in KEYS_T1:
            mine = np.asarray(p1["per_theta_ms_x"][k]); main_ = np.asarray(m["predictions"]["1y"]["per_theta_ms_x"][k])
            msdiag[k] = float(np.max(np.abs(mine - main_) / np.where(main_ != 0, np.abs(main_), 1)))
        # ---------------- Task 2
        p2 = T2.run_condition(cond, pulldowns, t1y, wd, n_inst=args.n_inst)
        arr = np.load(f"{MCDIR}/arrays_{cid}.npz")
        theta_main = arr["theta"]; b_target = arr["b_target"]; vsw_mc = arr["vsw_mc"]
        e_main = {"dac_remap": arr["err_pdk_dac_remap_1y"], "supply": arr["err_pdk_supply_1y"]}
        assert np.allclose(theta_main, np.arange(16) / 15.0, atol=1e-12)
        n = args.n_inst
        for k in range(n):
            d = p2["vsw"][k] - vsw_mc[k]
            vsw_rows.append(dict(condition=cid, instance=k, vsw_mine_V=p2["vsw"][k],
                                 vsw_main_V=vsw_mc[k], diff_mV=d * 1e3,
                                 passed=bool(abs(d) <= TOL["vsw_k_V"])))
        used = set(int(j) for j in p2["theta_idx"])
        for j in range(16):
            d = p2["M0"][j] - b_target[j]
            m0_rows.append(dict(condition=cid, j=j, theta=j / 15.0, used_in_task2=j in used,
                                m0_mine_V=p2["M0"][j], b_target_main_V=b_target[j],
                                diff_mV=d * 1e3, passed=bool(abs(d) <= TOL["m0_V"])))
        for ni, node in enumerate(p2["nodes"]):
            for k in range(n):
                for jj, j in enumerate(p2["theta_idx"]):
                    mine = p2["err"][ni, k, jj]; main_ = float(e_main[node][k, j])
                    err_rows.append(dict(condition=cid, node=node, instance=k, j=int(j),
                                         theta=j / 15.0, vsw_k_V=p2["vsw"][k],
                                         G_k_uS=p2["Gk"][k, jj] * 1e6,
                                         bound_V=p2["bound"][ni, k, jj],
                                         x_eff=p2["x_eff"][ni, k, jj],
                                         err_mine_lsb=mine, err_main_lsb=main_,
                                         diff_lsb=mine - main_,
                                         passed=bool(abs(mine - main_) <= TOL["err_lsb"])))
        np.savez(os.path.join(outdir, f"independent_{cid}.npz"), vsw=p2["vsw"], M0=p2["M0"],
                 theta_idx=p2["theta_idx"], Gk=p2["Gk"], bound=p2["bound"], x_eff=p2["x_eff"],
                 err=p2["err"], nodes=np.array(p2["nodes"]),
                 s_at_I=np.array([p1["s_at_I"][q] for q in ("vth0", "toxe", "voff", "nfactor", "lambda")]),
                 s_at_aI=np.array([p1["s_at_aI"][q] for q in ("vth0", "toxe", "voff", "nfactor", "lambda")]))
        rms_sub = {}
        for ni, node in enumerate(p2["nodes"]):
            sel = e_main[node][:n][:, p2["theta_idx"]]
            rms_sub[node] = dict(mine=float(np.sqrt(np.mean(p2["err"][ni] ** 2))),
                                 main=float(np.sqrt(np.mean(sel.astype(float) ** 2))))
        per_cond[cid] = dict(
            seconds=time.time() - t0,
            task1=dict(a=p1["a"], e_ln2_r=p1["e_ln2_r"], e_ln2_r_quad=p1["e_ln2_r_quad"],
                       n_dropped=p1["n_dropped"], pulldown=p1["pulldown"],
                       divider_diag=dict(nominal_resid_lnI_max=p1["diag"]["nominal_resid_lnI_max"],
                                         refine_shift_max_V=p1["diag"]["refine_shift_max_V"]),
                       sens_vs_main=sdiag, per_theta_ms_max_rel_diff=msdiag,
                       rms_lsb=p1["rms_lsb"]),
            task2=dict(vsw_nom=p2["vsw_nom"], vsl_nom=p2["vsl_nom"], vsl_supply_1y=p2["vsl_supply"],
                       selfcheck_zero_mismatch_max_abs_err_lsb=float(np.max(np.abs(p2["err_selfcheck"]))),
                       closure_max_abs_V=p2["closure_max_abs_V"], diag=p2["diag"],
                       rms_over_checked_pairs=rms_sub))
        print(f"{cid}: done in {time.time() - t0:.1f} s", flush=True)

    # ---------------- write tables
    def write_csv(name, rows):
        with open(os.path.join(outdir, name), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows:
                w.writerow({k: (repr(float(v)) if isinstance(v, (float, np.floating)) else
                                (bool(v) if isinstance(v, (bool, np.bool_)) else v))
                            for k, v in r.items()})
    write_csv("task1_predictor_compare.csv", t1_rows)
    write_csv("task2_vsw_compare.csv", vsw_rows)
    write_csv("task2_m0_compare.csv", m0_rows)
    write_csv("task2_error_compare.csv", err_rows)

    # ---------------- summary
    def worst(rows, key, filt=lambda r: True):
        rr = [r for r in rows if filt(r)]
        r = max(rr, key=lambda r: abs(r[key]))
        return r
    t1_judged = [r for r in t1_rows if r["passed"] is not None]
    rms_rows = [r for r in t1_rows if r["kind"] == "rms"]
    w_rms_rel = worst(rms_rows, "rel_diff")
    w_rms_abs = worst(rms_rows, "abs_diff")
    w_vswn = worst([r for r in t1_rows if r["kind"] == "vsw_nom"], "abs_diff")
    w_sig = worst([r for r in t1_rows if r["kind"] == "sigma"], "rel_diff")
    w_vsw = worst(vsw_rows, "diff_mV")
    w_m0 = worst(m0_rows, "diff_mV")
    w_m0u = worst(m0_rows, "diff_mV", lambda r: r["used_in_task2"])
    w_err = {node: worst(err_rows, "diff_lsb", lambda r, nd=node: r["node"] == nd)
             for node in ("dac_remap", "supply")}
    summary = dict(
        check="E2 independent check (protocol section 9)",
        started_jst=start.isoformat(), finished_jst=jst_now().isoformat(),
        conditions=cids, n_instances=args.n_inst, theta_checked=[1 / 15, 5 / 15, 10 / 15, 14 / 15],
        tolerances=TOL,
        task1=dict(
            passed=all(r["passed"] for r in t1_judged),
            n_judged=len(t1_judged), n_failed=sum(not r["passed"] for r in t1_judged),
            max_rel_diff_rms=dict(condition=w_rms_rel["condition"], quantity=w_rms_rel["quantity"],
                                  rel_diff=w_rms_rel["rel_diff"], abs_diff=w_rms_rel["abs_diff"],
                                  mine=w_rms_rel["mine"], main=w_rms_rel["main"]),
            max_abs_diff_rms=dict(condition=w_rms_abs["condition"], quantity=w_rms_abs["quantity"],
                                  abs_diff=w_rms_abs["abs_diff"], rel_diff=w_rms_abs["rel_diff"]),
            max_abs_diff_vsw_nom_mV=dict(condition=w_vswn["condition"], diff_mV=w_vswn["abs_diff"] * 1e3),
            max_rel_diff_sigma_delta=dict(condition=w_sig["condition"], rel_diff=w_sig["rel_diff"]),
        ),
        task2=dict(
            passed=all(r["passed"] for r in vsw_rows) and all(r["passed"] for r in m0_rows)
            and all(r["passed"] for r in err_rows),
            vsw_k=dict(n=len(vsw_rows), n_failed=sum(not r["passed"] for r in vsw_rows),
                       max_abs_diff_mV=w_vsw["diff_mV"], at=dict(condition=w_vsw["condition"], instance=w_vsw["instance"])),
            m0=dict(n=len(m0_rows), n_failed=sum(not r["passed"] for r in m0_rows),
                    max_abs_diff_mV=w_m0["diff_mV"], at=dict(condition=w_m0["condition"], j=w_m0["j"]),
                    max_abs_diff_mV_used_thetas=w_m0u["diff_mV"]),
            error={node: dict(n=sum(r["node"] == node for r in err_rows),
                              n_failed=sum((r["node"] == node) and not r["passed"] for r in err_rows),
                              max_abs_diff_lsb=w["diff_lsb"],
                              at=dict(condition=w["condition"], instance=w["instance"], j=w["j"]),
                              mine=w["err_mine_lsb"], main=w["err_main_lsb"])
                   for node, w in w_err.items()},
        ),
        per_condition=per_cond,
    )
    summary["overall_passed"] = summary["task1"]["passed"] and summary["task2"]["passed"]
    with open(os.path.join(outdir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1, default=float)

    # ---------------- provenance
    ngv = subprocess.run(["ngspice", "-v"], capture_output=True, text=True).stdout.strip().splitlines()
    files = [CONFIG, PRED] + [f"{MCDIR}/arrays_{c}.npz" for c in cids]
    files += [os.path.join(HERE, f) for f in sorted(os.listdir(HERE)) if f.endswith((".py", ".md"))]
    info = dict(command=" ".join(sys.argv), python=sys.version, numpy=np.__version__,
                ngspice=[l for l in ngv if "ngspice" in l.lower()][:2], workdir=workroot,
                sha256={f: sha256(f) for f in files},
                models_manifest_sha256=sha256(f"{ROOT}/third_party/sky130_fd_pr_subset_corners/MANIFEST.json"))
    with open(os.path.join(outdir, "run_info.json"), "w") as f:
        json.dump(info, f, indent=1)
    print(json.dumps({k: summary[k] for k in ("overall_passed",)}), "->", outdir)


if __name__ == "__main__":
    main()
