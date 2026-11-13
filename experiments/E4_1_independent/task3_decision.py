"""Task 3 (decision): the gate of the frozen plan, V1GS: pass if |ln(t_c,sim / t_c,pred)| <= 0.05
for at least 90 % of the 48 cases, recomputed
  A  from rows.csv alone (t_sim_ns, t_pred_ns), compared with rows.csv (ln_ratio, pass),
     checks.json and decision.json;
  B  with t_sim_ns of rows.csv and my t_c,pred (task 2; t_s(N) of the run of record);
  C  as B, but my t_c,pred built with my SPICE t_s(N) (task 1);
  D  for the 6 cases of task 1: my SPICE crossing time against my t_c,pred (the extra
     LBUF-dominated cases of task 1 are reported in the same way, outside the flags).
Also reported (plan, 'reported_without_gate'): the row decisions, model (S >= 1 + (2N - 1) w0)
versus simulation (t_c,sim <= t_s), their disagreements and near ties (|ln S| < 0.05), and the
gate voltage of every constrained pull-down 2 ns after release, model versus simulation (the
main run's numbers recomputed from bounds.csv; mine for the bounds of the 6 cases of task 1).

Usage: python experiments/E4_1_independent/task3_decision.py --out <folder with the outputs of
tasks 1 and 2>
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True           # no __pycache__ folders in the repository

import argparse            # noqa: E402
import csv                 # noqa: E402
import json                # noqa: E402
import time                # noqa: E402
from pathlib import Path   # noqa: E402

import numpy as np         # noqa: E402

import e41i_common as K    # noqa: E402

GROUP_ORDER = [f"{d}|{t}|{n}" for d in ("P1", "P3") for t, n in K.POINTS]


def gate_stats(keys, t_sim, t_pred) -> dict:
    """Per-case |ln ratio|, pass flags and the gate outcome."""
    ln_r = np.log(np.asarray(t_sim, float) / np.asarray(t_pred, float))
    ok = np.abs(ln_r) <= K.GATE_LN
    frac = float(ok.mean())
    by_group = {}
    for g in GROUP_ORDER:
        sel = [i for i, k in enumerate(keys) if k == g]
        if sel:
            by_group[g] = float(np.max(np.abs(ln_r[sel])))
    return {"n_cases": int(ln_r.size), "n_within": int(ok.sum()), "pass_fraction": frac,
            "max_abs_ln_ratio": float(np.max(np.abs(ln_r))), "max_abs_ln_ratio_by_group": by_group,
            "outcome": "PASS" if frac >= K.GATE_FRACTION else "FAIL",
            "ln_ratio": ln_r.tolist(), "within": ok.tolist()}


def decisions(S, t_sim, t_s, n_feat, w0) -> dict:
    S = np.asarray(S, float)
    t_sim = np.asarray(t_sim, float)
    pred = S >= 1.0 + (2 * n_feat - 1) * w0
    sim = t_sim <= t_s
    dis = pred != sim
    near = np.abs(np.log(S)) < 0.05
    return {"pred_mismatch": pred.tolist(), "sim_mismatch": sim.tolist(),
            "n_pred_mismatch": int(pred.sum()), "n_sim_mismatch": int(sim.sum()),
            "disagreements": int(dis.sum()), "disagreements_near_tie": int((dis & near).sum()),
            "cases_near_tie": int(near.sum()), "near_tie_case_ids": np.where(near)[0].tolist(),
            "disagreement_case_ids": np.where(dis)[0].tolist()}


def read_csv(path: Path) -> list:
    with open(path) as f:
        return list(csv.DictReader(f))


def run(out_dir: Path, log=print) -> dict:
    t0 = time.time()
    main = K.load_main()
    cases = main["cases"]["cases"]
    rows = main["rows"]
    n_feat = int(main["cases"]["n_feat"])
    w0 = float(main["cases"]["w0"])
    keys = [K.group_key(c) for c in cases]
    t1 = json.loads((out_dir / "task1_summary.json").read_text())
    mc = read_csv(out_dir / "task2_model_cases.csv")
    mb = read_csv(out_dir / "task2_model_bounds.csv")
    gs = read_csv(out_dir / "task1_gates_2ns.csv")
    assert [int(r["case"]) for r in mc] == list(range(len(cases)))

    t_sim_main = [float(r["t_sim_ns"]) for r in rows]
    t_pred_main = [float(r["t_pred_ns"]) for r in rows]
    t_s_main = float(rows[0]["t_s_ns"])
    S_main = [float(r["S"]) for r in rows]
    S_ind = [float(r["S_indep"]) for r in mc]
    t_pred_ind = [float(r["t_pred_ns_indep"]) for r in mc]
    t_s_ind = float(t1["t_s_indep_ns"])
    t_pred_ind_ts = [t_s_ind * (1.0 + (2 * n_feat - 1) * w0) / s for s in S_ind]

    A = gate_stats(keys, t_sim_main, t_pred_main)
    B = gate_stats(keys, t_sim_main, t_pred_ind)
    Cc = gate_stats(keys, t_sim_main, t_pred_ind_ts)
    sel = t1["selected_cases"]
    extra = t1.get("extra_cases_LBUF_dominated", [])
    t_sim_mine = {o["case"]: o["t_sim_indep_ns"] for o in t1["cases"]}

    def d_entry(i):
        return {"t_sim_indep_ns": t_sim_mine[i], "t_pred_indep_ns": t_pred_ind[i],
                "ln_ratio": float(np.log(t_sim_mine[i] / t_pred_ind[i])),
                "ln_ratio_with_indep_t_s": float(np.log(t_sim_mine[i] / t_pred_ind_ts[i])),
                "ln_ratio_main": A["ln_ratio"][i]}
    D = {str(i): d_entry(i) for i in sel}
    D_extra = {str(i): d_entry(i) for i in extra}
    D_max = K.max_abs(d["ln_ratio"] for d in D.values())

    # A against the run of record
    ch = main["checks"]["V1GS"]
    dj = main["decision"]["V1GS"]
    ln_csv = [float(r["ln_ratio"]) for r in rows]
    pass_csv = [r["pass"] == "True" for r in rows]
    a_cmp = {
        "max_abs_diff_ln_ratio_vs_rows_csv": K.max_abs(np.array(A["ln_ratio"]) - np.array(ln_csv)),
        "pass_flags_equal_rows_csv": A["within"] == pass_csv,
        "pass_fraction_checks_json": ch["pass_fraction"], "pass_fraction_decision_json": dj["pass_fraction"],
        "max_abs_ln_ratio_checks_json": ch["max_abs_ln_ratio"],
        "abs_diff_max_ln_ratio_vs_checks_json": abs(A["max_abs_ln_ratio"] - ch["max_abs_ln_ratio"]),
        "max_abs_diff_by_group_vs_checks_json": K.max_abs(A["max_abs_ln_ratio_by_group"][g]
                                                         - ch["max_abs_ln_ratio_by_group"][g] for g in GROUP_ORDER),
        "outcome_decision_json": dj["outcome"], "n_cases_decision_json": dj["cases"],
    }
    # decisions of the rows
    dec_main = decisions(S_main, t_sim_main, t_s_main, n_feat, w0)
    dec_ind = decisions(S_ind, t_sim_main, t_s_ind, n_feat, w0)
    dec_cmp = {
        "pred_flags_equal_rows_csv": dec_main["pred_mismatch"] == [r["decision_pred_mismatch"] == "True" for r in rows],
        "sim_flags_equal_rows_csv": dec_main["sim_mismatch"] == [r["decision_sim_mismatch"] == "True" for r in rows],
        "disagreements_checks_json": ch["decision_disagreements"],
        "near_tie_checks_json": ch["decision_disagreements_near_tie"],
        "indep_pred_flags_equal": dec_ind["pred_mismatch"] == dec_main["pred_mismatch"],
        "indep_sim_flags_equal": dec_ind["sim_mismatch"] == dec_main["sim_mismatch"],
        "spice_6_cases_sim_flags_equal": all((t_sim_mine[i] <= t_s_ind) == dec_main["sim_mismatch"][i] for i in sel),
    }
    # gate voltages 2 ns after release: model versus simulation
    bm = main["bounds"]
    gd_main = [(float(b["gate_model_V"]) - float(b["gate_sim_2ns_V"])) * 1e3 for b in bm]
    gmain = {"max_abs_diff_mV": K.max_abs(gd_main),
             "max_abs_diff_mV_lower": K.max_abs(d for d, b in zip(gd_main, bm) if b["kind"] == "0"),
             "max_abs_diff_mV_upper": K.max_abs(d for d, b in zip(gd_main, bm) if b["kind"] == "1"),
             "max_abs_diff_vs_gate_diff_mV_column": K.max_abs(d - float(b["gate_diff_mV"]) for d, b in zip(gd_main, bm))}
    gmain["checks_json"] = main["checks"]["gates_2ns"]
    gmain["reproduced"] = all(abs(gmain[k] - main["checks"]["gates_2ns"][k]) <= 1e-9
                              for k in ("max_abs_diff_mV", "max_abs_diff_mV_lower", "max_abs_diff_mV_upper"))
    mb_by = {(int(b["case"]), int(b["bound"])): b for b in mb}
    # the main run's model-minus-SPICE gate differences, by my window class (all 199 bounds)
    bm_by = {}
    for b in bm:
        bm_by.setdefault(int(b["case"]), []).append(b)
    by_cls = {}
    for (ci, bi), b in mb_by.items():
        key = f"{b['gate_class']}/{b['type']}"
        d = by_cls.setdefault(key, {"n_bounds": 0, "max_abs_diff_mV": 0.0, "max_w": 0.0})
        mrow = bm_by[ci][bi]
        d["n_bounds"] += 1
        d["max_abs_diff_mV"] = max(d["max_abs_diff_mV"],
                                   abs(float(mrow["gate_model_V"]) - float(mrow["gate_sim_2ns_V"])) * 1e3)
        d["max_w"] = max(d["max_w"], float(mrow["w_model"]))
    gmain["by_window_class"] = dict(sorted(by_cls.items()))
    grows = []
    for g in gs:
        b = mb_by[(int(g["case"]), int(g["bound"]))]
        sp = float(g["gate_spice_indep_2ns_V"])
        grows.append({"case": int(g["case"]), "set": g["set"], "bound": int(g["bound"]), "type": b["type"],
                      "gate_class": b["gate_class"], "w_indep": float(b["w_indep"]),
                      "gate_used_V": float(b["gate_used_V"]), "gate_uncut_V": float(b["gate_uncut_V"]),
                      "gate_spice_indep_2ns_V": sp,
                      "diff_used_mV": (float(b["gate_used_V"]) - sp) * 1e3,
                      "diff_uncut_mV": (float(b["gate_uncut_V"]) - sp) * 1e3})
    K.write_csv(out_dir / "task3_gates_model_vs_spice.csv", grows)
    gmine = {}
    for cl in ("in_window", "node_outside_transfer_window", "u_cut_gate_0V"):
        for typ in ("LBUF", "UINV"):
            sub = [r for r in grows if r["gate_class"] == cl and r["type"] == typ]
            gmine[f"{cl}/{typ}"] = {"n_bounds": len(sub),
                                    "max_abs_diff_used_mV": K.max_abs(r["diff_used_mV"] for r in sub),
                                    "max_abs_diff_uncut_mV": K.max_abs(r["diff_uncut_mV"] for r in sub),
                                    "max_w": K.max_abs(r["w_indep"] for r in sub)}

    per_case = []
    for i, c in enumerate(cases):
        per_case.append({"case": i, "group": keys[i], "S_main": S_main[i], "S_indep": S_ind[i],
                         "t_sim_main_ns": t_sim_main[i], "t_pred_main_ns": t_pred_main[i],
                         "t_pred_indep_ns": t_pred_ind[i], "t_pred_indep_with_indep_t_s_ns": t_pred_ind_ts[i],
                         "ln_ratio_A_rows_csv": A["ln_ratio"][i], "ln_ratio_B_indep_t_pred": B["ln_ratio"][i],
                         "ln_ratio_C_indep_t_pred_and_t_s": Cc["ln_ratio"][i],
                         "t_sim_spice_indep_ns": t_sim_mine.get(i, float("nan")),
                         "ln_ratio_D_spice_indep": float(np.log(t_sim_mine[i] / t_pred_ind[i]))
                         if i in t_sim_mine else float("nan"),
                         "within_A": A["within"][i], "within_B": B["within"][i], "within_C": Cc["within"][i],
                         "pred_mismatch_main": dec_main["pred_mismatch"][i],
                         "pred_mismatch_indep": dec_ind["pred_mismatch"][i],
                         "sim_mismatch_rows_csv": dec_main["sim_mismatch"][i],
                         "ln_S_indep": float(np.log(S_ind[i]))})
    K.write_csv(out_dir / "task3_gate_cases.csv", per_case)

    strip = lambda d: {k: v for k, v in d.items() if k not in ("ln_ratio", "within")}   # noqa: E731
    flags = {
        "A_outcome_equals_decision_json": A["outcome"] == dj["outcome"],
        "A_reproduces_rows_csv_and_checks_json": bool(
            a_cmp["max_abs_diff_ln_ratio_vs_rows_csv"] <= 1e-12 and a_cmp["pass_flags_equal_rows_csv"]
            and A["pass_fraction"] == ch["pass_fraction"] == dj["pass_fraction"]
            and a_cmp["abs_diff_max_ln_ratio_vs_checks_json"] <= 1e-12
            and a_cmp["max_abs_diff_by_group_vs_checks_json"] <= 1e-12),
        "B_outcome_equals_decision_json": B["outcome"] == dj["outcome"],
        "B_per_case_flags_equal_A": B["within"] == A["within"],
        "C_outcome_equals_decision_json": Cc["outcome"] == dj["outcome"],
        "D_all_six_within_gate": all(abs(d["ln_ratio"]) <= K.GATE_LN for d in D.values()),
        "row_decisions_reproduced": bool(dec_cmp["pred_flags_equal_rows_csv"] and dec_cmp["sim_flags_equal_rows_csv"]
                                         and dec_main["disagreements"] == ch["decision_disagreements"]
                                         and dec_main["disagreements_near_tie"] == ch["decision_disagreements_near_tie"]
                                         and dec_cmp["indep_pred_flags_equal"] and dec_cmp["indep_sim_flags_equal"]),
        "gates_2ns_checks_json_reproduced": gmain["reproduced"],
    }
    summary = {
        "rule": f"pass if |ln(t_c,sim / t_c,pred)| <= {K.GATE_LN} for at least {int(100 * K.GATE_FRACTION)} % of the 48 cases",
        "A_rows_csv": strip(A), "A_vs_run_of_record": a_cmp,
        "B_rows_csv_t_sim_indep_t_pred": strip(B),
        "C_rows_csv_t_sim_indep_t_pred_indep_t_s": dict(strip(Cc), t_s_indep_ns=t_s_ind),
        "B_minus_A_max_abs_ln_ratio": K.max_abs(np.array(B["ln_ratio"]) - np.array(A["ln_ratio"])),
        "C_minus_A_max_abs_ln_ratio": K.max_abs(np.array(Cc["ln_ratio"]) - np.array(A["ln_ratio"])),
        "D_spice_indep_six_cases": D, "D_max_abs_ln_ratio": D_max,
        "D_extra_LBUF_dominated_cases": D_extra,
        "D_extra_max_abs_ln_ratio": K.max_abs(d["ln_ratio"] for d in D_extra.values()),
        "D_extra_all_within_gate": all(abs(d["ln_ratio"]) <= K.GATE_LN for d in D_extra.values()),
        "row_decisions_main": {k: v for k, v in dec_main.items() if k not in ("pred_mismatch", "sim_mismatch")},
        "row_decisions_indep": {k: v for k, v in dec_ind.items() if k not in ("pred_mismatch", "sim_mismatch")},
        "row_decisions_compare": dec_cmp,
        "gates_2ns_main_recomputed": gmain,
        "gates_2ns_indep_model_vs_indep_spice": gmine,
        "flags": flags,
        "pass": bool(all(flags.values())),
        "elapsed_s": round(time.time() - t0, 1),
    }
    with open(out_dir / "task3_summary.json", "w") as f:
        json.dump(summary, f, indent=1)
    log(f"[task3] V1GS from rows.csv: {A['n_within']}/{A['n_cases']} within, max |ln| {A['max_abs_ln_ratio']:.4f} -> "
        f"{A['outcome']}; with my t_pred: {B['n_within']}/{B['n_cases']}, max {B['max_abs_ln_ratio']:.4f} -> {B['outcome']}; "
        f"with my t_pred and t_s: {Cc['n_within']}/{Cc['n_cases']} -> {Cc['outcome']}; my 6 SPICE rows: max |ln| "
        f"{D_max:.4f}; decision.json {dj['outcome']}; PASS={summary['pass']}")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="folder that holds the outputs of tasks 1 and 2")
    a = ap.parse_args()
    run(Path(a.out).resolve())
