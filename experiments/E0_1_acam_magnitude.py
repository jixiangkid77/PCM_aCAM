"""E0.1 magnitude probe for a PCM analog CAM running random forests.

Questions (frozen in experiments/E0_1_config.json):
  A  With programming noise only, is a PCM analog CAM close to the ideal quantized CAM?
  B  Without drift tracking, does the accuracy collapse within a day?
  C  Does global drift tracking or a ratio threshold keep it close for 30 days?

Usage
-----
    python experiments/E0_1_acam_magnitude.py            # full run
    python experiments/E0_1_acam_magnitude.py --smoke    # 2 small datasets

Outputs go to results/E0_1_<YYYYMMDD_HHMM>[_smoke]/ (never overwritten):
    config_used.json, run_info.json, checks.json, refs.csv, per_eval.csv,
    dataset_losses.csv, scheme_summary.csv, decision.json, run.log
"""

from __future__ import annotations

import argparse
import collections
import csv
import dataclasses
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pcm_acam import __version__ as PKG_VERSION  # noqa: E402
from pcm_acam.acam import (  # noqa: E402
    Scheme,
    cam_predict,
    device_states,
    device_targets,
    effective_theta,
    make_draws,
    replica_factor,
)
from pcm_acam.io_utils import (  # noqa: E402
    environment_info,
    make_logger,
    make_run_dir,
    now_jst,
    sha256_file,
    write_json,
)
from pcm_acam.pcm_model import PCMParams, nu_mean_std, prog_sigma  # noqa: E402
from pcm_acam.trees import dataset_bundle, quantize, traverse_quantized  # noqa: E402

EXPERIMENT = "E0_1"
SCHEME_TRACKS = [
    ("V1_abs_full", "none"),
    ("V1_abs_full", "replica"),
    ("V2_abs_high", "none"),
    ("V2_abs_high", "replica"),
    ("V3_ratio", "none"),
    ("V2d_abs_high_dual", "replica"),
]


def run_checks(cfg: dict, p: PCMParams, schemes: Dict[str, Scheme], datasets: List[str], log) -> dict:
    res: dict = {}
    worst_pred_mismatch, worst_integrity, worst_agree = 0, 1.0, 1.0
    info_30bit, info_ties = {}, {}
    for ds in datasets:
        b = dataset_bundle(ds, 0)
        rows, x, rf = b["rows"], b["x_te"], b["rf"]
        theta = quantize(rows.cons_thr, 4)
        pred, integ = cam_predict(x, rows, theta)
        ref = traverse_quantized(rf, x, 4).argmax(axis=1)
        worst_pred_mismatch = max(worst_pred_mismatch, int(np.sum(pred != ref)))
        worst_integrity = min(worst_integrity, integ)
        sk = rf.predict(x)
        predf, _ = cam_predict(x, rows, rows.cons_thr)
        agree = float(np.mean(rf.classes_[predf] == sk))
        worst_agree = min(worst_agree, agree)
        pred30, _ = cam_predict(x, rows, quantize(rows.cons_thr, 30))
        info_30bit[ds] = float(np.mean(rf.classes_[pred30] == sk))
        # test values lying exactly on a split threshold (possible with discrete features)
        info_ties[ds] = int(np.sum(x[:, rows.cons_feat] == rows.cons_thr[None, :]))
        log(f"  {ds}: 4-bit CAM vs traversal mismatches {int(np.sum(pred != ref))}, integrity {integ:.4f}; "
            f"float-threshold CAM vs sklearn agreement {agree:.4f} (30-bit: {info_30bit[ds]:.4f}, "
            f"exact value-threshold ties {info_ties[ds]})")
    res["check_1_cam_equals_quantized_tree_traversal"] = {
        "value": {"mismatched_predictions": worst_pred_mismatch, "min_integrity": worst_integrity},
        "pass": worst_pred_mismatch == 0 and worst_integrity == 1.0,
    }
    res["check_2_float_thresholds_equal_sklearn"] = {
        "value": worst_agree, "pass": worst_agree >= 0.999,
        "info_30bit_agreement": info_30bit, "info_exact_ties_sample_constraint_pairs": info_ties,
    }

    hand_sigma = {25.0: 0.26348 + 1.9650 - 1.1731, 0.0: 0.26348, 12.5: 0.26348 + 0.9825 - 0.293275}
    hand_nu = {25.0: (0.049, 0.008), 0.025: (0.1, 0.045),
               5.0: (0.0155 * 1.6094379124341003 + 0.0244, 0.0125 * 1.6094379124341003 - 0.0059)}
    w6 = 0.0
    for g, sv in hand_sigma.items():
        w6 = max(w6, abs(float(prog_sigma(np.array([g]), p)[0]) - sv))
    for g, (m, s) in hand_nu.items():
        mu, sd = nu_mean_std(np.array([g]), p)
        w6 = max(w6, abs(float(mu[0]) - m), abs(float(sd[0]) - s))
    res["check_3_pcm_hand_values"] = {"value": w6, "pass": w6 <= cfg["checks"]["check_3_pcm_hand_values"]}

    p0 = dataclasses.replace(p, prog_noise_scale=0.0, read_noise_scale=0.0, nu_std_min=0.0, nu_std_max=0.0)
    theta = np.arange(16) / 15.0
    draws = make_draws([cfg["base_seed"], 777], theta.size, len(cfg["times_inference_s"]), cfg["replica_cells"])
    w4 = 0.0
    for name, track in (("V2_abs_high", "replica"), ("V3_ratio", "none")):
        sch = schemes[name]
        tg = device_targets(theta, sch)
        for t_idx, t_inf in enumerate(cfg["times_inference_s"].values()):
            g = device_states(tg, draws, "all", t_idx, float(t_inf), p0)
            a_hat = replica_factor(sch, draws, "all", t_idx, float(t_inf), p0) if track == "replica" else 1.0
            w4 = max(w4, float(np.max(np.abs(effective_theta(g, sch, a_hat) - theta))))
    res["check_4_tracking_exact_without_spread"] = {"value": w4, "pass": w4 <= 1e-12}

    for k, v in res.items():
        log(f"  {k}: {v['value']} {'PASS' if v['pass'] else 'FAIL'}")
    return res


def write_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        for r in rows:
            wr.writerow({k: (repr(v) if isinstance(v, float) else v) for k, v in r.items()})


def aggregate(evals: List[dict], refs: List[dict]):
    ref_acc = {(r["dataset"], r["split"], r["bits"]): r["acc_ref"] for r in refs}
    per_ds = collections.defaultdict(list)
    integ = collections.defaultdict(list)
    for e in evals:
        loss = 100.0 * (ref_acc[(e["dataset"], e["split"], e["bits"])] - e["acc"])
        key = (e["bits"], e["scheme"], e["track"], e["condition"], e["t"], e["dataset"])
        per_ds[key].append(loss)
        integ[key].append(e["integrity"])
    ds_rows = []
    for key, v in sorted(per_ds.items()):
        ds_rows.append({"bits": key[0], "scheme": key[1], "track": key[2], "condition": key[3], "t": key[4],
                        "dataset": key[5], "n": len(v), "loss_pp_mean": float(np.mean(v)),
                        "loss_pp_std": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
                        "integrity_mean": float(np.mean(integ[key]))})
    by_scheme = collections.defaultdict(list)
    for r in ds_rows:
        by_scheme[(r["bits"], r["scheme"], r["track"], r["condition"], r["t"])].append(r)
    sch_rows = []
    for key, v in sorted(by_scheme.items()):
        losses = [r["loss_pp_mean"] for r in v]
        worst = max(v, key=lambda r: r["loss_pp_mean"])
        sch_rows.append({"bits": key[0], "scheme": key[1], "track": key[2], "condition": key[3], "t": key[4],
                         "n_datasets": len(v), "mean_loss_pp": float(np.mean(losses)),
                         "max_loss_pp": float(np.max(losses)), "worst_dataset": worst["dataset"],
                         "integrity_mean": float(np.mean([r["integrity_mean"] for r in v]))})
    return ds_rows, sch_rows


def decide(sch_rows: List[dict], cfg: dict) -> dict:
    b = cfg["decision"]["primary_bits"]

    def get(scheme, track, cond, t):
        for r in sch_rows:
            if (r["bits"], r["scheme"], r["track"], r["condition"], r["t"]) == (b, scheme, track, cond, t):
                return r
        raise KeyError((scheme, track, cond, t))

    prog = {s: get(s, "none", "prog", "t0") for s in ("V1_abs_full", "V2_abs_high", "V3_ratio")}
    a_ok = {s: (r["mean_loss_pp"] <= 2.0 and r["max_loss_pp"] <= 5.0) for s, r in prog.items()}
    gate_a = any(a_ok.values())
    best_abs = min(("V1_abs_full", "V2_abs_high"), key=lambda s: prog[s]["mean_loss_pp"])
    b_row = get(best_abs, "none", "all", "1d")
    gate_b = b_row["mean_loss_pp"] >= 10.0
    c_cands = {"V1_abs_full + replica": ("V1_abs_full", "replica"),
               "V2_abs_high + replica": ("V2_abs_high", "replica"),
               "V3_ratio": ("V3_ratio", "none")}
    c_detail = {}
    gate_c = False
    for label, (s, tr) in c_cands.items():
        r30 = get(s, tr, "all", "30d")
        ok = a_ok[s] and r30["mean_loss_pp"] <= 5.0
        c_detail[label] = {"meets_A": a_ok[s], "mean_loss_30d_pp": r30["mean_loss_pp"],
                           "max_loss_30d_pp": r30["max_loss_pp"], "pass": ok}
        gate_c = gate_c or ok
    verdict = "GO" if (gate_a and gate_b and gate_c) else "NO_GO"
    failing = [g for g, ok in (("A", gate_a), ("B", gate_b), ("C", gate_c)) if not ok]
    return {
        "bits": b,
        "gate_A": {"pass": gate_a, "prog_t0": {s: {"mean_loss_pp": r["mean_loss_pp"], "max_loss_pp": r["max_loss_pp"],
                                                     "worst_dataset": r["worst_dataset"], "meets": a_ok[s]}
                                                 for s, r in prog.items()}},
        "gate_B": {"pass": gate_b, "absolute_scheme": best_abs, "untracked_mean_loss_1d_pp": b_row["mean_loss_pp"],
                   "untracked_max_loss_1d_pp": b_row["max_loss_pp"]},
        "gate_C": {"pass": gate_c, "candidates": c_detail},
        "verdict": verdict,
        "failing_gates": failing,
        "rule": cfg["decision"],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "experiments" / "E0_1_config.json"))
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    cfg_sha = sha256_file(cfg_path)
    run_dir = make_run_dir(ROOT / "results", EXPERIMENT, args.smoke)
    log = make_logger(run_dir / "run.log")
    t_start = time.time()
    log(f"E0.1 (PCM_aCAM) start; run dir {run_dir.name}; config sha256 {cfg_sha}")
    write_json(run_dir / "config_used.json", cfg)
    run_info = {
        "experiment": cfg["experiment"], "project": cfg["project"], "plan_version": cfg["plan_version"],
        "smoke": args.smoke, "start_jst": now_jst().isoformat(), "config_sha256": cfg_sha,
        "script_sha256": sha256_file(Path(__file__)), "package_version": PKG_VERSION,
        "environment": environment_info(),
    }
    try:
        import sklearn
        run_info["environment"]["sklearn"] = sklearn.__version__
    except ImportError:
        pass
    write_json(run_dir / "run_info.json", run_info)

    p = PCMParams.from_dict(cfg["pcm_model"])
    schemes = {n: Scheme.from_cfg(n, d) for n, d in cfg["schemes"].items()}
    if args.smoke:
        datasets = cfg["smoke"]["datasets"]
        split_seeds = cfg["smoke"]["split_seeds"]
        n_dev = cfg["smoke"]["device_seeds"]
        bits_list = cfg["smoke"]["bits"]
    else:
        datasets = list(cfg["datasets"].keys())
        split_seeds = cfg["split_seeds"]
        n_dev = cfg["device_seeds"]
        bits_list = [cfg["bits_primary"]] + cfg["bits_secondary"]

    log("pre-run checks")
    checks = run_checks(cfg, p, schemes, datasets, log)
    write_json(run_dir / "checks.json", checks)
    if not all(c["pass"] for c in checks.values()):
        log("a check failed; stopping before any outcome is computed")
        return 2

    times = cfg["times_inference_s"]
    refs: List[dict] = []
    evals: List[dict] = []
    all_names = list(cfg["datasets"].keys())
    for ds in datasets:
        ds_idx = all_names.index(ds)
        for split in split_seeds:
            t0 = time.time()
            b = dataset_bundle(ds, split)
            rows, x, y = b["rows"], b["x_te"], b["y_te"]
            classes = b["rf"].classes_
            n_cons = rows.cons_thr.size
            for bits in bits_list:
                theta = quantize(rows.cons_thr, bits)
                pred_ref, integ_ref = cam_predict(x, rows, theta)
                acc_ref = float(np.mean(np.where(pred_ref >= 0, classes[np.maximum(pred_ref, 0)], -1) == y))
                refs.append({"dataset": ds, "split": split, "bits": bits, "n_test": int(y.size),
                             "n_rows": int(rows.row_tree.size), "n_constraints": int(n_cons),
                             "acc_float": b["acc_float"], "acc_ref": acc_ref, "integrity_ref": integ_ref})
                for dev in range(n_dev):
                    draws = make_draws([cfg["base_seed"], ds_idx, split, dev], n_cons, len(times), cfg["replica_cells"])
                    for sch_name, track in SCHEME_TRACKS:
                        sch = schemes[sch_name]
                        tg = device_targets(theta, sch)
                        conds = [("prog", "t0", 0, 0.0)] + [("all", lab, i, float(v)) for i, (lab, v) in enumerate(times.items())]
                        for cond, lab, t_idx, t_inf in conds:
                            g = device_states(tg, draws, cond, t_idx, t_inf, p)
                            a_hat = replica_factor(sch, draws, cond, t_idx, t_inf, p) if track == "replica" else 1.0
                            th = effective_theta(g, sch, a_hat)
                            pred, integ = cam_predict(x, rows, th)
                            acc = float(np.mean(np.where(pred >= 0, classes[np.maximum(pred, 0)], -1) == y))
                            evals.append({"dataset": ds, "split": split, "bits": bits, "device_seed": dev,
                                          "scheme": sch_name, "track": track, "condition": cond, "t": lab,
                                          "a_hat": a_hat, "acc": acc, "integrity": integ})
            log(f"{ds} split {split}: rows {rows.row_tree.size}, constraints {n_cons}, test {y.size}, "
                f"float acc {b['acc_float']:.4f}, 4-bit ref acc "
                f"{[r['acc_ref'] for r in refs if r['dataset'] == ds and r['split'] == split and r['bits'] == 4][0]:.4f} "
                f"({time.time() - t0:.1f}s)")
        write_csv(run_dir / "refs.csv", refs)
        write_csv(run_dir / "per_eval.csv", evals)

    ds_rows, sch_rows = aggregate(evals, refs)
    write_csv(run_dir / "dataset_losses.csv", ds_rows)
    write_csv(run_dir / "scheme_summary.csv", sch_rows)
    if args.smoke:
        decision = {"verdict": "SMOKE (not interpreted)"}
        try:
            decision.update(decide(sch_rows, cfg))
            decision["verdict_if_full"] = decision.pop("verdict")
            decision["verdict"] = "SMOKE (not interpreted)"
        except KeyError as exc:
            decision["note"] = f"decision not computable on smoke settings: {exc}"
    else:
        decision = decide(sch_rows, cfg)
    decision["config_sha256"] = cfg_sha
    decision["run_dir"] = run_dir.name
    write_json(run_dir / "decision.json", decision)
    run_info["end_jst"] = now_jst().isoformat()
    run_info["elapsed_s"] = round(time.time() - t_start, 1)
    write_json(run_dir / "run_info.json", run_info)
    log(f"verdict {decision['verdict']}; elapsed {run_info['elapsed_s']} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
