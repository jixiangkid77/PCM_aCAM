"""E0.2 magnitude probe: transistor mismatch x PCM drift coupling in an analog CAM.

Question (frozen in experiments/E0_2_config.json): with the search-path threshold
verification absorbing each cell's transistor mismatch into the PCM conductance, does global
drift compensation at the input DAC re-expose the mismatch as the PCM drifts, and does
compensating at the divider supply avoid it?

Usage
-----
    python experiments/E0_2_mismatch_coupling.py            # full run
    python experiments/E0_2_mismatch_coupling.py --smoke    # 2 small datasets

Outputs go to results/E0_2_<YYYYMMDD_HHMM>[_smoke]/ (never overwritten):
    config_used.json, run_info.json, checks.json, refs.csv, per_eval.csv, clipping.csv,
    dataset_losses.csv, summary.csv, decision.json, run.log
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
from pcm_acam.acam import Scheme, cam_predict, device_states, make_draws, replica_factor  # noqa: E402
from pcm_acam.cell import (  # noqa: E402
    NODES,
    CellParams,
    closed_form_uniform_drift,
    effective_theta,
    mismatch,
    program_targets,
)
from pcm_acam.io_utils import environment_info, make_logger, make_run_dir, now_jst, sha256_file, write_json  # noqa: E402
from pcm_acam.pcm_model import PCMParams  # noqa: E402
from pcm_acam.trees import dataset_bundle, quantize, traverse_quantized  # noqa: E402

EXPERIMENT = "E0_2"


def cell_params(cfg: dict, r_in: float) -> CellParams:
    c = cfg["cell_model"]
    return CellParams(v_sl=c["v_sl"], v_th=c["v_th"], v_tx=c["v_tx"], g_lo=c["g_lo"], g_hi=c["g_hi"],
                      g_max=c["g_max"], r_in=r_in)


def mismatch_z(cfg: dict, ds_idx: int, split: int, dev: int, k: int) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng([cfg["base_seed"], ds_idx, split, dev, 2])
    return {"z_vth": rng.standard_normal(k), "z_vtx": rng.standard_normal(k), "z_beta": rng.standard_normal(k)}


def configs(cfg: dict, settings: List[str]) -> List[tuple]:
    """(mismatch label, programming mode) pairs evaluated."""
    out = [(m, "threshold") for m in settings]
    if "m20" in settings:
        out.append(("m20", "conductance"))
    return out


def run_checks(cfg: dict, p: PCMParams, datasets: List[str], log) -> dict:
    res: dict = {}
    worst_mis, worst_int = 0, 1.0
    for ds in datasets:
        b = dataset_bundle(ds, 0)
        rows, x, rf = b["rows"], b["x_te"], b["rf"]
        theta = quantize(rows.cons_thr, cfg["bits"])
        pred, integ = cam_predict(x, rows, theta)
        ref = traverse_quantized(rf, x, cfg["bits"]).argmax(axis=1)
        worst_mis = max(worst_mis, int(np.sum(pred != ref)))
        worst_int = min(worst_int, integ)
    res["check_1_cam_equals_quantized_tree_traversal"] = {
        "value": {"mismatched_predictions": worst_mis, "min_integrity": worst_int},
        "pass": worst_mis == 0 and worst_int == 1.0}

    theta = np.arange(16) / 15.0
    k = theta.size
    z = {"z_vth": np.random.default_rng([cfg["base_seed"], 901]).standard_normal(k),
         "z_vtx": np.random.default_rng([cfg["base_seed"], 902]).standard_normal(k),
         "z_beta": np.random.default_rng([cfg["base_seed"], 903]).standard_normal(k)}
    cp = cell_params(cfg, cfg["mismatch"]["settings"]["m20"]["r_in_V"])
    zero = mismatch(z, 0.0, 0.0, 0.0)
    w2 = 0.0
    for mode in ("threshold", "conductance"):
        g, _ = program_targets(theta, cp, zero, mode)
        for node in NODES:
            w2 = max(w2, float(np.max(np.abs(effective_theta(g, cp, zero, node, 1.0) - theta))))
    res["check_2_nominal_mapping_exact"] = {"value": w2, "pass": w2 <= 1e-12}

    draws = make_draws([cfg["base_seed"], 777], k, len(cfg["times_inference_s"]), cfg["replica_cells"])
    rep = Scheme(name="replica", kind="absolute", devices=1, g_lo=cp.g_lo, g_hi=cp.g_hi)
    w3 = 0.0
    g_t, _ = program_targets(theta, cp, zero, "threshold")
    for t_idx, t_inf in enumerate(cfg["times_inference_s"].values()):
        g = device_states(g_t[:, None], draws, "all", t_idx, float(t_inf), p)[:, 0]
        a_hat = replica_factor(rep, draws, "all", t_idx, float(t_inf), p)
        th = [effective_theta(g, cp, zero, n, a_hat) for n in ("dac", "supply", "ideal")]
        w3 = max(w3, float(np.max(np.abs(th[0] - th[2]))), float(np.max(np.abs(th[1] - th[2]))))
    res["check_3_nodes_equal_without_mismatch"] = {"value": w3, "pass": w3 <= 1e-12}

    p_u = dataclasses.replace(p, nu_mean_a=0.0, nu_std_min=0.0, nu_std_max=0.0, prog_noise_scale=0.0,
                              read_noise_scale=0.0)
    s = cfg["mismatch"]["settings"]["m20"]
    mm = mismatch(z, s["sigma_vth_V"], s["sigma_vtx_V"], s["sigma_beta"])
    g_t, _ = program_targets(theta, cp, mm, "threshold")
    ok = (g_t > 0.0) & (g_t < cp.g_max)
    w4 = 0.0
    for t_idx, t_inf in enumerate(cfg["times_inference_s"].values()):
        g = device_states(g_t[:, None], draws, "all", t_idx, float(t_inf), p_u)[:, 0]
        a_hat = replica_factor(rep, draws, "all", t_idx, float(t_inf), p_u)
        a = (float(t_inf) + p.t0) / p.t0
        a = a ** (-0.049) if float(t_inf) > 0 else 1.0
        w4 = max(w4, abs(a_hat - a))
        for node in NODES:
            got = effective_theta(g, cp, mm, node, a_hat)
            want = closed_form_uniform_drift(theta, cp, mm, node, a)
            w4 = max(w4, float(np.max(np.abs(got[ok] - want[ok]))))
    res["check_4_closed_forms_uniform_drift"] = {"value": w4, "pass": w4 <= 1e-12,
                                                  "unclipped_cells": int(ok.sum()), "cells": k}
    for key, v in res.items():
        log(f"  {key}: {v['value']} {'PASS' if v['pass'] else 'FAIL'}")
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
    ref_acc = {(r["dataset"], r["split"]): r["acc_ref"] for r in refs}
    per_ds = collections.defaultdict(list)
    integ = collections.defaultdict(list)
    for e in evals:
        loss = 100.0 * (ref_acc[(e["dataset"], e["split"])] - e["acc"])
        key = (e["mismatch"], e["programming"], e["node"], e["condition"], e["t"], e["dataset"])
        per_ds[key].append(loss)
        integ[key].append(e["integrity"])
    ds_rows = []
    for key, v in sorted(per_ds.items()):
        ds_rows.append({"mismatch": key[0], "programming": key[1], "node": key[2], "condition": key[3], "t": key[4],
                        "dataset": key[5], "n": len(v), "loss_pp_mean": float(np.mean(v)),
                        "loss_pp_std": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
                        "integrity_mean": float(np.mean(integ[key]))})
    by = collections.defaultdict(list)
    for r in ds_rows:
        by[(r["mismatch"], r["programming"], r["node"], r["condition"], r["t"])].append(r)
    summ = []
    for key, v in sorted(by.items()):
        losses = [r["loss_pp_mean"] for r in v]
        worst = max(v, key=lambda r: r["loss_pp_mean"])
        summ.append({"mismatch": key[0], "programming": key[1], "node": key[2], "condition": key[3], "t": key[4],
                     "n_datasets": len(v), "mean_loss_pp": float(np.mean(losses)), "max_loss_pp": float(np.max(losses)),
                     "worst_dataset": worst["dataset"], "integrity_mean": float(np.mean([r["integrity_mean"] for r in v]))})
    return ds_rows, summ


def decide(summ: List[dict], cfg: dict) -> dict:
    def get(node, t, mism="m20", prog="threshold", cond="all"):
        for r in summ:
            if (r["mismatch"], r["programming"], r["node"], r["condition"], r["t"]) == (mism, prog, node, cond, t):
                return r
        raise KeyError((mism, prog, node, cond, t))

    d = {}
    for t in ("30d", "365d"):
        ideal = get("ideal", t)["mean_loss_pp"]
        d[t] = {n: {"mean_loss_pp": get(n, t)["mean_loss_pp"], "max_loss_pp": get(n, t)["max_loss_pp"],
                    "minus_ideal_pp": get(n, t)["mean_loss_pp"] - ideal} for n in NODES}
    gap_dac = d["365d"]["dac"]["minus_ideal_pp"]
    gap_supply = d["365d"]["supply"]["minus_ideal_pp"]
    gate_a = gap_dac >= 1.0
    if not gate_a:
        verdict = "NO_GO"
    else:
        verdict = "GO_STRONG" if gap_supply >= 1.0 else "GO_WEAK"
    return {"primary": cfg["decision"]["primary"], "gate_A": {"pass": gate_a, "dac_minus_ideal_365d_pp": gap_dac},
            "classification": {"supply_minus_ideal_365d_pp": gap_supply}, "verdict": verdict, "by_time": d,
            "rule": cfg["decision"]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "experiments" / "E0_2_config.json"))
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    cfg_sha = sha256_file(cfg_path)
    run_dir = make_run_dir(ROOT / "results", EXPERIMENT, args.smoke)
    log = make_logger(run_dir / "run.log")
    t_start = time.time()
    log(f"E0.2 (PCM_aCAM) start; run dir {run_dir.name}; config sha256 {cfg_sha}")
    write_json(run_dir / "config_used.json", cfg)
    run_info = {"experiment": cfg["experiment"], "project": cfg["project"], "plan_version": cfg["plan_version"],
                "smoke": args.smoke, "start_jst": now_jst().isoformat(), "config_sha256": cfg_sha,
                "script_sha256": sha256_file(Path(__file__)), "package_version": PKG_VERSION,
                "environment": environment_info()}
    try:
        import sklearn
        run_info["environment"]["sklearn"] = sklearn.__version__
    except ImportError:
        pass
    write_json(run_dir / "run_info.json", run_info)

    p = PCMParams.from_dict(cfg["pcm_model"])
    all_names = list(cfg["datasets"].keys())
    settings = list(cfg["mismatch"]["settings"].keys())
    if args.smoke:
        datasets, split_seeds, n_dev = cfg["smoke"]["datasets"], cfg["smoke"]["split_seeds"], cfg["smoke"]["device_seeds"]
    else:
        datasets, split_seeds, n_dev = all_names, cfg["split_seeds"], cfg["device_seeds"]

    log("pre-run checks")
    checks = run_checks(cfg, p, datasets, log)
    write_json(run_dir / "checks.json", checks)
    if not all(c["pass"] for c in checks.values()):
        log("a check failed; stopping before any outcome is computed")
        return 2

    times = cfg["times_inference_s"]
    conds = [("prog", "t0", 0, 0.0)] + [("all", lab, i, float(v)) for i, (lab, v) in enumerate(times.items())]
    refs: List[dict] = []
    evals: List[dict] = []
    clip_rows: List[dict] = []
    for ds in datasets:
        ds_idx = all_names.index(ds)
        for split in split_seeds:
            t0 = time.time()
            b = dataset_bundle(ds, split)
            rows, x, y = b["rows"], b["x_te"], b["y_te"]
            classes = b["rf"].classes_
            k = rows.cons_thr.size
            theta = quantize(rows.cons_thr, cfg["bits"])
            pred_ref, integ_ref = cam_predict(x, rows, theta)
            acc_ref = float(np.mean(np.where(pred_ref >= 0, classes[np.maximum(pred_ref, 0)], -1) == y))
            refs.append({"dataset": ds, "split": split, "n_test": int(y.size), "n_rows": int(rows.row_tree.size),
                         "n_constraints": int(k), "acc_float": b["acc_float"], "acc_ref": acc_ref,
                         "integrity_ref": integ_ref})
            for dev in range(n_dev):
                draws = make_draws([cfg["base_seed"], ds_idx, split, dev], k, len(times), cfg["replica_cells"])
                z = mismatch_z(cfg, ds_idx, split, dev, k)
                for mlabel, prog in configs(cfg, settings):
                    s = cfg["mismatch"]["settings"][mlabel]
                    cp = cell_params(cfg, s["r_in_V"])
                    mm = mismatch(z, s["sigma_vth_V"], s["sigma_vtx_V"], s["sigma_beta"])
                    g_t, n_clip = program_targets(theta, cp, mm, prog)
                    clip_rows.append({"dataset": ds, "split": split, "device_seed": dev, "mismatch": mlabel,
                                      "programming": prog, "clipped": n_clip, "constraints": int(k)})
                    rep = Scheme(name="replica", kind="absolute", devices=1, g_lo=cp.g_lo, g_hi=cp.g_hi)
                    for cond, lab, t_idx, t_inf in conds:
                        g = device_states(g_t[:, None], draws, cond, t_idx, t_inf, p)[:, 0]
                        a_hat = replica_factor(rep, draws, cond, t_idx, t_inf, p)
                        for node in NODES:
                            th = effective_theta(g, cp, mm, node, a_hat)
                            pred, integ = cam_predict(x, rows, th)
                            acc = float(np.mean(np.where(pred >= 0, classes[np.maximum(pred, 0)], -1) == y))
                            evals.append({"dataset": ds, "split": split, "device_seed": dev, "mismatch": mlabel,
                                          "programming": prog, "node": node, "condition": cond, "t": lab,
                                          "a_hat": a_hat, "acc": acc, "integrity": integ})
            log(f"{ds} split {split}: constraints {k}, test {y.size}, 4-bit ref acc {acc_ref:.4f} "
                f"({time.time() - t0:.1f}s)")
        write_csv(run_dir / "refs.csv", refs)
        write_csv(run_dir / "per_eval.csv", evals)
        write_csv(run_dir / "clipping.csv", clip_rows)

    ds_rows, summ = aggregate(evals, refs)
    write_csv(run_dir / "dataset_losses.csv", ds_rows)
    write_csv(run_dir / "summary.csv", summ)
    if args.smoke:
        decision = {"verdict": "SMOKE (not interpreted)"}
    else:
        decision = decide(summ, cfg)
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
