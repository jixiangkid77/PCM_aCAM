"""E5 post-hoc diagnosis (not pre-registered): bound failures (|e| > 15 LSB) with nonzero conductance, P1 6T2M (lower
and upper bounds), condition all, 1 year, input remap, full E4 set. Uses the E4 machinery only.
Usage: python experiments/E5_posthoc_failures.py > results/E5_<stamp>/posthoc_nonzero_failures.txt"""
import importlib.util, json, numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("e4", ROOT / "experiments/E4_forest_system.py")
e4 = importlib.util.module_from_spec(spec); spec.loader.exec_module(e4)
from pcm_acam.trees import dataset_bundle, quantize
from pcm_acam.pcm_model import PCMParams
from pcm_acam.acam import Scheme
cfg = json.loads((ROOT / "experiments/E4_config.json").read_text())
inst = e4.Instrument.load(ROOT / "results/E4_spice_20261017_0602/instrument.npz")
p = PCMParams.from_dict(cfg["pcm_model"]); rep = Scheme(name="replica", kind="absolute", devices=1, g_lo=2.5, g_hi=22.5)
pts = [("1y", list(cfg["times_inference_s"]).index("1y"), float(cfg["times_inference_s"]["1y"]), "dac_remap")]
names = list(cfg["datasets"])
for ds in names:
    for split in (0, 1, 2):
        rows = dataset_bundle(ds, split)["rows"]; th = quantize(rows.cons_thr, 4); kind = rows.cons_kind
        for seed in range(20):
            draws, idx = e4.forest_state(cfg, names.index(ds), split, seed, th.size, 2000)
            cm = e4.CellModel(inst, cfg, "P1", "6T2M")
            res, g_t, clipped = e4.theta_effs(cm, th, kind, idx, draws, "all", pts, p, rep)
            for tl, nd, a_hat, te, g_now in res:
                e = 15 * (te - th)
                bad = np.flatnonzero((np.abs(e) > 15) & (g_now > 0))
                for k in bad:
                    print(ds, split, seed, "kind", int(kind[k]), "theta*15", int(round(15 * th[k])), "e", round(float(e[k]), 2),
                          "x_eff", round(float(te[k]), 3), "g_now", round(float(g_now[k]), 3), "g_target", round(float(g_t[k]), 3),
                          "clipped", bool(clipped[k]), "a_hat", round(a_hat, 4))
