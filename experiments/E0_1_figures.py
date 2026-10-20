"""Figures for E0.1 (PCM_aCAM).

Usage
-----
    python experiments/E0_1_figures.py results/E0_1_<stamp>

Writes figures/E0_1_fig1_loss_vs_time.{png,pdf} and
figures/E0_1_fig2_datasets_integrity.{png,pdf}. Every plotted number is read
from scheme_summary.csv / dataset_losses.csv of the given run (4 bits).
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BLUE, ORANGE, GREEN, YELLOW, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6b6b6b"
T_LABELS = ["t0", "1h", "1d", "30d", "365d"]
X_TICKS = ["t$_0$\n(prog)", "t$_0$", "1 h", "1 d", "30 d", "1 y"]
DATASETS = ["iris", "wine", "breast_cancer", "digits", "synthetic"]


def read_csv(path: Path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def series(rows, scheme, track, field="mean_loss_pp", bits="4"):
    out = []
    for cond, t in [("prog", "t0")] + [("all", t) for t in T_LABELS]:
        r = [r for r in rows if (r["bits"], r["scheme"], r["track"], r["condition"], r["t"]) == (bits, scheme, track, cond, t)]
        out.append(float(r[0][field]))
    return out


def main() -> int:
    run = Path(sys.argv[1])
    summ = read_csv(run / "scheme_summary.csv")
    dsl = read_csv(run / "dataset_losses.csv")
    fig_dir = ROOT / "figures"
    fig_dir.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    x = list(range(len(X_TICKS)))

    # Figure 1: mean loss versus time, untracked (a) and tracked / ratio (b)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.6, 3.1), gridspec_kw={"width_ratios": [1, 1.12]})
    for sch, col, lab in (("V1_abs_full", BLUE, "V1 full range"), ("V2_abs_high", ORANGE, "V2 high range")):
        y = series(summ, sch, "none")
        ax1.plot(x, y, "-o", color=col, ms=4, lw=1.6)
        ax1.annotate(f"{lab}\n{y[3]:.1f} pp at 1 d", (x[-1], y[-1]), xytext=(6, -4),
                     textcoords="offset points", ha="left", va="center", color=col, fontsize=7.5)
    ax1.axhline(10.0, color=GREY, lw=0.8, ls="--")
    ax1.text(3.0, 5.0, "gate B: ≥ 10 pp at 1 d", color=GREY, fontsize=7.5)
    ax1.set_xlim(-0.3, len(x) + 0.9)
    ax1.set_xticks(x, X_TICKS)
    ax1.set_ylabel("accuracy loss vs ideal 4-bit CAM (pp)")
    ax1.set_title("(a) no drift tracking", fontsize=9, loc="left")
    ax1.set_ylim(-5, 70)

    tracked = (("V1_abs_full", "replica", BLUE, "-", "V1 + replica"),
               ("V2_abs_high", "replica", ORANGE, "-", "V2 + replica"),
               ("V3_ratio", "none", GREEN, "-", "V3 ratio"),
               ("V2d_abs_high_dual", "replica", YELLOW, "--", "V2 dual + replica (secondary)"))
    for sch, tr, col, ls, lab in tracked:
        y = series(summ, sch, tr)
        ax2.plot(x, y, ls, marker="o", color=col, ms=4, lw=1.6)
        ax2.annotate(f"{lab}\n{y[4]:.3f} pp at 30 d", (x[-1], y[-1]), xytext=(6, 0), textcoords="offset points",
                     ha="left", va="center", color=col, fontsize=7.5)
    ax2.set_xlim(-0.3, len(x) + 1.6)
    ax2.axhline(5.0, color=GREY, lw=0.8, ls="--")
    ax2.text(0.05, 5.25, "gate C: ≤ 5 pp at 30 d", color=GREY, fontsize=7.5)
    ax2.axhline(0.0, color="#bbbbbb", lw=0.6)
    ax2.set_xticks(x, X_TICKS)
    ax2.set_title("(b) replica tracking or ratio threshold", fontsize=9, loc="left")
    ax2.set_ylim(-1.5, 8)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(fig_dir / f"E0_1_fig1_loss_vs_time.{ext}", dpi=200)
    plt.close(fig)

    # Figure 2: per-dataset loss at 30 d (a) and path integrity versus time (b)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.0), gridspec_kw={"width_ratios": [1.25, 1]})
    cands = (("V1_abs_full", "replica", BLUE, "V1 + replica"), ("V2_abs_high", "replica", ORANGE, "V2 + replica"),
             ("V3_ratio", "none", GREEN, "V3 ratio"))
    w = 0.26
    for i, (sch, tr, col, lab) in enumerate(cands):
        vals = []
        for ds in DATASETS:
            r = [r for r in dsl if (r["bits"], r["scheme"], r["track"], r["condition"], r["t"], r["dataset"]) ==
                 ("4", sch, tr, "all", "30d", ds)]
            vals.append(float(r[0]["loss_pp_mean"]))
        xs = [k + (i - 1) * w for k in range(len(DATASETS))]
        ax1.bar(xs, vals, width=w, color=col, label=lab)
    ax1.axhline(0.0, color="#bbbbbb", lw=0.6)
    ax1.set_xticks(range(len(DATASETS)), ["iris", "wine", "breast\ncancer", "digits", "synthetic"])
    ax1.set_ylabel("accuracy loss at 30 d (pp)")
    ax1.set_title("(a) per dataset, 30 days, 4 bits", fontsize=9, loc="left")
    ax1.legend(frameon=False, fontsize=7.5, loc="upper left")

    for sch, tr, col, lab in cands + (("V1_abs_full", "none", GREY, "V1 untracked"),):
        y = series(summ, sch, tr, field="integrity_mean")
        ax2.plot(x, y, "-o", color=col, ms=3.5, lw=1.4)
        ax2.annotate(lab, (x[-1], y[-1]), xytext=(3, -3), textcoords="offset points", color=col, fontsize=7.5)
    ax2.axhline(1.0, color="#bbbbbb", lw=0.6)
    ax2.set_xticks(x, X_TICKS)
    ax2.set_ylim(0.4, 1.05)
    ax2.set_xlim(-0.3, len(x) + 1.4)
    ax2.set_ylabel("path integrity (exactly one match per tree)")
    ax2.set_title("(b) path integrity", fontsize=9, loc="left")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(fig_dir / f"E0_1_fig2_datasets_integrity.{ext}", dpi=200)
    plt.close(fig)
    print("figures written to", fig_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
