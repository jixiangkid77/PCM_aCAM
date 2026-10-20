"""Figures for E4 (PCM_aCAM): forest accuracy with transistor-level bound errors, PCM noise and match-line crosstalk.

Usage: python experiments/E4_figures.py results/E4_<stamp>
Reads summary.csv and deltas.csv; writes figures/E4_fig{1,2,3}_*.{png,pdf}.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BLUE, ORANGE, GREY, INK, INK2, BAND, DARK = "#2a78d6", "#eb6834", "#8a8a86", "#1f1f1f", "#52514e", "#e6e5e1", "#52514e"
NODE_COLOR = {"dac_remap": BLUE, "supply": ORANGE, "ideal": GREY}
NODE_LABEL = {"dac_remap": "input remap", "supply": "divider supply", "ideal": "ideal global scaling (reference)"}
TIMES = ["t0", "1d", "30d", "1y"]
TIME_LABEL = {"t0": "t0", "1d": "1 day", "30d": "30 days", "1y": "1 year"}
ARM_LABEL = {"xt": "6T2M as published (crosstalk)", "is": "6T2M, ideal sensing (reference)",
             "gs": "gain stage in the lower branch", "gi": "gain stage, ideal sensing (reference)"}
DATASETS = ["iris", "wine", "breast_cancer", "digits", "synthetic"]


def rcsv(p: Path):
    with open(p, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def style():
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": GREY, "xtick.color": INK2, "ytick.color": INK2,
                         "axes.labelcolor": INK2, "font.family": "DejaVu Sans"})


def save(fig, name):
    out = ROOT / "figures"
    out.mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def loss(summary, design, arm, cond, time, node):
    for r in summary:
        if (r["design"], r["arm"], r["condition"], r["time"], r["node"]) == (design, arm, cond, time, node):
            return float(r["mean_loss_pp"])
    raise KeyError((design, arm, cond, time, node))


def fig1(summary, cond="all"):
    """Mean accuracy loss over time: designs x arms (crosstalk modelled)."""
    fig, axs = plt.subplots(2, 2, figsize=(7.6, 5.4), sharex=True)
    xs = np.arange(len(TIMES))
    ymax = 0.0
    for i, did in enumerate(("P1", "P3")):
        for j, arm in enumerate(("xt", "gs")):
            ax = axs[i, j]
            t0v = loss(summary, did, arm, cond, "t0", "t0")
            for node in ("ideal", "dac_remap", "supply"):
                ys = [t0v] + [loss(summary, did, arm, cond, t, node) for t in TIMES[1:]]
                ymax = max(ymax, max(ys))
                ax.plot(xs, ys, marker="o", color=NODE_COLOR[node], ms=4.5, lw=2 if node != "ideal" else 1.2,
                        ls="--" if node == "ideal" else "-", label=NODE_LABEL[node], zorder=3 if node != "ideal" else 2)
            ref_arm = "is" if arm == "xt" else "gi"
            ys_ref = [loss(summary, did, ref_arm, cond, "t0", "t0")] + \
                     [loss(summary, did, ref_arm, cond, t, "supply") for t in TIMES[1:]]
            ax.plot(xs, ys_ref, ":", color=ORANGE, lw=1.0, alpha=0.8, zorder=1,
                    label="divider supply, ideal sensing")
            ys_ref = [loss(summary, did, ref_arm, cond, "t0", "t0")] + \
                     [loss(summary, did, ref_arm, cond, t, "dac_remap") for t in TIMES[1:]]
            ax.plot(xs, ys_ref, ":", color=BLUE, lw=1.0, alpha=0.8, zorder=1, label="input remap, ideal sensing")
            ax.set_title(f"{did}: {ARM_LABEL[arm]}", fontsize=8.5, loc="left", color=INK)
            ax.set_xticks(xs)
            ax.set_xticklabels([TIME_LABEL[t] for t in TIMES])
            if j == 0:
                ax.set_ylabel("accuracy loss (pp)")
    for ax in axs.flat:
        ax.set_ylim(0, ymax * 1.12 + 0.2)
        ax.grid(axis="y", color=BAND, lw=0.6)
    h, lab = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=3, fontsize=7.2, frameon=False, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, "E4_fig1_loss_over_time")


def fig2(deltas):
    """Paired difference loss(input remap) - loss(divider supply) at 1 year with 95 % CI."""
    rows = []
    for did in ("P1", "P3"):
        for arm in ("xt", "is", "gs", "gi"):
            for cond in ("all", "unif"):
                for r in deltas:
                    if (r["design"], r["arm"], r["condition"], r["time"], r["contrast"]) == (did, arm, cond, "1y",
                                                                                           "dac_minus_supply"):
                        rows.append((did, arm, cond, float(r["mean"]), float(r["ci_lo"]), float(r["ci_hi"])))
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    labels = []
    y = 0
    yt = []
    for did in ("P1", "P3"):
        for arm in ("xt", "is", "gs", "gi"):
            for cond, off, mk in (("all", 0.16, "o"), ("unif", -0.16, "s")):
                sel = [r for r in rows if r[0] == did and r[1] == arm and r[2] == cond]
                if not sel:
                    continue
                _, _, _, m, lo, hi = sel[0]
                col = ORANGE if lo > 0 else (BLUE if hi < 0 else GREY)
                ax.plot([lo, hi], [y + off, y + off], color=col, lw=1.6)
                ax.plot(m, y + off, mk, color=col, ms=5.5 if mk == "o" else 4.5,
                        mfc=col if cond == "all" else "white", mew=1.2)
            yt.append(y)
            labels.append(f"{did}  {ARM_LABEL[arm]}")
            y -= 1
        y -= 0.5
    ax.axvline(0, color=GREY, lw=0.8)
    ax.set_yticks(yt)
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.set_xlabel("loss(input remap) − loss(divider supply) at 1 year (pp)\n← input remap better      divider supply better →")
    h = [plt.Line2D([], [], marker="o", color=INK2, ls="none", ms=5, label="with PCM noise (primary)"),
         plt.Line2D([], [], marker="s", color=INK2, mfc="white", ls="none", ms=4.5, label="uniform drift, no PCM noise")]
    ax.legend(handles=h, fontsize=7, frameon=False, loc="lower right")
    ax.set_title("paired difference with 95 % CI (5 datasets x 3 splits x 20 device seeds)", fontsize=8.5, loc="left",
                 color=INK)
    fig.tight_layout()
    save(fig, "E4_fig2_node_difference")


def fig3(deltas, summary):
    """Crosstalk loss (xt minus is) per dataset at t0 and at 1 year per node (with PCM noise)."""
    fig, axs = plt.subplots(1, 2, figsize=(7.6, 3.2), sharey=True)
    groups = [("t0", "t0", DARK, "t0"), ("1y", "dac_remap", BLUE, "1 year, input remap"),
              ("1y", "supply", ORANGE, "1 year, divider supply")]
    for ax, did in zip(axs, ("P1", "P3")):
        xs = np.arange(len(DATASETS))
        width = 0.26
        for gi, (t, node, col, lab) in enumerate(groups):
            sel = [r for r in deltas if (r["design"], r["arm"], r["condition"], r["time"], r["contrast"]) ==
                   (did, "xt_minus_is", "all", t, node)]
            if not sel:
                continue
            vals = [float(sel[0][f"d_{ds}"]) for ds in DATASETS]
            ax.bar(xs + (gi - 1) * width, vals, width=width * 0.92, color=col, edgecolor="white", linewidth=1, label=lab)
        ax.set_xticks(xs)
        ax.set_xticklabels([d.replace("_", "\n") for d in DATASETS], fontsize=7.5)
        ax.set_title(f"{did}: accuracy lost to crosstalk (6T2M minus ideal sensing)", fontsize=8.5, loc="left", color=INK)
        ax.axhline(0, color=GREY, lw=0.8)
        ax.grid(axis="y", color=BAND, lw=0.6)
    axs[0].set_ylabel("crosstalk loss (pp)")
    h, lab = axs[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=3, fontsize=7.2, frameon=False, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    save(fig, "E4_fig3_crosstalk_by_dataset")


def main() -> int:
    run = Path(sys.argv[1])
    style()
    summary = rcsv(run / "summary.csv")
    deltas = rcsv(run / "deltas.csv")
    fig1(summary)
    fig2(deltas)
    fig3(deltas, summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
