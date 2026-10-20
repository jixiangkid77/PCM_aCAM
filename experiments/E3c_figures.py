"""Figures for E3c (PCM_aCAM): match-line transient sensing and row crosstalk.

Usage: python experiments/E3c_figures.py results/E3c_<stamp>
Reads t3_crosstalk.csv and the predictions named in run_info.json; writes figures/E3c_fig{1,2,3}_*.{png,pdf}.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FixedLocator, NullFormatter, NullLocator  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BLUE, ORANGE, GREY, INK, INK2, BAND, DARK = "#2a78d6", "#eb6834", "#8a8a86", "#1f1f1f", "#52514e", "#e6e5e1", "#52514e"
NODE_COLOR = {"t0": DARK, "dac_1y": BLUE, "supply_1y": ORANGE}
NODE_LABEL = {"t0": "no drift (t0)", "dac_1y": "1 year, input remap", "supply_1y": "1 year, supply"}
DESIGN_MARKER = {"P1": "o", "P3": "s"}
CONFIG_LABEL = {"k1d1": "1 cell, 1 LSB inside", "k1d3": "1 cell, 3 LSB", "k4d3": "4 cells, 3 LSB",
                "k4d6": "4 cells, 6 LSB", "k15d7.5": "15 cells, 7.5 LSB"}


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


def log_axis(ax, which, ticks):
    loc = FixedLocator(ticks)
    axis = ax.xaxis if which == "x" else ax.yaxis
    (ax.set_xscale if which == "x" else ax.set_yscale)("log")
    axis.set_major_locator(loc)
    (ax.set_xticklabels if which == "x" else ax.set_yticklabels)([f"{t:g}" for t in ticks])
    axis.set_minor_locator(NullLocator())
    axis.set_minor_formatter(NullFormatter())


def fig1(rows):
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    vals = [abs(float(r["sim_shift_lsb"])) for r in rows if float(r["theta"]) > 0]
    lo, hi = 0.01, max(vals) * 1.6
    xx = np.geomspace(lo, hi, 200)
    tol = np.maximum(0.10 * xx, 0.02)
    ax.fill_between(xx, np.maximum(xx - tol, lo), xx + tol, color=BAND, lw=0, zorder=0)
    ax.plot(xx, xx, color=GREY, lw=1, zorder=1)
    n_pass = sum(r["pass"] == "True" for r in rows)
    for r in rows:
        x, y = abs(float(r["sim_shift_lsb"])), abs(float(r["pred_shift_lsb"]))
        if x < lo:
            continue
        ax.plot(x, y, DESIGN_MARKER[r["design"]], ms=5, color=NODE_COLOR[r["node"]], mec="white", mew=0.6, ls="none", zorder=3)
    log_axis(ax, "x", [0.01, 0.1, 1, 10])
    log_axis(ax, "y", [0.01, 0.1, 1, 10])
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("full-row transient simulation: shift (LSB)")
    ax.set_ylabel("predicted shift (LSB)")
    ax.set_title(f"crosstalk shift of the cell under test: {n_pass}/{len(rows)}", fontsize=9, loc="left", color=INK)
    ax.text(0.03, 0.97, "band: max(10 %, 0.02 LSB)", transform=ax.transAxes, fontsize=7.5, color=INK2, va="top")
    h = [plt.Line2D([], [], marker="o", color=NODE_COLOR[n], ls="none", ms=5, label=NODE_LABEL[n]) for n in NODE_COLOR]
    h += [plt.Line2D([], [], marker=m, color=INK2, mfc="white", ls="none", ms=5, label=f"design {d}") for d, m in DESIGN_MARKER.items()]
    ax.legend(handles=h, loc="lower right", fontsize=7, frameon=False, handletextpad=0.3)
    fig.tight_layout()
    save(fig, "E3c_fig1_pred_vs_sim")


def fig2(preds, rows):
    """Shift at theta = 1 per configuration and node (simulation; predicted false mismatch marked)."""
    configs = list(CONFIG_LABEL)
    fig, axs = plt.subplots(1, 2, figsize=(7.6, 3.3), sharey=True)
    for ax, did in zip(axs, ("P1", "P3")):
        xs = np.arange(len(configs))
        width = 0.26
        for j, node in enumerate(NODE_COLOR):
            for i, cid in enumerate(configs):
                p = preds["t3"].get(f"{did}|{cid}|{node}")
                if p is None:
                    continue
                xpos = xs[i] + (j - 1) * width
                if p["false_mismatch"]:
                    ax.text(xpos, 0.03, "×", ha="center", va="bottom", fontsize=10, color=NODE_COLOR[node])
                    continue
                sel = [r for r in rows if r["design"] == did and r["config"] == cid and r["node"] == node and abs(float(r["theta"]) - 1) < 1e-9]
                if not sel:
                    continue
                v = abs(float(sel[0]["sim_shift_lsb"]))
                ax.bar(xpos, v, width=width * 0.92, color=NODE_COLOR[node], edgecolor="white", linewidth=1)
        ax.axhspan(0.12, 0.81, color=BAND, zorder=0)
        ax.set_yscale("log")
        ax.set_ylim(0.02, 30)
        ax.yaxis.set_major_locator(FixedLocator([0.03, 0.1, 0.3, 1, 3, 10]))
        ax.set_yticklabels(["0.03", "0.1", "0.3", "1", "3", "10"])
        ax.yaxis.set_minor_locator(NullLocator())
        ax.set_xticks(xs)
        ax.set_xticklabels([CONFIG_LABEL[c].replace(", ", ",\n") for c in configs], fontsize=7)
        ax.set_title(f"{did}: shift at θ = 1 (× = whole row misjudged)", fontsize=8.5, loc="left", color=INK)
    axs[0].set_ylabel("crosstalk shift (LSB)")
    h = [plt.Rectangle((0, 0), 1, 1, color=NODE_COLOR[n], label=NODE_LABEL[n]) for n in NODE_COLOR]
    h.append(plt.Rectangle((0, 0), 1, 1, color=BAND, label="drift-compensation errors of P1, P3 (E2 calibration, 0.12–0.81 LSB)"))
    fig.legend(handles=h, loc="lower center", ncol=4, fontsize=7.2, frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig, "E3c_fig2_shift_by_config")


def fig3(preds):
    fm = preds.get("false_mismatch_map", [])
    if not fm:
        return
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    for node in ("t0", "supply_1y"):
        pts = [(r["d_lsb"], r["k_max_no_false_mismatch"]) for r in fm if r["node"] == node]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "-o", color=NODE_COLOR[node], ms=4.5, lw=2,
                label={"t0": "no drift (t0)", "supply_1y": "supply, 1 year"}[node])
    ax.set_xlabel("distance of the other cells' inputs inside their bounds (LSB)")
    ax.set_ylabel("largest number of such cells\nwithout a false mismatch (of 15)")
    ax.set_ylim(0, 16)
    ax.legend(fontsize=7.5, frameon=False, loc="upper left")
    ax.set_title("row of 16 cells, P1, other bounds at θ = 0.5", fontsize=8.5, loc="left", color=INK)
    fig.tight_layout()
    save(fig, "E3c_fig3_false_mismatch_map")


def main() -> int:
    run = Path(sys.argv[1])
    style()
    info = json.loads((run / "run_info.json").read_text())
    preds = json.loads((ROOT / info["predictions_file"]).read_text())
    rows = rcsv(run / "t3_crosstalk.csv")
    fig1(rows)
    fig2(preds, rows)
    fig3(preds)
    return 0


if __name__ == "__main__":
    sys.exit(main())
