"""Figures for E3a (PCM_aCAM): PCM read nonlinearity at the two compensation nodes.

Usage: python experiments/E3a_figures.py results/E3a_<stamp>
Reads compare.csv, analysis_systematic.csv and the frozen predictions; writes figures/E3a_fig{1,2,3}_*.{png,pdf}.
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
BLUE, ORANGE, GREEN, GREY, INK, INK2, BAND = "#2a78d6", "#eb6834", "#1baf7a", "#8a8a86", "#1f1f1f", "#52514e", "#e6e5e1"
V0_GRID = [2.0, 1.0, 0.6, 0.4, 0.3, 0.2, 0.15]
X_LINEAR = 3.4            # where the linear PCM is drawn on the V0 axis
TITLES = {"P1": "P1  W/L 0.42/0.15 µm, D 0.2 V", "S1": "S1  W/L 0.42/0.15 µm, D 0.4 V",
          "G1": "G1  W/L 1/0.5 µm, D 0.2 V", "P3": "P3  W/L 0.42/4 µm, D 0.2 V"}


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


def lab(v0, beta):
    return "linear" if v0 is None else f"V0={v0:g}|beta={beta:+g}"


def fig1(rows):
    get = {(r["design"], r["time"], r["grid"], r["node"]): r for r in rows}
    fig, axs = plt.subplots(2, 2, figsize=(7.6, 6.2), sharex=True)
    series = [("supply_lin", 0.0, GREY, "o", "supply, linear rule (β = 0)"),
              ("supply_rep", 0.0, GREEN, "o", "supply, replica rule, β = 0"),
              ("supply_rep", 0.5, ORANGE, "^", "supply, replica rule, β = +0.5"),
              ("supply_rep", -0.5, ORANGE, "v", "supply, replica rule, β = −0.5")]
    for ax, did in zip(axs.flat, ["P1", "S1", "G1", "P3"]):
        ax.axvspan(0.14, 0.41, color=BAND, lw=0, zorder=0)
        dac = float(get[(did, "1y", "linear", "dac_remap")]["sim_rms_lsb"])
        ax.axhline(dac, color=BLUE, lw=1.8, zorder=2)
        sup_lin = float(get[(did, "1y", "linear", "supply_rep")]["sim_rms_lsb"])
        ax.annotate("input remap", (X_LINEAR, dac), xytext=(2, 4 if dac > sup_lin else -11), textcoords="offset points",
                    ha="left", fontsize=7.5, color=BLUE)
        for node, beta, col, mk, _ in series:
            xs = [X_LINEAR] + V0_GRID
            labs = [lab(None, 0)] + [lab(v, beta) for v in V0_GRID]
            pr = [float(get[(did, "1y", g, node)]["pred_rms_lsb"]) for g in labs]
            sm = [float(get[(did, "1y", g, node)]["sim_rms_lsb"]) for g in labs]
            ls = "--" if node == "supply_lin" else "-"
            ax.plot(xs, pr, ls, color=col, lw=1.3, zorder=3, alpha=0.9)
            ax.plot(xs, sm, mk, color=col, ms=4.2, mec="white", mew=0.5, ls="none", zorder=4)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ticks = [X_LINEAR, 2, 1, 0.6, 0.4, 0.3, 0.2, 0.15]
        ax.xaxis.set_major_locator(FixedLocator(ticks))
        ax.set_xticklabels(["lin."] + [f"{t:g}" for t in ticks[1:]], fontsize=7.5)
        ax.xaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_minor_formatter(NullFormatter())
        yt = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]
        ax.yaxis.set_major_locator(FixedLocator(yt))
        ax.set_yticklabels([f"{t:g}" for t in yt], fontsize=7.5)
        ax.yaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_ylim(0.008, 150)
        ax.set_title(TITLES[did], fontsize=8.5, loc="left", color=INK)
    axs[0, 0].set_xlim(X_LINEAR * 1.25, 0.13)          # shared x axis: linear PCM on the left, stronger nonlinearity to the right
    for ax in axs[1]:
        ax.set_xlabel("PCM nonlinearity V0_ref (V), stronger to the right")
    for ax in axs[:, 0]:
        ax.set_ylabel("rms threshold error, 1 year (LSB)")
    axs[0, 0].text(0.27, 0.011, "RESET-type\nliterature\nestimate", fontsize=6.8, color=INK2, ha="center", va="bottom")
    h = [plt.Line2D([], [], color=BLUE, lw=1.8, label="input remap (same for every V0)")]
    h += [plt.Line2D([], [], color=c, marker=m, ls="--" if n == "supply_lin" else "-", ms=4.2, label=t) for n, _, c, m, t in series]
    h += [plt.Line2D([], [], color=INK2, ls="-", label="line = frozen prediction"),
          plt.Line2D([], [], color=INK2, marker="o", ls="none", ms=4.2, label="marker = Monte Carlo")]
    fig.legend(handles=h, loc="lower center", ncol=3, fontsize=7, frameon=False, bbox_to_anchor=(0.5, -0.05))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, "E3a_fig1_error_vs_nonlinearity")


def fig2(rows):
    fig, ax = plt.subplots(figsize=(4.2, 3.9))
    lo, hi = 0.008, 150
    xx = np.geomspace(lo, hi, 300)
    tol = np.maximum(0.15 * xx, 0.02)
    ax.fill_between(xx, np.maximum(xx - tol, lo), xx + tol, color=BAND, lw=0, zorder=0)
    ax.plot(xx, xx, color=GREY, lw=1, zorder=1)
    col = {"dac_remap": BLUE, "supply_lin": GREY, "supply_rep": ORANGE}
    n = 0
    for r in rows:
        if r["time"] != "1y" or float(r["invalid_frac"]) > 0.01 or r["pred_rms_lsb"] in ("nan", ""):
            continue
        if r["node"] == "dac_remap" and r["grid"] != "linear":
            continue
        ax.plot(float(r["sim_rms_lsb"]), float(r["pred_rms_lsb"]), "o", ms=3.6, color=col[r["node"]], mec="white", mew=0.4,
                ls="none", zorder=3)
        n += r["node"] != "dac_remap"
    for a, which in ((ax.xaxis, "x"), (ax.yaxis, "y")):
        pass
    ax.set_xscale("log")
    ax.set_yscale("log")
    yt = [0.01, 0.1, 1, 10, 100]
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(FixedLocator(yt))
        axis.set_minor_locator(NullLocator())
        axis.set_minor_formatter(NullFormatter())
    ax.set_xticklabels([f"{t:g}" for t in yt])
    ax.set_yticklabels([f"{t:g}" for t in yt])
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Monte Carlo rms error (LSB)")
    ax.set_ylabel("predicted rms error (LSB)")
    ax.set_title(f"1 year: {n}/{n} supply predictions within tolerance", fontsize=8.5, loc="left", color=INK)
    h = [plt.Line2D([], [], marker="o", color=col[k], ls="none", ms=4, label=t) for k, t in
         (("supply_rep", "supply, replica rule"), ("supply_lin", "supply, linear rule"), ("dac_remap", "input remap"))]
    ax.legend(handles=h, loc="lower right", fontsize=7, frameon=False)
    fig.tight_layout()
    save(fig, "E3a_fig2_pred_vs_mc")


def fig3(syst):
    fig, ax = plt.subplots(figsize=(4.6, 3.1))
    sel = {(r["design"], r["grid"], r["node"]): [] for r in syst}
    for r in syst:
        sel[(r["design"], r["grid"], r["node"])].append((float(r["theta"]), float(r["systematic_error_lsb"])))
    for grid, col, mk, txt in (("V0=0.3|beta=+0.5", ORANGE, "^", "β = +0.5"), ("V0=0.3|beta=-0.5", ORANGE, "v", "β = −0.5"),
                               ("V0=0.3|beta=+0", GREEN, "o", "β = 0")):
        pts = sorted(sel[("P1", grid, "supply_rep")])
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "-" + mk, color=col, ms=4, lw=1.3, mec="white", mew=0.5, label=txt)
    ax.axvline(0.5, color=GREY, lw=0.8, ls=":")
    ax.text(0.51, ax.get_ylim()[1] * 0.85 if ax.get_ylim()[1] > 0 else 0.5, "replica level", fontsize=7, color=INK2)
    ax.axhline(0, color=GREY, lw=0.6)
    ax.set_xlabel("threshold θ")
    ax.set_ylabel("systematic error, 1 year (LSB)")
    ax.set_title("P1, replica-rule supply, V0_ref = 0.3 V", fontsize=8.5, loc="left", color=INK)
    ax.legend(fontsize=7, frameon=False, loc="lower left")
    fig.tight_layout()
    save(fig, "E3a_fig3_systematic_vs_theta")


def main() -> int:
    run = Path(sys.argv[1])
    style()
    rows = rcsv(run / "compare.csv")
    fig1(rows)
    fig2(rows)
    fig3(rcsv(run / "analysis_systematic.csv"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
