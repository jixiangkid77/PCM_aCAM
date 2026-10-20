"""Figures for E3b (PCM_aCAM): upper-bound branch with the inverter.

Usage: python experiments/E3b_figures.py results/E3b_<stamp>
Reads compare.csv, analysis_sigma_breakdown.csv and analysis_branch_comparison.csv (from E3b_analysis.py);
writes figures/E3b_fig{1,2,3}_*.{png,pdf}.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FixedLocator, NullFormatter, NullLocator  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BLUE, ORANGE, GREEN, GREY, INK, INK2, BAND = "#2a78d6", "#eb6834", "#1baf7a", "#8a8a86", "#1f1f1f", "#52514e", "#e6e5e1"
RAMP = ["#86b6ef", "#3987e5", "#1c5cab"]
NODE_COLOR = {"dac_remap": BLUE, "supply": ORANGE}
NODE_LABEL = {"dac_remap": "input remap (DAC)", "supply": "divider supply"}
INV_MARKER = {"INV_S": "o", "INV_M": "s"}
INV_OF = {"U1": "INV_S", "U2": "INV_S", "U3": "INV_S", "U4": "INV_M", "U5": "INV_M"}


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
    lab = [f"{t:g}" for t in ticks]
    axis = ax.xaxis if which == "x" else ax.yaxis
    (ax.set_xscale if which == "x" else ax.set_yscale)("log")
    axis.set_major_locator(loc)
    (ax.set_xticklabels if which == "x" else ax.set_yticklabels)(lab)
    axis.set_minor_locator(NullLocator())
    axis.set_minor_formatter(NullFormatter())


def fig1(cmp_rows):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.6, 3.5))
    lo, hi = 0.06, 1.4
    xx = np.geomspace(lo, hi, 200)
    a1.fill_between(xx, xx * 0.85, xx * 1.15, color=BAND, lw=0, zorder=0)
    a1.plot(xx, xx, color=GREY, lw=1, zorder=1)
    n_pass = n_all = 0
    sup_pts = {}
    for r in cmp_rows:
        if r["time"] != "1y" or r["setting"] != "pdk" or r["node"] not in NODE_COLOR:
            continue
        x, y = float(r["sim_rms_lsb"]), float(r["pred_rms_lsb"])
        a1.plot(x, y, INV_MARKER[INV_OF[r["condition"]]], ms=6, color=NODE_COLOR[r["node"]], mec="white", mew=0.8,
                ls="none", zorder=3)
        if r["node"] == "dac_remap":
            off = (5, 3) if r["condition"] == "U4" else (5, -5)
            a1.annotate(r["condition"], (x, y), xytext=off, textcoords="offset points", fontsize=7, color=INK2)
        else:
            sup_pts.setdefault(INV_OF[r["condition"]], []).append((x, y, r["condition"]))
        n_all += 1
        n_pass += r["pass"] == "True"
    for inv, pts in sup_pts.items():   # supply predictions coincide for the same inverter: one label per group
        xs, ys_ = [p[0] for p in pts], [p[1] for p in pts]
        ids = sorted(p[2] for p in pts)
        lab = f"{ids[0]}–{ids[-1]}" if len(ids) > 2 else ", ".join(ids)
        a1.annotate(lab, (max(xs), ys_[0]), xytext=(6, -3), textcoords="offset points", fontsize=7, color=INK2)
    log_axis(a1, "x", [0.1, 0.3, 1])
    log_axis(a1, "y", [0.1, 0.3, 1])
    a1.set_xlim(lo, hi)
    a1.set_ylim(lo, hi)
    a1.set_xlabel("Monte Carlo rms error (LSB)")
    a1.set_ylabel("predicted rms error (LSB)")
    a1.set_title(f"(a) upper branch, 1 year: {n_pass}/{n_all} within tolerance", fontsize=9, loc="left", color=INK)
    a1.text(0.03, 0.97, "grey band: ±15 % (tolerance)", transform=a1.transAxes, fontsize=7.5, color=INK2, va="top")
    hn = [plt.Line2D([], [], marker="o", color=NODE_COLOR[n], ls="none", ms=6, label=NODE_LABEL[n]) for n in NODE_COLOR]
    hs = [plt.Line2D([], [], marker=m, color=INK2, mfc="white", ls="none", ms=6,
                     label={"INV_S": "skewed inverter (INV_S)", "INV_M": "minimum inverter (INV_M)"}[k])
          for k, m in INV_MARKER.items()]
    a1.legend(handles=hn + hs, loc="lower right", fontsize=7, frameon=False, handletextpad=0.3)
    lo2, hi2 = 0.0005, 1.4
    xx = np.geomspace(lo2, hi2, 300)
    tol = np.maximum(0.15 * xx, 0.02)
    a2.fill_between(xx, np.maximum(xx - tol, lo2), xx + tol, color=BAND, lw=0, zorder=0)
    a2.plot(xx, xx, color=GREY, lw=1, zorder=1)
    sec = {("div_only", "dac_remap"): (BLUE, "o", "divider mismatch only, input remap"),
           ("y_only", "dac_remap"): (BLUE, "D", "inverter + pull-down mismatch only, input remap"),
           ("y_only", "supply"): (ORANGE, "o", "inverter + pull-down mismatch only, supply")}
    n_pass = n_all = 0
    for r in cmp_rows:
        k = (r["setting"], r["node"])
        if r["time"] != "1y" or k not in sec:
            continue
        col, m, _ = sec[k]
        a2.plot(float(r["sim_rms_lsb"]), float(r["pred_rms_lsb"]), m, ms=5, color=col,
                mfc="white" if m == "D" else col, mew=1.0, ls="none", zorder=3)
        n_all += 1
        n_pass += r["pass"] == "True"
    log_axis(a2, "x", [0.001, 0.01, 0.1, 1])
    log_axis(a2, "y", [0.001, 0.01, 0.1, 1])
    a2.set_xlim(lo2, hi2)
    a2.set_ylim(lo2, hi2)
    a2.set_xlabel("Monte Carlo rms error (LSB)")
    a2.set_title(f"(b) mismatch decomposition, 1 year: {n_pass}/{n_all}", fontsize=9, loc="left", color=INK)
    a2.text(0.03, 0.97, "band: max(15 %, 0.02 LSB)", transform=a2.transAxes, fontsize=7.5, color=INK2, va="top")
    h2 = [plt.Line2D([], [], marker=m, color=col, mfc="white" if m == "D" else col, ls="none", ms=5, label=lab)
          for col, m, lab in sec.values()]
    a2.legend(handles=h2, loc="lower right", fontsize=6.8, frameon=False, handletextpad=0.3)
    fig.tight_layout()
    save(fig, "E3b_fig1_pred_vs_mc")


def fig2(bc_rows):
    """Upper versus lower branch for the same divider and D (Monte Carlo, 1 year)."""
    groups = [((0.42, 0.15), "divider 0.42/0.15 µm"), ((1.0, 0.5), "divider 1/0.5 µm"), ((0.42, 4.0), "divider 0.42/4 µm")]
    rows = []
    for key, title in groups:
        sel = [r for r in bc_rows if (float(r["divider_w_um"]), float(r["divider_l_um"])) == key]
        if not sel:
            continue
        lo = sel[0]
        rows.append((title, "lower branch (pull-down only)", float(lo["lower_dac_remap"]), float(lo["lower_supply"]), True))
        for r in sorted(sel, key=lambda r: r["inverter"], reverse=True):
            lab = {"INV_S": "upper branch, skewed inverter", "INV_M": "upper branch, minimum inverter"}[r["inverter"]]
            rows.append((title, f"{lab} ({r['upper_condition']})", float(r["upper_dac_remap"]), float(r["upper_supply"]), False))
    fig, ax = plt.subplots(figsize=(7.6, 3.9))
    ys, labels = [], []
    ordered = rows
    ypos = []
    yy = 0.0
    prev = None
    for title, lab, dac, sup, first in ordered:
        if prev is not None and title != prev:
            yy += 0.7
        ypos.append(yy)
        prev = title
        yy += 1.0
    ymax = ypos[-1]
    for (title, lab, dac, sup, first), y0 in zip(ordered, ypos):
        yv = ymax - y0
        ax.plot([dac, sup], [yv, yv], "-", color=BAND, lw=3, zorder=1)
        ax.plot(dac, yv, "o", color=BLUE, ms=6.5, zorder=3)
        ax.plot(sup, yv, "o", color=ORANGE, ms=6.5, zorder=3)
        better = "input remap" if dac < sup else "supply"
        ax.text(1.0, yv, better, va="center", fontsize=7.5, color=BLUE if dac < sup else ORANGE)
        ys.append(yv)
        labels.append(lab)
        if first:
            ax.text(-0.02, yv + 0.55, title, fontsize=8, color=INK, fontweight="bold", transform=ax.get_yaxis_transform(),
                    ha="right")
    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.set_xlim(0, 0.98)
    ax.set_ylim(-0.6, ymax + 1.0)
    ax.set_xlabel("rms threshold error after 1 year (LSB), Monte Carlo")
    ax.grid(axis="x", color=BAND, lw=0.8)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.text(1.0, ymax + 0.7, "better", fontsize=7.5, color=INK2)
    h = [plt.Line2D([], [], marker="o", color=BLUE, ls="none", ms=6.5, label="input remap"),
         plt.Line2D([], [], marker="o", color=ORANGE, ls="none", ms=6.5, label="supply")]
    ax.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize=7.5, frameon=False)
    fig.tight_layout()
    save(fig, "E3b_fig2_upper_vs_lower")


def fig3(sig_rows):
    """Where sigma_delta_Y comes from: root of the variance contributed by each device group."""
    fig, ax = plt.subplots(figsize=(7.6, 2.6))
    cats = [("inv_nmos", "inverter nMOS"), ("inv_pmos", "inverter pMOS"), ("pulldown", "pull-down (÷ gain)")]
    rows = [(r["inverter"], r) for r in sig_rows]
    order = {"INV_S": 0, "INV_M": 1}
    rows.sort(key=lambda t: order[t[0]])
    labels = []
    for i, (inv, r) in enumerate(rows):
        left = 0.0
        for j, (k, lab) in enumerate(cats):
            share = float(r[f"share_{k}"])
            ax.barh(i, share, left=left, color=RAMP[j], height=0.55, edgecolor="white", linewidth=2)
            if share > 0.08:
                ax.text(left + share / 2, i, f"{lab}\n{100 * share:.0f} %", ha="center", va="center", fontsize=7,
                        color="white" if j > 0 else INK)
            left += share
        sp, sm = float(r["sigma_pred_mV"]), float(r["sigma_mc_mV"])
        gain = float(r["inverter_gain"])
        ax.text(1.02, i, f"σ = {sp:.1f} mV predicted, {sm:.1f} mV Monte Carlo; gain {gain:.1f}", va="center",
                fontsize=7.5, color=INK2, transform=ax.get_yaxis_transform())
        labels.append({"INV_S": "skewed inverter\n(nMOS 2.0/0.15, pMOS 0.42/4)",
                       "INV_M": "minimum inverter\n(0.42/0.15 both)"}[inv])
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of the variance of the trip point V_Y*")
    ax.invert_yaxis()
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_title("pull-down share (its mismatch divided by the inverter gain): < 1 % in both\n"
                 "lower branch for comparison: σ = 22.6 mV, all from the pull-down", fontsize=7.5, loc="left", color=INK2)
    fig.tight_layout()
    fig.subplots_adjust(right=0.62)
    save(fig, "E3b_fig3_sigma_breakdown")


def main() -> int:
    run = Path(sys.argv[1])
    style()
    fig1(rcsv(run / "compare.csv"))
    fig2(rcsv(run / "analysis_branch_comparison.csv"))
    fig3(rcsv(run / "analysis_sigma_breakdown.csv"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
