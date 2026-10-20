"""Figures for E1 (PCM_aCAM).

Usage: python experiments/E1_figures.py results/E1_<stamp>
Reads summary.csv, regime.csv, gmid_profile.csv, law_points.npz and law_points_current_mismatch.npz of the
run (the last two from the exploratory analyses) and writes figures/E1_fig{1,2,3}_*.{png,pdf}.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import NullFormatter, NullLocator  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BLUE, ORANGE, GREEN, GREY, INK, INK2 = "#2a78d6", "#eb6834", "#1baf7a", "#8a8a86", "#1f1f1f", "#52514e"
RAMP = {"P1": "#86b6ef", "P2": "#3987e5", "P3": "#1c5cab", "P4": "#0d366b"}
LBL = {"P1": "L = 0.15 µm", "P2": "L = 1 µm", "P3": "L = 4 µm", "P4": "L = 8 µm"}
PRIMARY = ["P1", "P2", "P3", "P4"]
L_UM = {"P1": 0.15, "P2": 1.0, "P3": 4.0, "P4": 8.0}


def rcsv(p: Path):
    with open(p, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def rms(rows, d, s, node, t="1y"):
    for r in rows:
        if (r["design"], r["setting"], r["node"], r["time"]) == (d, s, node, t):
            return float(r["rms"])
    raise KeyError((d, s, node, t))


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


def fig1(run: Path):
    reg = rcsv(run / "regime.csv")
    gm = rcsv(run / "gmid_profile.csv")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.6, 3.1))
    marker = {"sub": ("o", "none"), "sat": ("o", "full"), "triode": ("s", "full")}
    for d in PRIMARY:
        rr = [r for r in reg if r["design"] == d]
        th = np.array([float(r["theta"]) for r in rr])
        vb = np.array([float(r["vgs_t0_V"]) for r in rr])
        a1.plot(th, vb, "-", color=RAMP[d], lw=2)
        for region, (m, fill) in marker.items():
            sel = np.array([r["region_t0_and_supply"] == region for r in rr])
            if sel.any():
                a1.plot(th[sel], vb[sel], m, ms=4.5, color=RAMP[d], mfc="white" if fill == "none" else RAMP[d],
                        mew=1.2, ls="none")
        a1.annotate(LBL[d], (1.0, vb[-1]), xytext=(4, 0), textcoords="offset points", va="center",
                    fontsize=7.5, color=INK)
        g = [r for r in gm if r["design"] == d]
        a2.plot([float(r["theta"]) for r in g], [float(r["gmid_at_bound_per_V"]) for r in g], "-", color=RAMP[d], lw=2)
        a2.annotate(LBL[d], (1.0, float(g[-1]["gmid_at_bound_per_V"])), xytext=(4, 0), textcoords="offset points",
                    va="center", fontsize=7.5, color=INK)
    a1.set_xlim(-0.02, 1.32)
    a1.set_xlabel("normalized threshold θ")
    a1.set_ylabel("bound voltage V_DL (V)")
    a1.set_title("(a) bound at t₀ (D = 0.2 V)", fontsize=9, loc="left", color=INK)
    h = [plt.Line2D([], [], marker="o", mfc="white", color=INK2, ls="none", label="subthreshold"),
         plt.Line2D([], [], marker="o", color=INK2, ls="none", label="saturation"),
         plt.Line2D([], [], marker="s", color=INK2, ls="none", label="triode")]
    a1.legend(handles=h, frameon=False, fontsize=7.5, loc="upper left", title="divider at switching",
              title_fontsize=7.5)
    a2.set_xlim(-0.02, 1.32)
    a2.set_yscale("log")
    a2.set_xlabel("normalized threshold θ")
    a2.set_ylabel("gm/Id of the divider at the bound (1/V)")
    a2.set_title("(b) transconductance efficiency", fontsize=9, loc="left", color=INK)
    fig.tight_layout()
    save(fig, "E1_fig1_transfer_region")


def fig2(run: Path):
    rows = rcsv(run / "summary.csv")
    x = [L_UM[d] for d in PRIMARY]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.6, 3.1))
    series = [("dac_remap", ORANGE, "input remap (DAC)"), ("supply", GREEN, "divider supply"),
              ("dac_gain", GREY, "linear DAC gain (E0.2 style)")]
    for node, col, lab in series:
        y = [rms(rows, d, "pdk", node) for d in PRIMARY]
        ls = "--" if node == "dac_gain" else "-"
        a1.plot(x, y, ls, marker="o", ms=5, color=col, lw=2, label=lab)
    a1.set_xscale("log")
    a1.xaxis.set_minor_locator(NullLocator())
    a1.xaxis.set_minor_formatter(NullFormatter())
    a1.set_xticks(x, ["0.15", "1", "4", "8"])
    a1.set_xlabel("divider channel length L (µm)")
    a1.set_ylabel("rms bound error after 1 year (LSB)")
    a1.set_ylim(-0.12, 2.3)
    a1.set_yticks([0, 0.5, 1.0, 1.5, 2.0])
    a1.axhline(0, color="#cfcfcb", lw=0.6, zorder=0)
    a1.legend(frameon=False, fontsize=7.5, loc="upper right")
    a1.set_title("(a) PDK mismatch, both devices", fontsize=9, loc="left", color=INK)
    for d, xv in zip(PRIMARY, x):
        vals = {node: rms(rows, d, "pdk", node) for node in ("dac_remap", "supply")}
        hi = max(vals, key=vals.get)
        for node, v in vals.items():
            a1.annotate(f"{v:.2f}", (xv, v), xytext=(0, 5 if node == hi else -11), textcoords="offset points",
                        ha="center", fontsize=7, color=INK)
    attr = [("dac_remap", "div_only", ORANGE, "-", "input remap ← divider mismatch"),
            ("dac_remap", "pd_only", ORANGE, ":", "input remap ← pull-down mismatch"),
            ("supply", "pd_only", GREEN, "-", "supply ← pull-down mismatch"),
            ("supply", "div_only", GREEN, ":", "supply ← divider mismatch (= 0)")]
    for node, s, col, ls, lab in attr:
        a2.plot(x, [rms(rows, d, s, node) for d in PRIMARY], ls, marker="o", ms=4.5, color=col, lw=2, label=lab)
    a2.set_xscale("log")
    a2.xaxis.set_minor_locator(NullLocator())
    a2.xaxis.set_minor_formatter(NullFormatter())
    a2.set_xticks(x, ["0.15", "1", "4", "8"])
    a2.set_xlabel("divider channel length L (µm)")
    a2.set_ylabel("rms bound error after 1 year (LSB)")
    a2.set_ylim(-0.03, 0.9)
    a2.legend(frameon=False, fontsize=7, loc="upper right")
    a2.set_title("(b) which mismatch each node exposes", fontsize=9, loc="left", color=INK)
    fig.tight_layout()
    save(fig, "E1_fig2_nodes")


def fig3(run: Path):
    lp = np.load(run / "law_points.npz")
    lc = np.load(run / "law_points_current_mismatch.npz")
    laws = {r["design"] + r["law"]: r for r in rcsv(run / "law_check.csv")}
    lawc = {r["design"]: r for r in rcsv(run / "law_check_current_mismatch.csv")}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.6, 3.3))
    for ax, title, getter, r2s in (
            (a1, "(a) supply node, pull-down mismatch",
             lambda d: (lp[f"{d}|supply|pd_only|pred"], lp[f"{d}|supply|pd_only|sim"]),
             [float(laws[d + "supply|pd_only"]["r2_identity"]) for d in PRIMARY]),
            (a2, "(b) input remap, divider mismatch",
             lambda d: (lc[f"{d}|pred"], lc[f"{d}|sim"]),
             [float(lawc[d]["r2_identity"]) for d in PRIMARY])):
        lim = 0.0
        for d in PRIMARY:
            p, s = getter(d)
            ax.plot(p, s, ".", ms=1.6, alpha=0.35, color=RAMP[d], rasterized=True)
            lim = max(lim, float(np.max(np.abs(np.concatenate([p, s])))))
        lim = min(lim * 1.05, 4.0)
        ax.plot([-lim, lim], [-lim, lim], "-", color=GREY, lw=0.8)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.set_xlabel("first-order prediction (LSB)")
        ax.set_ylabel("transistor-level result (LSB)")
        ax.set_title(title, fontsize=9, loc="left", color=INK)
        txt = "R² " + ", ".join(f"{r:.3f}" for r in r2s) + "\n(L = 0.15, 1, 4, 8 µm)"
        ax.text(0.03, 0.97, txt, transform=ax.transAxes, va="top", fontsize=7, color=INK)
    h = [plt.Line2D([], [], marker="o", color=RAMP[d], ls="none", label=LBL[d]) for d in PRIMARY]
    a2.legend(handles=h, frameon=False, fontsize=7, loc="lower right")
    fig.tight_layout()
    save(fig, "E1_fig3_first_order_laws")


def main() -> int:
    run = Path(sys.argv[1])
    style()
    fig1(run)
    fig2(run)
    fig3(run)
    print("figures written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
