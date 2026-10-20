"""Figures for E2 (PCM_aCAM): frozen predictions versus Monte Carlo.

Usage: python experiments/E2_figures.py results/E2_<stamp>
Reads compare.csv, profile.csv and the predictions file named in run_info.json; writes
figures/E2_fig{1,2,3}_*.{png,pdf}.
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
NODE_COLOR = {"dac_remap": BLUE, "supply": ORANGE}
NODE_LABEL = {"dac_remap": "input remap (DAC)", "supply": "divider supply"}
SET_MARKER = {"geometry": "o", "corner": "s", "temperature": "^"}
ORDER = ["G1", "G2", "G3", "G4", "G5", "G6", "C1", "C2", "C3", "C4", "T1", "T2", "T3", "T4"]


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


def describe(c: dict) -> str:
    pd = "4×PD" if c["pulldown"] == "pd_4x" else ""
    w, l = c["w_div_um"], c["l_div_um"]
    s = f"{w:g}/{l:g} µm, D {c['d_v']:g} V"
    extra = []
    if c["corner"] != "tt":
        extra.append(c["corner"])
    if c["temp_c"] != 27:
        extra.append(f"{c['temp_c']:g} °C")
    if pd:
        extra.append(pd)
    return s + ("; " + ", ".join(extra) if extra else "")


def log_axis(ax, which, ticks):
    loc = FixedLocator(ticks)
    lab = [f"{t:g}" for t in ticks]
    if which == "x":
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(loc)
        ax.set_xticklabels(lab)
        ax.xaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_minor_formatter(NullFormatter())
    else:
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(loc)
        ax.set_yticklabels(lab)
        ax.yaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_minor_formatter(NullFormatter())


def fig1(cmp_rows, conds):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.6, 3.5))
    # (a) primary: pdk, 1 year
    lo, hi = 0.025, 1.4
    xx = np.geomspace(lo, hi, 200)
    a1.fill_between(xx, xx * 0.85, xx * 1.15, color=BAND, lw=0, zorder=0)
    a1.plot(xx, xx, color=GREY, lw=1, zorder=1)
    worst = None
    for r in cmp_rows:
        if r["time"] != "1y" or r["setting"] != "pdk" or r["node"] not in NODE_COLOR:
            continue
        x, y = float(r["sim_rms_lsb"]), float(r["pred_rms_lsb"])
        a1.plot(x, y, SET_MARKER[conds[r["condition"]]["set"]], ms=5.5, color=NODE_COLOR[r["node"]],
                mec="white", mew=0.8, ls="none", zorder=3)
        rel = abs(float(r["rel_diff"]))
        if worst is None or rel > worst[0]:
            worst = (rel, x, y, r["condition"], float(r["rel_diff"]))
    a1.annotate(f"{worst[3]}: {100 * worst[4]:+.1f} %", (worst[1], worst[2]), xytext=(6, -14),
                textcoords="offset points", fontsize=7.5, color=INK,
                arrowprops=dict(arrowstyle="-", color=GREY, lw=0.8))
    a1.text(0.03, 0.97, "grey band: ±15 % (tolerance)", transform=a1.transAxes, fontsize=7.5, color=INK2, va="top")
    log_axis(a1, "x", [0.03, 0.1, 0.3, 1])
    log_axis(a1, "y", [0.03, 0.1, 0.3, 1])
    a1.set_xlim(lo, hi)
    a1.set_ylim(lo, hi)
    a1.set_xlabel("Monte Carlo rms error (LSB)")
    a1.set_ylabel("predicted rms error (LSB)")
    a1.set_title("(a) PDK mismatch, 1 year: 28/28 within tolerance", fontsize=9, loc="left", color=INK)
    hn = [plt.Line2D([], [], marker="o", color=NODE_COLOR[n], ls="none", ms=5.5, label=NODE_LABEL[n]) for n in NODE_COLOR]
    hs = [plt.Line2D([], [], marker=m, color=INK2, mfc="white", ls="none", ms=5.5, label=f"{s} set")
          for s, m in SET_MARKER.items()]
    a1.legend(handles=hn + hs, loc="lower right", fontsize=7, frameon=False, handletextpad=0.3)
    # (b) secondary: decomposition settings, 1 year
    lo2, hi2 = 0.0008, 1.4
    xx = np.geomspace(lo2, hi2, 300)
    tol = np.maximum(0.15 * xx, 0.02)
    a2.fill_between(xx, np.maximum(xx - tol, lo2), xx + tol, color=BAND, lw=0, zorder=0)
    a2.plot(xx, xx, color=GREY, lw=1, zorder=1)
    sec = {("div_only", "dac_remap"): (BLUE, "o", "divider mismatch only, input remap"),
           ("pd_only", "dac_remap"): (BLUE, "D", "pull-down mismatch only, input remap"),
           ("pd_only", "supply"): (ORANGE, "o", "pull-down mismatch only, supply")}
    n_pass = n_all = 0
    for r in cmp_rows:
        k = (r["setting"], r["node"])
        if r["time"] != "1y" or k not in sec:
            continue
        col, m, _ = sec[k]
        a2.plot(float(r["sim_rms_lsb"]), float(r["pred_rms_lsb"]), m, ms=4.8, color=col,
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
    h2 = [plt.Line2D([], [], marker=m, color=col, mfc="white" if m == "D" else col, ls="none", ms=4.8, label=lab)
          for col, m, lab in sec.values()]
    a2.legend(handles=h2, loc="lower right", fontsize=7, frameon=False, handletextpad=0.3)
    fig.tight_layout()
    save(fig, "E2_fig1_pred_vs_mc")


def fig2(cmp_rows, conds):
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ys = {}
    y = 0
    for cid in ORDER[::-1]:
        ys[cid] = y
        y += 1
        if cid in ("T1", "C1"):
            y += 0.6
    get = {(r["condition"], r["node"], r["time"], r["setting"]): r for r in cmp_rows}
    for cid in ORDER:
        yy = ys[cid]
        for node, dy in (("dac_remap", 0.13), ("supply", -0.13)):
            r = get[(cid, node, "1y", "pdk")]
            p, s = float(r["pred_rms_lsb"]), float(r["sim_rms_lsb"])
            col = NODE_COLOR[node]
            ax.plot([p, s], [yy + dy, yy + dy], "-", color=col, lw=1.2, alpha=0.6)
            ax.plot(p, yy + dy, "o", color=col, mfc="white", ms=7.5, mew=1.2)
            ax.plot(s, yy + dy, "o", color=col, ms=4.2, mec=col, mew=0)
        rd, rs = get[(cid, "dac_remap", "1y", "pdk")], get[(cid, "supply", "1y", "pdk")]
        pdv, psv = float(rd["pred_rms_lsb"]), float(rs["pred_rms_lsb"])
        clear = abs(pdv - psv) >= 0.2 * max(pdv, psv)
        better = "input remap" if pdv < psv else "supply"
        ax.text(1.27, yy, f"{better}{'' if clear else ' (close)'}", va="center", fontsize=7.5,
                color=NODE_COLOR["dac_remap" if pdv < psv else "supply"])
    ax.set_yticks([ys[c] for c in ORDER])
    ax.set_yticklabels([f"{c}  {describe(conds[c])}" for c in ORDER], fontsize=7.5)
    ax.set_xlim(0, 1.25)
    ax.set_xlabel("rms threshold error after 1 year (LSB)")
    ax.text(1.27, ys["G1"] + 0.9, "predicted better\n(= Monte Carlo in 14/14)", fontsize=7.5, color=INK2, va="bottom")
    h = [plt.Line2D([], [], marker="o", color=BLUE, mfc="white", ls="none", ms=7.5, label="input remap, predicted"),
         plt.Line2D([], [], marker="o", color=BLUE, ls="none", ms=4.2, label="input remap, Monte Carlo"),
         plt.Line2D([], [], marker="o", color=ORANGE, mfc="white", ls="none", ms=7.5, label="supply, predicted"),
         plt.Line2D([], [], marker="o", color=ORANGE, ls="none", ms=4.2, label="supply, Monte Carlo")]
    ax.legend(handles=h, loc="upper center", fontsize=7.5, frameon=False, ncol=4, bbox_to_anchor=(0.5, -0.12),
              handletextpad=0.3, columnspacing=1.2)
    ax.grid(axis="x", color=BAND, lw=0.8)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    save(fig, "E2_fig2_ranking")


def fig3(preds, prof_rows, conds, picks=("C2", "G5", "G6")):
    fig, axs = plt.subplots(1, len(picks), figsize=(7.6, 2.8), sharey=False)
    for ax, cid in zip(axs, picks):
        pt = preds[cid]["predictions"]["1y"]["per_theta_ms_x"]
        theta = np.arange(16) / 15.0
        for node in ("dac_remap", "supply"):
            col = NODE_COLOR[node]
            pr = 15.0 * np.sqrt(np.array(pt[f"pdk|{node}"]))
            sim = [float(r["sim_rms"]) for r in prof_rows
                   if (r["condition"], r["setting"], r["node"], r["time"]) == (cid, "pdk", node, "1y")]
            th = [float(r["theta"]) for r in prof_rows
                  if (r["condition"], r["setting"], r["node"], r["time"]) == (cid, "pdk", node, "1y")]
            ax.plot(theta, pr, "--", color=col, lw=1.4)
            ax.plot(th, sim, "o", color=col, ms=3.8, mec="white", mew=0.5)
        ax.set_title(f"{cid}: {describe(conds[cid])}", fontsize=8, loc="left", color=INK)
        ax.set_xlabel("threshold θ")
        ax.set_xlim(-0.03, 1.03)
        ax.set_ylim(bottom=0)
    axs[0].set_ylabel("rms error at θ, 1 year (LSB)")
    h = [plt.Line2D([], [], color=INK2, ls="--", lw=1.4, label="predicted (frozen)"),
         plt.Line2D([], [], color=INK2, marker="o", ls="none", ms=3.8, label="Monte Carlo"),
         plt.Line2D([], [], color=BLUE, lw=2, label="input remap"),
         plt.Line2D([], [], color=ORANGE, lw=2, label="supply")]
    axs[-1].legend(handles=h, loc="upper left", fontsize=7, frameon=False)
    fig.tight_layout()
    save(fig, "E2_fig3_theta_profiles")


def main() -> int:
    run = Path(sys.argv[1])
    style()
    info = json.loads((run / "run_info.json").read_text())
    preds = json.loads((ROOT / info["predictions_file"]).read_text())
    conds = {k: v["condition"] for k, v in preds.items()}
    cmp_rows = rcsv(run / "compare.csv")
    fig1(cmp_rows, conds)
    fig2(cmp_rows, conds)
    fig3(preds, rcsv(run / "profile.csv"), conds)
    return 0


if __name__ == "__main__":
    sys.exit(main())
