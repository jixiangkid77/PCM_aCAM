"""Figures for E0.2 (PCM_aCAM).

Usage
-----
    python experiments/E0_2_figures.py results/E0_2_<stamp>

Writes figures/E0_2_fig1_coupling.{png,pdf}. Every plotted number is read from summary.csv of
the given run (4 bits).
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BLUE, ORANGE, GREEN, GREY, INK, INK2 = "#2a78d6", "#eb6834", "#1baf7a", "#8a8a86", "#1f1f1f", "#52514e"
T_LABELS = ["t0", "1h", "1d", "30d", "365d"]
X_TICKS = ["t$_0$\n(prog)", "t$_0$", "1 h", "1 d", "30 d", "1 y"]
SETTINGS = [("m10", "10 mV"), ("m20", "20 mV\n(primary)"), ("m30", "30 mV"), ("m20_r02", "20 mV,\n0.2 V span"),
            ("m20_vth_only", "V$_{TH}$ only\n20 mV"), ("m20_vtx_only", "V$_{TX}$ only\n20 mV")]


def read_csv(path: Path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def loss(rows, mism, node, cond, t, prog="threshold"):
    for r in rows:
        if (r["mismatch"], r["programming"], r["node"], r["condition"], r["t"]) == (mism, prog, node, cond, t):
            return float(r["mean_loss_pp"])
    raise KeyError((mism, prog, node, cond, t))


def main() -> int:
    run = Path(sys.argv[1])
    rows = read_csv(run / "summary.csv")
    fig_dir = ROOT / "figures"
    fig_dir.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": GREY, "xtick.color": INK2, "ytick.color": INK2, "axes.labelcolor": INK2})

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.6, 3.2), gridspec_kw={"width_ratios": [1, 1.25]})
    x = list(range(len(X_TICKS)))
    ends = []
    for node, col, lab in (("dac", ORANGE, "DAC gain"), ("supply", GREEN, "divider supply"), ("ideal", BLUE, "ideal (G / â)")):
        y = [loss(rows, "m20", node, "prog", "t0")] + [loss(rows, "m20", node, "all", t) for t in T_LABELS]
        ax1.plot(x, y, "-o", color=col, ms=4, lw=1.8)
        ends.append([y[-1], y[-1], col, lab])
    ends.sort(key=lambda e: -e[0])                 # top to bottom; keep labels >= 0.42 pp apart
    for i in range(1, len(ends)):
        ends[i][1] = min(ends[i][1], ends[i - 1][1] - 0.42)
    for y_end, y_lab, col, lab in ends:
        ax1.annotate(f"{lab}\n{y_end:.2f} pp at 1 y", (x[-1], y_end), xytext=(x[-1] + 0.25, y_lab),
                     textcoords="data", ha="left", va="center", color=INK, fontsize=7.5,
                     arrowprops={"arrowstyle": "-", "color": col, "lw": 0.8})
    ax1.set_xticks(x, X_TICKS)
    ax1.set_xlim(-0.3, len(x) + 1.3)
    ax1.set_ylim(-0.6, 3.2)
    ax1.axhline(0.0, color="#cfcfcb", lw=0.6)
    ax1.set_ylabel("accuracy loss vs ideal 4-bit CAM (pp)")
    ax1.set_title("(a) primary: 20 mV mismatch, threshold verify", fontsize=9, loc="left", color=INK)

    w = 0.36
    xs = list(range(len(SETTINGS)))
    dac = [loss(rows, m, "dac", "all", "365d") - loss(rows, m, "ideal", "all", "365d") for m, _ in SETTINGS]
    sup = [loss(rows, m, "supply", "all", "365d") - loss(rows, m, "ideal", "all", "365d") for m, _ in SETTINGS]
    ax2.bar([i - w / 2 - 0.01 for i in xs], dac, width=w, color=ORANGE, label="DAC gain − ideal")
    ax2.bar([i + w / 2 + 0.01 for i in xs], sup, width=w, color=GREEN, label="divider supply − ideal")
    for i, (d, s) in enumerate(zip(dac, sup)):
        if SETTINGS[i][0] == "m20" or d >= 1.0:
            ax2.annotate(f"{d:+.2f}", (i - w / 2, max(d, 0)), xytext=(0, 2), textcoords="offset points",
                         ha="center", va="bottom", fontsize=7, color=INK)
        if SETTINGS[i][0] == "m20" or s >= 1.0:
            ax2.annotate(f"{s:+.2f}", (i + w / 2, max(s, 0)), xytext=(0, 2), textcoords="offset points",
                         ha="center", va="bottom", fontsize=7, color=INK)
    ax2.axhline(1.0, color=GREY, lw=0.9, ls="--")
    ax2.text(len(xs) - 0.5, 1.06, "gate A: 1.0 pp", ha="right", va="bottom", fontsize=7.5, color=INK2)
    ax2.axhline(0.0, color="#cfcfcb", lw=0.6)
    ax2.set_xticks(xs, [lab for _, lab in SETTINGS], fontsize=7.5)
    ax2.set_ylabel("extra loss at 1 year vs ideal node (pp)")
    ax2.set_title("(b) coupling by mismatch setting, 1 year", fontsize=9, loc="left", color=INK)
    ax2.legend(frameon=False, fontsize=7.5, loc="upper left")
    ax2.set_ylim(-0.4, 3.0)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(fig_dir / f"E0_2_fig1_coupling.{ext}", dpi=200)
    plt.close(fig)
    print("figures written to", fig_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
