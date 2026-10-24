"""Markdown tables for the E5 results record (report only; selects and formats rows of the run's CSV files, no new
computation).

Usage: python experiments/E5_tables.py results/E5_<stamp>
Writes <run>/tables.md
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

ORDER_TYPES = [("6T2M", "L6T"), ("6T2M", "UINV"), ("GS", "LBUF"), ("GS", "UINV")]
NODES = ("dac_remap", "supply", "ideal")
NODE_ZH = {"dac_remap": "输入端", "supply": "电源端", "ideal": "理想", "none": "不补偿", "t0": "t0"}


def rcsv(p: Path):
    with open(p, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def f(x, nd=3):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    if v != v:
        return "—"
    return f"{v:.{nd}f}"


def pct(x, nd=1):
    try:
        return f"{100 * float(x):.{nd}f} %"
    except (TypeError, ValueError):
        return str(x)


def table(head, rows):
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def main() -> int:
    run = Path(sys.argv[1])
    md = [f"# E5 tables ({run.name})", ""]

    # ---------------------------------------------------------------- E5.1
    st = {(r["design"], r["cell_set"], r["type"], r["condition"], r["time"], r["node"]): r
          for r in rcsv(run / "e51_stats.csv")}
    for cond in ("all", "unif"):
        md += [f"## E5.1 condition {cond}: t0 and 1 year", ""]
        rows = []
        for d in ("P1", "P3"):
            for cs, ty in ORDER_TYPES:
                for tl, nd in [("t0", "t0")] + [("1y", n) for n in NODES]:
                    r = st[(d, cs, ty, cond, tl, nd)]
                    rows.append([d, f"{cs}/{ty}", f"{tl} {NODE_ZH[nd]}", f(r["rms_noext_lsb"]), f(r["rms_all_lsb"]),
                                 pct(r["p_exc_0.5"]), pct(r["p_exc_1"]), pct(r["p_exc_2"]), f(r["q99_abs_lsb"], 2),
                                 f(r["q99.9_abs_lsb"], 2), r["n_extreme"], r["n"]])
        md += [table(["设计", "单元/支路", "时刻 节点", "rms 不含失效", "rms 含失效", "P(|e|>0.5)", "P(|e|>1)",
                      "P(|e|>2)", "99 % |e|", "99.9 % |e|", "失效", "n"], rows), ""]
    clip = rcsv(run / "e51_clipping.csv")
    md += ["## E5.1 clipping (threshold verify needs more than 25 uS)", "",
           table(["设计", "单元", "支路", "截断", "界数", "比例"],
                 [[r["design"], r["cell_set"], r["type"], r["clipped"], r["constraints"], pct(r["fraction"], 2)]
                  for r in clip]), ""]

    # ---------------------------------------------------------------- E5.2
    lim = rcsv(run / "e52_limits.csv")
    for cond in ("all", "unif"):
        md += [f"## E5.2 condition {cond}, 1 year: gain limit and replica count", ""]
        rows = []
        for d in ("P1", "P3"):
            for cs, ty in ORDER_TYPES:
                for nd in NODES:
                    r = next(x for x in lim if (x["design"], x["cell_set"], x["type"], x["condition"], x["time"],
                                                x["node"]) == (d, cs, ty, cond, "1y", nd))
                    row = [d, f"{cs}/{ty}", NODE_ZH[nd], pct(r["g_max_pos"], 2), pct(r["g_max_neg"], 2),
                           str(r["updates_first_year"])]
                    if cond == "all":
                        row += [f(r["added_rms_E4_64"]), f(r["added_rms_N16"]), f(r["added_rms_N64"]),
                                f(r["added_rms_N256"]), f(r["added_rms_N1024"]), str(r["min_N_within_budget"])]
                    rows.append(row)
        head = ["设计", "单元/支路", "节点", "g 上限 (+)", "g 上限 (−)", "第一年更新次数"]
        if cond == "all":
            head += ["附加 rms E4 64", "N=16", "N=64", "N=256", "N=1024", "最小 N"]
        md += [table(head, rows), ""]
    ah = rcsv(run / "e52_ahat.csv")
    md += ["## E5.2 replica estimate relative to the 1e6 limit (condition all)", "",
           table(["时刻", "变体", "参数", "抽样数", "平均偏差", "rms 偏差"],
                 [[r["time"], r["variant"], r["param"], r["draws"], pct(r["mean_rel_to_limit"], 3),
                   pct(r["rms_rel_to_limit"], 3)] for r in ah if r["variant"] in ("a", "d")]), ""]

    # ---------------------------------------------------------------- E5.3
    mb = rcsv(run / "e53_min_bits.csv")
    for cond in ("all", "unif"):
        md += [f"## E5.3 condition {cond}: smallest DAC resolution with added rms <= 0.1 LSB", ""]
        rows = []
        for d in ("P1", "P3"):
            for cs, ty in ORDER_TYPES:
                def g(tl, nd, q, gr):
                    r = next(x for x in mb if (x["design"], x["cell_set"], x["type"], x["condition"], x["time"],
                                               x["node"], x["quantity"], x["grid"]) == (d, cs, ty, cond, tl, nd, q, gr))
                    return str(r["min_bits"])
                rows.append([d, f"{cs}/{ty}", g("t0", "t0", "DL", "full"), g("t0", "t0", "DL", "span"),
                             g("1y", "dac_remap", "DL", "full"), g("1y", "dac_remap", "DL", "span"),
                             g("1y", "supply", "DL", "full"), g("1y", "supply", "DL", "span"),
                             g("1y", "supply", "SL", "worst")])
        md += [table(["设计", "单元/支路", "t0 DL 满量程", "t0 DL 区间", "1 年输入端 DL 满量程", "1 年输入端 DL 区间",
                      "1 年电源端 DL 满量程", "1 年电源端 DL 区间", "1 年电源端 V_SL（最坏）"], rows), ""]

    # ---------------------------------------------------------------- E5.4
    bud = rcsv(run / "e54_budget.csv")
    for cond in ("unif", "all"):
        md += [f"## E5.4 condition {cond}: time to budget", ""]
        rows = []
        for d in ("P1", "P3"):
            for cs, ty in ORDER_TYPES:
                for nd in ("none", "dac_remap", "supply", "ideal"):
                    sel = [x for x in bud if (x["design"], x["cell_set"], x["type"], x["condition"], x["node"]) ==
                           (d, cs, ty, cond, nd)]
                    by = {(x["metric"], x["budget"]): x for x in sel}
                    t0v = by[("rms_noext", "0.5")]["value_t0"]
                    rows.append([d, f"{cs}/{ty}", NODE_ZH[nd], f(t0v)] +
                                [by[("rms_noext", b)]["t_budget"] for b in ("0.5", "1.0", "1.5")] +
                                [by[("p_exc_1", b)]["t_budget"] for b in ("0.05", "0.2")])
        md += [table(["设计", "单元/支路", "节点", "t0 rms", "rms 0.5", "rms 1.0", "rms 1.5", "P(|e|>1) 5 %",
                      "P(|e|>1) 20 %"], rows), ""]
    (run / "tables.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"wrote {run / 'tables.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
