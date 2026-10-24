"""Tables for the E6 results record (selection, averaging over splits and the energy products defined in the plan;
no new simulation).

Usage: python experiments/E6_tables.py results/E6_<stamp>
Writes <run>/tables.md and <run>/energy_search.csv.

Energy per search (plan section 4, decision D3): mean static power x window, window = t_s(n_features) + the largest
settling time (C_par 0.5 fF) of the two branch types of the cell set at that node and time (E6.1).
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

NODE_ZH = {"t0": "t0", "dac_remap": "输入端", "supply": "电源端"}
ORDER_NT = [("t0", "t0"), ("dac_remap", "1d"), ("dac_remap", "30d"), ("dac_remap", "1y"), ("supply", "1d"),
            ("supply", "30d"), ("supply", "1y")]
TYPES = [("6T2M", "L6T"), ("6T2M", "UINV"), ("GS", "LBUF"), ("GS", "UINV")]
CELL_TYPES = {"6T2M": ("L6T", "UINV"), "GS": ("LBUF", "UINV")}


def rcsv(p: Path):
    with open(p, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def f(x, nd=2):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    if v != v:
        return "—"
    return f"{v:.{nd}f}"


def table(head, rows):
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def main() -> int:
    run = Path(sys.argv[1])
    md = [f"# E6 tables ({run.name})", ""]
    cmain = "5e-16"
    # ------------------------------------------------------------------ E6.1
    ss = rcsv(run / "settle_summary.csv")
    key = {(r["design"], r["cell_set"], r["type"], r["node"], r["time"], r["c_par_f"]): r for r in ss}
    cpar_vals = sorted({r["c_par_f"] for r in ss}, key=float)
    cm = next(c for c in cpar_vals if abs(float(c) - 5e-16) < 1e-20)
    cap = rcsv(run / "capacitance.csv")
    ts = {int(r["drain_at"].split("=")[1]): float(r["t_s_ns"]) for r in cap if r["drain_at"].startswith("ML N=")}
    md += [f"## E6.1 最大建立时间（ns，16 个门限 × 2 个起点中的最大值；C_par 0.5 fF；t_s(16) = {f(ts.get(16, float('nan')), 1)} ns）", ""]
    rows = []
    for d in sorted({r["design"] for r in ss}):
        for cs, ty in TYPES:
            r0 = key.get((d, cs, ty, "t0", "t0", cm))
            if r0 is None:
                continue
            cells = []
            for node, t in ORDER_NT:
                r = key.get((d, cs, ty, node, t, cm))
                cells.append(f(r["t_settle_max_ns"], 1) if r else "")
            rows.append([d, f"{cs}/{ty}"] + cells)
    md += [table(["设计", "单元/支路"] + [f"{NODE_ZH[n]} {t}" for n, t in ORDER_NT], rows), ""]
    md += ["## E6.1 对 t0 的比值与占 t_s(16) 的比例（C_par 0.5 fF，1 年）", ""]
    rows = []
    for d in sorted({r["design"] for r in ss}):
        for cs, ty in TYPES:
            r0 = key.get((d, cs, ty, "t0", "t0", cm))
            if r0 is None:
                continue
            rr = [key.get((d, cs, ty, n, "1y", cm)) for n in ("dac_remap", "supply")]
            rows.append([d, f"{cs}/{ty}", f(r0["t_settle_max_ns"], 1)] +
                        [f(r["ratio_to_t0"], 2) if r else "" for r in rr] +
                        [f(float(r0["t_settle_max_ns"]) / ts[16] * 100, 1) + " %"] +
                        [f(float(r["t_settle_max_ns"]) / ts[16] * 100, 1) + " %" if r else "" for r in rr] +
                        [f"{r0['at_level']}/{r0['at_x_prev']}"] + [f"{r['at_level']}/{r['at_x_prev']}" if r else ""
                                                                   for r in rr])
    md += [table(["设计", "单元/支路", "t0 (ns)", "输入端 1y / t0", "电源端 1y / t0", "t0 / t_s", "输入端 1y / t_s",
                  "电源端 1y / t_s", "t0 最大处（级/起点）", "输入端最大处", "电源端最大处"], rows), ""]
    if len(cpar_vals) > 1:
        md += ["## E6.1 C_par 敏感性（最大建立时间 ns；t0 与 1 年，4 个门限）", ""]
        rows = []
        for d in sorted({r["design"] for r in ss}):
            for cs, ty in TYPES:
                for node, t in (("t0", "t0"), ("dac_remap", "1y"), ("supply", "1y")):
                    vals = []
                    for c in cpar_vals:
                        r = key.get((d, cs, ty, node, t, c))
                        vals.append(f(r["t_settle_max_ns"], 1) if r else "")
                    rows.append([d, f"{cs}/{ty}", f"{NODE_ZH[node]} {t}"] + vals)
        md += [table(["设计", "单元/支路", "状态"] + [f"C_par {float(c) * 1e15:g} fF" for c in cpar_vals], rows),
               "", "注：0.5 fF 一列是全部 16 个门限的最大值，0 与 2 fF 只跑了 4 个门限（0、5、10、15 级）", ""]
    # ------------------------------------------------------------------ E6.2
    pf = rcsv(run / "power_forest.csv")
    agg = defaultdict(list)
    for r in pf:
        agg[(r["dataset"], r["design"], r["cell_set"], r["node"], r["time"])].append(r)

    def mean(rs, k):
        return sum(float(x[k]) for x in rs) / len(rs)

    md += ["## E6.2 每次搜索的静态功率（µW，3 个划分的平均）与对 t0 的比值", ""]
    rows = []
    datasets = sorted({r["dataset"] for r in pf}, key=lambda s: ["iris", "wine", "breast_cancer", "digits",
                                                                  "synthetic"].index(s))
    designs = sorted({r["design"] for r in pf})
    for ds in datasets:
        for d in designs:
            for cs in ("6T2M", "GS"):
                base = agg.get((ds, d, cs, "t0", "t0"))
                if not base:
                    continue
                p0 = mean(base, "p_search_uw")
                cells = [f(p0, 0)]
                for node, t in ORDER_NT[1:]:
                    rs = agg.get((ds, d, cs, node, t))
                    cells.append(f(mean(rs, "p_search_uw") / p0, 3) if rs else "")
                rows.append([ds, d, cs, f(mean(base, "n_bounds"), 0)] + cells)
    md += [table(["数据集", "设计", "单元组", "界数", "t0 µW"] + [f"{NODE_ZH[n]} {t} / t0" for n, t in ORDER_NT[1:]],
                 rows), ""]
    md += ["## E6.2 每个界的平均功率（µW）：满足与违反分开（1 年）", ""]
    rows = []
    for ds in datasets:
        for d in designs:
            for cs in ("6T2M", "GS"):
                cells = []
                for node, t in (("t0", "t0"), ("dac_remap", "1y"), ("supply", "1y")):
                    rs = agg.get((ds, d, cs, node, t))
                    cells += [f(mean(rs, "p_per_bound_uw"), 3), f(mean(rs, "p_satisfied_uw"), 3),
                              f(mean(rs, "p_violated_uw"), 3)] if rs else ["", "", ""]
                base = agg.get((ds, d, cs, "t0", "t0"))
                rows.append([ds, d, cs, f(mean(base, "share_violated") * 100, 1) + " %"] + cells)
    md += [table(["数据集", "设计", "单元组", "违反比例", "t0 均值", "t0 满足", "t0 违反", "输入端 均值", "输入端 满足",
                  "输入端 违反", "电源端 均值", "电源端 满足", "电源端 违反"], rows), ""]
    # ------------------------------------------------------------------ energies
    dl = rcsv(run / "energy_dl_ml.csv")
    dag = defaultdict(list)
    for r in dl:
        dag[(r["dataset"], r["design"], r["cell_set"], r["node"], r["time"])].append(r)
    erows = []
    for (ds, d, cs, node, t), rs in sorted(agg.items()):
        p = mean(rs, "p_search_uw") * 1e-6
        dd = dag[(ds, d, cs, node, t)]
        n_feat = int(dd[0]["n_features"])
        st = max(float(key[(d, cs, ty, node, t, cm)]["t_settle_max_ns"]) for ty in CELL_TYPES[cs]) * 1e-9
        win = ts[n_feat] * 1e-9 + st
        erows.append({"dataset": ds, "design": d, "cell_set": cs, "node": node, "time": t, "n_features": n_feat,
                      "t_s_ns": ts[n_feat], "t_settle_max_ns": st * 1e9, "window_ns": win * 1e9,
                      "p_search_uw": p * 1e6, "e_static_nj": p * win * 1e9,
                      "e_dl_nj": mean(dd, "e_dl_pj") * 1e-3, "e_ml_nj": mean(dd, "e_ml_pj") * 1e-3})
    with open(run / "energy_search.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(erows[0]))
        w.writeheader()
        w.writerows(erows)
    ek = {(r["dataset"], r["design"], r["cell_set"], r["node"], r["time"]): r for r in erows}
    md += ["## E6.2–E6.3 每次搜索的能量（nJ）：静态 + DL + ML（t0 与 1 年）", ""]
    rows = []
    for ds in datasets:
        for d in designs:
            for cs in ("6T2M", "GS"):
                cells = []
                for node, t in (("t0", "t0"), ("dac_remap", "1y"), ("supply", "1y")):
                    r = ek.get((ds, d, cs, node, t))
                    cells += [f(r["window_ns"], 0), f(r["e_static_nj"], 3), f(r["e_dl_nj"], 4), f(r["e_ml_nj"], 4)] \
                        if r else ["", "", "", ""]
                rows.append([ds, d, cs] + cells)
    md += [table(["数据集", "设计", "单元组", "t0 窗口 ns", "t0 静态", "t0 DL", "t0 ML", "输入端 窗口", "输入端 静态",
                  "输入端 DL", "输入端 ML", "电源端 窗口", "电源端 静态", "电源端 DL", "电源端 ML"], rows), ""]
    # ------------------------------------------------------------------ E6.3 capacitance
    gq = [r for r in cap if not r["drain_at"].startswith("ML")]
    md += ["## E6.3 分压管栅电荷（栅 0 → 1.8 V，漏极接该支路的 V_sp）", "",
           table(["设计", "漏极所在支路", "V_drain (V)", "Q (fC)", "平均电容 (fF)"],
                 [[r["design"], r["drain_at"], f(r["v_drain_v"], 4), f(r["q_full_fc"], 3), f(r["c_avg_ff"], 3)]
                  for r in gq]), ""]
    mq = [r for r in cap if r["drain_at"].startswith("ML")]
    md += ["## E6.3 匹配线：感测时间与预充电", "",
           table(["N（特征数）", "t_s (ns)", "Q_ML (fC)", "预充电能量 (pJ/行)"],
                 [[r["drain_at"].split("=")[1], f(r["t_s_ns"], 1), f(r["q_ml_fc"], 1), f(r["e_precharge_pj"], 4)]
                  for r in mq]), ""]
    # ------------------------------------------------------------------ E6.4
    ov = rcsv(run / "overhead.csv")
    od = [r for r in ov if r["dataset"]]
    md += ["## E6.4 阵列与复制单元（3 个划分的平均）", "",
           table(["数据集", "行数", "特征数", "单元数", "PCM 器件", "已编程的界", "64 个复制单元占比", "256 个复制单元占比"],
                 [[r["dataset"], f(r["rows_mean"], 1), r["features"], f(r["cells"], 0), f(r["pcm_devices"], 0),
                   f(r["programmed_bounds_mean"], 1), f(float(r["replica_64_share"]) * 100, 2) + " %",
                   f(float(r["replica_256_share"]) * 100, 2) + " %"] for r in od]), ""]
    og = [r for r in ov if not r["dataset"]]
    md += ["## E6.4 补偿电路的计数与面积上限（每个单元组）", "",
           table(["设计", "满量程 DL 位数", "LUT 位数", "LUT 面积（触发器计，mm²）", "触发器面积 (µm²)", "电源 DAC 位数",
                  "可调电源个数", "分压管栅面积 (µm²/单元)", "第一年更新次数"],
                 [[r["design"], r["dl_bits_full_scale"], r["lut_bits"], f(float(r["lut_area_um2_ff"]) * 1e-6, 4),
                   f(r["ff_area_um2"], 2), r["vsl_dac_bits"], r["adjustable_supplies_per_cell_set"],
                   f(r["divider_gate_area_per_cell_um2"], 3),
                   f"{r['updates_first_year_min']}–{r['updates_first_year_max']}"] for r in og]), ""]
    (run / "tables.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"wrote {run / 'tables.md'} and energy_search.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
