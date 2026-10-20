"""E4 independent check: write report.md (Chinese, like the E4 protocol) from compare.json.

Usage: python experiments/E4_independent/make_report.py results/<check folder>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def _f(x, nd=3):
    return f"{x:.{nd}f}"


def _g(x, nd=2):
    return f"{x:.{nd}e}"


def write(out_dir: Path, cmp: dict, info: dict):
    t1, t2, t3 = cmp["task1"], cmp["task2"], cmp["task3"]
    pf = {True: "PASS", False: "FAIL"}
    files = sorted(p.relative_to(out_dir).as_posix() for p in out_dir.rglob("*") if p.is_file())
    files.append("report.md")
    files = sorted(set(files))
    spice_files = [f for f in files if f.startswith("spice_")]
    main_files = [f for f in files if not f.startswith("spice_")]

    L = []
    a = L.append
    a("# E4 独立复核报告")
    a("")
    a(f"- 运行目录：`{info['run_dir']}/`（{info['start_jst']} – {cmp['end_jst']} JST，用时 {cmp['elapsed_s']} s）")
    a("- 复核代码：`experiments/E4_independent/`（自己实现；只导入 `src/pcm_acam/trees.py`（dataset_bundle、quantize，"
      "单独加载，不执行包的其他文件）与 `src/pcm_acam/acam.py` 中的 make_draws（从源码只取该函数）；未读、未导入 "
      "`row_model.py` 与 `E4_forest_system.py`）")
    a(f"- 输入：`instrument.npz` 的 SHA-256 与方案第 10 节记录的值"
      f"{'一致' if info['instrument_sha256_matches_protocol'] else '不一致'}；全部输入与复核脚本的 SHA-256 见 `run_info.json`")
    env = info["environment"]
    a(f"- 环境：Python {env['python']}、numpy {env['numpy']}、scikit-learn {env['sklearn']}、{env['ngspice']}；"
      "SPICE（Simulation Program with Integrated Circuit Emphasis，电路仿真程序）进程同一时刻只运行一个")
    a("")
    a("## 1 结论")
    a("")
    a("| 任务 | 判定 | 关键数字 |")
    a("|---|---|---|")
    a(f"| 1 森林精度重算（{t1['n_evaluations']} 个评估） | {pf[t1['pass']]} | 精度相同 {t1['acc_identical']}/{t1['n_evaluations']}；"
      f"叶子丢失率相同 {t1['n_evaluations'] - t1['leaf_loss_differing']}/{t1['n_evaluations']}；"
      f"另外 â、多重匹配率、被串扰去掉的匹配数也全部相同 |")
    rr = t2["ratio_indep_to_pred_range"]
    a(f"| 2 V1（行验证：全行瞬态仿真）6 例，自己的 SPICE 网表 | {pf[t2['pass']]} | 越过时间与主运行最大相对差 "
      f"{100 * t2['max_abs_rel_diff_vs_main']:.4f} %（门槛 0.5 %）；与 t_pred 之比 {_f(rr[0], 4)}–{_f(rr[1], 4)} |")
    mx3 = max(v["max_abs_diff_uV_inst1_20"] for v in t3["per_type"].values())
    a(f"| 3 翻转点（每种支路实例 1–20） | {pf[t3['pass']]} | 最大差 {_f(mx3)} µV（门槛 50 µV） |")
    a("")
    a(f"三项全部{'通过' if cmp['all_pass'] else '未全部通过'}。")
    a("")

    # ---------------------------------------------------------------- task 1
    a("## 2 任务 1：森林精度（wine、breast_cancer，划分 0，种子 0–1）")
    a("")
    a("范围：P1、P3 两个设计 × xt、is、gs、gi 四个臂 × all、unif 两个条件 × 10 个点（t0；1 天、30 天、1 年 × "
      "dac_remap、supply、ideal）= 640 个评估，与 `per_eval.csv` 逐项比较。术语：LSB（least significant bit，4 bit 门限的最小间隔 1/15）；"
      "PCM（phase change memory，相变存储器）；â（64 个 PCM 复制单元估计的漂移因子）；xt/gs 为含匹配线（ML，match line）串扰的臂，"
      "is/gi 为理想感测的参照臂；θ 为量化门限，x_eff 为器件实现的有效门限。")
    a("")
    a("| 项 | 结果 |")
    a("|---|---|")
    for ds, v in t1["refs"].items():
        a(f"| {ds}：acc_ref、行数、约束数、下界数 | {v['acc_ref_indep']:.6f}、{v['n_rows_indep']}、{v['n_constraints_indep']}、"
          f"{v['n_lower_indep']}（与 refs.csv {'相同' if v['equal'] else '不同'}） |")
    a(f"| 精度（acc）完全相同 | {t1['acc_identical']}/{t1['n_evaluations']}（最大差 {t1['max_sample_difference']} 个样本） |")
    a(f"| 叶子丢失率（leaf_loss）完全相同 | {t1['n_evaluations'] - t1['leaf_loss_differing']}/{t1['n_evaluations']} |")
    a(f"| â 最大相对差 | {t1['a_hat_max_rel_difference']:.1e} |")
    a(f"| 多重匹配率、被串扰去掉的匹配数（brief 未定义，按“每个（样本、树）多于一行匹配的比例”与“界都满足但被串扰判据去掉的（样本、行）数”实现） | "
      f"不同的评估分别 {t1['multi_match_differing']}、{t1['removed_by_crosstalk_differing']} 个 |")
    a(f"| 串扰判据：被匹配的（样本、行）离门槛最近的相对距离 | {t1['min_rel_gap_to_crosstalk_threshold']:.1e}（没有落在 10⁻⁶ 以内的） |")
    c2 = t1["c2_reproduction"]
    a(f"| 检查 C2（无失配、无 PCM 噪声、均匀漂移时有效门限的精确性）在 checks.json 中的 32 个数值重算 | "
      f"40 步二分：{c2['40']['values_bit_identical']}/{c2['40']['values_total']} 逐位相同；"
      f"64 步二分：{c2['64']['values_bit_identical']}/{c2['64']['values_total']} |")
    a("")
    a("实现要点：分压管实例曲线 = 名义曲线（1 mV 网格线性插值）+ Catmull-Rom 三次插值的偏差；边界电压用二分法在 [vg_c[0], vg_c[-1]] 上反查"
      "（窗外 ±inf → x_eff = ±10）；输入映射 M0 用名义曲线的线性反插值；漏电按 brief 的负载线逐实例求节点电压（I–V 面：沿 V_G 线性插值 ln I，"
      "再沿 V_DS 线性插值 I），经实例自己的反相器传输曲线与下拉管放电曲线得到 w；UINV 在界内 ≥ 3 LSB、LBUF ≥ 0.45 LSB 时栅压取 0 V（w 用该实例下拉管在 0 V 的值）。")
    a("")
    a("二分步数：brief 只规定“二分法”。40 步、取区间中点时，C2 的 32 个数值（含 UINV 共用数据线残余 0.1026、0.0043、0.2253、0.0072 LSB）"
      "与 checks.json 逐位相同，所以主计算用 40 步。")
    a("")
    if "sensitivity" in t1:
        s64 = t1["sensitivity"]["bisection_64_steps"]
        sbl = t1["sensitivity"]["surface_bilinear_ln"]
        a("敏感性（只报告，不参与判定）：")
        a("")
        a("| 变体 | 精度不同 | 叶子丢失率不同 | 出现差别的点 |")
        a("|---|---|---|---|")
        for name, s in (("边界反查用 64 步二分（到浮点极限）", s64), ("I–V 面改为 ln I 双线性插值", sbl)):
            pts = "、".join(f"{p}（{d['leaf_loss']}）" for p, d in s["differences_by_point"].items()) or "无"
            a(f"| {name} | {s['acc_differing']}（最大 {s['max_sample_difference']} 个样本） | {s['leaf_loss_differing']}"
              f"（最大 {s['max_abs_leaf_loss_difference']:.2e}） | {pts} |")
        a("")
        pts = list(s64["differences_by_point"].keys())
        only_ties = all(p.startswith("unif/") and (p.endswith("/ideal") or p == "unif/t0/t0") for p in pts)
        ties = []
        for ds, v in t1["refs"].items():
            et = v["exact_ties_x_eq_theta"]
            xs = "、".join("0" if k == 0 else f"{k}/15" for k in et["x_times_15"])
            fs = "、".join(str(f_) for f_ in et["features"])
            ties.append(f"{ds} {et['pairs']} 个（x = {xs}；特征编号 {fs}）")
        a(f"64 步二分时，差别{'只' if only_ties else ''}出现在上表所列的点"
          f"{'（unif 条件的 t0 与 ideal：这些点上 x_eff 与 θ 只差舍入误差，不超过约 10⁻¹¹）' if only_ties else ''}。"
          f"测试输入恰好等于量化门限的（样本、界）对：{'；'.join(ties)}。x = 0 是等于或低于训练最小值（被截断）的输入，"
          "其余是取值离散的特征恰好落在 k/15。这些对的界是否满足取决于舍入误差的符号；用 40 步（与主运行相同）时差别全部消失。")
        a("")

    # ---------------------------------------------------------------- task 2
    a("## 3 任务 2：V1 行瞬态（breast_cancer，N = 30，2N = 60 个下拉管）")
    a("")
    a("t_sim：全行瞬态仿真中匹配线电压降到 0.9 V 的时间；t_pred = t_s(N)·(1 + (2N − 1)·w0)/S：倒数时间相加规则的预测"
      "（t_s(N) 为感测时间，S 为该行所有下拉管的相对漏电之和，w0 为栅压 0 V 的下拉管的相对漏电）。")
    a("")
    a(f"每个（设计、补偿点）组的第一个例子；ML 电容 C_SA + N·C_wire = {t2['c_ml_F'] * 1e15:.0f} fF；`.ic` 预充电 1.8 V（不用 UIC）；"
      f"0.9 V 越过时间；瞬态选项 `{t2['tran']['options']}`，最大步长 {t2['tran']['tmax_s'] * 1e9:.1f} ns；"
      "失配：`default_rng([20261017, 5])`，按 div_lo、pd_lo、b1n、b1p、b2n、b2p、div_hi、un、up、pd_hi 的顺序各抽 (2000, 4)，实例 k 用第 k − 1 行。")
    a("")
    a("| 例 | 设计 | 补偿点 | 界数 | t_sim 主运行 (ns) | t_sim 复核 (ns) | 相对差 | 复核 / t_pred |")
    a("|---|---|---|---|---|---|---|---|")
    for c in t2["cases"]:
        a(f"| {c['case']} | {c['design']} | {c['node']} | {c['n_bounds']} | {c['t_sim_main_ns']:.3f} | {c['t_sim_indep_ns']:.3f} | "
          f"{100 * c['rel_diff_vs_main']:+.4f} % | {c['ratio_indep_to_pred']:.4f} |")
    a("")
    ms = t2["model_side_all_48_cases"]
    a(f"- 感测时间 t_s(30)（名义下拉管栅压 V_sw,nom，其余 59 个 0 V）：复核 {t2['t_s_indep_ns']:.3f} ns，主运行 {t2['t_s_main_ns']:.3f} ns")
    a(f"- 与 t_pred 的最大 |ln 比| {t2['max_abs_ln_ratio_indep_to_pred']:.4f}（V1 门槛 0.05）")
    labels = {"default_tolerances": "ngspice 默认容差", "tmax_0.4ns": "最大步长 0.4 ns"}
    if t2.get("numerical_sensitivity"):
        a("- 数值敏感性（例 " + str(t2["numerical_sensitivity"][0]["case"]) + "）：" + "；".join(
            f"{labels.get(s['variant'], s['variant'])}，越过时间相对变化 {100 * s['rel_change']:+.4f} %"
            for s in t2["numerical_sensitivity"]))
    a(f"- 模型侧（全部 {ms['n_cases']} 例、{ms['n_bounds']} 个界，用任务 1 的实现重算 v1_cases.json）：实例编号全同；"
      f"电导的最大相对差 {ms['max_rel_g_us']:.1e}、x_eff 的最大差 {ms['max_abs_theta_eff']:.1e}（0 即逐位相同）；"
      f"数据线电压最大差 {ms['max_abs_v_dl_V']:.1e} V；漏电权重 w 最大相对差 {ms['max_rel_w']:.1e}；S（行的漏电和）最大相对差 {ms['max_rel_S']:.1e}")
    gc = t2["gate_2ns_by_class"]
    gi_, go_, gu_ = gc["in_window"], gc["node_outside_transfer_window"], gc["u_cut_gate_0V"]
    a(f"- 诊断：2 ns 时被测下拉管栅压，SPICE 与模型之差。节点在存储的反相器传输曲线窗口（V_sp ± 60 mV）以内的 {gi_['n_bounds']} 个界："
      f"最大 {gi_['max_abs_diff_mV']:.2f} mV；UINV 节点在窗口以外、模型取窗口端点值的 {go_['n_bounds']} 个界：最大 {go_['max_abs_diff_mV']:.2f} mV"
      f"（w ≤ {go_['max_w_model']:.1e}）；因 u 截断模型取 0 V 的 {gu_['n_bounds']} 个界：最大 {gu_['max_abs_diff_mV']:.2f} mV"
      f"（w ≤ {gu_['max_w_model']:.1e}）。后两类的 w 与 w0 = {t2['w0']:.2e}（截止下拉管的相对漏电）同量级，对越过时间没有影响。"
      "逐界数值见 `task2_gates_2ns.csv`")
    a("")

    # ---------------------------------------------------------------- task 3
    a("## 4 任务 3：翻转点（SPICE 直流扫描）")
    a("")
    a("L6T：下拉管（pd_lo）在 V_DS = 1.8 V 时导通 280 nA 的栅压（0.1 mV 步长，ln I 线性插值）；UINV：反相器（un、up）输出等于上界下拉管（pd_hi）"
      "V_sw 时的输入；LBUF：W* = 反相器 2（b2n、b2p）输出等于下界下拉管 V_sw 时的输入，X* = 反相器 1（b1n、b1p）输出等于 W* 时的输入"
      "（反相器 20 µV 步长，线性插值；三次反插值估计的插值误差 "
      f"{t3['interp_error_estimate_uV']['inverter']:.4f} µV）。")
    a("")
    a("| 量 | 实例 1–20 最大 \\|差\\| (µV) | 平均差 (µV) | 名义实例差 (µV) | 判定 |")
    a("|---|---|---|---|---|")
    for typ, v in t3["per_type"].items():
        a(f"| {typ} | {v['max_abs_diff_uV_inst1_20']:.3f} | {v['mean_diff_uV_inst1_20']:+.3f} | {v['nominal_diff_uV']:+.3f} | {pf[v['pass']]} |")
    a("")
    cg = t3["inverter_0p5mV_grid_emulation_max_abs_diff_uV"]
    pt = t3["per_type"]
    a(f"- V_sw,nom：复核 {t3['vsw_nominal_indep_V']:.10f} V，主运行 {t3['vsw_nominal_main_V']:.10f} V"
      f"（L6T 实例 1–20 最大差 {pt['L6T']['max_abs_diff_uV_inst1_20']:.3f} µV）")
    inv_max = max(pt[t]["max_abs_diff_uV_inst1_20"] for t in ("UINV", "LBUF", "LBUF_Wstar"))
    inv_mean = sum(pt[t]["mean_diff_uV_inst1_20"] for t in ("UINV", "LBUF", "LBUF_Wstar")) / 3
    a(f"- 反相器相关的三个量平均差约 {inv_mean:+.1f} µV（最大 {inv_max:.2f} µV）。把同一组 SPICE 数据缩减到配置中的反相器扫描步长"
      f"（0.5 mV 网格）再线性插值后，与主运行的差降到 UINV {cg['UINV']:.4f}、LBUF {cg['LBUF']:.4f}、W* {cg['LBUF_Wstar']:.4f} µV。"
      "即主运行在 0.5 mV 网格上线性插值，本复核用 20 µV 网格；差别是主运行的网格插值误差，比门槛小约两个数量级")
    if "extras" in t3:
        ex = t3["extras"]
        pdw = ex["pulldown_ln_w_max_abs_diff_Vsw-0.25..+0.05"]
        inv = ex["inverter_transfer_max_abs_diff_mV"]
        invk = ex["inverter_transfer_max_abs_diff_mV_from_0.5mV_knots"]
        dv = ex["divider_bound_voltage_max_abs_diff_uV_0.2-5uA"]
        a("- 附加（brief 未要求，检查任务 1 所用的存储曲线，实例 0–20）："
          f"下拉管放电曲线（V_DS = 1.35 V，V_sw − 0.25 至 + 0.05 V）|Δ ln w| 最大 L6T {pdw['L6T']:.1e}、UINV {pdw['UINV']:.1e}；"
          f"ln_ref 差 {ex['ln_ref_indep_minus_main']:.1e}；"
          f"反相器传输曲线最大差 {max(inv.values()):.3f} mV（按 0.5 mV 网格线性插值后 {max(invk.values()):.4f} mV，原因同上）；"
          f"分压管曲线在 0.2–5 µA 的边界电压最大差 {max(dv.values()):.2f} µV")
    a("")

    # ---------------------------------------------------------------- observations
    a("## 5 观察")
    a("")
    a("1. 森林阶段（任务 1）在 640 个评估上与主运行逐项相同，连 brief 未定义的两个附带量（多重匹配率、被串扰去掉的匹配数）也相同，"
      "说明 brief 描述的模型（编程、PCM 噪声、补偿点、逐实例漏电、倒数时间相加判据）与主运行的实现一致。")
    if "sensitivity" in t1:
        s64 = t1["sensitivity"]["bisection_64_steps"]
        n_acc64 = s64["acc_differing"]
        acc_txt = "仍全部相同" if n_acc64 == 0 else f"有 {n_acc64} 个评估不同"
        pairs = s64.get("max_sample_tree_pairs_difference", 0)
        pair_txt = f"，每个评估最多差 {pairs} 个（样本、树）对" if pairs else ""
        a("2. 唯一对数值细节敏感的地方是“输入恰好等于门限”的（样本、界）对：在 unif 条件的 t0 与 ideal 点，有效门限与 θ 只差舍入误差，"
          "结果由舍入的符号决定。用与主运行相同的二分约定（40 步，由 C2 数值逐位确认）时完全一致；用 64 步时精度"
          f"{acc_txt}，叶子丢失率有 {s64['leaf_loss_differing']} 个评估不同（最大差 {s64['max_abs_leaf_loss_difference']:.2e}"
          f"{pair_txt}）。这是测度为零的边界情形，不影响任何结论，但若今后改动反查精度，这些点的叶子丢失率会随之变化。")
    else:
        a("2. 主运行的二分约定（40 步、取中点）由 C2 数值逐位确认。")
    rr = t2["ratio_indep_to_pred_range"]
    a(f"3. V1 的 6 个越过时间与主运行的最大相对差 {t2['max_abs_rel_diff_vs_main']:.1e}（`meas` 打印精度为 7 位有效数字），"
      f"t_s(30) 相对差 {t2['t_s_rel_diff']:.1e}；倒数时间相加规则对这 6 例的预测误差为 {100 * (rr[0] - 1):+.2f} % 到 {100 * (rr[1] - 1):+.2f} %。")
    pt_ = t3["per_type"]
    inv_mx = max(pt_[t]["max_abs_diff_uV_inst1_20"] for t in ("UINV", "LBUF", "LBUF_Wstar"))
    a(f"4. 翻转点：L6T 最大差 {pt_['L6T']['max_abs_diff_uV_inst1_20']:.3f} µV；反相器相关的量最大差 {inv_mx:.2f} µV，"
      "来源是主运行反相器扫描的 0.5 mV 网格线性插值，已用同一组 SPICE 数据复现。")
    a("5. 本复核另外读了 SPICE 运行目录中的 `run_info.json`、`run.log`、`pool_stats.csv`（只含数字，不含代码），"
      "brief 列出的文件之外未读其他文件。")
    a("")
    a("## 6 文件")
    a("")
    a(f"本目录（`{info['run_dir']}/`，新建）：")
    a("")
    for f_ in main_files:
        a(f"- `{f_}`")
    if spice_files:
        dirs = sorted({f_.split('/')[0] for f_ in spice_files})
        a(f"- SPICE 网表、日志与原始数据：{'、'.join(f'`{d}/`' for d in dirs)}（共 {len(spice_files)} 个文件）")
    a("")
    a("复核代码（`experiments/E4_independent/`，新建）：`README.md`、`e4i_core.py`、`e4i_spice.py`、`task1_forest.py`、"
      "`task2_v1.py`、`task3_switching.py`、`run_check.py`、`make_report.py`。")
    (out_dir / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    d = Path(sys.argv[1]).resolve()
    with open(d / "compare.json") as f:
        cmp_ = json.load(f)
    with open(d / "run_info.json") as f:
        info_ = json.load(f)
    write(d, cmp_, info_)
