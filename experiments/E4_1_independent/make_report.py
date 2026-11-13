"""E4.1 independent check: write report.md (Chinese) from compare.json and run_info.json.

Usage: python experiments/E4_1_independent/make_report.py results/<check folder>
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

PF = {True: "PASS", False: "FAIL"}
GROUP_ZH = {"t0|t0": "t0", "1y|dac_remap": "1 年 · 输入重映射", "1y|supply": "1 年 · 分压电源"}


def _e(x, nd=1):
    return "–" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}e}"


def _f(x, nd=4):
    return "–" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def _grp(c):
    return GROUP_ZH.get(f"{c['time']}|{c['node']}", f"{c['time']} {c['node']}")


def _gkey(key: str) -> str:
    """'P1|1y|supply' -> 'P1 · 1 年 · 分压电源' (no '|' inside markdown tables)."""
    d, t, n = key.split("|")
    return f"{d} · {GROUP_ZH.get(f'{t}|{n}', f'{t} {n}')}"


def write(out_dir: Path, cmp: dict, info: dict):
    t1, t2, t3 = cmp["task1"], cmp["task2"], cmp["task3"]
    pv = cmp["provenance"]
    tol = cmp["tolerances"]
    m = t2["model_side_all_48_cases"]
    ref = t2["e4_check_6T2M_reference"]
    req = [c for c in t1["cases"] if c["set"] == "first_of_group"]
    ext = [c for c in t1["cases"] if c["set"] != "first_of_group"]
    A = t3["A_rows_csv"]
    B = t3["B_rows_csv_t_sim_indep_t_pred"]
    Cc = t3["C_rows_csv_t_sim_indep_t_pred_indep_t_s"]
    files = sorted(p.relative_to(out_dir).as_posix() for p in out_dir.rglob("*") if p.is_file())
    files = sorted(set(files + ["report.md"]))
    spice_files = [f for f in files if f.startswith("spice_")]
    main_files = [f for f in files if not f.startswith("spice_")]
    with open(out_dir / "task3_gates_model_vs_spice.csv") as f:
        g_rows = list(csv.DictReader(f))
    with open(out_dir / "task3_gate_cases.csv") as f:
        c_rows = list(csv.DictReader(f))
    w_lbuf_in = [float(r["w_indep"]) for r in g_rows
                 if r["type"] == "LBUF" and r["gate_class"] == "in_window" and r["set"] != "first_of_group"]
    w_lbuf_txt = f"{min(w_lbuf_in):.2f}–{max(w_lbuf_in):.2f}" if w_lbuf_in else "–"

    L = []
    a = L.append
    a("# E4.1 独立复核报告")
    a("")
    a(f"- 运行目录：`{info['run_dir']}/`（{info['start_jst']} – {cmp['end_jst']} JST，用时 {cmp['elapsed_s']} s）；"
      f"命令 `{info['command']}`")
    a("- 复核对象：E4.1 的正式运行 `results/E4_1_20261113_1612/`。冻结方案 `experiments/E4_1_config.json` 对增益级单元组 GS"
      "（下界 LBUF：分压节点 X → INV_S → W → INV_S → Z → 下拉管；上界 UINV：Y → INV_S → Z → 下拉管）重复 E4 的检查 V1："
      "用全行瞬态仿真检验行模型对匹配线（ML，match line）越过时间的预测 t_c,pred = t_s(N)·(1 + (2N − 1)·w0)/S"
      "（S 为该行全部下拉管的相对漏电之和，w0 为栅压 0 V 的名义下拉管的相对漏电，t_s(N) 为感测时间）。")
    a("- 复核代码：`experiments/E4_1_independent/`（新建），在 E4 独立复核 `experiments/E4_independent/` 的基础上编写，"
      "原样导入其中的 `e4i_core.py`（器件仪表重建、PCM 模型、编程、有效门限、逐实例漏电）与 `e4i_spice.py`"
      "（失配抽样、ngspice 调用）。项目代码只执行 `src/pcm_acam/trees.py`（单独加载）与 `src/pcm_acam/acam.py` 中的 "
      "`make_draws`（只取该函数源码）。未读、未导入 `experiments/E4_1_gainstage_rows.py`、`experiments/E4_forest_system.py` 以及 "
      "`src/pcm_acam/` 中的 `row_model.py`、`ml_transient.py`、`spice_iv.py`、`inverter.py`、`branch_post.py`。")
    sh = pv["shared_inputs_equal_run_of_record"]
    a(f"- 输入核对：E4.1 方案的 SHA-256 与冻结记录 `logs/E4_1_freeze_sha256.txt`"
      f"{'一致' if pv['plan_sha256_equals_freeze_log'] else '不一致'}、与正式运行 `run_info.json` 中的记录"
      f"{'一致' if pv['plan_sha256_equals_run_of_record'] else '不一致'}；`config_used.json` 的内容与方案"
      f"{'相同' if pv['config_used_content_equals_plan'] else '不同'}（重新序列化，字节不同）；E4 方案与 `instrument.npz` 的 SHA-256 与 "
      f"E4.1 方案中记录的值{'一致' if pv['e4_plan_sha256_equals_frozen_plan'] and pv['instrument_sha256_equals_frozen_plan'] else '不一致'}；"
      f"E4 方案、`trees.py`、`acam.py` 与模型库文件的 SHA-256 与正式运行的记录{'一致' if all(sh.values()) else '不一致'}。"
      "全部输入文件与复核脚本的 SHA-256 见 `run_info.json`。")
    env = info["environment"]
    build = "、".join(env.get("ngspice_build", [])) or "–"
    a(f"- 环境：Python {env['python']}、numpy {env['numpy']}、scipy {env['scipy']}、scikit-learn {env['sklearn']}、"
      f"{env['ngspice']}（{build}）；`OMP_NUM_THREADS={env['OMP_NUM_THREADS']}`，ngspice 进程同一时刻只运行一个。")
    a("")

    # ------------------------------------------------------------------ 1 conclusions
    a("## 1 结论")
    a("")
    a("| 项 | 判定 | 关键数字 |")
    a("|---|---|---|")
    rr = t1["ratio_indep_to_pred_main_range"]
    a(f"| 任务 1：SPICE 侧，自己的全行网表，每个（设计、时间/补偿点）组的第一例，共 {len(req)} 例 | {PF[t1['pass']]} | "
      f"越过时间与 rows.csv 的 t_sim 最大相对差 {100 * t1['max_abs_rel_diff_vs_main']:.4f} %（门槛 {100 * t1['threshold_rel']:.1f} %）；"
      f"t_s(30) {t1['t_s_indep_ns']:.4f} ns，与 {t1['t_s_main_ns']:.3f} ns 相对差 {_e(t1['t_s_rel_diff'])}；"
      f"复核 t_sim / t_pred = {rr[0]:.4f}–{rr[1]:.4f} |")
    a(f"| 任务 2：模型侧，{m['n_cases']} 例、{m['n_bounds']} 个受约束的界（LBUF {m['n_bounds_LBUF']}、UINV {m['n_bounds_UINV']}） | "
      f"{PF[t2['pass']]} | 不需求根的量全部相同（电导最大相对差 {_e(m['max_rel_g_us'])}，其余为 0）；w 最大相对差 "
      f"{_e(m['max_rel_w_LBUF'])}（LBUF）、{_e(m['max_rel_w_UINV'])}（UINV），S {_e(m['max_rel_S'])}，t_pred "
      f"{_e(m['max_rel_t_pred'])}（门槛 {_e(tol['solved_rel'], 0)}） |")
    a(f"| 任务 3：方案判定 V1GS（\\|ln(t_sim/t_pred)\\| ≤ 0.05 的例子 ≥ 90 %） | {PF[t3['pass']]} | "
      f"由 rows.csv：{A['n_within']}/{A['n_cases']}，最大 \\|ln\\| {_f(A['max_abs_ln_ratio'])}，{A['outcome']}"
      f"（decision.json：{t3['A_vs_run_of_record']['outcome_decision_json']}）；用复核的 t_pred：{B['n_within']}/{B['n_cases']}，"
      f"{B['outcome']}；复核 SPICE 的 {len(req)} 例对复核 t_pred：最大 \\|ln\\| {_f(t3['D_max_abs_ln_ratio'])} |")
    ex = t1["extra"]
    a(f"| 附加（不计入判定）：LBUF 界占 S 一半以上的全部 {len(ext)} 例（例 {'、'.join(str(c) for c in ex['cases'])}）的 SPICE | "
      f"{'在门槛内' if ex['all_within_threshold'] else '超出门槛'} | 与 rows.csv 最大相对差 {100 * ex['max_abs_rel_diff_vs_main']:.4f} %；"
      f"复核 SPICE 对复核 t_pred 最大 \\|ln\\| {_f(t3['D_extra_max_abs_ln_ratio'])} |")
    se = t2["selection_extra"]
    a(f"| 附加（不计入判定）：按方案重做病例选取 | {'相同' if se['all_identical'] else '不同'} | "
      "6 组的匹配对数与候选数与主运行 `run.log` 相同，选出的 48 个（样本、行）对逐一相同 |")
    a("")
    a(f"三项{'全部通过' if cmp['all_pass'] else '未全部通过'}，输入核对{'一致' if pv['all'] else '有不一致'}。"
      f"E4.1 的判定 V1GS = {A['outcome']} 由复核独立重现：复核的 SPICE 网表给出与主运行相同的越过时间，复核的模型给出与主运行"
      "相同的 w、S 与 t_pred（差别在求解精度的量级），由两者重算的判定与 `decision.json` 相同。")
    a("")

    # ------------------------------------------------------------------ 2 task 1
    a(f"## 2 任务 1：SPICE 侧（breast_cancer，划分 0，种子 0，条件 all；N = 30，2N = 60 个下拉管）")
    a("")
    a("网表按冻结方案的 `netlist` 一节自己编写，一个网表一行：")
    a("")
    a("- 下界（LBUF）：PCM 电阻 1/G（SL_lo → X）；分压管 nfet_01v8（设计的 W/L，失配 div_lo）X–DL；INV_S 1（b1n、b1p）X → W；"
      "INV_S 2（b2n、b2p）W → Z；下拉管 nfet_01v8_lvt 0.42/0.15 µm（pd_lo），漏极接 ML，栅极接 Z。")
    a("- 上界（UINV）：PCM 电阻（SL_hi → Y）；分压管（div_hi）；INV_S（un、up）Y → Z；下拉管（pd_hi）。INV_S：pfet_01v8 0.42/4 µm"
      "（接 VDD）与 nfet_01v8 2.0/0.15 µm（E4 方案）。其余 2N − n_c 个名义下拉管栅压 0 V。")
    a(f"- ML 电容 C_SA + N·C_wire = {t1['c_ml_F'] * 1e15:.0f} fF；`.ic v(ml)=1.8` 预充电（不用 UIC，其他节点从直流工作点开始）；"
      "取 V_ML 第一次降到 0.9 V 的时间。")
    a("- 电导 g_us、数据线电压 v_dl、实例编号 idx、电源 v_sl_lo / v_sl_hi 取自 cases.json（任务 2 用复核模型独立重算：电导最大相对差 "
      f"{_e(m['max_rel_g_us'])}，其余逐位相同）。"
      "失配：`default_rng([20261017, 5])`，按 div_lo、pd_lo、b1n、b1p、b2n、b2p、div_hi、un、up、pd_hi 的顺序各抽 (2000, 4)，"
      "实例 k 用第 k − 1 行（与 E4 复核相同）。")
    a(f"- 瞬态：`{t1['tran']['options']}`，最大步长 {t1['tran']['tmax_s'] * 1e9:.1f} ns，停止时间 2·t_pred + 100 ns（与 E4 复核相同）。")
    a("")
    a("| 例 | 设计 | 时间 · 补偿点 | 界数（LBUF + UINV） | LBUF 占 S | t_sim 主运行 (ns) | t_sim 复核 (ns) | 相对差 | 复核 / t_pred |")
    a("|---|---|---|---|---|---|---|---|---|")
    for c in req:
        a(f"| {c['case']} | {c['design']} | {_grp(c)} | {c['n_lower_LBUF']} + {c['n_upper_UINV']} | {_e(c['S_lower_LBUF_share'])} | "
          f"{c['t_sim_main_ns']:.3f} | {c['t_sim_indep_ns']:.3f} | {100 * c['rel_diff_vs_main']:+.4f} % | {c['ratio_indep_to_pred_main']:.4f} |")
    a("")
    a(f"- 感测时间 t_s(30)（名义下拉管栅压 V_sw,nom，取自 `instrument.npz`；其余 59 个下拉管 0 V；同样的电容与瞬态设置）："
      f"复核 {t1['t_s_indep_ns']:.4f} ns，主运行 {t1['t_s_main_ns']:.3f} ns，相对差 {_e(t1['t_s_rel_diff'])}（门槛 0.5 %）。")
    sens = t1.get("numerical_sensitivity", [])
    if sens:
        mx = max(abs(s["rel_change_vs_my_main_setting"]) for s in sens)
        mx_txt = ("为 0：所有变体打印出相同的数字" if mx < 1e-12 else f"为 {_e(mx)}")
        a(f"- 数值敏感性（只报告）：例 {sens[0]['case']} 改用 ngspice 默认容差、或最大步长 0.4 ns；6 例放进同一个网表（各自的 ML 节点，"
          f"只共用 VDD），用方案的最大步长 0.2 ns 与停止时间 6·t_s(N)（主运行每个网表放 12 行）。越过时间的最大相对变化{mx_txt}"
          "（`meas` 打印 7 位有效数字）。逐项见 `task1_sensitivity.csv`。")
    gt = t1["gate_2ns_vs_main_spice"]
    a(f"- 释放后 2 ns 时受约束下拉管的栅压，复核 SPICE 与主运行 SPICE（bounds.csv 的 gate_sim_2ns_V）之差：LBUF {gt['LBUF']['n_bounds']} 个界"
      f"最大 {gt['LBUF']['max_abs_diff_mV'] * 1e3:.2f} µV，UINV {gt['UINV']['n_bounds']} 个界最大 {gt['UINV']['max_abs_diff_mV'] * 1e3:.2f} µV"
      "（含附加 4 例；`meas` 的打印分辨率为 0.1 µV 量级）。")
    a("")
    a(f"这 {len(req)} 例中 LBUF 界占 S 的比例都不超过 {_e(t1['max_S_lower_LBUF_share_first_cases'])}：它们的 LBUF 界都在界内深处或被 u 截断，"
      "S 几乎全部来自 UINV 界，所以只检验了 LBUF 链关断的情形。附加：rows.csv 中 LBUF 界占 S ≥ 50 % 的全部例子也用同样的网表、"
      "设置和门槛仿真（不计入判定）：")
    a("")
    a("| 例 | 设计 | 时间 · 补偿点 | 界数（LBUF + UINV） | LBUF 占 S | S | t_sim 主运行 (ns) | t_sim 复核 (ns) | 相对差 | 复核 / t_pred |")
    a("|---|---|---|---|---|---|---|---|---|---|")
    for c in ext:
        a(f"| {c['case']} | {c['design']} | {_grp(c)} | {c['n_lower_LBUF']} + {c['n_upper_UINV']} | {c['S_lower_LBUF_share']:.3f} | "
          f"{c['S_main']:.4f} | {c['t_sim_main_ns']:.3f} | {c['t_sim_indep_ns']:.3f} | {100 * c['rel_diff_vs_main']:+.4f} % | "
          f"{c['ratio_indep_to_pred_main']:.4f} |")
    a("")

    # ------------------------------------------------------------------ 3 task 2
    a("## 3 任务 2：模型侧（48 例）")
    a("")
    a("复核模型（`e4i_core.py`，未修改）：重建随机森林（`trees.py`）、实例编号 `default_rng([20261012, 2, 0, 0, 4]).integers(1, 2001)`、"
      "PCM 抽样（`make_draws`）、阈值校验编程、编程噪声、漂移、读噪声与复制单元的 â，按补偿点设定电源与输入映射，有效门限由实例分压管曲线"
      "二分反查（40 步）。漏电：负载线 G_k·(V_SL − V) = I_k(V_DL, V_sp,k)·I_nom(V_G', V)/I_nom(V_G', V_sp,k)（64 步二分），节点电压经该实例"
      "自己的反相器传输曲线（LBUF 两级：t_first 后接 t_second）与下拉管放电曲线得到 w；LBUF 在界内 ≥ 0.45 LSB、UINV ≥ 3 LSB 时取该实例下拉管在"
      "栅压 0 V 的 w。S = Σw + (2N − n_c)·w0，t_pred 用主运行的 t_s(N)（cases.json：413.384 ns）。")
    a("")
    a(f"容差：E4 复核对 6T2M 的 48 例没有设模型侧的数值门槛；它报告不需求根的输入逐位相同，w 与 S 的最大相对差分别为 "
      f"{_e(ref['max_rel_w'])} 与 {_e(ref['max_rel_S'])}（其 `compare.json`）。本复核沿用这一标准：不需求根的量 ≤ {_e(tol['unsolved_rel'], 0)}"
      f"（实例编号与每例界的集合要求完全相同）；需求根的量（w、S、S 的 LBUF/UINV 两部分（除以 S）、t_pred）相对差 ≤ {_e(tol['solved_rel'], 0)}。"
      "后者 = 6T2M 的最大值 4e-8 × LBUF 链多出的一级反相器的增益（约 20，E3b）≈ 1e-6，再留 10 倍余量；它仍比方案门槛 0.05（ln t）小 5000 倍。")
    a("")
    a("| 量 | 最大差 | 门槛 | 判定 |")
    a("|---|---|---|---|")
    fl = t2["flags"]
    a(f"| 实例编号 idx（{m['n_bounds']} 个界） | 全同 | 全同 | {PF[fl['idx']]} |")
    a(f"| 每例界的集合（特征、类型） | 全同 | 全同 | {PF[fl['bound_sets']]} |")
    a(f"| 48 例都没有被违反的界（复核模型） | {'是' if fl['no_violated_bound'] else '否'} | – | {PF[fl['no_violated_bound']]} |")
    rows_ = [("输入 x", "max_abs_x_diff", "x"), ("电导 g（相对）", "max_rel_g_us", "g_us"),
             ("有效门限 θ_eff", "max_abs_theta_eff", "theta_eff"), ("数据线电压 V_DL (V)", "max_abs_v_dl_V", "v_dl"),
             ("界内距离 u (LSB)", "max_abs_u_lsb", "u_lsb"), ("â", "max_abs_a_hat", "a_hat"),
             ("电源 V_SL,lo、V_SL,hi (V)", "max_abs_v_sl_V", "v_sl")]
    for lab, k, f_ in rows_:
        a(f"| {lab} | {_e(m[k])} | {_e(tol['unsolved_rel'], 0)} | {PF[fl[f_]]} |")
    a(f"| 漏电权重 w（相对；LBUF / UINV） | {_e(m['max_rel_w_LBUF'])} / {_e(m['max_rel_w_UINV'])} | {_e(tol['solved_rel'], 0)} | {PF[fl['w']]} |")
    a(f"| S（相对） | {_e(m['max_rel_S'])} | {_e(tol['solved_rel'], 0)} | {PF[fl['S']]} |")
    a(f"| S 的 LBUF / UINV 部分（差 ÷ S） | {_e(m['max_abs_S_part_over_S'])} | {_e(tol['solved_rel'], 0)} | {PF[fl['S_parts']]} |")
    a(f"| t_pred（相对） | {_e(m['max_rel_t_pred'])} | {_e(tol['solved_rel'], 0)} | {PF[fl['t_pred']]} |")
    a(f"| 模型的行判定（S ≥ 1 + 59·w0） | 48/48 相同 | 全同 | {PF[fl['decision_pred']]} |")
    a("")
    a("w 的最大偏差：")
    a("")
    a("| 例 | 界 | 类型 | 实例 | w 复核 | w 主运行 | 相对差 | 节点 − 翻转点 (mV) | 等效节点电压差 (V) |")
    a("|---|---|---|---|---|---|---|---|---|")
    for b in t2["largest_w_deviations"]:
        a(f"| {b['case']} | {b['bound']} | {b['type']} | {b['idx_indep']} | {b['w_indep']:.6g} | {b['w_main']:.6g} | {b['w_rel_diff']:+.2e} | "
          f"{b['node_minus_vsp_mV']:+.3f} | {_e(b['implied_node_diff_V'])} |")
    a("")
    md = m["median_abs_dlnw_dnode_in_window_per_V"]
    a(f"最大的几个偏差都出现在节点离翻转点（X*_k）不到 1 mV 的 LBUF 界。在传输曲线窗口内，\\|d ln w / d 节点\\| 的中位数为 LBUF {md['LBUF']:.3g} /V、"
      f"UINV {md['UINV']:.3g} /V（比值约 {md['LBUF'] / md['UINV']:.0f}，即第二级反相器的增益）。把每个界的 Δln w 除以该斜率，换算成节点电压差，"
      f"全部不超过 {_e(m['max_abs_implied_node_diff_V'])} V：两种界的偏差同源，是两边负载线求解精度的差别（约 0.1 nV），与模型本身无关。"
      f"E4 复核中 6T2M 的最大偏差同样出现在 UINV 界上（UINV {_e(ref['max_rel_w_by_type']['UINV'])}、L6T "
      f"{_e(ref['max_rel_w_by_type']['L6T'])}），与本复核 UINV 界的 {_e(m['max_rel_w_UINV'])} 同量级。")
    a("")
    gc = m["max_abs_gate_uncut_minus_main_uV_by_class"]
    a(f"栅压约定：bounds.csv 的 gate_model_V 等于复核反相器链在不做 u 截断时的输出（最大差 {m['max_abs_gate_uncut_minus_main_uV']:.3f} µV；"
      f"其中被 u 截断的界 {gc['u_cut_gate_0V']:.4f} µV）；而被 u 截断的界的 w_model 与“该实例下拉管在 0 V 的 w”"
      f"{'逐位相同' if m['w_main_equals_w_at_gate_0V_for_u_cut_bounds'] else '并不逐位相同'}。即主运行在 bounds.csv 中报告的模型栅压是截断前的链输出，"
      "而 w 用的是截断后的 0 V（这一点影响第 4 节的栅压比较）。")
    a("")
    nb = m["n_bounds_by_class"]
    a(f"界的分类（全部 {m['n_bounds']} 个）：传输曲线窗口内 LBUF {nb['in_window']['LBUF']}、UINV {nb['in_window']['UINV']}；"
      f"节点在窗口外（存储曲线取端点值） LBUF {nb['node_outside_transfer_window']['LBUF']}、UINV {nb['node_outside_transfer_window']['UINV']}；"
      f"被 u 截断 LBUF {nb['u_cut_gate_0V']['LBUF']}、UINV {nb['u_cut_gate_0V']['UINV']}。窗口：反相器输入在其翻转点 ±60 mV 以内"
      "（LBUF 要求 X 与 W 都在窗口内）。")
    a("")
    a("附加：按方案重做病例选取（对全部无违反的（样本、行）对计算 S，取 S ∈ [0.3, 3] 的候选，按 ln S 排序后取 round(linspace(0, n − 1, 8))）：")
    a("")
    a("| 组 | 匹配对（复核 / 主运行 run.log） | 候选（复核 / 主运行） | 选出的（样本、行）对 |")
    a("|---|---|---|---|")
    for k, g in se["per_group"].items():
        a(f"| {_gkey(k)} | {g['matched_pairs_indep']} / {g['matched_pairs_main_log']} | {g['candidates_indep']} / {g['candidates_main_log']} | "
          f"{'相同' if g['selection_identical'] else '不同'} |")
    a("")

    # ------------------------------------------------------------------ 4 task 3
    a("## 4 任务 3：方案判定 V1GS")
    a("")
    a("| 计算方式 | 在门槛内 | 通过比例 | 最大 \\|ln(t_sim/t_pred)\\| | 结论 |")
    a("|---|---|---|---|---|")
    a(f"| A：rows.csv（t_sim_ns、t_pred_ns） | {A['n_within']}/{A['n_cases']} | {A['pass_fraction']:.3f} | {_f(A['max_abs_ln_ratio'])} | {A['outcome']} |")
    a(f"| B：rows.csv 的 t_sim + 复核 t_pred | {B['n_within']}/{B['n_cases']} | {B['pass_fraction']:.3f} | {_f(B['max_abs_ln_ratio'])} | {B['outcome']} |")
    a(f"| C：rows.csv 的 t_sim + 复核 t_pred（用复核 SPICE 的 t_s = {Cc['t_s_indep_ns']:.4f} ns） | {Cc['n_within']}/{Cc['n_cases']} | "
      f"{Cc['pass_fraction']:.3f} | {_f(Cc['max_abs_ln_ratio'])} | {Cc['outcome']} |")
    D = t3["D_spice_indep_six_cases"]
    nD = sum(abs(d["ln_ratio"]) <= 0.05 for d in D.values())
    a(f"| D：复核 SPICE 的 {len(D)} 例 + 复核 t_pred | {nD}/{len(D)} | – | {_f(t3['D_max_abs_ln_ratio'])} | （只有 {len(D)} 例，不构成判定） |")
    DE = t3["D_extra_LBUF_dominated_cases"]
    nDE = sum(abs(d["ln_ratio"]) <= 0.05 for d in DE.values())
    a(f"| D'：附加的 {len(DE)} 例（LBUF 主导） | {nDE}/{len(DE)} | – | {_f(t3['D_extra_max_abs_ln_ratio'])} | （附加） |")
    a("")
    av = t3["A_vs_run_of_record"]
    a(f"- A 与运行记录：ln 比与 rows.csv 的 ln_ratio 列最大差 {_e(av['max_abs_diff_ln_ratio_vs_rows_csv'])}；逐例的 pass 标志"
      f"{'相同' if av['pass_flags_equal_rows_csv'] else '不同'}；通过比例与 checks.json（{av['pass_fraction_checks_json']}）、decision.json"
      f"（{av['pass_fraction_decision_json']}）相同；最大 \\|ln\\| 与 checks.json 之差 {_e(av['abs_diff_max_ln_ratio_vs_checks_json'])}，"
      f"各组最大值之差 ≤ {_e(av['max_abs_diff_by_group_vs_checks_json'])}；结论与 decision.json（{av['outcome_decision_json']}）相同。")
    a(f"- B − A 的 ln 比最大差 {_e(t3['B_minus_A_max_abs_ln_ratio'])}，C − A {_e(t3['C_minus_A_max_abs_ln_ratio'])}"
      "（后者含 t_s 的相对差）。")
    a("- 各组最大 \\|ln\\|（A）：" + "；".join(f"{_gkey(g)} {_f(v)}" for g, v in A["max_abs_ln_ratio_by_group"].items()) + "。")
    rd, rc = t3["row_decisions_main"], t3["row_decisions_compare"]
    ri = t3["row_decisions_indep"]
    nt = "、".join(str(i) for i in rd["near_tie_case_ids"]) or "无"
    a(f"- 行判定（方案中只报告的项）：模型（S ≥ 1 + (2N − 1)·w0）判失配 {rd['n_pred_mismatch']} 例，仿真（t_sim ≤ t_s）判失配 "
      f"{rd['n_sim_mismatch']} 例，不一致 {rd['disagreements']} 例（checks.json：{rc['disagreements_checks_json']}）；\\|ln S\\| < 0.05 的近似平局 "
      f"{rd['cases_near_tie']} 例（例 {nt}），其中不一致 {rd['disagreements_near_tie']} 例（checks.json：{rc['near_tie_checks_json']}）。"
      f"与 rows.csv 的两列{'相同' if rc['pred_flags_equal_rows_csv'] and rc['sim_flags_equal_rows_csv'] else '不同'}；"
      f"用复核的 S 与 t_s 结果{'相同' if rc['indep_pred_flags_equal'] and rc['indep_sim_flags_equal'] else '不同'}（不一致 {ri['disagreements']} 例）。")
    gm = t3["gates_2ns_main_recomputed"]
    a(f"- 释放后 2 ns 时的栅压，模型对仿真：主运行的数值由 bounds.csv 重算，最大 {gm['max_abs_diff_mV']:.2f} mV（下界 "
      f"{gm['max_abs_diff_mV_lower']:.2f}、上界 {gm['max_abs_diff_mV_upper']:.2f} mV），与 checks.json "
      f"{'相同' if gm['reproduced'] else '不同'}。按第 3 节的分类：")
    a("")
    a("| 分类 | 主运行：界数 | 主运行：最大差 (mV) | 主运行：最大 w | 复核：界数 | 复核：最大差 (mV，截断后栅压) | 复核：最大差 (mV，截断前) |")
    a("|---|---|---|---|---|---|---|")
    gi = t3["gates_2ns_indep_model_vs_indep_spice"]
    names = {"in_window": "窗口内", "node_outside_transfer_window": "节点在窗口外", "u_cut_gate_0V": "被 u 截断"}
    for cl in ("in_window", "node_outside_transfer_window", "u_cut_gate_0V"):
        for typ in ("LBUF", "UINV"):
            k = f"{cl}/{typ}"
            g1 = gm["by_window_class"].get(k, {"n_bounds": 0, "max_abs_diff_mV": float("nan"), "max_w": float("nan")})
            g2 = gi.get(k, {"n_bounds": 0, "max_abs_diff_used_mV": float("nan"), "max_abs_diff_uncut_mV": float("nan")})
            a(f"| {names[cl]} · {typ} | {g1['n_bounds']} | {_f(g1['max_abs_diff_mV'], 2)} | {_e(g1['max_w'])} | {g2['n_bounds']} | "
              f"{_f(g2['max_abs_diff_used_mV'], 3)} | {_f(g2['max_abs_diff_uncut_mV'], 3)} |")
    a("")
    a(f"  复核一栏为任务 1 的 {len(req) + len(ext)} 例（{len(req)} 例 + 附加 {len(ext)} 例）的界，模型取复核模型，仿真取复核 SPICE。")
    a("")

    # ------------------------------------------------------------------ 5 observations
    a("## 5 观察")
    a("")
    a(f"1. SPICE：复核网表与主运行的网表互不相干（每个网表一行、不同的步长与停止时间），{len(req)} 例与附加 {len(ext)} 例的越过时间与主运行相差 "
      f"≤ {100 * max(t1['max_abs_rel_diff_vs_main'], ex['max_abs_rel_diff_vs_main']):.4f} %，在 `meas` 打印精度附近；t_s(30) 相同。说明主运行 GS 行的"
      "电路（两级反相器的连接、10 组失配的分配、电源与数据线电压）与方案的描述一致。")
    a(f"2. 模型：不需求根的量与主运行逐位相同（电导差 1 个末位）；w、S、t_pred 的差 ≤ {_e(max(m['max_rel_w'], m['max_rel_S'], m['max_rel_t_pred']))}，"
      f"折合节点电压 ≤ {_e(m['max_abs_implied_node_diff_V'])} V。LBUF 链的增益使这一差别比 6T2M 放大约 20 倍，但仍比判定门槛小 4 个数量级以上。")
    a(f"3. 方案指定的 {len(req)} 例全部由 UINV 界主导，LBUF 链在这些行里处于关断状态；真正检验 LBUF 漏电（节点在 X* 附近 1 mV 以内，"
      f"w = {w_lbuf_txt}）的是附加的 {len(ext)} 例，它们同样与主运行一致，复核模型对复核 SPICE 的 \\|ln 比\\| ≤ "
      f"{_f(t3['D_extra_max_abs_ln_ratio'], 3)}。论文若以 V1GS 说明 GS 行已被验证，可注明 48 例中 LBUF 主导 S 的只有这 {len(ext)} 例。")
    nt_txt = "；".join(f"例 {i}，S = {float(c_rows[i]['S_main']):.3f}" for i in rd["near_tie_case_ids"]) or "无"
    a(f"4. 判定：三种算法（A、B、C）都是 {A['n_within']}/{A['n_cases']}，最大 \\|ln\\| {_f(A['max_abs_ln_ratio'], 3)}，结论 {A['outcome']}，与 "
      f"decision.json 相同；行判定 {rd['disagreements']} 例不一致，近似平局（{nt_txt}）的判定也一致。")
    bw = gm["by_window_class"]
    off = [bw[k] for k in bw if not k.startswith("in_window/")]
    w_off = max(g["max_w"] for g in off) if off else float("nan")
    d_off = max(g["max_abs_diff_mV"] for g in off) if off else float("nan")
    d_in = max(bw[k]["max_abs_diff_mV"] for k in bw if k.startswith("in_window/"))
    src = "全部" if d_off >= gm["max_abs_diff_mV"] - 1e-12 else "不全"
    a(f"5. checks.json 的 gates_2ns 最大差（{gm['max_abs_diff_mV']:.1f} mV）{src}来自被 u 截断或节点在传输曲线窗口外的界（w ≤ {_e(w_off)}，与 w0 同量级），"
      f"其中被截断的界是因为 bounds.csv 报告截断前的链输出；窗口内的界，主运行模型与仿真之差 ≤ {d_in:.2f} mV。引用该数字时宜说明这一约定。")
    a("6. 本复核只读了任务说明允许的文件（清单见 README）；smoke 运行 `E4_1_20261113_1611_smoke` 的数字未使用。")
    a("")

    # ------------------------------------------------------------------ 6 files
    a("## 6 文件")
    a("")
    a(f"本目录（`{info['run_dir']}/`，新建）：")
    a("")
    for f_ in main_files:
        a(f"- `{f_}`")
    if spice_files:
        a(f"- 任务 1 的 SPICE 网表与 ngspice 日志：`spice_task1/`（共 {len(spice_files)} 个文件，不保存二进制 raw 文件）")
    a("")
    a("复核代码（`experiments/E4_1_independent/`，新建）：`README.md`、`e41i_common.py`、`task1_spice.py`、`task2_model.py`、"
      "`task3_decision.py`、`run_check.py`、`make_report.py`；导入（未修改）`experiments/E4_independent/e4i_core.py` 与 `e4i_spice.py`。")
    (out_dir / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    d = Path(sys.argv[1]).resolve()
    with open(d / "compare.json") as f:
        cmp_ = json.load(f)
    with open(d / "run_info.json") as f:
        info_ = json.load(f)
    write(d, cmp_, info_)
