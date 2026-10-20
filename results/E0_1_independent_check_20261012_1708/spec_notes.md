# E0.1 独立复算：规格歧义与处理记录（spec notes）

- 生成时间：[2026-10-12 17:08] JST
- 依据：`docs/E0_1_protocol.md`（冻结版）、`experiments/E0_1_config.json`（plan 1.1，SHA-256 `44d1d458564e7a36f32d67a24b7686a674fb4702b9b54353c0ca06232dcff25a`）、复核任务说明。未读取项目源码、项目脚本或其他结果目录。
- 复算脚本 SHA-256：`46158c19adcc6d2ceb1434e8c6830e03866ff99e6b3d6d740ba01f1bcf7a0d89`；配置常数与任务说明逐项断言一致（全部一致）。

## 1 数据与归一化

1. **归一化的浮点算式未规定**。本实现：span = max − min（训练集），span = 0 → 1；x_n = (x − min)/span；训练集不截断（构造上已在 [0,1]）；测试集 `np.clip` 到 [0,1]。sklearn `MinMaxScaler` 的算式 x·scale + min_ 与之可差 1 ulp，只在测试值恰好等于量化门限时改变比较结果。诊断：用 MinMaxScaler 重跑，森林全部逐位相同，测试特征有 49700 个值差 ≥1 ulp；理想 4-bit CAM 中测试值与内部门限（0<θ_q<1）恰好相等的（样本, 约束）对共 19100 个（iris/split0: 60, iris/split1: 96, wine/split0: 10, wine/split2: 12, breast_cancer/split0: 2, breast_cancer/split1: 3, digits/split0: 11260, digits/split1: 4388, digits/split2: 3269），相差在 1e-12 以内但不相等的对数 = 155；acc_ref 在两种算式下完全相同。
   θ_q ∈ {0, 1} 处的恰好相等共 447370 个（主要是 digits 的 0 像素与 θ_q = 0），这类相等在任何算式下都确定（0 与 1 在两种算式下都精确），按 x ≤ θ 匹配上界、x > θ 不匹配下界处理。
2. 数据以 float64 读入；标签直接用 sklearn 原值（classes_ = 0..C−1）；森林在归一化后的训练集上拟合（sklearn 内部转 float32，门限为两个 float32 值的中点，以 float64 保存）。CAM 与自检遍历都用 float64 特征与 float64 量化门限比较，不做 float32 转换。

## 2 树 → CAM 行

3. 最紧界在原始门限上取（下界取最大、上界取最小）后再量化；因量化单调不减，与先量化后取最紧界等价。
4. 无约束行（整棵树只有一个叶）：永远匹配。本次出现 0 棵单叶树。
5. 量化后不可满足的行（同一特征 L_q ≥ U_q，或 L_q ≥ 1、U_q < 0）照常保留在行序与随机数下标中，理想 CAM 下永不匹配；本次共 286 行。
6. 类别向量 = `tree_.value[leaf,0,:]` 除以其和（sklearn 1.9 已存比例，归一化只在 ulp 级别起作用）。

## 3 量化与随机数

7. round 用 `np.round`（半数取偶）。thr·15 小数部分恰为 0.5 的约束共 113 个，**全部是 thr = 0.5**（thr·15 = 7.5；iris 5 个、digits 108 个）：半数取偶与四舍五入都得 8，两种约定结果相同；只有“向下取整 / 半数向下”才会得 7（θ_q = 7/15 而非 8/15）。门限超出 [0,1] 的个数 = 0。〔[2026-10-12 17:11] 运行后手工更正：脚本模板原句把计数 113 直接接上“因此无影响”，理由不成立；结论经第 9 节的分析确认。〕
8. 随机数严格按 z_prog (K,2) → z_nu (K,2) → z_read (5,K,2) → z_rep_prog (64) → z_rep_nu (64) → z_rep_read (5,64) 的顺序以 `standard_normal` 抽取（C 序），prog 条件也照样全部抽取后只用 z_prog。V1 与 V2 共用同一组 z（第 0 列）与同一组复制单元 z（目标分别 12.5、16.61 µS），即成对比较。
9. 5 个时刻的读噪声是 5 组独立抽样（z_read[t_idx]），不是同一器件随时间的相关序列；编程噪声与漂移指数在各时刻共用。

## 4 器件与方案

10. t = t_inference + 20 s；t > 20 s 才乘漂移因子，t0 时因子 = 1；读噪声 sig 用这个 t（t0 时 t = 20 s），因此 t0 已有读噪声。Q 用 G_prog 计算，乘的是 |G_drift|（照协议原文）。
11. μ、σν 用编程目标 G 计算，g_rel = max(G/25, 1e-7)（下限只影响 G = 0，且被 clip 吸收，不改变数值）。
12. V3：r = 1/3 + θ_q/3（任务说明写法）；与配置的 r_lo + θ_q (r_hi − r_lo) 只差 ≤1 ulp，在连续噪声下无影响。G1+G2 = 0 时比值取 0.5（按任务说明）。
13. replica：â(t) = mean(64 个复制单元在 t 的 G_final) / mean(64 个复制单元的 G_prog)；prog 条件 â = 1，所以 (V1/V2, replica, prog) 与 (V1/V2, none, prog) 完全相同，仍各自输出一行。V3 的 â 记为 1.0。

## 5 匹配、投票与指标

14. 得分求和顺序：每棵树内用 one-hot 匹配向量 × 类别矩阵（通常只有一行匹配，结果精确），再按森林顺序逐棵累加；argmax 并列取最小下标。诊断：理想 CAM 中得分恰好并列的样本 3 个、0 < 差 ≤ 1e-9 的近并列 0 个；全部噪声评估中恰好并列 1161 个、近并列 0 个（近并列的样本其结果可能依赖浮点求和顺序）。敏感度诊断：改成“全部行一次矩阵乘”求和时，理想 CAM 预测变化 0 个、噪声评估预测变化 0 个（精度改变的评估 0 次）；改用未再归一化的 `tree_.value` 时（再归一化改变了 298 行的值），理想 CAM 预测变化 0 个、噪声评估预测变化 0 个（精度改变的评估 0 次）。
15. 所有树都没有匹配行的样本记为错误；噪声评估中这样的样本共 0 个（2250 次评估合计）。路径完整度在全部测试样本上计算（含无匹配样本）。
16. loss 先逐次评估算 100·(acc_ref − acc) 再在 3 分割 × 5 器件种子上平均（与先平均再相减数学上相同）；max_loss 并列时取数据集顺序中的第一个。per_eval.csv 的 t 列写时刻标签（t0/1h/1d/30d/365d），condition 列写 prog/all。

## 6 判定

17. 关口 A 用 track = none 的 prog 行（与 replica 相同）。关口 C 的候选各自用自己的 prog 行判定 A 标准。
18. 关口 B：V1、V2 的 mean_loss(prog) 若相等，规格未说取哪一个；本实现取 V1。本次未出现并列，规则不起作用。
19. 判定阈值按“≤ / ≥”含等号处理（照配置原文）。

## 7 自检

- 理想 4-bit CAM vs 量化门限直接遍历（split 0，全部数据集）：预测不一致 0 个，叶节点不一致或非恰好一行匹配 0 个（iris: 0/45, wine: 0/54, breast_cancer: 0/171, digits: 0/540, synthetic: 0/1500）。
- 理想 CAM 每棵树恰好一行匹配（全部分割）：非 1 的（样本, 树）对数 = iris/split0: 0, iris/split1: 0, iris/split2: 0, wine/split0: 0, wine/split1: 0, wine/split2: 0, breast_cancer/split0: 0, breast_cancer/split1: 0, breast_cancer/split2: 0, digits/split0: 0, digits/split1: 0, digits/split2: 0, synthetic/split0: 0, synthetic/split1: 0, synthetic/split2: 0。
- float64 门限 CAM vs sklearn predict（split 0）：iris: 1.0000, wine: 1.0000, breast_cancer: 1.0000, digits: 1.0000, synthetic: 1.0000。
- 无噪声、σν = 0 时 V2+replica 与 V3 复原 θ_q 的最大误差（split 0）：iris: 4.4e-16, wine: 4.4e-16, breast_cancer: 4.4e-16, digits: 4.4e-16, synthetic: 4.4e-16。
- PCM 公式手算值最大误差：0.0e+00。

## 8 判定结果（本复算）

- gate_A = PASS，gate_B = PASS，gate_C = PASS，verdict = GO。

## 9 运行后手工补充（[2026-10-12 17:11] JST，非脚本生成）

- 分析脚本 `/tmp/claude-0/-home-claude/6dfef7ba-89f2-5845-85d8-aa33235ce2d6/scratchpad/independent_check_acam/post_run_analysis.py`（只读本目录的输出），输出存于同目录 `post_run_analysis.log`。CSV、gates.json 等计算结果未改动；只改了本文件第 7 条并加了本节。
- summary.csv 的所有格子与 per_eval.csv + refs.csv 的精确有理数重算一致（最大差 1.4e-14）。
- V1_abs_full 的 prog max_loss 打印为 −1.48e-15（breast_cancer），精确值为 0（±1 个样本的翻转在浮点下没有完全抵消）。
- 关口余量（精确值）：A：V1 mean_loss(prog) = −5146/10125 = −0.508 pp（≤ 2.0），max = 0（≤ 5.0）；B：取 V1（mean_loss(prog) 较小），all@1d = 61991/3375 = 18.368 pp（≥ 10.0）；C：V1+replica all@30d = 16303/38475 = 0.424 pp，V2+replica = 3.689 pp（≤ 5.0）。三关的余量都远大于单样本粒度，结论 GO 对单样本级别的实现差异稳健。
- **刀口**：V3_ratio all@30d = 10133/2025 = 5.00395 pp，只比 5.0 高 0.00395 pp。iris、wine 或 breast_cancer 任一次评估中只要 1 个测试样本不同（digits 2 个，synthetic 5 个），它就会低于 5.0。因此“V3 是否满足关口 C”在两套实现之间可能不一致，但不影响关口 C 与总结论（V1+replica、V2+replica 已通过）。
- 配置版本：协议文件头写的配置 SHA-256（`83ace66b…`）是 plan 1.0；本次读取的配置为 plan 1.1（SHA-256 `44d1d458…`），与配置内 deviations 的记载一致（只改了检查 2 的比较对象）。本复算的 float64 门限 CAM 与 sklearn predict 100 % 一致，与该记载相符。
