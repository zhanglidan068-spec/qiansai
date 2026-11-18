# W8A8 高低四位渐进乘法实验

本次独立目录：`D:\codex\kwt_fp32\runs\nibble_w8a8_20261030_194829_241778`。本报告从本轮保存的预测和统计汇总生成，不启动模型推理。

筛选 47 项，全量独立配置 17 项，另有最终配置独立重跑 2 次、native 精确实现控制 1 项；控制和重跑均单独标记。

## 全量验证结论

本轮 FP32：**4340/4445＝97.6378%**；完整四项 W8A8：**4338/4445＝97.5928%**。

严格目标为相对本轮 W8A8 额外下降不超过 **0.1 个百分点**，至少正确 **4334 条**；另保留相对 FP32 下降不超过 **2 个百分点**的宽松目标，至少正确 **4252 条**。

严格目标下，已测试全量候选中节省全编码器部分积槽位最多的是 **guarded_all_dense_ratio_8_sll_only**：准确率 **97.5253%**，比 W8A8 额外下降 **0.0675 个百分点**，所选操作省 **26.2108%**，折算到全编码器省 **20.8384%**。

本次推荐的无量纲参数为 **λ＝8**：在 S3＝HH＋HL＋LH 后，取 B3＝15min(wL,aL)，满足 **S3≥8B3** 时允许省略 LL；这不是原始绝对阈值 **S3≥8**。`ll_only` 的近似退出只发生在第 3 项后；启用零尾部保护后，第 1、2 项后的退出全部来自精确零尾部条件。

相对 FP32 下降不超过 2 个百分点时，节省最多的已验证候选为 **guarded_all_dense_ratio_2_sll_only**，准确率 **97.3228%**，全编码器省部分积 **23.1733%**。这是一项单独的宽松目标，不代替上面的严格目标。

该 2 个百分点目标方案与精确控制 **all_dense_exact_zero_tail_control** 覆盖同一操作集合，相比该无损控制，全编码器部分积节省比例增加 **3.3989 个百分点**；总节省比例同时包含精确零尾部带来的收益。

精确零尾部控制 **exact_zero_tail_control** 覆盖 24 个 Dense 操作，logits 与 W8A8 是否逐位一致：**是**；全编码器省部分积 **18.3693%**。它利用 ratio λ＝65536 的精确零尾部条件，仅当保守余项界 B＝0 时提前停止，省去的尾部部分积必为零，不引入近似误差。

精确零尾部控制 **all_dense_exact_zero_tail_control** 覆盖 72 个 Dense 操作，logits 与 W8A8 是否逐位一致：**是**；全编码器省部分积 **19.7744%**。它利用 ratio λ＝65536 的精确零尾部条件，仅当保守余项界 B＝0 时提前停止，省去的尾部部分积必为零，不引入近似误差。

严格目标推荐与精确控制 **all_dense_exact_zero_tail_control** 覆盖完全相同的操作集合；相对该控制，全编码器部分积节省比例增加 **1.0639 个百分点**。因此不会把从 FFN 扩大到全部 Dense 的零尾部收益误算为近似独有收益。

补充的零尾部保护版本中，严格目标最佳为 **guarded_all_dense_ratio_8_sll_only**，准确率 **97.5253%**、全编码器省部分积 **20.8384%**。总体推荐已同时比较原始与保护版本，主实验 `selected_policy.json` 和 `completion.json` 保留原记录。

以下结果均来自完整 validation 的独立配置；筛选集的计数不用于最终推荐。

| 配置 | 正确数 | 准确率 | 比 W8A8 额外下降/百分点 | 比 FP32 下降/百分点 | 所选操作省部分积 | 全编码器省部分积 | 平均项数 |
|---|---:|---:|---:|---:|---:|---:|---:|
| [guarded_all_dense_ratio_2_sll_only](trials/guarded_all_dense_ratio_2_sll_only__full/result.json) | 4326/4445 | 97.3228% | 0.2700 | 0.3150 | 29.1477% | 23.1733% | 2.8341 |
| [guarded_all_dense_ratio_8_sll_only](trials/guarded_all_dense_ratio_8_sll_only__full/result.json) | 4335/4445 | 97.5253% | 0.0675 | 0.1125 | 26.2108% | 20.8384% | 2.9516 |
| [all_dense_ffn_ratio_32_sall](trials/all_dense_ffn_ratio_32_sall__full/result.json) | 4337/4445 | 97.5703% | 0.0225 | 0.0675 | 25.0852% | 19.9435% | 2.9966 |
| [ffn_ratio_2_sall](trials/ffn_ratio_2_sall__full/result.json) | 4317/4445 | 97.1204% | 0.4724 | 0.5174 | 37.4829% | 19.8667% | 2.5007 |
| [all_dense_exact_zero_tail_control](trials/all_dense_exact_zero_tail_control__full/result.json) | 4338/4445 | 97.5928% | 0.0000 | 0.0450 | 24.8725% | 19.7744% | 3.0051 |
| [guarded_ffn_ratio_2_sll_only](trials/guarded_ffn_ratio_2_sll_only__full/result.json) | 4326/4445 | 97.3228% | 0.2700 | 0.3150 | 36.7311% | 19.4682% | 2.5308 |
| [guarded_ffn_sigma_128_sll_only](trials/guarded_ffn_sigma_128_sll_only__full/result.json) | 4313/4445 | 97.0304% | 0.5624 | 0.6074 | 36.3138% | 19.2471% | 2.5474 |
| [guarded_ffn_sigma_256_sll_only](trials/guarded_ffn_sigma_256_sll_only__full/result.json) | 4332/4445 | 97.4578% | 0.1350 | 0.1800 | 35.4162% | 18.7713% | 2.5834 |
| [guarded_ffn_ratio_8_sll_only](trials/guarded_ffn_ratio_8_sll_only__full/result.json) | 4336/4445 | 97.5478% | 0.0450 | 0.0900 | 35.1943% | 18.6537% | 2.5922 |
| [guarded_ffn_ratio_16_sll_only](trials/guarded_ffn_ratio_16_sll_only__full/result.json) | 4330/4445 | 97.4128% | 0.1800 | 0.2250 | 34.8923% | 18.4937% | 2.6043 |
| [ffn_ratio_32_sall](trials/ffn_ratio_32_sall__full/result.json) | 4338/4445 | 97.5928% | 0.0000 | 0.0450 | 34.7291% | 18.4071% | 2.6108 |
| [ffn_ratio_64_sall](trials/ffn_ratio_64_sall__full/result.json) | 4335/4445 | 97.5253% | 0.0675 | 0.1125 | 34.6688% | 18.3752% | 2.6132 |
| [exact_zero_tail_control](trials/exact_zero_tail_control__full/result.json) | 4338/4445 | 97.5928% | 0.0000 | 0.0450 | 34.6578% | 18.3693% | 2.6137 |
| [masked_ffn_ffn_ratio_2_sall](trials/masked_ffn_ffn_ratio_2_sall__full/result.json) | 4317/4445 | 97.1204% | 0.4724 | 0.5174 | 35.6547% | 12.5985% | 2.5738 |
| [ffn_ratio_2_sll_only](trials/ffn_ratio_2_sll_only__full/result.json) | 4326/4445 | 97.3228% | 0.2700 | 0.3150 | 14.8188% | 7.8543% | 3.4072 |
| [ffn_sigma_64_sll_only](trials/ffn_sigma_64_sll_only__full/result.json) | 4308/4445 | 96.9179% | 0.6749 | 0.7199 | 2.7432% | 1.4540% | 3.8903 |
| [ffn_sigma_4096_sall](trials/ffn_sigma_4096_sall__full/result.json) | 4338/4445 | 97.5928% | 0.0000 | 0.0450 | 0.0001% | 0.0000% | 4.0000 |

最终入选配置从输入重新推理并保存完整逐层输出；下面的 `_final` 是独立重跑，不增加搜索配置数量。

| 配置 | 正确数 | 准确率 | 比 W8A8 额外下降/百分点 | 比 FP32 下降/百分点 | 所选操作省部分积 | 全编码器省部分积 | 平均项数 |
|---|---:|---:|---:|---:|---:|---:|---:|
| [all_dense_ffn_ratio_32_sall_final](trials/all_dense_ffn_ratio_32_sall_final__full/result.json) | 4337/4445 | 97.5703% | 0.0225 | 0.0675 | 25.0852% | 19.9435% | 2.9966 |
| [guarded_all_dense_ratio_8_sll_only_final](trials/guarded_all_dense_ratio_8_sll_only_final__full/result.json) | 4335/4445 | 97.5253% | 0.0675 | 0.1125 | 26.2108% | 20.8384% | 2.9516 |

## 算法、阈值和计数口径

权重和激活先取绝对值，再拆为无符号高四位和低四位，最后恢复乘积符号；不是直接切分有符号补码。权重 w＝16wH＋wL、激活 a＝16aH＋aL，按 HH＝256wHaH、HL＝16wHaL、LH＝16wLaH、LL＝wLaL 顺序相加。全算四项应与原 INT8 乘积逐对完全一致。

`sigma` 比较对齐后的当前整数乘积幅值 S≥σ。σ 使用原始量化整数乘积单位；实际反量化单位还需乘以 2^−(Fw＋Fa)，所以同一个 σ 在不同层的实际数值含义可能不同。σ 越大通常越保守。

`ratio` 比较 S≥λB，其中 B 是尚未计算部分积幅值的保守上界；λ 为无量纲，越大通常越保守，与原始绝对阈值 σ 区分。三处上界依次为 min(240wH＋15a,240aH＋15w)、15a、min(15wL,15aL)。比较相等时停止。单次乘法的余项界不能保证点积或分类误差，因为点积存在正负抵消。

`ll_only` 只允许在前三项后省略 LL；`all` 允许在第一、第二、第三项后停止。stage 1～4 表示执行了多少个四位子乘法槽位。主计数包含已执行但乘数为零的项，没有任意跳过中间零项的独立优化；B＝0 提前停止则自然跳过整个零尾部。非零部分积次数是另一项统计，当前模型运行记录的是槽位数，不能把这些节省全部解释为省去了非零乘法。

HH/HL/LH/LL 各按 1 个四位乘法槽位计数；CPU oracle 实际使用 LUT，槽位减少不表示 CPU 指令减少。精确零尾部控制只采用 B＝0 早停，并未穷尽所有无损优化，也没有实现 a＝0 或 w＝0 时零次乘法的直接旁路（当前最少计 1 项）。因此，相对该控制的近似增量不能解释为相对所有可能无损电路的优势。

补充策略的 `zero_tail_guard=True` 将执行阶段设为 min(原停止阶段, 精确零尾部停止阶段)。这可以在原阈值已经选出的近似乘积基础上省去额外零尾部，乘积 LUT 与原策略保持逐整数对一致；原模式与阈值的非零余项停止规则不变。保护版本及其 `_final` 仍重新前向推理并计数，不复用原策略预测。

所选操作节省比例＝1−所选操作实际子乘法槽位数/(4×所选操作 MAC 数)。全编码器比例以整个编码器的四项 MAC 槽位数为分母，未近似的操作按完整四项计，便于比较只改少量操作和改大量操作的候选。筛选表与全量表都使用各自候选传播后的真实激活计数；training_policy_metrics.csv 单独保存原 W8A8 训练操作数直方图上的预测值，不能与候选运行实测混用。

逐操作省算表保留每个被测 Dense 的四阶段计数；`stage1_count`～`stage4_count` 表示分别在第 1～4 项后结束的乘法对数量。逐层表将这些计数按 12 个编码器层汇总，并关联该层输出 MAE/RMSE/余弦误差。`saved_partial_product_fraction_selected` 仅以该行选中操作为分母；`saved_partial_product_fraction_layer` 以整层所有矩阵的完整四项槽位为分母，包含未近似的 QK/AV 等操作；`encoder_saved_fraction_contribution` 以整个编码器为分母，各层贡献之和等于候选的全编码器节省比例。本轮 12 层形状相同，整层分母为全编码器分母的 1/12。

这些是算术工作量统计，没有包含阈值比较器、符号/绝对值转换、控制、额外存储、阵列等待和搬运开销。未测量 ASIC 面积、26×4 阵列周期或能耗，不能把部分积减少百分比直接当成面积、时延或功耗减少百分比。软件推理耗时只反映当前 CPU oracle，不能当作硬件加速比。

## 实验边界和可复核文件

保留 W8A8 数值配置、N16 非线性与 Softmax→AV 的 P8/F7；FP32 软件输入准备和分类头保持相同，未选中的矩阵仍按完整 INT8 乘法执行。只改变 policy.json 列出的矩阵乘法部分积停止规则，不训练、不微调、不更新权重。

本轮每个候选重新前向计算预测；MFCC 缓存只包含输入特征，旧实验预测不是推理输入。训练划分只用于无梯度标定或操作数分布统计，validation 用于候选调试和全量精度验证；testing 使用数为 **0**。本报告仅从本轮保存的预测重新计分复核，不读取数据集。

筛选批次组织差异：完整 validation 按自然顺序分批，384 条平衡筛选子集会重新组织批次。FP32 patch/head GEMM 的微小舍入差异可经过量化边界放大，因此 screen 的逐层/logits 差异不应全部归因于近似乘法；即使乘法完全精确，也可能不与全量基线按索引抽出的浮点张量逐位相同。这是同轮独立前向的数值路径差异，不表示读取旧预测。最终精度与主要误差结论以同自然顺序的 4,445 条 full 为准，screen 仅用于筛选。

原始逐层表的 MAE、RMSE、最大误差和余弦相似度均是候选相对本轮 W8A8 的比较。通用统计器保留了旧字段名：`fp32_max_abs` 实际表示 W8A8 参考最大绝对值，`w16a16_max_abs` 实际表示当前候选最大绝对值。汇总文件 `normalized_layer_comparison.csv/json` 已分别规范命名为 `reference_w8a8_max_abs`、`candidate_max_abs`；原始实验文件保持原样。

| 文件 | 内容 |
|---|---|
| `run_metadata.json` | 本轮来源、执行配置和数据边界 |
| `baseline/fp32/`、`baseline/w8a8/` | 本轮重新推理的对照预测及 W8A8 逐层输出 |
| `trials/*__control/` | native 完整四项实现控制，单独统计 |
| `trials/*__screen/` | 筛选预测与候选真实操作数的部分积计数 |
| `trials/*__full/policy.json` | 操作列表、模式、阈值、停止阶段 |
| `trials/*__full/predictions.npz` | 完整验证 logits 和标签 |
| `trials/*__full/layer_comparison.csv/json` | 逐层误差对比 |
| `trials/*__full/operation_counts.json` | 候选实际操作数下的部分积阶段计数 |
| `all_results.csv/json` | 所有筛选和全量结果；scope 区分两者 |
| `recommendations.json` | 严格/宽松目标推荐、Pareto 候选和保存预测复核 |
| `recommended_strict_policy.json`、`recommended_2pp_policy.json` | 综合原始与保护版本后的推荐策略、完整格式/LUT来源、全量指标和可用的独立重跑目录 |
| `normalized_layer_comparison.csv/json` | 控制与全量候选逐层误差，规范参考/候选字段名 |
| `operation_count_comparison.csv/json` | 控制与全量候选逐 Dense 阶段计数、槽位节省及全编码器贡献 |
| `layer_compute_comparison.csv/json` | 逐层聚合省算并关联层输出误差，未选中层保留零贡献 |
| `training_policy_metrics.csv/json` | training 直方图上的节省量预测，单独保存 |
| `selected_policy.json`、`completion.json` | 主阶段最终独立重跑的入选策略和完成状态 |
| `guarded_completion.json` | 零尾部保护补充阶段完成状态；不覆盖主阶段记录 |
| `screening_batch_grouping_audit.json` | 精确零尾部控制在重组筛选批次及自然顺序全量批次下的 logits 对比 |

## 筛选结果

以下准确率和部分积计数均来自 screening validation 子集上的实际候选前向；子集结果不能代替完整 validation。

精确零尾部筛选控制 **exact_zero_tail_control** 相比从全量基线抽出的同样本结果：分类翻转 **0** 条，变化 logits 元素 **451** 个，logits 最大绝对差 **0.046283543**，patch 最大绝对差 **1.3113022e-06**。自然顺序全量精确控制已单独核验 logits 逐位一致。

| 配置 | 正确数 | 准确率 | 比 W8A8 额外下降/百分点 | 比 FP32 下降/百分点 | 所选操作省部分积 | 全编码器省部分积 | 平均项数 |
|---|---:|---:|---:|---:|---:|---:|---:|
| [guarded_all_dense_sigma_256_sll_only](trials/guarded_all_dense_sigma_256_sll_only__screen/result.json) | 377/384 | 98.1771% | -0.2604 | 0.0000 | 26.8994% | 21.3859% | 2.9240 |
| [guarded_all_dense_ratio_8_sll_only](trials/guarded_all_dense_ratio_8_sll_only__screen/result.json) | 377/384 | 98.1771% | -0.2604 | 0.0000 | 26.2128% | 20.8400% | 2.9515 |
| [single_layer_05_ffn_ratio_2_sall](trials/single_layer_05_ffn_ratio_2_sall__screen/result.json) | 377/384 | 98.1771% | -0.2604 | 0.0000 | 41.1715% | 1.8185% | 2.3531 |
| [ffn_sigma_256_sll_only](trials/ffn_sigma_256_sll_only__screen/result.json) | 377/384 | 98.1771% | -0.2604 | 0.0000 | 0.9052% | 0.4798% | 3.9638 |
| [guarded_all_dense_ratio_16_sll_only](trials/guarded_all_dense_ratio_16_sll_only__screen/result.json) | 376/384 | 97.9167% | 0.0000 | 0.2604 | 25.5014% | 20.2744% | 2.9799 |
| [all_dense_ffn_ratio_32_sall](trials/all_dense_ffn_ratio_32_sall__screen/result.json) | 376/384 | 97.9167% | 0.0000 | 0.2604 | 25.0877% | 19.9455% | 2.9965 |
| [ffn_ratio_2_sall](trials/ffn_ratio_2_sall__screen/result.json) | 376/384 | 97.9167% | 0.0000 | 0.2604 | 37.4855% | 19.8681% | 2.5006 |
| [ffn_ratio_8_sall](trials/ffn_ratio_8_sall__screen/result.json) | 376/384 | 97.9167% | 0.0000 | 0.2604 | 35.1972% | 18.6553% | 2.5921 |
| [ffn_ratio_32_sall](trials/ffn_ratio_32_sall__screen/result.json) | 376/384 | 97.9167% | 0.0000 | 0.2604 | 34.7327% | 18.4091% | 2.6107 |
| [ffn_ratio_64_sall](trials/ffn_ratio_64_sall__screen/result.json) | 376/384 | 97.9167% | 0.0000 | 0.2604 | 34.6738% | 18.3778% | 2.6130 |
| [exact_zero_tail_control](trials/exact_zero_tail_control__screen/result.json) | 376/384 | 97.9167% | 0.0000 | 0.2604 | 34.6626% | 18.3719% | 2.6135 |
| [masked_ffn_ffn_ratio_2_sall](trials/masked_ffn_ffn_ratio_2_sall__screen/result.json) | 376/384 | 97.9167% | 0.0000 | 0.2604 | 35.6600% | 12.6003% | 2.5736 |
| [ffn_ratio_2_sll_only](trials/ffn_ratio_2_sll_only__screen/result.json) | 376/384 | 97.9167% | 0.0000 | 0.2604 | 14.8187% | 7.8542% | 3.4073 |
| [ffn_ratio_8_sll_only](trials/ffn_ratio_8_sll_only__screen/result.json) | 376/384 | 97.9167% | 0.0000 | 0.2604 | 13.3223% | 7.0611% | 3.4671 |
| [ffn_ratio_32_sll_only](trials/ffn_ratio_32_sll_only__screen/result.json) | 376/384 | 97.9167% | 0.0000 | 0.2604 | 12.8604% | 6.8163% | 3.4856 |
| [ffn_ratio_64_sll_only](trials/ffn_ratio_64_sll_only__screen/result.json) | 376/384 | 97.9167% | 0.0000 | 0.2604 | 12.8013% | 6.7849% | 3.4879 |
| [single_layer_07_ffn_ratio_2_sall](trials/single_layer_07_ffn_ratio_2_sall__screen/result.json) | 376/384 | 97.9167% | 0.0000 | 0.2604 | 46.2286% | 2.0418% | 2.1509 |
| [single_layer_08_ffn_ratio_2_sall](trials/single_layer_08_ffn_ratio_2_sall__screen/result.json) | 376/384 | 97.9167% | 0.0000 | 0.2604 | 45.4227% | 2.0062% | 2.1831 |
| [single_layer_10_ffn_ratio_2_sall](trials/single_layer_10_ffn_ratio_2_sall__screen/result.json) | 376/384 | 97.9167% | 0.0000 | 0.2604 | 39.9209% | 1.7632% | 2.4032 |
| [single_layer_03_ffn_ratio_2_sall](trials/single_layer_03_ffn_ratio_2_sall__screen/result.json) | 376/384 | 97.9167% | 0.0000 | 0.2604 | 38.3115% | 1.6922% | 2.4675 |
| [single_layer_01_ffn_ratio_2_sall](trials/single_layer_01_ffn_ratio_2_sall__screen/result.json) | 376/384 | 97.9167% | 0.0000 | 0.2604 | 31.7763% | 1.4035% | 2.7289 |
| [single_layer_11_ffn_ratio_2_sall](trials/single_layer_11_ffn_ratio_2_sall__screen/result.json) | 376/384 | 97.9167% | 0.0000 | 0.2604 | 29.6162% | 1.3081% | 2.8154 |
| [ffn_sigma_128_sll_only](trials/ffn_sigma_128_sll_only__screen/result.json) | 376/384 | 97.9167% | 0.0000 | 0.2604 | 1.8846% | 0.9989% | 3.9246 |
| [single_layer_12_ffn_ratio_2_sall](trials/single_layer_12_ffn_ratio_2_sall__screen/result.json) | 376/384 | 97.9167% | 0.0000 | 0.2604 | 12.5766% | 0.5555% | 3.4969 |
| [ffn_sigma_1024_sll_only](trials/ffn_sigma_1024_sll_only__screen/result.json) | 376/384 | 97.9167% | 0.0000 | 0.2604 | 0.0747% | 0.0396% | 3.9970 |
| [ffn_sigma_2048_sll_only](trials/ffn_sigma_2048_sll_only__screen/result.json) | 376/384 | 97.9167% | 0.0000 | 0.2604 | 0.0050% | 0.0027% | 3.9998 |
| [ffn_sigma_4096_sall](trials/ffn_sigma_4096_sall__screen/result.json) | 376/384 | 97.9167% | 0.0000 | 0.2604 | 0.0001% | 0.0000% | 4.0000 |
| [ffn_sigma_4096_sll_only](trials/ffn_sigma_4096_sll_only__screen/result.json) | 376/384 | 97.9167% | 0.0000 | 0.2604 | 0.0001% | 0.0000% | 4.0000 |
| [ffn_sigma_8192_sll_only](trials/ffn_sigma_8192_sll_only__screen/result.json) | 376/384 | 97.9167% | 0.0000 | 0.2604 | 0.0000% | 0.0000% | 4.0000 |
| [ffn_sigma_8192_sall](trials/ffn_sigma_8192_sall__screen/result.json) | 376/384 | 97.9167% | 0.0000 | 0.2604 | 0.0000% | 0.0000% | 4.0000 |
| [ffn_sigma_12288_sall](trials/ffn_sigma_12288_sall__screen/result.json) | 376/384 | 97.9167% | 0.0000 | 0.2604 | 0.0000% | 0.0000% | 4.0000 |
| [ffn_sigma_16384_sall](trials/ffn_sigma_16384_sall__screen/result.json) | 376/384 | 97.9167% | 0.0000 | 0.2604 | 0.0000% | 0.0000% | 4.0000 |
| [guarded_all_dense_ratio_2_sll_only](trials/guarded_all_dense_ratio_2_sll_only__screen/result.json) | 375/384 | 97.6562% | 0.2604 | 0.5208 | 29.1480% | 23.1735% | 2.8341 |
| [ffn_ratio_4_sall](trials/ffn_ratio_4_sall__screen/result.json) | 375/384 | 97.6562% | 0.2604 | 0.5208 | 35.6454% | 18.8928% | 2.5742 |
| [ffn_ratio_16_sall](trials/ffn_ratio_16_sall__screen/result.json) | 375/384 | 97.6562% | 0.2604 | 0.5208 | 34.8967% | 18.4960% | 2.6041 |
| [ffn_ratio_4_sll_only](trials/ffn_ratio_4_sll_only__screen/result.json) | 375/384 | 97.6562% | 0.2604 | 0.5208 | 13.7415% | 7.2833% | 3.4503 |
| [ffn_ratio_16_sll_only](trials/ffn_ratio_16_sll_only__screen/result.json) | 375/384 | 97.6562% | 0.2604 | 0.5208 | 13.0243% | 6.9032% | 3.4790 |
| [single_layer_09_ffn_ratio_2_sall](trials/single_layer_09_ffn_ratio_2_sall__screen/result.json) | 375/384 | 97.6562% | 0.2604 | 0.5208 | 45.2301% | 1.9977% | 2.1908 |
| [single_layer_06_ffn_ratio_2_sall](trials/single_layer_06_ffn_ratio_2_sall__screen/result.json) | 375/384 | 97.6562% | 0.2604 | 0.5208 | 44.7091% | 1.9747% | 2.2116 |
| [single_layer_04_ffn_ratio_2_sall](trials/single_layer_04_ffn_ratio_2_sall__screen/result.json) | 375/384 | 97.6562% | 0.2604 | 0.5208 | 38.1701% | 1.6859% | 2.4732 |
| [single_layer_02_ffn_ratio_2_sall](trials/single_layer_02_ffn_ratio_2_sall__screen/result.json) | 375/384 | 97.6562% | 0.2604 | 0.5208 | 36.1161% | 1.5952% | 2.5554 |
| [ffn_sigma_64_sll_only](trials/ffn_sigma_64_sll_only__screen/result.json) | 375/384 | 97.6562% | 0.2604 | 0.5208 | 2.7412% | 1.4529% | 3.8904 |
| [ffn_sigma_512_sll_only](trials/ffn_sigma_512_sll_only__screen/result.json) | 374/384 | 97.3958% | 0.5208 | 0.7812 | 0.3909% | 0.2072% | 3.9844 |
| [guarded_all_dense_sigma_128_sll_only](trials/guarded_all_dense_sigma_128_sll_only__screen/result.json) | 373/384 | 97.1354% | 0.7812 | 1.0417 | 28.8706% | 22.9530% | 2.8452 |
| [ffn_sigma_1024_sall](trials/ffn_sigma_1024_sall__screen/result.json) | 373/384 | 97.1354% | 0.7812 | 1.0417 | 0.1560% | 0.0827% | 3.9938 |
| [ffn_sigma_0_sll_only](trials/ffn_sigma_0_sll_only__screen/result.json) | 369/384 | 96.0938% | 1.8229 | 2.0833 | 25.0000% | 13.2505% | 3.0000 |
| [ffn_sigma_256_sall](trials/ffn_sigma_256_sall__screen/result.json) | 367/384 | 95.5729% | 2.3438 | 2.6042 | 2.2196% | 1.1764% | 3.9112 |

推荐只表示本轮已测试配置中的结果，没有穷举所有逐层阈值，也不能保证独立最终评测集上的准确率。
