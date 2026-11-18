# W8A8 四位部分积提前停止实验

在固定 W8A8、N16 非线性、Softmax→AV P8/F7 的 KWT-1 基础上，测试可调门限的近似 Dense 乘法。不更新模型权重；训练划分仅用于标定/操作数统计，验证划分用于筛选与完整精度评价，禁止使用 testing。

本次已完成结果见 [2026-10-30 实验报告](backup/latest_nibble_results/README_四位渐进乘法实验结果.md)。在已测候选中，额外精度下降不超过 0.1 个百分点时，推荐全部编码器 Dense 使用 `ratio` 门限 λ=8、`ll_only` 并开启精确零尾部保护；4445 条验证样本正确 4335 条。本次报告和逐层结果均保存在新目录中。

## 在 VS Code 中运行

使用“终端 → 运行任务 → KWT-1: progressive four-bit product search (validation only)”，或执行：

```powershell
Set-Location D:\codex\kwt_fp32
& .\.conda\python.exe -B -u .\run_kwt_nibble_experiments.py
```

每次新建 `runs/nibble_w8a8_日期时间/`，拒绝覆盖既有实验。需要已有的 `runs/precision_search_20261028_162338_802026` 输入缓存、训练标定参数，以及本机官方模型。`--parent` 可替换为结构相同的准备目录。MFCC 缓存只提供模型输入；各候选重新计算 logits，不读取旧实验预测作推理。

运行依赖现有 `.conda` Python 3.8、TensorFlow 2.4、NumPy。Dense 的 CPU 实验内核由 `nibble_matmul.cpp` 编译为 DLL，通过 `nibble_native.py` 调用，避免构造巨大的逐乘法广播张量。默认编译器为本机 Vivado 附带的 GCC；可通过 `KWT_CXX` 指定其他兼容 GCC。开发机可复制已核验源码/二进制哈希的同机预编译 DLL；这只复用执行代码，不复用模型预测。

255×255 的乘积查表仅用于加速软件参考计算，不表示 RTL 必须增加这样一张 ROM。实际硬件应实现四位乘法、移位累加、门限比较和停止控制，再核算其面积与周期。

## 数值定义

先分离符号，将 `abs(a)`、`abs(w)` 分别拆成高四位和低四位，以 unsigned4×unsigned4 计算：

```text
HH = 256 * wH * aH
HL =  16 * wH * aL
LH =  16 * wL * aH
LL =       wL * aL
```

按 HH、HL、LH、LL 累加幅值，结束时恢复乘积符号。全部四项重建与原 signed INT8 乘法精确一致；不是把补码高低四位都当 signed4 相乘。矩阵整数内积完成后只做一次原有反量化，保留 bias、残差、非线性和软件 patch/head 的既有运算顺序。

- `sigma`：对齐后的乘积幅值部分和 `S >= sigma` 时停止。sigma 是整数乘积码域门限；各算子 F 不同，同一 sigma 对应的实数幅值不同。
- `ratio`：`S >= λ * B` 时停止，B 是便宜的剩余误差保守上界，λ 无量纲。推荐 λ=8 并非原始整数门限 sigma=8。
- 比较相等时停止；`ll_only` 仅允许近似省略 LL，`all` 允许三个提前停止点。
- `exact_zero_tail_control` 仅当保守界 B=0 时提前停，乘积仍完全精确。它用于区分无损省算和近似省算。
- 补充的 `zero_tail_guard` 取原规则与无损零尾部规则中更早的阶段，乘积与原规则完全相同，执行槽可能减少。

在幅值拆分下，精确乘积幅值为 `S + R`，且 `0 <= R <= B`。因此 ratio 门限为正时，提前停止乘积的相对误差至多为 `1 / (threshold + 1)`；这只约束单次乘积，不能保证矩阵点积相消后的相对误差或最终分类准确率。

推荐策略在前三项之后比较 `S3 >= 8 * B3`，其中 `B3 = 15 * min(wL, aL)`；满足时省略 LL，否则继续精确计算 LL。`ll_only` 下第一、第二阶段若停止，只能来自精确零尾部保护。

## 实验与记录

工作流先全量重跑 FP32、W8A8 基线，进行完整四项的逐层一致性检查，统计训练样本操作数分布，再筛选门限、逐层隔离、完整验证和扩大 Dense 覆盖范围。QK/AV 的矩阵乘保持精确 INT8；非线性格式保持 N16/P8F7。

严格目标为相对 W8A8 额外下降不超过 0.1 个百分点；同时记录相对 FP32 下降不超过 2 个百分点的原条件。门限在验证集调优，因此本结果属于开发集结果，不能视为未见样本上的最终保证。

输出包括：

- `baseline/`：本次重新计算的 FP32、W8A8 预测和 W8A8 的 15 个阶段输出；
- `trials/`：各策略、门限、LUT、预测、实际阶段计数及逐层误差；
- `screening_results.*`、`full_validation_results.*`：筛选/完整验证统计；
- 最佳配置的 `_final`：重新全量推理，保存 15 个阶段数组并校验重复结果；
- `README_四位渐进乘法实验结果.md`、`recommendations.json`：结果与推荐；
- `independent_audit.json`：独立重新计分、计数和来源审计。

四位部分积槽的减少比例，以每次乘法完整计算四项为对照。它不是原生单周期 INT8 乘法器的实测加速比，也不代表 SMIC40-2025 的面积、能耗或时序已经改善。26×4 阵列中的阶段分歧、比较/控制和访存，需要在 RTL 中另行核验。

结果目录属于 `.gitignore` 排除范围，仅运行本流程不会自动上传 GitHub。

## 2026-11-18 最新备份范围

最新源码与该次实验顶层报告、配置、审计和CSV/JSON摘要已纳入本仓库的`backup/latest_nibble_results/`。该目录不含权重、数据集、MFCC缓存、逐层数组和DLL；报告中的原始runs路径是来源记录，无法仅凭这些摘要重建完整历史运行目录。重新推理需按FP32手册重新准备模型和数据。
