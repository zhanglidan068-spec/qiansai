# 逐层混合精度实验

本轮按赛题边界评估：电脑端生成 FP32 编码器输入，硬件任务是 12 层编码器，最终接相同 FP32 软件分类头。输入投影、位置和分类 token 准备、分类头不参与硬件位宽优化。

## 数据与精度门槛

- training 选 256 条样本，前向标定，不训练、不微调、不更新权重。
- validation 全部 4,445 条用于最终精度验证；FP32 正确 4,340 条。
- 准确率下降不超过 2 个百分点，要求至少正确 **4,252 条**，对应 95.6580%。
- 从 validation 每类固定选 32 条，共 384 条，用于候选筛选。子集通过不能替代全量验证。
- 不读取 testing 音频，不用 testing 选择任何数值配置。

## 搜索内容

1. 对 12 层分别进行 8、10、12、16 bit 量化，其余层保持 FP32，共 48 组敏感性实验。
2. 测试矩阵权重 W4/W6/W8/W16，激活 A8/A10/A12/A16，Softmax→AV 概率 8/10/12/16 bit 及不同小数位。
3. 保留概率 int8 F=6 作为损失定位基准；赛题没有固定 F=6 的要求，因此同时比较 F=7、F=9 等方案。
4. 根据敏感层提高部分层位宽，测试混合配置。
5. 在完整 4,445 条上核验入选配置，分别保留精度优先、权重容量优先的候选，导出全部逐层输出及按实际位宽打包的权重。
6. 补充 9 组按输出通道标定权重的 W4/W6 候选。逐通道 F 直接由固定权重计算；激活仍使用 training 标定范围。筛选入选者在全部 validation 上再运行，另外保存逐层输出、打包权重和 6,912 字节通道 F 表。

非线性保持 N16 整数/查表运算。紧凑实现共享一张 1,025 项 GELU int16 表和一张 4,096 项 exp int16 表，载荷合计 10,242 字节；GELU 采用整数线性插值，exp 的零尾部隐式处理。宽中间量仍为 int64，不能把它误作全部寄存器都只有16位。

MFCC 是输入特征缓存，不是答案缓存。每个候选都会重新前向生成 logits；标签仅用于评分。标定范围固定来自 training，不使用 validation 重新拟合范围。

## VS Code / 终端运行

使用现有 `.conda` 环境：

```powershell
Set-Location D:\codex\kwt_fp32
& .\.conda\python.exe -B -u .\run_kwt_precision_experiments.py
```

默认新建带时间戳的目录，已有目录会拒绝覆盖。完整搜索需要较长时间；所有候选各自保存结果，筛选与全量评分有明确标记。

当前这轮目录为：`D:\codex\kwt_fp32\runs\precision_search_20261028_162338_802026`。主搜索状态以 `search_result.json` 为准，按通道搜索状态以 `per_channel_search_result.json` 为准，完整汇总为 `README_精度搜索结果.md`。

若主搜索中断，可在保留相同程序版本与参数的前提下继续尚未完成的候选：

```powershell
& .\.conda\python.exe -B -u .\search_kwt_precision.py --run .\runs\precision_search_20261028_162338_802026
```

已经完整结束的实验不要重复续跑；需要新一轮时使用总入口生成新目录。已完成 trial 的参数若不一致会被拒绝复用。

## 文件说明

| 文件 | 内容 |
|---|---|
| `layer_sensitivity.csv` | 12 层 × 4 位宽的敏感性实验，384 条筛选集 |
| `experiment_results.csv` | 各候选准确率、与同一批 FP32 的差值、screen/full/final 标记 |
| `search_result.json` | 本轮所有全量验证候选、两种推荐目标及是否达精度门槛 |
| `accuracy_layer_comparison.csv` / `efficiency_layer_comparison.csv` | 两种候选的15阶段逐层误差 |
| `trials/*__full/predictions.npz` | 全量验证 logits 和标签，重新计分可复核 |
| `trials/*__final/layer_outputs/` | 两种候选完整15阶段输出 |
| `accuracy_export/` / `efficiency_export/` | 实际位宽打包的权重二进制及逐张量布局、尺度 |
| `best_precision_config.json` | 精度优先配置的每个操作数位宽、小数位及非线性参数 |
| `calibration_ranges.json` | 训练集标定范围 |
| `search_source/` | 本轮搜索源码快照和哈希 |
| `per_channel_search/` | 按通道量化的候选、全量验证、精度配置、逐层输出和实际打包权重 |
| `pc_accuracy_layer_comparison.csv` / `pc_efficiency_layer_comparison.csv` | 按通道方案与 FP32 的15阶段误差 |
| `common_hardware_payload/` | 编码器 bias、LayerNorm 定点参数、共享非线性 ROM 及字节布局 |

## 面积验证边界

赛题要求新增逻辑、接口和内部 SRAM 总面积 ≤2.25 mm²，按指定标准单元库和官方 SRAM 宏面积核算，统一 150 MHz。详见 `area_requirements_audit.md`。

`area_proxy.py` 只计算 MAC、权重容量、位宽与多遍执行成本，不输出虚构的 mm²。`hardware_precision_spec.md/json` 给出共享 INT8 PE、打包、缓冲和综合核算模板。

当前缺指定工艺库、官方 SRAM 面积表和完整平台 RTL/综合网表，因此本轮可证明精度、可导出数值配置，**不能据此声称 ASIC 面积已经达标或全局最优**。最终精度、面积、时序必须来自同一个冻结配置。
