# KWT-1：W8A8 矩阵乘、N16 非线性、AV 概率 F=6

在 VS Code 中打开 `D:\codex\kwt_fp32`，使用现有 `.conda\python.exe` 环境运行 `run_kwt_int8_n16.py`。脚本每次创建一个新的 `runs\w8a8_n16_avf6_时间戳` 目录；之前的 FP32、W16A16 实验不会覆盖。

## 这次计算什么

| 计算部分 | 定点方式 |
|---|---|
| 全部 98 个模型矩阵乘 | 两个操作数均为 signed int8，范围 `[-127,127]`；包括 patch、Q/K/V、QKᵀ、AV、O、FFN 和分类头 |
| 矩阵乘累加 | int8 转为 int64 后做实际整数乘加；按两侧尺度还原数值 |
| Softmax | 输入 int16，整数指数查表、整数求和与整数除法，输出 int16、F=15 |
| Softmax → AV 左操作数 | int16 Q15 概率再量化成 int8、**固定 F=6**；尺度 `1/64`，没有再次归一化 |
| GELU | int16 输入码索引 65,535 项 int16 查找表；表格离线生成，前向使用整数索引 |
| LayerNorm | int16 输入/输出，均值与方差相关表达式、整数平方根和仿射计算使用 int64 中间量；gamma 为 int16 F=14，beta 为 int16 F=16 |
| 模型外侧和接口 | MFCC、偏置、嵌入、残差加法，以及定点模块间的反量化数值传递保留 FP32 |

F 表示小数位数，实值为 `整数码 × 2^(-F)`。矩阵乘尺度按操作数固定；除指定的 AV 概率 F=6 外，尺度来自训练样本标定。量化使用最近偶数舍入和对称饱和裁剪。

非线性的前向使用整数计算或整数查表。int64 中间量用于容纳乘积、累加和缩放表达式，非线性输入和输出的存储码仍为 int16。

F=6 的概率步长是 `0.015625`，较小的概率可能变成零，量化后的每行概率之和也可能偏离 1。这是本次指定格式的效果，脚本会记录该变化。CPU 运行耗时用于复现实验，不能作为 FPGA 性能结果。

## 数据使用范围

- 从 **training 划分选取 256 条样本**，仅做无梯度前向数值标定，激活范围留出 1.25 倍余量；不更新已有权重。
- 使用与之前相同顺序的 **validation 划分 4,445 条样本**，做功能核验、逐层对照和准确率验证。
- **不使用 testing 划分样本进行标定、推理或评估**；脚本不提供 testing 切换参数。
- 新预测先由本次前向计算并保存，随后才读取旧预测做历史结果对照。旧预测不参与模型输入和量化尺度确定。

## 在 VS Code 中运行

1. “终端 → 运行任务”，选择 **KWT-1: W8A8 + N16 + AV F6 full validation**，运行完整验证集。
2. 需要断点检查时，“运行和调试”选择 **KWT-1: W8A8 + N16 + AV F6 first 32 validation samples**，按 F5；只评估前 32 条验证样本，训练标定仍为 256 条。

原有 FP32 和 W16A16 任务、调试配置保留。

完整验证命令：

```powershell
Set-Location D:\codex\kwt_fp32
$env:TEMP = 'D:\codex\kwt_fp32\.tmp'
$env:TMP = 'D:\codex\kwt_fp32\.tmp'
& .\.conda\python.exe .\run_kwt_int8_n16.py --limit 0 --batch-size 32 --calibration-samples 256 --activation-headroom 1.25
```

把 `--limit 0` 改成 `--limit 32` 即可运行前 32 条验证样本。省略 `--out` 会自动使用唯一目录；显式传入已经存在的目录时，脚本会拒绝覆盖。

## 结果与逐层查看

本次完整实验目录为 `D:\codex\kwt_fp32\runs\w8a8_n16_avf6_20261028_115258_606490`。是否完成及最终准确率，以该目录生成的 `result.json` 和 `README_量化结果.md` 为准。

| 文件 | 内容 |
|---|---|
| `experiment_comparison.csv` | FP32、旧 W16A16、新 W8A8/N16 的准确率对照 |
| `layer_comparison.csv` | 15 个阶段相对 FP32 的 MAE、RMSE、最大误差、相对 L2 和余弦相似度 |
| `nonlinear_comparison.csv` | 每层 Softmax、GELU、两处 LayerNorm 的误差及裁剪统计 |
| `attention_av_f6_stats.csv` | 每层 AV 输入概率的零值比例、行和等统计 |
| `activation_quantization_stats.csv` | 矩阵乘各激活操作数的裁剪、端点和零值比例 |
| `quantization_config.json` | 全部固定尺度与非线性参数 |
| `first_sample_trace.npz` | 第一条样本的逐层值、矩阵乘 int8 操作数、非线性 int16 输入/输出 |
| `layer_outputs_manifest.json` | 54 个完整输出数组的路径、shape 和 dtype |

全部评估样本会记录到 **54 个 `.npy` 数组**：

- `layer_outputs\fp32`：15 个阶段的 FP32 对照值。
- `layer_outputs\w8a8_n16`：相同 15 个阶段的新模型输出，保存反量化后的实值以便直接比较。
- `layer_outputs\n16_codes`：12 层最终 LayerNorm 的 int16 原始码；各层 F 查阅 `quantization_config.json`。
- `layer_outputs\av_int8_f6`：12 层真正进入 AV 的 int8 概率码；每个码除以 64 得到对应概率实值。

阶段名包括 `patch_output`、`encoder_input`、`layer_01_output` 至 `layer_12_output`、`logits`。第 0 维对应 `sample_manifest.json` 的样本序号。

查看第 0 条验证样本、第 12 层、CLS token 的对照：

```powershell
& .\.conda\python.exe .\compare_kwt_layers.py --run .\runs\w8a8_n16_avf6_20261028_115258_606490 --quant-branch w8a8_n16 --stage layer_12_output --sample-index 0 --token 0
```

查看分类得分时，把 `--stage` 改成 `logits`，保持 `--token 0`。查看其他层可替换 `layer_12_output`。

自行分析大型数组时，使用 `np.load(绝对路径, mmap_mode='r', allow_pickle=False)`，然后按样本取值，避免一次载入全部数据。读取 `n16_codes` 后需要乘该层 `2^(-output_f)`；读取 `av_int8_f6` 后需要除以 64。

## 已完成的完整验证

| 模型 | 正确数 / 4445 | 准确率 |
|---|---:|---:|
| FP32 | 4340 | 97.637795% |
| 上次 W16A16＋浮点非线性 | 4339 | 97.615298% |
| 本次 W8A8＋N16，AV F=6 | 3761 | 84.611924% |

本次比FP32下降13.025872个百分点。训练标定256条、验证4445条，testing使用0条。54个完整输出数组约4.35GB，已完成文件和数值核验。

`layer_comparison_three_precisions.csv` 可直接比较三个实验在15个阶段的误差；`README_量化结果.md` 包含量化算法、逐层指标、AV概率行和与裁剪说明。

重新核验本次结果：

```powershell
& .\.conda\python.exe .\verify_kwt_int8_n16_results.py --run .\runs\w8a8_n16_avf6_20261028_115258_606490
```
