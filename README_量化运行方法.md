# KWT-1：16 bit 定点矩阵乘与浮点非线性

在 VS Code 中打开 `D:\codex\kwt_fp32`，继续使用现有 `.conda\python.exe` 环境、官方权重和 `data2` 数据，无需重新安装依赖。

## 本次实验计算方式

- 矩阵乘的权重与输入激活量化为有符号 int16；量化范围为 `[-32767, 32767]`。
- 每个矩阵操作数采用固定的 2 的幂尺度。激活尺度由训练集 256 条样本校准，并留出 1.25 倍范围；验证集用于评估。
- int16 数值转为 int64 后进行实际整数乘加，避免累加溢出；乘积按尺度恢复为 FP32。
- Softmax、GELU、LayerNorm，以及偏置、残差和嵌入加法保留 FP32。MFCC 前处理也保留原浮点实现。
- 包含 patch 投影、Q/K/V、注意力中的两次矩阵乘、输出投影、FFN 和分类头。

这是一次数值对照实验，CPU 运行时间不能当作 FPGA 加速性能。

## 在终端运行完整验证集

```powershell
Set-Location D:\codex\kwt_fp32
$env:TEMP = 'D:\codex\kwt_fp32\.tmp'
$env:TMP = 'D:\codex\kwt_fp32\.tmp'
& .\.conda\python.exe .\run_kwt_int16.py --repo .\keyword-transformer --data-dir .\data2 --baseline-run .\runs\validation_full --batch-size 32 --calibration-samples 256 --activation-headroom 1.25 --limit 0
```

`--limit 0` 运行完整验证集；首次检查流程时可以改成 `--limit 32`。

不指定 `--out`，脚本会在 `runs` 下自动生成新的唯一实验目录，并在终端打印实际路径。原来的 `runs\validation_full` 只作为 FP32 对照来源，不覆盖其中的结果。再次执行以上命令会产生另一份实验记录。

## 使用 VS Code 任务和调试

1. “终端 → 运行任务”选择 **KWT-1: W16A16 + FP32 nonlinear**，运行完整验证集。
2. 需要断点检查时，在“运行和调试”选择 **KWT-1: W16A16 first 32 validation samples**，按 F5。此配置只评估前 32 条验证样本，仍使用训练集 256 条校准。

原有 FP32 任务及调试配置保持原样。

## 查看对照结果

新实验会复用上次保存的验证样本及顺序，同时运行 FP32 与定点矩阵乘模型。每层会保存全部评估样本的完整输出，并汇总层间误差；另外记录准确率、逐样本预测、量化尺度和饱和情况。

完整张量使用 NumPy 文件保存，建议用 `np.load(path, mmap_mode='r')` 按需读取。完整验证集的逐层双分支结果占用数 GB 磁盘；终端与实验目录内的汇总记录用于确认本次运行实际完成情况。

### 直接对照某一层、某一个样本

```powershell
& .\.conda\python.exe .\compare_kwt_layers.py --run .\runs\w16a16_fpnl_20261028_103350_996076 --stage layer_12_output --sample-index 0 --token 0
```

命令将打印整条样本在该层的 MAE、RMSE、最大绝对误差、相对 L2、余弦相似度，并列出所选 token 的前 8 个通道数值。`--token 0` 是 CLS；`--stage` 可以换成 `patch_output`、`encoder_input`、`layer_01_output` 至 `layer_12_output` 或 `logits`。`logits` 没有 token 维度，使用 `--token 0`。

`layer_comparison.csv` 给出全部样本汇总的每层误差；`per_sample_layer_metrics.csv` 可按层名和样本序号筛选。验证清单中的样本序号与每个 `.npy` 文件的第 0 维一一对应。

## 已完成的完整实验

结果目录：`D:\codex\kwt_fp32\runs\w16a16_fpnl_20261028_103350_996076`。

| 模型 | 正确数 / 总数 | 准确率 |
|---|---:|---:|
| FP32 对照 | 4340 / 4445 | 97.637795% |
| W16A16 + FP32 非线性 | 4339 / 4445 | 97.615298% |

准确率下降 0.022497 个百分点，仅 1 条预测改变。完整输出约 3.15 GB；共有 8 个激活值饱和，细节、每层误差表、全部输出说明在实验目录的 `README_量化结果.md` 中。上一次实验的 12 个文件哈希均未改变。

独立核验命令：

```powershell
& .\.conda\python.exe .\verify_kwt_int16_results.py --run .\runs\w16a16_fpnl_20261028_103350_996076
```
