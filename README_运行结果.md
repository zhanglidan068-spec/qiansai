# KWT-1 浮点基线已跑通

实际工程目录：`D:\codex\kwt_fp32`。环境、源码、权重和完整 Speech Commands v2 数据均已准备好，可直接运行。

## 本次实测结果

| 检查 | 结果 |
|---|---|
| 指定代码版本 | 237e4097afd4f6851e0b82f0a8d65c2dfe8eae30 |
| 环境 | Python 3.8.20 / TensorFlow 2.4.0 / Addons 0.12.1 / NumPy 1.19.5 |
| 运行设备 | CPU |
| checkpoint 与模型权重逐项比较 | 162 个张量、607,308 个参数完全相等 |
| 零输入 smoke | 通过 |
| 32 条 validation | 31/32，96.875%（仅用于流程检查） |
| 完整 validation | **4340/4445，97.6378%** |
| 完整 validation 运行耗时 | 约 59 秒，包含数据处理器和模型准备，不包含安装与下载 |
| 逐层 trace 与原模型 CLS 最大绝对误差 | 0 |
| 逐层 trace 与原模型 logits 比较 | 通过 rtol=1e-5、atol=1e-5 断言 |
| Speech Commands v2 压缩包 | 2,428,923,189 字节，MD5 校验通过 |

本次使用完整 validation，尚未运行 testing。TensorFlow 2.4.0 正式版是原依赖 2.4.0rc1 的兼容替代；不声称两种运行时逐位一致。

## 在 VS Code 中继续运行

双击本目录的 `打开VSCode.cmd`，使用本工程专用的 VS Code 配置及扩展。Python 扩展已安装，Python Debugger 已固定为支持 Python 3.8 的 2024.2.0。

自动打开编辑器受到当前执行沙箱限制，未确认 GUI 窗口成功打开；手动双击上述文件即可从正常桌面环境启动。也可以用已有 VS Code 打开此目录，通过“终端 → 运行任务”执行下列任务，无需调试器。

- `Ctrl+Shift+B`：运行 `KWT-1: smoke`。
- “终端 → 运行任务 → KWT-1: validation 32”：运行 32 条样本。
- “终端 → 运行任务 → KWT-1: full validation”：运行完整验证集。

编辑器解释器应为 `D:\codex\kwt_fp32\.conda\python.exe`。终端可直接执行：

```powershell
Set-Location D:\codex\kwt_fp32
& .\.conda\python.exe .\run_kwt_fp32.py --repo .\keyword-transformer --data-dir .\data2 --split validation --limit 0 --batch-size 32 --out .\runs\validation_repeat
```

复查权重：

```powershell
& .\.conda\python.exe .\verify_checkpoint_weights.py --run .\runs\validation_repeat
```

使用新的输出目录可保留本次已建立的基线。原始安装步骤见 `README_先跑浮点模型.md`，当前不必重新安装。

## 关键结果

- `runs/validation_full/result.json`：完整验证集准确率及样本数。
- `runs/validation_full/predictions.csv`：逐条预测和正确性。
- `runs/validation_full/predictions.npz`：全部 logits、标签和混淆矩阵。
- `runs/validation_full/first_sample_trace.npz`：第一条样本的波形、MFCC、编码器输入及 12 层输出。
- `runs/validation_full/weights_fp32.npz`：FP32 权重，Dense kernel 为 `[输入通道, 输出通道]`。
- `runs/validation_full/checkpoint_verification.json`：权重逐项一致的核验结果。
- `runs/validation_full/run_metadata.json`：代码版本、checkpoint SHA256、环境和 batch。
- `runs/validation_full/sample_manifest.json`：本次实际样本和顺序。
- `requirements-lock.txt`：实际安装的完整包版本。
- `logs/validation_full.log`：完整运行日志。

## 已处理的兼容问题

原 checkpoint 加载时会报告 Dense/GELU 对象别名不一致的旧格式警告。额外检查已确认所有 162 个权重张量与 checkpoint 精确相等，且模型输出和中间结果正常。CPU 模式下 TensorFlow 可能仍打印 CUDA 初始化失败日志，实际推理继续使用 CPU。

本机安装时使用了 `http://127.0.0.1:7890` 代理以及清华 PyPI 镜像，设置仅作用于安装命令。日常离线推理不依赖这些网络设置。旧版 Python 在当前沙箱中调用 Path.resolve 会出现权限错误，入口已改用绝对路径处理；官方模型源码未修改。

下一步：固定这套浮点模型和样本清单，补充逐算子张量导出，再实现单层定点数值对照。当前 trace 仅导出第一条样本，不是赛题要求的完整数值验证集。
