# KWT-1 浮点对照模型：Windows + VS Code 操作手册

当前目标是加载题目指定的官方 KWT-1 预训练权重，运行 FP32 推理，保存后续定点模型要比较的数据。先用 CPU 即可，不需要配置 CUDA。

本机已实际跑通：完整验证集 4340/4445，准确率 97.6378%。实际目录是 `D:\codex\kwt_fp32`，环境和数据已准备完毕，请优先阅读 `README_运行结果.md`。以下安装步骤保留供重新部署参考，示例目录 `D:\kwt_fp32` 与本机实际目录不同。

## 1. 固定使用什么模型

- 官方仓库：<https://github.com/ARM-software/keyword-transformer>
- 代码版本：`237e4097`
- 权重前缀：`models_data_v2_12_labels/kwt1/best_weights`
- 配置：12 层，1 个注意力头，嵌入维度 64，FFN 隐层 256，PostNorm，精确 GELU。
- 输入：16 kHz、1 秒、单声道音频；官方前处理生成 98×40 MFCC。
- 编码器：加入 class token 后，输入及每层输出均为 99×64。
- 输出：12 个类别的 logits；类别顺序为 silence、unknown、yes、no、up、down、left、right、on、off、stop、go。

仓库的 `eval.sh` 默认配置为 KWT-3。原 `model_train_eval.py` 也不会读取已有 flags.json 来恢复配置，而会覆盖它并执行额外导出。因此本启动包提供独立入口，读取官方 flags.json，复用官方前处理、模型实现和权重，仅执行推理及结果导出。

## 2. 准备目录，在 VS Code 打开

建议把本启动包内的文件复制到 `D:\kwt_fp32`，让文件直接位于该目录下：

```text
D:\kwt_fp32\
├─ run_kwt_fp32.py
├─ requirements-fp32.txt
├─ README_先跑浮点模型.md
└─ .vscode\
   ├─ settings.json
   └─ launch.json
```

随后使用 VS Code 的“文件 → 打开文件夹”打开 `D:\kwt_fp32`。安装 Microsoft 发布的 Python 扩展，打开“终端 → 新建终端”，选择 PowerShell。

以下命令都在该 PowerShell 终端执行。可以先确认：

```powershell
Set-Location D:\kwt_fp32
Get-Location
```

不要用编辑器右上角的第三方 Code Runner 按钮。先按本文用终端运行，能明确使用哪个 Python 和工作目录。

## 3. 创建独立的 Python 3.8 环境

已发现你电脑上的 Anaconda 位于 `D:\anaconda3`。因此不必先处理 conda activate，可以直接调用完整路径创建环境：

```powershell
& D:\anaconda3\Scripts\conda.exe create --prefix D:\kwt_fp32\.conda python=3.8 -y
```

确认新环境的 Python 可以启动：

```powershell
& D:\kwt_fp32\.conda\python.exe --version
```

应显示 `Python 3.8.x`，然后安装依赖：

```powershell
& .\.conda\python.exe -m pip install pip==24.3.1 setuptools==59.8.0 wheel==0.37.1
& .\.conda\python.exe -m pip install -r .\requirements-fp32.txt
& .\.conda\python.exe -m pip check
```

`pip check` 应显示 `No broken requirements found.`。检查核心包与解释器：

```powershell
& .\.conda\python.exe -c "import sys, tensorflow as tf, tensorflow_addons as tfa, numpy as np; print(sys.executable); print('TF', tf.__version__, 'TFA', tfa.__version__, 'NumPy', np.__version__)"
```

预期版本：

```text
D:\kwt_fp32\.conda\python.exe
TF 2.4.0 TFA 0.12.1 NumPy 1.19.5
```

官方历史 requirements 指定 TensorFlow 2.4.0rc1；当前 PyPI 未提供该版本。本包采用有 Windows CPython 3.8 安装包的 2.4.0 正式版作为兼容方案，不能声称与原 rc1 运行时逐位相同。脚本会记录运行时版本。若后续赛方提供统一软件环境，应以赛方环境重新建立基准。

不要直接安装最新版 TensorFlow/Keras，也不要直接照装官方未锁定版本的 requirements.txt。独立推理入口不需要训练所用的 Transformers 或画模型图所用的 Graphviz。

在 VS Code 按 `Ctrl+Shift+P` → `Python: Select Interpreter` → 输入解释器路径：

```text
D:\kwt_fp32\.conda\python.exe
```

解释器选择主要影响编辑器和调试器；下面始终显式调用 `.conda\python.exe`，避免终端误用 base 环境或 Windows Store 的 Python 占位程序。

## 4. 获取指定版本的完整源码及权重

```powershell
git --version
git clone https://github.com/ARM-software/keyword-transformer.git keyword-transformer
git -C .\keyword-transformer checkout 237e4097
git -C .\keyword-transformer rev-parse HEAD
Get-ChildItem .\keyword-transformer\models_data_v2_12_labels\kwt1\best_weights*
```

提交号应以 `237e4097` 开头；权重目录至少包含：

```text
best_weights.index
best_weights.data-00000-of-00001
flags.json
labels.txt
```

其中 `best_weights` 是 checkpoint 的前缀，不是单个 .h5 文件。加载时不能把路径写成 `best_weights.index`。

需要完整仓库，不能只复制 kws_transformer.py 和 transformer_utils.py，前处理及模型注册模块还引用其他源码。

如果终端找不到 git，你电脑上的 Codex 自带 Git 可临时用于本次终端：

```powershell
$env:Path = 'C:\Users\iluoning\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\git\cmd;' + $env:Path
git --version
```

这是当前终端 PATH 的临时补充。也可以使用已安装的 Git for Windows。

## 5. 第一次运行：只检查加载与前向传播

此步骤不需要下载数据集：

```powershell
& .\.conda\python.exe .\run_kwt_fp32.py --repo .\keyword-transformer --smoke --out .\runs\smoke
```

脚本会完成以下检查：

1. 核对 KWT-1 配置、标签顺序和 checkpoint 文件。
2. 构建官方模型，严格确认当前模型变量均匹配 checkpoint。
3. 输入一段全零的 1 秒波形，经过官方前处理和 12 层编码器。
4. 复用同一组权重提取中间层，比较提取路径与原模型的最终 CLS 及 logits。
5. 检查中间张量和输出没有 NaN/Inf，并导出权重与第一条输入的中间结果。

你应看到类似下面的形状和最终 PASS，而不是某个固定预测类别：

```text
Checkpoint loaded; all current model objects matched.
MFCC: (1, 98, 40) Encoder input: (1, 99, 64) Layer 12: (1, 99, 64)
PASS: inference completed.
```

全零输入用于检查流程，不代表语音识别准确率。此模式下 result.json 的 accuracy 为 null。

## 6. 获取 Speech Commands v2 完整数据集

如果已有完整数据集，可跳过下载，后续 `--data-dir` 指向已有目录。

否则先在 PowerShell 下载官方压缩包。它是数 GB 级文件，预留压缩包及解压目录的空间：

```powershell
New-Item -ItemType Directory -Force .\downloads, .\data2
curl.exe -L --fail -C - "https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz" -o .\downloads\speech_commands_v0.02.tar.gz
```

确认下载成功后再解压：

```powershell
tar.exe -xzf .\downloads\speech_commands_v0.02.tar.gz -C .\data2
```

没有 tar.exe 时可使用 7-Zip 解压 .gz，再解包其中的 .tar。最终目录应直接包含：

```text
D:\kwt_fp32\data2\
├─ yes\*.wav
├─ no\*.wav
├─ up\*.wav
├─ ...其余单词目录，共 35 个...
├─ _background_noise_\
├─ validation_list.txt
└─ testing_list.txt
```

虽然模型输出 12 类，也必须准备完整 35 单词数据集。其余单词用于构造 unknown 类，不能只下载十个目标关键词。

脚本复用仓库 AudioProcessor 的哈希划分及 silence/unknown 构造逻辑，不另外自行按列表拼接标签。sample_manifest.json 会记录本次实际使用的样本及顺序，方便后续对照；不要假设不同平台的 unknown 抽样顺序天然相同。

## 7. 先跑 32 条验证样本，再跑完整验证集

先保守使用 batch 1：

```powershell
& .\.conda\python.exe .\run_kwt_fp32.py --repo .\keyword-transformer --data-dir .\data2 --split validation --limit 32 --batch-size 1 --out .\runs\validation_32
```

看到处理完 `32/32`、正常打印 accuracy 和 PASS 后，再评估完整 validation：

```powershell
& .\.conda\python.exe .\run_kwt_fp32.py --repo .\keyword-transformer --data-dir .\data2 --split validation --limit 0 --batch-size 32 --out .\runs\validation_full
```

`--limit 0` 表示完整划分；尾批会补齐后只统计有效样本，不丢弃末尾数据。内存不足时将 batch-size 改为 8 或 1，并使用新的输出目录。

32 条样本的准确率只用于流程检查。完整 validation 的结果才适合做当前开发基线。正式 testing 留待实现及参数冻结后评估，不用它选择量化参数；量化校准使用训练集。

脚本支持 `--split testing`，需要时将完整验证命令中的 split 和输出目录相应改为 testing 即可。

每次运行会覆盖指定 out 目录内的同名文件，比较不同环境或 batch 时应使用不同目录。浮点运算可能随运行时、设备或 batch 改变而出现小差异；固定参考时应连同这些条件一起记录。

## 8. 结果在哪里，分别有什么用

以 `runs\validation_full` 为例：

| 文件 | 作用 |
|---|---|
| result.json | 样本数、准确率、耗时和 trace 与原 CLS 的误差 |
| predictions.csv | 每条样本的真实标签、预测标签和是否正确 |
| predictions.npz | 全部 logits、目标标签、12×12 混淆矩阵 |
| first_sample_trace.npz | 第一条有效输入的波形、MFCC、编码器输入、12 层输出、CLS、logits |
| trace_shapes.json | 中间张量名称和形状 |
| weights_fp32.npz | 从已加载的官方模型导出的 FP32 权重 |
| weights_manifest.json | 权重名称、维度、数据类型和 Dense 核布局 |
| sample_manifest.json | 本次评估样本、标签和顺序；silence 标明使用合成零波形 |
| inference_flags.json | 实际推理配置；原官方 flags.json 保持不变 |
| run_metadata.json | 代码版本、核心依赖、checkpoint SHA256、设备和 batch |
| environment_freeze.txt | 当前环境的包版本，便于复现 |

第一条 trace 的主要形状（保留 batch 维度）：

```text
waveform          [1, 16000]
mfcc              [1, 98, 40]
encoder_input     [1, 99, 64]
layer_01_output   [1, 99, 64]
...
layer_12_output   [1, 99, 64]
official_cls      [1, 64]
logits            [1, 12]
```

权重里的 Dense kernel 使用 `[输入通道, 输出通道]` 布局，计算 `y = x @ kernel + bias`。以后导入 NumPy、PyTorch 或 RTL 时先明确布局，不要直接按另一框架的习惯解释。

此启动包仅保存第一条输入的逐层激活，不是完整的赛题 600 条数值验证集，也不是逐算子的 RTL 黄金数据生成器。等浮点基线跑通后，再固定样本清单并扩展 Q/K/V、注意力分数、Softmax、LayerNorm 和 FFN 等逐算子导出。

## 9. VS Code 调试及常见问题

终端命令是首选运行方式。启动包另附两个 launch 配置，可用于 checkpoint smoke 和 32 条 validation；F5 需要支持 Python 3.8 的 Python Debugger 扩展。若当前调试器要求更高 Python 版本，先继续使用本文终端命令，不要为了调试器升级模型运行环境。

| 现象 | 优先检查 |
|---|---|
| `No matching distribution found` | 是否确实使用 Windows x64 Python 3.8；有没有误装原 requirements 中的 rc1 |
| TensorFlow 提示 `DLL load failed` | 解释器是否 64 位；根据具体缺失 DLL 检查 Microsoft Visual C++ x64 运行库 |
| 找不到 kws_streaming | --repo 是否指向完整仓库根目录 |
| checkpoint 恢复失败 | commit、KWT-1 目录、权重数据分片是否齐全；不要改用 skip_mismatch |
| 找不到 WAV | --data-dir 是否直接包含 yes/no 等目录，而非它们的上层 |
| 数据目录数量不符 | 是否完整解压 v0.02 的 35 单词；是否多套了一层目录 |
| 程序 CPU 占用高 | CPU 推理正常现象，关注处理计数；内存压力大时减小 batch |
| accuracy 很低 | 先检查模型、数据版本、标签顺序、前处理和增强是否一致，保留 result 与配置排查 |
| 出现旧 API 的 deprecated 警告 | 旧版本项目可能有此类警告，以 traceback、加载断言及最后 PASS 为准 |

不要把安装失败后“升级全部包”作为修复方法。保留完整 traceback，先确认解释器路径及版本。

## 10. 做到哪一步再进入定点化

- checkpoint 严格匹配，零输入检查通过。
- 32 条真实语音推理完成。
- 完整 validation 得到并保存准确率，样本数和标签分布合理。
- 99×64 的逐层输出均有限值，提取路径与官方 CLS/logits 比较通过。
- 保存代码版本、checkpoint 哈希、依赖版本和样本清单。

接下来先以这些浮点张量为参考实现单层定点数值模型，再逐步扩展到 12 层和硬件实现。

## 来源

- [指定版本的官方实现](https://github.com/ARM-software/keyword-transformer/tree/237e4097)
- [官方 KWT-1 配置](https://github.com/ARM-software/keyword-transformer/blob/237e4097/models_data_v2_12_labels/kwt1/flags.json)
- [官方依赖文件](https://github.com/ARM-software/keyword-transformer/blob/237e4097/requirements.txt)
- [TensorFlow 2.4.0 包信息](https://pypi.org/project/tensorflow/2.4.0/)
- [TensorFlow Addons 0.12.1 包信息](https://pypi.org/project/tensorflow-addons/0.12.1/)
