# KWT 关键词识别量化与加速器研究

本项目包含 KWT-1 的 FP32 对照推理、定点量化、逐层精度搜索，以及加速器的软件行为模型和面积预算脚本。

GitHub 仓库：<https://github.com/zhanglidan068-spec/qiansai>

## 从哪里开始

- [FP32 环境、官方模型和数据准备](README_先跑浮点模型.md)
- [本机 FP32 运行结果](README_运行结果.md)
- [W16A16、浮点非线性运行方法](README_量化运行方法.md)
- [W8A8、N16 运行方法](README_W8A8_N16运行方法.md)
- [逐层精度搜索与独立复核](README_逐层精度搜索.md)
- [硬件行为模型、配置及测试](hw_model/README.md)
- [混合精度硬件接口规格](hardware_precision_spec.md)
- [赛题面积口径审计](area_requirements_audit.md)
- [Git 与 VS Code 日常操作](README_Git与VSCode.md)

## 运行原则

只使用训练划分样本进行无梯度数值标定，使用验证划分进行调试和精度评价。当前工作禁止使用 testing 样本；仓库中保留的历史 testing 工具不属于当前运行流程。已有实验通过独立目录保留，新运行不覆盖旧结果。

浮点/量化推理依赖 Python 3.8 和 TensorFlow 2.4，具体版本见 `requirements-fp32.txt`、`requirements-lock.txt`。`hw_model` 的行为仿真仅依赖 Python 标准库，运行说明以其 README 为准。

## 仓库范围

Git 保存源码、配置、测试、说明文档和 `.vscode` 配置。以下内容由 `.gitignore` 排除，仍保留在本机：

- `.conda/` 等环境与依赖缓存；
- `data2/` 数据集和 `downloads/` 下载文件；
- `keyword-transformer/` 官方上游源码、配置和预训练权重；
- `runs/`、`logs/`、`hw_model/reports/`、临时目录和生成文件。

因此，仅克隆本仓库不会包含模型权重、数据集和历史实验输出。完整推理前，请按 FP32 手册准备官方 `ARM-software/keyword-transformer`，参考版本 `237e4097`，以及对应 KWT-1 权重和数据。硬件行为模型可按自身说明独立运行。

硬件行为模型的回归参考摘要和配置保存在 `hw_model/reference_results/`，随仓库提供；它们用于校验此前结果，不包含训练样本或大体积 trace。使用 Python 3.8+ 可直接运行 `python -m hw_model.array_grid_study --out hw_model/reports/array_grid_new --workers 4`，生成本机的新验证报告。

## 面积估算

`estimate_kwt_area.py` 提供公开参考工艺的预算；`estimate_kwt_smic40_26x4.py` 针对 SMIC40-2025 目标和 26×4 阵列建立 40nm 参考预算。两者均不等同于指定工艺综合结果。实际面积、150 MHz 时序和赛题达标情况需使用赛方标准单元库、存储宏及完整 RTL 验证。
