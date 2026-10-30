# 可移植的回归参考数据

这里保存运行架构比较时必需的七份历史 JSON 元数据，约 3.24 MB。
它们是历史报告原文件的逐字节副本，不是新计算或删减后的预期结果。
`manifest.json` 记录每个文件的字节数和 SHA-256；加载时校验完整性。

| 参考目录 | 必需文件 | 使用入口 |
|---|---|---|
| `fixed300_20261029_v2` | `summary.json`、`run_config.json` | `architecture_study` |
| `architecture_study_20261029_v1` | `summary.json`、`run_config.json` | `architecture_study_ln64`、`architecture_study_32k_ln64` |
| `architecture_study_ln64_20261029_v1` | `summary.json`、`run_config.json`、`historical_regression.json` | `array_grid_study` |

这些文件来自原 `hw_model/reports/` 下同名目录，只包含数值、配置、
资源账本及回归证据，不包含逐事件轨迹、源码快照、模型权重或语音数据。
大型生成报告仍由 Git 忽略。克隆仓库后，运行新研究不需要先恢复其他本地报告。

`reference_data.reference_report_path()` 只读取这里的版本化参考数据。
文件缺失或校验失败会明确报错，不自动下载，也不会用本次计算生成或替换预期值。
新报告会保存所需参考副本；源码冻结清单还会包含本目录，以支持独立重放。
