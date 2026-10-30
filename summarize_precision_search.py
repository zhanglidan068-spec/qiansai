"""Summarize completed precision trials; read numerical outputs, never audio.

Run only after search_result.json exists. Existing model sources, previous
experiments, and audio files are never modified. All new outputs stay inside
the supplied precision-search directory.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import shutil
import sys
import time

import numpy as np


ROOT = Path(__file__).absolute().parent
STAGES = ["patch_output", "encoder_input"] + ["layer_{:02d}_output".format(i) for i in range(1, 13)] + ["logits"]


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def write_csv(path, rows):
    if not rows:
        raise ValueError("Cannot write an empty comparison table")
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        fields = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def compare_arrays(reference_path, candidate_path, chunk):
    reference = np.load(str(reference_path), mmap_mode="r", allow_pickle=False)
    candidate = np.load(str(candidate_path), mmap_mode="r", allow_pickle=False)
    if reference.shape != candidate.shape or reference.dtype != np.float32 or candidate.dtype != np.float32:
        raise ValueError("Incompatible layer arrays: {} / {}".format(reference_path, candidate_path))
    sums = dict(abs_error=0., squared_error=0., reference_squared=0., candidate_squared=0., dot=0.)
    maximum_error, reference_maximum, candidate_maximum = 0., 0., 0.
    nonzero = 0
    for offset in range(0, len(reference), chunk):
        base = np.asarray(reference[offset:offset + chunk], dtype=np.float64)
        got = np.asarray(candidate[offset:offset + chunk], dtype=np.float64)
        if not np.isfinite(base).all() or not np.isfinite(got).all():
            raise FloatingPointError("Nonfinite layer array")
        error = got - base
        sums["abs_error"] += float(np.abs(error).sum())
        sums["squared_error"] += float(np.square(error).sum())
        sums["reference_squared"] += float(np.square(base).sum())
        sums["candidate_squared"] += float(np.square(got).sum())
        sums["dot"] += float((base * got).sum())
        maximum_error = max(maximum_error, float(np.abs(error).max()))
        reference_maximum = max(reference_maximum, float(np.abs(base).max()))
        candidate_maximum = max(candidate_maximum, float(np.abs(got).max()))
        nonzero += int(np.count_nonzero(error))
    count = int(reference.size)
    rel_l2 = math.sqrt(sums["squared_error"] / sums["reference_squared"]) if sums["reference_squared"] else (0. if not sums["squared_error"] else None)
    norm_product = math.sqrt(sums["reference_squared"] * sums["candidate_squared"])
    cosine = sums["dot"] / norm_product if norm_product else (1. if not sums["squared_error"] else 0.)
    result = dict(samples=int(len(reference)), elements=count, shape="x".join(map(str, reference.shape)),
                  mae=sums["abs_error"] / count, rmse=math.sqrt(sums["squared_error"] / count),
                  max_abs_error=maximum_error, relative_l2_error=rel_l2,
                  cosine_similarity=max(-1., min(1., cosine)),
                  changed_elements=nonzero, changed_fraction=nonzero / float(count),
                  fp32_max_abs=reference_maximum, quantized_max_abs=candidate_maximum)
    reference._mmap.close()
    candidate._mmap.close()
    return result


def verify_trial(run, row, fresh_logits, targets, manifest, require_full=False, trial_root=None):
    directory = (trial_root or run / "trials") / row["trial"]
    indices = np.load(str(directory / "sample_indices.npy"), allow_pickle=False)
    with np.load(str(directory / "predictions.npz"), allow_pickle=False) as saved:
        logits, labels = saved["logits"], saved["target_ids"]
        if len(indices) != row["samples"] or logits.shape != (len(indices), fresh_logits.shape[1]):
            raise AssertionError("Trial sample count mismatch: " + row["trial"])
        np.testing.assert_array_equal(labels, targets[indices])
        if not np.isfinite(logits).all():
            raise FloatingPointError("Nonfinite saved trial logits")
        correct = int((logits.argmax(1) == labels).sum())
        baseline_correct = int((fresh_logits[indices].argmax(1) == labels).sum())
        if correct != row["correct"] or baseline_correct != row["fp32_correct"]:
            raise AssertionError("Saved trial accuracy does not reproduce: " + row["trial"])
        if require_full:
            np.testing.assert_array_equal(indices, np.arange(len(manifest)))
    return dict(trial=row["trial"], samples=int(len(indices)), recomputed_correct=correct,
                recomputed_fp32_correct=baseline_correct, verified=True)


def pareto_rows(rows):
    feasible = [row for row in rows if row["passes_2pp"] and row["is_full_validation"]]
    result = []
    for row in feasible:
        dominated = any(
            other["correct"] >= row["correct"] and other["matrix_weight_kib"] <= row["matrix_weight_kib"] and
            other["arithmetic_work_proxy"] <= row["arithmetic_work_proxy"] and
            (other["correct"] > row["correct"] or other["matrix_weight_kib"] < row["matrix_weight_kib"] or
             other["arithmetic_work_proxy"] < row["arithmetic_work_proxy"])
            for other in feasible)
        if not dominated:
            result.append(row)
    return result


def precision_table(config):
    widths = config.get("operand_bits", config.get("matrix_operand_bits"))
    rows = []
    for key, bits in sorted(widths.items()):
        accelerated = key.startswith("layer_")
        fractions = config["fractional_bits"][key]
        values = fractions if accelerated and isinstance(fractions, list) else [fractions]
        if not accelerated:
            values = [None]
        for channel, fraction in enumerate(values):
            fraction = int(fraction) if accelerated else None
            rows.append(dict(operand=key, scope="encoder accelerator" if accelerated else "FP32 software",
                             output_channel=channel if accelerated and isinstance(fractions, list) else None,
                             effective_bits=int(bits) if accelerated else 32, fractional_bits=fraction,
                             quantization_step=math.ldexp(1., -fraction) if accelerated else None,
                             symmetric_code_min=-((1 << (int(bits) - 1)) - 1) if accelerated else None,
                             symmetric_code_max=((1 << (int(bits) - 1)) - 1) if accelerated else None))
    return rows


def per_channel_summary(run, baseline_dir, fresh_logits, targets, manifest, chunk, primary_rows):
    path = run / "per_channel_search_result.json"
    if not path.is_file():
        return None
    from area_proxy import precision_accounting
    payload = load_json(path)
    if payload["testing_samples_used"] != 0 or payload["samples"] != len(targets):
        raise AssertionError("Invalid per-channel summary samples")
    directory = run / "per_channel_search"
    def normalize(row):
        return dict(row, trial=row["name"] + "__" + row["scope"], phase="per_channel",
                    scale_metadata_bytes=int(row.get("per_channel_scale_bytes", 6912)),
                    matrix_and_weight_F_kib=row["matrix_weight_kib"] + int(row.get("per_channel_scale_bytes", 6912)) / 1024.)
    full_rows = [normalize(row) for row in payload["full_validation_candidates"]]
    screen_rows = [normalize(row) for row in load_json(directory / "experiment_results.json") if row["scope"] == "screen"]
    checks = [verify_trial(run, row, fresh_logits, targets, manifest, False, directory) for row in screen_rows]
    checks += [verify_trial(run, row, fresh_logits, targets, manifest, True, directory) for row in full_rows]
    comparisons, exports, repeat_checks = {}, {}, []
    for role, winner_key in [("accuracy", "accuracy_winner"), ("efficiency", "efficient_winner")]:
        winner = payload[winner_key]
        final = normalize(payload["finals"][winner["name"]])
        final_dir = directory / final["trial"]
        checks.append(verify_trial(run, final, fresh_logits, targets, manifest, True, directory))
        with np.load(str(directory / (winner["name"] + "__full") / "predictions.npz"), allow_pickle=False) as old, np.load(str(final_dir / "predictions.npz"), allow_pickle=False) as new:
            np.testing.assert_array_equal(old["logits"], new["logits"])
        repeat_checks.append(dict(role="pc_" + role, name=winner["name"], logits_bitwise_equal=True))
        rows = []
        for stage in STAGES:
            got = compare_arrays(baseline_dir / (stage + ".npy"), final_dir / "layer_outputs" / (stage + ".npy"), chunk)
            rows.append(dict(stage=stage, trial=final["trial"], **got))
            print("pc_{}: {} MAE={:.6g} relL2={}".format(role, stage, got["mae"], got["relative_l2_error"]), flush=True)
        comparisons[role] = rows
        write_csv(run / ("pc_" + role + "_layer_comparison.csv"), rows)
        write_json(run / ("pc_" + role + "_layer_comparison.json"), rows)
        config = load_json(final_dir / "precision_config.json")
        precisions = precision_table(config)
        write_csv(run / ("pc_" + role + "_operand_precision.csv"), precisions)
        write_json(run / ("pc_" + role + "_operand_precision.json"), precisions)
        packed = final_dir / "weights_packed.bin"
        fractions = final_dir / "weight_channel_F_int8.bin"
        layout = load_json(final_dir / "weights_manifest.json")
        if packed.stat().st_size != layout["total_bytes"] or fractions.stat().st_size != layout["channel_scale_bytes"]:
            raise AssertionError("Per-channel packed storage mismatch")
        values = np.frombuffer(fractions.read_bytes(), dtype=np.int8)
        for key, entry in layout["tensors"].items():
            start, length = entry["f_table_offset"], entry["f_table_length"]
            np.testing.assert_array_equal(values[start:start + length], config["fractional_bits"][key])
        if len(values) != 6912:
            raise AssertionError("Expected 6912 encoder column F values")
        stats = precision_accounting(dict(default_weight_bits=config["policy"]["w"], default_activation_bits=config["policy"]["a"],
            av_probability_bits=config["policy"]["p"], av_fractional_bits=config["policy"]["pf"], matrix_operand_bits=config["matrix_operand_bits"]))[1]
        exports[role] = dict(trial=final["trial"], final_directory=final_dir.relative_to(run).as_posix(),
                             packed_weight_bytes=packed.stat().st_size, channel_F_table_bytes=fractions.stat().st_size,
                             total_weight_and_F_bytes=packed.stat().st_size + fractions.stat().st_size,
                             area_accounting=stats)
    write_csv(run / "per_channel_screening_comparison.csv", screen_rows)
    write_json(run / "per_channel_screening_comparison.json", screen_rows)
    write_csv(run / "per_channel_full_validation_comparison.csv", full_rows)
    write_json(run / "per_channel_full_validation_comparison.json", full_rows)
    combined = [dict(row, phase="per_tensor", scale_metadata_bytes=72,
                     matrix_and_weight_F_kib=row["matrix_weight_kib"] + 72. / 1024.) for row in primary_rows] + full_rows
    write_csv(run / "all_full_validation_comparison.csv", [dict(id=index, **row) for index, row in enumerate(combined, 1)])
    write_json(run / "all_full_validation_comparison.json", combined)
    valid = [row for row in combined if row["passes_2pp"]]
    accuracy = sorted(valid, key=lambda row: (-row["accuracy"], row["matrix_and_weight_F_kib"], row["arithmetic_work_proxy"]))[0]
    # This is explicitly an arithmetic-resource proxy selection, not a claim
    # of minimum mapped area or measured latency. F-table bytes remain visible.
    resource = sorted(valid, key=lambda row: (row["arithmetic_work_proxy"], row["matrix_and_weight_F_kib"], -row["accuracy"]))[0]
    frontier = []
    for row in valid:
        if not any(other["correct"] >= row["correct"] and other["matrix_and_weight_F_kib"] <= row["matrix_and_weight_F_kib"] and
                   other["arithmetic_work_proxy"] <= row["arithmetic_work_proxy"] and
                   (other["correct"] > row["correct"] or other["matrix_and_weight_F_kib"] < row["matrix_and_weight_F_kib"] or other["arithmetic_work_proxy"] < row["arithmetic_work_proxy"])
                   for other in valid):
            frontier.append(row)
    write_json(run / "pareto_frontier_all_candidates.json", frontier)
    recommendations = dict(accuracy_priority=accuracy, arithmetic_resource_priority=resource,
                           per_channel_balanced=normalize(payload["accuracy_winner"]),
                           selection_note="Accuracy and arithmetic-work proxy among evaluated full-validation candidates; no measured area optimum claim",
                           weight_F_metadata_note="6912 signed int8 channel F entries for PC, versus 72 scalar weight F entries for per-tensor; activation/LN metadata excluded from both",
                           area_compliance_verified=False)
    write_json(run / "combined_recommendations.json", recommendations)
    return dict(summary=payload, screen_rows=screen_rows, full_rows=full_rows, combined_rows=combined,
                comparisons=comparisons, exports=exports, verification_checks=checks,
                repeat_checks=repeat_checks, recommendations=recommendations, frontier=frontier)


def optional_plots(run, sensitivities, full_rows, baseline_accuracy):
    if importlib.util.find_spec("matplotlib") is None:
        return dict(generated=False, reason="Matplotlib is unavailable in this runtime; no package was installed")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    widths = sorted(set(row["effective_bits"] for row in sensitivities))
    heat = np.full((len(widths), 12), np.nan)
    for row in sensitivities:
        heat[widths.index(row["effective_bits"]), row["layer"] - 1] = row["accuracy_drop_pp"]
    fig, axis = plt.subplots(figsize=(12, 3.7), constrained_layout=True)
    image = axis.imshow(heat, cmap="YlOrRd", aspect="auto")
    axis.set_xticks(np.arange(12)); axis.set_xticklabels(np.arange(1, 13))
    axis.set_yticks(np.arange(len(widths))); axis.set_yticklabels([str(value) for value in widths])
    axis.set_xlabel("Quantized encoder block (other blocks FP32)")
    axis.set_ylabel("Effective matrix bits")
    axis.set_title("Single-block sensitivity: {} validation samples (screening only)".format(sensitivities[0]["samples"]))
    for y in range(len(widths)):
        for x in range(12):
            axis.text(x, y, "{:.2f}".format(heat[y, x]), ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=axis, label="Accuracy drop (percentage points)")
    for suffix in ("png", "svg"):
        fig.savefig(str(run / ("layer_sensitivity_heatmap." + suffix)), dpi=170)
    plt.close(fig)
    fig, axis = plt.subplots(figsize=(10, 6), constrained_layout=True)
    for index, row in enumerate(full_rows, 1):
        axis.scatter(row["matrix_weight_kib"], 100. * row["accuracy"], s=65)
        axis.annotate(str(index), (row["matrix_weight_kib"], 100. * row["accuracy"]), xytext=(4, 4), textcoords="offset points")
    axis.axhline(100. * baseline_accuracy, color="gray", linestyle="--", label="FP32")
    axis.axhline(100. * baseline_accuracy - 2., color="red", linestyle=":", label="FP32 - 2 pp")
    axis.set_xlabel("Packed encoder matrix weights (KiB); not ASIC area")
    axis.set_ylabel("Full-validation accuracy (%)")
    axis.set_title("Evaluated precision candidates; see full_validation_comparison.csv for IDs")
    axis.legend()
    for suffix in ("png", "svg"):
        fig.savefig(str(run / ("precision_tradeoff." + suffix)), dpi=170)
    plt.close(fig)
    return dict(generated=True, files=["layer_sensitivity_heatmap.png", "layer_sensitivity_heatmap.svg", "precision_tradeoff.png", "precision_tradeoff.svg"])


def percent(value):
    return "{:.4f}%".format(100. * value)


def policy_description(policy):
    text = "默认 W{} / A{}，AV 概率 {} bit、F={}，非线性 N16".format(
        policy["default_weight_bits"], policy["default_activation_bits"],
        policy["av_probability_bits"], policy["av_fractional_bits"])
    if policy.get("layer_overrides"):
        text += "；逐层覆盖：" + json.dumps(policy["layer_overrides"], ensure_ascii=False, sort_keys=True)
    if policy.get("matrix_operand_bits"):
        text += "；另有逐操作数覆盖，见 policy.json"
    return text


def report_text(run, summary, prep, full_rows, sensitivities, comparison, area_stats, plots, pareto):
    n = summary["samples"]
    accuracy = summary["accuracy_winner"]
    efficient = summary["efficient_winner"]
    baseline = summary["fp32_accuracy"]
    lines = ["# 逐层量化精度搜索结果", "",
        "本报告根据已经完成的数值实验生成。所有结果目录独立保留，未覆盖前两次实验；testing 样本使用量为 **0**。", "",
        "本次完整验证包含 **{}** 条样本，FP32 为 **{} / {}，{}**。允许下降不超过 **2 个百分点**，离散样本数要求是至少 **{} / {}**。".format(n, summary["fp32_correct"], n, percent(baseline), summary["minimum_correct"], n), "",
        "## 两个推荐候选", "",
        "| 角色 | 配置 | 正确数 | 准确率 | 相对 FP32 下降 | 编码器矩阵权重 | 算术位宽工作量代理 |", "|---|---|---:|---:|---:|---:|---:|"]
    for label, row in [("精度优先", accuracy), ("权重存储优先", efficient)]:
        lines.append("| {} | `{}` | {} / {} | {} | {:.4f} 个百分点 | {:.2f} KiB | {:.3f} × W8A8 |".format(label, row["name"], row["correct"], n, percent(row["accuracy"]), row["accuracy_drop_pp"], row["matrix_weight_kib"], row["arithmetic_work_proxy"]))
    lines += ["", "精度优先候选按准确率、权重存储、算术工作量依次排序；权重存储优先候选在满足 2 个百分点约束的候选中，按权重大小、算术工作量、准确率依次排序。它们是**已测试候选中的选择**，没有穷尽所有可能的逐层位宽组合。", ""]
    for role, label, row in [("accuracy", "精度优先", accuracy), ("efficiency", "权重存储优先", efficient)]:
        policy = load_json(run / "trials" / row["trial"] / "policy.json")
        lines += ["- **{}**：{}。完整配置见 [{}](trials/{}/policy.json)。".format(label, policy_description(policy), row["name"], row["trial"])]
    lines += ["", "## 如何读实验结果", "",
        "1. 从训练划分抽取 **{}** 条样本进行无梯度范围标定，固定每个操作数的 2 的幂尺度。模型权重没有重新训练；验证集未用于范围标定。".format(prep["calibration_samples"]),
        "2. **{}** 条验证样本用于快速筛选（每类抽样）；完成 **{}** 次单层实验：12 个编码器块 × 8 / 10 / 12 / 16 bit。每次只量化一个编码器块，其余块执行真正的 FP32。该实验同时包含被测块的 N16 非线性。".format(summary["screening_samples"], summary["layer_sensitivity_experiments"]),
        "3. 完整编码器候选先在筛选样本上比较，再对 **{}** 个配置执行全部 **{}** 条验证样本；只有完整验证结果用于证明 2 个百分点约束。".format(len(full_rows), n),
        "4. 两个推荐候选重新执行完整前向计算并保存 15 个阶段的全样本数组；final logits 与各自此前 full logits 逐位核对。每个候选都重新计算预测，没有读取旧预测来决定输出。", "",
        "**{} 条筛选样本上的结果不等于完整验证结果**，不能据此声称达标。单层敏感性也不能简单相加预测全模型精度。".format(summary["screening_samples"]), "",
        "## 完整验证配置", "",
        "| ID | 配置 | 正确数 | 准确率 | 下降 / 个百分点 | 满足 ≤2 个百分点 | 矩阵权重 / KiB | 算术工作量代理 |", "|---:|---|---:|---:|---:|---|---:|---:|"]
    for index, row in enumerate(full_rows, 1):
        lines.append("| {} | `{}` | {} | {} | {:.4f} | {} | {:.2f} | {:.3f} |".format(index, row["name"], row["correct"], percent(row["accuracy"]), row["accuracy_drop_pp"], "是" if row["passes_2pp"] else "否", row["matrix_weight_kib"], row["arithmetic_work_proxy"]))
    lines += ["", "完整验证的存储 / 算术工作量 / 精度非支配候选保存在 [pareto_frontier.json](pareto_frontier.json)，数量为 {}。这里的 Pareto 关系使用数值代理，尚未引入实测面积或吞吐率。".format(len(pareto)), "",
        "## 逐层敏感性", "",
        "下表为只量化某一层时，相对该筛选子集 FP32 的准确率下降，单位为**百分点**。8 bit 列仍采用 AV 概率 F=6，10 / 12 / 16 bit 分别采用 F=9 / 11 / 15。", "",
        "| 编码器块 | 8 bit | 10 bit | 12 bit | 16 bit |", "|---:|---:|---:|---:|---:|"]
    sensitivity_map = {(row["layer"], row["effective_bits"]): row for row in sensitivities}
    for layer in range(1, 13):
        lines.append("| {} | {} |".format(layer, " | ".join("{:.4f}".format(sensitivity_map[layer, bits]["accuracy_drop_pp"]) for bits in (8, 10, 12, 16))))
    lines += ["", "48 次实验的正确数、准确率、耗时和代理成本见 [layer_sensitivity_complete.csv](layer_sensitivity_complete.csv)。候选筛选与完整验证的所有配置见 [experiment_results.csv](experiment_results.csv)。", "",
        "## 每层输出误差", "",
        "基准使用上次实验保存的完整 FP32 层数组，比较前已核对样本顺序，并验证其 logits 与本次新算出的 FP32 logits 一致。误差按全部样本、全部元素分块累计；统计使用 float64。", "",
        "| 角色 | 阶段 | MAE | RMSE | 最大绝对误差 | 相对 L2 | 余弦相似度 |", "|---|---|---:|---:|---:|---:|---:|"]
    for role, label in [("accuracy", "精度优先"), ("efficiency", "权重存储优先")]:
        for row in comparison[role]:
            lines.append("| {} | {} | {:.5g} | {:.5g} | {:.5g} | {} | {:.8f} |".format(label, row["stage"], row["mae"], row["rmse"], row["max_abs_error"], "未定义" if row["relative_l2_error"] is None else "{:.5g}".format(row["relative_l2_error"]), row["cosine_similarity"]))
    lines += ["", "完整数据见 [accuracy_layer_comparison.csv](accuracy_layer_comparison.csv)、[efficiency_layer_comparison.csv](efficiency_layer_comparison.csv)，JSON 版本同时保留。", "",
        "## 运算边界与硬件含义", "",
        "- 加速范围是 **12 个编码器块**。MFCC、patch 投影和分类头在软件中使用 FP32；偏置、残差、位置 / 类别嵌入和算子之间的数值传递采用 FP32 表示。分类头输入已经包含前面量化累积的误差，所以其 logits 仍会变化。",
        "- 编码器矩阵乘的操作数按指定有效位宽量化，采用对称有符号范围、最近偶数舍入和饱和。软件以 int64 容器执行真实整数乘加，避免溢出；有效位宽和小数位数保存在每个 trial 的 quantization.json。",
        "- Softmax、GELU、LayerNorm 使用 N16 的整数 / 查表实现，边界为 int16；宽中间值使用 int64。共享紧凑 GELU 表为 **1025 × int16 = 2050 字节，约 2 KiB**，共享指数表为 **4096 × int16 = 8192 字节，8 KiB**。两张表合计 **10242 字节**，不包括控制、地址和插值逻辑。",
        "- 概率进入 AV 之前再次量化，位宽和 F 依候选而定。F=6 保留为消融配置；更高 F 的结果用于检查概率舍入是否是主要损失来源。",
        "- 打包权重已按实际位数导出到 [accuracy_export](accuracy_export) 和 [efficiency_export](efficiency_export)，其中 6 bit 等非整字节位宽真实按位打包。它们只包含编码器矩阵权重，不等于全部参数或 SRAM 总需求。", "",
        "| 角色 | 打包矩阵权重 / 字节 | 权重 / KiB | 最大输入位宽 | 最大第二操作数位宽 | 全范围安全累加位数 |", "|---|---:|---:|---:|---:|---:|"]
    for role, label in [("accuracy", "精度优先"), ("efficiency", "权重存储优先")]:
        stats = area_stats[role]
        packed = summary["hardware_weight_exports"][role]["packed_weight_bytes"]
        lines.append("| {} | {} | {:.2f} | {} | {} | {} |".format(label, packed, stats["matrix_weight_kib"], stats["maximum_a_bits"], stats["maximum_b_bits"], stats["maximum_safe_accumulator_bits_full_range"]))
    lines += ["", "## 面积结论与下一步", "",
        "题目面积上限为 **2.25 mm²**。**本轮只能确认数值精度和存储 / 运算量代理，尚未确认面积达标。** 当前没有指定标准单元库、官方 SRAM 宏面积表，以及完成综合映射的加速器 RTL / 网表，不能把位宽减少、权重大小或乘法位积直接换算成已满足面积要求。",
        "512 KiB 存储预算需要同时安排权重、激活、注意力临时值、非线性查表和双缓冲；表中的矩阵权重大小只能用于容量的第一步检查。共享乘法阵列的面积取决于原生支持的最大位宽；采用 INT8 多趟计算可以支持高精度操作数，但会增加周期、移位合并和控制开销，必须在 RTL 中核验。",
        "下一步应冻结一个通过完整验证的配置，把导出的每个操作数位宽 / F、舍入饱和、查表和累加规则落实到 RTL，使用指定工艺库与 SRAM 宏综合，再核对总面积、时序、周期及逐层整数结果。若面积超限，优先调整阵列并行度、缓存组织或用较低原生位宽多趟复用，并保留已验证的数值格式。", "",
        "## 结果文件与重现", "",
        "- [search_result.json](search_result.json)：搜索范围、完整验证结果与两个候选。",
        "- [summary_verification.json](summary_verification.json)：样本顺序、FP32 基准和 trial 准确率复算证据。",
        "- [accuracy_operand_precision.csv](accuracy_operand_precision.csv)、[efficiency_operand_precision.csv](efficiency_operand_precision.csv)：每个矩阵操作数实际使用的位宽、F 和尺度；软件边界标为 FP32。",
        "- `trials/<配置>__final/layer_outputs/*.npy`：15 个阶段的全部验证样本输出，使用 `np.load(path, mmap_mode='r')` 分块查看。",
        "- `accuracy_export/weights_manifest.json`、`efficiency_export/weights_manifest.json`：按位打包的格式、偏移、形状、位宽和 F。", "",
        "本报告程序只读取已完成实验的数组、配置和统计结果，没有读取音频或 testing 数据。"]
    if plots.get("generated"):
        lines += ["", "![单层敏感性](layer_sensitivity_heatmap.png)", "", "![精度与存储对照](precision_tradeoff.png)"]
    else:
        lines += ["", "当前结果汇总环境未提供 Matplotlib，未安装新包；数值对照表已完整保存。"]
    return "\n".join(lines) + "\n"


def per_channel_report(pc, summary):
    recommend = pc["recommendations"]
    lines = ["# 逐层与逐通道量化精度搜索结果", "",
        "**最终建议同时比较两个阶段的全部完整验证候选。** 第二阶段按权重输出通道确定 F，不改变激活范围的训练集标定。全部实验只使用训练标定和验证划分，testing 使用量为 **0**。", "",
        "FP32 基准为 **{} / {}，{}**；下降不超过 2 个百分点要求至少正确 **{}** 条。共完成 **48 次逐层敏感性实验、{} + {} 个筛选候选、{} 个完整验证候选**。{} 条筛选样本只用于缩小搜索范围，达标结论采用全部 {} 条验证样本。".format(summary["fp32_correct"], summary["samples"], percent(summary["fp32_accuracy"]), summary["minimum_correct"], summary["screened_candidates"], len(pc["screen_rows"]), len(pc["combined_rows"]), summary["screening_samples"], summary["samples"]), "",
        "## 综合推荐", "",
        "| 方向 | 配置 | 正确数 | 准确率 | 下降 / 个百分点 | 矩阵权重 / KiB | 权重 F 表 / 字节 | 算术工作量代理 |",
        "|---|---|---:|---:|---:|---:|---:|---:|"]
    for label, key in [("精度优先", "accuracy_priority"), ("低算术资源", "arithmetic_resource_priority"), ("接近 FP32 的折中", "per_channel_balanced")]:
        row = recommend[key]
        lines.append("| {} | `{}` | {} | {} | {:.4f} | {:.2f} | {} | {:.3f} × W8A8 |".format(label, row["name"], row["correct"], percent(row["accuracy"]), row["accuracy_drop_pp"], row["matrix_weight_kib"], row["scale_metadata_bytes"], row["arithmetic_work_proxy"]))
    lines += ["", "精度优先按完整验证准确率选择；低算术资源按满足精度约束后的位积工作量、权重与权重 F 元数据总字节数、准确率依次选择。第三项是逐通道阶段的最高精度候选。**工作量代理不等于实测面积、周期或吞吐率**，这些建议是已评估配置中的选择。", "",
        "`pc_` 配置按每个权重矩阵的输出通道单独设置 F；例如 W4A8 是权重 4 bit、矩阵激活 8 bit。概率位宽和 F 由名称后的 p / f 指明。非线性保持 N16，patch 和分类头保持软件 FP32。", "",
        "主阶段按矩阵权重容量选择的 `encoder_w4a16_p16` 仍单独保留，但**它不是综合两个阶段后的低算术资源推荐**。逐通道候选可使用更低激活 / 概率位宽；其新增 F 表也必须计入存储和控制成本。", "",
        "## 全部完整验证候选", "",
        "| ID | 阶段 | 配置 | 正确数 | 准确率 | 下降 / 个百分点 | 权重 / KiB | 权重 F / 字节 | 权重 + 权重 F / KiB | 工作量代理 |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for index, row in enumerate(pc["combined_rows"], 1):
        lines.append("| {} | {} | `{}` | {} | {} | {:.4f} | {:.2f} | {} | {:.3f} | {:.3f} |".format(index, "逐通道" if row["phase"] == "per_channel" else "逐张量", row["name"], row["correct"], percent(row["accuracy"]), row["accuracy_drop_pp"], row["matrix_weight_kib"], row["scale_metadata_bytes"], row["matrix_and_weight_F_kib"], row["arithmetic_work_proxy"]))
    lines += ["", "[all_full_validation_comparison.csv](all_full_validation_comparison.csv) 保留完整数值；[combined_recommendations.json](combined_recommendations.json) 保留综合选择；[pareto_frontier_all_candidates.json](pareto_frontier_all_candidates.json) 同时考虑精度、权重加权重 F 的容量、算术工作量，不把逐通道元数据视为零成本。", "",
        "## 逐通道阶段的筛选实验", "",
        "下面 {} 行仅为 {} 条验证样本的筛选结果，不能单独证明完整验证达标。".format(len(pc["screen_rows"]), summary["screening_samples"]), "",
        "| 配置 | 正确数 | 筛选准确率 | 相对筛选 FP32 下降 / 个百分点 | 工作量代理 |", "|---|---:|---:|---:|---:|"]
    for row in pc["screen_rows"]:
        lines.append("| `{}` | {} / {} | {} | {:.4f} | {:.3f} |".format(row["name"], row["correct"], row["samples"], percent(row["accuracy"]), row["accuracy_drop_pp"], row["arithmetic_work_proxy"]))
    lines += ["", "## 逐通道候选的全样本层输出", "",
        "两个候选都保存了全部 {} 条样本、15 个阶段的输出，并重新验证 final 与对应 full logits 逐位相同。逐层对照分别见 [pc_accuracy_layer_comparison.csv](pc_accuracy_layer_comparison.csv)、[pc_efficiency_layer_comparison.csv](pc_efficiency_layer_comparison.csv)。".format(summary["samples"]), "",
        "| 候选角色 | 层 | MAE | RMSE | 最大误差 | 相对 L2 | 余弦相似度 |", "|---|---|---:|---:|---:|---:|---:|"]
    for role, label in [("accuracy", "逐通道精度"), ("efficiency", "逐通道资源")]:
        for row in pc["comparisons"][role]:
            lines.append("| {} | {} | {:.5g} | {:.5g} | {:.5g} | {} | {:.8f} |".format(label, row["stage"], row["mae"], row["rmse"], row["max_abs_error"], "未定义" if row["relative_l2_error"] is None else "{:.5g}".format(row["relative_l2_error"]), row["cosine_similarity"]))
    lines += ["", "## 通道 F 元数据和导出", "",
        "编码器共有 `(64×4 + 256 + 64)×12 = 6912` 个权重输出通道，因此每个逐通道候选的 F 表是 **6912 个有符号 int8，6912 字节 = 6.75 KiB**。逐张量格式对应 72 个权重标量 F（按 int8 存储为 72 字节）；这两者都未包含激活、LayerNorm 等公共元数据。F 表必须有存储、读端口和按通道移位控制，不能认定为零面积开销。", "",
        "| 角色 | 打包权重 / 字节 | 通道 F / 字节 | 两者合计 / 字节 | 导出目录 |", "|---|---:|---:|---:|---|"]
    for role, label in [("accuracy", "逐通道精度"), ("efficiency", "逐通道资源")]:
        export = pc["exports"][role]
        relative = Path(export["final_directory"]).as_posix()
        lines.append("| {} | {} | {} | {} | [{}]({}) |".format(label, export["packed_weight_bytes"], export["channel_F_table_bytes"], export["total_weight_and_F_bytes"], export["trial"], relative))
    lines.append("")
    for role, label in [("accuracy", "逐通道精度"), ("efficiency", "逐通道资源")]:
        relative = Path(pc["exports"][role]["final_directory"]).as_posix()
        lines += ["{}：[`weights_packed.bin`]({}/weights_packed.bin)、[`weight_channel_F_int8.bin`]({}/weight_channel_F_int8.bin)、[`weights_manifest.json`]({}/weights_manifest.json)、[`precision_config.json`]({}/precision_config.json)。".format(label, relative, relative, relative, relative), ""]
    lines += ["逐列格式的完整位宽 / F 长表见 [pc_accuracy_operand_precision.csv](pc_accuracy_operand_precision.csv)、[pc_efficiency_operand_precision.csv](pc_efficiency_operand_precision.csv)。F 完全来自固定权重各列最大绝对值；激活范围沿用训练集标定，没有通过验证标签学习尺度。", "",
        "## 面积仍待综合核验", "",
        "**精度约束已经用完整验证样本核验；2.25 mm² 面积约束尚未核验。** 当前缺少指定标准单元库、官方 SRAM 宏面积表，以及映射后的加速器 RTL / 网表。共享 GELU 表约 2 KiB、exp 表 8 KiB、权重、通道 F 表、激活和双缓冲都需要计入适用的容量及面积规则。", "",
        "推荐以低算术资源候选启动 RTL 验证，并保留逐通道精度候选作为更小精度损失的备选；冻结数值格式后使用指定工艺库和 SRAM 宏综合。只有此后才能确认面积和时序，也才能确定资源上的最终最优点。下文保留主阶段原始选择及全部逐层实验，避免把主阶段选择误当成两个阶段的综合结论。", "",
        "独立复核见 [independent_verification.json](independent_verification.json)；偏置、LayerNorm 参数和共享查表等公共导出见 [common_hardware_payload](common_hardware_payload)。", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--reference-layer-dir", type=Path,
                        default=ROOT / "runs/w16a16_fpnl_20261028_103350_996076/layer_outputs/fp32")
    parser.add_argument("--chunk", type=int, default=64)
    args = parser.parse_args()
    if args.chunk < 1:
        parser.error("--chunk must be positive")
    run, baseline_dir = args.run.absolute(), args.reference_layer_dir.absolute()
    started = time.perf_counter()
    summary = load_json(run / "search_result.json")
    prep = load_json(run / "preparation_complete.json")
    manifest = load_json(run / "sample_manifest.json")
    baseline_run = baseline_dir.parent.parent
    old_manifest = load_json(baseline_run / "sample_manifest.json")
    if manifest != old_manifest:
        raise AssertionError("FP32 layer outputs have a different sample manifest")
    if summary["testing_samples_used"] != 0 or prep["sample_requests"]["testing"] != 0:
        raise AssertionError("Testing was unexpectedly used")
    with np.load(str(run / "fresh_fp32_reference.npz"), allow_pickle=False) as fresh:
        fresh_logits, targets = fresh["logits"].copy(), fresh["target_ids"].copy()
    np.testing.assert_array_equal(targets, [row["label_id"] for row in manifest])
    old_logits = np.load(str(baseline_dir / "logits.npy"), mmap_mode="r", allow_pickle=False)
    difference = float(np.max(np.abs(fresh_logits - old_logits)))
    np.testing.assert_allclose(fresh_logits, old_logits, rtol=1e-5, atol=1e-5)
    np.testing.assert_array_equal(fresh_logits.argmax(1), old_logits.argmax(1))
    old_logits._mmap.close()
    correct = int((fresh_logits.argmax(1) == targets).sum())
    if correct != summary["fp32_correct"] or len(targets) != summary["samples"]:
        raise AssertionError("Fresh FP32 summary cannot be reproduced")
    expected_minimum = int(math.ceil(correct - .02 * len(targets) - 1e-10))
    if expected_minimum != summary["minimum_correct"]:
        raise AssertionError("2 pp minimum count is inconsistent")
    sensitivities = []
    with (run / "layer_sensitivity.csv").open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            result = load_json(run / "trials" / row["trial"] / "result.json")
            sensitivities.append(dict(layer=int(row["layer"]), effective_bits=int(row["effective_bits"]), **result))
    sensitivities.sort(key=lambda row: (row["layer"], row["effective_bits"]))
    if len(sensitivities) != 48 or {(row["layer"], row["effective_bits"]) for row in sensitivities} != {(layer, bits) for layer in range(1, 13) for bits in (8, 10, 12, 16)}:
        raise AssertionError("Expected 48 unique single-block experiments")
    full_rows = summary["full_validation_candidates"]
    trial_checks = [verify_trial(run, row, fresh_logits, targets, manifest) for row in sensitivities]
    trial_checks += [verify_trial(run, row, fresh_logits, targets, manifest, True) for row in full_rows]
    for role in ("selected_final", "efficient_final"):
        trial_checks.append(verify_trial(run, summary[role], fresh_logits, targets, manifest, True))
    final_repeat_checks = []
    for role, winner_key, final_key in [("accuracy", "accuracy_winner", "selected_final"),
                                        ("efficiency", "efficient_winner", "efficient_final")]:
        full_path = run / "trials" / summary[winner_key]["trial"] / "predictions.npz"
        final_path = run / "trials" / summary[final_key]["trial"] / "predictions.npz"
        with np.load(str(full_path), allow_pickle=False) as full, np.load(str(final_path), allow_pickle=False) as final:
            np.testing.assert_array_equal(full["logits"], final["logits"])
        final_repeat_checks.append(dict(role=role, full_trial=summary[winner_key]["trial"],
                                        final_trial=summary[final_key]["trial"], logits_bitwise_equal=True))
    comparison, area_stats = {}, {}
    completed = {}
    for role, final_key in [("accuracy", "selected_final"), ("efficiency", "efficient_final")]:
        row = summary[final_key]
        directory = run / "trials" / row["trial"]
        if row["trial"] in completed:
            stage_rows = [dict(value) for value in completed[row["trial"]]]
        else:
            stage_rows = []
            for stage in STAGES:
                got = compare_arrays(baseline_dir / (stage + ".npy"), directory / "layer_outputs" / (stage + ".npy"), args.chunk)
                stage_rows.append(dict(stage=stage, trial=row["trial"], **got))
                print("{}: {} MAE={:.6g} relL2={}".format(role, stage, got["mae"], got["relative_l2_error"]), flush=True)
            completed[row["trial"]] = stage_rows
        comparison[role] = stage_rows
        write_csv(run / (role + "_layer_comparison.csv"), stage_rows)
        write_json(run / (role + "_layer_comparison.json"), stage_rows)
        area_stats[role] = load_json(directory / "area_accounting.json")
        quantization = load_json(directory / "quantization.json")
        precision_rows = []
        for key, bits in sorted(quantization["operand_bits"].items()):
            accelerated = key.startswith("layer_")
            fraction = int(quantization["fractional_bits"][key]) if accelerated else None
            precision_rows.append(dict(operand=key, scope="encoder accelerator" if accelerated else "FP32 software",
                                       effective_bits=int(bits) if accelerated else 32,
                                       fractional_bits=fraction,
                                       quantization_step=math.ldexp(1., -fraction) if accelerated else None,
                                       symmetric_code_min=-((1 << (int(bits) - 1)) - 1) if accelerated else None,
                                       symmetric_code_max=((1 << (int(bits) - 1)) - 1) if accelerated else None))
        write_csv(run / (role + "_operand_precision.csv"), precision_rows)
        write_json(run / (role + "_operand_precision.json"), precision_rows)
        export = summary["hardware_weight_exports"][role]
        packed_path = run / export["directory"] / "weights_packed.bin"
        if packed_path.stat().st_size != export["packed_weight_bytes"]:
            raise AssertionError("Packed weight byte count mismatch")
    write_csv(run / "layer_sensitivity_complete.csv", sensitivities)
    write_json(run / "layer_sensitivity_complete.json", sensitivities)
    write_csv(run / "full_validation_comparison.csv", [dict(id=i, **row) for i, row in enumerate(full_rows, 1)])
    write_json(run / "full_validation_comparison.json", full_rows)
    frontier = pareto_rows(full_rows)
    write_json(run / "pareto_frontier.json", frontier)
    plots = optional_plots(run, sensitivities, full_rows, summary["fp32_accuracy"])
    pc = per_channel_summary(run, baseline_dir, fresh_logits, targets, manifest, args.chunk, full_rows)
    write_json(run / "summary_verification.json", dict(
        samples=len(manifest), testing_samples_used=0, manifest_exact_match=True,
        old_fp32_layers_directory=str(baseline_dir), fresh_vs_saved_fp32_logits_max_abs_error=difference,
        fresh_fp32_correct=correct, minimum_correct_for_2pp=expected_minimum,
        recomputed_trial_accuracies=trial_checks,
        final_repeat_checks=final_repeat_checks,
        per_channel_trial_checks=pc["verification_checks"] if pc else [],
        per_channel_repeat_checks=pc["repeat_checks"] if pc else [],
        combined_full_validation_candidates=len(pc["combined_rows"]) if pc else len(full_rows),
        per_channel_weight_F_bytes=6912 if pc else None,
        numeric_accumulation_dtype="float64", layer_comparisons_per_role=len(STAGES),
        source_sha256=digest(Path(__file__)), search_result_sha256=digest(run / "search_result.json"),
        per_channel_search_result_sha256=digest(run / "per_channel_search_result.json") if pc else None,
        plots=plots, area_compliance_verified=False, elapsed_seconds=time.perf_counter() - started))
    report = report_text(run, summary, prep, full_rows, sensitivities, comparison, area_stats, plots, frontier)
    if pc:
        report = report.replace("# 逐层量化精度搜索结果", "## 第一阶段逐层搜索详细记录", 1)
        report = report.replace("## 两个推荐候选", "### 第一阶段独立选择（不是最终综合推荐）", 1)
        report = per_channel_report(pc, summary) + "\n\n" + report
    (run / "README_精度搜索结果.md").write_text(report, encoding="utf-8")
    source = run / "report_source"
    source.mkdir(exist_ok=True)
    shutil.copyfile(Path(__file__), source / Path(__file__).name)
    print("SUMMARY_COMPLETE=" + str(run / "README_精度搜索结果.md"), flush=True)


if __name__ == "__main__":
    main()
