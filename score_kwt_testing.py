"""Score committed predictions from exactly one independent testing run.

This module does not import a model runner, enumerate other runs, or read previous
predictions.  Inference must first commit the SHA-256 of its logits and manifest.
"""
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path

import numpy as np


OUTPUT_NAMES = (
    "predictions.npz", "predictions.csv", "changed_predictions.json",
    "result.json", "README_独立测试结果.md",
)


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path):
    with path.open("r", encoding="utf-8-sig") as stream:
        return json.load(stream)


def _write_json_new(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError("{} must be an integer >= {}".format(name, minimum))
    return value


def _seconds(commit, name):
    value = commit[name]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("{} must be a finite non-negative number".format(name))
    if not math.isfinite(value) or value < 0:
        raise ValueError("{} must be a finite non-negative number".format(name))
    return float(value)


def _subset(targets, fp32_ids, int16_ids, mask):
    count = int(np.count_nonzero(mask))
    fp_correct = int(np.count_nonzero((fp32_ids == targets) & mask))
    int_correct = int(np.count_nonzero((int16_ids == targets) & mask))
    return {
        "samples": count,
        "fp32_correct": fp_correct,
        "fp32_accuracy": fp_correct / count if count else None,
        "w16a16_correct": int_correct,
        "w16a16_accuracy": int_correct / count if count else None,
        "accuracy_drop_percentage_points": (
            100.0 * (fp_correct - int_correct) / count if count else None
        ),
    }


def score_run(run_dir):
    """Validate the inference commit and score it without reading old answers."""
    run_dir = Path(run_dir).absolute()
    existing = [name for name in OUTPUT_NAMES if (run_dir / name).exists()]
    if existing:
        raise FileExistsError("Refusing to overwrite scoring outputs: " + ", ".join(existing))

    commit = _read_json(run_dir / "inference_complete.json")
    count = _integer(commit["samples"], "samples", minimum=1)
    # Check both committed byte sequences before interpreting their contents.
    checked_hashes = {}
    for filename, field in (
        ("inference_logits.npz", "logits_sha256"),
        ("testing_manifest.json", "manifest_sha256"),
    ):
        expected = commit[field]
        if not isinstance(expected, str) or len(expected) != 64:
            raise ValueError("Invalid committed hash: " + field)
        actual = _sha256(run_dir / filename)
        if actual != expected.lower():
            raise ValueError("Committed SHA-256 mismatch: " + filename)
        checked_hashes[filename] = actual

    metadata = _read_json(run_dir / "run_metadata.json")
    labels = metadata["labels"]
    if (not isinstance(labels, list) or len(labels) != 12
            or not all(isinstance(label, str) and label for label in labels)
            or len(set(labels)) != 12):
        raise ValueError("run_metadata.labels must contain exactly 12 unique labels")
    if "_silence_" not in labels or "_unknown_" not in labels:
        raise ValueError("Expected _silence_ and _unknown_ in the 12-class label map")
    manifest = _read_json(run_dir / "testing_manifest.json")
    if not isinstance(manifest, list) or len(manifest) != count:
        raise ValueError("Manifest length does not match the inference commit")
    target_ids = np.empty(count, dtype=np.int64)
    for index, row in enumerate(manifest):
        if not isinstance(row, dict) or _integer(row["index"], "index") != index:
            raise ValueError("Manifest indices must be consecutive from zero")
        label_id = _integer(row["label_id"], "label_id")
        if label_id >= len(labels) or row["label"] != labels[label_id]:
            raise ValueError("Manifest label does not match the class map at index {}".format(index))
        if not isinstance(row["file"], str) or not isinstance(row["source_word"], str):
            raise ValueError("Manifest file/source_word must be strings")
        if not isinstance(row["synthetic_zero_waveform"], bool):
            raise ValueError("synthetic_zero_waveform must be a boolean")
        target_ids[index] = label_id

    with np.load(str(run_dir / "inference_logits.npz"), allow_pickle=False) as archive:
        if set(archive.files) != {"fp32_logits", "w16a16_logits"}:
            raise ValueError("Inference archive must contain only the two logits arrays, without targets")
        fp32_logits = archive["fp32_logits"]
        int16_logits = archive["w16a16_logits"]
    for name, array in (("fp32_logits", fp32_logits), ("w16a16_logits", int16_logits)):
        if array.shape != (count, 12) or array.dtype.kind != "f" or not np.isfinite(array).all():
            raise ValueError("{} must be a finite floating array of shape [{}, 12]".format(name, count))
    elapsed_seconds = _seconds(commit, "elapsed_seconds")
    inference_seconds = _seconds(commit, "inference_seconds")

    # Scoring begins only after both committed artifacts and their shapes pass.
    fp32_ids = np.argmax(fp32_logits, axis=1)
    int16_ids = np.argmax(int16_logits, axis=1)
    fp32_ok = fp32_ids == target_ids
    int16_ok = int16_ids == target_ids
    changed = fp32_ids != int16_ids
    all_scores = _subset(target_ids, fp32_ids, int16_ids, np.ones(count, dtype=bool))
    silence_id = labels.index("_silence_")
    unknown_id = labels.index("_unknown_")
    subsets = {
        "exclude_silence": _subset(
            target_ids, fp32_ids, int16_ids, target_ids != silence_id),
        "known_ten_keywords": _subset(
            target_ids, fp32_ids, int16_ids,
            (target_ids != silence_id) & (target_ids != unknown_id)),
    }
    fp32_cm = np.zeros((12, 12), dtype=np.int64)
    int16_cm = np.zeros((12, 12), dtype=np.int64)
    np.add.at(fp32_cm, (target_ids, fp32_ids), 1)
    np.add.at(int16_cm, (target_ids, int16_ids), 1)
    changed_rows = []
    csv_text = io.StringIO(newline="")
    writer = csv.writer(csv_text)
    writer.writerow(("index", "file", "source_word", "target_id", "target_label",
                     "fp32_prediction_id", "fp32_prediction", "fp32_correct",
                     "w16a16_prediction_id", "w16a16_prediction", "w16a16_correct",
                     "predictions_agree", "synthetic_zero_waveform"))
    for index, row in enumerate(manifest):
        fp_id, int_id = int(fp32_ids[index]), int(int16_ids[index])
        writer.writerow((index, row["file"], row["source_word"], int(target_ids[index]),
                         row["label"], fp_id, labels[fp_id], int(fp32_ok[index]),
                         int_id, labels[int_id], int(int16_ok[index]),
                         int(not changed[index]), row["synthetic_zero_waveform"]))
        if changed[index]:
            changed_rows.append({
                "index": index, "file": row["file"], "label": row["label"],
                "fp32_prediction": labels[fp_id], "w16a16_prediction": labels[int_id],
                "fp32_correct": bool(fp32_ok[index]), "w16a16_correct": bool(int16_ok[index]),
            })
    result = dict(all_scores)
    result.update({
        "mode": "paired independent testing", "split": "testing",
        "independent_from_previous_predictions": True,
        "phase": "committed inference then separate scoring",
        "prediction_agreement": float(np.mean(~changed)),
        "changed_predictions": int(np.count_nonzero(changed)),
        "correct_to_wrong": int(np.count_nonzero(fp32_ok & ~int16_ok)),
        "wrong_to_correct": int(np.count_nonzero(~fp32_ok & int16_ok)),
        "subsets": subsets,
        "inference_commit_verified": True, "checked_sha256": checked_hashes,
        "elapsed_seconds": elapsed_seconds, "inference_seconds": inference_seconds,
        "output_directory": str(run_dir),
        "note": "Static W16A16 integer matmul, int64 accumulation, FP32 nonlinear operations. "
                "No retraining; testing is separate from training calibration. "
                "This does not audit the official checkpoint training history.",
    })
    # All files are exclusive creates. An existing experiment is never overwritten.
    with (run_dir / "predictions.npz").open("xb") as stream:
        np.savez_compressed(
            stream, fp32_logits=fp32_logits, w16a16_logits=int16_logits,
            target_ids=target_ids, fp32_confusion_matrix=fp32_cm,
            w16a16_confusion_matrix=int16_cm)
    with (run_dir / "predictions.csv").open("x", encoding="utf-8-sig", newline="") as stream:
        stream.write(csv_text.getvalue())
    _write_json_new(run_dir / "changed_predictions.json", changed_rows)
    _write_json_new(run_dir / "result.json", result)
    report_rows = []
    for title, values in [("全部 12 类", all_scores), ("排除静音", subsets["exclude_silence"]),
                          ("10 个已知关键词", subsets["known_ten_keywords"])]:
        def accuracy_text(value):
            return "无样本" if value is None else "{:.6f}%".format(100 * value)
        report_rows.append("| {} | {} | {} / {}（{}） | {} / {}（{}） |".format(
            title, values["samples"], values["fp32_correct"], values["samples"],
            accuracy_text(values["fp32_accuracy"]), values["w16a16_correct"],
            values["samples"], accuracy_text(values["w16a16_accuracy"])))
    report = """# 独立 testing 集结果

本次在 testing 划分上重新进行 FP32 与 W16A16 前向推理，共 {count} 条样本。
两路推理共改变 {changed} 条预测，准确率下降 {drop:.6f} 个百分点。

| 统计范围 | 样本数 | FP32 正确数 / 总数（准确率） | W16A16 正确数 / 总数（准确率） |
|---|---:|---:|---:|
{rows}

矩阵乘的权重与输入激活按固定尺度量化为有符号 int16，使用 int64 累加；Softmax、GELU、LayerNorm 等非线性保持 FP32。
量化范围由训练集样本校准；testing 样本用于评分，不用于本次量化尺度校准或训练。

推理与评分分为两个阶段：推理阶段先生成只含两路 logits、没有标签的 `inference_logits.npz`，并在 `inference_complete.json` 中提交 logits 与 `testing_manifest.json` 的 SHA-256。
随后本评分脚本核对哈希、样本顺序、类别映射、数组形状与有限数值，再从本次 manifest 提取真实标签计分。
评分脚本只读取本次目录的四个输入文件：`inference_logits.npz`、`testing_manifest.json`、`run_metadata.json`、`inference_complete.json`；不读取其他实验的预测文件。
哈希校验固定的是本次提交文件，不能单独证明任何脚本的全部行为；推理脚本与源代码快照可用于进一步审查。

`predictions.csv` 记录每条样本的标签、两路预测及是否正确；`predictions.npz` 保存 logits、真实标签与混淆矩阵（行是真实类别，列是预测类别）；`changed_predictions.json` 记录两模型预测变化。
静音包含按数据协议合成的全零音频，因此同时列出排除静音及 10 关键词子集的成绩，便于观察统计范围的影响。

本次独立测试验证本地流程对 testing 划分的表现，不能据此审计官方 checkpoint 的训练历史或证明其训练过程没有使用测试数据。
CPU 整数乘法用于数值验证，本结果不代表硬件速度或延迟。

推理耗时：{inference:.3f} 秒；提交记录总耗时：{elapsed:.3f} 秒。
""".format(count=count, changed=result["changed_predictions"],
           drop=result["accuracy_drop_percentage_points"], rows="\n".join(report_rows),
           inference=inference_seconds, elapsed=elapsed_seconds)
    with (run_dir / "README_独立测试结果.md").open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(report)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(score_run(args.run), ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
