"""Summarize one newly generated progressive-four-bit experiment.

Reads only experiment metadata, policies, scores, and saved predictions.  It
does not read audio, dataset directories, or other runs, and does not perform
inference.  Candidates with full-validation measured operation counts alone
are eligible for the final recommendation.  Screening uses its own measured
candidate-operand counts; separate training histograms provide estimates.
"""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np


README_NAME = "README_四位渐进乘法实验结果.md"


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def _finite_float(value, field):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("Nonfinite field: " + field)
    return value


def _rows(path):
    value = load_json(path)
    if isinstance(value, list):
        return value
    for key in ("results", "trials", "rows"):
        if isinstance(value.get(key), list):
            return value[key]
    raise ValueError("Expected a result list: " + str(path))


def _prediction_audit(path, expected):
    """Recount scores when logits and target IDs are available in the NPZ."""
    if not path.exists():
        raise FileNotFoundError(str(path))
    with np.load(str(path), allow_pickle=False) as archive:
        target_key = next((key for key in ("target_ids", "labels", "targets", "y_true") if key in archive.files), None)
        prediction_key = next((key for key in ("logits", "predicted_ids", "prediction_ids", "predictions", "y_pred")
                               if key in archive.files), None)
        if target_key is None or prediction_key is None:
            return {"verified": False, "reason": "unrecognized_prediction_schema", "keys": archive.files}
        target = np.asarray(archive[target_key])
        prediction = np.asarray(archive[prediction_key])
        if target.ndim != 1:
            raise ValueError("Expected one-dimensional target IDs: " + str(path))
        if prediction.ndim == 2:
            if not np.isfinite(prediction).all():
                raise FloatingPointError("Nonfinite saved logits: " + str(path))
            prediction = prediction.argmax(axis=1)
        if prediction.ndim != 1 or prediction.shape != target.shape:
            raise ValueError("Saved prediction/target shapes differ: " + str(path))
        correct = int(np.count_nonzero(prediction == target))
        if len(target) != int(expected["samples"]) or correct != int(expected["correct"]):
            raise AssertionError("Saved predictions do not reproduce result: " + str(path))
        return {"verified": True, "samples": len(target), "correct": correct,
                "target_key": target_key, "prediction_key": prediction_key}


def _baseline(run, name):
    directory = run / "baseline" / name
    result = load_json(directory / "result.json")
    if int(result.get("testing_samples_used", 0)) != 0:
        raise ValueError("Experiment reports testing samples: " + name)
    samples, correct = int(result["samples"]), int(result["correct"])
    if samples <= 0 or not 0 <= correct <= samples:
        raise ValueError("Invalid baseline counts: " + name)
    return result, _prediction_audit(directory / "predictions.npz", result)


def _candidate(run, supplied, scope):
    row = dict(supplied)
    name = str(row["name"])
    directory = run / "trials" / (name + "__" + scope)
    stored = load_json(directory / "result.json")
    for key in ("name", "samples", "correct"):
        if key in row and row[key] != stored[key]:
            raise AssertionError("Result list disagrees with trial " + key + ": " + name)
    row.update(stored)
    policy = load_json(directory / "policy.json")
    if policy["name"] != name:
        raise AssertionError("Policy and result names disagree: " + name)
    row["scope"] = scope
    row["mode"] = policy["mode"]
    row["threshold"] = _finite_float(policy.get("threshold", 0), "threshold")
    row["stages"] = policy.get("stages", "all")
    row["operations"] = list(policy.get("operations", []))
    row["operation_count"] = len(row["operations"])
    row["zero_tail_guard"] = bool(policy.get("zero_tail_guard", False))
    row["is_final_repeat"] = name.endswith("_final")
    row["is_zero_tail_control"] = ("exact_zero_tail_control" in name
                                   or (row["mode"] == "exact" and row["zero_tail_guard"]))
    row["directory"] = str(directory.relative_to(run)).replace("\\", "/")
    row["policy_path"] = row["directory"] + "/policy.json"
    samples, correct = int(row["samples"]), int(row["correct"])
    if samples <= 0 or not 0 <= correct <= samples:
        raise ValueError("Invalid trial counts: " + name)
    fp32_correct = int(row["fp32_correct"])
    w8a8_correct = int(row["w8a8_correct"])
    row["accuracy_percent"] = 100.0 * correct / samples
    row["drop_vs_fp32_pp"] = 100.0 * (fp32_correct - correct) / samples
    row["additional_drop_vs_w8a8_pp"] = 100.0 * (w8a8_correct - correct) / samples
    row["passes_extra_0_1pp"] = row["additional_drop_vs_w8a8_pp"] <= 0.1 + 1e-12
    row["passes_fp32_2pp"] = row["drop_vs_fp32_pp"] <= 2.0 + 1e-12
    if int(row.get("testing_samples_used", 0)) != 0:
        raise ValueError("Experiment reports testing samples: " + name)
    for field in ("saved_partial_product_fraction_selected", "saved_partial_product_fraction_encoder"):
        value = _finite_float(row[field], field)
        if value < -1e-12 or value > 1 + 1e-12:
            raise ValueError("Saved partial-product fraction outside [0,1]: " + name)
        row[field] = value
    row["prediction_audit"] = _prediction_audit(directory / "predictions.npz", row)
    row["has_layer_comparison"] = (directory / "layer_comparison.csv").exists()
    row["has_operation_counts"] = (directory / "operation_counts.json").exists()
    if row.get("savings_measurement") != scope + "_validation_candidate_operands":
        raise ValueError("Trial lacks measured candidate-operand counts: " + name)
    if not row["has_operation_counts"] or not row["has_layer_comparison"]:
        raise FileNotFoundError("Trial lacks operation counts or layer comparison: " + name)
    row["operation_count_audit"] = _operation_count_audit(directory / "operation_counts.json", row)
    return row


def _operation_count_audit(path, row):
    counts = load_json(path)
    per_operation = counts["per_operation"]
    if set(per_operation) != set(row["operations"]):
        raise AssertionError("Count operation list does not match policy: " + str(path))
    total = np.zeros(4, dtype=np.int64)
    for value in per_operation.values():
        vector = np.asarray(value)
        if vector.shape != (4,) or not np.issubdtype(vector.dtype, np.integer) or np.any(vector < 0):
            raise ValueError("Invalid exit-stage counts: " + str(path))
        total += vector
    if not np.array_equal(total, np.asarray(counts["total_exit_counts"])):
        raise AssertionError("Per-operation counts do not sum to total: " + str(path))
    products = int(total.sum())
    expected_products = int(row["selected_macs_per_sample"]) * int(row["samples"])
    slots = int(np.dot(total, np.arange(1, 5, dtype=np.int64)))
    saved = 4 * products - slots
    denominator = int(counts["full_encoder_four_term_slots"])
    if (products != expected_products or products != int(counts["selected_scalar_products"])
            or slots != int(counts["executed_partial_product_slots"])
            or saved != int(counts["saved_partial_product_slots"])):
        raise AssertionError("Partial-product count accounting mismatch: " + str(path))
    if denominator <= 0 or counts.get("excludes_padding") is not True:
        raise AssertionError("Invalid denominator or padded sample accounting: " + str(path))
    selected_fraction = saved / float(4 * products) if products else 0.0
    encoder_fraction = saved / float(denominator)
    mean_terms = slots / float(products) if products else 4.0
    for actual, expected in ((selected_fraction, row["saved_partial_product_fraction_selected"]),
                             (encoder_fraction, row["saved_partial_product_fraction_encoder"]),
                             (mean_terms, row["mean_terms_selected"])):
        if not math.isclose(actual, float(expected), rel_tol=1e-12, abs_tol=1e-12):
            raise AssertionError("Result fraction does not match counted slots: " + str(path))
    return {"verified": True, "exit_stage_counts": total.tolist(),
            "selected_scalar_products": products, "executed_partial_product_slots": slots,
            "saved_partial_product_slots": saved, "full_encoder_four_term_slots": denominator}


def _ranking(row):
    return (row["saved_partial_product_fraction_encoder"], row["correct"],
            row["saved_partial_product_fraction_selected"], row["name"])


def _best(rows, criterion):
    feasible = [row for row in rows if row[criterion] and row["saved_partial_product_fraction_encoder"] > 0]
    return max(feasible, key=_ranking) if feasible else None


def _frontier(rows):
    result = []
    for row in rows:
        dominated = any(other["correct"] >= row["correct"]
                        and other["saved_partial_product_fraction_encoder"] >= row["saved_partial_product_fraction_encoder"]
                        and (other["correct"] > row["correct"]
                             or other["saved_partial_product_fraction_encoder"] > row["saved_partial_product_fraction_encoder"])
                        for other in rows)
        if not dominated:
            result.append(row["name"])
    return sorted(set(result))


def _brief(row):
    if row is None:
        return None
    fields = ("name", "scope", "samples", "correct", "accuracy_percent", "mode", "threshold", "stages",
              "operations", "operation_count", "drop_vs_fp32_pp", "additional_drop_vs_w8a8_pp",
              "passes_extra_0_1pp", "passes_fp32_2pp", "saved_partial_product_fraction_selected",
              "saved_partial_product_fraction_encoder", "mean_terms_selected", "selected_macs_per_sample",
              "savings_measurement", "directory", "policy_path", "prediction_audit", "is_final_repeat",
              "is_zero_tail_control", "zero_tail_guard", "operation_count_audit")
    return {field: row[field] for field in fields if field in row}


def _matched_zero_control(row, zero_controls):
    if row is None:
        return None
    matched = [control for control in zero_controls if set(control["operations"]) == set(row["operations"])]
    return max(matched, key=_ranking) if matched else None


def _zero_control_comparison(row, zero_controls):
    if row is None:
        return None
    control = _matched_zero_control(row, zero_controls)
    return {
        "name": row["name"],
        "matched_operations": control is not None,
        "zero_tail_control_name": control["name"] if control else None,
        "saved_encoder_fraction_increment":
            (row["saved_partial_product_fraction_encoder"] - control["saved_partial_product_fraction_encoder"]
             if control else None),
        "saved_selected_fraction_increment":
            (row["saved_partial_product_fraction_selected"] - control["saved_partial_product_fraction_selected"]
             if control else None),
        "note": "Only an exactly matching operation set provides a scope-matched zero-tail baseline.",
    }


def _percent(value):
    return "{:.4f}%".format(100.0 * float(value))


def _table(rows, header=True):
    lines = []
    if header:
        lines.extend(["| 配置 | 正确数 | 准确率 | 比 W8A8 额外下降/百分点 | 比 FP32 下降/百分点 | 所选操作省部分积 | 全编码器省部分积 | 平均项数 |",
                      "|---|---:|---:|---:|---:|---:|---:|---:|"])
    for row in rows:
        lines.append("| [{}]({}/result.json) | {}/{} | {:.4f}% | {:.4f} | {:.4f} | {} | {} | {:.4f} |".format(
            row["name"], row["directory"], row["correct"], row["samples"], row["accuracy_percent"],
            row["additional_drop_vs_w8a8_pp"], row["drop_vs_fp32_pp"],
            _percent(row["saved_partial_product_fraction_selected"]),
            _percent(row["saved_partial_product_fraction_encoder"]), row["mean_terms_selected"]))
    return lines


def _normalized_layers(run, rows):
    normalized = []
    for result in rows:
        source = run / result["directory"] / "layer_comparison.json"
        if not source.exists():
            continue
        for saved in load_json(source):
            row = dict(saved)
            row["reference_w8a8_max_abs"] = row.pop("fp32_max_abs")
            row["candidate_max_abs"] = row.pop("w16a16_max_abs")
            row.update(name=result["name"], scope=result["scope"],
                       is_final_repeat=result["is_final_repeat"],
                       source_path=str(source.relative_to(run)).replace("\\", "/"))
            normalized.append(row)
    if normalized:
        save_json(run / "normalized_layer_comparison.json", normalized)
        fields = list(dict.fromkeys(key for row in normalized for key in row))
        with (run / "normalized_layer_comparison.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(normalized)
    return len(normalized)


def _write_comparison_rows(run, basename, rows):
    save_json(run / (basename + ".json"), rows)
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (run / (basename + ".csv")).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _slot_metrics(stage_counts, encoder_slots):
    pairs = sum(int(value) for value in stage_counts)
    executed = sum((index + 1) * int(value) for index, value in enumerate(stage_counts))
    full = 4 * pairs
    saved = full - executed
    return {"stage1_count": int(stage_counts[0]), "stage2_count": int(stage_counts[1]),
            "stage3_count": int(stage_counts[2]), "stage4_count": int(stage_counts[3]),
            "pairs": pairs, "executed_partial_product_slots": executed,
            "full_four_term_slots": full, "saved_partial_product_slots": saved,
            "mean_terms_selected": executed / float(pairs) if pairs else None,
            "saved_partial_product_fraction_selected": saved / float(full) if full else 0.0,
            "encoder_saved_fraction_contribution": saved / float(encoder_slots)}


def _operation_comparisons(run, rows):
    """Expand recorded counts without invoking the model or changing trials."""
    operations, layers = [], []
    # Every KWT-1 encoder block has the same shapes/MAC count in this run.
    layer_names = ["layer_{:02d}".format(index) for index in range(1, 13)]
    for result in rows:
        directory = run / result["directory"]
        counts = load_json(directory / "operation_counts.json")
        encoder_slots = int(counts["full_encoder_four_term_slots"])
        if encoder_slots % len(layer_names):
            raise ValueError("Full encoder slot denominator is not divisible by 12 blocks")
        layer_slots = encoder_slots // len(layer_names)
        grouped = {layer: np.zeros(4, dtype=np.int64) for layer in layer_names}
        operation_numbers = {layer: 0 for layer in layer_names}
        identity = {"name": result["name"], "scope": result["scope"], "samples": int(result["samples"]),
                    "mode": result["mode"], "threshold": result["threshold"], "stages": result["stages"],
                    "zero_tail_guard": result["zero_tail_guard"], "is_final_repeat": result["is_final_repeat"],
                    "is_zero_tail_control": result["is_zero_tail_control"]}
        for operation, vector in sorted(counts["per_operation"].items()):
            layer = operation.split("/", 1)[0]
            if layer not in grouped:
                raise ValueError("Unexpected encoder operation name: " + operation)
            grouped[layer] += np.asarray(vector, dtype=np.int64)
            operation_numbers[layer] += 1
            row = dict(identity, layer=layer, operation=operation,
                       operation_kind=operation.split("/", 1)[1],
                       **_slot_metrics(vector, encoder_slots))
            row["saved_partial_product_fraction_operation"] = row["saved_partial_product_fraction_selected"]
            row["saved_slots_per_sample"] = row["saved_partial_product_slots"] / float(row["samples"])
            operations.append(row)
        error_rows = {row["stage"]: row for row in load_json(directory / "layer_comparison.json")}
        trial_layer_saved = 0
        for layer in layer_names:
            row = dict(identity, layer=layer, selected_operation_count=operation_numbers[layer],
                       **_slot_metrics(grouped[layer], encoder_slots))
            row["layer_full_encoder_four_term_slots"] = layer_slots
            row["saved_partial_product_fraction_layer"] = row["saved_partial_product_slots"] / float(layer_slots)
            row["saved_slots_per_sample"] = row["saved_partial_product_slots"] / float(row["samples"])
            error = error_rows.get(layer + "_output", {})
            for field in ("mae", "rmse", "max_abs_error", "relative_l2", "cosine"):
                row["layer_output_" + field + "_vs_w8a8"] = error.get(field)
            row["unselected_operations_counted_as_full_four_terms_in_layer_denominator"] = True
            layers.append(row)
            trial_layer_saved += row["saved_partial_product_slots"]
        if trial_layer_saved != int(counts["saved_partial_product_slots"]):
            raise AssertionError("Layer slot savings do not sum to trial total: " + result["name"])
    _write_comparison_rows(run, "operation_count_comparison", operations)
    _write_comparison_rows(run, "layer_compute_comparison", layers)
    return {"operation_rows": len(operations), "layer_rows": len(layers),
            "layer_count": len(layer_names),
            "stage_count_meaning": "stage1_count..stage4_count count product pairs exiting after 1..4 terms, not four independent attempted-term totals.",
            "layer_denominator": "One twelfth of full encoder four-term slots; all 12 KWT blocks have identical matrix shapes."}


def _screen_grouping_audit(run, screening, zero_controls):
    """Compare saved exact-control logits under screen/full sample grouping."""
    with np.load(str(run / "baseline/w8a8/predictions.npz"), allow_pickle=False) as saved:
        if "logits" not in saved.files or "target_ids" not in saved.files:
            return {"verified": False, "reason": "baseline_has_no_logits_for_grouping_check"}
        baseline_logits = saved["logits"].copy()
        baseline_targets = saved["target_ids"].copy()
    result = {"verified": True, "screen_controls": [], "full_controls": [],
              "interpretation": "Screen regrouping can alter FP32 patch/head rounding and downstream quantization boundaries. Full natural-order exact controls isolate arithmetic changes."}
    for row in screening + zero_controls:
        if not row["is_zero_tail_control"]:
            continue
        directory = run / row["directory"]
        with np.load(str(directory / "predictions.npz"), allow_pickle=False) as saved:
            logits, targets = saved["logits"], saved["target_ids"]
            indices = np.load(str(directory / "sample_indices.npy"), allow_pickle=False)
            if indices.ndim != 1 or len(indices) != int(row["samples"]):
                raise ValueError("Invalid exact-control sample indices: " + row["name"])
            if not np.array_equal(targets, baseline_targets[indices]):
                raise AssertionError("Exact-control target order differs: " + row["name"])
            reference = baseline_logits[indices]
            error = np.abs(logits.astype(np.float64) - reference.astype(np.float64))
            bit_exact = bool(np.array_equal(logits, reference))
            record = {"name": row["name"], "samples": len(indices), "logits_bit_exact_vs_full_baseline_subset": bit_exact,
                      "changed_logit_elements": int(np.count_nonzero(logits != reference)),
                      "max_abs_logit_difference": float(error.max()),
                      "classification_flips": int(np.count_nonzero(logits.argmax(1) != reference.argmax(1)))}
            layer_rows = load_json(directory / "layer_comparison.json")
            patch = next((layer for layer in layer_rows if layer["stage"] == "patch_output"), None)
            record["patch_max_abs_difference"] = float(patch["max_abs_error"]) if patch else None
            if row["scope"] == "full":
                if not np.array_equal(indices, np.arange(len(baseline_targets))):
                    raise AssertionError("Full exact control is not in natural validation order: " + row["name"])
                if not bit_exact:
                    raise AssertionError("Natural-order full exact zero-tail control is not bit exact: " + row["name"])
                result["full_controls"].append(record)
            else:
                result["screen_controls"].append(record)
    return result


def _export_recommended_policy(run, filename, objective, row, final_repeats, zero_controls):
    """Export a reviewable policy plus relative provenance without changing selection records."""
    if row is None:
        exported = {"objective": objective, "feasible_candidate_found": False, "policy": None,
                    "reason": "No full-validation candidate both meets the accuracy goal and saves partial-product slots."}
        save_json(run / filename, exported)
        return {"path": filename, "feasible_candidate_found": False}
    directory = run / row["directory"]
    quantization_path = run / "quantization.json"
    quantization = load_json(quantization_path)
    for field in ("operand_bits", "fractional_bits", "nonlinear"):
        if field not in quantization:
            raise ValueError("Incomplete fixed-point quantization reference: " + field)
    lut_path = directory / "arithmetic_lut.npz"
    if not lut_path.is_file():
        raise FileNotFoundError(str(lut_path))
    repeated = next((item for item in final_repeats if item["name"] == row["name"] + "_final"), None)
    repeat_record = None
    if repeated:
        repeat_directory = run / repeated["directory"]
        with np.load(str(directory / "predictions.npz"), allow_pickle=False) as first:
            with np.load(str(repeat_directory / "predictions.npz"), allow_pickle=False) as second:
                repeat_bit_exact = bool(np.array_equal(first["logits"], second["logits"]))
        if not repeat_bit_exact:
            raise AssertionError("Recommended policy's independent full repeat differs: " + row["name"])
        layer_directory = repeat_directory / "layer_outputs"
        repeat_record = {
            "trial_directory": repeated["directory"],
            "policy_path": repeated["policy_path"],
            "full_metrics_path": repeated["directory"] + "/result.json",
            "arithmetic_lut_path": repeated["directory"] + "/arithmetic_lut.npz",
            "layer_output_directory": repeated["directory"] + "/layer_outputs" if layer_directory.is_dir() else None,
            "layer_output_files": [str(path.relative_to(run)).replace("\\", "/")
                                   for path in sorted(layer_directory.glob("*.npy"))],
            "logits_bit_exact_to_original": repeat_bit_exact,
        }
    exported = {
        "objective": objective,
        "feasible_candidate_found": True,
        "policy": load_json(directory / "policy.json"),
        "provenance": {
            "path_base": "All paths are relative to this experiment directory, containing this exported JSON.",
            "trial_directory": row["directory"],
            "policy_path": row["policy_path"],
            "arithmetic_lut_path": row["directory"] + "/arithmetic_lut.npz",
            "arithmetic_lut_sha256": hashlib.sha256(lut_path.read_bytes()).hexdigest(),
            "full_metrics_path": row["directory"] + "/result.json",
            "operation_counts_path": row["directory"] + "/operation_counts.json",
            "layer_comparison_path": row["directory"] + "/layer_comparison.json",
            "quantization_path": "quantization.json",
            "quantization_sha256": hashlib.sha256(quantization_path.read_bytes()).hexdigest(),
            "quantization_maps": ["operand_bits", "fractional_bits", "nonlinear"],
            "quantization_note": "Complete fixed W8A8 bits/F and N16 formats; use together with the progressive multiplication policy.",
            "independent_full_repeat": repeat_record,
        },
        "full_validation_metrics": load_json(directory / "result.json"),
        "scope_matched_zero_tail_comparison": _zero_control_comparison(row, zero_controls),
        "testing_samples_used": 0,
        "hardware_speedup_measured": False,
        "selection_note": "Selected across original and guarded full-validation configurations; original selected_policy.json and guarded_selected_policy.json remain preserved.",
    }
    save_json(run / filename, exported)
    return {"path": filename, "feasible_candidate_found": True,
            "name": row["name"], "has_independent_full_repeat": repeat_record is not None}


def summarize(run_directory):
    run = Path(run_directory).absolute()
    metadata = load_json(run / "run_metadata.json")
    if int(metadata.get("testing_samples_used", 0)) != 0:
        raise ValueError("Run metadata reports testing samples")
    fp32, fp32_audit = _baseline(run, "fp32")
    w8a8, w8a8_audit = _baseline(run, "w8a8")
    samples = int(fp32["samples"])
    if samples != int(w8a8["samples"]):
        raise ValueError("FP32 and W8A8 baseline sample counts differ")
    screening = [_candidate(run, row, "screen") for row in _rows(run / "screening_results.json")]
    full = [_candidate(run, row, "full") for row in _rows(run / "full_validation_results.json")]
    controls = [_candidate(run, load_json(path / "result.json"), "control")
                for path in sorted((run / "trials").glob("*__control")) if (path / "result.json").exists()]
    if not full:
        raise ValueError("No completed full-validation candidates; screening cannot establish the final recommendation")
    for row in full:
        if (int(row["samples"]) != samples or int(row["fp32_correct"]) != int(fp32["correct"])
                or int(row["w8a8_correct"]) != int(w8a8["correct"])):
            raise AssertionError("Full trial baseline or sample-count mismatch: " + row["name"])
    unique_full = [row for row in full if not row["is_final_repeat"]]
    final_repeats = [row for row in full if row["is_final_repeat"]]
    strict = _best(unique_full, "passes_extra_0_1pp")
    relaxed = _best(unique_full, "passes_fp32_2pp")
    approximate = [row for row in unique_full if not row["is_zero_tail_control"]]
    strict_approximate = _best(approximate, "passes_extra_0_1pp")
    zero_controls = [row for row in unique_full if row["is_zero_tail_control"]]
    zero_tail = next((row for row in zero_controls if row["name"] == "exact_zero_tail_control"), None)
    guarded_full = [row for row in unique_full if row["zero_tail_guard"]]
    guarded_strict = _best(guarded_full, "passes_extra_0_1pp")
    strict_zero_comparison = _zero_control_comparison(strict, zero_controls)
    zero_comparisons = [_zero_control_comparison(row, zero_controls) for row in unique_full]
    grouping_audit = _screen_grouping_audit(run, screening, zero_controls)
    min_strict = int(math.ceil(int(w8a8["correct"]) - samples * 0.001 - 1e-12))
    min_relaxed = int(math.ceil(int(fp32["correct"]) - samples * 0.02 - 1e-12))
    audit = {"fp32": fp32_audit, "w8a8": w8a8_audit,
             "trials": {row["name"] + "__" + row["scope"]: row["prediction_audit"] for row in controls + screening + full}}
    recommendations = {
        "selection_basis": "Full validation only; maximize measured saved partial-product slots over the entire encoder, then correct count.",
        "strict_extra_drop_limit_pp": 0.1,
        "strict_min_correct": min_strict,
        "relaxed_fp32_drop_limit_pp": 2.0,
        "relaxed_min_correct": min_relaxed,
        "strict_recommendation": _brief(strict),
        "strict_approximate_recommendation": _brief(strict_approximate),
        "guarded_strict_recommendation": _brief(guarded_strict),
        "fp32_2pp_recommendation": _brief(relaxed),
        "exact_zero_tail_control": _brief(zero_tail),
        "exact_zero_tail_controls": [_brief(row) for row in zero_controls],
        "strict_scope_matched_zero_tail_comparison": strict_zero_comparison,
        "scope_matched_zero_tail_comparisons": zero_comparisons,
        "incremental_saved_encoder_fraction_over_zero_tail_control":
            (strict_zero_comparison["saved_encoder_fraction_increment"] if strict_zero_comparison else None),
        "accuracy_savings_pareto_names": _frontier(unique_full),
        "screening_trials": len(screening),
        "unique_full_validation_trials": len(unique_full),
        "independent_final_repeats": len(final_repeats),
        "guarded_unique_full_validation_trials": len(guarded_full),
        "guarded_stage_completed": (run / "guarded_completion.json").exists(),
        "native_control_trials": len(controls),
        "testing_samples_used": 0,
        "area_measured": False,
        "hardware_cycles_measured": False,
        "energy_measured": False,
        "prediction_audit": audit,
        "screening_batch_grouping_audit": grouping_audit,
    }
    recommendations["normalized_layer_rows"] = _normalized_layers(run, controls + full)
    recommendations["operation_comparison"] = _operation_comparisons(run, controls + full)
    recommendations["policy_exports"] = {
        "strict": _export_recommended_policy(run, "recommended_strict_policy.json",
            "Maximize whole-encoder saved partial-product slots with no more than 0.1 percentage-point additional drop against W8A8.",
            strict, final_repeats, zero_controls),
        "fp32_2pp": _export_recommended_policy(run, "recommended_2pp_policy.json",
            "Maximize whole-encoder saved partial-product slots with no more than 2 percentage-point drop against FP32.",
            relaxed, final_repeats, zero_controls),
    }

    save_json(run / "recommendations.json", recommendations)
    save_json(run / "screening_batch_grouping_audit.json", grouping_audit)
    save_json(run / "all_results.json", {"controls": controls, "screening": screening,
                                        "full_validation": full, "unique_full_validation": unique_full,
                                        "final_repeats": final_repeats})
    fields = ["name", "scope", "mode", "threshold", "stages", "operations", "operation_count", "samples", "correct",
              "accuracy_percent", "fp32_correct", "w8a8_correct", "drop_vs_fp32_pp", "additional_drop_vs_w8a8_pp",
              "passes_extra_0_1pp", "passes_fp32_2pp", "saved_partial_product_fraction_selected",
              "saved_partial_product_fraction_encoder", "mean_terms_selected", "selected_macs_per_sample",
              "savings_measurement", "elapsed_seconds", "testing_samples_used", "directory",
              "is_final_repeat", "is_zero_tail_control", "zero_tail_guard"]
    with (run / "all_results.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in controls + screening + full:
            csv_row = dict(row)
            csv_row["operations"] = ";".join(row["operations"])
            writer.writerow(csv_row)

    lines = ["# W8A8 高低四位渐进乘法实验", "",
             "本次独立目录：`{}`。本报告从本轮保存的预测和统计汇总生成，不启动模型推理。".format(run), "",
             "筛选 {} 项，全量独立配置 {} 项，另有最终配置独立重跑 {} 次、native 精确实现控制 {} 项；控制和重跑均单独标记。".format(
                 len(screening), len(unique_full), len(final_repeats), len(controls)), "",
             "## 全量验证结论", "",
             "本轮 FP32：**{}/{}＝{:.4f}%**；完整四项 W8A8：**{}/{}＝{:.4f}%**。".format(
                 fp32["correct"], samples, 100.0 * fp32["correct"] / samples,
                 w8a8["correct"], samples, 100.0 * w8a8["correct"] / samples), "",
             "严格目标为相对本轮 W8A8 额外下降不超过 **0.1 个百分点**，至少正确 **{} 条**；另保留相对 FP32 下降不超过 **2 个百分点**的宽松目标，至少正确 **{} 条**。".format(min_strict, min_relaxed), ""]
    if strict:
        lines.append("严格目标下，已测试全量候选中节省全编码器部分积槽位最多的是 **{}**：准确率 **{:.4f}%**，比 W8A8 额外下降 **{:.4f} 个百分点**，所选操作省 **{}**，折算到全编码器省 **{}**。".format(
            strict["name"], strict["accuracy_percent"], strict["additional_drop_vs_w8a8_pp"],
            _percent(strict["saved_partial_product_fraction_selected"]), _percent(strict["saved_partial_product_fraction_encoder"])))
        if strict["mode"] == "ratio" and strict["stages"] == "ll_only":
            lines.extend(["", "本次推荐的无量纲参数为 **λ＝{:g}**：在 S3＝HH＋HL＋LH 后，取 B3＝15min(wL,aL)，满足 **S3≥{:g}B3** 时允许省略 LL；这不是原始绝对阈值 **S3≥{:g}**。`ll_only` 的近似退出只发生在第 3 项后；启用零尾部保护后，第 1、2 项后的退出全部来自精确零尾部条件。".format(
                strict["threshold"], strict["threshold"], strict["threshold"])])
    else:
        lines.append("当前全量候选中，尚未找到同时满足额外下降不超过 0.1 个百分点且能减少部分积槽位的配置；保留完整 W8A8。")
    lines.append("")
    if relaxed:
        lines.append("相对 FP32 下降不超过 2 个百分点时，节省最多的已验证候选为 **{}**，准确率 **{:.4f}%**，全编码器省部分积 **{}**。这是一项单独的宽松目标，不代替上面的严格目标。".format(
            relaxed["name"], relaxed["accuracy_percent"], _percent(relaxed["saved_partial_product_fraction_encoder"])))
        relaxed_zero_comparison = _zero_control_comparison(relaxed, zero_controls)
        if relaxed_zero_comparison["matched_operations"]:
            lines.extend(["", "该 2 个百分点目标方案与精确控制 **{}** 覆盖同一操作集合，相比该无损控制，全编码器部分积节省比例增加 **{:.4f} 个百分点**；总节省比例同时包含精确零尾部带来的收益。".format(
                relaxed_zero_comparison["zero_tail_control_name"],
                100 * relaxed_zero_comparison["saved_encoder_fraction_increment"])])
    for zero_control in zero_controls:
        lines.extend(["", "精确零尾部控制 **{}** 覆盖 {} 个 Dense 操作，logits 与 W8A8 是否逐位一致：**{}**；全编码器省部分积 **{}**。它利用 ratio λ＝65536 的精确零尾部条件，仅当保守余项界 B＝0 时提前停止，省去的尾部部分积必为零，不引入近似误差。".format(
            zero_control["name"], zero_control["operation_count"],
            "是" if zero_control.get("logits_bit_exact_vs_w8a8") else "否",
            _percent(zero_control["saved_partial_product_fraction_encoder"]))])
    if strict_zero_comparison and strict_zero_comparison["matched_operations"]:
        lines.extend(["", "严格目标推荐与精确控制 **{}** 覆盖完全相同的操作集合；相对该控制，全编码器部分积节省比例增加 **{:.4f} 个百分点**。因此不会把从 FFN 扩大到全部 Dense 的零尾部收益误算为近似独有收益。".format(
            strict_zero_comparison["zero_tail_control_name"],
            100 * strict_zero_comparison["saved_encoder_fraction_increment"])])
    elif strict:
        lines.extend(["", "当前没有与严格推荐覆盖完全相同操作集合的精确零尾部控制；不计算近似相对零尾部优化的增量，避免混淆操作范围变化带来的收益。"])
    if guarded_strict:
        lines.extend(["", "补充的零尾部保护版本中，严格目标最佳为 **{}**，准确率 **{:.4f}%**、全编码器省部分积 **{}**。总体推荐已同时比较原始与保护版本，主实验 `selected_policy.json` 和 `completion.json` 保留原记录。".format(
            guarded_strict["name"], guarded_strict["accuracy_percent"], _percent(guarded_strict["saved_partial_product_fraction_encoder"]))])
    lines.extend(["", "以下结果均来自完整 validation 的独立配置；筛选集的计数不用于最终推荐。", ""])
    lines.extend(_table(sorted(unique_full, key=_ranking, reverse=True)))
    if final_repeats:
        lines.extend(["", "最终入选配置从输入重新推理并保存完整逐层输出；下面的 `_final` 是独立重跑，不增加搜索配置数量。", ""])
        lines.extend(_table(final_repeats))
    lines.extend(["", "## 算法、阈值和计数口径", "",
                  "权重和激活先取绝对值，再拆为无符号高四位和低四位，最后恢复乘积符号；不是直接切分有符号补码。权重 w＝16wH＋wL、激活 a＝16aH＋aL，按 HH＝256wHaH、HL＝16wHaL、LH＝16wLaH、LL＝wLaL 顺序相加。全算四项应与原 INT8 乘积逐对完全一致。", "",
                  "`sigma` 比较对齐后的当前整数乘积幅值 S≥σ。σ 使用原始量化整数乘积单位；实际反量化单位还需乘以 2^−(Fw＋Fa)，所以同一个 σ 在不同层的实际数值含义可能不同。σ 越大通常越保守。", "",
                  "`ratio` 比较 S≥λB，其中 B 是尚未计算部分积幅值的保守上界；λ 为无量纲，越大通常越保守，与原始绝对阈值 σ 区分。三处上界依次为 min(240wH＋15a,240aH＋15w)、15a、min(15wL,15aL)。比较相等时停止。单次乘法的余项界不能保证点积或分类误差，因为点积存在正负抵消。", "",
                  "`ll_only` 只允许在前三项后省略 LL；`all` 允许在第一、第二、第三项后停止。stage 1～4 表示执行了多少个四位子乘法槽位。主计数包含已执行但乘数为零的项，没有任意跳过中间零项的独立优化；B＝0 提前停止则自然跳过整个零尾部。非零部分积次数是另一项统计，当前模型运行记录的是槽位数，不能把这些节省全部解释为省去了非零乘法。", "",
                  "HH/HL/LH/LL 各按 1 个四位乘法槽位计数；CPU oracle 实际使用 LUT，槽位减少不表示 CPU 指令减少。精确零尾部控制只采用 B＝0 早停，并未穷尽所有无损优化，也没有实现 a＝0 或 w＝0 时零次乘法的直接旁路（当前最少计 1 项）。因此，相对该控制的近似增量不能解释为相对所有可能无损电路的优势。", "",
                  "补充策略的 `zero_tail_guard=True` 将执行阶段设为 min(原停止阶段, 精确零尾部停止阶段)。这可以在原阈值已经选出的近似乘积基础上省去额外零尾部，乘积 LUT 与原策略保持逐整数对一致；原模式与阈值的非零余项停止规则不变。保护版本及其 `_final` 仍重新前向推理并计数，不复用原策略预测。", "",
                  "所选操作节省比例＝1−所选操作实际子乘法槽位数/(4×所选操作 MAC 数)。全编码器比例以整个编码器的四项 MAC 槽位数为分母，未近似的操作按完整四项计，便于比较只改少量操作和改大量操作的候选。筛选表与全量表都使用各自候选传播后的真实激活计数；training_policy_metrics.csv 单独保存原 W8A8 训练操作数直方图上的预测值，不能与候选运行实测混用。", "",
                  "逐操作省算表保留每个被测 Dense 的四阶段计数；`stage1_count`～`stage4_count` 表示分别在第 1～4 项后结束的乘法对数量。逐层表将这些计数按 12 个编码器层汇总，并关联该层输出 MAE/RMSE/余弦误差。`saved_partial_product_fraction_selected` 仅以该行选中操作为分母；`saved_partial_product_fraction_layer` 以整层所有矩阵的完整四项槽位为分母，包含未近似的 QK/AV 等操作；`encoder_saved_fraction_contribution` 以整个编码器为分母，各层贡献之和等于候选的全编码器节省比例。本轮 12 层形状相同，整层分母为全编码器分母的 1/12。", "",
                  "这些是算术工作量统计，没有包含阈值比较器、符号/绝对值转换、控制、额外存储、阵列等待和搬运开销。未测量 ASIC 面积、26×4 阵列周期或能耗，不能把部分积减少百分比直接当成面积、时延或功耗减少百分比。软件推理耗时只反映当前 CPU oracle，不能当作硬件加速比。", "",
                  "## 实验边界和可复核文件", "",
                  "保留 W8A8 数值配置、N16 非线性与 Softmax→AV 的 P8/F7；FP32 软件输入准备和分类头保持相同，未选中的矩阵仍按完整 INT8 乘法执行。只改变 policy.json 列出的矩阵乘法部分积停止规则，不训练、不微调、不更新权重。", "",
                  "本轮每个候选重新前向计算预测；MFCC 缓存只包含输入特征，旧实验预测不是推理输入。训练划分只用于无梯度标定或操作数分布统计，validation 用于候选调试和全量精度验证；testing 使用数为 **0**。本报告仅从本轮保存的预测重新计分复核，不读取数据集。", "",
                  "筛选批次组织差异：完整 validation 按自然顺序分批，384 条平衡筛选子集会重新组织批次。FP32 patch/head GEMM 的微小舍入差异可经过量化边界放大，因此 screen 的逐层/logits 差异不应全部归因于近似乘法；即使乘法完全精确，也可能不与全量基线按索引抽出的浮点张量逐位相同。这是同轮独立前向的数值路径差异，不表示读取旧预测。最终精度与主要误差结论以同自然顺序的 4,445 条 full 为准，screen 仅用于筛选。", "",
                  "原始逐层表的 MAE、RMSE、最大误差和余弦相似度均是候选相对本轮 W8A8 的比较。通用统计器保留了旧字段名：`fp32_max_abs` 实际表示 W8A8 参考最大绝对值，`w16a16_max_abs` 实际表示当前候选最大绝对值。汇总文件 `normalized_layer_comparison.csv/json` 已分别规范命名为 `reference_w8a8_max_abs`、`candidate_max_abs`；原始实验文件保持原样。", "",
                  "| 文件 | 内容 |", "|---|---|",
                  "| `run_metadata.json` | 本轮来源、执行配置和数据边界 |",
                  "| `baseline/fp32/`、`baseline/w8a8/` | 本轮重新推理的对照预测及 W8A8 逐层输出 |",
                  "| `trials/*__control/` | native 完整四项实现控制，单独统计 |",
                  "| `trials/*__screen/` | 筛选预测与候选真实操作数的部分积计数 |",
                  "| `trials/*__full/policy.json` | 操作列表、模式、阈值、停止阶段 |",
                  "| `trials/*__full/predictions.npz` | 完整验证 logits 和标签 |",
                  "| `trials/*__full/layer_comparison.csv/json` | 逐层误差对比 |",
                  "| `trials/*__full/operation_counts.json` | 候选实际操作数下的部分积阶段计数 |",
                  "| `all_results.csv/json` | 所有筛选和全量结果；scope 区分两者 |",
                  "| `recommendations.json` | 严格/宽松目标推荐、Pareto 候选和保存预测复核 |",
                  "| `recommended_strict_policy.json`、`recommended_2pp_policy.json` | 综合原始与保护版本后的推荐策略、完整格式/LUT来源、全量指标和可用的独立重跑目录 |",
                  "| `normalized_layer_comparison.csv/json` | 控制与全量候选逐层误差，规范参考/候选字段名 |",
                  "| `operation_count_comparison.csv/json` | 控制与全量候选逐 Dense 阶段计数、槽位节省及全编码器贡献 |",
                  "| `layer_compute_comparison.csv/json` | 逐层聚合省算并关联层输出误差，未选中层保留零贡献 |",
                  "| `training_policy_metrics.csv/json` | training 直方图上的节省量预测，单独保存 |",
                  "| `selected_policy.json`、`completion.json` | 主阶段最终独立重跑的入选策略和完成状态 |",
                  "| `guarded_completion.json` | 零尾部保护补充阶段完成状态；不覆盖主阶段记录 |",
                  "| `screening_batch_grouping_audit.json` | 精确零尾部控制在重组筛选批次及自然顺序全量批次下的 logits 对比 |", "",
                  "## 筛选结果", "",
                  "以下准确率和部分积计数均来自 screening validation 子集上的实际候选前向；子集结果不能代替完整 validation。", ""])
    for record in grouping_audit.get("screen_controls", []):
        lines.extend(["精确零尾部筛选控制 **{}** 相比从全量基线抽出的同样本结果：分类翻转 **{}** 条，变化 logits 元素 **{}** 个，logits 最大绝对差 **{:.8g}**，patch 最大绝对差 **{}**。自然顺序全量精确控制已单独核验 logits 逐位一致。".format(
            record["name"], record["classification_flips"], record["changed_logit_elements"],
            record["max_abs_logit_difference"],
            "{:.8g}".format(record["patch_max_abs_difference"]) if record["patch_max_abs_difference"] is not None else "未记录"), ""])
    lines.extend(_table(sorted(screening, key=lambda row: (row["correct"], row["saved_partial_product_fraction_encoder"]), reverse=True)))
    lines.extend(["", "推荐只表示本轮已测试配置中的结果，没有穷举所有逐层阈值，也不能保证独立最终评测集上的准确率。", ""])
    (run / README_NAME).write_text("\n".join(lines), encoding="utf-8")
    return recommendations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-directory", "--run", dest="run_directory", required=True)
    args = parser.parse_args()
    result = summarize(args.run_directory)
    print(json.dumps({"strict_recommendation": result["strict_recommendation"],
                      "fp32_2pp_recommendation": result["fp32_2pp_recommendation"],
                      "testing_samples_used": 0}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
