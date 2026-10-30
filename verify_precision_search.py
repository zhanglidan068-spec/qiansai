"""Read-only independent audit of completed precision-search artifacts.

Only Python standard library and NumPy are used.  The verifier never imports
TensorFlow, loads audio, evaluates a model, or writes to an experiment folder.
Run after search_result.json has been produced; stdout is a JSON audit report.
"""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import sys

import numpy as np


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def same(left, right, message):
    require(np.array_equal(left, right), message)


def close(left, right, message):
    require(math.isclose(float(left), float(right), rel_tol=1e-12, abs_tol=1e-10), message)


def which_set(filename):
    # Same stable filename-hash rule as the pinned Speech Commands code.
    base = str(filename).replace("\\", "/").rsplit("/", 1)[-1]
    base = re.sub(r"_nohash_.*$", "", base)
    maximum = (1 << 27) - 1
    value = (int(hashlib.sha1(base.encode("utf-8")).hexdigest(), 16)
             % (maximum + 1)) * (100.0 / maximum)
    return "validation" if value < 10 else "testing" if value < 20 else "training"


def specs():
    for layer in range(1, 13):
        prefix = "layer_{:02d}".format(layer)
        for op in ("q", "k", "v", "o"):
            yield prefix + "/" + op, 99, 64, 64, True
        yield prefix + "/qk", 99, 64, 99, False
        yield prefix + "/av", 99, 99, 64, False
        yield prefix + "/fc1", 99, 64, 256, True
        yield prefix + "/fc2", 99, 256, 64, True


def accounting(widths):
    total_macs = total_weight_bits = total_weight_elements = work = passes = signed_passes = 0
    largest_accumulator = max_a = max_b = 0
    weight_counts = {}
    for op, m, k, n, is_weight in specs():
        a_bits = int(widths[op + "/a"])
        b_bits = int(widths[op + ("/w" if is_weight else "/b")])
        require(2 <= a_bits <= 32 and 2 <= b_bits <= 32, "Invalid operand width: " + op)
        macs = m * k * n
        total_macs += macs
        work += macs * a_bits * b_bits
        passes += macs * ((a_bits + 7) // 8) * ((b_bits + 7) // 8)
        signed_passes += macs * ((a_bits - 1 + 6) // 7) * ((b_bits - 1 + 6) // 7)
        bound = k * ((1 << (a_bits - 1)) - 1) * ((1 << (b_bits - 1)) - 1)
        largest_accumulator = max(largest_accumulator, bound.bit_length() + 1)
        max_a, max_b = max(max_a, a_bits), max(max_b, b_bits)
        if is_weight:
            count = k * n
            total_weight_elements += count
            total_weight_bits += count * b_bits
            weight_counts[str(b_bits)] = weight_counts.get(str(b_bits), 0) + count
    require(total_macs == 73446912, "Encoder MAC total changed")
    require(total_weight_elements == 589824, "Encoder matrix weight count changed")
    return {
        "macs_per_audio": total_macs,
        "matrix_weight_elements": total_weight_elements,
        "matrix_weight_bits": total_weight_bits,
        "matrix_weight_kib": total_weight_bits / 8192.0,
        "maximum_a_bits": max_a,
        "maximum_b_bits": max_b,
        "maximum_safe_accumulator_bits_full_range": largest_accumulator,
        "relative_bit_product_work_vs_w8a8": work / float(total_macs * 64),
        "relative_ideal_8bit_chunk_mac_work": passes / float(total_macs),
        "relative_signed7_chunk_mac_work": signed_passes / float(total_macs),
        "weight_elements_by_effective_bits": weight_counts,
    }


def check_area_unverified(value, path):
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "area_compliance_verified":
                require(child is False, "Unsubstantiated area compliance claim: " + path)
            if key in ("area_mm2", "synthesized_area_mm2"):
                require(child is None, "Mapped area evidence must be audited separately: " + path)
            check_area_unverified(child, path + "/" + key)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            check_area_unverified(child, path + "/" + str(index))


def check_npy(path, shape, dtype, sample_count=23):
    value = np.load(str(path), mmap_mode="r", allow_pickle=False)
    require(value.shape == tuple(shape), "Wrong array shape: " + str(path))
    require(value.dtype == np.dtype(dtype), "Wrong array dtype: " + str(path))
    require(path.stat().st_size == value.offset + value.nbytes,
            "Truncated or trailing NPY bytes: " + str(path))
    rows = np.unique(np.rint(np.linspace(0, value.shape[0] - 1,
                                        min(sample_count, value.shape[0]))).astype(np.int64))
    require(np.isfinite(value[rows]).all(), "Nonfinite sampled layer values: " + str(path))
    return value, {"shape": list(shape), "dtype": str(value.dtype),
                   "file_bytes": path.stat().st_size,
                   "finite_sample_rows": rows.tolist()}


def unpack_codes(payload, offset, length, bits, shape):
    count = int(np.prod(shape))
    require(length == (count * bits + 7) // 8, "Packed tensor length mismatch")
    block = memoryview(payload)[offset:offset + length]
    require(len(block) == length, "Packed tensor truncation")
    result = np.empty(count, np.int64)
    cursor = accumulator = used = 0
    mask, sign = (1 << bits) - 1, 1 << (bits - 1)
    for index in range(count):
        while used < bits:
            accumulator |= int(block[cursor]) << used
            cursor += 1
            used += 8
        code = accumulator & mask
        accumulator >>= bits
        used -= bits
        result[index] = code - (1 << bits) if code & sign else code
    require(accumulator == 0, "Nonzero unused packed padding bits")
    return result.reshape(shape)


def float_weight_key(operand_key):
    pieces = operand_key.split("/")
    if pieces[0] in ("patch", "head"):
        return pieces[0] + "_kernel"
    return pieces[0] + "_" + pieces[1] + "_kernel"


def verify_per_channel(run, reference, targets, screening, minimum, float_weights):
    """Optional second phase: per-column scales, including 4-bit signed data."""
    result_path = run / "per_channel_search_result.json"
    if not result_path.is_file():
        return {"present": False, "note": "Optional second-stage result is absent"}
    directory = run / "per_channel_search"
    summary = read_json(result_path)
    require(summary == read_json(directory / "search_result.json"), "PC parent/local summary mismatch")
    check_area_unverified(summary, "per_channel")
    n = len(targets)
    require(summary["samples"] == n and summary["testing_samples_used"] == 0, "PC sample provenance mismatch")
    require(summary["weight_scale_granularity"] == "per output channel", "PC granularity mismatch")
    require(summary["channel_F_table_bytes"] == 6912, "PC scale-table declared size mismatch")
    ranges = read_json(run / "calibration_ranges.json")["matrix_operands"]
    rows = read_json(directory / "experiment_results.json")
    trials, identified = {}, set()
    expected_weight_names = {op + "/w" for op, _, _, _, is_weight in specs() if is_weight}
    for row in rows:
        name, scope = row["name"], row["scope"]
        require(scope in ("screen", "full", "final"), "PC scope unknown")
        ident = name + "__" + scope
        require(ident not in identified, "Duplicate PC trial")
        identified.add(ident)
        trial_dir = directory / ident
        require(read_json(trial_dir / "result.json") == row, "PC trial summary mismatch: " + ident)
        check_area_unverified(row, ident)
        require(row["testing_samples_used"] == 0, "PC testing use not zero")
        indices = np.load(str(trial_dir / "sample_indices.npy"), allow_pickle=False)
        expected_indices = screening if scope == "screen" else np.arange(n, dtype=np.int64)
        same(indices, expected_indices, "PC sample indices mismatch: " + ident)
        require(row["samples"] == len(indices) and row["is_full_validation"] is (scope != "screen"),
                "PC screen/full flags mismatch")
        with np.load(str(trial_dir / "predictions.npz"), allow_pickle=False) as saved:
            logits, true, baseline = saved["logits"].copy(), saved["target_ids"].copy(), saved["baseline_logits"].copy()
        require(logits.shape == (len(indices), 12) and np.isfinite(logits).all(), "PC logits invalid")
        same(true, targets[indices], "PC labels differ from parent manifest")
        same(baseline, reference[indices], "PC reference differs from fresh parent FP32")
        correct = int((logits.argmax(1) == true).sum())
        base_correct = int((baseline.argmax(1) == true).sum())
        drop = 100.0 * (base_correct - correct) / len(indices)
        require(row["correct"] == correct and row["fp32_correct"] == base_correct, "PC score mismatch")
        close(row["accuracy"], correct / float(len(indices)), "PC accuracy mismatch")
        close(row["accuracy_drop_pp"], drop, "PC drop mismatch")
        require(row["passes_2pp"] is (drop <= 2 + 1e-12), "PC pass flag mismatch")
        if scope != "screen":
            require(row["passes_2pp"] is (correct >= minimum), "PC full minimum mismatch")
        precision = read_json(trial_dir / "precision_config.json")
        policy = precision["policy"]
        require(policy["name"] == name, "PC policy name mismatch")
        require(precision["testing_samples_used"] == 0, "PC precision testing use not zero")
        require(precision["weight_scale_granularity"] == "per output channel", "PC precision granularity mismatch")
        require(precision["accelerator_scope"] == "encoder only; FP32 patch/head", "PC hardware boundary mismatch")
        require(set(precision["matrix_operand_bits"]) == set(ranges), "PC operand set mismatch")
        for key, bits in precision["matrix_operand_bits"].items():
            expected_bits = policy["w"] if key.endswith("/w") else policy["p"] if key.endswith("/av/a") else policy["a"]
            require(bits == expected_bits, "PC operand width does not match policy")
            found_f = precision["fractional_bits"][key]
            if key.endswith("/w"):
                weights = float_weights[float_weight_key(key)]
                maximum = np.abs(weights).max(axis=0).astype(np.float64)
                safe = np.where(maximum == 0, 1.0, maximum)
                expected_f = np.floor(np.log2(((1 << (bits - 1)) - 1.0) / safe)).astype(np.int32)
                expected_f[maximum == 0] = 0
                same(np.asarray(found_f, np.int32), expected_f, "PC per-column F is not derived from frozen weights")
            else:
                maximum = float(ranges[key])
                expected_f = int(policy["pf"]) if key.endswith("/av/a") else 0 if maximum == 0 else int(
                    math.floor(math.log2(((1 << (bits - 1)) - 1.0) / (maximum * 1.25))))
                require(found_f == expected_f, "PC activation F differs from training calibration")
        calculated = accounting(precision["matrix_operand_bits"])
        close(row["matrix_weight_kib"], calculated["matrix_weight_kib"], "PC weight accounting mismatch")
        close(row["arithmetic_work_proxy"], calculated["relative_bit_product_work_vs_w8a8"], "PC work accounting mismatch")
        require(row["per_channel_scale_bytes"] == 6912, "PC row scale table size mismatch")
        trials[ident] = {"row": row, "logits": logits, "precision": precision, "accounting": calculated}
    require({p.parent.name for p in directory.glob("*/result.json")} == set(trials),
            "PC completed trial list mismatch")
    screen_rows = [row for row in rows if row["scope"] == "screen"]
    require(len(screen_rows) == 9 == summary["screened_candidates"], "PC nine screening candidates incomplete")
    require(len({row["name"] for row in screen_rows}) == 9, "Duplicate PC screen policy")
    full_rows = [row for row in rows if row["scope"] == "full"]
    require(full_rows == summary["full_validation_candidates"], "PC full candidate list mismatch")
    valid = [row for row in full_rows if row["passes_2pp"]]
    require(bool(valid), "No full passing PC candidate")
    best = sorted(valid, key=lambda row: (-row["accuracy"], row["matrix_weight_kib"], row["arithmetic_work_proxy"]))[0]
    efficient = sorted(valid, key=lambda row: (row["matrix_weight_kib"], row["arithmetic_work_proxy"], -row["accuracy"]))[0]
    require(best == summary["accuracy_winner"] and efficient == summary["efficient_winner"],
            "PC winner selection mismatch")
    require(set(summary["finals"]) == {best["name"], efficient["name"]}, "PC final winner set mismatch")
    final_reports = {}
    for name, final_row in summary["finals"].items():
        ident = name + "__final"
        require(final_row == trials[ident]["row"], "PC final row mismatch")
        require(final_row["correct"] >= minimum, "PC final misses accuracy threshold")
        same(trials[ident]["logits"], trials[name + "__full"]["logits"], "PC full-final logits not bitexact")
        final_dir = directory / ident
        layer_dir = final_dir / "layer_outputs"
        shapes = {"patch_output": (n, 98, 64), "encoder_input": (n, 99, 64), "logits": (n, 12)}
        shapes.update({"layer_{:02d}_output".format(layer): (n, 99, 64) for layer in range(1, 13)})
        require({path.stem for path in layer_dir.glob("*.npy")} == set(shapes), "PC 15 layer outputs incomplete")
        arrays = {}
        for stage, shape in shapes.items():
            values, arrays[stage] = check_npy(layer_dir / (stage + ".npy"), shape, np.float32)
            if stage == "logits":
                same(values, trials[ident]["logits"], "PC saved layer logits differ")
        description = read_json(final_dir / "weights_manifest.json")
        payload = (final_dir / "weights_packed.bin").read_bytes()
        scale_payload = (final_dir / "weight_channel_F_int8.bin").read_bytes()
        require(len(payload) == description["total_bytes"], "PC packed bytes mismatch")
        require(len(scale_payload) == description["channel_scale_bytes"] == 6912, "PC F table bytes mismatch")
        require(set(description["tensors"]) == expected_weight_names, "PC packed tensor set mismatch")
        with np.load(str(final_dir / "weights_codes.npz"), allow_pickle=False) as saved:
            codes = {key: saved[key].copy() for key in saved.files}
        require(set(codes) == {key.replace("/", "__") for key in expected_weight_names}, "PC code NPZ tensor set mismatch")
        data_ranges, scale_ranges, count, bit_count = [], [], 0, 0
        precision = trials[ident]["precision"]
        for key, tensor in description["tensors"].items():
            bits, offset, length = int(tensor["bits"]), int(tensor["byte_offset"]), int(tensor["byte_length"])
            require(bits in (4, 6), "PC export expected 4-bit or 6-bit data")
            require(bits == precision["matrix_operand_bits"][key], "PC export width mismatch")
            values = unpack_codes(payload, offset, length, bits, tuple(tensor["shape"]))
            same(values, codes[key.replace("/", "__")], "PC packed/code NPZ mismatch")
            f_offset, f_length = int(tensor["f_table_offset"]), int(tensor["f_table_length"])
            fs = np.frombuffer(scale_payload, dtype=np.int8, count=f_length, offset=f_offset).astype(np.int32)
            same(fs, np.asarray(tensor["fractional_bits"], np.int32), "PC packed/manifest F mismatch")
            same(fs, np.asarray(precision["fractional_bits"][key], np.int32), "PC packed/config F mismatch")
            fp32 = float_weights[float_weight_key(key)]
            require(values.shape == fp32.shape and f_length == fp32.shape[1], "PC matrix/F shape mismatch")
            maximum = np.abs(fp32).max(axis=0).astype(np.float64)
            expected_f = np.floor(np.log2(((1 << (bits - 1)) - 1.0) / np.where(maximum == 0, 1.0, maximum))).astype(np.int32)
            expected_f[maximum == 0] = 0
            same(fs, expected_f, "PC F bytes not derived from frozen FP32 column maxima")
            scaled = fp32.astype(np.float32) * np.ldexp(np.ones(fs.shape, np.float32), fs)
            bound = (1 << (bits - 1)) - 1
            expected_codes = np.clip(np.rint(scaled), -bound, bound).astype(np.int64)
            same(values, expected_codes, "PC codes differ from per-column FP32 quantization")
            data_ranges.append((offset, offset + length))
            scale_ranges.append((f_offset, f_offset + f_length))
            count += values.size
            bit_count += values.size * bits
        for ranges_to_check, total in ((data_ranges, len(payload)), (scale_ranges, len(scale_payload))):
            cursor = 0
            for begin, end in sorted(ranges_to_check):
                require(begin == cursor, "PC data/F table gap or overlap")
                cursor = end
            require(cursor == total, "PC unaccounted data/F bytes")
        require(count == 589824 and bit_count // 8 == len(payload), "PC total weight accounting mismatch")
        require(bit_count == trials[ident]["accounting"]["matrix_weight_bits"], "PC precision/packing accounting mismatch")
        final_reports[name] = {"correct": final_row["correct"], "packed_weight_bytes": len(payload),
                               "channel_F_bytes": len(scale_payload), "tensors": 72,
                               "codes_and_channel_F_match_frozen_weights": True, "layer_outputs": arrays}
    source_hashes = read_json(directory / "source_sha256.json")
    require(len(source_hashes) == 6, "PC expected six frozen source files")
    require({path.name for path in (directory / "source").iterdir() if path.is_file()} == set(source_hashes),
            "PC source manifest incomplete")
    for name, digest in source_hashes.items():
        require(file_sha256(directory / "source" / name) == digest, "PC source SHA mismatch: " + name)
    return {"present": True, "passed": True, "screen_candidates": len(screen_rows),
            "full_candidates": len(full_rows), "final_candidates": len(final_reports),
            "accuracy_winner": best, "efficient_winner": efficient,
            "finals": final_reports, "source_snapshot_sha256": source_hashes,
            "testing_samples_used": 0, "area_compliance_verified": False}


def verify(run, unit_report):
    report = {"run": str(run), "testing_audio_read_by_verifier": 0,
              "tensorflow_imported_by_verifier": False, "writes_to_run": False}
    require((run / "search_result.json").is_file(), "Search is incomplete: no search_result.json")
    summary = read_json(run / "search_result.json")
    prep = read_json(run / "preparation_complete.json")
    metadata = read_json(run / "run_metadata.json")
    manifest = read_json(run / "sample_manifest.json")
    calibration = read_json(run / "calibration_manifest.json")
    screening = np.asarray(read_json(run / "screening_indices.json"), np.int64)
    n = len(manifest)
    require(n == 4445 == prep["samples"] == summary["samples"], "Validation count mismatch")
    require(prep["sample_requests"] == {"training": len(calibration), "validation": n, "testing": 0},
            "Unexpected audio sample requests")
    require(len(calibration) == prep["calibration_samples"] == metadata["calibration_samples"],
            "Calibration count mismatch")
    for rows, split in ((manifest, "validation"), (calibration, "training")):
        for index, row in enumerate(rows):
            require(int(row["index"]) == index, "Manifest indices are not contiguous")
            if not row["synthetic_zero_waveform"]:
                require(which_set(row["file"]) == split, "Wrong manifest split: " + row["file"])
    for obj in (summary, metadata):
        require(obj.get("testing_samples_used") == 0, "Testing use is not explicitly zero")
        check_area_unverified(obj, "metadata")
    require(screening.ndim == 1 and len(screening) == prep["screening_samples"], "Screen count mismatch")
    require(np.all(screening >= 0) and np.all(screening < n), "Invalid screen indices")
    require(np.array_equal(np.unique(screening), screening), "Screen indices not unique and sorted")
    with np.load(str(run / "fresh_fp32_reference.npz"), allow_pickle=False) as found:
        reference, targets = found["logits"].copy(), found["target_ids"].copy()
    require(reference.shape == (n, 12) and np.isfinite(reference).all(), "Invalid fresh FP32 logits")
    same(targets, np.asarray([row["label_id"] for row in manifest], np.int64), "Manifest label mismatch")
    correct = int((reference.argmax(1) == targets).sum())
    require(correct == 4340 == prep["fp32_correct"] == summary["fp32_correct"], "Fresh FP32 score mismatch")
    minimum = int(math.ceil(correct - .02 * n - 1e-10))
    require(minimum == 4252 == prep["min_correct_for_2pp_drop"] == summary["minimum_correct"],
            "Wrong 2pp threshold")
    close(prep["fp32_accuracy"], correct / float(n), "FP32 accuracy mismatch")
    histogram = np.bincount(targets[screening], minlength=12)
    require(np.all(histogram == histogram[0]), "Screening is not class balanced")
    ranges = read_json(run / "calibration_ranges.json")
    require(ranges["calibration_split"] == "training" and ranges["samples"] == len(calibration),
            "Calibration provenance mismatch")
    for key, digest in (("feature_sha256", run / "validation_mfcc.npy"),
                        ("calibration_ranges_sha256", run / "calibration_ranges.json")):
        require(file_sha256(digest) == prep[key], "Preparation content hash mismatch: " + str(digest))
    for name, shape in (("validation_mfcc", (n, 98, 40)),
                        ("calibration_mfcc", (len(calibration), 98, 40))):
        _, details = check_npy(run / (name + ".npy"), shape, np.float32)
        report[name] = details
    report["preparation"] = {"validation": n, "training": len(calibration), "testing": 0,
                             "fp32_correct": correct, "minimum_correct": minimum,
                             "balanced_screen_per_class": int(histogram[0])}

    all_rows = read_json(run / "experiment_results.json")
    require(len({row["trial"] for row in all_rows}) == len(all_rows), "Duplicate trial summaries")
    trials = {}
    for row in all_rows:
        trial = row["trial"]
        directory = run / "trials" / trial
        saved = read_json(directory / "result.json")
        require(saved == row, "Trial summary does not match result.json: " + trial)
        scope = row["scope"]
        require(scope in ("screen", "full", "final"), "Unknown trial scope")
        require(trial.endswith("__" + scope), "Scope/name mismatch")
        expected_indices = screening if scope == "screen" else np.arange(n, dtype=np.int64)
        indices = np.load(str(directory / "sample_indices.npy"), allow_pickle=False)
        same(indices, expected_indices, "Trial indices do not match scope: " + trial)
        require(row["samples"] == len(indices), "Trial sample count mismatch: " + trial)
        require(row["is_full_validation"] is (scope != "screen"), "Screen/full status mismatch")
        require(row["testing_samples_used"] == 0, "Trial testing use not zero")
        with np.load(str(directory / "predictions.npz"), allow_pickle=False) as saved_predictions:
            logits = saved_predictions["logits"].copy()
            true = saved_predictions["target_ids"].copy()
            base = saved_predictions["baseline_logits"].copy()
        require(logits.shape == (len(indices), 12) and np.isfinite(logits).all(), "Invalid trial logits")
        same(true, targets[indices], "Trial ground truth mismatch: " + trial)
        same(base, reference[indices], "Trial baseline does not match fresh FP32: " + trial)
        trial_correct = int((logits.argmax(1) == true).sum())
        base_correct = int((base.argmax(1) == true).sum())
        drop = 100.0 * (base_correct - trial_correct) / len(indices)
        require(row["correct"] == trial_correct and row["fp32_correct"] == base_correct,
                "Trial score mismatch: " + trial)
        close(row["accuracy"], trial_correct / float(len(indices)), "Trial accuracy mismatch")
        close(row["accuracy_drop_pp"], drop, "Trial drop mismatch")
        require(row["passes_2pp"] is (drop <= 2.0 + 1e-12), "Trial pass flag mismatch")
        if scope != "screen":
            require(row["passes_2pp"] is (trial_correct >= minimum), "Full minimum-correct mismatch")
        quantization = read_json(directory / "quantization.json")
        require(quantization["testing_samples_used"] == 0, "Quantization testing use not zero")
        require(quantization["software_boundaries"] is True, "Encoder boundary unexpectedly quantized")
        stats = read_json(directory / "area_accounting.json")
        calculated = accounting(quantization["operand_bits"])
        for key, value in calculated.items():
            if key == "weight_elements_by_effective_bits":
                continue
            close(stats[key], value, "Accounting mismatch " + trial + "/" + key)
        close(row["matrix_weight_kib"], calculated["matrix_weight_kib"], "Weight KiB mismatch")
        close(row["arithmetic_work_proxy"], calculated["relative_bit_product_work_vs_w8a8"], "Work proxy mismatch")
        check_area_unverified(saved, trial)
        check_area_unverified(stats, trial + "/accounting")
        if row["name"] == "fp32_bypass_control":
            require(np.allclose(logits, base, rtol=1e-5, atol=1e-5), "FP32 bypass failed")
        trials[trial] = {"row": row, "logits": logits, "policy": read_json(directory / "policy.json"),
                         "quantization": quantization, "accounting": calculated}
    result_directories = {p.parent.name for p in (run / "trials").glob("*/result.json")}
    require(result_directories == set(trials), "Unlisted or missing completed trials")
    with (run / "layer_sensitivity.csv").open(encoding="utf-8-sig", newline="") as handle:
        sensitive = list(csv.DictReader(handle))
    pairs = {(int(row["layer"]), int(row["effective_bits"])) for row in sensitive}
    require(len(sensitive) == 48 and pairs == {(layer, bits) for layer in range(1, 13)
                                             for bits in (8, 10, 12, 16)}, "Incomplete 12 x 4 sensitivity grid")
    for row in sensitive:
        trial = row["trial"]
        require(trial in trials and trials[trial]["row"]["scope"] == "screen", "Sensitivity scope mismatch")
        require(int(row["correct"]) == trials[trial]["row"]["correct"], "Sensitivity score mismatch")
        require(trials[trial]["policy"].get("only_layer") == "layer_{:02d}".format(int(row["layer"])),
                "Sensitivity layer policy mismatch")
    report["trial_audit"] = {"completed_trials": len(trials), "sensitivity_grid": "12 x 4 complete",
                             "scopes": {scope: sum(t["row"]["scope"] == scope for t in trials.values())
                                        for scope in ("screen", "full", "final")}}

    final_details = {}
    role_fields = (("accuracy", "accuracy_winner", "selected_final"),
                   ("efficiency", "efficient_winner", "efficient_final"))
    for role, winner_field, final_field in role_fields:
        winner, final = summary[winner_field], summary[final_field]
        require(winner["scope"] == "full" and winner["passes_2pp"], "Winner lacks full passing validation")
        require(winner["correct"] >= minimum, "Winner misses 2pp requirement")
        require(final["scope"] == "final" and final["name"] == winner["name"], "Winner/final mismatch")
        require(winner == trials[winner["trial"]]["row"] and final == trials[final["trial"]]["row"],
                "Winner summary mismatch")
        same(trials[winner["trial"]]["logits"], trials[final["trial"]]["logits"],
             "Final does not reproduce full logits bit-for-bit")
        layer_dir = run / "trials" / final["trial"] / "layer_outputs"
        expected = {"patch_output": (n, 98, 64), "encoder_input": (n, 99, 64), "logits": (n, 12)}
        expected.update({"layer_{:02d}_output".format(layer): (n, 99, 64) for layer in range(1, 13)})
        require({p.stem for p in layer_dir.glob("*.npy")} == set(expected), "Final 15-stage outputs incomplete")
        detail = {}
        for key, shape in expected.items():
            values, detail[key] = check_npy(layer_dir / (key + ".npy"), shape, np.float32)
            if key == "logits":
                same(values, trials[final["trial"]]["logits"], "Layer logits differ from final predictions")
        final_details[role] = {"trial": final["trial"], "correct": final["correct"],
                               "layer_outputs": detail}
    report["final_outputs"] = final_details

    with np.load(str(run / "weights_fp32.npz"), allow_pickle=False) as saved:
        float_weights = {key: saved[key].copy() for key in saved.files}
    with np.load(str(run / "best_weights_mixed.npz"), allow_pickle=False) as saved:
        best_weights = {key: saved[key].copy() for key in saved.files}
    require(len(best_weights) == 72, "Expected 72 encoder weight matrices")
    export_results = {}
    for role, winner_field, _ in role_fields:
        exported = summary["hardware_weight_exports"][role]
        directory = run / exported["directory"]
        description = read_json(directory / "weights_manifest.json")
        payload = (directory / "weights_packed.bin").read_bytes()
        require(len(payload) == description["total_bytes"] == exported["packed_weight_bytes"],
                "Packed total byte mismatch")
        trial = trials[summary[winner_field]["trial"]]
        require(description["policy"] == trial["policy"], "Export policy mismatch")
        require(description["matrix_operand_bits"] == trial["quantization"]["operand_bits"],
                "Export widths mismatch")
        require(description["fractional_bits"] == trial["quantization"]["fractional_bits"],
                "Export scales mismatch")
        expected_names = {op + "/w" for op, _, _, _, is_weight in specs() if is_weight}
        require(set(description["tensors"]) == expected_names, "Packed matrix list mismatch")
        used_ranges, elements, bit_count = [], 0, 0
        for key, tensor in description["tensors"].items():
            bits, offset, length = int(tensor["bits"]), int(tensor["byte_offset"]), int(tensor["byte_length"])
            require(bits == description["matrix_operand_bits"][key], "Tensor width mismatch")
            require(int(tensor["fractional_bits"]) == description["fractional_bits"][key], "Tensor F mismatch")
            values = unpack_codes(payload, offset, length, bits, tuple(tensor["shape"]))
            layer, op, _ = key.split("/")
            fp32 = float_weights[layer + "_" + op + "_kernel"]
            require(values.shape == fp32.shape, "Packed matrix shape mismatch")
            scaled = fp32.astype(np.float32) * np.float32(math.ldexp(1.0, int(tensor["fractional_bits"])))
            maximum = (1 << (bits - 1)) - 1
            expected = np.clip(np.rint(scaled), -maximum, maximum).astype(np.int64)
            same(values, expected, "Packed codes differ from quantized official weights: " + role + "/" + key)
            if role == "accuracy":
                same(values, best_weights[key.replace("/", "__")], "Best weight NPZ/packed mismatch")
            used_ranges.append((offset, offset + length))
            elements += values.size
            bit_count += values.size * bits
        cursor = 0
        for begin, end in sorted(used_ranges):
            require(begin == cursor, "Gap or overlap in packed tensor offsets")
            cursor = end
        require(cursor == len(payload), "Packed payload has unaccounted bytes")
        require(elements == 589824, "Packed element count mismatch")
        require(bit_count == trial["accounting"]["matrix_weight_bits"], "Packed precision accounting mismatch")
        require(len(payload) == bit_count // 8, "Packed byte accounting mismatch")
        export_results[role] = {"tensors": 72, "elements": elements, "effective_bits": bit_count,
                                 "packed_bytes": len(payload),
                                 "elements_by_bits": trial["accounting"]["weight_elements_by_effective_bits"],
                                 "codes_match_fresh_weight_quantization": True}
    report["packed_weights"] = export_results

    source_hashes = read_json(run / "search_source_sha256.json")
    require({p.name for p in (run / "search_source").iterdir() if p.is_file()} == set(source_hashes),
            "Search source manifest incomplete")
    for name, digest in source_hashes.items():
        require(file_sha256(run / "search_source" / name) == digest, "Source snapshot SHA mismatch: " + name)
    report["source_snapshot_sha256"] = source_hashes
    report["preparation_source_sha256_computed"] = {
        path.name: file_sha256(path) for path in sorted((run / "source").iterdir()) if path.is_file()}
    require(unit_report.is_file(), "Compact nonlinear unit-test report missing")
    unit = read_json(unit_report)
    require(unit["passed"] is True and unit["testing_audio_used"] == 0, "Compact unit report failed")
    require(unit["exponential_lookup_all_8193_indices_bitexact"] is True, "Compact exp proof missing")
    require(unit["twelve_layer_nested_cond_test"] is True, "Shared table branch test missing")
    require(len(unit["shared_root_rom_constants"]) == 2 and not unit["branch_rom_constants"],
            "ROM constants are duplicated")
    gelu_meta, softmax_meta = unit["gelu_metadata"], unit["softmax_metadata"]
    require(gelu_meta["lut_entries"] == 1025 and softmax_meta["lut_entries"] == 4096,
            "Compact ROM entry count mismatch")
    require(gelu_meta["lut_payload_bytes"] + softmax_meta["lut_payload_bytes"] == 10242,
            "Compact ROM payload mismatch")
    report["compact_rom"] = {"total_payload_bytes": 10242, "shared_constants": 2,
                             "unit_report_sha256": file_sha256(unit_report)}
    report["per_channel_search"] = verify_per_channel(run, reference, targets, screening,
                                                      minimum, float_weights)
    report["area_compliance_verified"] = False
    report["area_status"] = "No physical area conclusion; requires official mapped netlist and SRAM areas"
    report["passed"] = True
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--unit-report", type=Path,
                        default=Path(__file__).absolute().parent / ".tmp/compact_nonlinear_unit_results.json")
    args = parser.parse_args()
    try:
        report = verify(args.run.absolute(), args.unit_report.absolute())
    except Exception as error:
        print(json.dumps({"passed": False, "run": str(args.run.absolute()),
                          "error_type": type(error).__name__, "error": str(error),
                          "testing_audio_read_by_verifier": 0}, ensure_ascii=False, indent=2))
        return 1
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
