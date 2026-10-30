"""KWT-1 encoder precision accounting, NOT a synthesized area estimate.

The contest scope keeps patch projection and classification in software.  This
module therefore defaults to only the 12 accelerated encoder blocks.  It uses
Python's standard library and never reads audio, predictions or dataset splits.
"""

import argparse
import csv
import json
import math
from pathlib import Path


def _bits(value):
    value = int(value)
    if not 2 <= value <= 32:
        raise ValueError("Precision must be between 2 and 32 bits")
    return value


def safe_signed_accumulator_bits(a_bits, b_bits, reduction_length):
    """Exact full-range bound for symmetric signed integer operands."""
    bound = int(reduction_length) * ((1 << (_bits(a_bits) - 1)) - 1) * \
        ((1 << (_bits(b_bits) - 1)) - 1)
    return max(1, bound.bit_length() + 1)


def operation_specs(accelerated_only=True):
    specs = []
    if not accelerated_only:
        specs.append(("patch", "patch", 98, 40, 64, True))
    for index in range(1, 13):
        prefix = "layer_{:02d}".format(index)
        for suffix in ("q", "k", "v"):
            specs.append((prefix + "/" + suffix, suffix, 99, 64, 64, True))
        specs.extend([
            (prefix + "/qk", "qk", 99, 64, 99, False),
            (prefix + "/av", "av", 99, 99, 64, False),
            (prefix + "/o", "o", 99, 64, 64, True),
            (prefix + "/fc1", "fc1", 99, 64, 256, True),
            (prefix + "/fc2", "fc2", 99, 256, 64, True),
        ])
    if not accelerated_only:
        specs.append(("head", "head", 1, 64, 12, True))
    return specs


def precision_accounting(policy=None, accelerated_only=True):
    """Return operation rows and storage/work proxies for a precision policy.

    Policy keys: default_activation_bits, default_weight_bits,
    av_probability_bits, av_fractional_bits, op_overrides (operation->dict with
    a_bits/b_bits), matrix_operand_bits (operand key->integer bits).
    No accuracy result is inferred from these analytical quantities.
    """
    policy = policy or {}
    default_a = _bits(policy.get("default_activation_bits", 8))
    default_w = _bits(policy.get("default_weight_bits", 8))
    av_bits = _bits(policy.get("av_probability_bits", default_a))
    av_f = int(policy.get("av_fractional_bits", av_bits - 2))
    operand_overrides = policy.get("matrix_operand_bits", {})
    overrides = policy.get("op_overrides", {})
    rows = []
    for name, group, m, k, n, is_weight in operation_specs(accelerated_only):
        local = overrides.get(name, {})
        a_bits = _bits(local.get("a_bits", av_bits if group == "av" else default_a))
        b_bits = _bits(local.get("b_bits", default_w if is_weight else default_a))
        a_bits = _bits(operand_overrides.get(name + "/a", a_bits))
        b_key = name + ("/w" if is_weight else "/b")
        b_bits = _bits(operand_overrides.get(b_key, b_bits))
        macs = m * k * n
        weights = k * n if is_weight else 0
        # Eight-bit chunks require unsigned low-digit handling/sign correction.
        ideal_passes = math.ceil(a_bits / 8.0) * math.ceil(b_bits / 8.0)
        # Seven-bit nonnegative low digits fit an ordinary signed INT8 PE.
        signed7_passes = math.ceil((a_bits - 1) / 7.0) * math.ceil((b_bits - 1) / 7.0)
        rows.append({
            "operation": name, "group": group, "rows_m": m,
            "reduction_k": k, "columns_n": n, "macs": macs,
            "a_bits": a_bits, "b_bits": b_bits,
            "matrix_weight_elements": weights,
            "matrix_weight_bits": weights * b_bits,
            "safe_accumulator_bits_full_range": safe_signed_accumulator_bits(a_bits, b_bits, k),
            "multiplier_bit_product_work_proxy": macs * a_bits * b_bits,
            "ideal_8bit_chunk_passes": ideal_passes,
            "signed7_chunk_passes": signed7_passes,
            "ideal_8bit_chunk_mac_work": macs * ideal_passes,
            "signed7_chunk_mac_work": macs * signed7_passes,
        })
    weight_bits = sum(row["matrix_weight_bits"] for row in rows)
    macs = sum(row["macs"] for row in rows)
    summary = {
        "scope": "encoder_12_blocks" if accelerated_only else "whole_model_matrix_ops",
        "synthesized_area_mm2": None,
        "contest_area_limit_mm2": 2.25,
        "area_compliance_verified": False,
        "macs_per_audio": macs,
        "matrix_weight_elements": sum(row["matrix_weight_elements"] for row in rows),
        "matrix_weight_bits": weight_bits,
        "matrix_weight_kib": weight_bits / 8.0 / 1024.0,
        "shared_sram_budget_kib": 512,
        "weight_only_excess_above_512kib": max(0.0, weight_bits / 8.0 / 1024.0 - 512),
        "maximum_a_bits": max(row["a_bits"] for row in rows),
        "maximum_b_bits": max(row["b_bits"] for row in rows),
        "maximum_safe_accumulator_bits_full_range": max(row["safe_accumulator_bits_full_range"] for row in rows),
        "weighted_mean_operand_bit_product": sum(row["multiplier_bit_product_work_proxy"] for row in rows) / float(macs),
        "relative_bit_product_work_vs_w8a8": sum(row["multiplier_bit_product_work_proxy"] for row in rows) / float(macs * 64),
        "relative_ideal_8bit_chunk_mac_work": sum(row["ideal_8bit_chunk_mac_work"] for row in rows) / float(macs),
        "relative_signed7_chunk_mac_work": sum(row["signed7_chunk_mac_work"] for row in rows) / float(macs),
        "av_probability_bits_default": av_bits,
        "av_fractional_bits_default": av_f,
        "warnings": [
            "Bit-product work is a numerical arithmetic-cost proxy, not cell or post-layout area.",
            "A shared native-width array must support its maximum operand widths; layer averages do not predict its area.",
            "A shared INT8 array can use multiple passes; eight-bit chunk counts exclude sign correction, shifts and scheduling overhead.",
            "Signed-seven-bit chunk counts permit signed INT8 partial products, but exclude partial-sum/rescale/control cost.",
            "The 512 KiB subtraction is a weight-only capacity comparison; activations, nonlinear tables and double buffers also consume storage.",
            "Required area must include added SRAM/ROM macros, interfaces, control, nonlinear units and timing fixes, following the contest exclusion rules.",
        ],
    }
    return rows, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--out-json", type=Path)
    parser.add_argument("--out-csv", type=Path)
    parser.add_argument("--include-software-ops", action="store_true")
    args = parser.parse_args()
    policy = json.loads(args.policy.read_text(encoding="utf-8")) if args.policy else {}
    rows, summary = precision_accounting(policy, not args.include_software_ops)
    payload = {"policy": policy, "summary": summary, "operations": rows}
    if args.out_json:
        with args.out_json.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
    if args.out_csv:
        with args.out_csv.open("x", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
