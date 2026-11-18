"""Independent exhaustive arithmetic checks for nibble_early_exit.

This verifier reads no audio, samples, model predictions, or experiment data.
It prints JSON and writes no files unless stdout is explicitly redirected.
"""

import json

import numpy as np

from nibble_early_exit import lookup_products, make_lut


def verify():
    domain = np.arange(-127, 128, dtype=np.int64)
    exact_products = domain[:, None] * domain[None, :]
    exact = make_lut("exact")
    assert np.array_equal(exact["product_lut"], exact_products)
    assert np.all(exact["stage_lut"] == 4)
    assert np.all(exact["remaining_bound_lut"] == 0)
    assert exact["statistics"]["bound_violation_count"] == 0
    assert exact["statistics"]["executed_term_slots"] == 65025 * 4
    assert np.array_equal(lookup_products(domain[:, None], domain[None, :], exact["product_lut"]), exact_products)

    # A separate scalar oracle checks decomposition, signs, all three bounds,
    # and stopping decisions, without using the module's intermediate arrays.
    configs = [("exact", 0.0, "all"),
               ("sigma", 0.0, "all"), ("sigma", 0.0, "ll_only"),
               ("sigma", 256.0, "all"), ("sigma", 1024.0, "all"),
               ("sigma", 1e20, "all"), ("sigma", 1e20, "ll_only"),
               ("ratio", 0.0, "all"), ("ratio", 1.0, "all"),
               ("ratio", 4.0, "all"), ("ratio", 16.0, "all"),
               ("ratio", 4.0, "ll_only")]
    outputs = [make_lut(mode, threshold, stages) for mode, threshold, stages in configs]
    pairs_checked = 0
    bound_checks = 0
    for activation in range(-127, 128):
        for weight in range(-127, 128):
            a = abs(activation)
            w = abs(weight)
            ah, al = divmod(a, 16)
            wh, wl = divmod(w, 16)
            terms = [wh * ah * 256, wh * al * 16, wl * ah * 16, wl * al]
            assert sum(terms) == a * w
            bounds = [min(240 * wh + 15 * a, 240 * ah + 15 * w),
                      15 * a, min(15 * wl, 15 * al)]
            for stage in range(3):
                omitted = sum(terms[stage + 1:])
                assert omitted <= bounds[stage]
                bound_checks += 1
            for (mode, threshold, stages), output in zip(configs, outputs):
                running = 0
                expected_stage = 4
                for stage, term in enumerate(terms, 1):
                    running += term
                    if stage == 4 or mode == "exact":
                        continue
                    if stages == "ll_only" and stage != 3:
                        continue
                    stop = running >= threshold if mode == "sigma" else running >= threshold * bounds[stage - 1]
                    if stop:
                        expected_stage = stage
                        break
                signed = -running if (activation < 0) != (weight < 0) else running
                row, column = activation + 127, weight + 127
                assert output["product_lut"][row, column] == signed
                assert output["stage_lut"][row, column] == expected_stage
                assert output["nonzero_lut"][row, column] == sum(term != 0 for term in terms[:expected_stage])
                expected_bound = 0 if expected_stage == 4 else bounds[expected_stage - 1]
                assert output["remaining_bound_lut"][row, column] == expected_bound
                assert abs(signed - activation * weight) <= expected_bound
                pairs_checked += 1

    assert np.all(outputs[1]["stage_lut"] == 1)
    assert np.all(outputs[2]["stage_lut"] == 3)
    assert np.all(outputs[5]["stage_lut"] == 4)
    assert np.all(outputs[6]["stage_lut"] == 4)
    # Ties stop: |16*16| reaches 256 immediately; check the documented rule.
    assert outputs[3]["stage_lut"][16 + 127, 16 + 127] == 1
    # Sign restoration avoids the direct two's-complement cancellation error.
    assert outputs[1]["product_lut"][-1 + 127, -1 + 127] == 0
    assert exact["product_lut"][-1 + 127, -1 + 127] == 1
    for output in outputs:
        stats = output["statistics"]
        assert sum(stats["stage_counts"].values()) == 65025
        assert stats["bound_violation_count"] == 0
        assert stats["magnitude_overshoot_count"] == 0
        assert stats["executed_term_slots"] == sum(stats["attempted_term_counts"].values())

    # Stronger ratio thresholds cannot make any operand pair stop earlier.
    ratio_stages = [output["stage_lut"] for config, output in zip(configs, outputs)
                    if config[0] == "ratio" and config[2] == "all"]
    for previous, current in zip(ratio_stages, ratio_stages[1:]):
        assert np.all(current >= previous)

    invalid_count = 0
    for kwargs in ({"mode": "unknown"}, {"stages": "unknown"},
                   {"threshold": -1}, {"threshold": float("nan")},
                   {"threshold": float("inf")}):
        try:
            make_lut(**kwargs)
        except ValueError:
            invalid_count += 1
        else:
            raise AssertionError("invalid LUT configuration was accepted")
    return {"passed": True, "unique_integer_pairs": 65025,
            "exact_reconstruction_pairs": 65025, "scalar_oracle_configurations": len(configs),
            "scalar_oracle_pair_checks": pairs_checked, "remaining_error_bound_checks": bound_checks,
            "invalid_configuration_checks": invalid_count,
            "zero_term_optimization": False,
            "statistics": [output["statistics"] for output in outputs]}


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2, ensure_ascii=False))
