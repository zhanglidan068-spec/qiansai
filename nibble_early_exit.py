"""Progressive signed INT8 multiplication through unsigned four-bit terms.

The quantizer's symmetric INT8 domain is [-127, 127].  Table axes are
``[activation + 127, weight + 127]``.  First take magnitudes, then evaluate
HH, HL, LH, LL in that order, and restore the product sign last::

    HH = 256 * wH * aH
    HL =  16 * wH * aL
    LH =  16 * wL * aH
    LL =       wL * aL

All comparisons use these aligned, nonnegative integer product magnitudes.
``sigma`` stops when S >= threshold.  ``ratio`` stops when S >= threshold*B,
where B is a conservative bound on the omitted magnitude.  A threshold tie
therefore stops.  ``ll_only`` permits stopping only after HH+HL+LH; ``all``
permits all three early-exit points.  ``exact`` always executes four terms.

There is deliberately no zero-term optimization: stage_lut is the number of
four-bit multiplication slots executed (1..4), including zero products.
nonzero_lut separately counts the executed terms whose product is nonzero.
Static statistics describe the uniform Cartesian INT8 domain, not a model's
observed operand distribution, energy, or physical accelerator speedup.

Only NumPy and the Python standard library are required.  No data is loaded
from disk and no outputs are written by this module.
"""

import math

import numpy as np


INT8_LIMIT = 127
LUT_SIDE = 255
LUT_OFFSET = 127


def _histogram(values):
    return {str(stage): int(np.count_nonzero(values == stage))
            for stage in range(1, 5)}


def make_lut(mode="exact", threshold=0.0, stages="all"):
    """Return deterministic product/operation-count LUTs and static statistics.

    Args:
        mode: ``exact``, ``sigma`` (absolute magnitude), or ``ratio``.
        threshold: finite nonnegative sigma; ignored by exact mode.  In ratio
            mode larger thresholds require a smaller omitted-error fraction.
        stages: ``ll_only`` or ``all`` (ignored by exact mode).

    Returns:
        A dict containing:
          product_lut: int32 [255, 255], signed approximate products.
          stage_lut: uint8 [255, 255], executed term count (including zeros).
          nonzero_lut: uint8 [255, 255], executed nonzero term count.
          remaining_bound_lut: uint16 [255, 255], bound at the chosen exit.
          statistics: JSON-serializable configuration/error/count metadata.

    Integer input containers used elsewhere (e.g. TensorFlow int64) do not
    change these LUT indices or the effective symmetric INT8 input domain.
    """
    if mode not in ("exact", "sigma", "ratio"):
        raise ValueError("mode must be exact, sigma, or ratio")
    if stages not in ("ll_only", "all"):
        raise ValueError("stages must be ll_only or all")
    threshold = float(threshold)
    if not math.isfinite(threshold) or threshold < 0:
        raise ValueError("threshold must be finite and nonnegative")

    domain = np.arange(-INT8_LIMIT, INT8_LIMIT + 1, dtype=np.int32)
    activation, weight = np.broadcast_arrays(domain[:, None], domain[None, :])
    a = np.abs(activation)
    w = np.abs(weight)
    ah, al = a >> 4, a & 15
    wh, wl = w >> 4, w & 15
    terms = (256 * wh * ah, 16 * wh * al, 16 * wl * ah, wl * al)
    partial = (terms[0], terms[0] + terms[1],
               terms[0] + terms[1] + terms[2])

    # Bounds require shifts/adds/min rather than the omitted 4x4 products.
    # B1 >= 16*wH*aL + wL*a; B2 >= wL*a; B3 >= wL*aL.
    bounds = (np.minimum(240 * wh + 15 * a, 240 * ah + 15 * w),
              15 * a, np.minimum(15 * wl, 15 * al))

    stop_stage = np.full((LUT_SIDE, LUT_SIDE), 4, dtype=np.uint8)
    if mode != "exact":
        allowed = (2,) if stages == "ll_only" else (0, 1, 2)
        for index in allowed:
            if mode == "sigma":
                eligible = partial[index] >= threshold
            else:
                eligible = partial[index].astype(np.float64) >= threshold * bounds[index]
            stop_stage[(stop_stage == 4) & eligible] = index + 1

    magnitude = np.zeros((LUT_SIDE, LUT_SIDE), dtype=np.int32)
    nonzero = np.zeros((LUT_SIDE, LUT_SIDE), dtype=np.uint8)
    remaining_bound = np.zeros((LUT_SIDE, LUT_SIDE), dtype=np.uint16)
    for index, term in enumerate(terms):
        executed = stop_stage >= index + 1
        magnitude += np.where(executed, term, 0)
        nonzero += (executed & (term != 0)).astype(np.uint8)
        if index < 3:
            remaining_bound[stop_stage == index + 1] = bounds[index][stop_stage == index + 1]
    negative = (activation < 0) != (weight < 0)
    product = np.where(negative, -magnitude, magnitude).astype(np.int32)
    exact = activation * weight
    absolute_error = np.abs(product - exact)
    full_nonzero = sum((term != 0).astype(np.uint8) for term in terms)
    total_pairs = int(product.size)
    attempted = int(stop_stage.astype(np.int64).sum())
    executed_nonzero = int(nonzero.astype(np.int64).sum())
    full_nonzero_count = int(full_nonzero.astype(np.int64).sum())
    exact_nonzero = exact != 0
    relative_errors = absolute_error[exact_nonzero].astype(np.float64) / np.abs(exact[exact_nonzero])
    stage_counts = _histogram(stop_stage)
    statistics = {
        "mode": mode,
        "threshold": threshold,
        "stages": stages,
        "domain": [-INT8_LIMIT, INT8_LIMIT],
        "axis_order": ["activation", "weight"],
        "offset": LUT_OFFSET,
        "partial_product_order": ["HH", "HL", "LH", "LL"],
        "threshold_comparison": ">= (ties stop)",
        "zero_term_optimization": False,
        "pair_count": total_pairs,
        "stage_counts": stage_counts,
        "attempted_term_counts": {
            name: int(np.count_nonzero(stop_stage >= index + 1))
            for index, name in enumerate(("HH", "HL", "LH", "LL"))},
        "full_term_slots": total_pairs * 4,
        "executed_term_slots": attempted,
        "mean_terms_per_pair": attempted / float(total_pairs),
        "saved_term_fraction": 1.0 - attempted / float(total_pairs * 4),
        "full_nonzero_terms": full_nonzero_count,
        "executed_nonzero_terms": executed_nonzero,
        "saved_nonzero_term_fraction": 1.0 - executed_nonzero / float(full_nonzero_count),
        "changed_pair_count": int(np.count_nonzero(absolute_error)),
        "max_abs_error": int(absolute_error.max()),
        "mean_abs_error": float(absolute_error.mean()),
        "root_mean_squared_error": float(np.sqrt(np.mean(absolute_error.astype(np.float64) ** 2))),
        "max_relative_error_nonzero_products": float(relative_errors.max()),
        "mean_relative_error_nonzero_products": float(relative_errors.mean()),
        "bound_violation_count": int(np.count_nonzero(absolute_error > remaining_bound)),
        "magnitude_overshoot_count": int(np.count_nonzero(np.abs(product) > np.abs(exact))),
        "distribution_note": "Uniform Cartesian integer-domain statistics; model-weighted counts must be measured separately.",
        "hardware_note": "Term-slot reductions exclude comparison/control overhead and do not establish PE-array cycle speedup.",
    }
    return {
        "product_lut": product,
        "stage_lut": stop_stage,
        "nonzero_lut": nonzero,
        "remaining_bound_lut": remaining_bound,
        "statistics": statistics,
    }


def lookup_products(activations, weights, product_lut):
    """NumPy reference lookup with domain checks and normal broadcasting."""
    activations = np.asarray(activations)
    weights = np.asarray(weights)
    if not np.issubdtype(activations.dtype, np.integer) or not np.issubdtype(weights.dtype, np.integer):
        raise TypeError("LUT operands must be integer arrays")
    if (np.any((activations < -INT8_LIMIT) | (activations > INT8_LIMIT))
            or np.any((weights < -INT8_LIMIT) | (weights > INT8_LIMIT))):
        raise ValueError("LUT operands must be in [-127, 127]")
    table = np.asarray(product_lut)
    if table.shape != (LUT_SIDE, LUT_SIDE):
        raise ValueError("product_lut must have shape [255, 255]")
    return table[activations.astype(np.int64) + LUT_OFFSET,
                 weights.astype(np.int64) + LUT_OFFSET]
