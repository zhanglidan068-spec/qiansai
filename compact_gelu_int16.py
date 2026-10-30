"""Shared 1025 x INT16 GELU LUT with integer linear interpolation.

The runtime nonlinear arithmetic uses integers.  FP32 tensors are transport
boundaries, exactly as in kwt_nonlinear_int16.  This module leaves that baseline
implementation unchanged.  One graph-level ROM serves all configured layers.
"""

import functools
import hashlib
import math
import weakref

import numpy as np

from kwt_nonlinear_int16 import (
    LIMIT, _f, _finish, quantize_input, rescale_int64,
)


LUT_F = 12
GRID_F = 6
DOMAIN = 8
LUT_ENTRIES = 2 * DOMAIN * (1 << GRID_F) + 1
_GRAPH_TABLES = weakref.WeakKeyDictionary()


@functools.lru_cache(maxsize=1)
def _gelu_lut():
    values = np.arange(LUT_ENTRIES, dtype=np.float64) / (1 << GRID_F) - DOMAIN
    outputs = np.fromiter(
        (0.5 * float(x) * (1.0 + math.erf(float(x) / math.sqrt(2.0)))
         for x in values), dtype=np.float64, count=LUT_ENTRIES)
    return np.clip(np.rint(outputs * (1 << LUT_F)), -LIMIT, LIMIT).astype(np.int16)


def _shared_table(tf):
    graph = tf.compat.v1.get_default_graph()
    # tf.cond builds one FuncGraph per branch.  Hoist the ROM to their common
    # outer graph so each branch captures the same physical-table candidate.
    while getattr(graph, "outer_graph", None) is not None:
        graph = graph.outer_graph
    if graph not in _GRAPH_TABLES:
        with graph.as_default(), tf.init_scope():
            _GRAPH_TABLES[graph] = tf.constant(
                _gelu_lut(), dtype=tf.int16, name="shared_compact_gelu_q12_rom")
    return _GRAPH_TABLES[graph]


def _safe_tail_rescale(tf, input_q, input_f, output_f):
    """Preserve output saturation without overflowing unusual F combinations."""
    delta = output_f - input_f
    value = tf.cast(input_q, tf.int64)
    if delta > 0:
        # An input beyond this bound will saturate even after rounding.  Capping
        # it retains _finish's clip flag while bounding the left-shift product.
        threshold = max(1, ((LIMIT + 1) + (1 << delta) - 1) >> delta)
        value = tf.clip_by_value(value, -threshold, threshold)
    return rescale_int64(tf, value, input_f, output_f)


def fixed_gelu(tf, x_fp32, input_f, output_f):
    """INT16 input/output GELU using the single shared 2,050-byte ROM.

    Values within [-8, 8] address adjacent Q12 table nodes at a 1/64 step.
    Interpolation retains its integer numerator until target-F rounding, so
    there is no extra intermediate rounding.  For x >= 8 GELU uses the input
    identity tail; for x <= -8 the output is zero.  Those tail approximations
    differ from exact GELU by less than 6e-15 in this domain boundary.

    The source input may have any F accepted by the existing nonlinear helper.
    LUT quantization error is absolute (about 2**-12) rather than a fixed count
    of output LSBs when a caller chooses output F substantially above 12.
    """
    input_f, output_f = _f(input_f), _f(output_f)
    q, input_clipped = quantize_input(tf, x_fp32, input_f)
    q64 = tf.cast(q, tf.int64)
    coordinate_f = max(input_f, GRID_F)
    coordinate = rescale_int64(tf, q64, input_f, coordinate_f)
    extent = DOMAIN * (1 << coordinate_f)
    denominator = 1 << (coordinate_f - GRID_F)
    position = coordinate + extent
    lower_index = tf.cast(tf.clip_by_value(
        tf.math.floordiv(position, denominator), 0, LUT_ENTRIES - 2), tf.int32)
    fraction = tf.math.floormod(position, denominator)
    table = _shared_table(tf)
    lower = tf.cast(tf.gather(table, lower_index), tf.int64)
    upper = tf.cast(tf.gather(table, lower_index + 1), tf.int64)
    numerator = lower * denominator + (upper - lower) * fraction
    interpolation_f = LUT_F + coordinate_f - GRID_F
    interpolated = rescale_int64(tf, numerator, interpolation_f, output_f)
    tail = _safe_tail_rescale(tf, q64, input_f, output_f)
    output_wide = tf.where(coordinate >= extent, tail, interpolated)
    output_wide = tf.where(coordinate <= -extent, tf.zeros_like(output_wide),
                           output_wide)
    return _finish(tf, q, input_clipped, output_wide, output_f)


def implementation_metadata():
    table = _gelu_lut()
    return {
        "algorithm": "shared compact GELU LUT with integer linear interpolation",
        "lut_entries": LUT_ENTRIES,
        "lut_dtype": "int16",
        "lut_fractional_bits": LUT_F,
        "lut_domain": [-DOMAIN, DOMAIN],
        "lut_step": 1.0 / (1 << GRID_F),
        "lut_payload_bytes": int(table.nbytes),
        "lut_last_node_units": int(table[-1]),
        "lut_sha256": hashlib.sha256(table.tobytes()).hexdigest(),
        "graph_table_sharing": "one TensorFlow Const per outer graph, captured by branches",
        "offline_function": "0.5*x*(1+erf(x/sqrt(2))) using Python math.erf",
        "runtime": "INT16 gather; INT64 interpolation; nearest-even rescale",
        "outside_domain": {"x<=-8": "zero", "x>=8": "input identity"},
        "rounding": "round-to-nearest, ties-to-even",
        "saturation": [-LIMIT, LIMIT],
        "note": "ROM payload is not a measured ASIC area; macro/control/interpolator costs remain",
    }


def export_lut_tables(nonlinear_config=None):
    """NPZ-ready copy of the exact single shared table; config is compatible."""
    del nonlinear_config
    return {"gelu_shared_compact_q12": _gelu_lut().copy()}
