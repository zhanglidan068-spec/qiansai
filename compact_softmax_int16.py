"""Bit-identical INT16 Softmax using a shared 4096-word exponential ROM.

The baseline exp(-distance) table has 8193 signed Q15 words.  Every entry from
index 4096 onwards is zero after its original nearest-even rounding.  This
implementation stores the first 4096 words and handles the zero tail with an
integer comparison.  Quantization, normalization and output remain identical
to kwt_nonlinear_int16.fixed_softmax; the baseline source is unchanged.
"""

import functools
import hashlib
import weakref

import numpy as np

from kwt_nonlinear_int16 import (
    LIMIT, SOFTMAX_GRID_F, SOFTMAX_DOMAIN, SOFTMAX_EXP_F,
    _f, _finish, _softmax_lut, quantize_input, rescale_int64, round_divide_even,
)


LUT_ENTRIES = 4096
_GRAPH_TABLES = weakref.WeakKeyDictionary()


@functools.lru_cache(maxsize=1)
def _compact_exp_lut():
    baseline = _softmax_lut()
    if not np.all(baseline[LUT_ENTRIES:] == 0):
        raise ValueError("Baseline Softmax table zero-tail invariant changed")
    return baseline[:LUT_ENTRIES].copy()


def _shared_table(tf):
    graph = tf.compat.v1.get_default_graph()
    while getattr(graph, "outer_graph", None) is not None:
        graph = graph.outer_graph
    if graph not in _GRAPH_TABLES:
        with graph.as_default(), tf.init_scope():
            _GRAPH_TABLES[graph] = tf.constant(
                _compact_exp_lut(), dtype=tf.int16,
                name="shared_compact_softmax_exp_q15_rom")
    return _GRAPH_TABLES[graph]


def fixed_softmax(tf, x_fp32, input_f, output_f=15):
    """Same five result fields and diagnostics as the baseline INT16 Softmax."""
    input_f, output_f = _f(input_f), _f(output_f)
    if output_f != 15:
        raise ValueError("This experiment requires Softmax INT16 output F=15")
    q, input_clipped = quantize_input(tf, x_fp32, input_f)
    q64 = tf.cast(q, tf.int64)
    distance = tf.reduce_max(q64, axis=-1, keepdims=True) - q64
    raw_index = rescale_int64(tf, distance, input_f, SOFTMAX_GRID_F)
    safe_index = tf.cast(tf.clip_by_value(raw_index, 0, LUT_ENTRIES - 1), tf.int32)
    gathered = tf.gather(_shared_table(tf), safe_index)
    exp_q15 = tf.where(raw_index < LUT_ENTRIES, gathered, tf.zeros_like(gathered))
    exp_wide = tf.cast(exp_q15, tf.int64)
    denominator = tf.reduce_sum(exp_wide, axis=-1, keepdims=True)
    output_wide = round_divide_even(tf, exp_wide * (1 << output_f), denominator)
    return _finish(tf, q, input_clipped, output_wide, output_f,
                   {"exp_int16": exp_q15,
                    "denominator_int64": denominator,
                    "row_sum_int16_units": tf.reduce_sum(
                        tf.clip_by_value(output_wide, 0, LIMIT), axis=-1)})


def implementation_metadata():
    table = _compact_exp_lut()
    return {
        "algorithm": "baseline integer Softmax with implicit zero exponential tail",
        "lut_entries": LUT_ENTRIES,
        "lut_dtype": "int16",
        "lut_fractional_bits": SOFTMAX_EXP_F,
        "lut_grid_fractional_bits": SOFTMAX_GRID_F,
        "lut_step": 1.0 / (1 << SOFTMAX_GRID_F),
        "stored_distance_domain": [0, (LUT_ENTRIES - 1) / float(1 << SOFTMAX_GRID_F)],
        "implicit_zero_distance_start": LUT_ENTRIES / float(1 << SOFTMAX_GRID_F),
        "baseline_distance_domain": [0, SOFTMAX_DOMAIN],
        "baseline_lut_entries": SOFTMAX_DOMAIN * (1 << SOFTMAX_GRID_F) + 1,
        "lut_payload_bytes": int(table.nbytes),
        "lut_sha256": hashlib.sha256(table.tobytes()).hexdigest(),
        "graph_table_sharing": "one TensorFlow Const per outer graph, captured by branches",
        "arithmetic_unchanged_from_baseline": True,
        "normalization": "INT64 sum and nearest-even integer division to signed Q15",
        "post_normalization": "no row renormalization",
        "note": "ROM payload is not a measured ASIC area; macro/control/normalizer costs remain",
    }


def export_lut_tables(nonlinear_config=None):
    del nonlinear_config
    return {"softmax_shared_compact_exp_q15": _compact_exp_lut().copy()}
