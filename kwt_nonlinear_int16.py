"""Integer/LUT nonlinear operators for the W8A8/N16 KWT experiment.

Only the operator boundaries convert FP32 transport tensors to/from fixed point.
Softmax, GELU and LayerNorm themselves use integer TensorFlow operations.  LUTs
are generated once, offline, with Python/NumPy floating point.  All LUT entries,
operator inputs/outputs and LayerNorm affine parameters are signed INT16.
Accumulators and intermediate expressions deliberately use INT64.

The symmetric range is [-32767, 32767]; -32768 is unused.  Rounding is nearest,
ties to even, at every integer division/rescale and at the FP32 boundary.
"""

import functools
import math
import operator

import numpy as np


LIMIT = 32767
SOFTMAX_GRID_F = 8
SOFTMAX_DOMAIN = 32
SOFTMAX_EXP_F = 15
LN_VARIANCE_EXTRA_F = 8
LN_NORMALIZED_F = 20


def _f(value):
    if isinstance(value, bool):
        raise TypeError("A fractional-bit count must be an integer")
    value = operator.index(value)
    if not -24 <= value <= 24:
        raise ValueError("Nonlinear fractional bits must be in [-24, 24]")
    return value


def _clip_count(tf, clipped):
    rank = clipped.shape.rank
    if rank is None or rank < 2:
        raise ValueError("Expected a nonlinear activation with rank >= 2")
    return tf.reduce_sum(tf.cast(clipped, tf.int64), axis=list(range(1, rank)))


def quantize_input(tf, x, fractional_bits):
    """Quantize an FP32 transport tensor, returning INT16 and per-sample clips."""
    fractional_bits = _f(fractional_bits)
    rounded = tf.math.rint(
        tf.cast(x, tf.float32) * tf.constant(math.ldexp(1.0, fractional_bits), tf.float32))
    clipped = tf.logical_or(rounded < -float(LIMIT), rounded > float(LIMIT))
    q = tf.cast(tf.clip_by_value(rounded, -float(LIMIT), float(LIMIT)), tf.int16)
    return q, _clip_count(tf, clipped)


def round_divide_even(tf, numerator, denominator):
    """Signed INT64 numerator / positive denominator, nearest ties to even.

    Used numerator magnitudes stay far below INT64_MIN, so abs() is safe.
    Denominators may broadcast and must be strictly positive.
    """
    numerator = tf.cast(numerator, tf.int64)
    denominator = tf.cast(denominator, tf.int64)
    magnitude = tf.abs(numerator)
    quotient = tf.math.floordiv(magnitude, denominator)
    remainder = tf.math.floormod(magnitude, denominator)
    twice = remainder * tf.constant(2, tf.int64)
    odd = tf.not_equal(tf.bitwise.bitwise_and(quotient, tf.constant(1, tf.int64)), 0)
    increment = tf.logical_or(twice > denominator,
                              tf.logical_and(tf.equal(twice, denominator), odd))
    rounded = quotient + tf.cast(increment, tf.int64)
    return tf.where(numerator < 0, -rounded, rounded)


def rescale_int64(tf, value, source_f, target_f):
    """Change a fixed-point integer scale without floating arithmetic."""
    source_f = operator.index(source_f)
    target_f = operator.index(target_f)
    delta = target_f - source_f
    value = tf.cast(value, tf.int64)
    if abs(delta) > 60:
        raise ValueError("Fixed point rescale exceeds the supported INT64 shift")
    if delta >= 0:
        return value * tf.constant(1 << delta, tf.int64)
    return round_divide_even(tf, value, tf.constant(1 << (-delta), tf.int64))


def integer_sqrt(tf, value):
    """Exact floor(sqrt(n)) for nonnegative INT64 n, using 32 restoring steps.

    The loop uses only comparisons, additions, subtractions and bit shifts;
    it has no floating point sqrt, reciprocal sqrt or division.
    """
    value = tf.cast(value, tf.int64)
    result = tf.zeros_like(value, tf.int64)
    bit = tf.constant(1 << 62, tf.int64)

    def condition(remainder, root, current_bit):
        del remainder, root
        return current_bit > 0

    def step(remainder, root, current_bit):
        trial = root + current_bit
        accept = remainder >= trial
        remainder = tf.where(accept, remainder - trial, remainder)
        half = tf.bitwise.right_shift(root, tf.constant(1, tf.int64))
        root = tf.where(accept, half + current_bit, half)
        return remainder, root, tf.bitwise.right_shift(current_bit, tf.constant(2, tf.int64))

    _, result, _ = tf.while_loop(condition, step, (value, result, bit),
                                 parallel_iterations=1,
                                 maximum_iterations=32)
    return result


def _finish(tf, input_q, input_clipped, output_wide, output_f, diagnostics=None):
    output_f = _f(output_f)
    output_wide = tf.cast(output_wide, tf.int64)
    output_clipped = _clip_count(tf, tf.logical_or(output_wide < -LIMIT,
                                                  output_wide > LIMIT))
    output_q = tf.cast(tf.clip_by_value(output_wide, -LIMIT, LIMIT), tf.int16)
    output_dequant = tf.cast(output_q, tf.float32) * tf.constant(
        math.ldexp(1.0, -output_f), tf.float32)
    return {"input_int16": input_q,
            "output_int16": output_q,
            "output_dequant": output_dequant,
            "input_clipped": input_clipped,
            "output_clipped": output_clipped,
            "diagnostics": diagnostics or {}}


@functools.lru_cache(maxsize=1)
def _softmax_lut():
    # exp(0) cannot represent +1 in signed Q15, hence its value is 32767.
    distances = np.arange(SOFTMAX_DOMAIN * (1 << SOFTMAX_GRID_F) + 1,
                          dtype=np.float64) / float(1 << SOFTMAX_GRID_F)
    return np.clip(np.rint(np.exp(-distances) * (1 << SOFTMAX_EXP_F)),
                   0, LIMIT).astype(np.int16)


def fixed_softmax(tf, x_fp32, input_f, output_f=15):
    """Stable integer Softmax over the last axis, with an INT16 exp LUT.

    Input is quantized to its calibrated INT16 scale first.  Differences are
    computed in INT64, then mapped to a 1/256 grid on [-32, 0].  Values farther
    below the maximum map to the table's zero tail.  The INT16 Q15 exp values
    are summed and normalized by rounded INT64 division.  Output is signed
    INT16 Q15: 1.0 is saturated to 32767/32768.  No row renormalization follows.
    """
    input_f, output_f = _f(input_f), _f(output_f)
    if output_f != 15:
        raise ValueError("This experiment requires Softmax INT16 output F=15")
    q, input_clipped = quantize_input(tf, x_fp32, input_f)
    q64 = tf.cast(q, tf.int64)
    distance = tf.reduce_max(q64, axis=-1, keepdims=True) - q64
    lut_index = rescale_int64(tf, distance, input_f, SOFTMAX_GRID_F)
    max_index = SOFTMAX_DOMAIN * (1 << SOFTMAX_GRID_F)
    lut_index = tf.cast(tf.clip_by_value(lut_index, 0, max_index), tf.int32)
    exp_q15 = tf.gather(tf.constant(_softmax_lut(), dtype=tf.int16), lut_index)
    exp_wide = tf.cast(exp_q15, tf.int64)
    denominator = tf.reduce_sum(exp_wide, axis=-1, keepdims=True)
    # At least one element has distance zero, exp=32767, so denominator > 0.
    output_wide = round_divide_even(tf, exp_wide * (1 << output_f), denominator)
    return _finish(tf, q, input_clipped, output_wide, output_f,
                   {"exp_int16": exp_q15, "denominator_int64": denominator,
                    "row_sum_int16_units": tf.reduce_sum(
                        tf.clip_by_value(output_wide, 0, LIMIT), axis=-1)})


@functools.lru_cache(maxsize=64)
def _gelu_lut(input_f, output_f):
    values = np.arange(-LIMIT, LIMIT + 1, dtype=np.float64)
    real = np.ldexp(values, -input_f)
    erf_values = np.fromiter((math.erf(float(item) / math.sqrt(2.0)) for item in real),
                             dtype=np.float64, count=real.size)
    result = 0.5 * real * (1.0 + erf_values)
    return np.clip(np.rint(np.ldexp(result, output_f)), -LIMIT, LIMIT).astype(np.int16)


def fixed_gelu(tf, x_fp32, input_f, output_f):
    """INT16 input indexes a full 65535-entry exact-erf GELU offline LUT.

    At runtime there is only boundary quantization and integer gather.  The
    table contains saturated INT16 output values at the calibrated output F.
    """
    input_f, output_f = _f(input_f), _f(output_f)
    q, input_clipped = quantize_input(tf, x_fp32, input_f)
    table = tf.constant(_gelu_lut(input_f, output_f), dtype=tf.int16)
    index = tf.cast(q, tf.int32) + LIMIT
    output_q = tf.gather(table, index)
    # The table already saturates.  Count which selected entries were clipped
    # using a second offline bit table, so runtime still uses integer gather.
    # Cache the output saturation lookup with the same offline LUT parameters.
    clip_table = tf.constant(_gelu_clip_lut(input_f, output_f), dtype=tf.bool)
    result = _finish(tf, q, input_clipped, tf.cast(output_q, tf.int64), output_f)
    result["output_clipped"] = _clip_count(tf, tf.gather(clip_table, index))
    return result


@functools.lru_cache(maxsize=64)
def _gelu_clip_lut(input_f, output_f):
    values = np.arange(-LIMIT, LIMIT + 1, dtype=np.float64)
    real = np.ldexp(values, -input_f)
    erf_values = np.fromiter((math.erf(float(item) / math.sqrt(2.0)) for item in real),
                             dtype=np.float64, count=real.size)
    units = np.rint(np.ldexp(0.5 * real * (1.0 + erf_values), output_f))
    return np.logical_or(units < -LIMIT, units > LIMIT)


def fixed_layer_norm(tf, x_fp32, input_f, output_f, gamma, beta,
                     epsilon=1e-6, gamma_f=14, beta_f=16):
    """Integer LayerNorm for KWT's last dimension N=64, epsilon=1e-6.

    Let c=N*q-sum(q), V=sum(c*c).  The normalized value is exactly
    c*sqrt(N)/sqrt(V+epsilon*2**(2*input_f)*N**3) before fixed point rounding.
    The radicand has 8 extra fractional bits; floor integer sqrt consequently
    has 4 extra bits.  The normalized intermediate uses Q20 in INT64.
    Gamma and beta are separately quantized INT16 Q14 and Q16, and the affine
    expression is accumulated at Q34 before one final rounded rescale.

    Current official weights fit these parameter formats: |gamma|<1.461,
    |beta|<0.254.  The helper returns parameter clipping diagnostics as well.
    """
    input_f, output_f = _f(input_f), _f(output_f)
    gamma_f, beta_f = _f(gamma_f), _f(beta_f)
    if x_fp32.shape.as_list()[-1] != 64:
        raise ValueError("Integer LayerNorm currently requires last dimension 64")
    if not (epsilon > 0 and math.isfinite(epsilon)):
        raise ValueError("LayerNorm epsilon must be finite and positive")
    if gamma_f + LN_NORMALIZED_F < beta_f:
        raise ValueError("Beta precision exceeds the affine accumulator precision")
    q, input_clipped = quantize_input(tf, x_fp32, input_f)
    q64 = tf.cast(q, tf.int64)
    n = 64
    centered = n * q64 - tf.reduce_sum(q64, axis=-1, keepdims=True)
    variance_numerator = tf.reduce_sum(centered * centered, axis=-1, keepdims=True)
    extra_scale = 1 << LN_VARIANCE_EXTRA_F
    epsilon_units = max(1, int(round(math.ldexp(
        float(epsilon) * (n ** 3), 2 * input_f + LN_VARIANCE_EXTRA_F))))
    max_variance = (n ** 3) * LIMIT * LIMIT * extra_scale
    if epsilon_units + max_variance >= (1 << 63):
        raise ValueError("LayerNorm calibrated input F exceeds safe INT64 radicand range")
    radicand = variance_numerator * extra_scale + tf.constant(epsilon_units, tf.int64)
    root = integer_sqrt(tf, radicand)
    # sqrt(64)=8, sqrt(2**8)=16: both constants are exact integers.
    numerator = centered * tf.constant(8 * 16 * (1 << LN_NORMALIZED_F), tf.int64)
    normalized = round_divide_even(tf, numerator, root)

    def affine_parameter(value, fractional_bits):
        units = tf.math.rint(tf.cast(value, tf.float32) * tf.constant(
            math.ldexp(1.0, fractional_bits), tf.float32))
        clips = tf.reduce_sum(tf.cast(tf.logical_or(units < -float(LIMIT),
                                                    units > float(LIMIT)), tf.int64))
        integer = tf.cast(tf.clip_by_value(units, -float(LIMIT), float(LIMIT)), tf.int16)
        return integer, clips

    gamma_q, gamma_clips = affine_parameter(gamma, gamma_f)
    beta_q, beta_clips = affine_parameter(beta, beta_f)
    affine_f = LN_NORMALIZED_F + gamma_f
    affine = normalized * tf.cast(gamma_q, tf.int64) + rescale_int64(
        tf, beta_q, beta_f, affine_f)
    output_wide = rescale_int64(tf, affine, affine_f, output_f)
    return _finish(tf, q, input_clipped, output_wide, output_f,
                   {"gamma_int16": gamma_q, "beta_int16": beta_q,
                    "gamma_clipped": gamma_clips, "beta_clipped": beta_clips,
                    "epsilon_radicand_units": tf.constant(epsilon_units, tf.int64),
                    "normalized_q20_int64": normalized})


def implementation_metadata():
    """JSON-serializable arithmetic description for frozen experiment reports."""
    return {"input_output_dtype": "int16", "symmetric_range": [-LIMIT, LIMIT],
            "rounding": "round-to-nearest, ties-to-even",
            "wide_intermediate_dtype": "int64",
            "softmax": {"input_f": "training-calibrated per layer", "output_f": 15,
                        "exp_lut_dtype": "int16", "exp_lut_f": SOFTMAX_EXP_F,
                        "lut_domain": [-SOFTMAX_DOMAIN, 0],
                        "lut_step": 1.0 / (1 << SOFTMAX_GRID_F),
                        "lut_entries": SOFTMAX_DOMAIN * (1 << SOFTMAX_GRID_F) + 1,
                        "exp_zero_units": LIMIT,
                        "normalization": "INT64 sum and nearest-even integer division",
                        "post_normalization": "no renormalization after INT16 or INT8 rounding"},
            "gelu": {"lut_entries": 2 * LIMIT + 1, "lut_dtype": "int16",
                     "offline_function": "0.5*x*(1+erf(x/sqrt(2)))",
                     "runtime": "integer LUT gather"},
            "layer_norm": {"width": 64, "epsilon": 1e-6,
                           "gamma_f": 14, "beta_f": 16,
                           "variance_extra_fractional_bits": LN_VARIANCE_EXTRA_F,
                           "normalized_intermediate_f": LN_NORMALIZED_F,
                           "sqrt": "exact floor integer sqrt, 32 restoring bit-pair steps",
                           "affine_accumulator_f": LN_NORMALIZED_F + 14},
            "floating_boundary": "FP32 transport quantization/dequantization only; offline LUT generation is floating point",
            "runtime_excluded_ops": ["Exp", "Softmax", "Erf", "Sqrt", "Rsqrt"]}


def export_lut_tables(nonlinear_config):
    """Export copies of the exact offline tables used by a configured graph.

    Keys are suitable for ``numpy.savez``: slash-separated nonlinear prefixes
    become double-underscore names.  GELU tables retain their per-operator
    input/output F configuration even when several operators share a table.
    The boolean saturation table records entries clipped during generation.
    """
    tables = {"softmax_exp_q15": _softmax_lut().copy()}
    for prefix, config in sorted(nonlinear_config.items()):
        if str(prefix).split("/")[-1] != "gelu":
            continue
        input_f = _f(config["input_f"])
        output_f = _f(config["output_f"])
        key = str(prefix).replace("/", "__")
        tables[key + "__int16"] = _gelu_lut(input_f, output_f).copy()
        tables[key + "__clipped"] = _gelu_clip_lut(input_f, output_f).copy()
    return tables
