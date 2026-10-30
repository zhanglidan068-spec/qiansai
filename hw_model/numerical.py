"""Small, explicit numerical references, separate from performance simulation.

There are no checkpoint/dataset accuracy claims. Quantized GEMM follows the
selected issue mapping: reductions inside an issue use exact integers, and the
finite accumulator applies overflow once per issue. Bias is added exactly once
after the final reduction. A narrower/truncating adder tree requires a different
numerical contract and is not implied by this reference.
"""

from fractions import Fraction
import math
from typing import List, Optional, Sequence, Union

from .compute import Shape, iter_issues


Number = Union[int, float, str, Fraction]


def _int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("{} must be an integer".format(name))
    return value


def _bits(bits: int) -> int:
    if _int(bits, "bits") < 1:
        raise ValueError("bits must be at least 1")
    return bits


def _fraction(value: Number) -> Fraction:
    if isinstance(value, bool):
        raise ValueError("boolean values are not numerical scales")
    if isinstance(value, Fraction):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("numerical values must be finite")
        # Use the stated decimal scale instead of its incidental binary encoding.
        return Fraction(str(value))
    try:
        return Fraction(value)
    except (TypeError, ValueError, ZeroDivisionError) as exc:
        raise ValueError("invalid exact numerical value {!r}".format(value)) from exc


def signed_overflow(value: int, bits: int, mode: str = "saturate") -> int:
    """Apply signed two's-complement saturation or modular wrap explicitly."""
    _int(value, "value")
    _bits(bits)
    half = 1 << (bits - 1)
    if mode == "saturate":
        return max(-half, min(half - 1, value))
    if mode == "wrap":
        return (value + half) % (1 << bits) - half
    raise ValueError("overflow mode must be 'saturate' or 'wrap'")


def round_even(value: Number) -> int:
    """Round an exact rational to the nearest integer, with ties to even."""
    exact = _fraction(value)
    quotient, remainder = divmod(exact.numerator, exact.denominator)
    twice = 2 * remainder
    if twice > exact.denominator or (twice == exact.denominator and quotient % 2):
        quotient += 1
    return quotient


def requantize(value: int, multiplier: Number = 1, bits: int = 16,
               overflow: str = "saturate") -> int:
    """Exact multiplier, nearest-even rounding, then signed output overflow."""
    _int(value, "value")
    scale = _fraction(multiplier)
    if scale < 0:
        raise ValueError("requantization multiplier must be nonnegative")
    return signed_overflow(round_even(value * scale), bits, overflow)


def _matrix(values: Sequence[Sequence[int]], name: str) -> List[List[int]]:
    matrix = [list(row) for row in values]
    if not matrix or not matrix[0]:
        raise ValueError("{} must be a nonempty matrix".format(name))
    width = len(matrix[0])
    for row in matrix:
        if len(row) != width:
            raise ValueError("{} must be rectangular".format(name))
        for value in row:
            _int(value, name)
    return matrix


def quantized_gemm(a: Sequence[Sequence[int]], b: Sequence[Sequence[int]],
                   architecture: str = "array_9x16",
                   bias: Optional[Sequence[int]] = None,
                   accumulator_bits: int = 32, overflow: str = "saturate",
                   output_bits: Optional[int] = None,
                   output_multiplier: Number = 1) -> List[List[int]]:
    """Evaluate the actual issue mapping with finite, per-issue accumulation.

    Input values are already quantized integers; input bit widths and zero points
    are not inferred. With output_bits=None, return the signed accumulators.
    Otherwise requantize after adding bias once, in accumulator units.
    """
    left, right = _matrix(a, "a"), _matrix(b, "b")
    if len(left[0]) != len(right):
        raise ValueError("GEMM reduction dimensions do not match")
    _bits(accumulator_bits)
    signed_overflow(0, accumulator_bits, overflow)
    shape = Shape(len(left), len(right[0]), len(right))
    if bias is not None:
        bias = list(bias)
        if len(bias) != shape.n:
            raise ValueError("bias length must match output columns")
        for value in bias:
            _int(value, "bias")
    if output_bits is None and _fraction(output_multiplier) != 1:
        raise ValueError("output_multiplier requires output_bits")
    if output_bits is not None:
        _bits(output_bits)
        requantize(0, output_multiplier, output_bits, overflow)
    result = [[0] * shape.n for _ in range(shape.m)]
    for issue in iter_issues(shape, architecture):
        for row in issue.rows:
            for column in issue.columns:
                partial = sum(left[row][k] * right[k][column] for k in issue.reduction)
                result[row][column] = signed_overflow(
                    result[row][column] + partial, accumulator_bits, overflow)
    for row in range(shape.m):
        for column in range(shape.n):
            value = result[row][column]
            if bias is not None:
                value = signed_overflow(value + bias[column], accumulator_bits, overflow)
            if output_bits is not None:
                value = requantize(value, output_multiplier, output_bits, overflow)
            result[row][column] = value
    return result


def _scales(value: Union[Number, Sequence[Number]], size: int, name: str) -> List[Fraction]:
    if isinstance(value, (list, tuple)):
        if len(value) != size:
            raise ValueError("{} must have one scale per channel".format(name))
        scales = [_fraction(item) for item in value]
    else:
        scales = [_fraction(value)] * size
    if any(scale <= 0 for scale in scales):
        raise ValueError("{} must be positive".format(name))
    return scales


def residual_add(main: Sequence[int], residual: Sequence[int],
                 main_scale: Union[Number, Sequence[Number]] = 1,
                 residual_scale: Union[Number, Sequence[Number]] = 1,
                 output_scale: Number = 1, bits: int = 16,
                 overflow: str = "saturate") -> List[int]:
    """Align each branch to one common LN scale, round each, add, then clip.

    Alignment temporaries are exact/unbounded; only the sum is constrained to
    bits. Per-channel source scales are supported, but output_scale is common
    to the whole row, so downstream LN need not average incompatible integers.
    """
    if len(main) != len(residual) or not main:
        raise ValueError("residual branches must have the same nonzero length")
    common = _fraction(output_scale)
    if common <= 0:
        raise ValueError("output_scale must be positive")
    left = _scales(main_scale, len(main), "main_scale")
    right = _scales(residual_scale, len(main), "residual_scale")
    result = []
    for channel, (value, skip) in enumerate(zip(main, residual)):
        _int(value, "main value")
        _int(skip, "residual value")
        aligned = round_even(value * left[channel] / common)
        aligned_skip = round_even(skip * right[channel] / common)
        result.append(signed_overflow(aligned + aligned_skip, bits, overflow))
    return result


def layer_norm_float(values: Sequence[float],
                     gamma: Optional[Sequence[float]] = None,
                     beta: Optional[Sequence[float]] = None,
                     epsilon: float = 1e-6) -> List[float]:
    """Two-pass real-scale LN formula in Python floats, not fixed-point LN.

    Dequantize into real units before calling. epsilon remains in squared real
    units; an integer epsilon of one is not substituted for the KWT value.
    Population variance divides by N, including for a one-channel test vector.
    """
    row = [float(value) for value in values]
    if not row or not all(math.isfinite(value) for value in row):
        raise ValueError("LN values must be a nonempty finite row")
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")
    count = len(row)
    gain = [1.0] * count if gamma is None else [float(value) for value in gamma]
    offset = [0.0] * count if beta is None else [float(value) for value in beta]
    if len(gain) != count or len(offset) != count:
        raise ValueError("gamma and beta must match the row width")
    if not all(math.isfinite(value) for value in gain + offset):
        raise ValueError("gamma and beta must be finite")
    mean = math.fsum(row) / count
    variance = math.fsum((value - mean) ** 2 for value in row) / count
    inverse_std = 1.0 / math.sqrt(variance + epsilon)
    return [gain[j] * (row[j] - mean) * inverse_std + offset[j]
            for j in range(count)]
