from fractions import Fraction
import math
import random
import unittest

from hw_model.compute import ARCHITECTURES
from hw_model.numerical import (
    layer_norm_float, quantized_gemm, requantize, residual_add, round_even,
    signed_overflow,
)


class NumericalTests(unittest.TestCase):
    def test_signed_saturation_and_wrap(self):
        self.assertEqual([signed_overflow(x, 8) for x in (-129, -128, 127, 128)],
                         [-128, -128, 127, 127])
        self.assertEqual([signed_overflow(x, 8, "wrap") for x in (-129, -128, 127, 128)],
                         [127, -128, 127, -128])
        self.assertEqual(signed_overflow((1 << 130) + 3, 8, "wrap"), 3)
        for bits in (0, -2, True, 8.5):
            with self.assertRaises(ValueError):
                signed_overflow(0, bits)
        with self.assertRaises(ValueError):
            signed_overflow(0, 8, "unknown")

    def test_nearest_even_exact_ties_both_signs(self):
        self.assertEqual([round_even(Fraction(n, 2)) for n in (-7, -5, -3, -1, 1, 3, 5, 7)],
                         [-4, -2, -2, 0, 0, 2, 2, 4])
        self.assertEqual(round_even(Fraction(2**80 + 1, 2)), 2**79)
        self.assertEqual(round_even("2.5000000000000000001"), 3)
        self.assertEqual(requantize(511, Fraction(1, 2), bits=8), 127)
        self.assertEqual(requantize(511, Fraction(1, 2), bits=8, overflow="wrap"), 0)
        for value in (float("nan"), float("inf"), True):
            with self.assertRaises(ValueError):
                round_even(value)

    def test_issue_mappings_match_independent_dense_reference(self):
        rng = random.Random(1977)
        for m, n, r in ((1, 1, 1), (8, 15, 31), (10, 17, 65), (2, 67, 99)):
            a = [[rng.randrange(-7, 8) for _ in range(r)] for _ in range(m)]
            b = [[rng.randrange(-7, 8) for _ in range(n)] for _ in range(r)]
            bias = [rng.randrange(-10, 11) for _ in range(n)]
            expected = [[sum(a[i][k] * b[k][j] for k in range(r)) + bias[j]
                         for j in range(n)] for i in range(m)]
            for architecture in ARCHITECTURES:
                with self.subTest(shape=(m, n, r), architecture=architecture):
                    self.assertEqual(quantized_gemm(a, b, architecture, bias=bias), expected)

    def test_bias_added_once_after_reduction_continuation(self):
        for architecture in ARCHITECTURES:
            with self.subTest(architecture=architecture):
                self.assertEqual(quantized_gemm([[1] * 129], [[1, 2]] * 129,
                                               architecture, bias=[7, -3]), [[136, 255]])
                self.assertEqual(quantized_gemm([[1] * 129], [[1, 2]] * 129,
                                               architecture, bias=[7, -3],
                                               output_bits=8,
                                               output_multiplier=Fraction(1, 2)), [[68, 127]])

    def test_finite_accumulator_issue_boundaries_are_explicit(self):
        a, b = [[100, 100, -100]], [[1], [1], [1]]
        # Per-issue saturation means reduction grouping can change the answer;
        # the reference does not pretend saturation is associative.
        self.assertEqual(quantized_gemm(a, b, "array_9x16", accumulator_bits=8), [[27]])
        self.assertEqual(quantized_gemm(a, b, "row_dot128", accumulator_bits=8), [[100]])
        self.assertEqual(quantized_gemm(a, b, "row_av64", accumulator_bits=8), [[27]])
        for architecture in ARCHITECTURES:
            self.assertEqual(quantized_gemm(a, b, architecture, accumulator_bits=8,
                                           overflow="wrap"), [[100]])

    def test_residual_scale_alignment_per_channel_and_common_output(self):
        result = residual_add([10, 10, -5], [4, -4, 4],
                              main_scale=["0.5", "0.25", "0.5"],
                              residual_scale="0.25", output_scale="0.25")
        self.assertEqual(result, [24, 6, -6])
        # Each branch rounds before adding; this is an explicit numerical policy.
        self.assertEqual(residual_add([1], [1], main_scale="0.5",
                                      residual_scale="0.5"), [0])
        self.assertEqual(residual_add([127], [127], bits=8), [127])
        self.assertEqual(residual_add([127], [127], bits=8, overflow="wrap"), [-2])
        with self.assertRaises(ValueError):
            residual_add([1, 2], [1, 2], main_scale=[1])
        with self.assertRaises(ValueError):
            residual_add([1], [1], output_scale=0)

    def test_ln_population_variance_epsilon_and_affine(self):
        expected = 1 / math.sqrt(1 + 1e-6)
        result = layer_norm_float([0, 2])
        self.assertAlmostEqual(result[0], -expected, places=14)
        self.assertAlmostEqual(result[1], expected, places=14)
        affine = layer_norm_float([0, 2], gamma=[2, 3], beta=[4, 5])
        self.assertAlmostEqual(affine[0], 4 - 2 * expected, places=14)
        self.assertAlmostEqual(affine[1], 5 + 3 * expected, places=14)
        self.assertEqual(layer_norm_float([7] * 64, beta=[2] * 64), [2] * 64)
        tiny = layer_norm_float([0, 0.002])
        self.assertAlmostEqual(tiny[1], 1 / math.sqrt(2), places=14)
        self.assertEqual(layer_norm_float([4]), [0.0])

    def test_invalid_gemm_and_ln_inputs(self):
        for a, b in (([], [[1]]), ([[1, 2]], [[1]]), ([[1], [2, 3]], [[1]])):
            with self.assertRaises(ValueError):
                quantized_gemm(a, b)
        with self.assertRaises(ValueError):
            quantized_gemm([[1]], [[1]], bias=[1, 2])
        with self.assertRaises(ValueError):
            quantized_gemm([[1]], [[1]], output_multiplier=2)
        for kwargs in ({"epsilon": 0}, {"gamma": [1]}, {"beta": [1]},
                       {"epsilon": float("nan")}):
            with self.assertRaises(ValueError):
                layer_norm_float([1, 2], **kwargs)
        with self.assertRaises(ValueError):
            layer_norm_float([1, float("nan")])


if __name__ == "__main__":
    unittest.main()
