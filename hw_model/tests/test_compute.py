import itertools
import unittest

from hw_model.compute import (
    ARCHITECTURES, Shape, benchmarks, ideal_fused, ideal_summary, iter_issues,
)


class ComputeTests(unittest.TestCase):
    def test_document_operator_cycles_and_macs(self):
        report = benchmarks()
        expected = {
            "q": (2816, 3168), "k": (2816, 3168), "v": (2816, 3168),
            "qk": (4928, 4950), "av": (4356, 4950),
            "projection": (2816, 3168), "ffn1": (11264, 12672),
            "ffn2": (11264, 12672),
        }
        for operator in report["operators"]:
            self.assertEqual((operator["array_9x16"]["issue_cycles"],
                              operator["row_dot128"]["issue_cycles"]),
                             expected[operator["operator"]])
        self.assertEqual(report["totals"], {
            "array_9x16": 43076, "row_dot128": 47916, "useful_mac": 6120576})
        av = next(op for op in report["operators"] if op["operator"] == "av")
        self.assertEqual(av["row_dot128"]["architecture"], "row_av64")
        self.assertEqual(12 * report["totals"]["useful_mac"], 73446912)

    def test_document_first_rows_and_ideal_ln(self):
        for shape, architecture, matrix_first, ln_last in (
                (Shape(99, 64, 64), "array_9x16", 256, 6592),
                (Shape(99, 64, 64), "row_dot128", 32, 6368),
                (Shape(99, 64, 256), "array_9x16", 1024, 11840),
                (Shape(99, 64, 256), "row_dot128", 128, 12736)):
            with self.subTest(shape=shape, architecture=architecture):
                result = ideal_fused(shape, architecture)
                self.assertEqual(result["matrix"]["first_row_cycles"], matrix_first)
                self.assertEqual(result["first_row_cycles"], matrix_first + 64)
                self.assertEqual(result["total_cycles"], ln_last)
                self.assertEqual(result["scope"], "ideal_infinite_buffer")

    def test_all_document_boundary_combinations_partition_every_product(self):
        # Keep reduction coverage as bit sets per output tile, avoiding a large
        # set of individual MAC tuples. Output tiles must also partition each row.
        dimensions = itertools.product(
            (1, 8, 9, 10, 99), (1, 15, 16, 17, 64, 99, 256),
            (1, 31, 32, 33, 64, 99, 256))
        for m, n, r in dimensions:
            shape = Shape(m, n, r)
            for architecture in ARCHITECTURES:
                with self.subTest(shape=shape, architecture=architecture):
                    tiles = {}
                    issue_count = macs = 0
                    lanes = 144 if architecture == "array_9x16" else 128
                    for issue in iter_issues(shape, architecture):
                        for indices, limit in ((issue.rows, m), (issue.columns, n),
                                               (issue.reduction, r)):
                            self.assertTrue(indices)
                            self.assertGreaterEqual(indices[0], 0)
                            self.assertLess(indices[-1], limit)
                            self.assertEqual(indices, tuple(range(indices[0], indices[-1] + 1)))
                        useful = len(issue.rows) * len(issue.columns) * len(issue.reduction)
                        self.assertEqual(issue.useful_mac, useful)
                        self.assertLessEqual(useful, lanes)
                        mask = ((1 << len(issue.reduction)) - 1) << issue.reduction[0]
                        key = (issue.rows, issue.columns)
                        previous = tiles.get(key, 0)
                        self.assertEqual(previous & mask, 0, "duplicate reduction product")
                        tiles[key] = previous | mask
                        issue_count += 1
                        macs += useful
                    columns_seen = [0] * m
                    for (rows, columns), mask in tiles.items():
                        self.assertEqual(mask, (1 << r) - 1, "missing reduction product")
                        col_mask = ((1 << len(columns)) - 1) << columns[0]
                        for row in rows:
                            self.assertEqual(columns_seen[row] & col_mask, 0,
                                             "overlapping output tiles")
                            columns_seen[row] |= col_mask
                    self.assertEqual(columns_seen, [(1 << n) - 1] * m)
                    summary = ideal_summary(shape, architecture)
                    self.assertEqual(issue_count, summary["issue_cycles"])
                    self.assertEqual(macs, m * n * r)

    def test_array_nine_rows_and_row_order(self):
        shape = Shape(10, 17, 33)
        array = ideal_summary(shape, "array_9x16")
        self.assertEqual(array["row_ready_cycles"], [66] * 9 + [132])
        row = ideal_summary(shape, "row_dot128")
        self.assertEqual(row["row_ready_cycles"], list(range(9, 91, 9)))
        issues = list(iter_issues(shape, "array_9x16"))
        self.assertEqual(issues[0].rows, tuple(range(9)))
        self.assertEqual(issues[0].columns, tuple(range(16)))
        self.assertEqual(issues[33].columns, (16,))
        self.assertEqual(issues[66].rows, (9,))

    def test_ln_latency_ii_and_capacity_are_independent(self):
        shape = Shape(4, 1, 1)
        pipelined = ideal_fused(shape, "row_dot128", ln_latency=7, ln_ii=2, max_inflight=2)
        self.assertEqual(pipelined["row_accept_cycles"], [1, 3, 8, 10])
        self.assertEqual(pipelined["row_done_cycles"], [8, 10, 15, 17])
        serial = ideal_fused(shape, "row_dot128", ln_latency=7, ln_ii=2, max_inflight=1)
        self.assertEqual(serial["row_accept_cycles"], [1, 8, 15, 22])

    def test_invalid_parameters(self):
        for dimensions in ((0, 1, 1), (1, -1, 1), (1, 1, 0), (True, 1, 1), (1.5, 1, 1)):
            with self.assertRaises(ValueError):
                Shape(*dimensions)
        with self.assertRaises(ValueError):
            list(iter_issues(Shape(1, 1, 1), "unknown"))
        for kwargs in ({"ln_latency": 0}, {"ln_ii": 0}, {"max_inflight": 0}):
            with self.assertRaises(ValueError):
                ideal_fused(Shape(1, 1, 1), "array_9x16", **kwargs)


if __name__ == "__main__":
    unittest.main()
