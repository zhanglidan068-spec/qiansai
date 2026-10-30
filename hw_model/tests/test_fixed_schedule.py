import itertools
import random
import unittest

from hw_model.compute import Shape
from hw_model.fixed_schedule import (
    MAPPINGS, arithmetic_checks, fixed_l0_benchmarks, mapping_traits,
    schedule_packets, weight_byte_offset,
)


class FixedScheduleTests(unittest.TestCase):
    def test_fixed_weight_and_ln128_l0_values(self):
        report = arithmetic_checks()
        self.assertEqual(report, fixed_l0_benchmarks())
        self.assertEqual(report["scope"], "weight_transfer_only_plus_ln128")
        expected = {"projection": (23680, 202880, 2048),
                    "ffn2": (91264, 811136, 8192)}
        for operator in report["operators"]:
            a, b, first_batch = expected[operator["operator"]]
            self.assertEqual(operator["A"]["total_cycles"], a)
            self.assertEqual(operator["B0"]["total_cycles"], b)
            self.assertEqual(operator["A"]["first_batch_weight_cycles"], first_batch)
            self.assertEqual(operator["B0"]["first_batch_weight_cycles"], first_batch)
            self.assertEqual(operator["A"]["first_row_cycles"], first_batch + 128)

    def test_document_weight_reuse_ledgers(self):
        expected = {
            ("projection", "A"): 45056, ("projection", "B0"): 405504,
            ("projection", "B1_G9"): 45056, ("projection", "B1_G99"): 4096,
            ("ffn2", "A"): 180224, ("ffn2", "B0"): 1622016,
            ("ffn2", "B1_G9"): 180224, ("ffn2", "B1_G99"): 16384,
        }
        for entry in fixed_l0_benchmarks()["weight_ledger"]:
            key = (entry["operator"], entry["mapping"])
            self.assertEqual(entry["weight_bytes"], expected[key])
            self.assertEqual(entry["physical_weight_bytes"], expected[key])
            self.assertEqual(entry["useful_mac"], 405504 if key[0] == "projection" else 1622016)

    def test_registers_and_local_workspace_are_separate_payloads(self):
        shape = Shape(99, 64, 256)
        a = mapping_traits(shape, "A")
        b0 = mapping_traits(shape, "B0")
        b1 = mapping_traits(shape, "B1_G99")
        self.assertEqual(a["activation_register_bytes"], 18)
        self.assertEqual(b0["activation_register_bytes"], 256)
        self.assertEqual(b1["activation_register_bytes"], 128)
        self.assertEqual(b1["accumulator_register_bytes"], 8)
        self.assertEqual(b1["operand_partial_register_bytes"], 16)
        self.assertEqual(b1["partial_sum_workspace_bytes"], 792)
        self.assertEqual(b1["row_assembly_bytes"], 12672)

    def test_all_tail_combinations_partition_products_and_reuse_weights(self):
        for m, n, r in itertools.product((1, 8, 9, 10, 99),
                                         (1, 15, 16, 17, 64),
                                         (1, 31, 32, 33, 64, 65, 99, 256)):
            shape = Shape(m, n, r)
            for mapping in MAPPINGS:
                with self.subTest(shape=shape, mapping=mapping):
                    traits = mapping_traits(shape, mapping)
                    masks, identifiers, address_uses = {}, set(), {}
                    macs = finals = 0
                    for packet in schedule_packets(shape, mapping):
                        self.assertNotIn(packet.packet_id, identifiers)
                        identifiers.add(packet.packet_id)
                        self.assertEqual(packet.weight_bytes,
                                         len(packet.columns) * len(packet.reduction))
                        self.assertEqual(packet.physical_weight_bytes % 4, 0)
                        self.assertGreaterEqual(packet.physical_weight_bytes, packet.weight_bytes)
                        self.assertLessEqual(packet.physical_weight_bytes, traits["weight_packet_slot_bytes"])
                        self.assertEqual(packet.first_reduction, packet.reduction[0] == 0)
                        self.assertEqual(packet.last_reduction, packet.reduction[-1] == r - 1)
                        self.assertEqual(packet.last_tile_in_group,
                                         packet.last_reduction and packet.columns[-1] == n - 1)
                        finals += int(packet.last_tile_in_group)
                        for indices, bound in ((packet.rows, m), (packet.columns, n),
                                               (packet.reduction, r)):
                            self.assertEqual(indices, tuple(range(indices[0], indices[-1] + 1)))
                            self.assertTrue(0 <= indices[0] <= indices[-1] < bound)
                        key = (packet.rows, packet.columns)
                        mask = ((1 << len(packet.reduction)) - 1) << packet.reduction[0]
                        self.assertEqual(masks.get(key, 0) & mask, 0)
                        masks[key] = masks.get(key, 0) | mask
                        address_uses[packet.weight_address] = address_uses.get(packet.weight_address, 0) + 1
                        macs += packet.useful_mac
                    output_masks = [0] * m
                    for (rows, columns), mask in masks.items():
                        self.assertEqual(mask, (1 << r) - 1)
                        column_mask = ((1 << len(columns)) - 1) << columns[0]
                        for row in rows:
                            self.assertEqual(output_masks[row] & column_mask, 0)
                            output_masks[row] |= column_mask
                    group_count = (m + traits["group_rows"] - 1) // traits["group_rows"]
                    self.assertEqual(output_masks, [(1 << n) - 1] * m)
                    self.assertEqual(macs, m * n * r)
                    self.assertEqual(finals, group_count)
                    self.assertTrue(all(count == group_count for count in address_uses.values()))

    def test_prepacked_valid_weight_addresses_are_bijective_and_packet_contiguous(self):
        for n, r in itertools.product((1, 2, 3, 15, 16, 17, 64), (1, 33, 64, 65, 99)):
            shape = Shape(1, n, r)
            for mapping in MAPPINGS:
                addresses = {weight_byte_offset(shape, mapping, k, column)
                             for k in range(r) for column in range(n)}
                self.assertEqual(len(addresses), n * r)
                transferred = set()
                for packet in schedule_packets(shape, mapping):
                    useful = {weight_byte_offset(shape, mapping, k, column)
                              for k in packet.reduction for column in packet.columns}
                    self.assertEqual(useful, set(range(packet.weight_address,
                                                       packet.weight_address + packet.weight_bytes)))
                    self.assertFalse(useful.intersection(transferred))
                    transferred.update(useful)
                self.assertEqual(transferred, addresses)
                self.assertLess(max(addresses), mapping_traits(shape, mapping)["weight_storage_bytes"])

    def test_packet_consumption_matches_independent_dense_products(self):
        shape = Shape(10, 17, 65)
        rng = random.Random(42)
        a = [[rng.randrange(-3, 4) for _ in range(shape.r)] for _ in range(shape.m)]
        b = [[rng.randrange(-3, 4) for _ in range(shape.n)] for _ in range(shape.r)]
        expected = [[sum(a[i][k] * b[k][j] for k in range(shape.r))
                     for j in range(shape.n)] for i in range(shape.m)]
        for mapping in MAPPINGS:
            result = [[0] * shape.n for _ in range(shape.m)]
            for packet in schedule_packets(shape, mapping):
                for row in packet.rows:
                    for column in packet.columns:
                        result[row][column] += sum(a[row][k] * b[k][column] for k in packet.reduction)
            self.assertEqual(result, expected)

    def test_invalid_mapping_or_weight_indices_rejected(self):
        shape = Shape(1, 1, 1)
        with self.assertRaises(ValueError):
            list(schedule_packets(shape, "B1"))
        for k, column in ((-1, 0), (1, 0), (0, 1), (True, 0)):
            with self.assertRaises(ValueError):
                weight_byte_offset(shape, "A", k, column)


if __name__ == "__main__":
    unittest.main()
