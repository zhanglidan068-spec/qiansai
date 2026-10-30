"""Fixed-profile timing, bounded weight packets, and physical SRAM invariants."""

import unittest

from hw_model.compute import Shape
from hw_model.fixed_memory import LocalLayout, LocalMemory, WeightPort


def drive_weights(port, end):
    for cycle in range(port._cycle + 1, end + 1):
        port.complete(cycle)
        port.issue(cycle)


def drain_local(memory, end=100):
    cycle = memory._cycle + 1
    while memory.pending and cycle < end:
        memory.complete(cycle)
        memory.arbitrate(cycle)
        cycle += 1
    if memory.pending:
        raise AssertionError("local memory did not drain")


class WeightPortTests(unittest.TestCase):
    def test_two_packets_have_continuous_ii_two_without_response_bubble(self):
        port = WeightPort(16)
        self.assertTrue(port.reserve("a", 0, 16, 1))
        self.assertTrue(port.reserve("b", 16, 16, 1))
        self.assertFalse(port.reserve("full", 32, 16, 1))
        for cycle in range(18):
            port.complete(cycle)
            self.assertEqual(port.ready("a"), cycle >= 9)
            self.assertEqual(port.ready("b"), cycle >= 17)
            port.issue(cycle)
        issues = [event for event in port.trace if event["event"] == "WEIGHT_BEAT_ISSUE"]
        self.assertEqual([event["cycle"] for event in issues], list(range(0, 16, 2)))
        self.assertEqual([event["packet_id"] for event in issues], ["a"] * 4 + ["b"] * 4)
        self.assertEqual(issues[4]["cycle"], 8)  # a's final response is cycle 9.
        stats = port.stats()
        self.assertEqual(stats["max_outstanding"], 2)
        self.assertEqual(stats["useful_weight_bytes"], 32)
        self.assertEqual(stats["physical_weight_bytes"], 32)
        self.assertEqual(stats["bus_busy_cycles"], 8)
        self.assertEqual(stats["occupied_service_cycles"], 16)
        self.assertEqual(stats["fill_count"], 2)
        self.assertEqual(stats["peak_reserved_bytes"], 32)
        self.assertEqual(stats["peak_live_bytes"], 32)
        self.assertEqual(port.valid_mask("a"), 65535)
        self.assertTrue(port.pending)  # READY registers remain allocated.
        port.consume("a", 18)
        self.assertTrue(port.reserve("c", 32, 16, 1, ready_cycle=18))
        port.consume("b", 18)
        self.assertEqual(port.stats()["active_packets"], 1)

    def test_packet_128_sustains_the_fixed_300_mb_per_second_port(self):
        port = WeightPort(128)
        port.reserve("a", 0, 128, 1)
        port.reserve("b", 128, 128, 1)
        drive_weights(port, 129)
        issues = [event for event in port.trace if event["event"] == "WEIGHT_BEAT_ISSUE"]
        self.assertEqual([event["cycle"] for event in issues], list(range(0, 128, 2)))
        self.assertEqual([event["cycle"] for event in port.trace if event["event"] == "PACKET_READY"], [65, 129])
        self.assertEqual(issues[32]["cycle"], 64)
        self.assertEqual(port.stats()["physical_weight_bytes"], 256)
        self.assertEqual(port.stats()["occupied_service_cycles"], 128)
        self.assertEqual(port.stats()["outstanding_stall_cycles"], 0)

    def test_packet_cannot_be_consumed_until_all_bytes_arrive(self):
        port = WeightPort(16)
        port.reserve("p", 0, 16, 3)
        drive_weights(port, 3)
        self.assertEqual(port.valid_mask("p"), 15)
        self.assertFalse(port.ready("p"))
        with self.assertRaisesRegex(ValueError, "READY"):
            port.consume("p", 3)
        drive_weights(port, 9)
        port.consume("p", 9)
        self.assertTrue(port.ready("p"))
        self.assertEqual(port.stats()["reserved_bytes"], 16)
        port.consume("p", 10)
        self.assertTrue(port.pending)
        port.consume("p", 11)
        self.assertFalse(port.pending)
        self.assertFalse(port.ready("p"))
        self.assertEqual(port.stats()["consumers_completed"], 3)
        self.assertEqual(port.stats()["reuse_count"], 2)
        self.assertEqual(port.stats()["released_packets"], 1)
        with self.assertRaises(ValueError):
            port.consume("p", 11)

    def test_unaligned_source_and_tail_masks_count_real_physical_beats(self):
        port = WeightPort(16)
        port.reserve("odd", 3, 5, 1, tag={"tensor": "tail"})
        drive_weights(port, 5)
        issues = [event for event in port.trace if event["event"] == "WEIGHT_BEAT_ISSUE"]
        self.assertEqual([event["address"] for event in issues], [0, 4])
        self.assertEqual([event["valid_bytes"] for event in issues], [1, 4])
        self.assertEqual([event["byte_mask"] for event in issues], [8, 15])
        self.assertEqual([event["packet_byte_mask"] for event in issues], [1, 30])
        self.assertEqual(port.valid_mask("odd"), 31)
        self.assertEqual(port.stats()["useful_weight_bytes"], 5)
        self.assertEqual(port.stats()["physical_weight_bytes"], 8)
        self.assertEqual(port.stats()["live_bytes"], 5)
        self.assertEqual(port.stats()["reserved_bytes"], 16)
        self.assertTrue(all(event["tag"] == {"tensor": "tail"} for event in issues))

    def test_ready_cycle_and_positive_consumer_count(self):
        port = WeightPort(16)
        port.reserve("p", 0, 1, 1, ready_cycle=4)
        drive_weights(port, 3)
        self.assertEqual(port.stats()["accepted_beats"], 0)
        drive_weights(port, 7)
        self.assertTrue(port.ready("p"))
        self.assertEqual(port.stats()["physical_weight_bytes"], 4)
        for amount in (0, 2, -1, True):
            with self.assertRaises(ValueError):
                port.consume("p", 7, amount)
        self.assertEqual(port.stats()["consumers_completed"], 0)

    def test_bounds_duplicate_ids_and_cycle_protocol(self):
        port = WeightPort(16)
        for address, useful in ((-1, 1), (524288, 1), (524287, 2), (0, 17), (0, 0)):
            with self.assertRaises(ValueError):
                port.reserve("bad", address, useful, 1)
        port.reserve("last", 524287, 1, 1)
        with self.assertRaises(ValueError):
            port.reserve("last", 0, 1, 1)
        port.complete(0)
        port.issue(0)
        with self.assertRaises(ValueError):
            port.complete(0)
        with self.assertRaises(ValueError):
            port.issue(0)
        drive_weights(port, 3)
        self.assertTrue(port.ready("last"))
        self.assertEqual(port.stats()["physical_weight_bytes"], 4)
        port.consume("last", 3)
        with self.assertRaises(ValueError):
            port.reserve("last", 0, 1, 1)

    def test_trace_can_be_disabled_and_stats_are_snapshots(self):
        port = WeightPort(16, capture_trace=False)
        port.reserve("p", 0, 1, 1)
        drive_weights(port, 3)
        snapshot = port.stats()
        snapshot["valid_byte_masks"]["p"] = 0
        self.assertEqual(port.valid_mask("p"), 1)
        self.assertEqual(port.trace, [])


class LocalMemoryTests(unittest.TestCase):
    def test_physical_eight_banks_read_or_write_once_per_cycle(self):
        memory, completed = LocalMemory(), []
        memory.submit("read", "matrix", False, [(bank, 0, 3) for bank in range(8)],
                      lambda: completed.append(("read", memory._cycle)))
        memory.submit("write", "ln", True, [(bank, 1, 1) for bank in range(8)],
                      lambda: completed.append(("write", memory._cycle)))
        memory.complete(0)
        memory.arbitrate(0)
        self.assertEqual(completed, [])
        memory.complete(1)
        self.assertEqual(completed, [("read", 1)])
        memory.arbitrate(1)
        memory.complete(2)
        self.assertEqual(completed, [("read", 1), ("write", 2)])
        self.assertFalse(memory.pending)
        grants = [event for event in memory.trace if event["event"] in ("LOCAL_READ", "LOCAL_WRITE")]
        counts = {}
        for event in grants:
            key = (event["cycle"], event["bank"])
            counts[key] = counts.get(key, 0) + 1
            self.assertEqual(event["complete_cycle"], event["cycle"] + 1)
        self.assertTrue(all(count == 1 for count in counts.values()))
        stats = memory.stats()
        self.assertEqual(stats["physical_read_words"], 8)
        self.assertEqual(stats["physical_write_words"], 8)
        self.assertEqual(stats["read_useful_bytes"], 16)
        self.assertEqual(stats["write_useful_bytes"], 8)
        self.assertEqual(stats["peak_outstanding_words"], 8)

    def test_requests_progress_partially_and_owner_arbitration_is_fair(self):
        memory, done = LocalMemory(), []
        for owner in ("a", "b"):
            memory.submit(owner, owner, False, [(0, word, 1) for word in range(4)],
                          lambda owner=owner: done.append((owner, memory._cycle)))
        drain_local(memory)
        grants = [event for event in memory.trace if event["event"] == "LOCAL_READ"]
        self.assertEqual([event["owner"] for event in grants], ["a", "b"] * 4)
        self.assertEqual(done, [("a", 7), ("b", 8)])
        self.assertEqual(memory.stats()["completed_words"], 8)
        self.assertGreater(memory.stats()["denied_cycles"], 0)
        self.assertGreater(memory.stats()["per_owner"]["b"]["denied_cycles"], 0)

    def test_mask_union_reads_packed_bytes_without_free_physical_copies(self):
        memory, done = LocalMemory(), []
        memory.submit("pair", "matrix", False, [(2, 4, 1), (2, 4, 2), (2, 4, 1)],
                      lambda: done.append("pair"))
        drain_local(memory)
        self.assertEqual(done, ["pair"])
        self.assertEqual(memory.stats()["physical_read_words"], 1)
        self.assertEqual(memory.stats()["read_useful_bytes"], 2)
        event = next(item for item in memory.trace if item["event"] == "LOCAL_READ")
        self.assertEqual(event["byte_mask"], 3)
        memory.submit("again", "matrix", False, [(2, 4, 1)], lambda: done.append("again"))
        drain_local(memory)
        self.assertEqual(memory.stats()["physical_read_words"], 2)

    def test_same_owner_fifo_and_callback_can_submit_next_action(self):
        memory, done = LocalMemory(), []
        def first_done():
            done.append(("first", memory._cycle))
            memory.submit("next", "owner", True, [(1, 0, 3)],
                          lambda: done.append(("next", memory._cycle)))
        memory.submit("first", "owner", False, [(0, 0, 3), (0, 1, 3)], first_done)
        drain_local(memory)
        self.assertEqual(done, [("first", 2), ("next", 3)])
        self.assertEqual(memory.stats()["submitted_requests"], 2)
        self.assertEqual(memory.stats()["completed_requests"], 2)
        self.assertEqual(memory.stats()["pending_requests"], 0)

    def test_bounds_and_bad_transaction_do_not_allocate_or_call_back(self):
        memory, done = LocalMemory(), []
        for accesses in ([(8, 0, 1)], [(0, 4096, 1)], [(0, -1, 1)],
                         [(0, 0, 0)], [(0, 0, 4)], [(0, 0)], []):
            with self.assertRaises(ValueError):
                memory.submit("bad", "owner", False, accesses, lambda: done.append("bad"))
        self.assertFalse(memory.pending)
        self.assertEqual(memory.stats()["submitted_requests"], 0)
        self.assertEqual(done, [])
        memory.submit("last", "owner", True, [(7, 4095, 2)], lambda: done.append("last"))
        drain_local(memory)
        self.assertEqual(done, ["last"])
        with self.assertRaises(ValueError):
            memory.submit("last", "owner", True, [(0, 0, 1)], lambda: None)

    def test_cycle_protocol_trace_toggle_and_stats_snapshot(self):
        memory = LocalMemory(capture_trace=False)
        memory.submit("p", "owner", False, [(0, 0, 1)], lambda: None)
        memory.complete(0)
        memory.arbitrate(0)
        with self.assertRaises(ValueError):
            memory.complete(0)
        with self.assertRaises(ValueError):
            memory.arbitrate(0)
        memory.complete(1)
        snapshot = memory.stats()
        snapshot["per_owner"]["owner"]["read_useful_bytes"] = -1
        snapshot["per_bank"][0]["read_words"] = -1
        self.assertEqual(memory.stats()["per_owner"]["owner"]["read_useful_bytes"], 1)
        self.assertEqual(memory.stats()["per_bank"][0]["read_words"], 1)
        self.assertEqual(memory.trace, [])


class LocalLayoutTests(unittest.TestCase):
    def test_fixed_64_kib_layout_for_all_four_mappings(self):
        shape = Shape(99, 64, 256)
        expected = {"A": 50944, "B0": 50688, "B1_G9": 50760, "B1_G99": 51480}
        for mapping, used in expected.items():
            with self.subTest(mapping=mapping):
                group = 99 if mapping == "B1_G99" else 9
                layout = LocalLayout(shape, mapping, group)
                self.assertEqual(layout.used_bytes, used)
                self.assertEqual(layout.capacity_bytes, 65536)
                self.assertEqual(layout.free_bytes + used, 65536)
                self.assertNotIn("weight", layout.tensors)
                self.assertEqual("partial" in layout.tensors, mapping.startswith("B1"))
                extents = [(item["base_byte_addr"], item["base_byte_addr"] + item["padded_bytes"])
                           for item in layout.tensors.values()]
                self.assertTrue(all(first[1] <= second[0] for first, second in zip(extents, extents[1:])))

    def test_array_input_is_k_major_with_even_token_stride(self):
        layout = LocalLayout(Shape(3, 5, 3), "A", 9)
        self.assertEqual(layout.tensors["input"]["logical_bytes"], 9)
        self.assertEqual(layout.tensors["input"]["padded_bytes"], 12)
        self.assertEqual([layout.address("input", row, 0) for row in range(3)], [0, 1, 2])
        self.assertEqual([layout.address("input", 0, k) for k in range(3)], [0, 4, 8])
        self.assertEqual(layout.accesses("input", 1, [0, 1, 2]),
                         [(0, 0, 2), (2, 0, 2), (4, 0, 2)])

    def test_row_input_compact_row_major_can_share_an_odd_row_boundary(self):
        layout = LocalLayout(Shape(3, 5, 3), "B0", 1)
        self.assertEqual(layout.address("input", 1, 0), 3)
        self.assertEqual(layout.accesses("input", 0, [2]), [(1, 0, 1)])
        self.assertEqual(layout.accesses("input", 1, [0]), [(1, 0, 2)])
        self.assertEqual(layout.tensors["input"]["padded_bytes"], 10)

    def test_int16_and_int32_accesses_include_every_physical_byte(self):
        layout = LocalLayout(Shape(10, 3, 5), "B1_G9", 9)
        self.assertEqual(layout.tensors["partial"]["bytes"], 72)
        base = layout.tensors["partial"]["base_byte_addr"]
        self.assertEqual(layout.address("partial", 2, 1), base + (2 * 2 + 1) * 4)
        partial = layout.accesses("partial", 2, [0, 1])
        self.assertEqual(len(partial), 4)
        self.assertTrue(all(mask == 3 for _, _, mask in partial))
        pre_ln = layout.accesses("pre_ln", 2, [0, 1, 2])
        self.assertEqual(len(pre_ln), 3)
        self.assertTrue(all(mask == 3 for _, _, mask in pre_ln))
        tail = LocalLayout(Shape(3, 3, 5), "B1_G99", 99)
        self.assertEqual(tail.tensors["partial"]["bytes"], 24)

    def test_exact_capacity_fits_and_excess_is_rejected(self):
        for mapping in ("A", "B0"):
            layout = LocalLayout(Shape(128, 64, 256), mapping, 1)
            self.assertEqual(layout.used_bytes, 65536)
            self.assertEqual(layout.address("output", 127, 63), 65535)
            self.assertEqual(layout.accesses("output", 127, [63]), [(7, 4095, 2)])
        with self.assertRaisesRegex(ValueError, "65536"):
            LocalLayout(Shape(128, 64, 256), "B1_G9", 1)
        with self.assertRaisesRegex(ValueError, "65536"):
            LocalLayout(Shape(129, 64, 256), "B0", 1)

    def test_bad_mapping_indices_and_descriptor_snapshots(self):
        with self.assertRaises(ValueError):
            LocalLayout(Shape(1, 1, 1), "unknown", 1)
        layout = LocalLayout(Shape(3, 5, 7), "A", 9)
        for tensor, row, col in (("weight", 0, 0), ("input", -1, 0),
                                 ("input", 3, 0), ("input", 0, 7), ("partial", 0, 0)):
            with self.assertRaises(ValueError):
                layout.address(tensor, row, col)
        snapshot = layout.stats()
        snapshot["tensors"]["input"]["base_byte_addr"] = -1
        self.assertEqual(layout.address("input", 0, 0), 0)


if __name__ == "__main__":
    unittest.main()
