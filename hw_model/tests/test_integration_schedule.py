"""End-to-end schedule invariants for the fixed weight-port contract."""
from collections import Counter
import unittest

from hw_model.compute import Shape
from hw_model.config import load_config
from hw_model.core import Capacity, EventQueue
from hw_model.fixed_schedule import MAPPINGS, schedule_packets
from hw_model.fixed_simulation import FusedSimulator, simulate


def events(trace, kind):
    return [item for item in trace if item["event"] == kind]


def row_cycles(trace, kind):
    return {item["token"]: item["cycle"] for item in events(trace, kind)}


class IntegrationScheduleTests(unittest.TestCase):
    def assert_complete(self, result, trace, shape, mapping):
        self.assertEqual(result["useful_mac"], shape.m * shape.n * shape.r)
        self.assertEqual([item["cycle"] for item in trace],
                         sorted(item["cycle"] for item in trace))
        confirmed = events(trace, "INTERNAL_OUTPUT_CONFIRMED")
        self.assertEqual(sorted(item["token"] for item in confirmed), list(range(shape.m)))
        self.assertEqual(max(item["cycle"] for item in confirmed), result["total_cycles"])
        packets = {packet.packet_id: packet for packet in schedule_packets(shape, mapping)}
        issues = events(trace, "COMPUTE_ISSUE")
        self.assertEqual(sum(item["useful_mac"] for item in issues), shape.useful_mac)
        seen, consumers = set(), Counter()
        for item in issues:
            packet = packets[item["packet_id"]]
            consumers[packet.packet_id] += 1
            if mapping == "A":
                self.assertEqual(item["useful_mac"], packet.useful_mac)
            else:
                self.assertIn(item["token"], packet.rows)
                key = (packet.packet_id, item["token"])
                self.assertNotIn(key, seen)
                seen.add(key)
                self.assertEqual(item["useful_mac"], len(packet.columns) * len(packet.reduction))
        for packet_id, packet in packets.items():
            self.assertEqual(consumers[packet_id], 1 if mapping == "A" else len(packet.rows))
        self.assertEqual(result["issued_updates"], len(issues))
        self.assertEqual(result["output_rows"], shape.m)
        self.assertEqual(result["output_elements"], shape.m * shape.n)
        self.assertEqual(result["row_output_cycles"],
                         [row_cycles(trace, "INTERNAL_OUTPUT_CONFIRMED")[row] for row in range(shape.m)])
        self.assertEqual(result["last_internal_output_confirm"], result["total_cycles"])
        self.assertEqual(result["shared_activation_read_bytes"], 0)
        self.assertEqual(result["shared_activation_write_bytes"], 0)
        self.assertEqual(result["register_activity"]["bias_reads_once"], shape.m * shape.n)
        self.assertLessEqual(result["peak_operand_packets"], 2)
        self.assertLessEqual(result["peak_post_jobs"], 2)
        for name, buffer in result["buffers"].items():
            self.assertLessEqual(buffer["peak_live_bytes"], buffer["peak_reserved_bytes"])
            self.assertLessEqual(buffer["peak_reserved_bytes"], buffer["capacity_bytes"])
            self.assertEqual(buffer["final_reserved_bytes"],
                             shape.m * shape.n if name == "resident_output" else 0)
        self.assertEqual(result["buffers"]["resident_output"]["peak_live_bytes"], shape.m * shape.n)
        weight = result["weight_port"]
        self.assertEqual(weight["useful_weight_bytes"], sum(p.weight_bytes for p in packets.values()))
        self.assertEqual(weight["physical_weight_bytes"], sum(p.physical_weight_bytes for p in packets.values()))
        self.assertEqual(weight["fill_count"], len(packets))
        self.assertEqual(weight["released_packets"], len(packets))
        self.assertEqual(weight["consumers_completed"], len(issues))
        self.assertEqual(weight["accepted_beats"], weight["physical_weight_bytes"] // 4)
        self.assertEqual(weight["completed_beats"], weight["accepted_beats"])
        self.assertLessEqual(weight["max_outstanding"], 2)
        self.assertEqual(weight["active_packets"], 0)
        self.assertEqual(weight["reserved_bytes"], 0)
        self.assertEqual(weight["live_bytes"], 0)
        self.assertEqual(min(item["cycle"] for item in events(trace, "WEIGHT_BEAT_ISSUE")), 0)

    def test_fixed_mappings_complete_small_tail_workloads(self):
        config = load_config()
        for shape in (Shape(1, 1, 1), Shape(10, 17, 65)):
            for mapping in MAPPINGS:
                with self.subTest(shape=shape, mapping=mapping):
                    result, trace = simulate(config, mapping, shape=shape)
                    self.assert_complete(result, trace, shape, mapping)

    def test_repeated_runs_produce_identical_complete_event_traces(self):
        config = load_config()
        for mapping in MAPPINGS:
            with self.subTest(mapping=mapping):
                first = simulate(config, mapping, shape=Shape(3, 17, 33))
                second = simulate(config, mapping, shape=Shape(3, 17, 33))
                self.assertEqual(first, second)

    def test_weight_packet_must_be_complete_before_any_mac_consumer(self):
        config = load_config()
        for mapping in MAPPINGS:
            with self.subTest(mapping=mapping):
                shape = Shape(10, 17, 65)
                result, trace = simulate(config, mapping, shape=shape)
                self.assert_complete(result, trace, shape, mapping)
                ready = {item["packet_id"]: item for item in events(trace, "PACKET_READY")}
                for item in events(trace, "COMPUTE_ISSUE"):
                    self.assertIn(item["packet_id"], ready)
                    self.assertGreaterEqual(item["cycle"], ready[item["packet_id"]]["cycle"])
                for item in ready.values():
                    self.assertEqual(item["valid_mask"], (1 << item["useful_bytes"]) - 1)
                for item in events(trace, "WEIGHT_BEAT_ISSUE"):
                    self.assertEqual(item["complete_cycle"], item["cycle"] + 3)
                issued = {(item["packet_id"], tuple(item["rows"])): item
                          for item in events(trace, "COMPUTE_ISSUE")}
                last_mac = {}
                for item in events(trace, "MAC_COMPLETE"):
                    key = (item["packet_id"], tuple(item["rows"]))
                    self.assertEqual(item["cycle"], issued[key]["complete_cycle"])
                    last_mac[item["packet_id"]] = max(last_mac.get(item["packet_id"], 0), item["cycle"])
                for item in events(trace, "PACKET_RELEASE"):
                    self.assertEqual(item["cycle"], last_mac[item["packet_id"]])
                allocated = set()
                for item in trace:
                    if item["event"] == "PACKET_RESERVED":
                        self.assertNotIn(item["packet_id"], allocated)
                        allocated.add(item["packet_id"])
                        self.assertLessEqual(len(allocated), 2)
                    elif item["event"] == "PACKET_RELEASE":
                        allocated.remove(item["packet_id"])
                self.assertFalse(allocated)

    def test_ln_uses_full_postprocessed_rows_and_fixed128_contract(self):
        config = load_config()
        shape = Shape(10, 17, 33)
        for mapping in MAPPINGS:
            with self.subTest(mapping=mapping):
                result, trace = simulate(config, mapping, shape=shape)
                self.assert_complete(result, trace, shape, mapping)
                matrix = row_cycles(trace, "MATRIX_ROW_COMPLETE")
                ready = row_cycles(trace, "LN_INPUT_READY")
                accepts = events(trace, "LN_ACCEPT")
                done = row_cycles(trace, "LN_CORE_DONE")
                confirmed = row_cycles(trace, "INTERNAL_OUTPUT_CONFIRMED")
                self.assertEqual(set(matrix), set(range(shape.m)))
                self.assertEqual(set(ready), set(range(shape.m)))
                for item in accepts:
                    row = item["token"]
                    self.assertLessEqual(matrix[row], ready[row])
                    self.assertLessEqual(ready[row], item["cycle"])
                    self.assertEqual(done[row], item["cycle"] + 128)
                    self.assertGreaterEqual(confirmed[row], done[row])
                self.assertEqual(len(accepts), shape.m)
                for previous, current in zip(accepts, accepts[1:]):
                    self.assertGreaterEqual(current["cycle"] - previous["cycle"], 128)
                    self.assertGreaterEqual(current["cycle"], confirmed[previous["token"]])

    def test_matrix_and_post_channel_masks_are_independently_complete(self):
        shape = Shape(10, 17, 129)
        for mapping in MAPPINGS:
            with self.subTest(mapping=mapping):
                result, trace = simulate(load_config(), mapping, shape=shape)
                self.assert_complete(result, trace, shape, mapping)
                matrix, pre_ln = [set() for _ in range(shape.m)], [set() for _ in range(shape.m)]
                live_pre_ln = peak_pre_ln = 0
                for item in trace:
                    if item["event"] in ("MAT_FRAGMENT_READY", "POST_WRITE_DONE"):
                        row = item["token"]
                        payload = matrix[row] if item["event"] == "MAT_FRAGMENT_READY" else pre_ln[row]
                        self.assertFalse(payload.intersection(item["channels"]))
                        self.assertTrue(all(0 <= channel < shape.n for channel in item["channels"]))
                        payload.update(item["channels"])
                        if item["event"] == "POST_WRITE_DONE":
                            live_pre_ln += len(item["channels"]) * 2
                            peak_pre_ln = max(peak_pre_ln, live_pre_ln)
                    elif item["event"] == "MATRIX_ROW_COMPLETE":
                        self.assertEqual(matrix[item["token"]], set(range(shape.n)))
                    elif item["event"] == "LN_INPUT_READY":
                        self.assertEqual(pre_ln[item["token"]], set(range(shape.n)))
                        live_pre_ln -= shape.n * 2
                        self.assertGreaterEqual(live_pre_ln, 0)
                self.assertEqual(matrix, [set(range(shape.n)) for _ in range(shape.m)])
                self.assertEqual(pre_ln, matrix)
                self.assertEqual(live_pre_ln, 0)
                self.assertEqual(peak_pre_ln, result["buffers"]["pre_ln_active_rows"]["peak_live_bytes"])

    def test_b1_partial_sums_saved_and_read_once_per_continuation(self):
        shape = Shape(10, 3, 129)
        for mapping in ("B1_G9", "B1_G99"):
            with self.subTest(mapping=mapping):
                result, trace = simulate(load_config(), mapping, shape=shape)
                self.assert_complete(result, trace, shape, mapping)
                packets = {p.packet_id: p for p in schedule_packets(shape, mapping)}
                saved = events(trace, "PARTIAL_SUM_SAVED")
                expected = sum(len(p.rows) for p in packets.values() if not p.last_reduction)
                self.assertEqual(len(saved), expected)
                reads = [item for item in events(trace, "LOCAL_REQUEST") if item["owner"] == "partial_read"]
                writes = [item for item in events(trace, "LOCAL_REQUEST") if item["owner"] == "partial_write"]
                self.assertEqual(len(reads), expected)
                self.assertEqual(len(writes), expected)
                save_cycles = {(packets[item["packet_id"]].tile_id, item["token"],
                                packets[item["packet_id"]].reduction[-1] + 1): item["cycle"]
                               for item in saved}
                for item in reads:
                    packet = packets[item["tag"]["packet_id"]]
                    key = (packet.tile_id, item["tag"]["token"], packet.reduction[0])
                    self.assertGreaterEqual(item["cycle"], save_cycles[key])
                self.assertGreater(result["buffers"]["partial_sums"]["peak_live_bytes"], 0)

    def test_b1_group_reuse_can_delay_first_row_while_reducing_total_time(self):
        shape = Shape(9, 17, 129)
        b0, _ = simulate(load_config(), "B0", shape=shape, capture_trace=False)
        b1, _ = simulate(load_config(), "B1_G9", shape=shape, capture_trace=False)
        self.assertGreater(b1["first_matrix_row"], b0["first_matrix_row"])
        self.assertGreater(b1["first_row_cycles"], b0["first_row_cycles"])
        self.assertLess(b1["total_cycles"], b0["total_cycles"])
        self.assertEqual(b0["weight_port"]["useful_weight_bytes"],
                         b1["weight_port"]["useful_weight_bytes"] * 9)

    def test_configured_feedback_and_multiply_latency_preserve_all_products(self):
        config = load_config(overrides={"architecture": {
            "accumulator_feedback": 7, "multiply_latency": 11}})
        shape = Shape(3, 3, 129)
        for mapping in MAPPINGS:
            with self.subTest(mapping=mapping):
                result, trace = simulate(config, mapping, shape=shape)
                self.assert_complete(result, trace, shape, mapping)
                for item in events(trace, "COMPUTE_ISSUE"):
                    self.assertEqual(item["complete_cycle"], item["cycle"] + 11)
                # For one token/output pair or one array output tile, reduction
                # continuations cannot bypass the configured feedback gate.
                packets = {p.packet_id: p for p in schedule_packets(shape, mapping)}
                last_issue = {}
                for item in events(trace, "COMPUTE_ISSUE"):
                    packet = packets[item["packet_id"]]
                    key = (packet.tile_id, item.get("token") if mapping != "A" else None)
                    if not packet.first_reduction:
                        self.assertGreaterEqual(item["cycle"] - last_issue[key], 7)
                    last_issue[key] = item["cycle"]

    def test_partial_capacity_live_bytes_do_not_release_reserved_space(self):
        capacity = Capacity("partial", 20)
        capacity.reserve("a", 12)
        capacity.reserve("b", 8)
        capacity.set_live("a", 4)
        capacity.set_live("b", 3)
        self.assertEqual((capacity.live, capacity.reserved), (7, 20))
        self.assertFalse(capacity.can_reserve(1))
        capacity.set_live("a", 2)
        self.assertEqual((capacity.live, capacity.reserved), (5, 20))
        capacity.ready("a")
        self.assertEqual((capacity.live, capacity.reserved), (15, 20))
        capacity.release("a")
        self.assertEqual((capacity.live, capacity.reserved), (3, 8))
        with self.assertRaises(AssertionError):
            capacity.set_live("b", 9)
        self.assertEqual((capacity.live, capacity.reserved), (3, 8))
        with self.assertRaises(AssertionError):
            capacity.reserve("overflow", 13)
        capacity.release("b")
        self.assertEqual(capacity.stats(), {
            "capacity_bytes": 20, "peak_reserved_bytes": 20,
            "peak_live_bytes": 15, "final_reserved_bytes": 0})

    def test_event_boundaries_apply_phases_then_stable_sequence_order(self):
        queue = EventQueue()
        observed = []
        def collect(event):
            observed.append((event.cycle, event.phase, event.kind))
        queue.schedule(5, 3, "later_phase", None, collect, phase=1)
        queue.schedule(5, 3, "first", None, collect)
        queue.schedule(5, 3, "second", None, collect)
        queue.complete(7)
        self.assertEqual(observed, [])
        self.assertTrue(queue.pending)
        queue.complete(8)
        self.assertEqual(observed, [(8, 0, "first"), (8, 0, "second"),
                                    (8, 1, "later_phase")])
        self.assertFalse(queue.pending)
        with self.assertRaises(ValueError):
            queue.schedule(8, 0, "zero", None, collect)


if __name__ == "__main__":
    unittest.main()
