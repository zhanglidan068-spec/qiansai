"""Independent numerical and physical-contract checks for optional mappings.

Expected geometry and traffic are specified here independently of mapping_traits.
No performance-cycle result is frozen: these tests validate work, data lifetime,
physical capacity, port service and deterministic scheduling instead.
"""
from collections import Counter, defaultdict
import random
import unittest
from unittest.mock import patch

from hw_model.compute import Shape
from hw_model.config import load_config
from hw_model.fixed_memory import LocalLayout, WeightPort
from hw_model.fixed_schedule import (
    ALL_MAPPINGS, EXPERIMENTAL_MAPPINGS, MAPPINGS, mapping_traits,
    schedule_packets, weight_byte_offset,
)
from hw_model.fixed_simulation import FusedSimulator


# token group, output columns, reduction chunk, multiplier count
GEOMETRY = {"A18x8": (18, 8, 1, 144), "A16x8": (16, 8, 1, 128),
            "A32x4": (32, 4, 1, 128), "B1_G11": (11, 2, 64, 128)}


def ceildiv(value, divisor):
    return (value + divisor - 1) // divisor


def addresses(event):
    base = (event["word_index"] * 8 + event["bank"]) * 2
    return [base + lane for lane in (0, 1) if event["byte_mask"] & (1 << lane)]


class ExperimentalArchitectureTests(unittest.TestCase):
    def _run(self, mapping, shape, trace=True):
        sim = FusedSimulator(load_config(), mapping, shape=shape, capture_trace=trace)
        return sim, sim.run(max_cycles=300000)

    def _ledger_and_resources(self, mapping, shape, result):
        group, columns, reduction, multipliers = GEOMETRY[mapping]
        array = mapping.startswith("A")
        groups = ceildiv(shape.m, group)
        col_tiles, red_tiles = ceildiv(shape.n, columns), ceildiv(shape.r, reduction)
        packet_count = groups * col_tiles * red_tiles
        issues = packet_count if array else shape.m * col_tiles * red_tiles
        physical_per_group = sum(ceildiv(min(columns, shape.n-c) * min(reduction, shape.r-k), 4) * 4
                                 for c in range(0, shape.n, columns) for k in range(0, shape.r, reduction))
        weight = result["weight_port"]
        self.assertEqual(result["useful_mac"], shape.m * shape.n * shape.r)
        self.assertEqual(result["issued_updates"], issues)
        self.assertEqual(result["physical_multipliers"], multipliers)
        self.assertEqual(result["issue_lane_utilization"], shape.useful_mac / (multipliers * issues))
        self.assertEqual(weight["useful_weight_bytes"], groups * shape.n * shape.r)
        self.assertEqual(weight["physical_weight_bytes"], groups * physical_per_group)
        self.assertEqual(weight["fill_count"], packet_count)
        self.assertEqual(weight["consumers_completed"], issues)
        self.assertEqual(weight["released_packets"], packet_count)
        self.assertEqual(weight["reuse_count"], issues - packet_count)
        self.assertEqual(weight["physical_weight_bytes"], 4 * weight["accepted_beats"])
        self.assertEqual(weight["occupied_service_cycles"], 2 * weight["accepted_beats"])
        self.assertEqual(weight["accepted_beats"], weight["completed_beats"])
        self.assertLessEqual(weight["max_outstanding"], 2)
        self.assertEqual(weight["active_packets"], 0)
        self.assertEqual(weight["outstanding_beats"], 0)
        expected_owners = {"matrix_activation": (shape.m * shape.r * col_tiles, 0),
                           "post_residual": (shape.m * shape.n, 0),
                           "post_pre_ln": (0, 2 * shape.m * shape.n),
                           "ln_input": (2 * shape.m * shape.n, 0),
                           "ln_output": (0, shape.m * shape.n)}
        partial = 4 * shape.m * shape.n * (red_tiles - 1)
        if not array and partial:
            expected_owners.update(partial_read=(partial, 0), partial_write=(0, partial))
        actual = {owner: (stats["read_useful_bytes"], stats["write_useful_bytes"])
                  for owner, stats in result["local_memory"]["per_owner"].items()}
        self.assertEqual(actual, expected_owners)
        self.assertEqual(result["shared_activation_read_bytes"], 0)
        self.assertEqual(result["shared_activation_write_bytes"], 0)
        resources = result["resources"]
        regs = resources["register_payload_bytes"]
        self.assertEqual(resources["multipliers"], multipliers)
        self.assertEqual(resources["macro_instances"], {"4096_single": 8})
        self.assertEqual(resources["local_sram_bytes"], 65536)
        self.assertEqual(resources["weight_sram_bytes"], 0)
        self.assertEqual(regs["weight_register_bytes"], 2 * columns if array else 256)
        self.assertEqual(regs["activation_register_bytes"], 2 * group if array else 128)
        self.assertEqual(regs["accumulator_register_bytes"], group * columns * 4 if array else 8)
        self.assertEqual(regs["partial_operand_register_bytes"], 0 if array else 16)
        self.assertEqual(resources["b1_partial_sram_bytes"], 0 if array else 8 * min(group, shape.m))
        layout = result["physical_local_layout"]
        input_bytes = shape.r * ceildiv(shape.m, 2) * 2 if array else ceildiv(shape.m * shape.r, 2) * 2
        used = input_bytes + 2 * ceildiv(shape.m * shape.n, 2) * 2 + 2 * shape.m * shape.n
        if not array:
            used += 8 * min(group, shape.m)
        self.assertEqual(layout["used_bytes"], used)
        self.assertLessEqual(used, 65536)
        for name, stats in result["buffers"].items():
            self.assertTrue(0 <= stats["peak_live_bytes"] <= stats["peak_reserved_bytes"] <= stats["capacity_bytes"])
            self.assertEqual(stats["final_reserved_bytes"], shape.m * shape.n if name == "resident_output" else 0)
        self.assertEqual(result["buffers"]["activation_registers"]["capacity_bytes"], regs["activation_register_bytes"])
        self.assertEqual(result["buffers"]["accumulators"]["capacity_bytes"], regs["accumulator_register_bytes"])
        self.assertLessEqual(result["peak_operand_packets"], 2)
        self.assertLessEqual(result["peak_post_jobs"], 2)
        self.assertEqual(result["local_memory"]["pending_requests"], 0)
        self.assertEqual(result["local_memory"]["outstanding_words"], 0)
        self.assertEqual(sum(result["cycle_breakdown"].values()), result["total_cycles"])

    def _trace_contract(self, sim, shape):
        bank_uses, physical = Counter(), defaultdict(Counter)
        valid = {}
        for tensor in ("input", "residual"):
            desc = sim.layout.tensors[tensor]
            for row in range(desc["rows"]):
                for col in range(desc["cols"]):
                    valid[sim.layout.address(tensor, row, col)] = tensor
        for event in sim.local.trace:
            if event["event"] == "LOCAL_WORD_DONE" and event["write"]:
                for address in addresses(event):
                    valid[address] = event["tag"]["tensor"]
            if event["event"] not in ("LOCAL_READ", "LOCAL_WRITE"):
                continue
            self.assertTrue(0 <= event["bank"] < 8 and 0 <= event["word_index"] < 4096)
            bank_uses[(event["cycle"], event["bank"])] += 1
            self.assertEqual(event["complete_cycle"], event["cycle"] + 1)
            if not event["write"]:
                for address in addresses(event):
                    self.assertEqual(valid.get(address), event["tag"]["tensor"], "uninitialized/stale SRAM read")
            direction = "write" if event["write"] else "read"
            physical[event["owner"]]["physical_" + direction + "_words"] += 1
        self.assertTrue(all(count == 1 for count in bank_uses.values()))
        for owner, stats in sim.local.stats()["per_owner"].items():
            self.assertEqual(stats["physical_read_words"], physical[owner]["physical_read_words"])
            self.assertEqual(stats["physical_write_words"], physical[owner]["physical_write_words"])
        ready, consumer_done, released, reserved, active = {}, defaultdict(list), {}, {}, {}
        for event in sim.weights.trace:
            packet_id = event.get("packet_id")
            if event["event"] == "PACKET_RESERVED":
                self.assertNotIn(event["slot"], active)
                active[event["slot"]] = packet_id
                reserved[packet_id] = event
                self.assertLessEqual(len(active), 2)
            elif event["event"] == "PACKET_READY":
                self.assertEqual(event["valid_mask"], (1 << reserved[packet_id]["useful_bytes"]) - 1)
                ready[packet_id] = event["cycle"]
            elif event["event"] == "PACKET_CONSUME":
                consumer_done[packet_id].extend([event["cycle"]] * event["amount"])
            elif event["event"] == "PACKET_RELEASE":
                self.assertEqual(active.pop(event["slot"]), packet_id)
                released[packet_id] = event["cycle"]
        self.assertEqual(active, {})
        mac_done = defaultdict(list)
        matrix_ready, post_written, ln_input, ln_accept = {}, defaultdict(int), {}, {}
        output_rows = set()
        for event in sim.trace:
            kind, cycle = event["event"], event["cycle"]
            if kind == "COMPUTE_ISSUE":
                self.assertGreaterEqual(cycle, ready[event["packet_id"]])
                self.assertLessEqual(event["useful_mac"], GEOMETRY[sim.mapping][3])
                self.assertTrue(all(0 <= row < shape.m for row in event["rows"]))
                self.assertTrue(all(0 <= col < shape.n for col in event["columns"]))
            elif kind == "MAC_COMPLETE":
                mac_done[event["packet_id"]].append(cycle)
            elif kind == "MAT_FRAGMENT_READY":
                for col in event["channels"]:
                    key = (event["token"], col)
                    self.assertNotIn(key, matrix_ready)
                    matrix_ready[key] = cycle
            elif kind == "POST_ISSUE":
                self.assertTrue(all(matrix_ready[(event["token"], col)] <= cycle for col in event["channels"]))
            elif kind == "POST_WRITE_DONE":
                row = event["token"]
                mask = sum(1 << col for col in event["channels"])
                self.assertFalse(post_written[row] & mask)
                post_written[row] |= mask
            elif kind == "LN_INPUT_READY":
                self.assertEqual(post_written[event["token"]], (1 << shape.n) - 1)
                ln_input[event["token"]] = cycle
            elif kind == "LN_ACCEPT":
                self.assertGreaterEqual(cycle, ln_input[event["token"]])
                ln_accept[event["token"]] = cycle
                self.assertEqual(event["complete_cycle"], cycle + 128)
            elif kind == "LN_CORE_DONE":
                self.assertEqual(cycle, ln_accept[event["token"]] + 128)
            elif kind == "INTERNAL_OUTPUT_CONFIRMED":
                self.assertGreaterEqual(cycle, ln_accept[event["token"]] + 129)
                output_rows.add(event["token"])
        self.assertEqual(len(matrix_ready), shape.m * shape.n)
        self.assertEqual(output_rows, set(range(shape.m)))
        for packet_id, reservation in reserved.items():
            self.assertEqual(len(mac_done[packet_id]), reservation["consumers"])
            self.assertEqual(sorted(consumer_done[packet_id]), sorted(mac_done[packet_id]))
            self.assertEqual(released[packet_id], max(mac_done[packet_id]))

    def test_experimental_opt_in_does_not_change_original_mapping_set(self):
        self.assertEqual(MAPPINGS, ("A", "B0", "B1_G9", "B1_G99"))
        self.assertEqual(set(EXPERIMENTAL_MAPPINGS), set(GEOMETRY))
        # Later opt-in studies may add candidates, while these two established
        # comparison sets and their ordering remain unchanged.
        established = MAPPINGS + EXPERIMENTAL_MAPPINGS
        self.assertEqual(ALL_MAPPINGS[:len(established)], established)
        self.assertEqual(len(ALL_MAPPINGS), len(set(ALL_MAPPINGS)))
        for mapping, (rows, columns, reduction, multipliers) in GEOMETRY.items():
            traits = mapping_traits(Shape(99, 64, 256), mapping)
            self.assertEqual((traits["group_rows"], traits["column_lanes"], traits["reduction_lanes"],
                              traits["physical_multipliers"]), (rows, columns, reduction, multipliers))

    def test_random_int8_packed_schedule_equals_independent_integer_gemm(self):
        random_values = random.Random(20261029)
        for mapping, (group, columns, _, _) in GEOMETRY.items():
            for shape in (Shape(group + 3, columns + 3, 65), Shape(3, 5, 129)):
                with self.subTest(mapping=mapping, shape=shape):
                    left = [[random_values.randint(-128, 127) for _ in range(shape.r)] for _ in range(shape.m)]
                    right = [[random_values.randint(-128, 127) for _ in range(shape.n)] for _ in range(shape.r)]
                    expected = [[sum(left[i][k] * right[k][j] for k in range(shape.r))
                                 for j in range(shape.n)] for i in range(shape.m)]
                    traits = mapping_traits(shape, mapping)
                    layout = LocalLayout(shape, mapping, group)
                    local = {layout.address("input", i, k): left[i][k]
                             for i in range(shape.m) for k in range(shape.r)}
                    self.assertEqual(len(local), shape.m * shape.r)
                    packed = [None] * traits["weight_storage_bytes"]
                    for k in range(shape.r):
                        for j in range(shape.n):
                            address = weight_byte_offset(shape, mapping, k, j)
                            self.assertIsNone(packed[address], "weight values collide in packed source")
                            packed[address] = right[k][j]
                    actual, coverage = [[0] * shape.n for _ in range(shape.m)], [[0] * shape.n for _ in range(shape.m)]
                    mac_count = 0
                    for packet in schedule_packets(shape, mapping):
                        block = packed[packet.weight_address:packet.weight_address + packet.weight_bytes]
                        self.assertEqual(len(block), packet.weight_bytes)
                        self.assertNotIn(None, block)
                        self.assertEqual(packet.useful_mac, len(packet.rows) * len(packet.columns) * len(packet.reduction))
                        mac_count += packet.useful_mac
                        for row in packet.rows:
                            for cj, col in enumerate(packet.columns):
                                for kk, k in enumerate(packet.reduction):
                                    self.assertFalse(coverage[row][col] & (1 << k), "duplicate useful product")
                                    coverage[row][col] |= 1 << k
                                    actual[row][col] += local[layout.address("input", row, k)] * block[kk * len(packet.columns) + cj]
                    self.assertEqual(actual, expected)
                    self.assertEqual(mac_count, shape.useful_mac)
                    self.assertTrue(all(mask == (1 << shape.r) - 1 for row in coverage for mask in row))

    def test_tail_and_complete_64_channel_traces_obey_fixed_memory_contract(self):
        for mapping, (group, columns, _, _) in GEOMETRY.items():
            for shape in (Shape(group + 3, columns + 3, 65), Shape(3, 64, 129)):
                with self.subTest(mapping=mapping, shape=shape):
                    sim, result = self._run(mapping, shape)
                    self._ledger_and_resources(mapping, shape, result)
                    self._trace_contract(sim, shape)

    def test_full_99_token_projection_and_ffn2_traffic_resources(self):
        for mapping in GEOMETRY:
            for reduction in (64, 256):
                with self.subTest(mapping=mapping, reduction=reduction):
                    _, result = self._run(mapping, Shape(99, 64, reduction), trace=False)
                    self._ledger_and_resources(mapping, Shape(99, 64, reduction), result)

    def test_small_trace_replay_is_deterministic(self):
        for mapping in GEOMETRY:
            with self.subTest(mapping=mapping):
                first, first_result = self._run(mapping, Shape(3, 9, 65))
                second, second_result = self._run(mapping, Shape(3, 9, 65))
                self.assertEqual(first_result, second_result)
                self.assertEqual(first.merged_trace(), second.merged_trace())

    def test_capacity_rejection_precedes_weight_packet_allocation(self):
        for mapping in GEOMETRY:
            with self.subTest(mapping=mapping):
                shape = Shape(128 if mapping == "B1_G11" else 129, 64, 256)
                with patch("hw_model.fixed_simulation.WeightPort", wraps=WeightPort) as port:
                    with self.assertRaises(ValueError):
                        FusedSimulator(load_config(), mapping, shape=shape)
                    port.assert_not_called()


if __name__ == "__main__":
    unittest.main()
