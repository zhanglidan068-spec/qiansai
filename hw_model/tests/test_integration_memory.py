"""Independent memory/lifetime audit of the fixed 300 MB/s contract."""
from collections import Counter, defaultdict
import unittest
from unittest.mock import patch

from hw_model.compute import Shape
from hw_model.config import load_config
from hw_model.fixed_memory import WeightPort
from hw_model.fixed_schedule import schedule_packets
from hw_model.fixed_simulation import FusedSimulator


MAPPINGS = ("A", "B0", "B1_G9", "B1_G99")


def ceildiv(value, divisor):
    return (value + divisor - 1) // divisor


def byte_addresses(event):
    base = (event["word_index"] * 8 + event["bank"]) * 2
    return [base + lane for lane in (0, 1) if event["byte_mask"] & (1 << lane)]


class IntegrationMemoryTests(unittest.TestCase):
    def _run(self, mapping, shape):
        simulator = FusedSimulator(load_config(), mapping, shape=shape)
        return simulator, simulator.run(max_cycles=100000)

    def _check_weight_ledger(self, simulator, result):
        shape, mapping = simulator.shape, simulator.mapping
        group = {"A": 9, "B0": 1, "B1_G9": 9, "B1_G99": 99}[mapping]
        groups = ceildiv(shape.m, group)
        columns, reduction = (16, 1) if mapping == "A" else (2, 64)
        packet_count = groups * ceildiv(shape.n, columns) * ceildiv(shape.r, reduction)
        physical = sum(ceildiv(min(columns, shape.n-c) * min(reduction, shape.r-k), 4) * 4
                       for c in range(0, shape.n, columns) for k in range(0, shape.r, reduction))
        weight = result["weight_port"]
        self.assertEqual(weight["useful_weight_bytes"], groups * shape.n * shape.r)
        self.assertEqual(weight["physical_weight_bytes"], groups * physical)
        self.assertEqual(weight["fill_count"], packet_count)
        self.assertEqual(weight["released_packets"], packet_count)
        self.assertEqual(weight["accepted_beats"], weight["completed_beats"])
        self.assertEqual(weight["accepted_beats"] * 4, weight["physical_weight_bytes"])
        self.assertEqual(weight["occupied_service_cycles"], weight["accepted_beats"] * 2)
        self.assertEqual(weight["bus_busy_cycles"], weight["accepted_beats"])
        for key in ("reserved_bytes", "active_packets", "outstanding_beats"):
            self.assertEqual(weight[key], 0)
        self.assertLessEqual(weight["max_outstanding"], 2)
        self.assertLessEqual(weight["peak_reserved_bytes"], 32 if mapping == "A" else 256)
        self.assertLessEqual(weight["peak_live_bytes"], weight["peak_reserved_bytes"])
        consumers = packet_count if mapping == "A" else shape.m * ceildiv(shape.n, 2) * ceildiv(shape.r, 64)
        self.assertEqual(weight["consumers_completed"], consumers)
        self.assertEqual(weight["reuse_count"], consumers - packet_count)
        self.assertEqual(result["shared_activation_read_bytes"], 0)
        self.assertEqual(result["shared_activation_write_bytes"], 0)

    def _check_packet_lifetimes(self, simulator):
        reserved, ready, releases, active = {}, {}, {}, {}
        masks, issues, consumes, completions = defaultdict(int), defaultdict(list), defaultdict(list), defaultdict(list)
        bus_cycles, lifetimes = [], []
        for event in simulator.trace:
            if event["event"] == "COMPUTE_ISSUE":
                issues[event["packet_id"]].append(event)
            elif event["event"] == "MAC_COMPLETE":
                completions[event["packet_id"]].append(event)
        for event in simulator.weights.trace:
            kind, packet_id = event["event"], event.get("packet_id")
            if kind == "PACKET_RESERVED":
                self.assertNotIn(packet_id, reserved)
                self.assertNotIn(event["slot"], active)
                active[event["slot"]] = packet_id
                self.assertLessEqual(len(active), 2)
                reserved[packet_id] = event
            elif kind == "WEIGHT_BEAT_ISSUE":
                self.assertEqual(active[event["slot"]], packet_id)
                self.assertEqual(event["address"] % 4, 0)
                self.assertEqual(event["complete_cycle"], event["cycle"] + 3)
                self.assertTrue(1 <= event["valid_bytes"] <= 4)
                bus_cycles.append(event["cycle"])
                lifetimes.append((event["cycle"], 1, 1))
            elif kind == "WEIGHT_BEAT_DONE":
                self.assertFalse(masks[packet_id] & event["packet_byte_mask"])
                masks[packet_id] |= event["packet_byte_mask"]
                self.assertEqual(event["valid_mask"], masks[packet_id])
                lifetimes.append((event["cycle"], 0, -1))
            elif kind == "PACKET_READY":
                self.assertEqual(masks[packet_id], (1 << reserved[packet_id]["useful_bytes"]) - 1)
                ready[packet_id] = event["cycle"]
            elif kind == "PACKET_CONSUME":
                consumes[packet_id].append(event["cycle"])
            elif kind == "PACKET_RELEASE":
                self.assertEqual(active.pop(event["slot"]), packet_id)
                releases[packet_id] = event["cycle"]
        self.assertEqual(active, {})
        self.assertTrue(all(b-a >= 2 for a,b in zip(bus_cycles,bus_cycles[1:])))
        live = 0
        for _, _, change in sorted(lifetimes):
            live += change
            self.assertTrue(0 <= live <= 2)
        self.assertEqual(live, 0)
        self.assertEqual(set(reserved), set(ready))
        self.assertEqual(set(reserved), set(releases))
        for packet_id, reservation in reserved.items():
            self.assertEqual(len(issues[packet_id]), reservation["consumers"])
            self.assertEqual(len(completions[packet_id]), reservation["consumers"])
            self.assertEqual(sorted(consumes[packet_id]), sorted(e["cycle"] for e in completions[packet_id]))
            self.assertEqual(releases[packet_id], max(consumes[packet_id]))
            self.assertEqual({(tuple(e["rows"]),e["complete_cycle"]) for e in issues[packet_id]},
                             {(tuple(e["rows"]),e["cycle"]) for e in completions[packet_id]})
            self.assertTrue(all(e["cycle"] >= ready[packet_id] for e in issues[packet_id]))

    def _check_local_ledger(self, simulator, result):
        shape, mapping = simulator.shape, simulator.mapping
        matrix = shape.m * shape.r
        if mapping == "A":
            matrix *= ceildiv(shape.n, 16)
        elif mapping.startswith("B1"):
            matrix *= ceildiv(shape.n, 2)
        expected = {"matrix_activation": (matrix, 0), "post_residual": (shape.m*shape.n, 0),
                    "post_pre_ln": (0, 2*shape.m*shape.n), "ln_input": (2*shape.m*shape.n, 0),
                    "ln_output": (0, shape.m*shape.n)}
        partial = 4 * shape.m * shape.n * (ceildiv(shape.r, 64)-1)
        if mapping.startswith("B1") and partial:
            expected.update(partial_read=(partial,0),partial_write=(0,partial))
        owners = result["local_memory"]["per_owner"]
        self.assertEqual(set(owners), set(expected))
        physical, bank_cycles = defaultdict(Counter), Counter()
        for event in simulator.local.trace:
            if event["event"] not in ("LOCAL_READ", "LOCAL_WRITE"):
                continue
            self.assertTrue(0 <= event["bank"] < 8 and 0 <= event["word_index"] < 4096)
            self.assertIn(event["byte_mask"], (1,2,3))
            self.assertEqual(event["complete_cycle"], event["cycle"]+1)
            bank_cycles[(event["cycle"],event["bank"])] += 1
            direction = "write" if event["write"] else "read"
            physical[event["owner"]]["physical_"+direction+"_words"] += 1
            physical[event["owner"]][direction+"_useful_bytes"] += len(byte_addresses(event))
        self.assertTrue(all(count == 1 for count in bank_cycles.values()))
        self.assertEqual({owner:(s["read_useful_bytes"],s["write_useful_bytes"]) for owner,s in owners.items()}, expected)
        for owner, stats in owners.items():
            for key in ("physical_read_words","physical_write_words","read_useful_bytes","write_useful_bytes"):
                self.assertEqual(stats[key], physical[owner][key])
        local = result["local_memory"]
        self.assertEqual(local["completed_requests"], local["submitted_requests"])
        self.assertEqual(local["pending_requests"], 0)
        self.assertEqual(local["outstanding_words"], 0)
        self.assertLessEqual(local["peak_outstanding_words"], 8)
        self.assertLessEqual(local["peak_pending_requests"], 8)

    def _check_initialized_reads(self, simulator):
        # Initial input/residual bytes are resident. All other valid bytes must
        # originate from an acknowledged physical SRAM write inside the window.
        valid = {}
        for tensor in ("input","residual"):
            desc = simulator.layout.tensors[tensor]
            for row in range(desc["rows"]):
                for col in range(desc["cols"]):
                    valid[simulator.layout.address(tensor,row,col)] = (tensor,None,row)
        packets = {e["packet_id"]:e["tag"] for e in simulator.weights.trace if e["event"] == "PACKET_RESERVED"}
        for event in simulator.local.trace:
            tag = event.get("tag",{})
            tensor = tag.get("tensor")
            if event["event"] == "LOCAL_WORD_DONE" and event["write"]:
                for address in byte_addresses(event):
                    valid[address] = (tensor,tag.get("packet_id"),tag.get("token"))
            elif event["event"] == "LOCAL_READ":
                for address in byte_addresses(event):
                    self.assertIn(address,valid,"read before initialization/write ACK")
                    origin, previous_packet, previous_token = valid[address]
                    self.assertEqual(origin,tensor)
                    if tensor == "partial":
                        current,previous = packets[tag["packet_id"]],packets[previous_packet]
                        self.assertEqual(previous["tile_id"],current["tile_id"])
                        self.assertEqual(previous_token,tag["token"])
                        self.assertEqual(previous["reduction"][0]+64,current["reduction"][0])

    def _check_ln_and_buffers(self, simulator, result):
        shape = simulator.shape
        written, input_ready, assembled, accepted, outputs = {}, {}, {}, {}, {}
        masks = defaultdict(int)
        for event in simulator.trace:
            kind,row,cycle = event["event"],event.get("token"),event["cycle"]
            if kind == "POST_WRITE_DONE":
                mask = sum(1 << c for c in event["channels"])
                self.assertFalse(masks[row]&mask)
                masks[row] |= mask
                written[row] = cycle
            elif kind == "LN_ROW_ASSEMBLED":
                self.assertEqual(masks[row],(1 << shape.n)-1)
                self.assertEqual(cycle,written[row])
                assembled[row] = cycle
            elif kind == "LN_INPUT_READY":
                self.assertGreaterEqual(cycle,assembled[row]+1)
                input_ready[row] = cycle
            elif kind == "LN_ACCEPT":
                self.assertGreaterEqual(cycle,input_ready[row])
                self.assertEqual(event["complete_cycle"],cycle+128)
                accepted[row] = cycle
            elif kind == "LN_CORE_DONE":
                self.assertEqual(cycle,accepted[row]+128)
            elif kind == "INTERNAL_OUTPUT_CONFIRMED":
                self.assertGreaterEqual(cycle,accepted[row]+129)
                outputs[row] = cycle
        self.assertEqual(set(outputs),set(range(shape.m)))
        starts = sorted(accepted.values())
        self.assertTrue(all(b-a >= 128 for a,b in zip(starts,starts[1:])))
        self.assertEqual(result["total_cycles"],max(outputs.values()))
        for name,stats in result["buffers"].items():
            self.assertLessEqual(stats["peak_live_bytes"],stats["peak_reserved_bytes"])
            self.assertLessEqual(stats["peak_reserved_bytes"],stats["capacity_bytes"])
            self.assertEqual(stats["final_reserved_bytes"],shape.m*shape.n if name == "resident_output" else 0)
        self.assertLessEqual(result["peak_operand_packets"],2)
        self.assertLessEqual(result["peak_post_jobs"],2)
        self.assertFalse(simulator.weights.pending or simulator.local.pending or simulator.events.pending)
        self.assertEqual(result["physical_local_layout"]["capacity_bytes"],65536)
        self.assertLessEqual(result["physical_local_layout"]["used_bytes"],65536)
        resources = result["resources"]
        self.assertEqual(resources["weight_sram_bytes"],0)
        self.assertEqual(resources["macro_instances"],{"4096_single":8})
        self.assertEqual(resources["register_payload_bytes"]["partial_operand_register_bytes"],16 if simulator.b1 else 0)
        self.assertFalse(result["initialization"]["included_in_main_window"])
        self.assertIsNone(result["initialization"]["external_activation_load_cycles"])

    def test_all_mappings_tail_masks_multigroup_and_partial_reduction_chunks(self):
        for mapping in MAPPINGS:
            for shape in (Shape(1,1,1),Shape(3,5,7),Shape(10,17,65),Shape(3,5,256)):
                with self.subTest(mapping=mapping,shape=shape):
                    simulator,result = self._run(mapping,shape)
                    self._check_weight_ledger(simulator,result)
                    self._check_packet_lifetimes(simulator)
                    self._check_local_ledger(simulator,result)
                    self._check_initialized_reads(simulator)
                    self._check_ln_and_buffers(simulator,result)

    def test_full_operator_weight_ledger_matches_explicit_reference_counts(self):
        expected = {"A":(45056,180224),"B0":(405504,1622016),"B1_G9":(45056,180224),"B1_G99":(4096,16384)}
        for mapping,counts in expected.items():
            for reduction,reference in zip((64,256),counts):
                with self.subTest(mapping=mapping,reduction=reduction):
                    self.assertEqual(sum(p.weight_bytes for p in schedule_packets(Shape(99,64,reduction),mapping)),reference)

    def test_capacity_failure_precedes_any_weight_register_creation(self):
        for mapping,shape in (("A",Shape(129,64,256)),("B0",Shape(129,64,256)),
                              ("B1_G9",Shape(128,64,256)),("B1_G99",Shape(128,64,256)),("A",Shape(1,8193,64))):
            with self.subTest(mapping=mapping,shape=shape):
                with patch("hw_model.fixed_simulation.WeightPort",wraps=WeightPort) as port:
                    with self.assertRaises(ValueError):
                        FusedSimulator(load_config(),mapping,shape=shape)
                    port.assert_not_called()


if __name__ == "__main__":
    unittest.main()
