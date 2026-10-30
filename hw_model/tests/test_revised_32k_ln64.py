"""Independent audit of the explicitly selected 4-bank/LN64 revision.

Strict tensor residency is preserved: no spilling, tensor aliasing, token count
reduction, or changed measurement boundary is introduced by these tests.
"""
from collections import Counter, defaultdict
import random
import unittest
from unittest.mock import patch

from hw_model.compute import Shape
from hw_model.config import REVISED_CONFIG, REVISED_PROFILE_ID, load_config
from hw_model.fixed_memory import LocalCapacityError, LocalLayout, WeightPort
from hw_model.fixed_schedule import (
    ALL_MAPPINGS, MAPPINGS, EXPERIMENTAL_MAPPINGS, REVISION_MAPPINGS,
    schedule_packets, weight_byte_offset,
)
from hw_model.fixed_simulation import FusedSimulator


REVISION_GEOMETRY = {"A33x4": (33,4), "A33x8": (33,8), "A32x8": (32,8),
                     "A36x4": (36,4), "A17x4": (17,4), "A33x2": (33,2),
                     "A9x8": (9,8), "A11x8": (11,8), "A3x16": (3,16),
                     "A3x32": (3,32), "A33x16": (33,16), "A17x8": (17,8)}
ARRAY_GEOMETRY = {"A": (9,16), "A18x8": (18,8), "A16x8": (16,8), "A32x4": (32,4)}
ARRAY_GEOMETRY.update(REVISION_GEOMETRY)
ROW_GROUPS = {"B0": 1, "B1_G9": 9, "B1_G11": 11, "B1_G99": 99}
STUDY_MAPPINGS = MAPPINGS + EXPERIMENTAL_MAPPINGS + REVISION_MAPPINGS


def ceildiv(value, divisor):
    return (value + divisor - 1) // divisor


def required_bytes(shape, mapping):
    array = mapping in ARRAY_GEOMETRY
    input_bytes = 2 * ceildiv(shape.m,2) * shape.r if array else 2 * ceildiv(shape.m*shape.r,2)
    common = input_bytes + 4 * ceildiv(shape.m*shape.n,2) + 2*shape.m*shape.n
    return common + (8*min(shape.m,ROW_GROUPS[mapping]) if mapping.startswith("B1") else 0)


def physical_addresses(event):
    base = (event["word_index"] * 4 + event["bank"]) * 2
    return [base+lane for lane in (0,1) if event["byte_mask"] & (1 << lane)]


class Revised32kLn64Tests(unittest.TestCase):
    def _config(self):
        return load_config(REVISED_CONFIG)

    def _run(self, mapping, shape, trace=False):
        sim = FusedSimulator(self._config(),mapping,shape=shape,capture_trace=trace)
        return sim,sim.run(max_cycles=400000)

    def _check_projection(self,mapping,result):
        shape = Shape(99,64,64)
        array = mapping in ARRAY_GEOMETRY
        group,cols = ARRAY_GEOMETRY[mapping] if array else (ROW_GROUPS[mapping],2)
        groups = ceildiv(99,group)
        reduction = 1 if array else 64
        packets = groups*ceildiv(64,cols)*ceildiv(64,reduction)
        issue_count = packets if array else 99*32
        multipliers = group*cols if array else 128
        physical_per_group = sum(4*ceildiv(min(cols,64-c)*min(reduction,64-k),4)
                                 for c in range(0,64,cols) for k in range(0,64,reduction))
        weight = result["weight_port"]
        self.assertEqual(weight["useful_weight_bytes"],groups*4096)
        self.assertEqual(weight["physical_weight_bytes"],groups*physical_per_group)
        self.assertEqual(weight["accepted_beats"]*4,weight["physical_weight_bytes"])
        self.assertEqual(weight["accepted_beats"],weight["completed_beats"])
        self.assertEqual(weight["occupied_service_cycles"],2*weight["accepted_beats"])
        self.assertEqual(weight["fill_count"],packets)
        self.assertEqual(weight["consumers_completed"],issue_count)
        self.assertEqual(weight["released_packets"],packets)
        self.assertEqual(weight["active_packets"],0)
        self.assertEqual(weight["outstanding_beats"],0)
        self.assertLessEqual(weight["max_outstanding"],2)
        if mapping == "A33x2":
            self.assertEqual(weight["useful_weight_bytes"],12288)
            self.assertEqual(weight["physical_weight_bytes"],24576)
        self.assertEqual(result["useful_mac"],405504)
        self.assertEqual(result["issued_updates"],issue_count)
        self.assertEqual(result["physical_multipliers"],multipliers)
        self.assertEqual(result["issue_lane_utilization"],405504/(multipliers*issue_count))
        expected_input = 6336 if mapping == "B0" else 6336*ceildiv(64,cols)
        expected = {"matrix_activation":(expected_input,0),"post_residual":(6336,0),
                    "post_pre_ln":(0,12672),"ln_input":(12672,0),"ln_output":(0,6336)}
        self.assertEqual({owner:(stats["read_useful_bytes"],stats["write_useful_bytes"])
                          for owner,stats in result["local_memory"]["per_owner"].items()},expected)
        self.assertEqual(result["physical_local_layout"]["used_bytes"],required_bytes(shape,mapping))
        self.assertEqual(result["physical_local_layout"]["capacity_bytes"],32768)
        local,resources = result["local_memory"],result["resources"]
        self.assertEqual(local["banks"],4)
        self.assertEqual(local["capacity_bytes"],32768)
        self.assertEqual(local["pending_requests"],0)
        self.assertEqual(local["outstanding_words"],0)
        self.assertLessEqual(local["peak_outstanding_words"],4)
        self.assertEqual(resources["macro_instances"],{"4096_single":4})
        self.assertEqual(resources["local_sram_bytes"],32768)
        self.assertEqual(resources["weight_sram_bytes"],0)
        regs = resources["register_payload_bytes"]
        self.assertEqual(regs["weight_register_bytes"],2*cols if array else 256)
        self.assertEqual(regs["activation_register_bytes"],2*group if array else 64 if mapping == "B0" else 128)
        self.assertEqual(regs["accumulator_register_bytes"],4*group*cols if array else 8)
        self.assertEqual(result["shared_activation_read_bytes"],0)
        self.assertEqual(result["shared_activation_write_bytes"],0)
        for name,stats in result["buffers"].items():
            self.assertTrue(0 <= stats["peak_live_bytes"] <= stats["peak_reserved_bytes"] <= stats["capacity_bytes"])
            self.assertEqual(stats["final_reserved_bytes"],6336 if name == "resident_output" else 0)
        self.assertEqual(sum(result["cycle_breakdown"].values()),result["total_cycles"])
        self.assertFalse(result["initialization"]["included_in_main_window"])
        self.assertIsNone(result["initialization"]["external_activation_load_cycles"])

    def _check_trace(self,sim,shape):
        bank_cycles = Counter()
        valid = {}
        for tensor in ("input","residual"):
            desc = sim.layout.tensors[tensor]
            for row in range(desc["rows"]):
                for col in range(desc["cols"]):
                    valid[sim.layout.address(tensor,row,col)] = tensor
        for event in sim.local.trace:
            if event["event"] == "LOCAL_WORD_DONE" and event["write"]:
                for address in physical_addresses(event):
                    valid[address] = event["tag"]["tensor"]
            if event["event"] not in ("LOCAL_READ","LOCAL_WRITE"):
                continue
            self.assertTrue(0 <= event["bank"] < 4 and 0 <= event["word_index"] < 4096)
            bank_cycles[(event["cycle"],event["bank"])] += 1
            self.assertEqual(event["complete_cycle"],event["cycle"]+1)
            if not event["write"]:
                self.assertTrue(all(valid.get(addr) == event["tag"]["tensor"] for addr in physical_addresses(event)))
        self.assertTrue(all(count == 1 for count in bank_cycles.values()))
        matrix,post_masks,inputs,ln_accept,outputs = {},defaultdict(int),{},{},set()
        ready,mac_done,consume,released,active,reservations = {},defaultdict(list),defaultdict(list),{},{},{}
        beat_lifetimes,issue_cycles = [],[]
        for event in sim.weights.trace:
            packet = event.get("packet_id")
            if event["event"] == "PACKET_RESERVED":
                self.assertNotIn(event["slot"],active)
                active[event["slot"]] = packet
                reservations[packet] = event
                self.assertLessEqual(len(active),2)
            elif event["event"] == "PACKET_READY":
                self.assertEqual(event["valid_mask"],(1 << reservations[packet]["useful_bytes"])-1)
                ready[packet] = event["cycle"]
            elif event["event"] == "WEIGHT_BEAT_ISSUE":
                self.assertEqual(event["complete_cycle"],event["cycle"]+3)
                beat_lifetimes.append((event["cycle"],1,1))
                issue_cycles.append(event["cycle"])
            elif event["event"] == "WEIGHT_BEAT_DONE":
                beat_lifetimes.append((event["cycle"],0,-1))
            elif event["event"] == "PACKET_CONSUME":
                consume[packet].append(event["cycle"])
            elif event["event"] == "PACKET_RELEASE":
                self.assertEqual(active.pop(event["slot"]),packet)
                released[packet] = event["cycle"]
        self.assertEqual(active,{})
        self.assertTrue(all(b-a >= 2 for a,b in zip(issue_cycles,issue_cycles[1:])))
        outstanding = 0
        for _,_,change in sorted(beat_lifetimes):
            outstanding += change
            self.assertTrue(0 <= outstanding <= 2)
        self.assertEqual(outstanding,0)
        for event in sim.trace:
            kind,cycle,row = event["event"],event["cycle"],event.get("token")
            if kind == "COMPUTE_ISSUE":
                self.assertGreaterEqual(cycle,ready[event["packet_id"]])
            elif kind == "MAC_COMPLETE":
                mac_done[event["packet_id"]].append(cycle)
            elif kind == "MAT_FRAGMENT_READY":
                for col in event["channels"]:
                    self.assertNotIn((row,col),matrix)
                    matrix[(row,col)] = cycle
            elif kind == "POST_ISSUE":
                self.assertTrue(all(matrix[(row,col)] <= cycle for col in event["channels"]))
            elif kind == "POST_WRITE_DONE":
                mask = sum(1 << col for col in event["channels"])
                self.assertFalse(post_masks[row]&mask)
                post_masks[row] |= mask
            elif kind == "LN_INPUT_READY":
                self.assertEqual(post_masks[row],(1 << shape.n)-1)
                inputs[row] = cycle
            elif kind == "LN_ACCEPT":
                self.assertGreaterEqual(cycle,inputs[row])
                self.assertEqual(event["complete_cycle"],cycle+64)
                ln_accept[row] = cycle
            elif kind == "LN_CORE_DONE":
                self.assertEqual(cycle,ln_accept[row]+64)
            elif kind == "INTERNAL_OUTPUT_CONFIRMED":
                self.assertGreaterEqual(cycle,ln_accept[row]+65)
                outputs.add(row)
        starts = sorted(ln_accept.values())
        self.assertTrue(all(b-a >= 64 for a,b in zip(starts,starts[1:])))
        self.assertEqual(len(matrix),shape.m*shape.n)
        self.assertEqual(outputs,set(range(shape.m)))
        for packet,cycles in mac_done.items():
            self.assertEqual(cycles,consume[packet])
            self.assertEqual(len(cycles),reservations[packet]["consumers"])
            self.assertEqual(released[packet],max(cycles))
        if sim.mapping == "A33x2":
            issues = [e for e in sim.weights.trace if e["event"] == "WEIGHT_BEAT_ISSUE"]
            self.assertEqual({e["byte_mask"] for e in issues},{3,12})
            self.assertTrue(all(e["valid_bytes"] == 2 for e in issues))
            stats = sim.weights.stats()
            self.assertEqual(stats["physical_weight_bytes"],2*stats["useful_weight_bytes"])

    def test_named_revision_is_exact_and_default_contract_remains_separate(self):
        revised,default = self._config(),load_config()
        self.assertEqual(revised["provenance"]["scenario_id"],REVISED_PROFILE_ID)
        self.assertEqual(revised["local_memory"]["banks"],4)
        self.assertEqual((revised["ln"]["latency_cycles"],revised["ln"]["initiation_interval_cycles"]),(64,64))
        self.assertEqual(default["local_memory"]["banks"],8)
        self.assertEqual((default["ln"]["latency_cycles"],default["ln"]["initiation_interval_cycles"]),(128,128))
        for override in ({"ln":{"latency_cycles":32}}, {"ln":{"initiation_interval_cycles":128}},
                         {"local_memory":{"banks":8}}):
            with self.assertRaises(ValueError):
                load_config(REVISED_CONFIG,overrides=override)
        for override in ({"ln":{"latency_cycles":64,"initiation_interval_cycles":64}},
                         {"local_memory":{"banks":4}}):
            with self.assertRaises(ValueError):
                load_config(overrides=override)
        self.assertEqual(tuple(REVISION_MAPPINGS),tuple(REVISION_GEOMETRY))
        self.assertEqual(set(STUDY_MAPPINGS),set(ARRAY_GEOMETRY)|set(ROW_GROUPS))
        self.assertTrue(set(STUDY_MAPPINGS).issubset(ALL_MAPPINGS))

    def test_all_twenty_full_projection_mappings_respect_closed_form_ledgers(self):
        for mapping in STUDY_MAPPINGS:
            with self.subTest(mapping=mapping):
                _,result = self._run(mapping,Shape(99,64,64))
                self._check_projection(mapping,result)

    def test_all_twenty_full_ffn2_layouts_are_structured_capacity_failures(self):
        shape = Shape(99,64,256)
        for mapping in STUDY_MAPPINGS:
            with self.subTest(mapping=mapping):
                with patch("hw_model.fixed_simulation.WeightPort",wraps=WeightPort) as port:
                    with self.assertRaises(LocalCapacityError) as failure:
                        FusedSimulator(self._config(),mapping,operator="ffn2",shape=shape)
                    port.assert_not_called()
                error = failure.exception
                self.assertEqual(error.mapping,mapping)
                self.assertEqual(error.capacity_bytes,32768)
                self.assertEqual(error.required_bytes,required_bytes(shape,mapping))
                self.assertGreater(error.required_bytes,error.capacity_bytes)
                self.assertEqual(sum(t["padded_bytes"] for t in error.tensors.values()),error.required_bytes)
                self.assertEqual(set(error.tensors),{"input","residual","pre_ln","output","partial"}
                                 if mapping.startswith("B1") else {"input","residual","pre_ln","output"})
                self.assertEqual(error.tensors["input"]["rows"],99)
                self.assertEqual(error.tensors["input"]["cols"],256)
                self.assertEqual(error.tensors["pre_ln"]["logical_bytes"],12672)
                self.assertEqual(error.tensors["output"]["logical_bytes"],6336)

    def test_revision_random_int8_gemm_and_four_bank_byte_mapping(self):
        rng = random.Random(2026102932)
        for mapping,(group,columns) in REVISION_GEOMETRY.items():
            for reduction in (65,129):
                shape = Shape(group+1,columns+1,reduction)
                with self.subTest(mapping=mapping,shape=shape):
                    layout = LocalLayout(shape,mapping,group,banks=4)
                    self.assertEqual(layout.capacity_bytes,32768)
                    left = [[rng.randint(-128,127) for _ in range(shape.r)] for _ in range(shape.m)]
                    right = [[rng.randint(-128,127) for _ in range(shape.n)] for _ in range(shape.r)]
                    expected = [[sum(left[i][k]*right[k][j] for k in range(shape.r))
                                 for j in range(shape.n)] for i in range(shape.m)]
                    local = {layout.address("input",i,k):left[i][k] for i in range(shape.m) for k in range(shape.r)}
                    self.assertEqual(len(local),shape.m*shape.r)
                    for tensor,desc in layout.tensors.items():
                        for row in (0,desc["rows"]-1):
                            logical = {layout.address(tensor,row,col)+lane for col in range(desc["cols"])
                                       for lane in range(desc["element_bytes"])}
                            physical = set()
                            for bank,word,mask in layout.accesses(tensor,row,range(desc["cols"])):
                                self.assertTrue(0 <= bank < 4 and 0 <= word < 4096)
                                physical.update((word*4+bank)*2+lane for lane in (0,1) if mask & (1 << lane))
                            self.assertEqual(physical,logical)
                    packed = {}
                    for k in range(shape.r):
                        for col in range(shape.n):
                            address = weight_byte_offset(shape,mapping,k,col)
                            self.assertNotIn(address,packed)
                            packed[address] = right[k][col]
                    actual = [[0]*shape.n for _ in range(shape.m)]
                    coverage = [[0]*shape.n for _ in range(shape.m)]
                    for packet in schedule_packets(shape,mapping):
                        for row in packet.rows:
                            for cj,col in enumerate(packet.columns):
                                for kk,k in enumerate(packet.reduction):
                                    self.assertFalse(coverage[row][col] & (1 << k))
                                    coverage[row][col] |= 1 << k
                                    actual[row][col] += local[layout.address("input",row,k)]*packed[
                                        packet.weight_address+kk*len(packet.columns)+cj]
                    self.assertEqual(actual,expected)
                    self.assertTrue(all(mask == (1 << shape.r)-1 for row in coverage for mask in row))

    def test_all_twenty_tail_traces_keep_banks_ln_and_full_channel_gates_honest(self):
        for mapping in STUDY_MAPPINGS:
            group = ARRAY_GEOMETRY[mapping][0] if mapping in ARRAY_GEOMETRY else ROW_GROUPS[mapping]
            # G99 uses a partial group to preserve capacity while exercising
            # retained weights and actual R65 partial-SRAM dependencies.
            shape = Shape(min(group+1,37),64,65)
            with self.subTest(mapping=mapping):
                sim,result = self._run(mapping,shape,trace=True)
                self._check_trace(sim,shape)
                self.assertLessEqual(result["peak_operand_packets"],2)
                self.assertFalse(sim.weights.pending or sim.local.pending or sim.events.pending)


if __name__ == "__main__":
    unittest.main()
