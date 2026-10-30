"""Independent validation of LN64 with the original eight-bank 64 KiB SRAM.

The historical LN128/64 KiB and LN64/32 KiB profiles remain separate. These
checks do not predict/freeze performance cycles or alter residency boundaries.
"""
from collections import Counter, defaultdict
import random
import re
import unittest

from hw_model.compute import Shape
from hw_model.config import (
    LN64_LOCAL64K_CONFIG, LN64_LOCAL64K_PROFILE_ID, REVISED_CONFIG, load_config,
)
from hw_model.fixed_memory import LocalLayout
from hw_model.fixed_schedule import (
    ALL_MAPPINGS, EXTENDED_MAPPINGS, REVISION_MAPPINGS, mapping_traits,
    schedule_packets, weight_byte_offset,
)
from hw_model.fixed_simulation import FusedSimulator


EXPECTED_EXTENSIONS = {"A33x32", "A11x4", "A99x4", "A99x8", "A66x4",
                       "A99x16", "A66x8", "A66x16"}
NEW_ARRAYS = REVISION_MAPPINGS + EXTENDED_MAPPINGS


def ceildiv(value, divisor):
    return (value + divisor - 1) // divisor


def array_shape(mapping):
    if mapping == "A":
        return 9,16
    match = re.fullmatch(r"A(\d+)x(\d+)",mapping)
    return tuple(map(int,match.groups())) if match else None


def group_rows(mapping):
    geometry = array_shape(mapping)
    return geometry[0] if geometry else 1 if mapping == "B0" else int(mapping.split("_G")[1])


def physical_addresses(event):
    base = (event["word_index"]*8+event["bank"])*2
    return [base+lane for lane in (0,1) if event["byte_mask"] & (1 << lane)]


class Ln64Local64kTests(unittest.TestCase):
    def _config(self):
        return load_config(LN64_LOCAL64K_CONFIG)

    def _run(self,mapping,shape,trace=False):
        sim = FusedSimulator(self._config(),mapping,shape=shape,capture_trace=trace)
        return sim,sim.run(max_cycles=400000)

    def _array_ledger(self,mapping,shape,result):
        rows,cols = array_shape(mapping)
        groups,col_tiles = ceildiv(shape.m,rows),ceildiv(shape.n,cols)
        packets = groups*col_tiles*shape.r
        per_group_physical = sum(4*ceildiv(min(cols,shape.n-c),4) for c in range(0,shape.n,cols))*shape.r
        weight,local = result["weight_port"],result["local_memory"]
        self.assertEqual(result["physical_multipliers"],rows*cols)
        self.assertEqual(result["issued_updates"],packets)
        self.assertEqual(result["useful_mac"],shape.useful_mac)
        self.assertEqual(result["issue_lane_utilization"],shape.useful_mac/(rows*cols*packets))
        self.assertEqual(weight["useful_weight_bytes"],groups*shape.r*shape.n)
        self.assertEqual(weight["physical_weight_bytes"],groups*per_group_physical)
        self.assertEqual(weight["physical_weight_bytes"],weight["accepted_beats"]*4)
        self.assertEqual(weight["occupied_service_cycles"],weight["accepted_beats"]*2)
        self.assertEqual(weight["accepted_beats"],weight["completed_beats"])
        self.assertEqual(weight["fill_count"],packets)
        self.assertEqual(weight["consumers_completed"],packets)
        self.assertEqual(weight["released_packets"],packets)
        self.assertEqual(weight["active_packets"],0)
        self.assertEqual(weight["outstanding_beats"],0)
        owners = {owner:(stats["read_useful_bytes"],stats["write_useful_bytes"])
                  for owner,stats in local["per_owner"].items()}
        self.assertEqual(owners,{"matrix_activation":(shape.m*shape.r*col_tiles,0),
                                 "post_residual":(shape.m*shape.n,0),
                                 "post_pre_ln":(0,shape.m*shape.n*2),
                                 "ln_input":(shape.m*shape.n*2,0),
                                 "ln_output":(0,shape.m*shape.n)})
        self.assertEqual(local["banks"],8)
        self.assertEqual(local["capacity_bytes"],65536)
        self.assertEqual(local["pending_requests"],0)
        self.assertEqual(local["outstanding_words"],0)
        resources = result["resources"]
        self.assertEqual(resources["multipliers"],rows*cols)
        self.assertEqual(resources["macro_instances"],{"4096_single":8})
        self.assertEqual(resources["weight_sram_bytes"],0)
        registers = resources["register_payload_bytes"]
        self.assertEqual(registers["weight_register_bytes"],cols*2)
        self.assertEqual(registers["activation_register_bytes"],rows*2)
        self.assertEqual(registers["accumulator_register_bytes"],rows*cols*4)
        self.assertEqual(registers["partial_operand_register_bytes"],0)
        self.assertEqual(resources["b1_partial_sram_bytes"],0)
        self.assertEqual(result["buffers"]["ln_workspace"]["capacity_bytes"],shape.n*3)
        for name,stats in result["buffers"].items():
            self.assertTrue(0 <= stats["peak_live_bytes"] <= stats["peak_reserved_bytes"] <= stats["capacity_bytes"])
            self.assertEqual(stats["final_reserved_bytes"],shape.m*shape.n if name == "resident_output" else 0)
        self.assertEqual(result["shared_activation_read_bytes"],0)
        self.assertEqual(result["shared_activation_write_bytes"],0)
        self.assertFalse(result["initialization"]["included_in_main_window"])

    def _trace_contract(self,sim,shape):
        bank_cycles = Counter()
        initialized = {}
        for tensor in ("input","residual"):
            desc = sim.layout.tensors[tensor]
            for row in range(desc["rows"]):
                for col in range(desc["cols"]):
                    initialized[sim.layout.address(tensor,row,col)] = tensor
        in_request,in_done,out_request,out_done = {},{},{},{}
        for event in sim.local.trace:
            row = event.get("tag",{}).get("token")
            if event["event"] == "LOCAL_REQUEST":
                if event["owner"] == "ln_input":
                    self.assertNotIn(row,in_request)
                    in_request[row] = event["cycle"]
                elif event["owner"] == "ln_output":
                    out_request[row] = event["cycle"]
            elif event["event"] == "LOCAL_REQUEST_DONE":
                if event["owner"] == "ln_input":
                    in_done[row] = event["cycle"]
                elif event["owner"] == "ln_output":
                    out_done[row] = event["cycle"]
            if event["event"] == "LOCAL_WORD_DONE" and event["write"]:
                for address in physical_addresses(event):
                    initialized[address] = event["tag"]["tensor"]
            if event["event"] in ("LOCAL_READ","LOCAL_WRITE"):
                self.assertTrue(0 <= event["bank"] < 8 and 0 <= event["word_index"] < 4096)
                bank_cycles[(event["cycle"],event["bank"])] += 1
                self.assertEqual(event["complete_cycle"],event["cycle"]+1)
                if not event["write"]:
                    self.assertTrue(all(initialized.get(addr) == event["tag"]["tensor"]
                                        for addr in physical_addresses(event)))
        self.assertTrue(all(count == 1 for count in bank_cycles.values()))
        active,ready,consume,released = {},{},defaultdict(list),{}
        for event in sim.weights.trace:
            packet = event.get("packet_id")
            if event["event"] == "PACKET_RESERVED":
                self.assertNotIn(event["slot"],active)
                active[event["slot"]] = packet
                self.assertLessEqual(len(active),2)
            elif event["event"] == "PACKET_READY":
                ready[packet] = event["cycle"]
            elif event["event"] == "PACKET_CONSUME":
                consume[packet].append(event["cycle"])
            elif event["event"] == "PACKET_RELEASE":
                self.assertEqual(active.pop(event["slot"]),packet)
                released[packet] = event["cycle"]
        self.assertEqual(active,{})
        matrix,post_masks,accept,core_done,mac_done = {},defaultdict(int),{},{},defaultdict(list)
        outputs = {}
        for event in sim.trace:
            kind,row,cycle = event["event"],event.get("token"),event["cycle"]
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
            elif kind == "LN_ACCEPT":
                self.assertEqual(post_masks[row],(1 << shape.n)-1)
                self.assertGreaterEqual(cycle,in_done[row])
                self.assertEqual(event["complete_cycle"],cycle+64)
                accept[row] = cycle
            elif kind == "LN_CORE_DONE":
                self.assertEqual(cycle,accept[row]+64)
                core_done[row] = cycle
            elif kind == "INTERNAL_OUTPUT_CONFIRMED":
                self.assertEqual(cycle,out_done[row])
                outputs[row] = cycle
        self.assertEqual(len(matrix),shape.m*shape.n)
        self.assertEqual(set(outputs),set(range(shape.m)))
        read_min,write_min = ceildiv(shape.n,8),ceildiv(shape.n,16)
        for row in range(shape.m):
            self.assertGreaterEqual(in_done[row]-in_request[row],read_min)
            self.assertGreaterEqual(out_request[row],core_done[row])
            self.assertGreaterEqual(out_done[row]-out_request[row],write_min)
            self.assertGreaterEqual(out_done[row]-in_request[row],64+read_min+write_min)
        ordered = sorted(in_request,key=in_request.get)
        for previous,current in zip(ordered,ordered[1:]):
            self.assertGreaterEqual(in_request[current],out_done[previous],"two LN jobs overlap")
            self.assertGreaterEqual(accept[current]-accept[previous],64+read_min+write_min)
        for packet,cycles in mac_done.items():
            self.assertEqual(cycles,consume[packet])
            self.assertEqual(released[packet],max(cycles))
        if sim.mapping == "A33x2":
            beats = [event for event in sim.weights.trace if event["event"] == "WEIGHT_BEAT_ISSUE"]
            self.assertEqual({event["byte_mask"] for event in beats},{3,12})
            self.assertTrue(all(event["valid_bytes"] == 2 for event in beats))
            self.assertEqual(sim.weights.stats()["physical_weight_bytes"],2*sim.weights.stats()["useful_weight_bytes"])

    def test_three_profiles_remain_distinct_and_mixed_contracts_are_rejected(self):
        new,old32,old64 = self._config(),load_config(REVISED_CONFIG),load_config()
        self.assertEqual(new["provenance"]["scenario_id"],LN64_LOCAL64K_PROFILE_ID)
        self.assertEqual((new["local_memory"]["banks"],new["ln"]["latency_cycles"],new["ln"]["initiation_interval_cycles"]),(8,64,64))
        self.assertEqual((old32["local_memory"]["banks"],old32["ln"]["latency_cycles"]),(4,64))
        self.assertEqual((old64["local_memory"]["banks"],old64["ln"]["latency_cycles"]),(8,128))
        for override in ({"local_memory":{"banks":4}},{"ln":{"latency_cycles":128}},
                         {"ln":{"initiation_interval_cycles":128}},{"ln":{"latency_cycles":32}}):
            with self.assertRaises(ValueError):
                load_config(LN64_LOCAL64K_CONFIG,overrides=override)
        self.assertEqual(set(EXTENDED_MAPPINGS),EXPECTED_EXTENSIONS)
        self.assertEqual(len(ALL_MAPPINGS),len(set(ALL_MAPPINGS)))
        self.assertTrue(set(NEW_ARRAYS).issubset(ALL_MAPPINGS))

    def test_all_twenty_eight_full_tensor_layouts_fit_both_operators(self):
        for mapping in ALL_MAPPINGS:
            for reduction in (64,256):
                with self.subTest(mapping=mapping,reduction=reduction):
                    shape = Shape(99,64,reduction)
                    geometry = array_shape(mapping)
                    layout = LocalLayout(shape,mapping,group_rows(mapping),banks=8)
                    initial = 100*reduction if geometry else 2*ceildiv(99*reduction,2)
                    expected = initial+25344+(8*min(99,group_rows(mapping)) if mapping.startswith("B1") else 0)
                    self.assertEqual(layout.used_bytes,expected)
                    self.assertEqual(layout.capacity_bytes,65536)
                    self.assertLessEqual(layout.used_bytes,65536)
                    self.assertEqual(layout.tensors["input"]["rows"],99)
                    self.assertEqual(layout.tensors["input"]["cols"],reduction)
                    self.assertEqual(layout.tensors["pre_ln"]["bytes"],12672)
                    self.assertEqual(layout.tensors["output"]["bytes"],6336)
                    for tensor,desc in layout.tensors.items():
                        for row in (0,desc["rows"]-1):
                            logical = {layout.address(tensor,row,col)+lane for col in range(desc["cols"])
                                       for lane in range(desc["element_bytes"])}
                            actual = set()
                            for bank,word,mask in layout.accesses(tensor,row,range(desc["cols"])):
                                self.assertTrue(0 <= bank < 8 and 0 <= word < 4096)
                                actual.update((word*8+bank)*2+lane for lane in (0,1) if mask & (1 << lane))
                            self.assertEqual(actual,logical)

    def test_all_new_geometries_packed_random_int8_gemm_and_tail_products(self):
        rng = random.Random(2026102964)
        for mapping in NEW_ARRAYS:
            rows,cols = array_shape(mapping)
            for reduction in (65,129):
                shape = Shape(rows+1,cols+1,reduction)
                with self.subTest(mapping=mapping,reduction=reduction):
                    traits = mapping_traits(shape,mapping)
                    self.assertEqual((traits["group_rows"],traits["column_lanes"],traits["reduction_lanes"],traits["physical_multipliers"]),
                                     (rows,cols,1,rows*cols))
                    left = [[rng.randint(-128,127) for _ in range(shape.r)] for _ in range(shape.m)]
                    right = [[rng.randint(-128,127) for _ in range(shape.n)] for _ in range(shape.r)]
                    expected = [[sum(left[i][k]*right[k][j] for k in range(shape.r))
                                 for j in range(shape.n)] for i in range(shape.m)]
                    layout = LocalLayout(shape,mapping,rows,banks=8)
                    stride = 2*ceildiv(shape.m,2)
                    image = {k*stride+row:left[row][k] for row in range(shape.m) for k in range(shape.r)}
                    packed = {}
                    for k in range(shape.r):
                        for col in range(shape.n):
                            address = ((col//cols)*shape.r+k)*cols+col%cols
                            self.assertEqual(weight_byte_offset(shape,mapping,k,col),address)
                            packed[address] = right[k][col]
                    actual,coverage = [[0]*shape.n for _ in range(shape.m)],[[0]*shape.n for _ in range(shape.m)]
                    for packet in schedule_packets(shape,mapping):
                        self.assertLessEqual(packet.useful_mac,rows*cols)
                        for row in packet.rows:
                            for cj,col in enumerate(packet.columns):
                                for k in packet.reduction:
                                    self.assertEqual(layout.address("input",row,k),k*stride+row)
                                    self.assertFalse(coverage[row][col] & (1 << k))
                                    coverage[row][col] |= 1 << k
                                    actual[row][col] += image[layout.address("input",row,k)]*packed[packet.weight_address+cj]
                    self.assertEqual(actual,expected)
                    self.assertTrue(all(mask == (1 << shape.r)-1 for row in coverage for mask in row))

    def test_focus_full_projection_and_ffn2_have_explicit_resource_and_byte_ledgers(self):
        for mapping in ("A33x4","A33x2","A99x8"):
            for reduction in (64,256):
                with self.subTest(mapping=mapping,reduction=reduction):
                    shape = Shape(99,64,reduction)
                    sim,result = self._run(mapping,shape,trace=reduction == 64)
                    self._array_ledger(mapping,shape,result)
                    if reduction == 64:
                        self._trace_contract(sim,shape)

    def test_all_new_geometry_tail_traces_have_single_ln64_job_and_eight_sp_banks(self):
        for mapping in NEW_ARRAYS:
            rows,_ = array_shape(mapping)
            shape = Shape(min(rows+1,12),64,65)
            with self.subTest(mapping=mapping):
                sim,result = self._run(mapping,shape,trace=True)
                self._array_ledger(mapping,shape,result)
                self._trace_contract(sim,shape)


if __name__ == "__main__":
    unittest.main()
