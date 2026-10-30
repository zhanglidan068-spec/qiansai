"""Independent opt-in P=20..33, Q=4/6/8 grid validation.

Physical traffic is reconstructed from absolute byte addresses, not from the
packet's accounting formula. Compact six-byte slots are never aligned or merged
implicitly. Existing default mapping sets and tests are left unchanged.
"""
from collections import Counter, defaultdict
import random
import unittest

from hw_model.compute import Shape
from hw_model.config import LN64_LOCAL64K_CONFIG, load_config
from hw_model.fixed_memory import LocalLayout, WeightPort
from hw_model.fixed_schedule import (
    ALL_MAPPINGS, GRID_MAPPINGS, SUPPORTED_MAPPINGS, mapping_traits,
    schedule_packets, weight_byte_offset,
)
from hw_model.fixed_simulation import FusedSimulator


EXPECTED_GRID = tuple("A{}x{}".format(p,q) for p in range(20,34) for q in (4,6,8))


def geometry(mapping):
    return tuple(map(int,mapping[1:].split("x")))


def ceildiv(value, divisor):
    return (value+divisor-1)//divisor


def expected_beats(address, useful_bytes):
    masks = {}
    for byte in range(address,address+useful_bytes):
        base = byte//4*4
        masks[base] = masks.get(base,0) | (1 << (byte%4))
    return masks


def local_bytes(event):
    base = (event["word_index"]*8+event["bank"])*2
    return [base+lane for lane in (0,1) if event["byte_mask"] & (1 << lane)]


class ArrayGridTests(unittest.TestCase):
    def _run(self,mapping,shape,trace=False):
        sim = FusedSimulator(load_config(LN64_LOCAL64K_CONFIG),mapping,shape=shape,capture_trace=trace)
        return sim,sim.run(max_cycles=400000)

    def _ledger(self,mapping,shape,result):
        p,q = geometry(mapping)
        groups,columns = ceildiv(shape.m,p),ceildiv(shape.n,q)
        packet_count = groups*columns*shape.r
        # Sum addressed 4-byte words separately for every request. A word
        # fetched by adjacent compact packets is charged again, as the bus does.
        physical = sum(len(expected_beats((ct*shape.r+k)*q,min(q,shape.n-ct*q)))*4
                       for ct in range(columns) for k in range(shape.r))*groups
        weight = result["weight_port"]
        self.assertEqual(weight["useful_weight_bytes"],groups*shape.r*shape.n)
        self.assertEqual(weight["physical_weight_bytes"],physical)
        self.assertEqual(weight["accepted_beats"]*4,physical)
        self.assertEqual(weight["accepted_beats"],weight["completed_beats"])
        self.assertEqual(weight["occupied_service_cycles"],2*weight["accepted_beats"])
        self.assertEqual(weight["fill_count"],packet_count)
        self.assertEqual(weight["released_packets"],packet_count)
        self.assertEqual(weight["consumers_completed"],packet_count)
        self.assertEqual(weight["outstanding_beats"],0)
        self.assertEqual(weight["active_packets"],0)
        self.assertLessEqual(weight["max_outstanding"],2)
        self.assertLessEqual(weight["peak_reserved_bytes"],2*q)
        self.assertEqual(result["useful_mac"],shape.useful_mac)
        self.assertEqual(result["physical_multipliers"],p*q)
        self.assertEqual(result["issued_updates"],packet_count)
        self.assertEqual(result["issue_lane_utilization"],shape.useful_mac/(p*q*packet_count))
        self.assertEqual({owner:(stats["read_useful_bytes"],stats["write_useful_bytes"])
                          for owner,stats in result["local_memory"]["per_owner"].items()},
                         {"matrix_activation":(shape.m*shape.r*columns,0),
                          "post_residual":(shape.m*shape.n,0),"post_pre_ln":(0,2*shape.m*shape.n),
                          "ln_input":(2*shape.m*shape.n,0),"ln_output":(0,shape.m*shape.n)})
        resources = result["resources"]
        self.assertEqual(resources["multipliers"],p*q)
        self.assertEqual(resources["macro_instances"],{"4096_single":8})
        self.assertEqual(resources["weight_sram_bytes"],0)
        regs = resources["register_payload_bytes"]
        self.assertEqual(regs["weight_register_bytes"],2*q)
        self.assertEqual(regs["activation_register_bytes"],2*p)
        self.assertEqual(regs["accumulator_register_bytes"],p*q*4)
        self.assertEqual(resources["b1_partial_sram_bytes"],0)
        for name,stats in result["buffers"].items():
            self.assertTrue(0 <= stats["peak_live_bytes"] <= stats["peak_reserved_bytes"] <= stats["capacity_bytes"])
            self.assertEqual(stats["final_reserved_bytes"],shape.m*shape.n if name == "resident_output" else 0)
        self.assertLessEqual(result["peak_operand_packets"],2)
        self.assertLessEqual(result["peak_post_jobs"],2)
        self.assertEqual(result["shared_activation_read_bytes"],0)
        self.assertEqual(result["shared_activation_write_bytes"],0)

    def _trace(self,sim,shape):
        descriptions = {packet.packet_id:packet for packet in schedule_packets(shape,sim.mapping)}
        actual_beats,ready,consumed,released,active = defaultdict(dict),{},defaultdict(list),{},{}
        for event in sim.weights.trace:
            kind,packet = event["event"],event.get("packet_id")
            if kind == "PACKET_RESERVED":
                self.assertNotIn(event["slot"],active)
                active[event["slot"]] = packet
                self.assertLessEqual(len(active),2)
                self.assertEqual(event["address"],descriptions[packet].weight_address)
            elif kind == "WEIGHT_BEAT_ISSUE":
                self.assertNotIn(event["address"],actual_beats[packet])
                actual_beats[packet][event["address"]] = event["byte_mask"]
                self.assertEqual(event["valid_bytes"],bin(event["byte_mask"]).count("1"))
                self.assertEqual(event["complete_cycle"],event["cycle"]+3)
            elif kind == "PACKET_READY":
                desc = descriptions[packet]
                self.assertEqual(event["valid_mask"],(1 << desc.weight_bytes)-1)
                ready[packet] = event["cycle"]
            elif kind == "PACKET_CONSUME":
                consumed[packet].append(event["cycle"])
            elif kind == "PACKET_RELEASE":
                self.assertEqual(active.pop(event["slot"]),packet)
                released[packet] = event["cycle"]
        self.assertEqual(active,{})
        self.assertEqual(set(actual_beats),set(descriptions))
        for packet,desc in descriptions.items():
            expected = expected_beats(desc.weight_address,desc.weight_bytes)
            self.assertEqual(actual_beats[packet],expected)
            self.assertEqual(desc.physical_weight_bytes,len(expected)*4)
        initialized = {}
        for tensor in ("input","residual"):
            d = sim.layout.tensors[tensor]
            for row in range(d["rows"]):
                for col in range(d["cols"]):
                    initialized[sim.layout.address(tensor,row,col)] = tensor
        bank_cycles,in_start,in_done,out_start,out_done = Counter(),{},{},{},{}
        for event in sim.local.trace:
            row = event.get("tag",{}).get("token")
            if event["event"] == "LOCAL_REQUEST":
                if event["owner"] == "ln_input": in_start[row] = event["cycle"]
                elif event["owner"] == "ln_output": out_start[row] = event["cycle"]
            elif event["event"] == "LOCAL_REQUEST_DONE":
                if event["owner"] == "ln_input": in_done[row] = event["cycle"]
                elif event["owner"] == "ln_output": out_done[row] = event["cycle"]
            if event["event"] == "LOCAL_WORD_DONE" and event["write"]:
                for address in local_bytes(event): initialized[address] = event["tag"]["tensor"]
            if event["event"] in ("LOCAL_READ","LOCAL_WRITE"):
                self.assertTrue(0 <= event["bank"] < 8 and 0 <= event["word_index"] < 4096)
                bank_cycles[(event["cycle"],event["bank"])] += 1
                self.assertEqual(event["complete_cycle"],event["cycle"]+1)
                if not event["write"]:
                    self.assertTrue(all(initialized.get(addr) == event["tag"]["tensor"] for addr in local_bytes(event)))
        self.assertTrue(all(count == 1 for count in bank_cycles.values()))
        matrix,post,accept,core_done,done = {},defaultdict(int),{},{},defaultdict(list)
        outputs = set()
        for event in sim.trace:
            kind,row,cycle = event["event"],event.get("token"),event["cycle"]
            if kind == "COMPUTE_ISSUE": self.assertGreaterEqual(cycle,ready[event["packet_id"]])
            elif kind == "MAC_COMPLETE": done[event["packet_id"]].append(cycle)
            elif kind == "MAT_FRAGMENT_READY":
                for col in event["channels"]:
                    self.assertNotIn((row,col),matrix)
                    matrix[(row,col)] = cycle
            elif kind == "POST_ISSUE": self.assertTrue(all(matrix[(row,c)] <= cycle for c in event["channels"]))
            elif kind == "POST_WRITE_DONE":
                mask = sum(1 << c for c in event["channels"])
                self.assertFalse(post[row]&mask)
                post[row] |= mask
            elif kind == "LN_ACCEPT":
                self.assertEqual(post[row],(1 << shape.n)-1)
                self.assertGreaterEqual(cycle,in_done[row])
                self.assertEqual(event["complete_cycle"],cycle+64)
                accept[row] = cycle
            elif kind == "LN_CORE_DONE":
                self.assertEqual(cycle,accept[row]+64)
                core_done[row] = cycle
            elif kind == "INTERNAL_OUTPUT_CONFIRMED":
                self.assertEqual(cycle,out_done[row])
                outputs.add(row)
        self.assertEqual(len(matrix),shape.m*shape.n)
        self.assertEqual(outputs,set(range(shape.m)))
        ordered = sorted(in_start,key=in_start.get)
        for row in ordered:
            self.assertGreaterEqual(in_done[row]-in_start[row],ceildiv(shape.n,8))
            self.assertGreaterEqual(out_start[row],core_done[row])
            self.assertGreaterEqual(out_done[row]-out_start[row],ceildiv(shape.n,16))
        for previous,current in zip(ordered,ordered[1:]):
            self.assertGreaterEqual(in_start[current],out_done[previous])
            self.assertGreaterEqual(accept[current]-accept[previous],64+ceildiv(shape.n,8)+ceildiv(shape.n,16))
        for packet,cycles in done.items():
            self.assertEqual(cycles,consumed[packet])
            self.assertEqual(released[packet],max(cycles))

    def test_grid_registry_geometry_and_full_operator_capacity(self):
        self.assertEqual(GRID_MAPPINGS,EXPECTED_GRID)
        self.assertEqual(len(ALL_MAPPINGS),28)
        self.assertEqual(set(SUPPORTED_MAPPINGS),set(ALL_MAPPINGS)|set(EXPECTED_GRID))
        self.assertEqual(len(SUPPORTED_MAPPINGS),len(set(SUPPORTED_MAPPINGS)))
        for mapping in GRID_MAPPINGS:
            p,q = geometry(mapping)
            for reduction in (64,256):
                with self.subTest(mapping=mapping,reduction=reduction):
                    shape = Shape(99,64,reduction)
                    traits = mapping_traits(shape,mapping)
                    self.assertEqual((traits["group_rows"],traits["column_lanes"],traits["physical_multipliers"]),(p,q,p*q))
                    self.assertEqual(traits["weight_packet_slot_bytes"],q)
                    layout = LocalLayout(shape,mapping,p,banks=8)
                    self.assertEqual(layout.used_bytes,100*reduction+25344)
                    self.assertEqual(layout.capacity_bytes,65536)
                    self.assertNotIn("partial",layout.tensors)

    def test_all_42_random_int8_gemm_absolute_packing_and_tail_lanes(self):
        rng = random.Random(2026102942)
        for mapping in GRID_MAPPINGS:
            p,q = geometry(mapping)
            for shape in (Shape(p+1,64,5),Shape(3,64,65)):
                with self.subTest(mapping=mapping,shape=shape):
                    left = [[rng.randint(-128,127) for _ in range(shape.r)] for _ in range(shape.m)]
                    right = [[rng.randint(-128,127) for _ in range(shape.n)] for _ in range(shape.r)]
                    expected = [[sum(left[i][k]*right[k][j] for k in range(shape.r))
                                 for j in range(shape.n)] for i in range(shape.m)]
                    layout = LocalLayout(shape,mapping,p,banks=8)
                    stride = 2*ceildiv(shape.m,2)
                    image = {k*stride+i:left[i][k] for i in range(shape.m) for k in range(shape.r)}
                    packed = {}
                    for k in range(shape.r):
                        for col in range(shape.n):
                            address = ((col//q)*shape.r+k)*q+col%q
                            self.assertEqual(weight_byte_offset(shape,mapping,k,col),address)
                            self.assertNotIn(address,packed)
                            packed[address] = right[k][col]
                    actual,coverage = [[0]*shape.n for _ in range(shape.m)],[[0]*shape.n for _ in range(shape.m)]
                    for packet in schedule_packets(shape,mapping):
                        self.assertEqual(packet.physical_weight_bytes,len(expected_beats(packet.weight_address,packet.weight_bytes))*4)
                        self.assertLessEqual(packet.useful_mac,p*q)
                        if q == 6 and packet.columns[0] == 60: self.assertEqual(packet.columns,(60,61,62,63))
                        for row in packet.rows:
                            for cj,col in enumerate(packet.columns):
                                for k in packet.reduction:
                                    self.assertFalse(coverage[row][col] & (1 << k))
                                    coverage[row][col] |= 1 << k
                                    self.assertEqual(layout.address("input",row,k),k*stride+row)
                                    actual[row][col] += image[layout.address("input",row,k)]*packed[packet.weight_address+cj]
                    self.assertEqual(actual,expected)
                    self.assertTrue(all(mask == (1 << shape.r)-1 for row in coverage for mask in row))

    def test_six_byte_port_slots_charge_repeated_words_and_tail_crossings(self):
        for address_a,address_b,size in ((0,6,6),(12,18,4)):
            with self.subTest(size=size):
                port = WeightPort(6)
                port.reserve("a",address_a,size,1)
                port.reserve("b",address_b,size,1)
                for cycle in range(12):
                    port.complete(cycle)
                    port.issue(cycle)
                actual = defaultdict(dict)
                for event in port.trace:
                    if event["event"] == "WEIGHT_BEAT_ISSUE":
                        actual[event["packet_id"]][event["address"]] = event["byte_mask"]
                self.assertEqual(actual["a"],expected_beats(address_a,size))
                self.assertEqual(actual["b"],expected_beats(address_b,size))
                self.assertEqual(port.stats()["useful_weight_bytes"],size*2)
                self.assertEqual(port.stats()["physical_weight_bytes"],16 if size == 6 else 12)
                if size == 6:
                    self.assertEqual(set(actual["a"])&set(actual["b"]),{4})
                else:
                    self.assertEqual((len(actual["a"]),len(actual["b"])),(1,2))

    def test_all_42_small_traces_keep_banks_packet_bytes_and_ln64_honest(self):
        for mapping in GRID_MAPPINGS:
            p,_ = geometry(mapping)
            shape = Shape(p+1,64,5)
            with self.subTest(mapping=mapping):
                sim,result = self._run(mapping,shape,trace=True)
                self._ledger(mapping,shape,result)
                self._trace(sim,shape)

    def test_focus_full_operator_useful_and_physical_traffic_closed_forms(self):
        for mapping in ("A26x6","A26x8","A33x6"):
            p,q = geometry(mapping)
            for reduction in (64,256):
                with self.subTest(mapping=mapping,reduction=reduction):
                    shape = Shape(99,64,reduction)
                    _,result = self._run(mapping,shape)
                    self._ledger(mapping,shape,result)
                    groups = ceildiv(99,p)
                    self.assertEqual(result["weight_port"]["useful_weight_bytes"],groups*64*reduction)
                    self.assertEqual(result["weight_port"]["physical_weight_bytes"],groups*(86 if q == 6 else 64)*reduction)
                    self.assertEqual(result["initialization"]["shared_weights_packed_address_bytes"],ceildiv(64,q)*q*reduction)


if __name__ == "__main__":
    unittest.main()
