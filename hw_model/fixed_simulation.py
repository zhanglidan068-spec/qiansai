"""Finite-resource A/B schedules under the amendment's fixed 300 MB/s port.

Main window: first weight request to final local INT8 output acknowledgement.
Only weights cross the shared interface. Values are metadata; LN uses the
configured row-event latency and interval, not a bit-accurate fixed-point circuit
or Python wall-clock timing. The default remains the original LN128 profile.
"""
from collections import Counter, deque
from dataclasses import asdict

from .compute import Shape
from .config import validate_config
from .core import Capacity, EventQueue
from .fixed_memory import LocalLayout, LocalMemory, WeightPort
from .fixed_schedule import mapping_traits, schedule_packets
from .resources import resource_inventory


class FusedSimulator:
    def __init__(self, config, mapping, operator="projection", shape=None, capture_trace=True):
        validate_config(config)
        if operator not in ("projection", "ffn2"):
            raise ValueError("fixed comparison supports projection and ffn2")
        self.cfg, self.mapping, self.operator = config, mapping, operator
        m = config["model"]
        self.shape = shape or Shape(m["tokens"], m["hidden"], m["hidden"] if operator == "projection" else m["ffn"])
        self.traits = mapping_traits(self.shape, mapping)
        self.array, self.b1 = self.traits["is_array"], self.traits["is_b1"]
        local_banks = config["local_memory"]["banks"]
        self.layout = LocalLayout(self.shape, mapping, self.traits["group_rows"], banks=local_banks)
        if self.traits["weight_storage_bytes"] > config["platform"]["shared_bytes"]:
            raise ValueError("packed weights exceed shared SRAM")
        self.weights = WeightPort(self.traits["weight_packet_slot_bytes"], capture_trace)
        self.local = LocalMemory(capture_trace, banks=local_banks)
        self.capture_trace, self.trace, self.cycle = capture_trace, [], 0
        self.events, self.request_sequence = EventQueue(), 0
        self.schedule = iter(schedule_packets(self.shape, mapping))
        self.schedule_exhausted, self.packets = False, deque()
        self.a = config["architecture"]
        s = self.shape
        input_register_bytes = self.traits["activation_register_bytes"]
        self.operand_storage = Capacity("activation_registers", input_register_bytes)
        self.accumulators = Capacity("accumulators", self.traits["accumulator_register_bytes"])
        self.post_storage = Capacity("post_pipeline", 2*56)
        self.pre_ln = Capacity("pre_ln_active_rows", s.m*s.n*2)
        self.output = Capacity("resident_output", s.m*s.n)
        self.ln_workspace = Capacity("ln_workspace",s.n*3)
        self.partial = Capacity("partial_sums", max(1, 8*min(s.m, self.traits["group_rows"])))
        self.acc_owner = None
        self.matrix_busy, self.partial_hold = False, False
        self.next_matrix_issue = 0
        self.draining = deque()
        self.latch_row, self.latch_ready, self.latch_pending = None, False, False
        self.post_jobs, self.next_post_issue = [], 0
        self.matrix_masks, self.ln_masks = {}, {}
        self.ready_rows, self.outputs = deque(), {}
        self.ln_job, self.next_ln_start = None, 0
        self.partial_ready = set()
        self.issued, self.useful_mac = 0, 0
        self.milestones, self.cycle_breakdown = {}, Counter()
        self.register_activity = Counter()
        self.logical_weight_consumers = 0
        self.peak_operand_packets = 0
        self.row_matrix_cycles, self.row_ln_cycles = {}, {}
        self.peak_post_jobs = 0

    def _log(self, event, **fields):
        if self.capture_trace:
            self.trace.append(dict(cycle=self.cycle, event=event, **fields))

    def _first(self, key):
        self.milestones.setdefault(key, self.cycle)

    def _later(self, delay, event, callback, **fields):
        def done(_):
            self._log(event, **fields)
            callback()
        self.events.schedule(self.cycle, delay, event, fields, done)

    def _request(self, owner, write, accesses, callback, **tag):
        request_id = "local{:08d}".format(self.request_sequence)
        self.request_sequence += 1
        self.local.submit(request_id, owner, write, accesses, callback, tag)

    def _prefetch_weights(self):
        while len(self.packets) < 2 and not self.schedule_exhausted:
            try:
                desc = next(self.schedule)
            except StopIteration:
                self.schedule_exhausted = True
                break
            consumers = 1 if self.array else len(desc.rows)
            assert self.weights.reserve(desc.packet_id, desc.weight_address, desc.weight_bytes,
                                        consumers, self.cycle,
                                        {"tile_id":desc.tile_id,"rows":list(desc.rows),
                                         "columns":list(desc.columns),"reduction":list(desc.reduction),
                                         "logical_reuse":len(desc.rows)})
            self.packets.append({"desc":desc,"operands":deque(),"next_row":0,"completed":0})

    def _partial_key(self, desc, row):
        return (desc.tile_id, row)

    def _operand(self, packet, row=None):
        desc = packet["desc"]
        rows = desc.rows if self.array else (row,)
        key = desc.packet_id if self.array else desc.packet_id+"_t"+str(row)
        size = len(rows)*len(desc.reduction)
        operand = {"owner":key,"row":row,"rows":rows,"input_ready":False,
                   "partial_ready":not self.b1 or desc.first_reduction,"issued":False}
        self.operand_storage.reserve(key,size)
        accesses = []
        for token in rows:
            accesses.extend(self.layout.accesses("input",token,desc.reduction))
        def input_ready():
            operand["input_ready"] = True
            self.operand_storage.ready(key)
            self._log("ACTIVATION_OPERANDS_READY",packet_id=desc.packet_id,token=row,rows=list(rows))
        self._request("matrix_activation",False,accesses,input_ready,tensor="input",packet_id=desc.packet_id,token=row)
        if self.b1 and not desc.first_reduction:
            partial_key = self._partial_key(desc,row)
            assert partial_key in self.partial_ready
            local_row = row-desc.rows[0]
            def partial_read():
                operand["partial_ready"] = True
                self.partial_ready.remove(partial_key)
                self.partial.release(str(row))
            self._request("partial_read",False,self.layout.accesses("partial",local_row,range(len(desc.columns))),
                          partial_read,tensor="partial",packet_id=desc.packet_id,token=row)
        packet["operands"].append(operand)
        return operand

    def _prepare_operands(self):
        if not self.packets:
            return
        if self.mapping == "B0":
            desc = self.packets[0]["desc"]
            row = desc.rows[0]
            if self.latch_row != row and not self.matrix_busy:
                if self.latch_row is not None:
                    self.operand_storage.release("row_latch")
                self.latch_row, self.latch_pending, self.latch_ready = row, True, False
                self.operand_storage.reserve("row_latch",self.shape.r)
                def ready():
                    self.latch_pending, self.latch_ready = False,True
                    self.operand_storage.ready("row_latch")
                    self._log("INPUT_ROW_LATCHED",token=row,bytes=self.shape.r)
                self._request("matrix_activation",False,self.layout.accesses("input",row,range(self.shape.r)),
                              ready,tensor="input",token=row)
            return
        if self.array:
            for packet in self.packets:
                if not packet["operands"]:
                    self._operand(packet)
        else:
            packet = self.packets[0]
            desc = packet["desc"]
            while len(packet["operands"]) < 2 and packet["next_row"] < len(desc.rows):
                row = desc.rows[packet["next_row"]]
                if not desc.first_reduction and self._partial_key(desc,row) not in self.partial_ready:
                    break
                self._operand(packet,row)
                packet["next_row"] += 1
        self.peak_operand_packets = max(self.peak_operand_packets,
                                       sum(len(p["operands"]) for p in self.packets))

    def _matrix_row_masks(self,desc,rows):
        mask = sum(1<<channel for channel in desc.columns)
        for row in rows:
            if self.matrix_masks.get(row,0)&mask:
                raise AssertionError("matrix channel produced twice")
            self.matrix_masks[row] = self.matrix_masks.get(row,0)|mask
            self._first("first_matrix_fragment")
            self._log("MAT_FRAGMENT_READY",token=row,channels=list(desc.columns),packet_id=desc.packet_id)
            if self.matrix_masks[row] == (1<<self.shape.n)-1:
                self.row_matrix_cycles[row] = self.cycle
                self._first("first_matrix_row")
                self._log("MATRIX_ROW_COMPLETE",token=row)

    def _finish_packet_consumer(self,packet,operand):
        desc = packet["desc"]
        rows = desc.rows if self.array else desc.rows if self.mapping == "B0" else (operand["row"],)
        self.matrix_busy = False
        self.weights.consume(desc.packet_id,self.cycle)
        packet["completed"] += 1
        self.logical_weight_consumers += len(rows)
        if self.mapping != "B0":
            assert packet["operands"].popleft() is operand
            self.operand_storage.release(operand["owner"])
        self.accumulators.ready(self.acc_owner)
        self._log("MAC_COMPLETE",packet_id=desc.packet_id,rows=list(rows))
        if self.b1 and not desc.last_reduction:
            row = rows[0]
            local_row = row-desc.rows[0]
            partial_key = self._partial_key(desc,row)
            self.partial.reserve(str(row),len(desc.columns)*4)
            self.partial_hold = True
            def saved():
                self.partial.ready(str(row))
                self.partial_ready.add(partial_key)
                self.accumulators.release(self.acc_owner)
                self.acc_owner = None
                self.partial_hold = False
                self._log("PARTIAL_SUM_SAVED",packet_id=desc.packet_id,token=row)
            self._request("partial_write",True,self.layout.accesses("partial",local_row,range(len(desc.columns))),
                          saved,tensor="partial",packet_id=desc.packet_id,token=row)
        elif desc.last_reduction:
            self._matrix_row_masks(desc,rows)
            for row in rows:
                for index in range(0,len(desc.columns),self.a["post_channels_per_cycle"]):
                    self.draining.append({"row":row,"channels":desc.columns[index:index+self.a["post_channels_per_cycle"]],
                                          "tile_id":desc.tile_id})
        if packet["completed"] == (1 if self.array else len(desc.rows)):
            assert not packet["operands"]
            assert self.packets.popleft() is packet

    def _matrix_step(self):
        if self.matrix_busy or self.partial_hold or self.draining or not self.packets:
            return False
        packet = self.packets[0]
        desc = packet["desc"]
        if not self.weights.ready(desc.packet_id) or self.cycle < self.next_matrix_issue:
            return False
        if self.mapping == "B0":
            if not self.latch_ready or self.latch_row != desc.rows[0]:
                return False
            operand = None
            rows = desc.rows
        else:
            if not packet["operands"]:
                return False
            operand = packet["operands"][0]
            if not operand["input_ready"] or not operand["partial_ready"] or operand["issued"]:
                return False
            operand["issued"] = True
            rows = operand["rows"]
        owner = desc.tile_id if not self.b1 else desc.packet_id+"_t"+str(rows[0])
        if self.acc_owner is None:
            self.accumulators.reserve(owner,len(rows)*len(desc.columns)*4)
            self.acc_owner = owner
        elif self.acc_owner != owner:
            raise AssertionError("accumulator overwritten before previous result consumed")
        self.matrix_busy = True
        self.next_matrix_issue = self.cycle+self.a["accumulator_feedback"]
        self.issued += 1
        useful = len(rows)*len(desc.columns)*len(desc.reduction)
        self.useful_mac += useful
        self.register_activity["activation_element_reads"] += len(rows)*len(desc.reduction)
        self.register_activity["weight_element_reads"] += len(desc.columns)*len(desc.reduction)
        self.register_activity["accumulator_updates"] += len(rows)*len(desc.columns)
        self._log("COMPUTE_ISSUE",packet_id=desc.packet_id,rows=list(rows),token=rows[0],
                  columns=list(desc.columns),reduction_start=desc.reduction[0],
                  reduction_count=len(desc.reduction),useful_mac=useful,
                  complete_cycle=self.cycle+self.a["multiply_latency"])
        self._later(self.a["multiply_latency"],"MULTIPLY_PIPELINE_DONE",
                    lambda:self._finish_packet_consumer(packet,operand),packet_id=desc.packet_id)
        return True

    def _drain_step(self):
        if not self.draining or len(self.post_jobs) >= 2:
            return
        fragment = self.draining.popleft()
        row, columns = fragment["row"],fragment["channels"]
        owner = "post{}".format(self.request_sequence)
        # Two bounded post packages each hold <=8 INT32,8 INT8,8 INT16 values.
        self.post_storage.reserve(owner,len(columns)*7)
        if row not in self.pre_ln.allocations:
            self.pre_ln.reserve(row,self.shape.n*2)
        job = {"owner":owner,"row":row,"channels":columns,"state":"residual_wait"}
        self.post_jobs.append(job)
        self.peak_post_jobs = max(self.peak_post_jobs,len(self.post_jobs))
        def ready():
            job["state"] = "arithmetic_wait"
            self.post_storage.ready(owner)
            self._log("POST_RESIDUAL_READY",token=row,channels=list(columns))
        self._request("post_residual",False,self.layout.accesses("residual",row,columns),ready,
                      tensor="residual",token=row,channels=list(columns))
        self._log("ACCUMULATOR_FRAGMENT_COPIED",token=row,channels=list(columns))
        if not self.draining:
            self.accumulators.release(self.acc_owner)
            self.acc_owner = None

    def _post_step(self):
        if self.cycle < self.next_post_issue:
            return
        job = next((j for j in self.post_jobs if j["state"] == "arithmetic_wait"),None)
        if job is None:
            return
        row,columns = job["row"],job["channels"]
        job["state"] = "arithmetic_run"
        self.next_post_issue = self.cycle+1
        self.register_activity["bias_reads_once"] += len(columns)
        self._log("POST_ISSUE",token=row,channels=list(columns),complete_cycle=self.cycle+self.a["post_latency"])
        def arithmetic_done():
            job["state"] = "write_wait"
            def written():
                mask = sum(1<<channel for channel in columns)
                if self.ln_masks.get(row,0)&mask:
                    raise AssertionError("duplicate pre-LN channel")
                self.ln_masks[row] = self.ln_masks.get(row,0)|mask
                self.pre_ln.set_live(row,bin(self.ln_masks[row]).count("1")*2)
                self.post_storage.release(job["owner"])
                self.post_jobs.remove(job)
                self._log("POST_WRITE_DONE",token=row,channels=list(columns))
                if self.ln_masks[row] == (1<<self.shape.n)-1:
                    self.ready_rows.append(row)
                    self._first("first_ln_input_row")
                    self._log("LN_ROW_ASSEMBLED",token=row)
            self._request("post_pre_ln",True,self.layout.accesses("pre_ln",row,columns),written,
                          tensor="pre_ln",token=row,channels=list(columns))
        self._later(self.a["post_latency"],"POST_ARITHMETIC_DONE",arithmetic_done,
                    token=row,channels=list(columns))

    def _ln_step(self):
        if self.ln_job is not None or not self.ready_rows:
            return
        row = self.ready_rows.popleft()
        self.output.reserve(row,self.shape.n)
        self.ln_workspace.reserve(row,self.shape.n*3)
        self.ln_job = {"row":row,"state":"input_read"}
        def input_ready():
            self.pre_ln.release(row)
            self.ln_workspace.set_live(row,self.shape.n*2)
            self.ln_job["state"] = "core_wait"
            self._log("LN_INPUT_READY",token=row)
        self._request("ln_input",False,self.layout.accesses("pre_ln",row,range(self.shape.n)),input_ready,
                      tensor="pre_ln",token=row)

    def _ln_core_step(self):
        if self.ln_job is None or self.ln_job["state"] != "core_wait" or self.cycle < self.next_ln_start:
            return
        job, row = self.ln_job,self.ln_job["row"]
        assert self.ln_masks[row] == (1<<self.shape.n)-1
        assert self.matrix_masks[row] == (1<<self.shape.n)-1
        job["state"] = "core_run"
        latency = self.cfg["ln"]["latency_cycles"]
        self.next_ln_start = self.cycle+self.cfg["ln"]["initiation_interval_cycles"]
        self._first("first_ln_accept")
        self.register_activity["gamma_beta_reads"] += self.shape.n*2
        self._log("LN_ACCEPT",token=row,complete_cycle=self.cycle+latency)
        def core_done():
            job["state"] = "output_write"
            self.ln_workspace.set_live(row,self.shape.n)
            self.row_ln_cycles[row] = self.cycle
            self._first("first_ln_row_done")
            def output_done():
                self.outputs[row] = self.cycle
                self.output.ready(row)
                self.ln_workspace.release(row)
                self.ln_job = None
                self._first("first_internal_output_confirm")
                self._log("INTERNAL_OUTPUT_CONFIRMED",token=row,bytes=self.shape.n,dtype="int8")
            self._request("ln_output",True,self.layout.accesses("output",row,range(self.shape.n)),output_done,
                          tensor="output",token=row)
        self._later(latency,"LN_CORE_DONE",core_done,token=row)

    def _cycle_reason(self,issued):
        if issued:
            return "matrix_issue"
        if self.matrix_busy:
            return "multiply_pipeline"
        if self.partial_hold:
            return "partial_sum_writeback"
        if self.draining:
            return "matrix_result_post_backpressure"
        if self.packets:
            if not self.weights.ready(self.packets[0]["desc"].packet_id):
                return "weight_packet_fill"
            if self.cycle < self.next_matrix_issue:
                return "accumulator_feedback"
            return "local_activation_or_partial_read"
        return "post_ln_local_output_tail"

    def run(self,max_cycles=10000000):
        for cycle in range(max_cycles):
            self.cycle = cycle
            self.weights.complete(cycle)
            self.local.complete(cycle)
            self.events.complete(cycle)
            if len(self.outputs) == self.shape.m:
                if self.weights.pending or self.local.pending or self.events.pending:
                    raise AssertionError("final output left pending operations")
                return self._result()
            self._prefetch_weights()
            self._prepare_operands()
            self._ln_core_step()
            self._ln_step()
            self._post_step()
            self._drain_step()
            issued = self._matrix_step()
            self.weights.issue(cycle)
            self.local.arbitrate(cycle)
            self.cycle_breakdown[self._cycle_reason(issued)] += 1
        raise RuntimeError("fixed model timeout: cycle={} packets={} matrix_busy={} partial_hold={} post={} LN={} local={}".format(
            self.cycle,[(p["desc"].packet_id,p["completed"]) for p in self.packets],self.matrix_busy,
            self.partial_hold,len(self.post_jobs),self.ln_job,self.local.stats()))

    def _initialization(self):
        s = self.shape
        writes = {}
        counts = [0]*self.local.banks
        for tensor,rows,cols in (("input",s.m,s.r),("residual",s.m,s.n)):
            accesses = {}
            for row in range(rows):
                for bank,word,mask in self.layout.accesses(tensor,row,range(cols)):
                    accesses[(bank,word)] = accesses.get((bank,word),0)|mask
            per_bank = [0]*self.local.banks
            for bank,_ in accesses:
                per_bank[bank] += 1
                counts[bank] += 1
            writes[tensor] = {"useful_bytes":rows*cols,"physical_word_writes":len(accesses),"per_bank":per_bank}
        parameters = self.shape.n*(self.cfg["numeric"]["bias_bits"]+2*self.cfg["numeric"]["ln_parameter_bits"])//8
        return {"included_in_main_window":False,
                "activation_residual_local_image":writes,
                "local_image_install_cycles":max(counts)+1,
                "local_image_install_method":"prearranged_image_one_write_per_bank_per_cycle_plus_ack",
                "external_activation_load_cycles":None,
                "external_activation_bandwidth":"not specified; not charged to dedicated weight port",
                "activation_layout_repack_bytes":s.m*s.r,
                "activation_layout_repack_cycles":None,
                "parameter_register_bytes":parameters,"parameter_install_cycles":None,
                "parameter_source":"separate pre-test bias/gamma/beta image; metadata timing mode does not load checkpoint values",
                "activation_source":"prearranged local input/residual image; external transfer and repacking precede the main window",
                "shared_weights_layout":self.traits["weight_layout"],
                "shared_weights_valid_bytes":s.r*s.n,
                "shared_weights_packed_address_bytes":self.traits["weight_storage_bytes"],
                "weight_repack_read_bytes":s.r*s.n,"weight_repack_write_bytes":s.r*s.n,
                "weight_repack_cycles":None,
                "status":"local image installed before measurement; source transfer and rearrangement timing unknown, separately disclosed"}

    def _result(self):
        s = self.shape
        assert self.useful_mac == s.useful_mac
        assert self.register_activity["bias_reads_once"] == s.m*s.n
        assert set(self.outputs) == set(range(s.m))
        assert all(v == (1<<s.n)-1 for v in self.matrix_masks.values())
        assert all(v == (1<<s.n)-1 for v in self.ln_masks.values())
        assert len(self.matrix_masks) == len(self.ln_masks) == s.m
        assert sum(self.cycle_breakdown.values()) == self.cycle
        assert self.pre_ln.reserved == self.post_storage.reserved == self.accumulators.reserved == 0
        assert self.partial.reserved == 0
        assert self.ln_workspace.reserved == 0
        if self.mapping == "B0" and self.latch_row is not None:
            self.operand_storage.release("row_latch")
        assert self.operand_storage.reserved == 0
        buffers = {b.name:b.stats() for b in (self.operand_storage,self.accumulators,self.post_storage,self.pre_ln,self.output,self.ln_workspace)}
        if self.b1:
            buffers[self.partial.name] = self.partial.stats()
        resources = resource_inventory(self.cfg,self.mapping,s,{
            "weight_register_bytes":2*self.traits["weight_packet_slot_bytes"],
            "activation_register_bytes":self.operand_storage.capacity_bytes,
            "accumulator_register_bytes":self.accumulators.capacity_bytes,
            "post_pipeline_register_bytes":self.post_storage.capacity_bytes,
            "partial_operand_register_bytes":16 if self.b1 else 0,
            "ln_workspace_bytes":s.n*3,
            "b1_partial_sram_bytes":8*min(s.m,self.traits["group_rows"]) if self.b1 else 0,
            "local_reserved_bytes":self.layout.used_bytes})
        multiplier_count = self.traits["physical_multipliers"]
        weight_stats,local_stats = self.weights.stats(),self.local.stats()
        buffers["weight_registers"] = {"capacity_bytes":weight_stats["register_capacity_bytes"],
                                     "peak_reserved_bytes":weight_stats["register_peak_reserved_bytes"],
                                     "peak_live_bytes":weight_stats["register_peak_live_bytes"],
                                     "final_reserved_bytes":weight_stats["reserved_bytes"]}
        weight_stats["logical_token_consumers"] = self.logical_weight_consumers
        weight_stats["token_reuse_beyond_first"] = self.logical_weight_consumers-weight_stats["fill_count"]
        return {"mapping":self.mapping,"architecture":self.traits["architecture"],
                "operator":self.operator,"shape":asdict(s),
                "scope":"first_weight_request_to_final_internal_int8_output_ack",
                "model_level":"fixed300_weight_port_local_banks_ln{}_event_metadata".format(self.cfg["ln"]["latency_cycles"]),
                "total_cycles":self.cycle,"first_row_cycles":self.milestones["first_internal_output_confirm"],
                "milliseconds_at_assumed_clock":self.cycle/self.cfg["platform"]["clock_hz"]*1000,
                "milestones":dict(self.milestones),
                "first_matrix_fragment":self.milestones["first_matrix_fragment"],
                "first_matrix_row":self.milestones["first_matrix_row"],
                "first_ln_row_done":self.milestones["first_ln_row_done"],
                "last_internal_output_confirm":max(self.outputs.values()),
                "row_matrix_cycles":[self.row_matrix_cycles[r] for r in range(s.m)],
                "row_ln_core_done_cycles":[self.row_ln_cycles[r] for r in range(s.m)],
                "row_output_cycles":[self.outputs[r] for r in range(s.m)],
                "useful_mac":self.useful_mac,"issued_updates":self.issued,
                "physical_multipliers":multiplier_count,
                "issue_lane_utilization":self.useful_mac/(multiplier_count*self.issued),
                "issue_duty_cycle":self.issued/self.cycle,
                "compute_utilization":self.useful_mac/(multiplier_count*self.cycle),
                "utilization_denominators":{"issue_lane_utilization":multiplier_count*self.issued,
                                             "issue_duty_cycle":self.cycle,
                                             "compute_utilization":multiplier_count*self.cycle},
                "weight_port":weight_stats,"local_memory":local_stats,
                "shared_activation_read_bytes":0,"shared_activation_write_bytes":0,
                "buffers":buffers,"physical_local_layout":self.layout.stats(),
                "cycle_breakdown":dict(self.cycle_breakdown),
                "register_activity":dict(self.register_activity),
                "peak_operand_packets":self.peak_operand_packets,"peak_post_jobs":self.peak_post_jobs,
                "resources":resources,"initialization":self._initialization(),
                "output_rows":s.m,"output_elements":s.m*s.n,
                "claims":{"bit_accurate_fixed_point_ln":False,"rtl_calibrated":False,
                          "accuracy_measured":False,"system_cold_start_timed":False}}

    def merged_trace(self):
        return sorted(self.trace+self.weights.trace+self.local.trace,key=lambda item:item["cycle"])


def simulate(config,mapping,operator="projection",shape=None,capture_trace=True):
    simulator = FusedSimulator(config,mapping,operator,shape,capture_trace)
    result = simulator.run()
    return result,simulator.merged_trace() if capture_trace else []
