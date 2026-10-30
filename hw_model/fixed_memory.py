"""Memory primitives for the fixed 150 MHz, 300 MB/s behavior model.

Weights use a dedicated 4-byte port, II=2, response latency=3, and at most two
outstanding beats. Its two packet registers are the only local weight storage.
Activations, residuals, pre-LN values, results and B1 partials occupy four or
eight 4096-word, 16-bit single-port banks, with latency one. The default remains
eight banks (64 KiB); the revised profile has four banks (32 KiB).
"""

from collections import deque
from dataclasses import dataclass
import heapq
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

from .compute import Shape
from .fixed_schedule import SUPPORTED_MAPPINGS, mapping_traits
from .memory import BankedSRAM


LOCAL_BYTES = 65536
WEIGHT_ADDRESS_BYTES = 524288


def _int(name, value, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError("{} must be an integer >= {}".format(name, minimum))


def _name(name, value):
    if not isinstance(value, str) or not value:
        raise ValueError("{} must be a nonempty string".format(name))


def _mask_bytes(mask):
    return (mask & 1) + ((mask >> 1) & 1)


def _bank_count(banks):
    _int("banks", banks, 1)
    if banks not in (4, 8):
        raise ValueError("local SRAM must have four or eight banks")
    return banks


class LocalCapacityError(ValueError):
    """A valid layout cannot reside in the selected physical local SRAM.

    This distinguishes a real capacity failure from malformed parameters or
    unsupported mappings. No spilling, resizing, or tensor lifetime change is
    implied by this exception. Descriptors are preserved for experiment logs.
    """

    def __init__(self, required_bytes, capacity_bytes, tensors, mapping):
        self.required_bytes = required_bytes
        self.capacity_bytes = capacity_bytes
        self.tensors = {name: dict(descriptor) for name, descriptor in tensors.items()}
        self.mapping = mapping
        super().__init__("local tensors require {} bytes, exceeding fixed {}-byte SRAM".format(
            required_bytes, capacity_bytes))


@dataclass
class _Packet:
    packet_id: str
    slot: int
    address: int
    useful_bytes: int
    consumers: int
    ready_cycle: int
    tag: Any
    beats: List[Tuple[int, int, int, int]]
    next_beat: int = 0
    valid_mask: int = 0
    completed_consumers: int = 0
    fill_cycle: Optional[int] = None


class WeightPort:
    """Fixed-rate, pipelined weight bus with two bounded packet registers.

    ``consume`` denotes consumer *completion*, not issue. A READY packet remains
    allocated until every declared consumer has completed. ``complete`` is
    applied once per cycle; ``issue`` accepts at most one beat per call/cycle
    and applies due responses first if the caller has not already done so.
    """

    beat_bytes = 4
    accept_interval = 2
    response_latency = 3
    max_outstanding = 2
    slots = 2
    address_capacity_bytes = WEIGHT_ADDRESS_BYTES

    def __init__(self, packet_bytes, capture_trace=True):
        _int("packet_bytes", packet_bytes, 1)
        self.packet_bytes = packet_bytes
        self.capture_trace, self.trace = capture_trace, []
        self._packets = {}  # type: Dict[str, _Packet]
        self._order = []  # type: List[str]
        self._ids = set()
        self._inflight = []
        self._sequence = 0
        self._cycle = self._last_issue = -1
        self._next_accept = 0
        self._stats = {
            "reserved_packets": 0, "fill_count": 0, "reuse_count": 0,
            "released_packets": 0, "consumers_completed": 0,
            "useful_weight_bytes": 0, "physical_weight_bytes": 0,
            "accepted_beats": 0, "completed_beats": 0,
            "bus_busy_cycles": 0, "occupied_service_cycles": 0,
            "max_outstanding": 0, "peak_reserved_bytes": 0,
            "peak_live_bytes": 0, "reservation_denied": 0,
            "outstanding_stall_cycles": 0,
        }

    def _event(self, event, cycle, packet=None, **fields):
        if self.capture_trace:
            item = {"event": event, "cycle": cycle}
            if packet is not None:
                item.update(packet_id=packet.packet_id, slot=packet.slot, tag=packet.tag)
            item.update(fields)
            self.trace.append(item)

    def _peaks(self):
        reserved = len(self._packets) * self.packet_bytes
        live = sum(bin(packet.valid_mask).count("1") for packet in self._packets.values())
        self._stats["peak_reserved_bytes"] = max(self._stats["peak_reserved_bytes"], reserved)
        self._stats["peak_live_bytes"] = max(self._stats["peak_live_bytes"], live)

    def reserve(self, packet_id, address, useful_bytes, consumers,
                ready_cycle=0, tag=None):
        _name("packet_id", packet_id)
        _int("address", address)
        _int("useful_bytes", useful_bytes, 1)
        _int("consumers", consumers, 1)
        _int("ready_cycle", ready_cycle)
        if useful_bytes > self.packet_bytes:
            raise ValueError("useful packet bytes exceed physical packet register")
        if address + useful_bytes > self.address_capacity_bytes:
            raise ValueError("weight source exceeds the fixed 512 KiB address space")
        if packet_id in self._ids:
            raise ValueError("duplicate packet_id: " + packet_id)
        if len(self._packets) == self.slots:
            self._stats["reservation_denied"] += 1
            return False
        occupied = {packet.slot for packet in self._packets.values()}
        slot = next(index for index in range(self.slots) if index not in occupied)
        start, end = address // 4 * 4, address + useful_bytes
        beats = []
        for base in range(start, end, 4):
            lo, hi = max(base, address), min(base + 4, end)
            bus_mask = ((1 << (hi - lo)) - 1) << (lo - base)
            packet_mask = ((1 << (hi - lo)) - 1) << (lo - address)
            beats.append((base, hi - lo, bus_mask, packet_mask))
        packet = _Packet(packet_id, slot, address, useful_bytes, consumers,
                         ready_cycle, tag, beats)
        self._packets[packet_id] = packet
        self._order.append(packet_id)
        self._ids.add(packet_id)
        self._stats["reserved_packets"] += 1
        self._peaks()
        self._event("PACKET_RESERVED", max(0, self._cycle), packet,
                    address=address, useful_bytes=useful_bytes,
                    consumers=consumers, ready_cycle=ready_cycle)
        return True

    def ready(self, packet_id):
        packet = self._packets.get(packet_id)
        return packet is not None and packet.fill_cycle is not None

    def valid_mask(self, packet_id):
        if packet_id not in self._packets:
            raise ValueError("packet is not reserved: " + str(packet_id))
        return self._packets[packet_id].valid_mask

    def complete(self, cycle):
        _int("cycle", cycle)
        if cycle <= self._cycle:
            raise ValueError("weight completion cycles must be strictly increasing")
        self._cycle = cycle
        while self._inflight and self._inflight[0][0] <= cycle:
            due, _, packet_id, base, valid_bytes, byte_mask, packet_mask = heapq.heappop(self._inflight)
            packet = self._packets[packet_id]
            if packet.valid_mask & packet_mask:
                raise AssertionError("weight byte arrived twice")
            packet.valid_mask |= packet_mask
            self._stats["completed_beats"] += 1
            self._event("WEIGHT_BEAT_DONE", due, packet, address=base,
                        valid_bytes=valid_bytes, byte_mask=byte_mask,
                        packet_byte_mask=packet_mask, valid_mask=packet.valid_mask)
            if packet.valid_mask == (1 << packet.useful_bytes) - 1:
                packet.fill_cycle = due
                self._stats["fill_count"] += 1
                self._event("PACKET_READY", due, packet,
                            useful_bytes=packet.useful_bytes,
                            valid_mask=packet.valid_mask)
            self._peaks()

    def issue(self, cycle):
        _int("cycle", cycle)
        if cycle <= self._last_issue:
            raise ValueError("weight issue cycles must be strictly increasing")
        if cycle < self._cycle:
            raise ValueError("weight cycles must not go backwards")
        if cycle > self._cycle:
            self.complete(cycle)
        self._last_issue = cycle
        if cycle < self._next_accept:
            return
        packet = next((self._packets[packet_id] for packet_id in self._order
                       if self._packets[packet_id].ready_cycle <= cycle and
                       self._packets[packet_id].next_beat < len(self._packets[packet_id].beats)), None)
        if packet is None:
            return
        if len(self._inflight) >= self.max_outstanding:
            self._stats["outstanding_stall_cycles"] += 1
            return
        base, valid_bytes, byte_mask, packet_mask = packet.beats[packet.next_beat]
        beat = packet.next_beat
        packet.next_beat += 1
        self._sequence += 1
        due = cycle + self.response_latency
        heapq.heappush(self._inflight, (due, self._sequence, packet.packet_id,
                                      base, valid_bytes, byte_mask, packet_mask))
        self._next_accept = cycle + self.accept_interval
        self._stats["accepted_beats"] += 1
        self._stats["bus_busy_cycles"] += 1
        self._stats["occupied_service_cycles"] += self.accept_interval
        self._stats["useful_weight_bytes"] += valid_bytes
        self._stats["physical_weight_bytes"] += self.beat_bytes
        self._stats["max_outstanding"] = max(self._stats["max_outstanding"], len(self._inflight))
        self._event("WEIGHT_BEAT_ISSUE", cycle, packet, address=base, beat=beat,
                    valid_bytes=valid_bytes, byte_mask=byte_mask,
                    packet_byte_mask=packet_mask, complete_cycle=due)

    def consume(self, packet_id, cycle, amount=1):
        _int("amount", amount, 1)
        _int("cycle", cycle)
        if cycle < self._cycle:
            raise ValueError("weight cycles must not go backwards")
        if cycle > self._cycle:
            self.complete(cycle)
        packet = self._packets.get(packet_id)
        if packet is None:
            raise ValueError("packet is not reserved: " + str(packet_id))
        if packet.fill_cycle is None:
            raise ValueError("packet is not READY")
        if packet.completed_consumers + amount > packet.consumers:
            raise ValueError("too many packet consumers completed")
        previous = packet.completed_consumers
        packet.completed_consumers += amount
        self._stats["consumers_completed"] += amount
        self._stats["reuse_count"] += amount - (1 if previous == 0 else 0)
        self._event("PACKET_CONSUME", cycle, packet, amount=amount,
                    consumers_completed=packet.completed_consumers,
                    consumers=packet.consumers)
        if packet.completed_consumers == packet.consumers:
            self._event("PACKET_RELEASE", cycle, packet)
            del self._packets[packet_id]
            self._order.remove(packet_id)
            self._stats["released_packets"] += 1

    @property
    def pending(self):
        return bool(self._packets) or bool(self._inflight)

    def stats(self):
        result = dict(self._stats)
        result.update(packet_bytes=self.packet_bytes,
                      register_capacity_bytes=self.slots * self.packet_bytes,
                      reserved_bytes=len(self._packets) * self.packet_bytes,
                      live_bytes=sum(bin(p.valid_mask).count("1") for p in self._packets.values()),
                      active_packets=len(self._packets), outstanding_beats=len(self._inflight),
                      register_peak_reserved_bytes=self._stats["peak_reserved_bytes"],
                      register_peak_live_bytes=self._stats["peak_live_bytes"],
                      valid_byte_masks={key: p.valid_mask for key, p in self._packets.items()})
        return result


@dataclass
class _LocalRequest:
    request_id: str
    owner: str
    write: bool
    queues: List[Deque[Tuple[int, int]]]
    remaining: int
    callback: Callable[[], None]
    tag: Any
    outstanding: int = 0


class LocalMemory:
    """One fixed local SRAM, with per-bank round-robin owner arbitration.

    Requests are FIFO within each owner and may complete over several cycles.
    Duplicate word accesses inside one request combine byte masks; a read of a
    packed pair therefore uses one physical word port. Cross-request broadcasts
    are not assumed. The caller bounds the number of submitted control requests;
    this object adds no uncounted data buffers or tensor copies.
    """

    banks = 8
    depth_words = 4096
    word_bytes = 2
    latency = 1
    capacity_bytes = LOCAL_BYTES

    def __init__(self, capture_trace=True, banks=8):
        self.banks = _bank_count(banks)
        self.capacity_bytes = self.banks * self.depth_words * self.word_bytes
        self.capture_trace, self.trace = capture_trace, []
        self.sram = BankedSRAM("fixed_local", self.banks, self.depth_words,
                              "single", self.latency, capture_trace=False)
        self._queues, self._owners = {}, []
        self._requests, self._ids = {}, set()
        self._cursor = [0] * self.banks
        self._inflight, self._sequence = [], 0
        self._cycle = self._last_arbitrate = -1
        self._stats = {"submitted_requests": 0, "completed_requests": 0,
                       "read_useful_bytes": 0, "write_useful_bytes": 0,
                       "physical_read_words": 0, "physical_write_words": 0,
                       "completed_words": 0, "denied_cycles": 0,
                       "peak_pending_requests": 0, "peak_outstanding_words": 0}
        self._per_owner = {}
        self._per_bank = [{"read_words": 0, "write_words": 0,
                           "read_useful_bytes": 0, "write_useful_bytes": 0,
                           "denied_cycles": 0} for _ in range(self.banks)]

    def _event(self, event, cycle, request=None, **fields):
        if self.capture_trace:
            item = {"event": event, "cycle": cycle}
            if request is not None:
                item.update(request_id=request.request_id, owner=request.owner,
                            write=request.write, tag=request.tag)
            item.update(fields)
            self.trace.append(item)

    def submit(self, request_id, owner, write, accesses, callback, tag=None):
        _name("request_id", request_id)
        _name("owner", owner)
        if request_id in self._ids:
            raise ValueError("duplicate local request_id: " + request_id)
        if not isinstance(write, bool):
            raise ValueError("write must be a bool")
        if not callable(callback):
            raise ValueError("callback must be callable")
        masks = {}
        for access in accesses:
            if not isinstance(access, (tuple, list)) or len(access) != 3:
                raise ValueError("local access must be (bank, word, byte_mask)")
            bank, word, mask = access
            _int("bank", bank)
            _int("word", word)
            _int("byte_mask", mask, 1)
            if bank >= self.banks or word >= self.depth_words or mask > 3:
                raise ValueError("local address or byte mask exceeds physical SRAM")
            masks[(bank, word)] = masks.get((bank, word), 0) | mask
        if not masks:
            raise ValueError("local requests must contain at least one physical word")
        queues = [deque() for _ in range(self.banks)]
        for (bank, word), mask in masks.items():
            queues[bank].append((word, mask))
        request = _LocalRequest(request_id, owner, write, queues, len(masks), callback, tag)
        if owner not in self._queues:
            self._queues[owner] = deque()
            self._owners.append(owner)
            self._per_owner[owner] = {"read_useful_bytes": 0, "write_useful_bytes": 0,
                                      "physical_read_words": 0, "physical_write_words": 0,
                                      "completed_requests": 0, "denied_cycles": 0}
        self._queues[owner].append(request)
        self._requests[request_id] = request
        self._ids.add(request_id)
        self._stats["submitted_requests"] += 1
        self._stats["peak_pending_requests"] = max(self._stats["peak_pending_requests"], len(self._requests))
        self._event("LOCAL_REQUEST", max(0, self._cycle), request,
                    words=len(masks), useful_bytes=sum(_mask_bytes(mask) for mask in masks.values()))

    def complete(self, cycle):
        _int("cycle", cycle)
        if cycle <= self._cycle:
            raise ValueError("local completion cycles must be strictly increasing")
        self._cycle = cycle
        while self._inflight and self._inflight[0][0] <= cycle:
            due, _, request_id, bank, word, mask = heapq.heappop(self._inflight)
            request = self._requests[request_id]
            request.outstanding -= 1
            self._stats["completed_words"] += 1
            self._event("LOCAL_WORD_DONE", due, request, bank=bank, word_index=word,
                        byte_mask=mask)
            if request.remaining == 0 and request.outstanding == 0:
                assert self._queues[request.owner][0] is request
                self._queues[request.owner].popleft()
                del self._requests[request_id]
                self._stats["completed_requests"] += 1
                self._per_owner[request.owner]["completed_requests"] += 1
                self._event("LOCAL_REQUEST_DONE", due, request)
                request.callback()

    def arbitrate(self, cycle):
        _int("cycle", cycle)
        if cycle <= self._last_arbitrate:
            raise ValueError("local arbitration cycles must be strictly increasing")
        if cycle < self._cycle:
            raise ValueError("local cycles must not go backwards")
        if cycle > self._cycle:
            self.complete(cycle)
        self._last_arbitrate = cycle
        owner_count = len(self._owners)
        if not owner_count:
            return
        for bank in range(self.banks):
            for offset in range(owner_count):
                index = (self._cursor[bank] + offset) % owner_count
                owner_queue = self._queues[self._owners[index]]
                if not owner_queue or not owner_queue[0].queues[bank]:
                    continue
                request = owner_queue[0]
                word, mask = request.queues[bank][0]
                address = (bank, word)
                assert self.sram.try_access(cycle, [] if request.write else [address],
                                            [address] if request.write else [])
                request.queues[bank].popleft()
                request.remaining -= 1
                request.outstanding += 1
                self._cursor[bank] = (index + 1) % owner_count
                self._sequence += 1
                heapq.heappush(self._inflight, (cycle + 1, self._sequence,
                                               request.request_id, bank, word, mask))
                direction = "write" if request.write else "read"
                useful_key, physical_key = direction + "_useful_bytes", "physical_" + direction + "_words"
                self._stats[useful_key] += _mask_bytes(mask)
                self._stats[physical_key] += 1
                self._per_owner[request.owner][useful_key] += _mask_bytes(mask)
                self._per_owner[request.owner][physical_key] += 1
                self._per_bank[bank][direction + "_words"] += 1
                self._per_bank[bank][useful_key] += _mask_bytes(mask)
                self._event("LOCAL_WRITE" if request.write else "LOCAL_READ", cycle,
                            request, bank=bank, word_index=word, byte_mask=mask,
                            useful_bytes=_mask_bytes(mask), physical_bytes=2,
                            complete_cycle=cycle + 1)
                break
            if any(queue and queue[0].queues[bank] for queue in self._queues.values()):
                self._per_bank[bank]["denied_cycles"] += 1
        denied = False
        for owner, queue in self._queues.items():
            if queue and queue[0].remaining:
                self._per_owner[owner]["denied_cycles"] += 1
                denied = True
        if denied:
            self._stats["denied_cycles"] += 1
        self._stats["peak_outstanding_words"] = max(self._stats["peak_outstanding_words"], len(self._inflight))

    @property
    def pending(self):
        return bool(self._requests) or bool(self._inflight)

    def stats(self):
        result = dict(self._stats)
        result.update(capacity_bytes=self.capacity_bytes, banks=self.banks,
                      depth_words=self.depth_words, word_bytes=self.word_bytes,
                      latency=self.latency, pending_requests=len(self._requests),
                      outstanding_words=len(self._inflight),
                      physical_read_bytes=self._stats["physical_read_words"] * 2,
                      physical_write_bytes=self._stats["physical_write_words"] * 2,
                      per_owner={owner: dict(stats) for owner, stats in self._per_owner.items()},
                      per_bank=[dict(stats) for stats in self._per_bank])
        return result


class LocalLayout:
    """Byte-accurate allocation in the selected fixed local SRAM.

    Input A for the array is k-major with an even token stride. Row/B1 input is
    row-major. Other tensors are row-major. B1 partial storage contains ``group``
    token slots, two INT32 channels each; its row index is local to the group.
    """

    capacity_bytes = LOCAL_BYTES
    banks = 8
    depth_words = 4096
    word_bytes = 2

    def __init__(self, shape, mapping, group, banks=8):
        if not isinstance(shape, Shape):
            raise ValueError("shape must be a Shape")
        self.banks = _bank_count(banks)
        self.capacity_bytes = self.banks * self.depth_words * self.word_bytes
        _int("group", group, 1)
        aliases = {"array_9x16": "A", "array": "A", "A": "A", "B0": "B0",
                   "row_dot128": "B0", "row": "B0", "B": "B0", "B1": "B1_G9",
                   "B1_G9": "B1_G9", "B1_G99": "B1_G99"}
        aliases.update((name, name) for name in SUPPORTED_MAPPINGS)
        if mapping not in aliases:
            raise ValueError("unknown fixed local mapping: " + str(mapping))
        self.shape, self.mapping, self.group = shape, aliases[mapping], min(group, shape.m)
        self.tensors, cursor = {}, 0
        traits = mapping_traits(shape, self.mapping)
        array = traits["is_array"]
        input_padded = shape.r * ((shape.m + 1) // 2 * 2) if array else (shape.m * shape.r + 1) // 2 * 2
        specs = [("input", shape.m, shape.r, 1, input_padded),
                 ("residual", shape.m, shape.n, 1, (shape.m * shape.n + 1) // 2 * 2),
                 ("pre_ln", shape.m, shape.n, 2, shape.m * shape.n * 2),
                 ("output", shape.m, shape.n, 1, (shape.m * shape.n + 1) // 2 * 2)]
        if traits["is_b1"]:
            specs.append(("partial", self.group, 2, 4, self.group * 8))
        for name, rows, cols, width, padded in specs:
            descriptor = {"base_byte_addr": cursor, "bytes": rows * cols * width,
                          "logical_bytes": rows * cols * width, "padded_bytes": padded,
                          "element_bytes": width, "rows": rows, "cols": cols,
                          "bits": width * 8,
                          "layout": "k_major" if name == "input" and array else "row_major"}
            if name == "input" and array:
                descriptor.update(row_stride_bytes=1, col_stride_bytes=(shape.m + 1) // 2 * 2)
            else:
                descriptor.update(row_stride_bytes=cols * width, col_stride_bytes=width)
            self.tensors[name] = descriptor
            cursor += padded
        if cursor > self.capacity_bytes:
            raise LocalCapacityError(cursor, self.capacity_bytes, self.tensors, self.mapping)
        self.descriptors = self.tensors
        self.used_bytes, self.padded_bytes = cursor, cursor
        self.free_bytes = self.capacity_bytes - cursor

    def _tensor(self, tensor):
        aliases = {"a": "input", "input_a": "input", "ln_input": "pre_ln",
                   "pre_ln16": "pre_ln", "output8": "output", "partial_sum": "partial"}
        name = aliases.get(tensor, tensor)
        if name not in self.tensors:
            raise ValueError("unknown local tensor: " + str(tensor))
        return self.tensors[name]

    def address(self, tensor, row, col):
        descriptor = self._tensor(tensor)
        _int("row", row)
        _int("col", col)
        if row >= descriptor["rows"] or col >= descriptor["cols"]:
            raise ValueError("local tensor element is out of range")
        return (descriptor["base_byte_addr"] + row * descriptor["row_stride_bytes"] +
                col * descriptor["col_stride_bytes"])

    def accesses(self, tensor, row, cols):
        descriptor, masks = self._tensor(tensor), {}
        for col in cols:
            address = self.address(tensor, row, col)
            for byte in range(address, address + descriptor["element_bytes"]):
                linear_word = byte // 2
                key = (linear_word % self.banks, linear_word // self.banks)
                masks[key] = masks.get(key, 0) | (1 << (byte % 2))
        return [(bank, word, mask) for (bank, word), mask in masks.items()]

    def stats(self):
        return {"capacity_bytes": self.capacity_bytes, "used_bytes": self.used_bytes,
                "padded_bytes": self.used_bytes, "free_bytes": self.free_bytes,
                "mapping": self.mapping, "group": self.group,
                "tensors": {name: dict(descriptor) for name, descriptor in self.tensors.items()}}
