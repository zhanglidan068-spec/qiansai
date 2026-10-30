"""Deterministic, cycle-driven memory resources (Python 3.8, stdlib only).

``SharedMemory`` models timing and traffic, not stored values.  Requests carry
useful byte ranges; the interface transfers whole aligned beats, including the
padding at either end.  Requests from one requester complete in FIFO order.
Call ``tick`` once per simulated cycle, after submitting that cycle's requests.

``BankedSRAM`` is an atomic port-allocation model.  Multiple calls in a cycle
share the same bank ports.  Reads of one address broadcast without another port.
The model deliberately rejects simultaneous read/write of the same address.
"""

from collections import deque
from dataclasses import dataclass
import heapq
from typing import Any, Deque, Dict, List, Optional, Set, Tuple


def _integer(name: str, value: int, minimum: int = 0) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError("{} must be an integer >= {}".format(name, minimum))


@dataclass(frozen=True)
class MemRequest:
    request_id: str
    requester: str
    write: bool
    address: int
    useful_bytes: int
    ready_cycle: int = 0
    tag: Any = None


@dataclass
class _RequestState:
    request: MemRequest
    beats: List[Tuple[int, int, int]]
    next_beat: int = 0
    completed_beats: int = 0
    first_issue: Optional[int] = None


class SharedMemory:
    """Finite-outstanding memory with deterministic requester round robin.

    ``accept_interval`` applies to each physical channel, not to a requester.
    A split interface can accept one read and one write in the same cycle, but
    both use the global ``max_outstanding`` limit.  An accepted beat occupies an
    outstanding entry until its acknowledgement, including writes.

    A positive ``background_period`` reserves every Nth zero-based cycle
    (N-1, 2*N-1, ...).  A reservation consumes each otherwise eligible channel's
    acceptance slot and advances its cooldown.  It represents competing traffic
    outside the modeled requesters, so it is separately counted and does not
    create modeled requester beats or outstanding entries.
    """

    def __init__(self, capacity_bytes: int, beat_bytes: int, read_latency: int,
                 write_latency: int, accept_interval: int,
                 max_outstanding: int, read_write_shared: bool = True,
                 background_period: int = 0, capture_trace: bool = True):
        for name, value in (("capacity_bytes", capacity_bytes),
                            ("beat_bytes", beat_bytes),
                            ("read_latency", read_latency),
                            ("write_latency", write_latency),
                            ("accept_interval", accept_interval),
                            ("max_outstanding", max_outstanding)):
            _integer(name, value, 1)
        _integer("background_period", background_period)
        self.capacity_bytes = capacity_bytes
        self.beat_bytes = beat_bytes
        self.read_latency = read_latency
        self.write_latency = write_latency
        self.accept_interval = accept_interval
        self.max_outstanding = max_outstanding
        self.read_write_shared = bool(read_write_shared)
        self.background_period = background_period
        self.capture_trace = capture_trace
        self.trace = []  # type: List[Dict[str, Any]]
        self._queues = {}  # type: Dict[str, Deque[_RequestState]]
        self._requesters = []  # type: List[str]
        self._ids = set()  # type: Set[str]
        self._inflight = []  # type: List[Tuple[int, int, _RequestState, int, int, int]]
        self._sequence = 0
        self._last_cycle = -1
        channels = ("shared",) if self.read_write_shared else ("read", "write")
        self._next_accept = {channel: 0 for channel in channels}
        self._cursor = {channel: 0 for channel in channels}
        self._next_split_lane = "read"
        self._counters = {
            "submitted_requests": 0, "completed_requests": 0,
            "accepted_beats": 0, "completed_beats": 0,
            "useful_read_bytes": 0, "useful_write_bytes": 0,
            "physical_read_bytes": 0, "physical_write_bytes": 0,
            "max_outstanding": 0, "background_slots": 0,
            "background_bytes": 0, "outstanding_stall_cycles": 0,
        }
        self._requester_wait = {}  # type: Dict[str, int]
        self._requester_first_wait = {}  # type: Dict[str, int]

    def _event(self, event: str, cycle: int,
               request: Optional[MemRequest] = None, **fields: Any) -> None:
        if not self.capture_trace:
            return
        item = {"event": event, "cycle": cycle}
        if request is not None:
            item.update(request_id=request.request_id,
                        requester=request.requester, write=request.write,
                        address=request.address, tag=request.tag)
        item.update(fields)
        self.trace.append(item)

    def submit(self, request: MemRequest) -> None:
        """Queue one request; a duplicate ID or out-of-range request is an error."""
        if not isinstance(request, MemRequest):
            raise TypeError("request must be a MemRequest")
        if not isinstance(request.request_id, str) or not request.request_id:
            raise ValueError("request_id must be a nonempty string")
        if not isinstance(request.requester, str) or not request.requester:
            raise ValueError("requester must be a nonempty string")
        if not isinstance(request.write, bool):
            raise ValueError("write must be a bool")
        _integer("address", request.address)
        _integer("useful_bytes", request.useful_bytes, 1)
        _integer("ready_cycle", request.ready_cycle)
        if request.address + request.useful_bytes > self.capacity_bytes:
            raise ValueError("request useful byte range exceeds memory capacity")
        if request.request_id in self._ids:
            raise ValueError("duplicate request_id: {}".format(request.request_id))
        start = request.address - request.address % self.beat_bytes
        end = request.address + request.useful_bytes
        beats = []
        for address in range(start, end, self.beat_bytes):
            useful_start = max(address, request.address)
            valid_bytes = min(address + self.beat_bytes, end) - useful_start
            beats.append((address, valid_bytes, useful_start - address))
        if request.requester not in self._queues:
            self._queues[request.requester] = deque()
            self._requesters.append(request.requester)
            self._requester_wait[request.requester] = 0
            self._requester_first_wait[request.requester] = 0
        self._queues[request.requester].append(_RequestState(request, beats))
        self._ids.add(request.request_id)
        self._counters["submitted_requests"] += 1
        # With no explicit submit-cycle argument, use the most recently visited
        # cycle (or cycle zero before the first tick); ready_cycle stays explicit.
        self._event("request", max(0, self._last_cycle), request,
                    useful_bytes=request.useful_bytes,
                    ready_cycle=request.ready_cycle, beats=len(beats))

    @property
    def pending(self) -> bool:
        return bool(self._inflight) or any(self._queues.values())

    def _eligible(self, state: _RequestState, cycle: int, channel: str) -> bool:
        request = state.request
        return (request.ready_cycle <= cycle and
                state.next_beat < len(state.beats) and
                (channel == "shared" or request.write == (channel == "write")))

    def _accept(self, cycle: int, channel: str) -> bool:
        if cycle < self._next_accept[channel] or not self._requesters:
            return False
        count = len(self._requesters)
        for offset in range(count):
            index = (self._cursor[channel] + offset) % count
            requester = self._requesters[index]
            queue = self._queues[requester]
            if not queue or not self._eligible(queue[0], cycle, channel):
                continue
            state = queue[0]
            request = state.request
            address, valid_bytes, valid_offset = state.beats[state.next_beat]
            state.next_beat += 1
            if state.first_issue is None:
                state.first_issue = cycle
                self._requester_first_wait[requester] += cycle - request.ready_cycle
            latency = self.write_latency if request.write else self.read_latency
            due = cycle + latency
            self._sequence += 1
            heapq.heappush(self._inflight,
                           (due, self._sequence, state, address,
                            valid_bytes, valid_offset))
            direction = "write" if request.write else "read"
            self._counters["useful_{}_bytes".format(direction)] += valid_bytes
            self._counters["physical_{}_bytes".format(direction)] += self.beat_bytes
            self._counters["accepted_beats"] += 1
            self._counters["max_outstanding"] = max(
                self._counters["max_outstanding"], len(self._inflight))
            self._next_accept[channel] = cycle + self.accept_interval
            self._cursor[channel] = (index + 1) % count
            self._event("beat_issue", cycle, request, address=address,
                        valid_bytes=valid_bytes, valid_offset=valid_offset,
                        physical_bytes=self.beat_bytes, complete_cycle=due,
                        channel=channel, beat=state.next_beat - 1)
            return True
        return False

    def tick(self, cycle: int) -> List[MemRequest]:
        """Apply acknowledgements, then arbitrate this cycle's acceptance slots.

        Cycles must be strictly increasing.  If cycles are skipped, all due
        acknowledgements are applied, but acceptance slots are not replayed.
        """
        _integer("cycle", cycle)
        if cycle <= self._last_cycle:
            raise ValueError("tick cycles must be strictly increasing")
        self._last_cycle = cycle
        completed = []
        while self._inflight and self._inflight[0][0] <= cycle:
            due, _, state, address, valid_bytes, valid_offset = heapq.heappop(self._inflight)
            request = state.request
            state.completed_beats += 1
            self._counters["completed_beats"] += 1
            self._event("beat_done", due, request, address=address,
                        valid_bytes=valid_bytes, valid_offset=valid_offset,
                        physical_bytes=self.beat_bytes)
            if state.completed_beats == len(state.beats):
                queue = self._queues[request.requester]
                assert queue[0] is state
                queue.popleft()
                self._counters["completed_requests"] += 1
                completed.append(request)
                self._event("request_done", due, request,
                            useful_bytes=request.useful_bytes)

        waiting = [name for name, queue in self._queues.items()
                   if queue and self._eligible(queue[0], cycle, "shared")]
        accepted_requesters = set()
        channels = (["shared"] if self.read_write_shared else
                    [self._next_split_lane,
                     "write" if self._next_split_lane == "read" else "read"])
        background = (self.background_period > 0 and
                      (cycle + 1) % self.background_period == 0)
        capacity_blocked = False
        for channel in channels:
            if cycle < self._next_accept[channel]:
                continue
            if len(self._inflight) >= self.max_outstanding:
                capacity_blocked = True
                continue
            if background:
                self._next_accept[channel] = cycle + self.accept_interval
                self._counters["background_slots"] += 1
                self._counters["background_bytes"] += self.beat_bytes
                self._event("background", cycle, channel=channel,
                            physical_bytes=self.beat_bytes)
                continue
            before = self._sequence
            if self._accept(cycle, channel):
                # The sequence identifies the newly accepted heap entry even
                # when a later-due beat sorts after earlier entries.
                accepted = next(item for item in self._inflight if item[1] == before + 1)
                accepted_requesters.add(accepted[2].request.requester)
                if not self.read_write_shared:
                    self._next_split_lane = "write" if channel == "read" else "read"
        if capacity_blocked and waiting:
            self._counters["outstanding_stall_cycles"] += 1
        for requester in waiting:
            if requester not in accepted_requesters:
                self._requester_wait[requester] += 1
        return completed

    def stats(self) -> Dict[str, Any]:
        result = dict(self._counters)
        result.update(outstanding_beats=len(self._inflight),
                      pending_requests=sum(len(queue) for queue in self._queues.values()),
                      requester_wait=dict(self._requester_wait),
                      requester_first_issue_wait=dict(self._requester_first_wait))
        return result


class BankedSRAM:
    """Cycle-local SRAM bank ports, with fixed 2-byte words.

    ``single`` permits one read or write per bank/cycle. ``1r1w`` permits one
    distinct read and one write. ``2rw`` permits any two accesses. Repeated reads
    of an identical address broadcast, even across calls in the same cycle.
    Duplicate writes and read/write of one address are rejected in every mode.
    A successful access completes at ``cycle + latency``. No values are stored.
    """

    word_bytes = 2

    def __init__(self, name: str, banks: int, depth_words: int,
                 port_mode: str = "single", latency: int = 1,
                 capture_trace: bool = True):
        if not isinstance(name, str) or not name:
            raise ValueError("name must be a nonempty string")
        _integer("banks", banks, 1)
        _integer("depth_words", depth_words, 1)
        _integer("latency", latency, 1)
        if port_mode not in ("single", "1r1w", "2rw"):
            raise ValueError("port_mode must be single, 1r1w, or 2rw")
        self.name = name
        self.banks = banks
        self.depth_words = depth_words
        self.port_mode = port_mode
        self.latency = latency
        self.capacity_bytes = banks * depth_words * self.word_bytes
        self.capture_trace = capture_trace
        self.trace = []  # type: List[Dict[str, Any]]
        self._cycle = -1
        self._reads = [set() for _ in range(banks)]  # type: List[Set[int]]
        self._writes = [set() for _ in range(banks)]  # type: List[Set[int]]
        self._counters = {"attempts": 0, "grants": 0, "conflicts": 0,
                          "read_requests": 0, "write_requests": 0,
                          "read_accesses": 0, "write_accesses": 0,
                          "read_broadcasts": 0}
        self._per_bank = [{"reads": 0, "writes": 0, "conflicts": 0}
                          for _ in range(banks)]

    def completion_cycle(self, cycle: int) -> int:
        _integer("cycle", cycle)
        return cycle + self.latency

    def _validate(self, accesses: List[Tuple[int, int]]) -> None:
        for access in accesses:
            if not isinstance(access, (tuple, list)) or len(access) != 2:
                raise ValueError("an access must be (bank, word)")
            bank, word = access
            _integer("bank", bank)
            _integer("word", word)
            if bank >= self.banks or word >= self.depth_words:
                raise ValueError("SRAM bank or word address is out of range")

    def try_access(self, cycle: int, reads: List[Tuple[int, int]],
                   writes: List[Tuple[int, int]]) -> bool:
        """Atomically grant all accesses, or count a conflict and grant none."""
        _integer("cycle", cycle)
        if cycle < self._cycle:
            raise ValueError("SRAM cycles must not go backwards")
        self._validate(reads)
        self._validate(writes)
        # Arbitration commonly probes one scalar word in a many-bank cache.
        # Validate/copy only touched banks, then commit them together. Expiring
        # all old ports happens once per successful new cycle, never per probe.
        touched = {bank for bank, _ in reads}
        touched.update(bank for bank, _ in writes)
        new_cycle = cycle != self._cycle
        old_reads = {bank: set() if new_cycle else self._reads[bank]
                     for bank in touched}
        old_writes = {bank: set() if new_cycle else self._writes[bank]
                      for bank in touched}
        new_reads = {bank: words.copy() for bank, words in old_reads.items()}
        new_writes = {bank: words.copy() for bank, words in old_writes.items()}
        conflicts = set()
        for bank, word in reads:
            new_reads[bank].add(word)
        for bank, word in writes:
            if word in new_writes[bank]:
                conflicts.add(bank)
            new_writes[bank].add(word)
        for bank in touched:
            nr, nw = len(new_reads[bank]), len(new_writes[bank])
            if new_reads[bank] & new_writes[bank]:
                conflicts.add(bank)
            if ((self.port_mode == "single" and nr + nw > 1) or
                    (self.port_mode == "1r1w" and (nr > 1 or nw > 1)) or
                    (self.port_mode == "2rw" and nr + nw > 2)):
                conflicts.add(bank)
        self._counters["attempts"] += 1
        if conflicts:
            self._counters["conflicts"] += 1
            for bank in conflicts:
                self._per_bank[bank]["conflicts"] += 1
            if self.capture_trace:
                self.trace.append({"event": "bank_conflict", "cycle": cycle,
                                   "name": self.name, "banks": sorted(conflicts)})
            return False
        physical_reads = sum(len(new_reads[bank]) - len(old_reads[bank])
                             for bank in touched)
        self._counters["grants"] += 1
        self._counters["read_requests"] += len(reads)
        self._counters["write_requests"] += len(writes)
        self._counters["read_accesses"] += physical_reads
        self._counters["write_accesses"] += len(writes)
        self._counters["read_broadcasts"] += len(reads) - physical_reads
        if new_cycle:
            self._reads = [set() for _ in range(self.banks)]
            self._writes = [set() for _ in range(self.banks)]
        for bank in touched:
            self._per_bank[bank]["reads"] += len(new_reads[bank]) - len(old_reads[bank])
            self._per_bank[bank]["writes"] += len(new_writes[bank]) - len(old_writes[bank])
            self._reads[bank], self._writes[bank] = new_reads[bank], new_writes[bank]
        self._cycle = cycle
        if self.capture_trace:
            self.trace.append({"event": "bank_grant", "cycle": cycle,
                               "complete_cycle": cycle + self.latency,
                               "name": self.name, "reads": [list(x) for x in reads],
                               "writes": [list(x) for x in writes]})
        return True

    def stats(self) -> Dict[str, Any]:
        result = dict(self._counters)
        result.update(name=self.name, capacity_bytes=self.capacity_bytes,
                      word_bytes=self.word_bytes,
                      per_bank=[dict(item) for item in self._per_bank])
        return result
