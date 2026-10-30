"""Cycle boundaries, bounded reservations, and explicit byte addressing."""
import heapq
from dataclasses import dataclass, field


@dataclass(order=True)
class Event:
    cycle: int
    phase: int
    sequence_id: int
    kind: str = field(compare=False)
    payload: object = field(compare=False)
    callback: object = field(compare=False, repr=False)


class EventQueue:
    def __init__(self):
        self._heap = []
        self._sequence = 0

    def schedule(self, now, delay, kind, payload, callback, phase=0):
        if delay < 1:
            raise ValueError("new actions must have positive latency")
        event = Event(now + delay, phase, self._sequence, kind, payload, callback)
        self._sequence += 1
        heapq.heappush(self._heap, event)
        return event

    def complete(self, cycle):
        while self._heap and self._heap[0].cycle <= cycle:
            event = heapq.heappop(self._heap)
            if event.cycle != cycle:
                raise AssertionError("an event was skipped")
            event.callback(event)

    @property
    def pending(self):
        return bool(self._heap)


class Capacity:
    """Reserved includes in-flight data; live is the completed subset."""
    def __init__(self, name, capacity_bytes):
        if capacity_bytes < 1:
            raise ValueError("buffer capacity must be positive")
        self.name = name
        self.capacity_bytes = capacity_bytes
        self.allocations = {}
        self.peak_reserved = 0
        self.peak_live = 0

    @property
    def reserved(self):
        return sum(v[0] for v in self.allocations.values())

    @property
    def live(self):
        return sum(v[1] for v in self.allocations.values())

    def can_reserve(self, size):
        return self.reserved + size <= self.capacity_bytes

    def reserve(self, owner, size):
        if owner in self.allocations or size < 1 or not self.can_reserve(size):
            raise AssertionError("invalid reservation: " + self.name)
        self.allocations[owner] = (size, 0)
        self._check()

    def ready(self, owner):
        size, _ = self.allocations[owner]
        self.allocations[owner] = (size, size)
        self._check()

    def set_live(self, owner, live_bytes):
        size, _ = self.allocations[owner]
        if not 0 <= live_bytes <= size:
            raise AssertionError("live payload exceeds reservation")
        self.allocations[owner] = (size, live_bytes)
        self._check()

    def release(self, owner):
        del self.allocations[owner]
        self._check()

    def _check(self):
        assert 0 <= self.live <= self.reserved <= self.capacity_bytes
        self.peak_reserved = max(self.peak_reserved, self.reserved)
        self.peak_live = max(self.peak_live, self.live)

    def stats(self):
        return {"capacity_bytes": self.capacity_bytes,
                "peak_reserved_bytes": self.peak_reserved,
                "peak_live_bytes": self.peak_live,
                "final_reserved_bytes": self.reserved}


def align(value, quantum):
    return ((value + quantum - 1) // quantum) * quantum
