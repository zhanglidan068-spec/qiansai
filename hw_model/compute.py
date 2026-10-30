"""Pure compute mappings and explicitly ideal, unlimited-bandwidth checks.

An issue occupies [t, t + 1) and its arithmetic is visible at t + 1.
These functions omit memory, finite queues, pipeline fill and drain, and residual
work. They are arithmetic/scheduler oracles, not a complete hardware simulator.
"""

from dataclasses import dataclass
import heapq
from typing import Dict, Iterator, List, Tuple


ARCHITECTURES = ("array_9x16", "row_dot128", "row_av64")


def _positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("{} must be a positive integer".format(name))
    return value


@dataclass(frozen=True)
class Shape:
    """C[m, n] = A[m, r] @ B[r, n], with strictly positive dimensions."""

    m: int
    n: int
    r: int

    def __post_init__(self) -> None:
        for name in ("m", "n", "r"):
            _positive_int(getattr(self, name), name)

    @property
    def useful_mac(self) -> int:
        return self.m * self.n * self.r


@dataclass(frozen=True)
class Issue:
    """Valid indices only; their Cartesian product specifies useful products.

    Padded physical lanes are absent, so tail masks never invent useful MACs.
    A row-dot issue reduces up to 64 products for each of two outputs. AV mode
    reduces up to two products for each of 64 outputs.
    """

    rows: Tuple[int, ...]
    columns: Tuple[int, ...]
    reduction: Tuple[int, ...]
    useful_mac: int


def _mapping(architecture: str) -> Tuple[int, int, int, int]:
    """Return row, column, reduction lanes and physical multiplier count."""
    if architecture == "array_9x16":
        return 9, 16, 1, 144
    if architecture == "row_dot128":
        return 1, 2, 64, 128
    if architecture == "row_av64":
        return 1, 64, 2, 128
    raise ValueError("unknown architecture {!r}; choose {}".format(
        architecture, ", ".join(ARCHITECTURES)))


def iter_issues(shape: Shape, architecture: str) -> Iterator[Issue]:
    """Emit token-group, column-tile, reduction-chunk order deterministically."""
    row_lanes, col_lanes, red_lanes, _ = _mapping(architecture)
    for row_start in range(0, shape.m, row_lanes):
        rows = tuple(range(row_start, min(row_start + row_lanes, shape.m)))
        for col_start in range(0, shape.n, col_lanes):
            columns = tuple(range(col_start, min(col_start + col_lanes, shape.n)))
            for red_start in range(0, shape.r, red_lanes):
                reduction = tuple(range(red_start, min(red_start + red_lanes, shape.r)))
                yield Issue(rows, columns, reduction,
                            len(rows) * len(columns) * len(reduction))


def _ceildiv(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def ideal_summary(shape: Shape, architecture: str) -> Dict[str, object]:
    """Closed-form independent oracle for the issue generator's schedule."""
    row_lanes, col_lanes, red_lanes, multipliers = _mapping(architecture)
    cycles_per_group = _ceildiv(shape.n, col_lanes) * _ceildiv(shape.r, red_lanes)
    issue_cycles = _ceildiv(shape.m, row_lanes) * cycles_per_group
    row_ready = [(row // row_lanes + 1) * cycles_per_group for row in range(shape.m)]
    return {
        "architecture": architecture,
        "issue_cycles": issue_cycles,
        "useful_mac": shape.useful_mac,
        "physical_multipliers": multipliers,
        "first_row_cycles": row_ready[0],
        "row_ready_cycles": row_ready,
        "compute_utilization": shape.useful_mac / (multipliers * issue_cycles),
    }


def ideal_fused(shape: Shape, architecture: str, ln_latency: int = 64,
                ln_ii: int = 64, max_inflight: int = 1) -> Dict[str, object]:
    """Schedule ideal matrix rows through an L0 LN with unlimited row storage.

    LN latency, initiation interval and in-flight capacity are independent.
    Completions at t release capacity before an acceptance at the same t.
    This intentionally excludes residual/transfer costs and finite backpressure.
    """
    _positive_int(ln_latency, "ln_latency")
    _positive_int(ln_ii, "ln_ii")
    _positive_int(max_inflight, "max_inflight")
    matrix = ideal_summary(shape, architecture)
    inflight = []  # type: List[int]
    accepts = []  # type: List[int]
    dones = []  # type: List[int]
    next_accept = 0
    for ready in matrix["row_ready_cycles"]:
        start = max(ready, next_accept)
        while inflight and inflight[0] <= start:
            heapq.heappop(inflight)
        if len(inflight) >= max_inflight:
            start = max(start, heapq.heappop(inflight))
            while inflight and inflight[0] <= start:
                heapq.heappop(inflight)
        done = start + ln_latency
        heapq.heappush(inflight, done)
        accepts.append(start)
        dones.append(done)
        next_accept = start + ln_ii
    return {
        "scope": "ideal_infinite_buffer",
        "matrix": matrix,
        "row_accept_cycles": accepts,
        "row_done_cycles": dones,
        "first_row_cycles": dones[0],
        "total_cycles": max(dones),
        "ln_latency": ln_latency,
        "ln_ii": ln_ii,
        "max_inflight": max_inflight,
    }


def benchmarks() -> Dict[str, object]:
    """Reproduce document section 12; AV row mode is explicitly reconfigured."""
    operators = []
    totals = {"array_9x16": 0, "row_dot128": 0, "useful_mac": 0}
    for name, shape in (
            ("q", Shape(99, 64, 64)),
            ("k", Shape(99, 64, 64)),
            ("v", Shape(99, 64, 64)),
            ("qk", Shape(99, 99, 64)),
            ("av", Shape(99, 64, 99)),
            ("projection", Shape(99, 64, 64)),
            ("ffn1", Shape(99, 256, 64)),
            ("ffn2", Shape(99, 64, 256))):
        array = ideal_summary(shape, "array_9x16")
        row = ideal_summary(shape, "row_av64" if name == "av" else "row_dot128")
        operators.append({
            "operator": name,
            "shape": {"m": shape.m, "n": shape.n, "r": shape.r},
            "array_9x16": array,
            "row_dot128": row,
        })
        totals["array_9x16"] += array["issue_cycles"]
        totals["row_dot128"] += row["issue_cycles"]
        totals["useful_mac"] += shape.useful_mac
    return {
        "scope": "encoder_gemm_only",
        "operators": operators,
        "totals": totals,
        "av_row_mapping": "row_av64",
    }
