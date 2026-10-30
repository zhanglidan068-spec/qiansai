"""Weight schedules for the fixed 300 MB/s A/B0/B1 comparison.

Addresses are relative to a separately initialized, architecture-specific
prepacked shared weight tensor. Packing is not a free operation inside the
comparison: its initialization cost belongs in a separate report boundary.
Each packet's useful values are contiguous inside a fixed per-mapping slot.
Only useful bytes are requested; the weight port accounts for every addressed
4-byte beat, including repeated beats across separately requested packets.
"""
from dataclasses import asdict, dataclass
from typing import Dict, Iterator, Tuple

from .compute import Shape


MAPPINGS = ("A", "B0", "B1_G9", "B1_G99")
# Keep the established comparison set stable. Experimental shapes require an
# explicit selection and change only the compute mapping/register geometry.
EXPERIMENTAL_MAPPINGS = ("A18x8", "A16x8", "A32x4", "B1_G11")
REVISION_MAPPINGS = ("A33x4", "A33x8", "A32x8", "A36x4", "A17x4", "A33x2",
                     "A9x8", "A11x8", "A3x16", "A3x32", "A33x16", "A17x8")
EXTENDED_MAPPINGS = ("A33x32", "A11x4", "A99x4", "A99x8", "A66x4",
                     "A99x16", "A66x8", "A66x16")
ALL_MAPPINGS = MAPPINGS + EXPERIMENTAL_MAPPINGS + REVISION_MAPPINGS + EXTENDED_MAPPINGS
# This bounded grid is opt-in. Keep ALL_MAPPINGS stable for prior studies.
_GRID_ARRAY_SHAPES = {"A{}x{}".format(rows, columns): (rows, columns)
                      for rows in range(20, 34) for columns in (4, 6, 8)}
GRID_MAPPINGS = tuple(_GRID_ARRAY_SHAPES)
SUPPORTED_MAPPINGS = ALL_MAPPINGS + tuple(name for name in GRID_MAPPINGS if name not in ALL_MAPPINGS)


@dataclass(frozen=True)
class ComputePacket:
    packet_id: str
    weight_address: int
    weight_bytes: int
    columns: Tuple[int, ...]
    reduction: Tuple[int, ...]
    rows: Tuple[int, ...]
    group_id: int
    tile_id: str
    first_reduction: bool
    last_reduction: bool
    last_tile_in_group: bool

    @property
    def physical_weight_bytes(self) -> int:
        return ((self.weight_address % 4 + self.weight_bytes + 3) // 4) * 4

    @property
    def useful_mac(self) -> int:
        return len(self.rows) * len(self.columns) * len(self.reduction)


def _ceildiv(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def mapping_traits(shape: Shape, mapping: str) -> Dict[str, object]:
    """Physical issue shape and maximum group payload, with explicit units.

    B1 has one active pair of accumulator registers plus an SRAM partial-sum
    workspace for that pair across all group rows. Row assembly is an INT16
    local-SRAM payload budget, not an additional register tensor; the complete
    pre-LN tensor allocation is common to all mappings in the detailed model.
    """
    if mapping not in SUPPORTED_MAPPINGS:
        raise ValueError("mapping must be one of {}".format(", ".join(SUPPORTED_MAPPINGS)))
    array_shapes = {"A": (9, 16), "A18x8": (18, 8),
                    "A16x8": (16, 8), "A32x4": (32, 4),
                    "A33x4": (33, 4), "A33x8": (33, 8), "A32x8": (32, 8),
                    "A36x4": (36, 4), "A17x4": (17, 4), "A33x2": (33, 2),
                    "A9x8": (9, 8), "A11x8": (11, 8), "A3x16": (3, 16),
                    "A3x32": (3, 32), "A33x16": (33, 16), "A17x8": (17, 8),
                    "A33x32": (33, 32), "A11x4": (11, 4), "A99x4": (99, 4),
                    "A99x8": (99, 8), "A66x4": (66, 4), "A99x16": (99, 16),
                    "A66x8": (66, 8), "A66x16": (66, 16)}
    if mapping in _GRID_ARRAY_SHAPES:
        array_shapes[mapping] = _GRID_ARRAY_SHAPES[mapping]
    array = mapping in array_shapes
    if array:
        group_rows, columns = array_shapes[mapping]
        reduction, slot = 1, columns
        architecture = "array_{}x{}".format(group_rows, columns)
    else:
        group_rows = {"B0": 1, "B1_G9": 9, "B1_G99": 99, "B1_G11": 11}[mapping]
        columns, reduction, slot, architecture = 2, 64, 128, "row_dot128"
    b1 = not array and mapping != "B0"
    accumulator_bytes = group_rows * columns * 4 if array else 8
    return {
        "mapping": mapping,
        "architecture": architecture,
        "is_array": array,
        "is_b1": b1,
        "is_experimental": mapping not in MAPPINGS,
        "array_rows": group_rows if array else 1,
        "array_columns": columns,
        "activation_operand_slots": 2,
        "group_rows": group_rows,
        "column_lanes": columns,
        "reduction_lanes": reduction,
        "physical_multipliers": group_rows * columns if array else 128,
        "weight_packet_slot_bytes": slot,
        "weight_hold_bytes": slot,
        "weight_storage_bytes": _ceildiv(shape.n, columns) * _ceildiv(shape.r, reduction) * slot,
        "activation_register_bytes": 2 * group_rows if array else (shape.r if mapping == "B0" else 128),
        "accumulator_register_bytes": accumulator_bytes,
        "operand_partial_register_bytes": 16 if b1 else 0,
        "partial_sum_workspace_bytes": min(shape.m, group_rows) * 8 if b1 else 0,
        "row_assembly_bytes": min(shape.m, group_rows) * shape.n * 2,
        "sequential_row_consumers": not array,
        "weight_layout": "column_tile_reduction_packet_contiguous_valid_values",
    }


def weight_byte_offset(shape: Shape, mapping: str, reduction_index: int,
                       column: int) -> int:
    """Byte address of one INT8 B[k,n] in the prepacked shared layout."""
    traits = mapping_traits(shape, mapping)
    for value, limit, name in ((reduction_index, shape.r, "reduction_index"),
                               (column, shape.n, "column")):
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < limit:
            raise ValueError("{} outside tensor".format(name))
    col_lanes, red_lanes = traits["column_lanes"], traits["reduction_lanes"]
    col_tile, local_column = divmod(column, col_lanes)
    red_tile, local_reduction = divmod(reduction_index, red_lanes)
    width = min(col_lanes, shape.n - col_tile * col_lanes)
    packet = col_tile * _ceildiv(shape.r, red_lanes) + red_tile
    return packet * traits["weight_packet_slot_bytes"] + local_reduction * width + local_column


def schedule_packets(shape: Shape, mapping: str) -> Iterator[ComputePacket]:
    """Emit group -> output tile -> reduction packet order deterministically.

    An A packet has one parallel MAC consumer across its rows. B0/B1 packets
    have one sequential MAC consumer per row, retaining the same weight bytes
    until every row consumes them. No tail padding creates useful products.
    last_tile_in_group is a final-packet marker (last column AND reduction).
    """
    traits = mapping_traits(shape, mapping)
    group_rows, col_lanes, red_lanes = (traits[key] for key in
                                        ("group_rows", "column_lanes", "reduction_lanes"))
    red_tiles = _ceildiv(shape.r, red_lanes)
    for group_id, row_start in enumerate(range(0, shape.m, group_rows)):
        rows = tuple(range(row_start, min(row_start + group_rows, shape.m)))
        for col_tile, col_start in enumerate(range(0, shape.n, col_lanes)):
            columns = tuple(range(col_start, min(col_start + col_lanes, shape.n)))
            tile_id = "g{:04d}_c{:04d}".format(group_id, col_start)
            for red_tile, red_start in enumerate(range(0, shape.r, red_lanes)):
                reduction = tuple(range(red_start, min(red_start + red_lanes, shape.r)))
                last_reduction = red_start + len(reduction) == shape.r
                yield ComputePacket(
                    packet_id="{}_{}_k{:04d}".format(mapping, tile_id, red_start),
                    weight_address=(col_tile * red_tiles + red_tile) * traits["weight_packet_slot_bytes"],
                    weight_bytes=len(columns) * len(reduction),
                    columns=columns, reduction=reduction, rows=rows,
                    group_id=group_id, tile_id=tile_id,
                    first_reduction=red_start == 0,
                    last_reduction=last_reduction,
                    last_tile_in_group=last_reduction and col_start + len(columns) == shape.n)


def _weight_only_l0(shape: Shape, mapping: str) -> Dict[str, int]:
    """Transfer-only L0: 2 B/cycle, LN latency/II128, one LN row at a time."""
    cycle = useful = physical = count = 0
    first_batch = first_output = None
    ln_done = 0
    for packet in schedule_packets(shape, mapping):
        count += 1
        useful += packet.weight_bytes
        physical += packet.physical_weight_bytes
        cycle += packet.physical_weight_bytes // 2
        if packet.last_tile_in_group:
            if first_batch is None:
                first_batch = cycle
            for _ in packet.rows:
                ln_done = max(cycle, ln_done) + 128
                if first_output is None:
                    first_output = ln_done
    return {
        "packet_count": count,
        "weight_bytes": useful,
        "physical_weight_bytes": physical,
        "weight_cycles": cycle,
        "first_batch_weight_cycles": first_batch,
        "first_row_cycles": first_output,
        "total_cycles": ln_done,
    }


def fixed_l0_benchmarks() -> Dict[str, object]:
    """Independent, explicit weight-transfer + LN128 comparison baseline.

    This is not the previous preloaded-weight LN64 arithmetic baseline. It
    excludes compute, activation/residual traffic, conversion and writeback.
    B1 is reported only in the weight ledger because sequential group consumers
    can exceed packet transfer time and must be timed by the detailed model.
    """
    operators, ledger = [], []
    for name, shape in (("projection", Shape(99, 64, 64)),
                        ("ffn2", Shape(99, 64, 256))):
        operators.append({"operator": name, "shape": asdict(shape),
                          "A": _weight_only_l0(shape, "A"),
                          "B0": _weight_only_l0(shape, "B0")})
        for mapping in MAPPINGS:
            packets = list(schedule_packets(shape, mapping))
            traits = mapping_traits(shape, mapping)
            ledger.append({
                "operator": name, "mapping": mapping,
                "group_rows": traits["group_rows"], "packet_count": len(packets),
                "useful_mac": sum(packet.useful_mac for packet in packets),
                "weight_bytes": sum(packet.weight_bytes for packet in packets),
                "physical_weight_bytes": sum(packet.physical_weight_bytes for packet in packets),
                "weight_storage_bytes": traits["weight_storage_bytes"],
            })
    return {
        "scope": "weight_transfer_only_plus_ln128",
        "clock_hz": 150000000,
        "bandwidth_bytes_per_second": 300000000,
        "weight_bytes_per_cycle": 2,
        "ln_latency_cycles": 128,
        "ln_ii_cycles": 128,
        "operators": operators,
        "weight_ledger": ledger,
    }


def arithmetic_checks() -> Dict[str, object]:
    result = fixed_l0_benchmarks()
    expected = {"projection": {"A": 23680, "B0": 202880},
                "ffn2": {"A": 91264, "B0": 811136}}
    for operator in result["operators"]:
        for mapping in ("A", "B0"):
            if operator[mapping]["total_cycles"] != expected[operator["operator"]][mapping]:
                raise AssertionError("fixed weight/LN128 arithmetic baseline mismatch")
    return result
