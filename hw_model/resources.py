"""Profiled local SRAM and explicit small register inventories."""
from .compute import Shape
from .fixed_schedule import mapping_traits


def _shape_values(shape):
    if isinstance(shape, dict):
        return tuple(shape[key] for key in ('m', 'n', 'r'))
    if all(hasattr(shape, key) for key in ('m', 'n', 'r')):
        return shape.m, shape.n, shape.r
    return tuple(shape)


def resource_inventory(config, mapping, shape, measured=None):
    m, n, r = _shape_values(shape)
    traits = mapping_traits(Shape(m, n, r), mapping)
    partial_bytes = traits['partial_sum_workspace_bytes']
    registers = {
        'weight_register_bytes': 2*traits['weight_packet_slot_bytes'],
        'activation_register_bytes': traits['activation_register_bytes'],
        'accumulator_register_bytes': traits['accumulator_register_bytes'],
        'row_assembly_register_bytes': 0,
        'ln_workspace_bytes': n*2+n,
        'parameter_register_bytes': n*(config['numeric']['bias_bits']+2*config['numeric']['ln_parameter_bits'])//8,
    }
    measured = measured or {}
    non_register = ('b1_partial_sram_bytes', 'local_sram_bytes', 'local_reserved_bytes', 'local_payload_bytes')
    for key, value in measured.items():
        if key.endswith('_bytes') and key not in non_register:
            if type(value) is not int or value < 0:
                raise ValueError('measured %s must be a nonnegative byte count' % key)
            registers[key] = value
    unit_area = config['area']['macro_areas_mm2']['4096_single']
    cell_area = config['area']['standard_cells_mm2']
    local_banks = config['local_memory']['banks']
    local_bytes = local_banks*config['local_memory']['depth_words']*config['local_memory']['word_bytes']
    macro_area = None if unit_area is None else unit_area*local_banks
    total_area = None if macro_area is None or cell_area is None else macro_area+cell_area
    limit = config['area']['limit_mm2']
    return {
        'mapping': mapping, 'architecture': traits['architecture'],
        'multipliers': traits['physical_multipliers'],
        'macro_instances': {'4096_single': local_banks}, 'local_sram_bytes': local_bytes,
        'activation_macro_instances': {'4096_single': local_banks}, 'weight_sram_macro_instances': {},
        'weight_sram_bytes': 0, 'shared_weight_sram_bytes': config['platform']['shared_bytes'],
        'register_payload_bytes': registers, 'total_register_payload_bytes': sum(registers.values()),
        'b1_partial_sram_bytes': measured.get('b1_partial_sram_bytes', partial_bytes),
        'common_tensor_payload_bytes': m*r+4*m*n,
        'register_payload_excludes_control_tags_and_interconnect': True,
        'source_weight_repacking_initialization_separate': True,
        'local_capacity_must_include_layout_padding_and_workspace': True,
        'area': {'status': 'unknown' if total_area is None else 'provided_estimate_not_synthesis',
                 'local_sram_mm2': macro_area, 'standard_cells_mm2': cell_area,
                 'total_mm2': total_area, 'limit_mm2': limit,
                 'within_limit': None if total_area is None or limit is None else total_area <= limit,
                 'existing_shared_sram_excluded_from_incremental_area': True},
    }
