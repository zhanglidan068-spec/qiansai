"""Named schema-2 contracts with the same fixed 300 MB/s weight interface."""
import copy
import hashlib
import json
import math
from pathlib import Path
import platform

CONFIG_DIR = Path(__file__).absolute().parent / 'configs'
DEFAULT_CONFIG = CONFIG_DIR / 'baseline.json'
REVISED_CONFIG = CONFIG_DIR / 'ln64_local32k.json'
REVISED_PROFILE_ID = 'fixed300_local32k_ln64'
LN64_LOCAL64K_CONFIG = CONFIG_DIR / 'ln64_local64k.json'
LN64_LOCAL64K_PROFILE_ID = 'fixed300_local64k_ln64'


def _read(path):
    def invalid_constant(value):
        raise ValueError('non-finite JSON constant: %s' % value)
    with Path(path).open(encoding='utf-8-sig') as stream:
        result = json.load(stream, parse_constant=invalid_constant)
    if not isinstance(result, dict):
        raise ValueError('configuration must be a JSON object')
    return result


def _merge(target, patch, where='config'):
    if not isinstance(patch, dict):
        raise ValueError('%s must be an object' % where)
    for key, value in patch.items():
        if key not in target:
            raise ValueError('unknown configuration field: %s.%s (legacy configurations and sweeps are not supported)' % (where, key))
        if isinstance(target[key], dict):
            _merge(target[key], value, where+'.'+key)
        else:
            target[key] = copy.deepcopy(value)


def load_config(path=None, overrides=None):
    result = _read(DEFAULT_CONFIG)
    if path is not None and Path(path).absolute() != DEFAULT_CONFIG.absolute():
        _merge(result, _read(path))
    if overrides is not None:
        _merge(result, overrides)
    validate_config(result)
    return result


def validate_config(config):
    template = _read(DEFAULT_CONFIG)
    def keys(value, expected, where):
        if not isinstance(value, dict):
            raise ValueError('%s must be an object' % where)
        missing, extra = sorted(set(expected)-set(value)), sorted(set(value)-set(expected))
        if missing or extra:
            raise ValueError('%s fields missing=%r unknown=%r' % (where, missing, extra))
        for key in expected:
            if isinstance(expected[key], dict):
                keys(value[key], expected[key], where+'.'+key)
    keys(config, template, 'config')
    # The user explicitly requested this second, fixed platform profile. Keep
    # the original baseline and its rejection of arbitrary LN/bank sweeps.
    if config['provenance']['scenario_id'] == REVISED_PROFILE_ID:
        template['local_memory']['banks'] = 4
        template['ln']['latency_cycles'] = 64
        template['ln']['initiation_interval_cycles'] = 64
    elif config['provenance']['scenario_id'] == LN64_LOCAL64K_PROFILE_ID:
        # This independent profile changes only LN. It does not inherit the
        # separate 32 KiB experiment's bank count or its capacity limitations.
        template['ln']['latency_cycles'] = 64
        template['ln']['initiation_interval_cycles'] = 64
    def positive(section, key):
        value = config[section][key]
        if type(value) is not int or value < 1:
            raise ValueError('%s.%s must be a positive integer' % (section, key))
    def frozen(section, exceptions=()):
        for key, expected in template[section].items():
            if key not in exceptions and (type(config[section][key]) is not type(expected) or config[section][key] != expected):
                raise ValueError('%s.%s is fixed at %r by the 300 MB/s contract' % (section, key, expected))
    if type(config['schema_version']) is not int or config['schema_version'] != 2:
        raise ValueError('schema_version must equal 2; use frozen source to replay legacy experiments')
    if config['scope'] != 'fused_subgraphs_fixed300':
        raise ValueError('scope must equal fused_subgraphs_fixed300')
    frozen('model', ('tokens', 'ffn'))
    positive('model', 'tokens')
    positive('model', 'ffn')
    frozen('numeric')
    frozen('platform')
    frozen('local_memory')
    frozen('architecture', ('multiply_latency', 'accumulator_feedback'))
    positive('architecture', 'multiply_latency')
    positive('architecture', 'accumulator_feedback')
    frozen('ln')
    for key in ('scenario_id', 'checkpoint_id', 'assumption_status'):
        if not isinstance(config['provenance'][key], str) or not config['provenance'][key]:
            raise ValueError('provenance.%s must be a nonempty string' % key)
    if config['provenance']['initial_state'] != template['provenance']['initial_state']:
        raise ValueError('provenance.initial_state must describe resident local activations/parameters and empty weight registers')
    seed = config['provenance']['seed']
    if type(seed) is not int or seed < 0:
        raise ValueError('provenance.seed must be a nonnegative integer')
    revision = config['provenance']['platform_revision']
    if revision is not None and (not isinstance(revision, str) or not revision):
        raise ValueError('provenance.platform_revision must be null or nonempty text')
    for key in ('source_document_sha256', 'amendment_sha256'):
        value = config['provenance'][key]
        if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
            raise ValueError('provenance.%s must be a lowercase SHA-256 digest' % key)
    for key, value in [('limit_mm2', config['area']['limit_mm2']), ('standard_cells_mm2', config['area']['standard_cells_mm2'])] + list(config['area']['macro_areas_mm2'].items()):
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0):
            raise ValueError('area.%s must be null or finite and positive' % key)
    return config


def source_manifest(root):
    root = Path(root).absolute()
    model_root = root if root.name == 'hw_model' else root/'hw_model'
    if not model_root.is_dir():
        raise ValueError('model source directory does not exist: %s' % model_root)
    files = {}
    for path in sorted(model_root.rglob('*')):
        if path.is_file() and path.suffix.lower() in ('.py', '.json', '.html', '.md') and not any(
                part in ('reports', 'runs', '__pycache__', '.pytest_cache') for part in path.relative_to(model_root).parts):
            files[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    encoded = json.dumps(files, sort_keys=True, separators=(',', ':')).encode('utf-8')
    return {'schema_version': 2, 'algorithm': 'sha256', 'files': files,
            'aggregate_sha256': hashlib.sha256(encoded).hexdigest(), 'python_version': platform.python_version()}
