"""Fixed-contract CLI, immutable source bundles, and exact result replay."""
import argparse
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
import shutil
import sys

from .compute import benchmarks
from .config import load_config, source_manifest
from .fixed_schedule import fixed_l0_benchmarks
from .reporting import write_report

ROOT = Path(__file__).absolute().parent.parent
MAPPINGS = ('A', 'B0', 'B1_G9', 'B1_G99')
OPERATORS = ('projection', 'ffn2')


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    sort_keys=True, allow_nan=False)+'\n', encoding='utf-8')


def config_hash(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()


def arithmetic_checks():
    pure = benchmarks()
    if pure['totals'] != {'array_9x16': 43076, 'row_dot128': 47916, 'useful_mac': 6120576}:
        raise AssertionError('preloaded compute mapping oracle mismatch')
    return {'schema_version': 2,
            'fixed300_weight_dominated_l0': fixed_l0_benchmarks(),
            'preloaded_compute_only_legacy_mapping_oracle': pure,
            'warning': 'L0 checks omit finite local accesses and request tails; use detailed results for ranking.'}


def run_scenarios(config, capture_trace=False):
    from .fixed_simulation import simulate
    results, trace = [], []
    for operator in OPERATORS:
        for mapping in MAPPINGS:
            result, events = simulate(config, mapping, operator, capture_trace=capture_trace)
            results.append(result)
            for event in events:
                item = dict(event)
                item.update(mapping=mapping, architecture=result['architecture'], operator=operator)
                trace.append(item)
            print('{} {}: {} cycles, internal output confirmed at {}'.format(
                operator, mapping, result['total_cycles'], result.get('last_internal_output_confirm')), flush=True)
    return results, trace


def _freeze_sources(destination, manifest):
    source_root = destination/'source'
    for relative, digest in manifest['files'].items():
        content = (ROOT/relative).read_bytes()
        if hashlib.sha256(content).hexdigest() != digest:
            raise ValueError('source changed while freezing: %s' % relative)
        target = source_root/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    for specification in sorted((ROOT/'hw_model'/'spec').glob('*.html')):
        target = source_root/'hw_model'/'spec'/specification.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(str(specification), str(target))
    readme = ROOT/'hw_model'/'README.md'
    if readme.exists():
        shutil.copyfile(str(readme), str(source_root/'hw_model'/'README.md'))


def compare(args):
    config = load_config(args.config)
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S_%f')
    destination = Path(args.out) if args.out else ROOT/'hw_model'/'reports'/('fixed300_'+stamp)
    destination.mkdir(parents=True, exist_ok=False)
    manifest = source_manifest(ROOT)
    _freeze_sources(destination, manifest)
    dump(destination/'arithmetic_baselines.json', arithmetic_checks())
    dump(destination/'invocation.json', {'argv': sys.argv, 'python': sys.version,
                                        'executable': sys.executable, 'client_timezone': 'Asia/Shanghai',
                                        'contract': 'fixed300_local64k_schema2'})
    capture_trace = bool(getattr(args, 'trace', False)) and not bool(getattr(args, 'no_trace', False))
    results, trace = run_scenarios(config, capture_trace=capture_trace)
    write_report(destination, config, results, manifest, trace)
    print('Report: {}'.format(destination.absolute()/'report.html'), flush=True)
    return 0


def replay(args):
    report = Path(args.report)
    original = json.loads((report/'summary.json').read_text(encoding='utf-8'))
    if original.get('schema_version') != 2:
        raise ValueError('legacy report: replay from its original frozen source/ directory')
    config = load_config(report/'run_config.json')
    if config_hash(config) != original['config_sha256']:
        raise ValueError('saved run_config.json hash does not match the report')
    manifest = source_manifest(ROOT)
    saved = original['source_manifest']
    if manifest['aggregate_sha256'] != saved['aggregate_sha256']:
        raise ValueError('source hashes differ; replay using the report frozen source/ directory')
    if manifest['python_version'] != saved['python_version']:
        raise ValueError('Python runtime differs from the saved experiment')
    results, _ = run_scenarios(config, capture_trace=False)
    if results != original['results']:
        raise AssertionError('replay differs from the saved experiment')
    if arithmetic_checks() != json.loads((report/'arithmetic_baselines.json').read_text(encoding='utf-8')):
        raise AssertionError('L0 checks differ from the saved experiment')
    print('Exact replay PASS: {} fixed-condition results'.format(len(results)))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description='KWT fixed 300 MB/s two-architecture scheduling comparison')
    commands = parser.add_subparsers(dest='command', required=True)
    command = commands.add_parser('compare', help='run A/B0/B1_G9/B1_G99 on projection and FFN2')
    command.add_argument('--config', help='schema-2 JSON configuration; timing contract is fixed')
    command.add_argument('--out', help='new output directory; existing results cannot be overwritten')
    capture = command.add_mutually_exclusive_group()
    capture.add_argument('--trace', action='store_true', help='capture potentially large detailed events')
    capture.add_argument('--no-trace', action='store_true', help='explicitly omit detailed trace (default)')
    command.set_defaults(func=compare)
    command = commands.add_parser('benchmarks', help='fixed300 L0 and preloaded compute mapping checks')
    command.add_argument('--out')
    def benchmark_command(args):
        answer = arithmetic_checks()
        if args.out:
            dump(args.out, answer)
        else:
            print(json.dumps(answer, ensure_ascii=False, indent=2))
        return 0
    command.set_defaults(func=benchmark_command)
    command = commands.add_parser('replay', help='exactly reproduce the eight fixed-condition results')
    command.add_argument('--report', required=True)
    command.set_defaults(func=replay)
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (ValueError, AssertionError, FileExistsError) as error:
        parser.exit(2, 'error: {}\n'.format(error))
