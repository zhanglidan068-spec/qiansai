"""Regression references must work in a checkout with no generated reports."""
import copy
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from unittest.mock import patch

from hw_model import reference_data
from hw_model.architecture_study import _hash, baseline_compatibility
from hw_model.array_grid_study import compare_overlap, OVERLAP_MAPPINGS
from hw_model.config import load_config, source_manifest
from hw_model.experiments import config_hash


# Pin the original report bytes independently of the distributable manifest.
# An intentional new reference should have a new identifier, not replace these.
ORIGINAL_FILES = {
    'fixed300_20261029_v2/run_config.json': (
        1955, 'a743bc52ba1503c1701a53dc4a879f5084c858371634ef7527cde7ca6e7ac417'),
    'fixed300_20261029_v2/summary.json': (
        155146, '880e58e9697dd04b61d5af9c5b9a9c5560e5e3922ffd7cecf17022b16b96e8a9'),
    'architecture_study_20261029_v1/run_config.json': (
        1955, 'a743bc52ba1503c1701a53dc4a879f5084c858371634ef7527cde7ca6e7ac417'),
    'architecture_study_20261029_v1/summary.json': (
        339169, '7d4ec95f40f0fae8677335f1e036f2ae02e60556dfe595ab449d174a8bdc49d7'),
    'architecture_study_ln64_20261029_v1/run_config.json': (
        1970, '1b7375f78d13e5d8ffbba603e5cbc1b829958f3b1f33588eb42f01fc592745f1'),
    'architecture_study_ln64_20261029_v1/summary.json': (
        2433382, '2e445d9ae15e1e92018cb4ce6a905994c77022e8611055953bdae86f92c8feb6'),
    'architecture_study_ln64_20261029_v1/historical_regression.json': (
        304440, 'dc18117f7ee5c55f08f3e50cfd17ef9048478a777d3965ccad0abdfd763125c0'),
}

CLI_REFERENCES = (
    ('architecture_study', 'BASELINE_REPORT', 'fixed300_20261029_v2'),
    ('architecture_study_32k_ln64', 'REFERENCE_REPORT', 'architecture_study_20261029_v1'),
    ('architecture_study_ln64', 'REFERENCE_REPORT', 'architecture_study_20261029_v1'),
    ('array_grid_study', 'REFERENCE_REPORT', 'architecture_study_ln64_20261029_v1'),
)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


class ReferenceDataTests(unittest.TestCase):
    def test_seven_references_preserve_original_bytes_and_source_manifest(self):
        root = reference_data.REFERENCE_RESULTS
        manifest = read_json(root / 'manifest.json')
        self.assertEqual(manifest['schema_version'], 1)
        self.assertEqual(manifest['algorithm'], 'sha256')
        required = {reference_id + '/' + filename
                    for reference_id, filenames in reference_data.REFERENCE_FILES.items()
                    for filename in filenames}
        self.assertEqual(required, set(ORIGINAL_FILES))
        self.assertEqual(set(manifest['files']), required)
        self.assertEqual({path.relative_to(root).as_posix()
                          for path in root.glob('*/*.json')}, required)
        model_manifest = source_manifest(Path(__file__).absolute().parents[2])
        for relative, (size, digest) in ORIGINAL_FILES.items():
            with self.subTest(file=relative):
                content = (root / relative).read_bytes()
                self.assertEqual(len(content), size)
                self.assertEqual(hashlib.sha256(content).hexdigest(), digest)
                self.assertEqual(manifest['files'][relative], {'bytes': size, 'sha256': digest})
                self.assertEqual(model_manifest['files']['hw_model/reference_results/' + relative], digest)
        for reference_id in reference_data.REFERENCE_FILES:
            self.assertEqual(reference_data.reference_report_path(reference_id), root / reference_id)

    def test_cli_paths_and_reference_configuration_result_chain(self):
        for module_name, constant, reference_id in CLI_REFERENCES:
            with self.subTest(module=module_name):
                module = importlib.import_module('hw_model.' + module_name)
                path = getattr(module, constant)
                self.assertEqual(path, reference_data.REFERENCE_RESULTS / reference_id)
                self.assertNotIn('reports', path.parts)
                self.assertTrue(callable(module.main))
        datasets = {}
        for reference_id, count in (
                ('fixed300_20261029_v2', 8),
                ('architecture_study_20261029_v1', 16),
                ('architecture_study_ln64_20261029_v1', 56)):
            directory = reference_data.reference_report_path(reference_id)
            config = load_config(directory / 'run_config.json')
            summary = read_json(directory / 'summary.json')
            datasets[reference_id] = summary
            self.assertEqual(config_hash(config), summary['config_sha256'])
            self.assertEqual(len(summary['results']), count)
            self.assertEqual(len({(row['operator'], row['mapping'])
                                  for row in summary['results']}), count)
            self.assertEqual(config['local_memory']['banks'], 8)
            self.assertEqual(config['ln']['latency_cycles'],
                             64 if 'ln64' in reference_id else 128)
            for row in summary['results']:
                model = config['model']
                self.assertEqual(row['shape'], {
                    'm': model['tokens'], 'n': model['hidden'],
                    'r': model['hidden'] if row['operator'] == 'projection' else model['ffn']})
        baseline = datasets['fixed300_20261029_v2']
        old = datasets['architecture_study_20261029_v1']
        current = datasets['architecture_study_ln64_20261029_v1']
        self.assertEqual(old['baseline_reference_sha256'], _hash(baseline))
        self.assertTrue(baseline_compatibility(old['results'], baseline)['passed'])
        self.assertEqual(current['historical_reference_summary_sha256'], _hash(old))
        self.assertEqual(current['historical_reference_config_sha256'], old['config_sha256'])
        evidence = read_json(reference_data.reference_report_path(
            'architecture_study_ln64_20261029_v1') / 'historical_regression.json')
        self.assertTrue(evidence['passed'])
        self.assertFalse(evidence['mixed_into_new_ranking'])
        self.assertEqual(evidence['reference_summary_sha256'], _hash(old))
        self.assertEqual(evidence['reference_config_sha256'], old['config_sha256'])
        self.assertEqual(evidence['results'], old['results'])
        self.assertEqual(len(evidence['comparisons']), 16)
        self.assertTrue(all(row['exact_dict_equal'] and not row['differences']
                            for row in evidence['comparisons']))

    def test_grid_eight_overlap_reference_is_usable_and_detects_changed_result(self):
        summary = read_json(reference_data.reference_report_path(
            'architecture_study_ln64_20261029_v1') / 'summary.json')
        cases = [copy.deepcopy(case) for case in summary['cases']
                 if case['mapping'] in OVERLAP_MAPPINGS]
        comparison = compare_overlap(cases, summary)
        self.assertEqual(len(cases), 8)
        self.assertTrue(comparison['passed'])
        self.assertEqual(comparison['case_count'], 8)
        self.assertEqual(comparison['additional_simulation_count'], 0)
        self.assertTrue(all(row['exact_dict_equal'] and not row['differences']
                            for row in comparison['comparisons']))
        cases[0]['result']['total_cycles'] += 1
        changed = compare_overlap(cases, summary)
        self.assertFalse(changed['passed'])
        failures = [row for row in changed['comparisons'] if not row['exact_dict_equal']]
        self.assertEqual(len(failures), 1)
        self.assertTrue(any('total_cycles' in item for item in failures[0]['differences']))

    def test_missing_or_tampered_references_fail_without_regenerating_expectations(self):
        workspace = Path(__file__).absolute().parents[2]
        with tempfile.TemporaryDirectory(prefix='kwt-reference-test-', dir=str(workspace)) as temporary:
            fixture = Path(temporary) / 'reference_results'
            shutil.copytree(str(reference_data.REFERENCE_RESULTS), str(fixture))
            with patch.object(reference_data, 'REFERENCE_RESULTS', fixture):
                with self.assertRaisesRegex(ValueError, 'unknown'):
                    reference_data.reference_report_path('unknown_study')
                for relative in ORIGINAL_FILES:
                    with self.subTest(file=relative):
                        reference_id = relative.split('/')[0]
                        path = fixture / relative
                        original = path.read_bytes()
                        path.unlink()
                        with self.assertRaisesRegex(FileNotFoundError, 'missing'):
                            reference_data.reference_report_path(reference_id)
                        self.assertFalse(path.exists(), 'missing expectations must not be generated')
                        # A valid JSON trailing newline becomes a space: same length,
                        # same parsed data, different historical source bytes.
                        self.assertEqual(original[-1:], b'\n')
                        path.write_bytes(original[:-1] + b' ')
                        with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                            reference_data.reference_report_path(reference_id)
                        path.write_bytes(original + b'\n')
                        with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                            reference_data.reference_report_path(reference_id)
                        path.write_bytes(original)
                manifest_path = fixture / 'manifest.json'
                original_manifest = manifest_path.read_bytes()
                manifest_path.unlink()
                with self.assertRaisesRegex(FileNotFoundError, 'manifest is missing'):
                    reference_data.reference_report_path('fixed300_20261029_v2')
                self.assertFalse(manifest_path.exists())
                manifest_path.write_bytes(original_manifest)
                manifest = read_json(manifest_path)
                del manifest['files']['fixed300_20261029_v2/summary.json']
                manifest_path.write_text(json.dumps(manifest), encoding='utf-8')
                with self.assertRaisesRegex(ValueError, 'no checksum'):
                    reference_data.reference_report_path('fixed300_20261029_v2')

    def test_package_without_reports_imports_all_clis_and_runs_grid_two_case_smoke(self):
        workspace = Path(__file__).absolute().parents[2]
        manifest = source_manifest(workspace)
        with tempfile.TemporaryDirectory(prefix='kwt-reference-test-', dir=str(workspace)) as temporary:
            clean = Path(temporary)
            # Copy distributable source types only; generated reports, runs and
            # bytecode are excluded by the same source-freezing inventory.
            for relative in manifest['files']:
                source = workspace / relative
                target = clean / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(str(source), str(target))
            self.assertFalse((clean / 'hw_model' / 'reports').exists())
            script = textwrap.dedent('''
                import builtins
                import importlib
                import io
                from pathlib import Path

                # Block both builtins.open and pathlib's io.open even if an
                # import accidentally reaches an original-workspace report.
                def guarded(original):
                    def opened(file, *args, **kwargs):
                        if isinstance(file, (str, bytes, Path)):
                            path = str(file).replace('\\\\', '/').lower()
                            if 'reports' in path.split('/'):
                                raise AssertionError('generated reports access: ' + path)
                        return original(file, *args, **kwargs)
                    return opened
                builtins.open = guarded(builtins.open)
                io.open = guarded(io.open)
                import hw_model
                from hw_model.config import load_config, LN64_LOCAL64K_CONFIG, source_manifest
                from hw_model import reference_data
                from hw_model import array_grid_study as grid
                assert Path(hw_model.__file__).absolute().parent == Path.cwd() / 'hw_model'
                assert not (Path.cwd() / 'hw_model' / 'reports').exists()
                for name, constant, reference_id in (
                        ('architecture_study', 'BASELINE_REPORT', 'fixed300_20261029_v2'),
                        ('architecture_study_32k_ln64', 'REFERENCE_REPORT', 'architecture_study_20261029_v1'),
                        ('architecture_study_ln64', 'REFERENCE_REPORT', 'architecture_study_20261029_v1'),
                        ('array_grid_study', 'REFERENCE_REPORT', 'architecture_study_ln64_20261029_v1')):
                    module = importlib.import_module('hw_model.' + name)
                    assert getattr(module, constant) == reference_data.reference_report_path(reference_id)
                    try:
                        module.main(['--help'])
                    except SystemExit as done:
                        assert done.code == 0
                    else:
                        raise AssertionError('CLI help did not use argparse')
                files = source_manifest(Path.cwd())['files']
                for reference_id, names in reference_data.REFERENCE_FILES.items():
                    for name in names:
                        assert 'hw_model/reference_results/' + reference_id + '/' + name in files
                config = load_config(LN64_LOCAL64K_CONFIG, overrides={
                    'model': {'tokens': 1, 'ffn': 5}})
                cases = grid.run_cases(config, mappings=('A26x6',), workers=1)
                assert len(cases) == 2
                assert [case['operator'] for case in cases] == ['projection', 'ffn2']
                assert [case['shape']['r'] for case in cases] == [64, 5]
                for case in cases:
                    assert case['status'] == 'feasible'
                    assert case['result']['useful_mac'] == case['shape']['m'] * 64 * case['shape']['r']
                    assert case['result']['weight_port']['physical_weight_bytes'] == case['weight_ledger']['physical_weight_bytes']
                    assert case['result']['last_internal_output_confirm'] == case['result']['total_cycles']
                print('CLEAN_REFERENCE_GRID_SMOKE_OK')
            ''')
            environment = dict(os.environ)
            environment['PYTHONPATH'] = str(clean)
            environment['PYTHONDONTWRITEBYTECODE'] = '1'
            result = subprocess.run([sys.executable, '-c', script], cwd=str(clean),
                                    env=environment, stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE, universal_newlines=True,
                                    encoding='utf-8', timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn('CLEAN_REFERENCE_GRID_SMOKE_OK', result.stdout)


if __name__ == '__main__':
    unittest.main()
