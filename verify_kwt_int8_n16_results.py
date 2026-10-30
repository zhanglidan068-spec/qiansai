"""Read-only W8A8/N16 artifact verification using NumPy and the standard library.

This script never loads audio or TensorFlow. It reads mmap headers plus samples
0, 31 and the final sample; complete logits and small parameter/LUT files are
checked in full. It prints JSON and never changes an experiment directory.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys

import numpy as np


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def inventory(directory):
    return {str(path.relative_to(directory)):
            dict(size=path.stat().st_size, mtime_ns=path.stat().st_mtime_ns)
            for path in sorted(directory.rglob('*')) if path.is_file()}


def which_set(filename, validation_percentage, testing_percentage):
    # Independent copy of the official filename-only split rule; no audio read.
    base = str(filename).replace('\\', '/').rsplit('/', 1)[-1]
    name = re.sub(r'_nohash_.*$', '', base)
    maximum = 2 ** 27 - 1
    percentage = ((int(hashlib.sha1(name.encode('utf-8')).hexdigest(), 16) %
                   (maximum + 1)) * (100.0 / maximum))
    if percentage < validation_percentage:
        return 'validation'
    if percentage < validation_percentage + testing_percentage:
        return 'testing'
    return 'training'


def verify(run):
    def load(name):
        return json.loads((run / name).read_text(encoding='utf-8'))

    require((run / 'result.json').is_file(), 'Incomplete run: missing result.json')
    result, metadata, specs = load('result.json'), load('run_metadata.json'), load('layer_outputs_manifest.json')
    config, proof, committed = load('quantization_config.json'), load('verification.json'), load('inference_complete.json')
    sample_manifest, calibration = load('sample_manifest.json'), load('calibration_manifest.json')
    n = result['samples']
    require(n > 0 and len(sample_manifest) == n, 'Invalid evaluated sample count')
    indices = sorted({0, min(31, n - 1), n - 1})
    stages = ['patch_output', 'encoder_input'] + ['layer_%02d_output' % i for i in range(1, 13)] + ['logits']
    expected = {branch + '/' + stage for branch in ('fp32', 'w8a8_n16') for stage in stages}
    expected.update('n16_codes/layer_%02d_output' % i for i in range(1, 13))
    expected.update('av_int8_f6/layer_%02d_attention' % i for i in range(1, 13))
    require(set(specs) == expected, '54-array manifest key coverage')
    actual_paths = {os.path.normcase(os.path.abspath(str(path))) for path in (run / 'layer_outputs').rglob('*.npy')}
    arrays, stored_paths, total_bytes = {}, set(), 0
    for key, spec in specs.items():
        path = run / spec['file']
        absolute = os.path.normcase(os.path.abspath(str(path)))
        require(os.path.commonpath([str(run), os.path.abspath(str(path))]) == str(run), 'Array path escapes run: ' + key)
        stored_paths.add(absolute)
        array = arrays[key] = np.load(str(path), mmap_mode='r', allow_pickle=False)
        branch, stage = key.split('/', 1)
        if branch == 'av_int8_f6':
            tail, dtype = [1, 99, 99], 'int8'
        else:
            tail = [12] if stage == 'logits' else [98 if stage == 'patch_output' else 99, 64]
            dtype = 'int16' if branch == 'n16_codes' else 'float32'
        require(list(array.shape) == spec['shape'] == [n] + tail, 'Shape mismatch: ' + key)
        require(str(array.dtype) == spec['dtype'] == dtype, 'Dtype mismatch: ' + key)
        require(array.flags.c_contiguous, 'Unexpected Fortran layout: ' + key)
        require(path.stat().st_size == array.offset + array.nbytes, 'Header/file extent mismatch: ' + key)
        total_bytes += path.stat().st_size
    require(actual_paths == stored_paths and len(arrays) == 54, 'Unexpected or missing NPY arrays')

    selected, seen = {}, set()
    with (run / 'per_sample_layer_metrics.csv').open(newline='', encoding='utf-8-sig') as stream:
        for row in csv.DictReader(stream):
            index, stage = int(row['sample_index']), row['stage']
            require(0 <= index < n and stage in stages, 'Invalid sample metric key')
            key = (index, stage)
            require(key not in seen, 'Duplicate sample metric key: ' + str(key))
            seen.add(key)
            if index in indices:
                selected[key] = row
    require(len(seen) == n * 15 and len(selected) == len(indices) * 15, 'Metric coverage mismatch')
    metric_columns = ['mae', 'rmse', 'max_abs_error', 'relative_l2', 'cosine']
    for index in indices:
        for stage in stages:
            r, q = [arrays[branch + '/' + stage][index].astype(np.float64) for branch in ('fp32', 'w8a8_n16')]
            require(np.isfinite(r).all() and np.isfinite(q).all(), 'Nonfinite sampled stage')
            error, r2, q2 = q - r, np.square(r).sum(), np.square(q).sum()
            values = [np.abs(error).mean(), np.sqrt(np.square(error).mean()), np.abs(error).max(),
                      np.sqrt(np.square(error).sum() / r2) if r2 else np.nan,
                      (r * q).sum() / np.sqrt(r2 * q2) if r2 * q2 else np.nan]
            np.testing.assert_allclose(values, [float(selected[(index, stage)][key]) for key in metric_columns],
                                       rtol=1e-10, atol=1e-12, equal_nan=True,
                                       err_msg='Sample %s stage %s metrics' % (index, stage))
        for layer in range(1, 13):
            name = 'layer_%02d_output' % layer
            codes = arrays['n16_codes/' + name][index]
            require(codes.min() >= -32767, 'Unused -32768 N16 code encountered')
            fraction = config['nonlinear']['layer_%02d/ln2' % layer]['output_f']
            reconstructed = codes.astype(np.float32) * np.float32(math.ldexp(1.0, -fraction))
            np.testing.assert_array_equal(reconstructed, arrays['w8a8_n16/' + name][index])
            codes = arrays['av_int8_f6/layer_%02d_attention' % layer][index]
            require(codes.min() >= 0 and codes.max() <= 64, 'AV sampled probability code outside [0,64]')

    with np.load(str(run / 'first_sample_trace.npz'), allow_pickle=False) as first:
        for key in first.files:
            require(not first[key].dtype.hasobject, 'Object array in first trace: ' + key)
        for layer in range(1, 13):
            prefix = 'nonlinear__layer_%02d__softmax' % layer
            codes15 = first[prefix + '__output_int16']
            require(codes15.dtype == np.int16 and codes15.min() >= 0, 'Softmax trace is not nonnegative int16')
            codes6 = np.rint(codes15.astype(np.float64) / 512.0).astype(np.int8)
            np.testing.assert_array_equal(codes6, first['first_q__layer_%02d__av__a' % layer])
            np.testing.assert_array_equal(codes6[0], arrays['av_int8_f6/layer_%02d_attention' % layer][0])
        for stage in stages:
            for prefix, branch in [('fp32', 'fp32'), ('quantized', 'w8a8_n16')]:
                np.testing.assert_array_equal(first[prefix + '__' + stage][0], arrays[branch + '/' + stage][0])
        for layer in range(1, 13):
            for kind in ('softmax', 'ln1', 'gelu', 'ln2'):
                for side in ('input_int16', 'output_int16'):
                    require(first['nonlinear__layer_%02d__%s__%s' % (layer, kind, side)].dtype == np.int16,
                            'Nonlinear trace boundary dtype')

    weight_keys = {key.replace('/', '__') for key in config['matmul_operands'] if key.endswith('/w')}
    require(len(weight_keys) == 74, 'Expected 74 configured weight matrices')
    with np.load(str(run / 'weights_int8.npz'), allow_pickle=False) as weights:
        require(set(weights.files) == weight_keys, 'Weight matrix coverage')
        for key in weights.files:
            value = weights[key]
            require(value.dtype == np.int8 and value.ndim == 2 and value.min() >= -127, 'Weight format: ' + key)
    with np.load(str(run / 'nonlinear_luts_int16.npz'), allow_pickle=False) as tables:
        require(set(tables.files) == {'softmax_exp_q15'} | {
            'layer_%02d__gelu__%s' % (layer, suffix) for layer in range(1, 13) for suffix in ('int16', 'clipped')},
            'LUT coverage')
        exp = tables['softmax_exp_q15']
        require(exp.dtype == np.int16 and exp.shape == (8193,), 'Softmax LUT format')
        require(exp[0] == 32767 and exp[-1] == 0 and exp.min() == 0 and np.all(np.diff(exp.astype(np.int32)) <= 0), 'Softmax LUT endpoints/order')
        for layer in range(1, 13):
            table, clipped = tables['layer_%02d__gelu__int16' % layer], tables['layer_%02d__gelu__clipped' % layer]
            require(table.dtype == np.int16 and table.shape == (65535,) and table[32767] == 0 and table.min() >= -32767, 'GELU LUT format')
            require(clipped.dtype == np.bool_ and clipped.shape == table.shape, 'GELU clip LUT format')

    with np.load(str(run / 'predictions.npz'), allow_pickle=False) as saved:
        r, q, targets = saved['fp32_logits'], saved['w8a8_n16_logits'], saved['target_ids']
        require(r.shape == q.shape == (n, 12) and targets.shape == (n,), 'Prediction shapes')
        require(r.dtype == q.dtype == np.float32 and np.issubdtype(targets.dtype, np.integer), 'Prediction dtypes')
        require(np.isfinite(r).all() and np.isfinite(q).all(), 'Nonfinite logits')
        np.testing.assert_array_equal(targets, [row['label_id'] for row in sample_manifest])
        for branch, logits in [('fp32', r), ('w8a8_n16', q)]:
            np.testing.assert_array_equal(logits, arrays[branch + '/logits'])
            confusion = np.zeros((12, 12), np.int64)
            np.add.at(confusion, (targets, logits.argmax(axis=1)), 1)
            np.testing.assert_array_equal(confusion, saved[branch + '_confusion_matrix'])
    with np.load(str(run / 'fresh_inference_logits.npz'), allow_pickle=False) as fresh:
        np.testing.assert_array_equal(r, fresh['fp32_logits'])
        np.testing.assert_array_equal(q, fresh['w8a8_n16_logits'])
    require(digest(run / 'fresh_inference_logits.npz') == committed['sha256'], 'Committed logits SHA256')
    ri, qi = r.argmax(axis=1), q.argmax(axis=1)
    ro, qo = ri == targets, qi == targets
    counts = dict(fp32_correct=int(ro.sum()), w8a8_n16_correct=int(qo.sum()),
                  changed_predictions_vs_fp32=int((ri != qi).sum()),
                  correct_to_wrong=int((ro & ~qo).sum()), wrong_to_correct=int((~ro & qo).sum()))
    for key, value in counts.items():
        require(result[key] == value, 'Result count mismatch: ' + key)
    require(result['fp32_accuracy'] == float(ro.mean()) and result['w8a8_n16_accuracy'] == float(qo.mean()), 'Accuracy count mismatch')
    require(len(load('changed_predictions.json')) == counts['changed_predictions_vs_fp32'], 'Changed prediction coverage')
    with np.load(str(Path(metadata['w16_reference']) / 'predictions.npz'), allow_pickle=False) as previous:
        w16 = previous['w16a16_logits'][:n]
        np.testing.assert_array_equal(targets, previous['target_ids'][:n])
        w16_ok = w16.argmax(axis=1) == targets
    require(result['previous_w16a16_correct'] == int(w16_ok.sum()) and
            result['previous_w16a16_accuracy'] == float(w16_ok.mean()), 'Previous W16 accuracy/count mismatch')
    require(result['accuracy_drop_vs_fp32_percentage_points'] == float((ro.mean() - qo.mean()) * 100), 'FP32 accuracy drop')
    require(result['accuracy_drop_vs_w16_percentage_points'] == float((w16_ok.mean() - qo.mean()) * 100), 'W16 accuracy drop')

    require(config['softmax_to_av']['fractional_bits'] == 6 and config['softmax_to_av']['row_renormalization'] is False, 'AV F6 contract')
    require(all(config['matmul_operands']['layer_%02d/av/a' % layer]['fractional_bits'] == 6 for layer in range(1, 13)), 'Per-layer AV F6 contract')
    require(config['softmax_output']['fractional_bits'] == 15, 'Softmax Q15 contract')
    require(proof['integer_matmuls'] == 98 and proof['int8_weight_matrices'] == 74, 'Integer graph coverage')
    require(proof['forbidden_floating_nonlinear_ops_in_quantized_graph'] == [], 'Forbidden floating nonlinear operations')
    require(proof['previous_prediction_reads_before_fresh_inference'] == [] and committed['previous_prediction_reads'] == [], 'Previous prediction read before fresh inference')
    for document in (proof, committed):
        requests = document['split_sample_requests']
        require(requests == dict(training=len(calibration), validation=n, testing=0), 'Split request mismatch')
    require(result['testing_samples_used'] == proof['testing_samples_used'] == 0, 'Testing samples used')
    require(config['calibration_split'] == metadata['calibration_split'] == 'training', 'Calibration split')
    require(result['calibration_samples'] == config['calibration_samples'] == len(calibration), 'Calibration sample count')
    flags = load('inference_flags.json')
    for rows, split in [(calibration, 'training'), (sample_manifest, 'validation')]:
        for row in rows:
            if not row['synthetic_zero_waveform']:
                require(which_set(row['file'], flags['validation_percentage'], flags['testing_percentage']) == split,
                        'Manifest filename belongs to a different split: ' + row['file'])

    before, after = load('previous_experiment_inventory_before.json'), load('previous_experiment_inventory_after.json')
    require(before == after, 'Historical inventory changed during run')
    for directory, expected_inventory in before.items():
        require(inventory(Path(directory)) == expected_inventory, 'Historical files changed: ' + directory)
    hashes = metadata['source_sha256']
    require(set(hashes) == {'run_kwt_int8_n16.py', 'kwt_int8_n16_graph.py', 'kwt_nonlinear_int16.py', 'run_kwt_fp32.py', 'run_kwt_int16.py'}, 'Frozen source coverage')
    for name, expected_hash in hashes.items():
        require(digest(run / 'source' / name) == expected_hash, 'Frozen source SHA256 mismatch: ' + name)

    return dict(status='passed', run=str(run), samples=n, array_files_verified=54,
                array_file_bytes=total_bytes, array_check_scope='All headers/extents; sample indices %s; complete logits' % indices,
                sampled_stage_metric_checks=len(indices) * 15, metric_rows_verified=len(seen),
                n16_reconstruction_checks=len(indices) * 12, av_range_sample_checks=len(indices) * 12,
                softmax_q15_to_av_q6_first_sample_checks=12, nonlinear_int16_boundary_trace_checks=96,
                int8_weight_matrices_verified=74, int16_luts_verified=13, frozen_source_hashes_verified=5,
                previous_experiments_unchanged=True, previous_predictions_read_before_new_logits=False,
                testing_samples_used=0, fp32_accuracy=float(ro.mean()), w8a8_n16_accuracy=float(qo.mean()), **counts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    run = parser.parse_args().run.absolute()
    try:
        report = verify(run)
    except Exception as error:
        print(json.dumps(dict(status='failed', run=str(run), error_type=type(error).__name__, error=str(error)), ensure_ascii=False, indent=2))
        return 1
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
