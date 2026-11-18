"""Independent NumPy-only audit of a completed nibble W8A8 experiment.

Reads experiment artifacts and acoustic-cache hashes, never audio or a model.
No inference module is imported.  Use --allow-incomplete for a read-only audit
of committed trial result files while an experiment is running.  --output
creates a new JSON file exclusively and refuses to overwrite existing files.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import ntpath
from pathlib import Path
import re

import numpy as np


ENCODER_MACS = 73446912
STAGES = ['patch_output', 'encoder_input'] + [
    'layer_{:02d}_output'.format(i) for i in range(1, 13)] + ['logits']


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def close(actual, expected, message, atol=1e-11, rtol=1e-9):
    if actual is None or expected is None:
        require(actual is expected, message + ': null mismatch')
    else:
        require(math.isclose(float(actual), float(expected), rel_tol=rtol, abs_tol=atol),
                '{}: {} != {}'.format(message, actual, expected))


def split_of(filename):
    # Independently reproduce Speech Commands' official speaker grouping.
    name = re.sub(r'_nohash_.*$', '', ntpath.basename(filename))
    maximum = 2 ** 27 - 1
    percent = (int(hashlib.sha1(name.encode('utf-8')).hexdigest(), 16) %
               (maximum + 1)) * (100. / maximum)
    return 'validation' if percent < 10 else ('testing' if percent < 20 else 'training')


def macs(op):
    require(re.fullmatch(r'layer_(0[1-9]|1[0-2])/(q|k|v|o|fc1|fc2)', op) is not None,
            'Unknown approximated operation: ' + op)
    return 99 * 64 * (256 if op.endswith(('/fc1', '/fc2')) else 64)


def rebuild_lut(policy):
    domain = np.arange(-127, 128, dtype=np.int64)
    a, w = np.broadcast_arrays(domain[:, None], domain[None, :])
    u, v = np.abs(a), np.abs(w)
    ah, al, wh, wl = u // 16, u % 16, v // 16, v % 16
    terms = [256 * wh * ah, 16 * wh * al, 16 * wl * ah, wl * al]
    sums = np.cumsum(np.stack(terms), axis=0)
    bounds = [np.minimum(240 * wh + 15 * u, 240 * ah + 15 * v),
              15 * u, np.minimum(15 * wl, 15 * al)]
    counts = np.full((255, 255), 4, dtype=np.uint8)
    mode, threshold, allowed = policy['mode'], policy['threshold'], policy['stages']
    require(mode in ('exact', 'sigma', 'ratio') and allowed in ('all', 'll_only'), 'Invalid LUT policy')
    require(math.isfinite(threshold) and threshold >= 0, 'Invalid threshold')
    require(policy['magnitude_split'] is True and policy['comparison'] == '>=', 'Unexpected arithmetic definition')
    if mode != 'exact':
        for stage in ([2] if allowed == 'll_only' else [0, 1, 2]):
            decision = sums[stage] >= (threshold if mode == 'sigma' else threshold * bounds[stage])
            counts[(counts == 4) & decision] = stage + 1
    magnitude = np.take_along_axis(sums, counts[None, :, :].astype(np.int64) - 1, axis=0)[0]
    products = np.where((a < 0) ^ (w < 0), -magnitude, magnitude).astype(np.int32)
    if policy.get('zero_tail_guard', False):
        # A zero remaining-tail bound permits an exact early stop.  The
        # numerical product stays the raw policy's product; only its scheduled
        # term count changes.  65536 exceeds every possible nonzero-bound ratio
        # in the symmetric INT8 domain, matching the experiment's exact guard.
        exact_stops = np.full((255, 255), 4, dtype=np.uint8)
        for stage in range(3):
            safe = sums[stage] >= 65536 * bounds[stage]
            exact_stops[(exact_stops == 4) & safe] = stage + 1
        counts = np.minimum(counts, exact_stops).astype(np.uint8)
    return products, counts


def verify_source_records(run):
    records = [(run / 'source', run / 'source_sha256.json')]
    records += [(p.parent, p) for p in sorted((run / 'source_versions').glob('*/source_sha256.json'))]
    versions = []
    for directory, record in records:
        hashes = read_json(record)
        for name, expected in hashes.items():
            require(sha256(directory / name) == expected, 'Source snapshot hash differs: ' + str(directory / name))
        versions.append(dict(directory=str(directory.relative_to(run)), files=len(hashes), hashes=hashes))
    require(bool(versions), 'No source snapshots')
    native = read_json(run / 'native/build_manifest.json')
    require(sha256(run / 'native/nibble_matmul.cpp') == native['source_sha256'], 'Native C++ snapshot mismatch')
    library = run / 'native/nibble_matmul.dll'
    if not library.exists():
        library = run / 'native/nibble_matmul.so'
    require(sha256(library) == native['library_sha256'], 'Native compiled-library mismatch')
    require(all(v['hashes']['nibble_matmul.cpp'] == native['source_sha256'] for v in versions),
            'Native arithmetic changed between source versions')
    if len(versions) > 1:
        fix = read_json(run / 'development_fix_record.json')
        require(fix['affected_trials'] == [] and fix['arithmetic_changed'] is False,
                'Unexpected source-version development history')
    guarded_source = run / 'guarded_source'
    guarded_hashes = None
    if guarded_source.exists():
        guarded_hashes = read_json(guarded_source / 'source_sha256.json')
        for name, expected in guarded_hashes.items():
            require(sha256(guarded_source / name) == expected, 'Guarded source snapshot differs: ' + name)
    return dict(versions=versions, native_library_sha256=native['library_sha256'],
                native_source_sha256=native['source_sha256'], guarded_source_hashes=guarded_hashes)


def layer_metrics(reference, candidate, indices):
    totals = dict(elements=0, abs_sum=0., err2=0., ref2=0., candidate2=0., dot=0.,
                  max_abs_error=0., reference_max_abs=0., candidate_max_abs=0.)
    for start in range(0, len(indices), 32):
        r = np.asarray(reference[indices[start:start + 32]], dtype=np.float64)
        q = np.asarray(candidate[start:start + 32], dtype=np.float64)
        require(np.isfinite(r).all() and np.isfinite(q).all(), 'Nonfinite stage output')
        e = q - r
        totals['elements'] += r.size
        for name, value in [('abs_sum', np.abs(e).sum()), ('err2', (e * e).sum()),
                            ('ref2', (r * r).sum()), ('candidate2', (q * q).sum()), ('dot', (r * q).sum())]:
            totals[name] += float(value)
        for name, value in [('max_abs_error', np.abs(e).max()),
                            ('reference_max_abs', np.abs(r).max()), ('candidate_max_abs', np.abs(q).max())]:
            totals[name] = max(totals[name], float(value))
    n, er, rr, qq = totals['elements'], totals['err2'], totals['ref2'], totals['candidate2']
    return dict(elements=n, mae=totals['abs_sum'] / n, rmse=math.sqrt(er / n),
                max_abs_error=totals['max_abs_error'], relative_l2=math.sqrt(er / rr) if rr else None,
                cosine=totals['dot'] / math.sqrt(rr * qq) if rr * qq else None,
                sqnr_db=10 * math.log10(rr / er) if rr and er else None,
                fp32_max_abs=totals['reference_max_abs'], w16a16_max_abs=totals['candidate_max_abs'])


def audit(run, allow_incomplete=False):
    run = Path(run).absolute()
    metadata = read_json(run / 'run_metadata.json')
    manifest = read_json(run / 'sample_manifest.json')
    calibration = read_json(run / 'calibration_manifest.json')
    targets = np.asarray([row['label_id'] for row in manifest], dtype=np.int64)
    screen = np.asarray(read_json(run / 'screening_indices.json'), dtype=np.int64)
    require(len(targets) == 4445 and len(calibration) == 256 and len(screen) == 384, 'Unexpected sample counts')
    require(len(np.unique(screen)) == len(screen) and screen.min() >= 0 and screen.max() < len(targets), 'Invalid screening indices')
    require(np.array_equal(np.bincount(targets[screen], minlength=12), np.full(12, 32)), 'Screen is not class-balanced')
    for rows, expected in [(manifest, 'validation'), (calibration, 'training')]:
        for row in rows:
            if not row['synthetic_zero_waveform']:
                require(split_of(row['file']) == expected, 'Manifest split mismatch')
    for key in ('testing_samples_used', 'audio_files_opened'):
        require(metadata[key] == 0, 'Forbidden data-use metadata: ' + key)
    for key in ('training_or_weight_updates', 'historical_prediction_files_read',
                'synthesized', 'hardware_speedup_measured', 'area_compliance_verified'):
        require(metadata[key] is False, 'Unexpected verification claim: ' + key)
    for filename, digest in metadata['input_sha256'].items():
        require(sha256(filename) == digest, 'Input cache/provenance hash mismatch: ' + filename)
    parent = Path(metadata['parent'])
    require(sha256(parent / 'trials/encoder_w8a8_p8f7__full/quantization.json') == metadata['quantization_sha256'],
            'Parent quantization hash mismatch')
    quantization = read_json(run / 'quantization.json')
    require(quantization == read_json(parent / 'trials/encoder_w8a8_p8f7__full/quantization.json'),
            'Quantization contents changed')
    require(len(quantization['operand_bits']) == 196 and set(quantization['operand_bits'].values()) == {8},
            'W8A8 operand widths changed')
    require(all(value == 7 for key, value in quantization['fractional_bits'].items() if key.endswith('/av/a')),
            'Softmax-to-AV no longer P8/F7')
    for name, digest in metadata['checkpoint_sha256'].items():
        require(sha256(run.parent.parent / 'keyword-transformer/models_data_v2_12_labels/kwt1' / name) == digest,
                'Official checkpoint changed')
    provenance = verify_source_records(run)
    baselines = {}
    for name, expected_correct in [('fp32', 4340), ('w8a8', 4338)]:
        with np.load(str(run / 'baseline' / name / 'predictions.npz'), allow_pickle=False) as saved:
            baselines[name] = saved['logits'].copy()
            require(np.array_equal(saved['target_ids'], targets), 'Baseline targets mismatch')
        require(baselines[name].shape == (4445, 12) and baselines[name].dtype == np.float32, 'Baseline logits format differs')
        require(int((baselines[name].argmax(1) == targets).sum()) == expected_correct, 'Fresh baseline score mismatch')
    base_arrays = {stage: np.load(str(run / 'baseline/w8a8/layer_outputs' / (stage + '.npy')),
                                 mmap_mode='r', allow_pickle=False) for stage in STAGES}
    for stage, array in base_arrays.items():
        expected_shape = (4445, 12) if stage == 'logits' else (4445, 98 if stage == 'patch_output' else 99, 64)
        require(array.shape == expected_shape and array.dtype == np.float32, 'Baseline stage shape differs: ' + stage)
        for start in range(0, len(array), 256):
            require(np.isfinite(array[start:start + 256]).all(), 'Nonfinite baseline stage: ' + stage)
    require(np.array_equal(base_arrays['logits'], baselines['w8a8']), 'Baseline stored logits stage mismatch')
    trials = []
    matching_predictions = {}
    exact_screen_roundoff = []
    paths = sorted((run / 'baseline').glob('*/result.json')) + sorted((run / 'trials').glob('*/result.json'))
    for result_path in paths:
        directory = result_path.parent
        result, policy = read_json(result_path), read_json(directory / 'policy.json')
        indices = np.load(str(directory / 'sample_indices.npy'), allow_pickle=False)
        require(indices.dtype == np.int64 and indices.ndim == 1 and len(np.unique(indices)) == len(indices), 'Invalid trial indices')
        require(len(indices) == result['samples'] and indices.min() >= 0 and indices.max() < len(targets), 'Trial sample count mismatch')
        if result['scope'] == 'full':
            require(np.array_equal(indices, np.arange(4445)), 'Full trial does not cover exact validation order')
        elif result['scope'] == 'screen':
            require(np.array_equal(indices, screen), 'Screen sample order changed')
        else:
            require(result['scope'] == 'control' and np.array_equal(indices, np.arange(64)), 'Unexpected control indices')
        with np.load(str(directory / 'predictions.npz'), allow_pickle=False) as saved:
            logits, true = saved['logits'].copy(), saved['target_ids'].copy()
        require(logits.dtype == np.float32 and logits.shape == (len(indices), 12) and np.isfinite(logits).all(), 'Invalid trial logits')
        require(true.dtype == np.int64 and np.array_equal(true, targets[indices]), 'Trial targets mismatch')
        prediction, truth = logits.argmax(1), targets[indices]
        correct = int((prediction == truth).sum())
        fp_correct = int((baselines['fp32'][indices].argmax(1) == truth).sum())
        w8_prediction = baselines['w8a8'][indices].argmax(1)
        w8_correct = int((w8_prediction == truth).sum())
        require(correct == result['correct'] and fp_correct == result['fp32_correct'], 'Independent re-score differs: ' + result['name'])
        close(result['accuracy'], correct / float(len(indices)), 'Accuracy differs')
        close(result['drop_vs_fp32_pp'], 100. * (fp_correct - correct) / len(indices), 'FP32 drop differs')
        if result['w8a8_correct'] is not None:
            require(w8_correct == result['w8a8_correct'], 'W8A8 score differs')
            close(result['additional_drop_vs_w8a8_pp'], 100. * (w8_correct - correct) / len(indices), 'Additional accuracy drop differs')
            require(result['passes_extra_0_1pp'] == (100. * (w8_correct - correct) / len(indices) <= .1 + 1e-12), 'Strict accuracy flag differs')
        if 'prediction_flips_vs_w8a8' in result:
            require(result['prediction_flips_vs_w8a8'] == int((prediction != w8_prediction).sum()), 'Flip count differs')
            require(result['newly_wrong_vs_w8a8'] == int(((w8_prediction == truth) & (prediction != truth)).sum()), 'New error count differs')
            require(result['corrected_vs_w8a8'] == int(((w8_prediction != truth) & (prediction == truth)).sum()), 'Corrected count differs')
            require(result['logits_bit_exact_vs_w8a8'] == bool(np.array_equal(logits, baselines['w8a8'][indices])), 'Bit-exact flag differs')
        require(result['passes_fp32_2pp'] == (100. * (fp_correct - correct) / len(indices) <= 2. + 1e-12), 'FP32 threshold flag differs')
        require(result['testing_samples_used'] == 0 and result['hardware_speedup_measured'] is False, 'Invalid result claim')
        products, exits = rebuild_lut(policy)
        digest = hashlib.sha256(products.tobytes() + exits.tobytes()).hexdigest()
        require(digest == policy['lut_sha256'], 'Independent LUT hash differs')
        with np.load(str(directory / 'arithmetic_lut.npz'), allow_pickle=False) as saved:
            require(np.array_equal(saved['product_lut'], products) and np.array_equal(saved['stage_lut'], exits), 'Saved LUT differs')
            if policy.get('zero_tail_guard', False):
                require(policy['zero_tail_guard_mode'] == 'ratio' and policy['zero_tail_guard_threshold'] == 65536. and
                        policy['zero_tail_guard_stages'] == 'all', 'Unexpected exact zero-tail guard')
                raw_policy = dict(policy, zero_tail_guard=False)
                raw_products, raw_exits = rebuild_lut(raw_policy)
                zero_products, zero_exits = rebuild_lut(dict(raw_policy, mode='ratio', threshold=65536., stages='all'))
                require(np.array_equal(saved['raw_stage_lut'], raw_exits), 'Saved raw stage LUT differs')
                require(np.array_equal(saved['zero_tail_stage_lut'], zero_exits), 'Saved exact-tail stage LUT differs')
                require(np.array_equal(raw_products, products), 'Zero-tail guard changed raw numerical products')
                smaller = zero_exits < raw_exits
                require(np.array_equal(raw_products[smaller], zero_products[smaller]), 'Guard reduced an inexact numerical tail')
                require(hashlib.sha256(raw_products.tobytes()).hexdigest() == policy['raw_product_lut_sha256'],
                        'Raw product hash differs')
        domain = np.arange(-127, 128, dtype=np.int32)
        arithmetic_exact = bool(np.array_equal(products, domain[:, None] * domain[None, :]))
        if arithmetic_exact and directory != run / 'baseline/fp32':
            if result['scope'] in ('full', 'control'):
                require(np.array_equal(logits, baselines['w8a8'][indices]),
                        'An exact numerical product LUT changed predictions: ' + str(directory))
                if (directory / 'layer_comparison.json').exists():
                    require(all(row['max_abs_error'] == 0 for row in read_json(directory / 'layer_comparison.json')),
                            'An exact numerical product LUT changed intermediate stages')
            else:
                # Screen inputs occupy different batch slots from full inputs.
                # FP32 patch/head GEMM roundoff can cross quantization boundaries.
                # This is measured separately; full same-order controls are strict.
                exact_screen_roundoff.append(dict(name=result['name'],
                    max_logit_difference_vs_full_batch_baseline=float(np.abs(logits - baselines['w8a8'][indices]).max()),
                    prediction_flips=int((prediction != w8_prediction).sum())))
        numeric_key = json.dumps({key: policy[key] for key in
                                 ('mode', 'threshold', 'stages', 'operations', 'magnitude_split', 'comparison')}, sort_keys=True)
        matching_predictions.setdefault((numeric_key, result['scope']), {})[
            bool(policy.get('zero_tail_guard', False))] = logits.copy()
        counts = read_json(directory / 'operation_counts.json')
        require(set(counts['per_operation']) == set(policy['operations']), 'Count operation set differs')
        total = np.zeros(4, dtype=np.int64)
        for op, row in counts['per_operation'].items():
            row = np.asarray(row, dtype=np.int64)
            require(row.shape == (4,) and (row >= 0).all() and int(row.sum()) == macs(op) * len(indices), 'Per-operation count differs')
            total += row
        selected = sum(macs(op) for op in policy['operations'])
        slots = int(total.dot(np.arange(1, 5, dtype=np.int64)))
        saved_slots = 4 * selected * len(indices) - slots
        require(total.tolist() == counts['total_exit_counts'] == result['exit_stage_counts'], 'Exit counts differ')
        require(counts['selected_scalar_products'] == selected * len(indices), 'Selected product count differs')
        require(counts['executed_partial_product_slots'] == slots and counts['saved_partial_product_slots'] == saved_slots, 'Term-slot counts differ')
        require(counts['full_encoder_four_term_slots'] == 4 * ENCODER_MACS * len(indices), 'Encoder denominator differs')
        require(counts['excludes_padding'] is True and counts['physical_cycle_measurement'] is False, 'Count semantics differ')
        close(result['selected_macs_per_sample'], selected, 'Selected MACs differ')
        close(result['mean_terms_selected'], slots / float(selected * len(indices)) if selected else 4., 'Mean terms differ')
        close(result['saved_partial_product_fraction_selected'], saved_slots / float(4 * selected * len(indices)) if selected else 0., 'Selected savings differ')
        close(result['saved_partial_product_fraction_encoder'], saved_slots / float(4 * ENCODER_MACS * len(indices)), 'Encoder savings differ')
        stages_checked = []
        layer_directory = directory / 'layer_outputs'
        if layer_directory.is_dir():
            rows = read_json(directory / 'layer_comparison.json') if (directory / 'layer_comparison.json').exists() else []
            comparison = {row['stage']: row for row in rows}
            for stage in STAGES:
                array = np.load(str(layer_directory / (stage + '.npy')), mmap_mode='r', allow_pickle=False)
                require(array.dtype == np.float32 and array.shape == (len(indices),) + base_arrays[stage].shape[1:], 'Candidate stage shape differs')
                if stage == 'logits':
                    require(np.array_equal(array, logits), 'Saved candidate logits stage differs')
                if comparison:
                    measured = layer_metrics(base_arrays[stage], array, indices)
                    for key, value in measured.items():
                        close(comparison[stage][key], value, 'Stage metric differs: ' + stage + '/' + key)
                stages_checked.append(stage)
                del array
        if policy['name'] == 'native_four_term_exact_control':
            require(result['logits_bit_exact_vs_w8a8'], 'Native exact control differs')
            require(all(row['max_abs_error'] == 0 for row in read_json(directory / 'layer_comparison.json')), 'Native intermediate control differs')
        trials.append(dict(directory=directory.relative_to(run).as_posix(), name=result['name'], scope=result['scope'],
                           samples=len(indices), correct=correct, recomputed=True, counts_verified=True,
                           zero_tail_guard=bool(policy.get('zero_tail_guard', False)),
                           lut_sha256=digest, full_stage_arrays_checked=stages_checked))
    histogram_checks = 0
    training_metric_checks = 0
    hist_path = run / 'training_pair_histograms.npz'
    if hist_path.exists():
        histograms = {}
        with np.load(str(hist_path), allow_pickle=False) as saved:
            require(len(saved.files) == 72, 'Training histograms incomplete')
            for key in saved.files:
                op = key.replace('__', '/')
                hist = saved[key]
                require(hist.dtype == np.int64 and hist.shape == (255, 255) and (hist >= 0).all(), 'Invalid training histogram')
                require(int(hist.sum()) == macs(op) * 256, 'Training histogram sample/product count differs')
                histograms[op] = hist.copy()
                histogram_checks += 1
        metrics_path = run / 'training_policy_metrics.json'
        if metrics_path.exists():
            for row in read_json(metrics_path):
                candidate_policy = read_json(run / 'trials' / (row['name'] + '__screen') / 'policy.json')
                unused, exits = rebuild_lut(candidate_policy)
                histogram = sum((histograms[op] for op in candidate_policy['operations']),
                                np.zeros((255, 255), dtype=np.int64))
                scalar_products = int(histogram.sum())
                executed_slots = int((histogram * exits).sum())
                require(row['training_scalar_products'] == scalar_products, 'Training metric product count differs')
                require(row['selected_macs_per_sample'] == sum(macs(op) for op in candidate_policy['operations']),
                        'Training selected MAC count differs')
                close(row['training_mean_terms'], executed_slots / float(scalar_products) if scalar_products else 4.,
                      'Training mean term count differs')
                close(row['training_saved_fraction_selected'], 1. - executed_slots / float(scalar_products * 4) if scalar_products else 0.,
                      'Training selected saving differs')
                close(row['training_saved_fraction_encoder'], (scalar_products * 4 - executed_slots) / float(256 * ENCODER_MACS * 4),
                      'Training encoder saving differs')
                training_metric_checks += 1
        del histograms
    matching_guarded_raw_checks = 0
    for matches in matching_predictions.values():
        if False in matches and True in matches:
            require(np.array_equal(matches[False], matches[True]), 'Guarded candidate changed raw candidate predictions')
            matching_guarded_raw_checks += 1
    del matching_predictions
    completion = None
    if (run / 'completion.json').exists():
        completion = read_json(run / 'completion.json')
        require(completion['completed'] is True and completion['full_repeat_logits_bit_exact'] is True, 'Completion flag differs')
        selected = read_json(run / 'selected_policy.json')
        require(selected['name'].endswith('_final'), 'Selected repeat name unexpected')
        first = run / 'trials' / (selected['name'][:-6] + '__full') / 'predictions.npz'
        final = run / 'trials' / (selected['name'] + '__full') / 'predictions.npz'
        with np.load(str(first), allow_pickle=False) as old, np.load(str(final), allow_pickle=False) as new:
            require(np.array_equal(old['logits'], new['logits']), 'Final-repeat logits differ')
        selected_result = read_json(final.parent / 'result.json')
        require(selected_result == completion['selected'], 'Completion selected result differs')
        require(selected_result['passes_extra_0_1pp'] and selected_result['passes_fp32_2pp'], 'Selected candidate misses accuracy target')
        first_policy = read_json(first.parent / 'policy.json')
        require({key: value for key, value in first_policy.items() if key not in ('name', 'lut_sha256')} ==
                {key: value for key, value in selected.items() if key not in ('name', 'lut_sha256')},
                'Selected repeat policy differs from original')
        require(len([r for r in trials if r['name'] == selected['name'] and len(r['full_stage_arrays_checked']) == 15]) == 1,
                'Selected full stage arrays absent')
        for filename, scope in [('screening_results.json', 'screen'), ('full_validation_results.json', 'full')]:
            summaries = read_json(run / filename)
            saved_names = {row['name'] for row in summaries}
            actual_names = {row['name'] for row in trials if row['scope'] == scope and not row['directory'].startswith('baseline/')}
            require(saved_names == actual_names and len(summaries) == len(actual_names), 'Summary trial set differs')
            for row in summaries:
                require(row == read_json(run / 'trials' / (row['name'] + '__' + scope) / 'result.json'), 'Summary result values differ')
        main_names = {row['name'] for row in trials if not row['zero_tail_guard'] and
                      row['name'] != 'all_dense_exact_zero_tail_control'}
        feasible = [row for row in read_json(run / 'full_validation_results.json')
                    if row['passes_extra_0_1pp'] and row['name'] in main_names]
        best = max(feasible, key=lambda row: (row['saved_partial_product_fraction_encoder'], row['correct']))
        close(selected_result['saved_partial_product_fraction_encoder'], best['saved_partial_product_fraction_encoder'],
              'Selected candidate is not best measured saving among feasible candidates')
        require(selected_result['correct'] == best['correct'], 'Selected candidate tie-break differs')
    else:
        require(allow_incomplete, 'Experiment is not complete; use --allow-incomplete for a partial read-only audit')
    guarded_completion = None
    guarded_present = any(row['zero_tail_guard'] for row in trials) or (run / 'guarded_source').exists()
    if (run / 'guarded_completion.json').exists():
        guarded_completion = read_json(run / 'guarded_completion.json')
        require(guarded_completion['completed'] is True and guarded_completion['full_repeat_logits_bit_exact'] is True,
                'Guarded completion flags differ')
        require(guarded_completion['testing_samples_used'] == 0 and guarded_completion['older_run_prediction_files_read'] is False,
                'Guarded provenance claim differs')
        require(guarded_completion['main_completion_sha256'] == sha256(run / 'completion.json'),
                'Main completion was changed by guarded phase')
        selected = read_json(run / 'guarded_selected_policy.json')
        require(selected['zero_tail_guard'] is True and selected['name'].endswith('_final'), 'Unexpected guarded final policy')
        first = run / 'trials' / (guarded_completion['selected_first_full'] + '__full')
        final = run / 'trials' / (selected['name'] + '__full')
        require(selected == read_json(final / 'policy.json'), 'Guarded selected policy differs from final trial policy')
        with np.load(str(first / 'predictions.npz'), allow_pickle=False) as old, np.load(str(final / 'predictions.npz'), allow_pickle=False) as new:
            require(np.array_equal(old['logits'], new['logits']), 'Guarded final repeat differs')
        selected_result = read_json(final / 'result.json')
        require(selected_result == guarded_completion['selected'], 'Guarded completion selected result differs')
        require(selected_result['passes_extra_0_1pp'] and selected_result['passes_fp32_2pp'], 'Guarded selected result misses target')
        require(len([row for row in trials if row['name'] == selected['name'] and len(row['full_stage_arrays_checked']) == 15]) == 1,
                'Guarded final stage arrays incomplete')
        guarded_names = {row['name'] for row in trials if row['zero_tail_guard']}
        feasible = [row for row in read_json(run / 'full_validation_results.json')
                    if row['passes_extra_0_1pp'] and row['name'] in guarded_names]
        best = max(feasible, key=lambda row: (row['saved_partial_product_fraction_encoder'], row['correct']))
        close(selected_result['saved_partial_product_fraction_encoder'], best['saved_partial_product_fraction_encoder'],
              'Guarded selection is not best measured saving among guarded feasible candidates')
        require(selected_result['correct'] == best['correct'], 'Guarded selection tie-break differs')
        comparisons = read_json(run / 'guarded_same_scope_comparison.json')
        for row in comparisons:
            candidate = read_json(run / 'trials' / (row['name'] + '__full') / 'result.json')
            candidate_policy = read_json(run / 'trials' / (row['name'] + '__full') / 'policy.json')
            require(row['selected_operations'] == len(candidate_policy['operations']), 'Same-scope operation count differs')
            close(row['encoder_saved_fraction'], candidate['saved_partial_product_fraction_encoder'], 'Same-scope candidate saving differs')
            if row.get('same_operation_scope'):
                control_dir = run / 'trials' / (row['exact_zero_tail_control'] + '__full')
                control, control_policy = read_json(control_dir / 'result.json'), read_json(control_dir / 'policy.json')
                require(set(control_policy['operations']) == set(candidate_policy['operations']), 'Same-scope control operation set differs')
                require(control['logits_bit_exact_vs_w8a8'], 'Same-scope exact control changed logits')
                close(row['control_encoder_saved_fraction'], control['saved_partial_product_fraction_encoder'], 'Same-scope control saving differs')
                close(row['additional_encoder_slot_saving_over_same_scope_control'],
                      candidate['saved_partial_product_fraction_encoder'] - control['saved_partial_product_fraction_encoder'],
                      'Same-scope incremental saving differs')
    elif guarded_present:
        require(allow_incomplete, 'Guarded supplement is not yet complete')
    overall_best = None
    if completion is not None:
        feasible = [row for row in read_json(run / 'full_validation_results.json')
                    if row['passes_extra_0_1pp'] and row['passes_fp32_2pp']]
        best = max(feasible, key=lambda row: (row['saved_partial_product_fraction_encoder'], row['correct']))
        overall_best = {key: best[key] for key in ('name', 'correct', 'accuracy',
                       'additional_drop_vs_w8a8_pp', 'saved_partial_product_fraction_encoder')}
    for array in base_arrays.values():
        array._mmap.close()
    return dict(passed=True, complete=completion is not None and (not guarded_present or guarded_completion is not None),
                main_phase_complete=completion is not None, guarded_phase_complete=guarded_completion is not None,
                run=str(run),
                verified_at_utc=datetime.now(timezone.utc).isoformat(),
                verifier_sha256=sha256(Path(__file__)), inference_executed=False, audio_files_opened=0,
                testing_samples_used=0, historical_parent_prediction_files_read=False,
                baseline_correct=dict(fp32=4340, w8a8=4338),
                trial_count=len(trials), trials=trials, training_histograms_checked=histogram_checks,
                training_policy_metrics_checked=training_metric_checks,
                guarded_vs_raw_bit_exact_checks=matching_guarded_raw_checks,
                exact_lut_screen_vs_full_batch_roundoff=exact_screen_roundoff,
                best_full_validation_candidate_among_all_phases=overall_best,
                source_and_native=provenance,
                limitations=['Counts are arithmetic term slots; no hardware timing/energy/area is established.',
                             'Recorded per-operation counts are checked algebraically; inference is not repeated by this verifier.',
                             'Screen-vs-full layer errors can include FP32 batch-slot roundoff; final full-order controls are bit-exact.',
                             'Legacy layer metric names fp32_max_abs/w16a16_max_abs mean W8A8-reference/candidate here.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--allow-incomplete', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.output and args.output.exists():
        raise FileExistsError('Refusing to overwrite existing audit: ' + str(args.output))
    result = audit(args.run, args.allow_incomplete)
    if args.output:
        with args.output.open('x', encoding='utf-8') as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2)
    print(json.dumps({key: value for key, value in result.items() if key not in ('trials', 'source_and_native')},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
