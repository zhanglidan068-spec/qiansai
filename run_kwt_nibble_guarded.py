"""Supplement a completed nibble run with fair zero-tail-guarded thresholds.

Only the identified run's freshly inferred baseline predictions are read.
The older parent provides MFCC inputs and training-calibrated formats, never
predictions.  Every new candidate performs fresh model inference.  Testing
audio and training/weight updates are never used.  The main completion and
selected-policy files are preserved; this supplement has separate records.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np

from nibble_early_exit import make_lut
from nibble_native import ApproxDenseEngine
from run_kwt_fp32 import make_flags, sha256, write_json
from run_kwt_int16 import ErrorAccumulator
from run_kwt_nibble_search import (ENCODER_MACS, ROOT, SOURCE_NAMES, lut_hash,
                                  policy, save_table, selected_macs, timestamp)


DEFAULT_RUN = ROOT / 'runs/nibble_w8a8_20261030_194829_241778'


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def guarded_policy(name, mode, threshold, stages, operations):
    return dict(policy(name, mode, threshold, stages, operations),
                zero_tail_guard=True, zero_tail_guard_mode='ratio',
                zero_tail_guard_threshold=65536.0, zero_tail_guard_stages='all')


def make_guarded_lut(mode='sigma', threshold=128, stages='ll_only'):
    """Preserve raw products exactly while allowing earlier exact-zero tails.

    All 65,025 signed-INT8 pairs are checked on every LUT construction.
    Changing exit stages cannot change the approximate arithmetic function.
    """
    raw = make_lut(mode, threshold, stages)
    zero = make_lut('ratio', 65536, 'all')
    exact = make_lut('exact')
    if not np.array_equal(zero['product_lut'], exact['product_lut']):
        raise AssertionError('The supposed zero-tail guard is not exact')
    earlier = zero['stage_lut'] < raw['stage_lut']
    if not np.array_equal(raw['product_lut'][earlier], zero['product_lut'][earlier]):
        raise AssertionError('Zero-tail guard would change a raw product')
    stage = np.minimum(raw['stage_lut'], zero['stage_lut']).astype(np.uint8)
    product = raw['product_lut'].copy()
    if not np.array_equal(product, raw['product_lut']):
        raise AssertionError('Raw product LUT changed')
    raw_hash = hashlib.sha256(raw['product_lut'].tobytes()).hexdigest()
    statistics = dict(raw['statistics'], zero_tail_guard=True,
        zero_tail_guard_threshold=65536, exhaustive_domain_pairs=255 * 255,
        guarded_product_bit_exact_to_raw=True, exact_zero_tail_control_verified=True,
        guard_earlier_pair_count=int(earlier.sum()), raw_product_lut_sha256=raw_hash,
        raw_exit_stage_histogram={str(i): int((raw['stage_lut'] == i).sum()) for i in range(1, 5)},
        guarded_exit_stage_histogram={str(i): int((stage == i).sum()) for i in range(1, 5)},
        uniform_domain_only_not_observed_model_savings=True)
    # Product/error statistics are unchanged, but operation statistics must
    # describe the guarded stages rather than the raw stopping rule.
    statistics['raw_statistics'] = raw['statistics']
    statistics['zero_term_optimization'] = True
    statistics['stage_counts'] = statistics['guarded_exit_stage_histogram']
    statistics['attempted_term_counts'] = {name: int((stage >= index + 1).sum())
        for index, name in enumerate(('HH', 'HL', 'LH', 'LL'))}
    statistics['executed_term_slots'] = int(stage.astype(np.int64).sum())
    statistics['mean_terms_per_pair'] = statistics['executed_term_slots'] / float(255 * 255)
    statistics['saved_term_fraction'] = 1. - statistics['executed_term_slots'] / float(255 * 255 * 4)
    return dict(product_lut=product, stage_lut=stage,
                raw_stage_lut=raw['stage_lut'], zero_tail_stage_lut=zero['stage_lut'],
                statistics=statistics, raw_product_lut_sha256=raw_hash)


def math_check():
    rows = []
    for mode, thresholds in [('sigma', [0, 64, 128, 256, 512, 16384]),
                             ('ratio', [1, 1.5, 2, 4, 8, 16, 65536]), ('exact', [0])]:
        for threshold in thresholds:
            for stages in ('ll_only', 'all'):
                lut = make_guarded_lut(mode, threshold, stages)
                rows.append(dict(mode=mode, threshold=threshold, stages=stages,
                    guard_earlier_pair_count=lut['statistics']['guard_earlier_pair_count'],
                    product_bit_exact_to_raw=True, lut_sha256=lut_hash(lut)))
    return dict(checked_policies=len(rows), pairs_per_policy=65025, passed=True, policies=rows,
                model_inference_performed=False, audio_files_opened=0, testing_samples_used=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=DEFAULT_RUN)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--third-sigma', type=int, choices=(64, 512),
                        help='Optional extra FFN LL-only sigma full; defaults to prioritizing conservative ratio8/16')
    parser.add_argument('--check-only', action='store_true', help='Pure LUT checks only; no run artifacts or TensorFlow')
    args = parser.parse_args()
    if args.check_only:
        print(json.dumps(math_check(), indent=2))
        return
    out = args.run.absolute()
    completion_path = out / 'completion.json'
    if not completion_path.is_file() or not read_json(completion_path).get('completed'):
        raise RuntimeError('Main experiment must finish before this supplement starts (completion.json gate)')
    if (out / 'guarded_completion.json').is_file():
        if read_json(out / 'guarded_completion.json').get('completed'):
            print('GUARDED_SUPPLEMENT_ALREADY_COMPLETE=' + str(out), flush=True)
            return
    lock_path = out / '.guarded_supplement.lock'
    # Exclusive creation avoids two supplement processes editing shared lists.
    descriptor = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    try:
        with os.fdopen(descriptor, 'w') as stream:
            stream.write(json.dumps(dict(pid=os.getpid(), created_time=timestamp())))
        run_supplement(out, args.threads, args.third_sigma)
    finally:
        lock_path.unlink()


def run_supplement(out, threads, third_sigma=None):
    completion_hash = sha256(out / 'completion.json')
    metadata = read_json(out / 'run_metadata.json')
    parent = Path(metadata['parent'])
    quantization = read_json(out / 'quantization.json')
    ranges = read_json(parent / 'calibration_ranges.json')
    manifest = read_json(out / 'sample_manifest.json')
    calibration_manifest = read_json(out / 'calibration_manifest.json')
    targets = np.asarray([row['label_id'] for row in manifest], np.int64)
    indices_screen = np.asarray(read_json(out / 'screening_indices.json'), np.int64)
    n, batch_size = len(targets), int(metadata['batch_size'])
    if n != 4445 or len(indices_screen) != 384 or len(calibration_manifest) != 256:
        raise ValueError('Unexpected sample count')
    if ranges['calibration_split'] != 'training' or metadata['testing_samples_used'] != 0:
        raise ValueError('Unexpected split provenance')
    if sha256(parent / 'calibration_ranges.json') != metadata['input_sha256'][str(parent / 'calibration_ranges.json')]:
        raise ValueError('Training-calibrated operand ranges changed')
    for filename in ('sample_manifest.json', 'calibration_manifest.json'):
        expected = metadata['input_sha256'][str(parent / filename)]
        if sha256(out / filename) != expected:
            raise ValueError('Same-run manifest differs from recorded source')
    for name, expected in metadata['checkpoint_sha256'].items():
        if sha256(ROOT / 'keyword-transformer/models_data_v2_12_labels/kwt1' / name) != expected:
            raise ValueError('Checkpoint changed')
    if sha256(parent / 'validation_mfcc.npy') != metadata['input_sha256'][str(parent / 'validation_mfcc.npy')]:
        raise ValueError('MFCC input cache changed')
    if sha256(out / 'quantization.json') != metadata['quantization_sha256']:
        raise ValueError('Quantization changed')
    features = np.load(str(parent / 'validation_mfcc.npy'), mmap_mode='r', allow_pickle=False)
    if features.shape != (4445, 98, 40):
        raise ValueError('Unexpected MFCC input shape')
    baselines = {}
    for name, expected in [('fp32', 4340), ('w8a8', 4338)]:
        with np.load(str(out / 'baseline' / name / 'predictions.npz'), allow_pickle=False) as saved:
            baselines[name] = saved['logits'].copy()
            if not np.array_equal(saved['target_ids'], targets):
                raise ValueError('Same-run baseline target order changed')
        if baselines[name].shape != (4445, 12) or int((baselines[name].argmax(1) == targets).sum()) != expected:
            raise ValueError('Same-run fresh baseline differs')
    all_indices = np.arange(n, dtype=np.int64)
    base_screen_correct = int((baselines['w8a8'][indices_screen].argmax(1) == targets[indices_screen]).sum())
    results_screen = read_json(out / 'screening_results.json')
    results_full = read_json(out / 'full_validation_results.json')
    main_screen = [r for r in results_screen if not r.get('zero_tail_guard')]
    main_full = [r for r in results_full if not r.get('zero_tail_guard')]
    initial_names_screen = {r['name'] for r in results_screen if not r['name'].startswith('guarded_')}
    initial_names_full = {r['name'] for r in results_full if not r['name'].startswith('guarded_')}
    fingerprints = {name: sha256(out / name) for name in ('screening_results.json', 'full_validation_results.json')}

    def update_collection(scope, result):
        name = 'full_validation_results.json' if scope == 'full' else 'screening_results.json'
        if sha256(out / name) != fingerprints[name]:
            raise RuntimeError('Concurrent shared-result edit detected: ' + name)
        if sha256(out / 'completion.json') != completion_hash:
            raise RuntimeError('Main completion changed while supplement was running')
        collection = results_full if scope == 'full' else results_screen
        collection[:] = [r for r in collection if r['name'] != result['name']]
        collection.append(result)
        write_json(out / name, collection)
        fingerprints[name] = sha256(out / name)
        save_table(out / name.replace('.json', '.csv'), collection)

    sources = out / 'guarded_source'
    sources.mkdir(exist_ok=True)
    source_names = list(SOURCE_NAMES) + ['run_kwt_nibble_guarded.py']
    source_hashes = {name: sha256(ROOT / name) for name in source_names}
    source_record = sources / 'source_sha256.json'
    if source_record.exists() and read_json(source_record) != source_hashes:
        raise RuntimeError('Guarded supplement source changed; retain the previous artifacts')
    for name in source_names:
        destination = sources / name
        if not destination.exists():
            shutil.copyfile(str(ROOT / name), str(destination))
        elif sha256(destination) != source_hashes[name]:
            raise RuntimeError('Guarded source snapshot changed')
    write_json(source_record, source_hashes)
    write_json(out / 'guarded_arithmetic_verification.json', math_check())
    supplement_metadata = dict(created_time=timestamp(), main_completion_sha256=completion_hash,
        same_run_fresh_baselines_reused=True, older_run_prediction_files_read=False,
        baseline_prediction_sha256={name: sha256(out / 'baseline' / name / 'predictions.npz') for name in baselines},
        input_cache=str(parent / 'validation_mfcc.npy'), feature_sha256=sha256(parent / 'validation_mfcc.npy'),
        calibration_split='training', validation_samples=n, testing_samples_used=0,
        audio_files_opened=0, training_or_weight_updates=False,
        preserves_main_completion_and_selected_policy=True,
        optional_third_sigma=third_sigma,
        slot_comparison='Compare guarded approximation against an exact-zero-tail control over the same selected operations',
        synthesized=False, hardware_speedup_measured=False)
    if not (out / 'guarded_metadata.json').exists():
        write_json(out / 'guarded_metadata.json', supplement_metadata)
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
    os.environ.setdefault('OMP_NUM_THREADS', str(threads))
    os.environ['TEMP'] = os.environ['TMP'] = str(ROOT / '.tmp')
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(ROOT / 'keyword-transformer'))
    import tensorflow as tf
    tf.compat.v1.disable_eager_execution()
    tf.compat.v1.disable_control_flow_v2()
    from kws_streaming.models import models as registry
    from kws_streaming.models import model_flags
    from kws_streaming.models.transformer_utils import KWSTransformer
    from kws_streaming.train import base_parser
    from kwt_nibble_graph import NibbleGraph
    model_dir = ROOT / 'keyword-transformer/models_data_v2_12_labels/kwt1'
    engine = ApproxDenseEngine(out / 'native', threads=threads)
    flags = make_flags(argparse.Namespace(repo=ROOT / 'keyword-transformer', data_dir=ROOT / 'data2',
        smoke=False, out=out, batch_size=batch_size), model_dir, base_parser, registry.kws_transformer, model_flags)
    config = tf.compat.v1.ConfigProto(device_count={'GPU': 0}, intra_op_parallelism_threads=8,
                                    inter_op_parallelism_threads=2)
    policies = {}
    with tf.compat.v1.Session(config=config) as sess:
        tf.compat.v1.keras.backend.set_session(sess)
        tf.keras.backend.set_learning_phase(0)
        model = registry.kws_transformer.model(flags)
        kwt = next(layer for layer in model.layers if isinstance(layer, KWSTransformer))
        graph = NibbleGraph(tf, model, kwt, ranges['matrix_operands'], quantization['nonlinear'], engine, batch_size)
        sess.run(tf.compat.v1.global_variables_initializer())
        status = model.load_weights(str(model_dir / 'best_weights'))
        status.assert_existing_objects_matched()
        status.expect_partial()
        all_dense = sorted(graph.native_enabled)
        ffn = [op for op in all_dense if op.endswith(('/fc1', '/fc2'))]
        assert len(all_dense) == 72 and len(ffn) == 24
        baseline_layers = {key: np.load(str(out / 'baseline/w8a8/layer_outputs' / (key + '.npy')),
                            mmap_mode='r', allow_pickle=False) for key in graph.outputs}

        def evaluate(p, scope='full', save_outputs=False):
            name = p['name']
            directory = out / 'trials' / (name + '__' + scope)
            directory.mkdir(parents=True, exist_ok=True)
            lut = make_guarded_lut(p['mode'], p['threshold'], p['stages']) if p.get('zero_tail_guard') else make_lut(p['mode'], p['threshold'], p['stages'])
            p = dict(p, lut_sha256=lut_hash(lut))
            if p.get('zero_tail_guard'):
                p['raw_product_lut_sha256'] = lut['raw_product_lut_sha256']
            policies[name] = p
            result_path = directory / 'result.json'
            if result_path.exists():
                if read_json(directory / 'policy.json') != p:
                    raise RuntimeError('Existing supplement policy differs: ' + name)
                result = read_json(result_path)
                update_collection(scope, result)
                print('GUARDED_RESUME_COMPLETED=' + name, flush=True)
                return result
            indices = all_indices if scope == 'full' else indices_screen
            write_json(directory / 'policy.json', p)
            write_json(directory / 'lut_statistics.json', lut['statistics'])
            np.save(str(directory / 'sample_indices.npy'), indices)
            np.savez_compressed(str(directory / 'arithmetic_lut.npz'),
                **{key: value for key, value in lut.items() if key.endswith('_lut') and isinstance(value, np.ndarray)})
            feed = graph.feed(quantization, lut['product_lut'], lut['stage_lut'], p['operations'])
            graph.collect_stats = True
            stage_counts = {op: np.zeros(4, np.int64) for op in p['operations']}
            errors = {key: ErrorAccumulator() for key in graph.outputs}
            arrays = {}
            if save_outputs:
                layer_dir = directory / 'layer_outputs'
                layer_dir.mkdir(exist_ok=True)
                for key, tensor in graph.outputs.items():
                    shape = (len(indices),) + tuple(tensor.shape.as_list()[1:])
                    arrays[key] = np.lib.format.open_memmap(str(layer_dir / (key + '.npy')), mode='w+', dtype=np.float32, shape=shape)
            started = time.perf_counter()
            predictions = []
            for offset in range(0, len(indices), batch_size):
                sample_ids = indices[offset:offset + batch_size]
                count = len(sample_ids)
                padded = np.empty((batch_size, 98, 40), np.float32)
                padded[:count] = np.asarray(features[sample_ids], np.float32)
                padded[count:] = padded[count - 1]
                graph.valid_samples = count
                feed[kwt.input] = padded
                graph.measured_counts.clear()
                graph.native_call_counts.clear()
                got = sess.run(graph.outputs, feed)
                logits = got['logits'][:count]
                if not np.isfinite(logits).all():
                    raise FloatingPointError('Nonfinite guarded logits')
                predictions.append(logits.copy())
                for key in errors:
                    errors[key].update(np, np.asarray(baseline_layers[key][sample_ids]), got[key][:count])
                    if key in arrays:
                        arrays[key][offset:offset + count] = got[key][:count]
                for op in p['operations']:
                    if graph.native_call_counts.get(op) != 1:
                        raise AssertionError('Expected one measured native callback: ' + op)
                    stage_counts[op] += graph.measured_counts[op]
                if scope == 'full' and (offset == 0 or (offset // batch_size + 1) % 20 == 0 or offset + count == len(indices)):
                    print('{} {}/{} ({:.1f}s)'.format(name, offset + count, len(indices), time.perf_counter() - started), flush=True)
            logits = np.concatenate(predictions)
            for array in arrays.values():
                array.flush()
                array._mmap.close()
            truth = targets[indices]
            current_prediction = logits.argmax(1)
            reference_prediction = baselines['w8a8'][indices].argmax(1)
            correct = int((current_prediction == truth).sum())
            fp_correct = int((baselines['fp32'][indices].argmax(1) == truth).sum())
            w8_correct = int((reference_prediction == truth).sum())
            selected = selected_macs(p['operations'])
            total_counts = sum(stage_counts.values(), np.zeros(4, np.int64))
            if int(total_counts.sum()) != selected * len(indices):
                raise AssertionError('Guarded callback counts include padding or omit operations')
            slots = int(np.dot(total_counts, np.arange(1, 5, dtype=np.int64)))
            saved_slots = selected * len(indices) * 4 - slots
            result = dict(name=name, scope=scope, samples=len(indices), correct=correct,
                accuracy=correct / float(len(indices)), fp32_correct=fp_correct, w8a8_correct=w8_correct,
                drop_vs_fp32_pp=100. * (fp_correct - correct) / len(indices),
                additional_drop_vs_w8a8_pp=100. * (w8_correct - correct) / len(indices),
                elapsed_seconds=time.perf_counter() - started, selected_macs_per_sample=selected,
                saved_partial_product_fraction_selected=saved_slots / float(selected * len(indices) * 4),
                saved_partial_product_fraction_encoder=saved_slots / float(ENCODER_MACS * len(indices) * 4),
                mean_terms_selected=slots / float(selected * len(indices)),
                exit_stage_counts=total_counts.tolist(), savings_measurement=scope + '_validation_candidate_operands',
                testing_samples_used=0, hardware_speedup_measured=False, mode=p['mode'], threshold=p['threshold'],
                stages=p['stages'], zero_tail_guard=bool(p.get('zero_tail_guard')),
                prediction_flips_vs_w8a8=int((reference_prediction != current_prediction).sum()),
                newly_wrong_vs_w8a8=int(((reference_prediction == truth) & (current_prediction != truth)).sum()),
                corrected_vs_w8a8=int(((reference_prediction != truth) & (current_prediction == truth)).sum()),
                logits_bit_exact_vs_w8a8=bool(np.array_equal(logits, baselines['w8a8'][indices])))
            result['passes_extra_0_1pp'] = result['additional_drop_vs_w8a8_pp'] <= .1 + 1e-12
            result['passes_fp32_2pp'] = result['drop_vs_fp32_pp'] <= 2. + 1e-12
            np.savez_compressed(str(directory / 'predictions.npz'), logits=logits, target_ids=truth)
            write_json(directory / 'operation_counts.json', dict(per_operation={op: count.tolist() for op, count in stage_counts.items()},
                total_exit_counts=total_counts.tolist(), selected_scalar_products=selected * len(indices),
                executed_partial_product_slots=slots, saved_partial_product_slots=saved_slots,
                full_encoder_four_term_slots=ENCODER_MACS * len(indices) * 4,
                excludes_padding=True, physical_cycle_measurement=False))
            layer_rows = [dict(stage=key, **acc.result()) for key, acc in errors.items()]
            write_json(directory / 'layer_comparison.json', layer_rows)
            save_table(directory / 'layer_comparison.csv', layer_rows)
            write_json(result_path, result)
            update_collection(scope, result)
            print('{}__{}: {}/{}; encoder_slots_saved={:.3f}%; extra_drop={:.6f}pp'.format(
                name, scope, correct, len(indices), 100 * result['saved_partial_product_fraction_encoder'],
                result['additional_drop_vs_w8a8_pp']), flush=True)
            return result

        # The completed main run already measured aggressive thresholds.  Use
        # more conservative ratios to search the strict 0.1 percentage-point limit.
        candidate_results = []
        for sigma in [128, 256] + ([third_sigma] if third_sigma is not None else []):
            candidate_results.append(evaluate(guarded_policy('guarded_ffn_sigma_{}_sll_only'.format(sigma),
                'sigma', sigma, 'll_only', ffn)))
        for ratio in (2, 8, 16):
            candidate_results.append(evaluate(guarded_policy('guarded_ffn_ratio_{}_sll_only'.format(ratio),
                'ratio', ratio, 'll_only', ffn)))

        # At most one promising all-stage raw-sigma policy gets a full guarded run.
        sigma_all = [r for r in main_full if r['mode'] == 'sigma' and r['stages'] == 'all'
                     and r['passes_extra_0_1pp'] and r['name'].startswith('ffn_')]
        optional_decisions = [dict(optional_third_sigma=third_sigma,
            performed=third_sigma is not None,
            reason=('Explicit --third-sigma request' if third_sigma is not None else
                    'Prioritize conservative ratio8/16; main aggressive sigma full already exceeds strict accuracy loss'),
            same_run_aggressive_sigma_results=[dict(name=r['name'], correct=r['correct'],
                additional_drop_vs_w8a8_pp=r['additional_drop_vs_w8a8_pp']) for r in main_full
                if r['name'] in ('ffn_sigma_64_sll_only', 'ffn_sigma_512_sll_only')])]
        if sigma_all:
            row = max(sigma_all, key=lambda r: (r['saved_partial_product_fraction_encoder'], r['correct']))
            original = read_json(out / 'trials' / (row['name'] + '__full') / 'policy.json')
            guarded = make_guarded_lut('sigma', row['threshold'], 'all')
            additional_slots = guarded['zero_tail_stage_lut'].astype(np.int64) - guarded['stage_lut'].astype(np.int64)
            estimate = None
            histogram_path = out / 'training_pair_histograms.npz'
            if histogram_path.exists():
                with np.load(str(histogram_path), allow_pickle=False) as saved:
                    saved_slots = sum(int((saved[op.replace('/', '__')] * additional_slots).sum())
                                      for op in original['operations'])
                estimate = saved_slots / float(256 * ENCODER_MACS * 4)
            worth_full = estimate > .0001 if estimate is not None else bool(np.any(additional_slots))
            optional_decisions.append(dict(raw_policy=row['name'], guarded_full_performed=worth_full,
                training_estimated_extra_encoder_slot_saving_over_zero_tail=estimate,
                minimum_extra_fraction=.0001, estimate_is_not_validation_measurement=True))
            if worth_full:
                candidate_results.append(evaluate(guarded_policy('guarded_' + row['name'], 'sigma', row['threshold'],
                    'all', original['operations'])))
        write_json(out / 'guarded_optional_decisions.json', optional_decisions)

        dense_screens = []
        for mode, threshold in [('sigma', 128), ('sigma', 256), ('ratio', 2), ('ratio', 8), ('ratio', 16)]:
            p = guarded_policy('guarded_all_dense_{}_{}_sll_only'.format(mode, threshold), mode, threshold, 'll_only', all_dense)
            dense_screens.append(evaluate(p, 'screen'))
        eligible = [r for r in dense_screens if r['correct'] >= base_screen_correct - 1]
        if eligible:
            # Compare expanded scope to its own exact zero-tail baseline.
            control = evaluate(policy('all_dense_exact_zero_tail_control', 'ratio', 65536, 'all', all_dense))
            if not control['logits_bit_exact_vs_w8a8']:
                raise AssertionError('All-Dense zero-tail control must match fresh W8A8 bit-for-bit')
            ordered = sorted(eligible, key=lambda r: (r['saved_partial_product_fraction_encoder'], r['correct']), reverse=True)
            full_plan = ordered[:1]
            remaining = ordered[1:]
            conservative = [r for r in remaining if r['mode'] == 'ratio' and r['threshold'] in (8, 16)]
            if conservative or remaining:
                # If the most economical screen fails the strict full target,
                # spend the second full run on a conservative ratio direction.
                full_plan.append((conservative or remaining)[0])
            for screened in full_plan:
                full = evaluate(policies[screened['name']])
                candidate_results.append(full)
                if full['passes_extra_0_1pp']:
                    break

        strict = [r for r in candidate_results if r['passes_extra_0_1pp']]
        if not strict:
            strict = [evaluate(guarded_policy('guarded_ffn_exact_zero_tail_fallback', 'exact', 0, 'all', ffn))]
        winner = max(strict, key=lambda r: (r['saved_partial_product_fraction_encoder'], r['correct']))
        final_policy = dict(policies[winner['name']], name=winner['name'] + '_final')
        final = evaluate(final_policy, save_outputs=True)
        with np.load(str(out / 'trials' / (winner['name'] + '__full') / 'predictions.npz'), allow_pickle=False) as saved:
            first_logits = saved['logits'].copy()
        with np.load(str(out / 'trials' / (final['name'] + '__full') / 'predictions.npz'), allow_pickle=False) as saved:
            final_logits = saved['logits'].copy()
        if not np.array_equal(first_logits, final_logits):
            raise AssertionError('Fresh guarded repeat is not bit exact')
        if not np.array_equal(np.load(str(out / 'trials' / (final['name'] + '__full') / 'layer_outputs/logits.npy'),
                                      allow_pickle=False), final_logits):
            raise AssertionError('Final stage logits disagree with fresh predictions')
        # Explicitly quantify gains over the same-scope lossless baseline.
        comparison_rows = []
        for row in candidate_results + [final]:
            original = policies[row['name']]
            dense_scope = set(original['operations']) == set(all_dense)
            control_name = 'all_dense_exact_zero_tail_control' if dense_scope else 'exact_zero_tail_control'
            control_path = out / 'trials' / (control_name + '__full') / 'result.json'
            comparison = dict(name=row['name'], exact_zero_tail_control=control_name,
                selected_operations=len(original['operations']),
                encoder_saved_fraction=row['saved_partial_product_fraction_encoder'])
            if control_path.exists():
                control = read_json(control_path)
                control_policy = read_json(control_path.parent / 'policy.json')
                if set(control_policy['operations']) == set(original['operations']):
                    comparison.update(control_encoder_saved_fraction=control['saved_partial_product_fraction_encoder'],
                        additional_encoder_slot_saving_over_same_scope_control=
                            row['saved_partial_product_fraction_encoder'] - control['saved_partial_product_fraction_encoder'],
                        same_operation_scope=True)
                else:
                    comparison.update(same_operation_scope=False, reason='Operation subset has no matched full lossless control')
            comparison_rows.append(comparison)
        write_json(out / 'guarded_same_scope_comparison.json', comparison_rows)
        save_table(out / 'guarded_same_scope_comparison.csv', comparison_rows)
        write_json(out / 'guarded_selected_policy.json', policies[final['name']])
        write_json(out / 'guarded_completion.json', dict(completed=True, selected=final,
            selected_first_full=winner['name'], full_repeat_logits_bit_exact=True,
            screening_trials_added=len([r for r in results_screen if r['name'] not in initial_names_screen]),
            full_trials_added=len([r for r in results_full if r['name'] not in initial_names_full]),
            main_completion_sha256=completion_hash, older_run_prediction_files_read=False,
            same_run_fresh_baselines_reused=True, testing_samples_used=0,
            area_compliance_verified=False, hardware_speedup_measured=False))
        print('GUARDED_SUPPLEMENT_COMPLETE=' + str(out), flush=True)
        for array in baseline_layers.values():
            array._mmap.close()
    features._mmap.close()


if __name__ == '__main__':
    main()
