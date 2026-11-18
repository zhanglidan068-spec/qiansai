"""Fresh W8A8 progressive four-bit product experiments, validation only.

Reuses acoustic feature inputs and training-calibrated numeric formats, never
historical predictions. Does not access audio, testing data, or update weights.
"""
import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np

from run_kwt_fp32 import make_flags, sha256, write_json
from run_kwt_int16 import ErrorAccumulator, dump_csv
from nibble_early_exit import make_lut
from nibble_native import ApproxDenseEngine


ROOT = Path(__file__).absolute().parent
ENCODER_MACS = 73446912
SOURCE_NAMES = ('run_kwt_nibble_search.py', 'kwt_nibble_graph.py', 'kwt_mixed_graph.py',
    'nibble_early_exit.py', 'verify_nibble_early_exit.py', 'nibble_native.py', 'nibble_matmul.cpp',
    'kwt_nonlinear_int16.py', 'compact_gelu_int16.py', 'compact_softmax_int16.py',
    'kwt_int8_n16_graph.py', 'run_kwt_fp32.py', 'run_kwt_int16.py')


def timestamp():
    return datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S_%f')


def policy(name, mode='exact', threshold=0, stages='all', operations=()):
    return dict(name=name, mode=mode, threshold=float(threshold), stages=stages,
                operations=sorted(operations), magnitude_split=True, comparison='>=')


def save_table(path, rows):
    if not rows:
        return
    fields = sorted(set().union(*(r.keys() for r in rows)))
    flat = [{k: (json.dumps(r.get(k), ensure_ascii=False) if isinstance(r.get(k), (dict, list))
                 else r.get(k)) for k in fields} for r in rows]
    dump_csv(path, flat)


def selected_macs(operations):
    return sum(99 * (64 * 256 if op.endswith(('/fc1', '/fc2')) else 64 * 64)
               for op in operations)


def lut_hash(lut):
    return hashlib.sha256(lut['product_lut'].tobytes() + lut['stage_lut'].tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, default=ROOT / 'runs/precision_search_20261028_162338_802026')
    parser.add_argument('--out', type=Path)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--max-full', type=int, default=10)
    parser.add_argument('--native-library', type=Path)
    args = parser.parse_args()
    if args.out and args.resume:
        parser.error('--out and --resume are mutually exclusive')
    out = (args.resume or args.out or ROOT / 'runs' / ('nibble_w8a8_' + timestamp())).absolute()
    if args.resume:
        if not (out / 'run_metadata.json').is_file():
            raise ValueError('Only an explicitly identified nibble experiment can be resumed')
    else:
        out.mkdir(parents=True, exist_ok=False)
    print('NIBBLE_EXPERIMENT_DIRECTORY=' + str(out), flush=True)
    parent = args.parent.absolute()
    prep = json.loads((parent / 'preparation_complete.json').read_text(encoding='utf-8'))
    manifest = json.loads((parent / 'sample_manifest.json').read_text(encoding='utf-8'))
    calibration_manifest = json.loads((parent / 'calibration_manifest.json').read_text(encoding='utf-8'))
    ranges = json.loads((parent / 'calibration_ranges.json').read_text(encoding='utf-8'))
    quantization_path = parent / 'trials/encoder_w8a8_p8f7__full/quantization.json'
    quantization = json.loads(quantization_path.read_text(encoding='utf-8'))
    indices_screen = np.asarray(json.loads((parent / 'screening_indices.json').read_text(encoding='utf-8')), np.int64)
    targets = np.asarray([row['label_id'] for row in manifest], np.int64)
    features = np.load(str(parent / 'validation_mfcc.npy'), mmap_mode='r', allow_pickle=False)
    training_features = np.load(str(parent / 'calibration_mfcc.npy'), mmap_mode='r', allow_pickle=False)
    n, batch_size = len(targets), 32
    if features.shape != (4445, 98, 40) or training_features.shape != (256, 98, 40):
        raise ValueError('Unexpected feature cache shape')
    if n != 4445 or len(calibration_manifest) != 256 or ranges['calibration_split'] != 'training':
        raise ValueError('Unexpected data split metadata')
    if sha256(parent / 'validation_mfcc.npy') != prep['feature_sha256']:
        raise ValueError('Validation acoustic cache hash mismatch')
    if np.any(indices_screen < 0) or np.any(indices_screen >= n) or len(np.unique(indices_screen)) != len(indices_screen):
        raise ValueError('Invalid validation screening indices')
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
    os.environ.setdefault('OMP_NUM_THREADS', str(args.threads))
    os.environ.setdefault('MKL_NUM_THREADS', str(args.threads))
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
    from kws_streaming.data.input_data import which_set
    from kwt_nibble_graph import NibbleGraph
    from verify_nibble_early_exit import verify
    for rows, split in ((manifest, 'validation'), (calibration_manifest, 'training')):
        for row in rows:
            if not row['synthetic_zero_waveform'] and which_set(row['file'], 10, 10) != split:
                raise ValueError('Wrong split in manifest: ' + split)
    exact_lut = make_lut()
    write_json(out / 'nibble_arithmetic_verification.json', verify())
    sources = out / 'source'
    sources.mkdir(exist_ok=True)
    source_hashes = {name: sha256(ROOT / name) for name in SOURCE_NAMES}
    source_record = out / 'source_sha256.json'
    if source_record.is_file() and json.loads(source_record.read_text(encoding='utf-8')) != source_hashes:
        version = out / 'source_versions' / timestamp()
        version.mkdir(parents=True)
        for name in SOURCE_NAMES:
            shutil.copyfile(ROOT / name, version / name)
        write_json(version / 'source_sha256.json', source_hashes)
        print('SOURCE_VERSION_RECORD=' + str(version), flush=True)
    elif not source_record.exists():
        for name in SOURCE_NAMES:
            shutil.copyfile(ROOT / name, sources / name)
        write_json(source_record, source_hashes)
    for name in ('sample_manifest.json', 'calibration_manifest.json', 'screening_indices.json'):
        destination = out / name
        if not destination.exists():
            shutil.copyfile(parent / name, destination)
    write_json(out / 'quantization.json', quantization)
    model_dir = ROOT / 'keyword-transformer/models_data_v2_12_labels/kwt1'
    metadata = dict(purpose='Progressive four-bit multiplication on frozen W8A8/N16/P8F7',
        created_time=timestamp(), parent=str(parent), batch_size=batch_size,
        calibration_samples=256, validation_samples=n, screening_samples=len(indices_screen),
        testing_samples_used=0, audio_files_opened=0, training_or_weight_updates=False,
        historical_prediction_files_read=False, input_reuse='MFCC acoustic inputs only; logits computed afresh',
        fp32_reference_expected_correct=4340, w8a8_reference_expected_correct=4338,
        target_extra_drop_vs_w8a8_pp=.1, original_limit_drop_vs_fp32_pp=2.,
        encoder_macs_per_sample=ENCODER_MACS, native_threads=args.threads,
        tensorflow=tf.__version__, numpy=np.__version__,
        input_sha256={str(parent / name): sha256(parent / name) for name in
            ('validation_mfcc.npy', 'calibration_mfcc.npy', 'calibration_ranges.json', 'sample_manifest.json', 'calibration_manifest.json')},
        quantization_sha256=sha256(quantization_path),
        checkpoint_sha256={p.name: sha256(p) for p in model_dir.glob('best_weights.*')},
        synthesized=False, hardware_speedup_measured=False, area_compliance_verified=False)
    if not (out / 'run_metadata.json').exists():
        write_json(out / 'run_metadata.json', metadata)
    else:
        previous = json.loads((out / 'run_metadata.json').read_text(encoding='utf-8'))
        if (previous['input_sha256'] != metadata['input_sha256'] or
                previous['quantization_sha256'] != metadata['quantization_sha256'] or
                previous['checkpoint_sha256'] != metadata['checkpoint_sha256']):
            raise ValueError('Resume inputs/configuration changed')
    library = args.native_library
    if library is None and (ROOT / '.tmp/native_nibble_dev2/nibble_matmul.dll').is_file():
        library = ROOT / '.tmp/native_nibble_dev2/nibble_matmul.dll'
    engine = ApproxDenseEngine(out / 'native', threads=args.threads, library_path=library)
    flags_args = argparse.Namespace(repo=ROOT / 'keyword-transformer', data_dir=ROOT / 'data2',
        smoke=False, out=out, batch_size=batch_size)
    flags = make_flags(flags_args, model_dir, base_parser, registry.kws_transformer, model_flags)
    all_indices = np.arange(n, dtype=np.int64)
    results_screen, results_full = [], []
    if (out / 'screening_results.json').exists():
        results_screen = json.loads((out / 'screening_results.json').read_text(encoding='utf-8'))
    if (out / 'full_validation_results.json').exists():
        results_full = json.loads((out / 'full_validation_results.json').read_text(encoding='utf-8'))
    policies = {}
    config = tf.compat.v1.ConfigProto(device_count={'GPU': 0}, intra_op_parallelism_threads=8,
                                      inter_op_parallelism_threads=2)
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
        assert selected_macs(ffn) == 38928384
        baseline_layers = None
        logits_fp32 = logits_w8 = None

        def batch(data, indices):
            found = np.asarray(data[indices], np.float32)
            padded = np.empty((batch_size, 98, 40), np.float32)
            padded[:len(found)] = found
            padded[len(found):] = found[-1]
            graph.valid_samples = len(found)
            return padded

        def evaluate(p, scope='screen', fp32=False, baseline=False, save_outputs=False, control_indices=None):
            nonlocal logits_fp32, logits_w8, baseline_layers
            policies[p['name']] = p
            directory = out / 'baseline' / ('fp32' if fp32 else 'w8a8') if baseline else out / 'trials' / (p['name'] + '__' + scope)
            directory.mkdir(parents=True, exist_ok=True)
            lut = make_lut(p['mode'], p['threshold'], p['stages'])
            p = dict(p, lut_sha256=lut_hash(lut))
            result_path = directory / 'result.json'
            if result_path.is_file():
                old_policy = json.loads((directory / 'policy.json').read_text(encoding='utf-8'))
                if old_policy != p:
                    raise ValueError('Resume policy differs: ' + p['name'])
                print('RESUME_COMPLETED=' + p['name'] + '__' + scope, flush=True)
                return json.loads(result_path.read_text(encoding='utf-8'))
            indices = control_indices if control_indices is not None else (all_indices if scope == 'full' else indices_screen)
            write_json(directory / 'policy.json', p)
            write_json(directory / 'lut_statistics.json', lut['statistics'])
            np.save(str(directory / 'sample_indices.npy'), indices)
            np.savez_compressed(str(directory / 'arithmetic_lut.npz'), product_lut=lut['product_lut'], stage_lut=lut['stage_lut'])
            feed = graph.feed(quantization, lut['product_lut'], lut['stage_lut'], p['operations'], fp32=fp32)
            graph.collect_stats = True
            stage_counts = {op: np.zeros(4, np.int64) for op in p['operations']}
            error_acc = {key: ErrorAccumulator() for key in graph.outputs} if baseline_layers is not None and not baseline else {}
            layer_arrays = {}
            if save_outputs:
                layer_dir = directory / 'layer_outputs'
                layer_dir.mkdir(exist_ok=True)
                for key, tensor in graph.outputs.items():
                    shape = (len(indices),) + tuple(tensor.shape.as_list()[1:])
                    layer_arrays[key] = np.lib.format.open_memmap(str(layer_dir / (key + '.npy')), mode='w+', dtype=np.float32, shape=shape)
            started = time.perf_counter()
            predictions = []
            for offset in range(0, len(indices), batch_size):
                sample_ids = indices[offset:offset + batch_size]
                feed[kwt.input] = batch(features, sample_ids)
                requests = {'outputs': graph.outputs if save_outputs or error_acc else {'logits': graph.outputs['logits']}}
                # TF2.4 control-flow-v1 inner-branch tensors cannot be fetched
                # directly. Collect the native function's returned counts at
                # execution instead; verify exactly one call per active Dense.
                graph.measured_counts.clear()
                graph.native_call_counts.clear()
                got = sess.run(requests, feed)
                count = len(sample_ids)
                logits = got['outputs']['logits'][:count]
                if not np.isfinite(logits).all():
                    raise FloatingPointError('Nonfinite logits: ' + p['name'])
                predictions.append(logits.copy())
                for key, array in layer_arrays.items():
                    array[offset:offset + count] = got['outputs'][key][:count]
                for key, accumulator in error_acc.items():
                    accumulator.update(np, np.asarray(baseline_layers[key][sample_ids]), got['outputs'][key][:count])
                for op in p['operations']:
                    if graph.native_call_counts.get(op) != 1:
                        raise AssertionError('Expected one native call per selected Dense: ' + op)
                    stage_counts[op] += graph.measured_counts[op]
                if scope == 'full' and (offset == 0 or (offset // batch_size + 1) % 20 == 0 or offset + count == len(indices)):
                    print('{} {}/{} ({:.1f}s)'.format(p['name'], offset + count, len(indices), time.perf_counter() - started), flush=True)
            logits = np.concatenate(predictions)
            for array in layer_arrays.values():
                array.flush()
                array._mmap.close()
            correct = int((logits.argmax(1) == targets[indices]).sum())
            fp_correct = correct if fp32 else int((logits_fp32[indices].argmax(1) == targets[indices]).sum())
            w8_correct = correct if baseline and not fp32 else (int((logits_w8[indices].argmax(1) == targets[indices]).sum()) if logits_w8 is not None else None)
            selected = selected_macs(p['operations'])
            total_counts = sum(stage_counts.values(), np.zeros(4, np.int64))
            if int(total_counts.sum()) != selected * len(indices):
                raise AssertionError('Count mismatch (padding or skipped operations): ' + p['name'])
            slots = int(np.dot(total_counts, np.arange(1, 5, dtype=np.int64)))
            saved_slots = selected * len(indices) * 4 - slots
            result = dict(name=p['name'], scope=scope, samples=len(indices), correct=correct,
                accuracy=correct / float(len(indices)), fp32_correct=fp_correct, w8a8_correct=w8_correct,
                drop_vs_fp32_pp=100. * (fp_correct - correct) / len(indices),
                additional_drop_vs_w8a8_pp=(100. * (w8_correct - correct) / len(indices) if w8_correct is not None else None),
                elapsed_seconds=time.perf_counter() - started, selected_macs_per_sample=selected,
                saved_partial_product_fraction_selected=(saved_slots / float(selected * len(indices) * 4) if selected else 0.),
                saved_partial_product_fraction_encoder=saved_slots / float(ENCODER_MACS * len(indices) * 4),
                mean_terms_selected=(slots / float(selected * len(indices)) if selected else 4.),
                exit_stage_counts=total_counts.tolist(), savings_measurement=scope + '_validation_candidate_operands',
                testing_samples_used=0, hardware_speedup_measured=False, mode=p['mode'], threshold=p['threshold'], stages=p['stages'])
            result['passes_extra_0_1pp'] = result['additional_drop_vs_w8a8_pp'] is not None and result['additional_drop_vs_w8a8_pp'] <= .1 + 1e-12
            result['passes_fp32_2pp'] = result['drop_vs_fp32_pp'] <= 2. + 1e-12
            if logits_w8 is not None:
                reference_prediction = logits_w8[indices].argmax(1)
                current_prediction = logits.argmax(1)
                truth = targets[indices]
                result.update(prediction_flips_vs_w8a8=int((reference_prediction != current_prediction).sum()),
                    newly_wrong_vs_w8a8=int(((reference_prediction == truth) & (current_prediction != truth)).sum()),
                    corrected_vs_w8a8=int(((reference_prediction != truth) & (current_prediction == truth)).sum()),
                    logits_bit_exact_vs_w8a8=bool(np.array_equal(logits, logits_w8[indices])))
            np.savez_compressed(str(directory / 'predictions.npz'), logits=logits, target_ids=targets[indices])
            write_json(directory / 'operation_counts.json', dict(per_operation={op: counts.tolist() for op, counts in stage_counts.items()},
                total_exit_counts=total_counts.tolist(), selected_scalar_products=selected * len(indices),
                executed_partial_product_slots=slots, saved_partial_product_slots=saved_slots,
                full_encoder_four_term_slots=ENCODER_MACS * len(indices) * 4,
                excludes_padding=True, physical_cycle_measurement=False))
            if error_acc:
                layer_rows = [dict(stage=key, **acc.result()) for key, acc in error_acc.items()]
                write_json(directory / 'layer_comparison.json', layer_rows)
                save_table(directory / 'layer_comparison.csv', layer_rows)
            write_json(result_path, result)
            if not baseline and scope in ('screen', 'full'):
                collection = results_full if scope == 'full' else results_screen
                collection[:] = [r for r in collection if r['name'] != p['name']]
                collection.append(result)
                write_json(out / ('full_validation_results.json' if scope == 'full' else 'screening_results.json'), collection)
                save_table(out / ('full_validation_results.csv' if scope == 'full' else 'screening_results.csv'), collection)
            print('{}__{}: {}/{}={:.5f}%; extra_drop={}pp; encoder_slots_saved={:.3f}% ({:.1f}s)'.format(
                p['name'], scope, correct, len(indices), result['accuracy'] * 100,
                result['additional_drop_vs_w8a8_pp'], result['saved_partial_product_fraction_encoder'] * 100,
                result['elapsed_seconds']), flush=True)
            return result

        # Independent baselines are freshly inferred before any approximation.
        evaluate(policy('fp32_fresh'), 'full', fp32=True, baseline=True)
        with np.load(str(out / 'baseline/fp32/predictions.npz'), allow_pickle=False) as saved:
            logits_fp32 = saved['logits'].copy()
        if int((logits_fp32.argmax(1) == targets).sum()) != 4340:
            raise AssertionError('Fresh FP32 accuracy differs from established reference')
        evaluate(policy('w8a8_exact_tf'), 'full', baseline=True, save_outputs=True)
        with np.load(str(out / 'baseline/w8a8/predictions.npz'), allow_pickle=False) as saved:
            logits_w8 = saved['logits'].copy()
        if int((logits_w8.argmax(1) == targets).sum()) != 4338:
            raise AssertionError('Fresh W8A8 accuracy differs from established reference')
        baseline_layers = {key: np.load(str(out / 'baseline/w8a8/layer_outputs' / (key + '.npy')), mmap_mode='r', allow_pickle=False)
                           for key in graph.outputs}
        native_control = evaluate(policy('native_four_term_exact_control', operations=all_dense),
                                  'control', control_indices=np.arange(64, dtype=np.int64))
        if not native_control['logits_bit_exact_vs_w8a8']:
            raise AssertionError('Full four-term native graph must match exact TF bit-for-bit')
        control_layers = json.loads((out / 'trials/native_four_term_exact_control__control/layer_comparison.json').read_text(encoding='utf-8'))
        if any(row['max_abs_error'] != 0 for row in control_layers):
            raise AssertionError('Native control intermediate stages differ')

        # Train-only operand histograms characterize ranges and expected savings.
        hist_path = out / 'training_pair_histograms.npz'
        if not hist_path.exists():
            print('Collecting TRAIN-only W8A8 operand distributions (256 samples)', flush=True)
            hist_a, hist_w = {}, {}
            feed = graph.feed(quantization, exact_lut['product_lut'], exact_lut['stage_lut'])
            graph.collect_stats = False
            fetch = {op + '/a': graph.records['q_operands'][op + '/a'] for op in all_dense}
            fetch.update({op + '/w': graph.records['q_operands'][op + '/w'] for op in all_dense})
            for offset in range(0, len(training_features), batch_size):
                sample_ids = np.arange(offset, min(offset + batch_size, len(training_features)))
                feed[kwt.input] = batch(training_features, sample_ids)
                got = sess.run(fetch, feed)
                for op in all_dense:
                    a_codes = got[op + '/a'][:len(sample_ids)].reshape(-1, got[op + '/a'].shape[-1]).astype(np.int32)
                    w_codes = got[op + '/w'].astype(np.int32)
                    if op not in hist_a:
                        hist_a[op] = np.zeros((a_codes.shape[1], 255), np.int64)
                        hist_w[op] = np.stack([np.bincount(row + 127, minlength=255) for row in w_codes])
                    for k in range(a_codes.shape[1]):
                        hist_a[op][k] += np.bincount(a_codes[:, k] + 127, minlength=255)
                print('Training operands {}/256'.format(offset + len(sample_ids)), flush=True)
            pair_hist = {}
            for op in all_dense:
                cross = np.dot(hist_a[op].T.astype(np.float64), hist_w[op].astype(np.float64))
                rounded = np.rint(cross).astype(np.int64)
                if not np.allclose(cross, rounded, rtol=0, atol=1e-5) or int(rounded.sum()) != selected_macs([op]) * 256:
                    raise AssertionError('Operand histogram accounting mismatch')
                pair_hist[op.replace('/', '__')] = rounded
            np.savez_compressed(str(hist_path), **pair_hist)
        with np.load(str(hist_path), allow_pickle=False) as saved:
            histograms = {op: saved[op.replace('/', '__')].copy() for op in all_dense}

        training_metrics = []
        def estimate(p):
            lut = make_lut(p['mode'], p['threshold'], p['stages'])
            histogram = sum((histograms[op] for op in p['operations']), np.zeros((255, 255), np.int64))
            products = int(histogram.sum())
            slots = int((histogram * lut['stage_lut']).sum())
            row = dict(name=p['name'], mode=p['mode'], threshold=p['threshold'], stages=p['stages'],
                selected_macs_per_sample=selected_macs(p['operations']),
                training_scalar_products=products, training_mean_terms=slots / float(products) if products else 4.,
                training_saved_fraction_selected=1. - slots / float(products * 4) if products else 0.,
                training_saved_fraction_encoder=(products * 4 - slots) / float(256 * ENCODER_MACS * 4))
            training_metrics.append(row)
            return row

        candidates = [policy('exact_zero_tail_control', 'ratio', 65536, 'all', ffn)]
        for stages, sigmas in (('ll_only', (0, 64, 128, 256, 512, 1024, 2048, 4096, 8192)),
                               ('all', (256, 1024, 4096, 8192, 12288, 16384))):
            candidates += [policy('ffn_sigma_{}_s{}'.format(value, stages), 'sigma', value, stages, ffn) for value in sigmas]
        for stages in ('ll_only', 'all'):
            candidates += [policy('ffn_ratio_{}_s{}'.format(value, stages), 'ratio', value, stages, ffn)
                           for value in (2, 4, 8, 16, 32, 64)]
        screened = []
        for p in candidates:
            estimate(p)
            screened.append(evaluate(p))
        save_table(out / 'training_policy_metrics.csv', training_metrics)
        write_json(out / 'training_policy_metrics.json', training_metrics)
        base_screen_correct = int((logits_w8[indices_screen].argmax(1) == targets[indices_screen]).sum())
        shortlist = ['exact_zero_tail_control']
        for mode in ('sigma', 'ratio'):
            for stages in ('ll_only', 'all'):
                eligible = [r for r in screened if r['name'] != 'exact_zero_tail_control' and
                            r['mode'] == mode and r['stages'] == stages and r['correct'] >= base_screen_correct - 1]
                if eligible:
                    shortlist.append(max(eligible, key=lambda r: (r['saved_partial_product_fraction_encoder'], r['correct']))['name'])
        shortlist += ['ffn_ratio_32_sall', 'ffn_ratio_64_sall']
        shortlist = list(dict.fromkeys(shortlist))
        for name in shortlist:
            if len(results_full) >= args.max_full:
                break
            row = evaluate(policies[name], 'full')
            if name == 'exact_zero_tail_control' and not row['logits_bit_exact_vs_w8a8']:
                raise AssertionError('Zero-tail control must remain bit exact')

        # Isolate each block, then test combined masks at one promising threshold.
        approximate = [r for r in screened if r['name'] != 'exact_zero_tail_control' and
                       r['correct'] >= base_screen_correct - 4]
        if approximate:
            aggressive = max(approximate, key=lambda r: r['saved_partial_product_fraction_encoder'])
            original = policies[aggressive['name']]
            single_results = []
            for number in range(1, 13):
                prefix = 'layer_{:02d}/'.format(number)
                ops = [op for op in ffn if op.startswith(prefix)]
                p = policy('single_layer_{:02d}_{}'.format(number, original['name']), original['mode'],
                           original['threshold'], original['stages'], ops)
                row = evaluate(p)
                single_results.append(dict(layer=number, **row))
            write_json(out / 'layer_sensitivity.json', single_results)
            save_table(out / 'layer_sensitivity.csv', single_results)
            eligible_layers = [r['layer'] for r in single_results if r['correct'] >= base_screen_correct]
            combined_ops = [op for op in ffn if int(op.split('/')[0].split('_')[1]) in eligible_layers]
            if combined_ops:
                p = policy('masked_ffn_' + original['name'], original['mode'], original['threshold'], original['stages'], combined_ops)
                row = evaluate(p)
                if row['correct'] >= base_screen_correct - 1 and len(results_full) < args.max_full:
                    evaluate(p, 'full')

        # One broader Dense-only trial; QK/AV and all nonlinear formats stay fixed.
        strict = [r for r in results_full if r['passes_extra_0_1pp'] and r['name'] != 'exact_zero_tail_control']
        if strict:
            best = max(strict, key=lambda r: r['saved_partial_product_fraction_encoder'])
            original = policies[best['name']]
            p = policy('all_dense_' + original['name'], original['mode'], original['threshold'], original['stages'], all_dense)
            row = evaluate(p)
            if row['correct'] >= base_screen_correct - 1 and len(results_full) < args.max_full:
                evaluate(p, 'full')

        strict = [r for r in results_full if r['passes_extra_0_1pp']]
        if not strict:
            raise RuntimeError('No full candidate satisfies the strict accuracy target')
        winner = max(strict, key=lambda r: (r['saved_partial_product_fraction_encoder'], r['correct']))
        winner_policy = policies[winner['name']]
        # Fresh repeat with full arrays for the selected result, not cached predictions.
        final_policy = dict(winner_policy, name=winner_policy['name'] + '_final')
        final_result = evaluate(final_policy, 'full', save_outputs=True)
        with np.load(str(out / 'trials' / (winner['name'] + '__full') / 'predictions.npz'), allow_pickle=False) as saved:
            first_logits = saved['logits'].copy()
        with np.load(str(out / 'trials' / (final_policy['name'] + '__full') / 'predictions.npz'), allow_pickle=False) as saved:
            final_logits = saved['logits'].copy()
        if not np.array_equal(first_logits, final_logits):
            raise AssertionError('Selected configuration repeat is not bit exact')
        write_json(out / 'selected_policy.json', final_policy)
        write_json(out / 'completion.json', dict(completed=True, selected=final_result,
            screening_trials=len(results_screen), full_trials=len(results_full),
            full_repeat_logits_bit_exact=True, historical_prediction_files_read=False,
            testing_samples_used=0, area_compliance_verified=False, hardware_speedup_measured=False))
        print('NIBBLE_EXPERIMENT_COMPLETE=' + str(out), flush=True)
    features._mmap.close()
    training_features._mmap.close()


if __name__ == '__main__':
    main()
