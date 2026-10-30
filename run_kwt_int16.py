"""Paired KWT W16A16 integer-matmul / FP32 nonlinear validation experiment.

Every invocation creates a new directory. No training or checkpoint mutation.
Run with the existing .conda Python 3.8 / TensorFlow 2.4 environment.
"""
import argparse
import csv
from datetime import datetime, timedelta, timezone
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import time

from run_kwt_fp32 import build_probe, export_weights, git_revision, make_flags, sha256, write_json


def arguments():
    root = Path(__file__).absolute().parent
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, default=root / 'keyword-transformer')
    p.add_argument('--data-dir', type=Path, default=root / 'data2')
    p.add_argument('--baseline-run', type=Path, default=root / 'runs/validation_full')
    p.add_argument('--out', type=Path)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--calibration-samples', type=int, default=256)
    p.add_argument('--activation-headroom', type=float, default=1.25)
    p.add_argument('--limit', type=int, default=0, help='0: full original validation manifest')
    a = p.parse_args()
    if a.batch_size < 1 or a.calibration_samples < 1 or a.limit < 0 or a.activation_headroom < 1:
        p.error('Invalid batch size, calibration count, limit, or headroom')
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S_%f')
    a.out = (a.out or root / 'runs' / ('w16a16_fpnl_' + stamp)).absolute()
    a.repo, a.data_dir, a.baseline_run = a.repo.absolute(), a.data_dir.absolute(), a.baseline_run.absolute()
    a.smoke = False
    return a


def fingerprint(directory):
    return {str(p.relative_to(directory)): sha256(p) for p in sorted(directory.rglob('*')) if p.is_file()}


def choose_fractional_bits(max_abs, headroom):
    if not math.isfinite(max_abs):
        raise FloatingPointError('Non-finite calibration range')
    # Explicit zero convention: scale=1. Actual nonzero ranges use power-of-two scales.
    return 0 if max_abs == 0 else int(math.floor(math.log2(32767.0 / (max_abs * headroom))))


class ErrorAccumulator:
    def __init__(self):
        self.n = 0
        self.abs_sum = self.err2 = self.ref2 = self.quant2 = self.dot = 0.0
        self.max_abs = self.ref_max_abs = self.quant_max_abs = 0.0

    def update(self, np, ref, quant):
        r, q = ref.astype(np.float64), quant.astype(np.float64)
        error = q - r
        self.n += r.size
        self.abs_sum += float(np.abs(error).sum())
        self.err2 += float(np.square(error).sum())
        self.ref2 += float(np.square(r).sum())
        self.quant2 += float(np.square(q).sum())
        self.dot += float((r * q).sum())
        self.max_abs = max(self.max_abs, float(np.abs(error).max()))
        self.ref_max_abs = max(self.ref_max_abs, float(np.abs(r).max()))
        self.quant_max_abs = max(self.quant_max_abs, float(np.abs(q).max()))
        axes = tuple(range(1, r.ndim))
        e2, r2, q2 = np.square(error).sum(axis=axes), np.square(r).sum(axis=axes), np.square(q).sum(axis=axes)
        denominator = np.sqrt(r2 * q2)
        cosine = np.divide((r * q).sum(axis=axes), denominator,
                           out=np.full(r.shape[0], np.nan), where=denominator != 0)
        rel = np.divide(np.sqrt(e2), np.sqrt(r2), out=np.full(r.shape[0], np.nan), where=r2 != 0)
        return np.stack([np.abs(error).mean(axis=axes), np.sqrt(np.square(error).mean(axis=axes)),
                         np.abs(error).max(axis=axes), rel, cosine], axis=1)

    def result(self):
        return dict(elements=self.n, mae=self.abs_sum / self.n, rmse=math.sqrt(self.err2 / self.n),
                    max_abs_error=self.max_abs,
                    relative_l2=math.sqrt(self.err2 / self.ref2) if self.ref2 else None,
                    cosine=self.dot / math.sqrt(self.ref2 * self.quant2) if self.ref2 * self.quant2 else None,
                    sqnr_db=10 * math.log10(self.ref2 / self.err2) if self.ref2 and self.err2 else None,
                    fp32_max_abs=self.ref_max_abs, w16a16_max_abs=self.quant_max_abs)


def dump_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def numerical_self_check(tf, np, session, quantize):
    values = tf.constant([0., .5, 1.5, 2.5, -.5, -1.5, -2.5, 32767., 32768., -32768.], tf.float32)
    q = quantize(tf, values, 0)
    # A dot product deliberately exceeding int32 demonstrates actual accumulator width.
    operand = quantize(tf, tf.constant([[32767., 32767., 32767.]], tf.float32), 0)
    acc = tf.matmul(tf.cast(operand, tf.int64), tf.cast(operand, tf.int64), transpose_b=True)
    observed_q, observed_acc = session.run([q, acc])
    np.testing.assert_array_equal(observed_q, [0, 0, 2, 2, 0, -2, -2, 32767, 32767, -32767])
    if observed_q.dtype != np.int16 or observed_acc.dtype != np.int64:
        raise AssertionError('Quantization/accumulation dtype mismatch')
    np.testing.assert_array_equal(observed_acc, [[3 * 32767 * 32767]])
    return dict(rounding='nearest, ties to even: passed', saturation='symmetric int16: passed',
                int64_dot=int(observed_acc[0, 0]), int32_overflow_case='passed')


def main():
    args = arguments()
    # Reject existing output directories, including old experiments.
    args.out.mkdir(parents=True, exist_ok=False)
    print('OUTPUT_DIRECTORY=' + str(args.out), flush=True)
    started = time.perf_counter()
    old_hashes = fingerprint(args.baseline_run)
    write_json(args.out / 'previous_experiment_sha256_before.json', old_hashes)
    manifest = json.loads((args.baseline_run / 'sample_manifest.json').read_text(encoding='utf-8'))
    total = min(len(manifest), args.limit) if args.limit else len(manifest)
    manifest = manifest[:total]
    write_json(args.out / 'sample_manifest.json', manifest)
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(args.repo))
    import numpy as np
    import tensorflow as tf
    import tensorflow_addons as tfa
    from kwt_int16_graph import build_reference, build_quantized, quantize_int16
    tf.compat.v1.disable_eager_execution()
    from kws_streaming.models import models as registry
    from kws_streaming.models import model_flags
    from kws_streaming.models.transformer_utils import KWSTransformer
    from kws_streaming.train import base_parser
    from kws_streaming.data import input_data
    if (tf.__version__, np.__version__, tfa.__version__) != ('2.4.0', '1.19.5', '0.12.1'):
        raise RuntimeError('Use the pinned .conda Python environment')
    random.seed(59185)
    np.random.seed(59185)
    tf.compat.v1.set_random_seed(59185)
    model_dir = args.repo / 'models_data_v2_12_labels/kwt1'
    flags = make_flags(args, model_dir, base_parser, registry.kws_transformer, model_flags)
    labels = input_data.prepare_words_list(flags.wanted_words.split(','), flags.split_data)
    write_json(args.out / 'inference_flags.json', vars(flags))
    source_dir = args.out / 'source'
    source_dir.mkdir()
    for name in ('run_kwt_int16.py', 'kwt_int16_graph.py', 'run_kwt_fp32.py'):
        shutil.copyfile(Path(__file__).absolute().parent / name, source_dir / name)
    metadata = dict(purpose='W16A16 matmul with FP32 nonlinearities; paired inference, no retraining',
                    observed_commit=git_revision(args.repo), python=sys.version,
                    tensorflow=tf.__version__, numpy=np.__version__, tensorflow_addons=tfa.__version__,
                    device='CPU', batch_size=args.batch_size, labels=labels,
                    baseline_run=str(args.baseline_run), data_dir=str(args.data_dir),
                    upstream_runtime='TF 2.4.0rc1; installed stable 2.4.0 compatibility substitute',
                    checkpoint_sha256={p.name: sha256(p) for p in sorted(model_dir.glob('best_weights.*'))},
                    source_sha256={p.name: sha256(p) for p in source_dir.iterdir()})
    write_json(args.out / 'run_metadata.json', metadata)
    with np.load(str(args.baseline_run / 'predictions.npz')) as old:
        old_logits, old_targets = old['logits'][:total].copy(), old['target_ids'][:total].copy()
    config = tf.compat.v1.ConfigProto(device_count={'GPU': 0})
    with tf.compat.v1.Session(config=config) as session:
        tf.compat.v1.keras.backend.set_session(session)
        tf.keras.backend.set_learning_phase(0)
        processor = input_data.AudioProcessor(flags)
        for row in manifest:
            if processor.word_to_index[row['source_word']] != row['label_id']:
                raise AssertionError('Original manifest label mismatch')
            if not (args.data_dir / row['file']).is_file():
                raise FileNotFoundError(row['file'])
        processor.data_index['validation'] = [dict(label=r['source_word'], file=str(args.data_dir / r['file'])) for r in manifest]
        # Training order is deterministic; record all selected samples for reproducibility.
        calibration = processor.data_index['training'][:args.calibration_samples]
        if len(calibration) != args.calibration_samples:
            raise ValueError('Insufficient training calibration data')
        calibration_manifest = [dict(index=i, file=os.path.relpath(r['file'], str(args.data_dir)),
                                     source_word=r['label'], label_id=int(processor.word_to_index[r['label']]))
                                for i, r in enumerate(calibration)]
        write_json(args.out / 'calibration_manifest.json', calibration_manifest)
        model = registry.kws_transformer.model(flags)
        kwt = next(layer for layer in model.layers if isinstance(layer, KWSTransformer))
        official = build_probe(tf, model, kwt)
        reference, ref_operands = build_reference(tf, model, kwt)
        session.run(tf.compat.v1.global_variables_initializer())
        status = model.load_weights(str(model_dir / 'best_weights'))
        status.assert_existing_objects_matched()
        status.expect_partial()
        print('Checkpoint restored strictly; calibrating on {} training samples'.format(len(calibration)), flush=True)
        checks = numerical_self_check(tf, np, session, quantize_int16)
        export_weights(np, session, kwt, model, args.out)
        with np.load(str(args.out / 'weights_fp32.npz')) as new, np.load(str(args.baseline_run / 'weights_fp32.npz')) as old:
            if set(new.files) != set(old.files):
                raise AssertionError('Checkpoint export keys differ from previous run')
            for key in new.files:
                np.testing.assert_array_equal(new[key], old[key], err_msg=key)
        checks['checkpoint_arrays_exact_match_previous'] = True
        maximum_ops = {k: tf.reduce_max(tf.abs(v)) for k, v in ref_operands.items()}
        maxima = {k: 0.0 for k in ref_operands}

        def get_batch(split, offset, count):
            audio, target = processor.get_data(count, offset, flags, 0., 0., 0, split, 0., 0., session)
            padded = np.zeros((args.batch_size, 16000), np.float32)
            padded[:count] = np.asarray(audio, np.float32)
            # Repeating the final valid sample excludes artificial silence from calibration maxima.
            if count < args.batch_size:
                padded[count:] = padded[count - 1]
            return padded, np.asarray(target, np.int64)

        for offset in range(0, len(calibration), args.batch_size):
            count = min(args.batch_size, len(calibration) - offset)
            audio, _ = get_batch('training', offset, count)
            observed = session.run(maximum_ops, {model.inputs[0]: audio})
            for key, value in observed.items():
                if not np.isfinite(value):
                    raise FloatingPointError('Non-finite calibration operand: ' + key)
                maxima[key] = max(maxima[key], float(value))
            print('Calibration {}/{}'.format(offset + count, len(calibration)), flush=True)
        fractions = {k: choose_fractional_bits(v, 1.0 if k.endswith('/w') else args.activation_headroom)
                     for k, v in maxima.items()}
        qconfig = dict(scheme='static per-operand symmetric signed int16, power-of-two scale',
                       formula='s=2^(-F); q=clip(round_ties_even(x/s), -32767,32767); x_hat=q*s',
                       multiply='cast(q_left,int64) @ cast(q_right,int64); cast(acc,float32)*s_left*s_right',
                       accumulator_bits=64, zero_tensor_fractional_bits=0, integer_range=[-32767, 32767],
                       activation_headroom=args.activation_headroom, calibration_split='training',
                       calibration_samples=len(calibration), calibration_manifest_sha256=sha256(args.out / 'calibration_manifest.json'),
                       floating_operations=['MFCC frontend', 'bias', 'embedding', 'residual addition', 'attention division by sqrt(64)',
                                            'softmax', 'exact GELU', 'LayerNorm epsilon=1e-6'],
                       quantized_operations='98 matmuls: patch + 12*(Q,K,V,QK^T,AV,O,FC1,FC2) + classifier',
                       worst_case_dot_bound=256 * 32767 ** 2,
                       operands={k: dict(fractional_bits=fractions[k], scale=2. ** -fractions[k],
                                         calibration_max_abs=v, headroom=1.0 if k.endswith('/w') else args.activation_headroom,
                                         representable_max=32767 * 2. ** -fractions[k], shape=ref_operands[k].shape.as_list())
                                 for k, v in maxima.items()})
        write_json(args.out / 'quantization_config.json', qconfig)
        variable_count = len(tf.compat.v1.global_variables())
        print('Building 98 integer matmuls; TensorFlow graph construction can take several minutes', flush=True)
        hybrid, hybrid_operands, q_operands, clips = build_quantized(tf, model, kwt, fractions)
        print('Integer graph built; exporting quantized weights', flush=True)
        if len(tf.compat.v1.global_variables()) != variable_count:
            raise AssertionError('Quantized graph created unexpected variables')
        int64_multiplications = [o.name for o in tf.compat.v1.get_default_graph().get_operations()
                                if o.type in ('MatMul', 'BatchMatMul', 'BatchMatMulV2') and o.outputs[0].dtype == tf.int64]
        if len(int64_multiplications) < 98:
            raise AssertionError('Missing integer matmuls: {}'.format(len(int64_multiplications)))
        checks['int64_matmul_operations_in_graph_including_selfcheck'] = len(int64_multiplications)
        weight_values = session.run({k: v for k, v in q_operands.items() if k.endswith('/w')})
        np.savez_compressed(str(args.out / 'weights_int16.npz'), **{k.replace('/', '__'): v for k, v in weight_values.items()})
        original_weights = session.run({k: v for k, v in ref_operands.items() if k.endswith('/w')})
        weight_stats = []
        for key, value in weight_values.items():
            source = original_weights[key]
            reconstructed = value.astype(np.float32) * (2. ** -fractions[key])
            rounded = np.rint(source * (2. ** fractions[key]))
            weight_stats.append(dict(operand=key, elements=int(value.size), dtype=str(value.dtype),
                                     fractional_bits=fractions[key], scale=2. ** -fractions[key],
                                     clipped=int(np.count_nonzero((rounded < -32767) | (rounded > 32767))),
                                     max_abs_error=float(np.abs(reconstructed - source).max()),
                                     mae=float(np.abs(reconstructed - source).mean())))
        dump_csv(args.out / 'weight_quantization_stats.csv', weight_stats)
        print('74 int16 weight matrices exported; preparing paired layer outputs', flush=True)
        non_weight = [k for k in q_operands if not k.endswith('/w')]
        diagnostics = {}
        for key in non_weight:
            x, q = hybrid_operands[key], q_operands[key]
            axes = list(range(1, x.shape.ndims))
            diagnostics[key] = dict(max_abs=tf.reduce_max(tf.abs(x), axis=axes),
                                    clipped=clips[key],
                                    endpoints=tf.reduce_sum(tf.cast(tf.equal(tf.abs(tf.cast(q, tf.int32)), 32767), tf.int64), axis=axes),
                                    zeros=tf.reduce_sum(tf.cast(tf.equal(q, 0), tf.int64), axis=axes))
        quant_stats = {k: dict(operand=k, elements=0, clipped=0, endpoints=0, zeros=0, evaluation_max_abs=0.0,
                               fractional_bits=fractions[k], scale=2. ** -fractions[k]) for k in non_weight}
        stage_names = list(reference)
        if set(reference) != set(hybrid):
            raise AssertionError('Paired stage names differ')
        arrays, error_stats = {}, {k: ErrorAccumulator() for k in stage_names}
        output_manifest = {}
        for branch, tensors in [('fp32', reference), ('w16a16', hybrid)]:
            directory = args.out / 'layer_outputs' / branch
            directory.mkdir(parents=True)
            arrays[branch] = {}
            for key, tensor in tensors.items():
                shape = [total] + tensor.shape.as_list()[1:]
                arrays[branch][key] = np.lib.format.open_memmap(str(directory / (key + '.npy')), mode='w+', dtype=np.float32, shape=tuple(shape))
                output_manifest[branch + '/' + key] = dict(file=str((directory / (key + '.npy')).relative_to(args.out)), shape=shape, dtype='float32')
        write_json(args.out / 'layer_outputs_manifest.json', output_manifest)
        all_ref, all_quant, all_targets = [], [], []
        reconstruction_max = {k: 0.0 for k in official if k in reference}
        old_max_error = 0.0
        evaluation_started = time.perf_counter()
        sample_metrics = (args.out / 'per_sample_layer_metrics.csv').open('w', newline='', encoding='utf-8-sig')
        metric_writer = csv.writer(sample_metrics)
        metric_writer.writerow(['sample_index', 'stage', 'mae', 'rmse', 'max_abs_error', 'relative_l2', 'cosine'])
        try:
            for offset in range(0, total, args.batch_size):
                count = min(args.batch_size, total - offset)
                audio, targets = get_batch('validation', offset, count)
                expected_targets = np.asarray([r['label_id'] for r in manifest[offset:offset + count]], np.int64)
                np.testing.assert_array_equal(targets, expected_targets)
                np.testing.assert_array_equal(targets, old_targets[offset:offset + count])
                fetch = dict(fp32=reference, w16a16=hybrid, diagnostics=diagnostics,
                             official={k: official[k] for k in reconstruction_max})
                if offset == 0:
                    fetch.update(first_ref_operands=ref_operands, first_hybrid_operands=hybrid_operands,
                                 first_int16_operands=q_operands, mfcc=official['mfcc'])
                values = session.run(fetch, {model.inputs[0]: audio})
                for key in reconstruction_max:
                    err = float(np.max(np.abs(values['fp32'][key][:count] - values['official'][key][:count])))
                    reconstruction_max[key] = max(reconstruction_max[key], err)
                    np.testing.assert_allclose(values['fp32'][key][:count], values['official'][key][:count], rtol=1e-5, atol=1e-5, err_msg=key)
                for key in stage_names:
                    ref, quant = values['fp32'][key][:count], values['w16a16'][key][:count]
                    if not np.isfinite(ref).all() or not np.isfinite(quant).all():
                        raise FloatingPointError('Non-finite layer output: ' + key)
                    arrays['fp32'][key][offset:offset + count] = ref
                    arrays['w16a16'][key][offset:offset + count] = quant
                    per_sample = error_stats[key].update(np, ref, quant)
                    for i, metrics in enumerate(per_sample):
                        metric_writer.writerow([offset + i, key] + metrics.tolist())
                for key, observed in values['diagnostics'].items():
                    if not np.isfinite(observed['max_abs'][:count]).all():
                        raise FloatingPointError('Non-finite evaluation operand: ' + key)
                    stats = quant_stats[key]
                    stats['elements'] += count * int(np.prod(hybrid_operands[key].shape.as_list()[1:]))
                    stats['clipped'] += int(observed['clipped'][:count].sum())
                    stats['endpoints'] += int(observed['endpoints'][:count].sum())
                    stats['zeros'] += int(observed['zeros'][:count].sum())
                    stats['evaluation_max_abs'] = max(stats['evaluation_max_abs'], float(observed['max_abs'][:count].max()))
                logits_ref, logits_q = values['fp32']['logits'][:count], values['w16a16']['logits'][:count]
                old_max_error = max(old_max_error, float(np.abs(logits_ref - old_logits[offset:offset + count]).max()))
                np.testing.assert_allclose(logits_ref, old_logits[offset:offset + count], rtol=1e-5, atol=1e-5)
                if offset == 0:
                    first = {'waveform': audio[:1], 'mfcc': values['mfcc'][:1], 'target_id': targets[:1]}
                    for branch in ('fp32', 'w16a16'):
                        for k, v in values[branch].items():
                            first[branch + '__' + k] = v[:1]
                    for branch in ('first_ref_operands', 'first_hybrid_operands', 'first_int16_operands'):
                        for k, v in values[branch].items():
                            first[branch + '__' + k.replace('/', '__')] = v if k.endswith('/w') else v[:1]
                    np.savez_compressed(str(args.out / 'first_sample_trace.npz'), **first)
                all_ref.append(logits_ref)
                all_quant.append(logits_q)
                all_targets.append(targets)
                if offset == 0 or (offset // args.batch_size + 1) % 10 == 0 or offset + count == total:
                    print('Validation {}/{}; paired inference {:.1f}s'.format(offset + count, total, time.perf_counter() - evaluation_started), flush=True)
        finally:
            sample_metrics.close()
            for branch in arrays.values():
                for array in branch.values():
                    array.flush()
        logits_ref, logits_q, targets = np.concatenate(all_ref), np.concatenate(all_quant), np.concatenate(all_targets)
        ref_ids, q_ids = np.argmax(logits_ref, axis=1), np.argmax(logits_q, axis=1)
        np.testing.assert_array_equal(ref_ids, np.argmax(old_logits, axis=1))
        ref_ok, q_ok = ref_ids == targets, q_ids == targets
        confusion_ref, confusion_q = np.zeros((12, 12), np.int64), np.zeros((12, 12), np.int64)
        np.add.at(confusion_ref, (targets, ref_ids), 1)
        np.add.at(confusion_q, (targets, q_ids), 1)
        np.savez_compressed(str(args.out / 'predictions.npz'), fp32_logits=logits_ref, w16a16_logits=logits_q,
                            target_ids=targets, fp32_confusion_matrix=confusion_ref, w16a16_confusion_matrix=confusion_q)
        prediction_rows = [dict(index=i, file=manifest[i]['file'], target=labels[int(targets[i])],
                                fp32_prediction=labels[int(ref_ids[i])], w16a16_prediction=labels[int(q_ids[i])],
                                fp32_correct=bool(ref_ok[i]), w16a16_correct=bool(q_ok[i]), same_prediction=bool(ref_ids[i] == q_ids[i]))
                           for i in range(total)]
        dump_csv(args.out / 'predictions.csv', prediction_rows)
        write_json(args.out / 'changed_predictions.json', [r for r in prediction_rows if not r['same_prediction']])
        stage_rows = [dict(stage=k, **error_stats[k].result()) for k in stage_names]
        dump_csv(args.out / 'layer_comparison.csv', stage_rows)
        write_json(args.out / 'layer_comparison.json', stage_rows)
        for stats in quant_stats.values():
            stats['clipping_rate'] = stats['clipped'] / stats['elements']
            stats['endpoint_rate'] = stats['endpoints'] / stats['elements']
            stats['zero_rate'] = stats['zeros'] / stats['elements']
        dump_csv(args.out / 'activation_quantization_stats.csv', list(quant_stats.values()))
        write_json(args.out / 'activation_quantization_stats.json', quant_stats)
        checks.update(reference_vs_official_layer_max_abs=reconstruction_max,
                      fp32_logits_vs_previous_max_abs=old_max_error, fp32_predictions_match_previous=True,
                      labels_match_previous=True)
        old_after = fingerprint(args.baseline_run)
        write_json(args.out / 'previous_experiment_sha256_after.json', old_after)
        if old_hashes != old_after:
            raise AssertionError('Previous experiment changed during this run')
        checks['previous_experiment_all_files_unchanged'] = True
        write_json(args.out / 'verification.json', checks)
        result = dict(mode='paired validation' if total == len(old_targets) and not args.limit else 'partial pipeline check',
                      samples=total, calibration_samples=len(calibration), fp32_correct=int(ref_ok.sum()),
                      activation_headroom=args.activation_headroom,
                      fp32_accuracy=float(ref_ok.mean()), w16a16_correct=int(q_ok.sum()), w16a16_accuracy=float(q_ok.mean()),
                      accuracy_drop_percentage_points=float((ref_ok.mean() - q_ok.mean()) * 100),
                      prediction_agreement=float((ref_ids == q_ids).mean()), changed_predictions=int((ref_ids != q_ids).sum()),
                      correct_to_wrong=int((ref_ok & ~q_ok).sum()), wrong_to_correct=int((~ref_ok & q_ok).sum()),
                      activation_clipped_elements=sum(v['clipped'] for v in quant_stats.values()),
                      activation_total_elements=sum(v['elements'] for v in quant_stats.values()),
                      validation_seconds=time.perf_counter() - evaluation_started, elapsed_seconds=time.perf_counter() - started,
                      output_directory=str(args.out), previous_experiment_unchanged=True,
                      note='Integer matmul numerical simulation on CPU; no hardware latency/speedup claim. No retraining.')
        write_json(args.out / 'result.json', result)
        make_report(args.out, result, stage_rows, quant_stats, checks)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


def make_report(out, result, stages, quant_stats, checks):
    lines = ['# KWT-1 W16A16 / FP32 非线性对照实验', '',
             '同一官方 checkpoint、同一验证清单，执行成对推理；没有重新训练。', '',
             '| 指标 | 结果 |', '|---|---|',
             '| 验证样本 | {} |'.format(result['samples']),
             '| FP32 | {}/{}，{:.6f}% |'.format(result['fp32_correct'], result['samples'], result['fp32_accuracy'] * 100),
             '| W16A16 | {}/{}，{:.6f}% |'.format(result['w16a16_correct'], result['samples'], result['w16a16_accuracy'] * 100),
             '| 准确率下降 | {:.6f} 个百分点 |'.format(result['accuracy_drop_percentage_points']),
             '| 预测改变 | {} 条；正确变错误 {} 条，错误变正确 {} 条 |'.format(result['changed_predictions'], result['correct_to_wrong'], result['wrong_to_correct']),
             '| FP32 对上次 logits 最大误差 | {:.8g} |'.format(checks['fp32_logits_vs_previous_max_abs']),
             '| 激活实际饱和 | {} / {} |'.format(result['activation_clipped_elements'], result['activation_total_elements']), '',
             '## 数值定义', '',
             '98 个矩阵乘的两个操作数都先量化为 signed int16（-32767…32767），再扩展为 int64 乘加，之后反量化为 FP32。'
             '权重每个张量一个固定尺度；激活每个矩阵操作数一个固定尺度。尺度均为 2 的幂，用训练集 {} 条样本校准，激活留 {} 倍余量。'.format(result['calibration_samples'], result['activation_headroom']) +
             '量化采用 ties-to-even 舍入和饱和裁剪。量化格式、实际尺度、校准范围见 quantization_config.json。', '',
             'MFCC、偏置、嵌入、残差相加、注意力除以 8、Softmax、GELU、LayerNorm 继续采用 FP32。'
             'Softmax 的输出进入 AV 矩阵乘前也量化为 int16。int64 是累加器宽度；本次不是整个网络所有张量都保持 int16。'
             '本实验是 CPU 上的数值仿真，运行时间不能用作 FPGA 加速比。', '',
             '## 每层误差', '', '| 阶段 | MAE | RMSE | 最大绝对误差 | 相对 L2 |', '|---|---:|---:|---:|---:|']
    for row in stages:
        lines.append('| {} | {:.6g} | {:.6g} | {:.6g} | {} |'.format(row['stage'], row['mae'], row['rmse'], row['max_abs_error'],
                     '{:.6g}'.format(row['relative_l2']) if row['relative_l2'] is not None else '未定义'))
    lines += ['', '## 文件', '',
              '- `layer_outputs/fp32/` 与 `layer_outputs/w16a16/`：所有验证样本的 patch、encoder 输入、12 层输出和 logits，逐样本顺序一致。',
              '- `layer_outputs_manifest.json`：每个输出数组的路径、形状、数据类型。用 `numpy.load(path, mmap_mode="r")` 可按需读取。',
              '- `layer_comparison.csv` / `.json`：每层全局误差；`per_sample_layer_metrics.csv`：每个样本逐层误差。',
              '- `first_sample_trace.npz`：首个样本每个矩阵乘的浮点操作数、量化后 int16 操作数以及逐层输出。',
              '- `weights_int16.npz`：量化权重；`weights_fp32.npz`：对应原权重和浮点偏置/归一化参数。尺度在 `quantization_config.json`。',
              '- `activation_quantization_stats.csv`：每量化点实际饱和数、端点数、零值数与最大激活值，端点数不等于饱和数。',
              '- `weight_quantization_stats.csv`：每个权重矩阵的量化格式、实际误差、饱和数。',
              '- `predictions.csv` / `.npz`：两条路径的预测、logits、标签、混淆矩阵；`changed_predictions.json`：预测改变的样本。',
              '- `calibration_manifest.json`：训练校准清单；`sample_manifest.json`：与上次完全一致的验证清单。',
              '- `verification.json`：舍入/饱和/int64 累加自检、逐层 FP32 官方图一致性、旧结果一致性检查。',
              '- `source/`：本次代码快照；`run_metadata.json`：版本与 checkpoint/source 哈希。', '',
              '旧实验全部文件已在运行前后逐文件计算 SHA256，确认未改变。', '']
    (out / 'README_量化结果.md').write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    main()
