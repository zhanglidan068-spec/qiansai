"""Fresh W8A8 + fixed16 nonlinear validation; never evaluate testing samples.

Train-only no-gradient calibration, real integer matmuls/nonlinear kernels.
Old prediction files become readable only after new logits have been committed.
All experiment directories are exclusive, so prior results are preserved.
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
from run_kwt_int16 import ErrorAccumulator, dump_csv


ROOT = Path(__file__).absolute().parent


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, default=ROOT / 'keyword-transformer')
    p.add_argument('--data-dir', type=Path, default=ROOT / 'data2')
    p.add_argument('--fp32-reference', type=Path, default=ROOT / 'runs/validation_full')
    p.add_argument('--w16-reference', type=Path, default=ROOT / 'runs/w16a16_fpnl_20261028_103350_996076')
    p.add_argument('--out', type=Path)
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--calibration-samples', type=int, default=256)
    p.add_argument('--activation-headroom', type=float, default=1.25)
    a = p.parse_args()
    if a.batch_size < 1 or a.calibration_samples < 1 or a.limit < 0 or not math.isfinite(a.activation_headroom) or a.activation_headroom < 1:
        p.error('Invalid batch/calibration/limit/headroom')
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S_%f')
    a.out = (a.out or ROOT / 'runs' / ('w8a8_n16_avf6_' + stamp)).absolute()
    for name in ('repo', 'data_dir', 'fp32_reference', 'w16_reference'):
        setattr(a, name, getattr(a, name).absolute())
    a.smoke = False
    return a


def fraction(maximum, bits, headroom=1.):
    if not math.isfinite(maximum):
        raise FloatingPointError('Nonfinite calibration maximum')
    return 0 if maximum == 0 else int(math.floor(math.log2(((1 << (bits - 1)) - 1.) / (maximum * headroom))))


def stat_inventory(directory):
    return {str(p.relative_to(directory)): dict(size=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns)
            for p in sorted(directory.rglob('*')) if p.is_file()}


class PredictionReadGuard:
    def __init__(self, output):
        self.output = os.path.normcase(os.path.abspath(str(output)))
        self.complete = False
        self.before_completion_reads = []
        self.after_completion_reads = []

    def hook(self, event, args):
        if event != 'open' or not args or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = os.path.normcase(os.path.abspath(os.fsdecode(args[0])))
        if path == self.output or path.startswith(self.output + os.sep):
            return
        base = os.path.basename(path)
        if base in ('predictions.npz', 'predictions.csv', 'inference_logits.npz', 'changed_predictions.json'):
            if not self.complete:
                self.before_completion_reads.append(path)
                raise PermissionError('Old predictions cannot be read before fresh inference is complete')
            self.after_completion_reads.append(path)


def main():
    a = arguments()
    a.out.mkdir(parents=True, exist_ok=False)
    print('OUTPUT_DIRECTORY=' + str(a.out), flush=True)
    started = time.perf_counter()
    before = {str(p): stat_inventory(p) for p in (a.fp32_reference, a.w16_reference)}
    write_json(a.out / 'previous_experiment_inventory_before.json', before)
    guard = PredictionReadGuard(a.out)
    sys.addaudithook(guard.hook)
    # The old sample list aligns the comparison; no old predictions are loaded.
    all_manifest = json.loads((a.fp32_reference / 'sample_manifest.json').read_text(encoding='utf-8'))
    total = min(a.limit, len(all_manifest)) if a.limit else len(all_manifest)
    manifest = all_manifest[:total]
    write_json(a.out / 'sample_manifest.json', manifest)
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(a.repo))
    import numpy as np
    import tensorflow as tf
    import tensorflow_addons as tfa
    from kwt_int8_n16_graph import build_reference, build_quantized
    from kwt_nonlinear_int16 import implementation_metadata, export_lut_tables
    tf.compat.v1.disable_eager_execution()
    from kws_streaming.models import models as registry
    from kws_streaming.models import model_flags
    from kws_streaming.models.transformer_utils import KWSTransformer
    from kws_streaming.train import base_parser
    from kws_streaming.data import input_data
    if (tf.__version__, np.__version__, tfa.__version__) != ('2.4.0', '1.19.5', '0.12.1'):
        raise RuntimeError('Use the pinned .conda environment')
    random.seed(59185)
    np.random.seed(59185)
    tf.compat.v1.set_random_seed(59185)
    model_dir = a.repo / 'models_data_v2_12_labels/kwt1'
    flags = make_flags(a, model_dir, base_parser, registry.kws_transformer, model_flags)
    labels = input_data.prepare_words_list(flags.wanted_words.split(','), flags.split_data)
    if labels != (model_dir / 'labels.txt').read_text(encoding='utf-8').split():
        raise AssertionError('Checkpoint class order mismatch')
    write_json(a.out / 'inference_flags.json', vars(flags))
    source = a.out / 'source'
    source.mkdir()
    for name in ('run_kwt_int8_n16.py', 'kwt_int8_n16_graph.py', 'kwt_nonlinear_int16.py', 'run_kwt_fp32.py', 'run_kwt_int16.py'):
        shutil.copyfile(ROOT / name, source / name)
    metadata = dict(mode='validation only; testing prohibited', labels=labels, batch_size=a.batch_size,
                    python=sys.version, tensorflow=tf.__version__, numpy=np.__version__, tensorflow_addons=tfa.__version__,
                    observed_commit=git_revision(a.repo), device='CPU', calibration_split='training',
                    fp32_reference=str(a.fp32_reference), w16_reference=str(a.w16_reference),
                    checkpoint_sha256={p.name: sha256(p) for p in sorted(model_dir.glob('best_weights.*'))},
                    source_sha256={p.name: sha256(p) for p in source.iterdir()},
                    inference='No previous prediction is readable until new logits are committed',
                    runtime_note='TF2.4.0 stable compatibility substitute for upstream rc1')
    write_json(a.out / 'run_metadata.json', metadata)
    split_requests = dict(training=0, validation=0, testing=0)
    with tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(device_count={'GPU': 0})) as sess:
        tf.compat.v1.keras.backend.set_session(sess)
        tf.keras.backend.set_learning_phase(0)
        processor = input_data.AudioProcessor(flags)
        for row in manifest:
            if not row['synthetic_zero_waveform'] and input_data.which_set(row['file'], flags.validation_percentage, flags.testing_percentage) != 'validation':
                raise AssertionError('Requested waveform does not belong to validation')
            if processor.word_to_index[row['source_word']] != row['label_id']:
                raise AssertionError('Manifest label mismatch')
        processor.data_index['validation'] = [dict(label=row['source_word'], file=str(a.data_dir / row['file'])) for row in manifest]
        calibration = processor.data_index['training'][:a.calibration_samples]
        if len(calibration) != a.calibration_samples:
            raise ValueError('Insufficient training calibration samples')
        calibration_manifest = [dict(index=i, file=os.path.relpath(row['file'], str(a.data_dir)), source_word=row['label'],
                                     label_id=int(processor.word_to_index[row['label']]), synthetic_zero_waveform=row['label'] == '_silence_')
                                for i, row in enumerate(calibration)]
        for row in calibration_manifest:
            if not row['synthetic_zero_waveform'] and input_data.which_set(row['file'], flags.validation_percentage, flags.testing_percentage) != 'training':
                raise AssertionError('Calibration waveform is not in training split')
        write_json(a.out / 'calibration_manifest.json', calibration_manifest)
        model = registry.kws_transformer.model(flags)
        kwt = next(layer for layer in model.layers if isinstance(layer, KWSTransformer))
        official = build_probe(tf, model, kwt)
        reference, ref_operands, nl_reference = build_reference(tf, model, kwt)
        sess.run(tf.compat.v1.global_variables_initializer())
        restore = model.load_weights(str(model_dir / 'best_weights'))
        restore.assert_existing_objects_matched()
        restore.expect_partial()
        export_weights(np, sess, kwt, model, a.out)

        def batch(split, offset, count):
            if split not in ('training', 'validation'):
                raise RuntimeError('Testing sample access is prohibited')
            split_requests[split] += count
            audio, targets = processor.get_data(count, offset, flags, 0., 0., 0, split, 0., 0., sess)
            padded = np.zeros((a.batch_size, 16000), np.float32)
            padded[:count] = np.asarray(audio, np.float32)
            if count < a.batch_size:
                padded[count:] = padded[count - 1]
            return padded, np.asarray(targets, np.int64)

        ranges = dict(ref_operands)
        ranges.update({'nl/' + key: value for key, value in nl_reference.items()})
        maximum_ops = {key: tf.reduce_max(tf.abs(value)) for key, value in ranges.items()}
        maxima = {key: 0. for key in ranges}
        for offset in range(0, len(calibration), a.batch_size):
            count = min(a.batch_size, len(calibration) - offset)
            audio, _ = batch('training', offset, count)
            observed = sess.run(maximum_ops, {model.inputs[0]: audio})
            for key, value in observed.items():
                if not np.isfinite(value):
                    raise FloatingPointError('Nonfinite calibration range: ' + key)
                maxima[key] = max(maxima[key], float(value))
            print('Training calibration {}/{}'.format(offset + count, len(calibration)), flush=True)
        fractions = {key: (6 if key.endswith('/av/a') else fraction(maxima[key], 8, 1. if key.endswith('/w') else a.activation_headroom)) for key in ref_operands}
        nonlinear_config = {}
        for i in range(1, 13):
            for kind in ('softmax', 'ln1', 'gelu', 'ln2'):
                key = 'layer_{:02d}/{}'.format(i, kind)
                nonlinear_config[key] = dict(input_f=fraction(maxima['nl/' + key + '/input'], 16, a.activation_headroom),
                    output_f=15 if kind == 'softmax' else fraction(maxima['nl/' + key + '/output'], 16, a.activation_headroom))
                if kind.startswith('ln'):
                    nonlinear_config[key].update(gamma_f=14, beta_f=16, epsilon=1e-6)
        config = dict(scheme='W8A8 + integer/LUT N16; fixed Softmax-to-AV Q6', integer_range8=[-127, 127], integer_range16=[-32767, 32767],
                      round_mode='nearest ties to even at quantization boundaries', accumulator_bits=64,
                      calibration_split='training', calibration_samples=len(calibration), activation_headroom=a.activation_headroom,
                      softmax_to_av=dict(dtype='int8', fractional_bits=6, scale=1 / 64., row_renormalization=False),
                      softmax_output=dict(dtype='int16', fractional_bits=15),
                      matmul_operands={key: dict(fractional_bits=fractions[key], scale=2. ** -fractions[key],
                                                 calibration_max_abs=maxima[key], shape=ref_operands[key].shape.as_list()) for key in ref_operands},
                      nonlinear=nonlinear_config,
                      nonlinear_implementation_metadata=implementation_metadata(),
                      floating_boundary_operations=['MFCC frontend', 'embedding/residual additions', 'Dense bias', 'power-of-two scale transport', 'attention score division by 8'],
                      nonlinear_implementation='Integer/LUT forward; LUTs generated offline. Int64 wide intermediate arithmetic. See source/kwt_nonlinear_int16.py')
        write_json(a.out / 'quantization_config.json', config)
        print('Building W8A8 + fixed16 nonlinear graph; no validation-based recalibration', flush=True)
        variable_count = len(tf.compat.v1.global_variables())
        hybrid, hybrid_operands, q_operands, clips, nonlinear_records = build_quantized(tf, model, kwt, fractions, nonlinear_config)
        if variable_count != len(tf.compat.v1.global_variables()):
            raise AssertionError('Quantized graph introduced model variables')
        integer_matmuls = [op for op in tf.compat.v1.get_default_graph().get_operations() if op.type in ('MatMul', 'BatchMatMul', 'BatchMatMulV2') and op.outputs[0].dtype == tf.int64]
        if len(integer_matmuls) != 98:
            raise AssertionError('Expected 98 integer matrix multiplies')
        forbidden_nl_ops = [dict(name=op.name, type=op.type) for op in tf.compat.v1.get_default_graph().get_operations()
                            if op.name.startswith('kwt_w8a8_n16/') and op.type in ('Exp', 'Softmax', 'Erf', 'Sqrt', 'Rsqrt', 'Tanh')]
        if forbidden_nl_ops:
            raise AssertionError('Floating nonlinear function in quantized graph: ' + str(forbidden_nl_ops))
        # Inside nonlinear kernels, FP operations are permitted only at explicit
        # input quantization / output transport boundaries, not function evaluation.
        weights = sess.run({key: value for key, value in q_operands.items() if key.endswith('/w')})
        if len(weights) != 74 or any(v.dtype != np.int8 for v in weights.values()):
            raise AssertionError('Expected 74 int8 weight matrices')
        np.savez_compressed(str(a.out / 'weights_int8.npz'), **{key.replace('/', '__'): value for key, value in weights.items()})
        np.savez_compressed(str(a.out / 'nonlinear_luts_int16.npz'), **export_lut_tables(nonlinear_config))
        config['nonlinear'] = nonlinear_config
        write_json(a.out / 'quantization_config.json', config)
        print('Integer graph ready; preparing full layer outputs', flush=True)
        qbranch = 'w8a8_n16'
        stages = list(reference)
        arrays, output_specs = {}, {}
        definitions = [('fp32', reference), (qbranch, hybrid),
                       ('n16_codes', {'layer_{:02d}_output'.format(i): nonlinear_records['layer_{:02d}/ln2'.format(i)]['output_int16'] for i in range(1, 13)}),
                       ('av_int8_f6', {'layer_{:02d}_attention'.format(i): q_operands['layer_{:02d}/av/a'.format(i)] for i in range(1, 13)})]
        for branch, tensors in definitions:
            directory = a.out / 'layer_outputs' / branch
            directory.mkdir(parents=True)
            arrays[branch] = {}
            for key, tensor in tensors.items():
                shape = [total] + tensor.shape.as_list()[1:]
                dtype = tensor.dtype.as_numpy_dtype
                path = directory / (key + '.npy')
                arrays[branch][key] = np.lib.format.open_memmap(str(path), mode='w+', dtype=dtype, shape=tuple(shape))
                output_specs[branch + '/' + key] = dict(file=str(path.relative_to(a.out)), shape=shape, dtype=str(np.dtype(dtype)))
        write_json(a.out / 'layer_outputs_manifest.json', output_specs)
        errors = {key: ErrorAccumulator() for key in stages}
        nl_errors = {key: ErrorAccumulator() for key in nonlinear_records}
        diagnostics = {}
        for key in clips:
            x, q = hybrid_operands[key], q_operands[key]
            axes = list(range(1, x.shape.ndims))
            diagnostics[key] = dict(clipped=clips[key], max_abs=tf.reduce_max(tf.abs(x), axis=axes),
                                    zeros=tf.reduce_sum(tf.cast(tf.equal(q, 0), tf.int64), axis=axes),
                                    endpoints=tf.reduce_sum(tf.cast(tf.equal(tf.abs(tf.cast(q, tf.int32)), 127), tf.int64), axis=axes))
        mat_stats = {key: dict(operand=key, elements=0, clipped=0, zeros=0, endpoints=0, evaluation_max_abs=0., fractional_bits=fractions[key], scale=2. ** -fractions[key]) for key in clips}
        nl_stats = {key: dict(block=key, input_elements=0, output_elements=0, input_clipped=0, output_clipped=0,
                             input_f=nonlinear_config[key]['input_f'], output_f=nonlinear_config[key]['output_f']) for key in nonlinear_records}
        attention_stats = {i: dict(layer=i, rows=0, row_sum_min=float('inf'), row_sum_max=0., row_sum_sum=0., row_sum_abs_error_sum=0., zeros=0, elements=0) for i in range(1, 13)}
        official_error = {key: 0. for key in stages if key in official}
        logits_ref, logits_quant, targets_all = [], [], []
        infer_start = time.perf_counter()
        csv_path = a.out / 'per_sample_layer_metrics.csv'
        with csv_path.open('w', newline='', encoding='utf-8-sig') as metrics_stream:
            metric_writer = csv.writer(metrics_stream)
            metric_writer.writerow(['sample_index', 'stage', 'mae', 'rmse', 'max_abs_error', 'relative_l2', 'cosine'])
            for offset in range(0, total, a.batch_size):
                count = min(a.batch_size, total - offset)
                audio, targets = batch('validation', offset, count)
                np.testing.assert_array_equal(targets, [row['label_id'] for row in manifest[offset:offset + count]])
                fetch = dict(fp32=reference, quantized=hybrid, diagnostics=diagnostics,
                             official={key: official[key] for key in official_error},
                             nonlinear={key: {k: record[k] for k in ('input_clipped', 'output_clipped', 'output_dequant')} for key, record in nonlinear_records.items()},
                             nonlinear_reference={key: nl_reference[key + '/output'] for key in nonlinear_records},
                             n16_codes={key: tensor for branch, tensors in definitions if branch == 'n16_codes' for key, tensor in tensors.items()},
                             av_codes={key: tensor for branch, tensors in definitions if branch == 'av_int8_f6' for key, tensor in tensors.items()})
                if offset == 0:
                    fetch.update(first_operands=hybrid_operands, first_q=q_operands, first_nonlinear=nonlinear_records, mfcc=official['mfcc'])
                values = sess.run(fetch, {model.inputs[0]: audio})
                for key in official_error:
                    r, o = values['fp32'][key][:count], values['official'][key][:count]
                    official_error[key] = max(official_error[key], float(np.abs(r - o).max()))
                    np.testing.assert_allclose(r, o, rtol=1e-5, atol=1e-5, err_msg=key)
                for key in stages:
                    r, q = values['fp32'][key][:count], values['quantized'][key][:count]
                    if not np.isfinite(r).all() or not np.isfinite(q).all():
                        raise FloatingPointError('Nonfinite output: ' + key)
                    arrays['fp32'][key][offset:offset + count] = r
                    arrays[qbranch][key][offset:offset + count] = q
                    for i, metrics in enumerate(errors[key].update(np, r, q)):
                        metric_writer.writerow([offset + i, key] + metrics.tolist())
                for branch, field in [('n16_codes', 'n16_codes'), ('av_int8_f6', 'av_codes')]:
                    for key, tensor in values[field].items():
                        arrays[branch][key][offset:offset + count] = tensor[:count]
                for key, observed in values['diagnostics'].items():
                    stat = mat_stats[key]
                    stat['elements'] += count * int(np.prod(hybrid_operands[key].shape.as_list()[1:]))
                    for name in ('clipped', 'zeros', 'endpoints'):
                        stat[name] += int(observed[name][:count].sum())
                    if not np.isfinite(observed['max_abs'][:count]).all():
                        raise FloatingPointError('Nonfinite matrix operand: ' + key)
                    stat['evaluation_max_abs'] = max(stat['evaluation_max_abs'], float(observed['max_abs'][:count].max()))
                for key, observed in values['nonlinear'].items():
                    q = observed['output_dequant'][:count]
                    nl_errors[key].update(np, values['nonlinear_reference'][key][:count], q)
                    stat = nl_stats[key]
                    stat['input_elements'] += count * int(np.prod(nonlinear_records[key]['input_int16'].shape.as_list()[1:]))
                    stat['output_elements'] += q.size
                    for name in ('input_clipped', 'output_clipped'):
                        stat[name] += int(observed[name][:count].sum())
                for i in range(1, 13):
                    q = values['av_codes']['layer_{:02d}_attention'.format(i)][:count]
                    sums = q.astype(np.int64).sum(axis=-1) / 64.
                    stat = attention_stats[i]
                    stat['rows'] += sums.size
                    stat['row_sum_min'] = min(stat['row_sum_min'], float(sums.min()))
                    stat['row_sum_max'] = max(stat['row_sum_max'], float(sums.max()))
                    stat['row_sum_sum'] += float(sums.sum())
                    stat['row_sum_abs_error_sum'] += float(np.abs(sums - 1).sum())
                    stat['zeros'] += int(np.count_nonzero(q == 0))
                    stat['elements'] += q.size
                logits_ref.append(values['fp32']['logits'][:count])
                logits_quant.append(values['quantized']['logits'][:count])
                targets_all.append(targets)
                if offset == 0:
                    first = dict(waveform=audio[:1], mfcc=values['mfcc'][:1], target_id=targets[:1])
                    for branch in ('fp32', 'quantized'):
                        for key, tensor in values[branch].items():
                            first[branch + '__' + key] = tensor[:1]
                    for branch in ('first_operands', 'first_q'):
                        for key, tensor in values[branch].items():
                            first[branch + '__' + key.replace('/', '__')] = tensor if key.endswith('/w') else tensor[:1]
                    for key, record in values['first_nonlinear'].items():
                        def flatten_first(prefix, item):
                            if isinstance(item, dict):
                                for subkey, value in item.items():
                                    flatten_first(prefix + '__' + subkey, value)
                            else:
                                tensor = np.asarray(item)
                                if tensor.dtype == np.dtype('object'):
                                    raise TypeError('Object array is not a numerical trace: ' + prefix)
                                first[prefix] = tensor[:1] if tensor.ndim and tensor.shape[0] == a.batch_size else tensor
                        flatten_first('nonlinear__' + key.replace('/', '__'), record)
                    np.savez_compressed(str(a.out / 'first_sample_trace.npz'), **first)
                if offset == 0 or (offset // a.batch_size + 1) % 10 == 0 or offset + count == total:
                    print('Validation {}/{}; {:.1f}s'.format(offset + count, total, time.perf_counter() - infer_start), flush=True)
        for branch in arrays.values():
            for array in branch.values():
                array.flush()
                array._mmap.close()
        ref, quant, targets = np.concatenate(logits_ref), np.concatenate(logits_quant), np.concatenate(targets_all)
        np.savez_compressed(str(a.out / 'fresh_inference_logits.npz'), fp32_logits=ref, w8a8_n16_logits=quant)
        write_json(a.out / 'inference_complete.json', dict(samples=total, sha256=sha256(a.out / 'fresh_inference_logits.npz'),
                   validation_seconds=time.perf_counter() - infer_start, phase='new logits committed; previous predictions have not been read',
                   previous_prediction_reads=guard.before_completion_reads, split_sample_requests=split_requests))
        guard.complete = True
        # Only comparison/verification below can read historical predictions.
        with np.load(str(a.fp32_reference / 'predictions.npz'), allow_pickle=False) as previous:
            previous_ref, previous_targets = previous['logits'][:total], previous['target_ids'][:total]
            np.testing.assert_array_equal(targets, previous_targets)
            np.testing.assert_allclose(ref, previous_ref, rtol=1e-5, atol=1e-5)
            old_fp32_max_error = float(np.abs(ref - previous_ref).max())
        with np.load(str(a.w16_reference / 'predictions.npz'), allow_pickle=False) as previous:
            w16, w16_targets = previous['w16a16_logits'][:total].copy(), previous['target_ids'][:total]
            np.testing.assert_array_equal(targets, w16_targets)
        if json.loads((a.w16_reference / 'sample_manifest.json').read_text(encoding='utf-8'))[:total] != manifest:
            raise AssertionError('W16 sample order mismatch')
        ref_ids, q_ids, w16_ids = ref.argmax(1), quant.argmax(1), w16.argmax(1)
        ref_ok, q_ok, w16_ok = ref_ids == targets, q_ids == targets, w16_ids == targets
        confusion_ref, confusion_q = np.zeros((12, 12), np.int64), np.zeros((12, 12), np.int64)
        np.add.at(confusion_ref, (targets, ref_ids), 1)
        np.add.at(confusion_q, (targets, q_ids), 1)
        np.savez_compressed(str(a.out / 'predictions.npz'), fp32_logits=ref, w8a8_n16_logits=quant, target_ids=targets,
                            fp32_confusion_matrix=confusion_ref, w8a8_n16_confusion_matrix=confusion_q)
        prediction_rows = [dict(index=i, file=manifest[i]['file'], target=labels[int(targets[i])],
                                fp32_prediction=labels[int(ref_ids[i])], w16a16_prediction=labels[int(w16_ids[i])], w8a8_n16_prediction=labels[int(q_ids[i])],
                                fp32_correct=bool(ref_ok[i]), w16a16_correct=bool(w16_ok[i]), w8a8_n16_correct=bool(q_ok[i]), same_as_fp32=bool(ref_ids[i] == q_ids[i])) for i in range(total)]
        dump_csv(a.out / 'predictions.csv', prediction_rows)
        write_json(a.out / 'changed_predictions.json', [row for row in prediction_rows if not row['same_as_fp32']])
        rows = [dict(stage=key, **stat.result()) for key, stat in errors.items()]
        dump_csv(a.out / 'layer_comparison.csv', rows)
        write_json(a.out / 'layer_comparison.json', rows)
        nonlinear_rows = [dict(block=key, **stat.result(), **{k: v for k, v in nl_stats[key].items() if k != 'block'}) for key, stat in nl_errors.items()]
        dump_csv(a.out / 'nonlinear_comparison.csv', nonlinear_rows)
        write_json(a.out / 'nonlinear_comparison.json', nonlinear_rows)
        for stat in mat_stats.values():
            for name in ('clipped', 'zeros', 'endpoints'):
                stat[name + '_rate'] = stat[name] / stat['elements']
        dump_csv(a.out / 'activation_quantization_stats.csv', list(mat_stats.values()))
        write_json(a.out / 'activation_quantization_stats.json', mat_stats)
        for stat in attention_stats.values():
            stat['row_sum_mean'] = stat['row_sum_sum'] / stat['rows']
            stat['row_sum_mean_abs_error_from_one'] = stat['row_sum_abs_error_sum'] / stat['rows']
            stat['zero_rate'] = stat['zeros'] / stat['elements']
        dump_csv(a.out / 'attention_av_f6_stats.csv', list(attention_stats.values()))
        write_json(a.out / 'attention_av_f6_stats.json', attention_stats)
        comparison = [dict(model=name, samples=total, correct=int(ok.sum()), accuracy=float(ok.mean())) for name, ok in [('FP32', ref_ok), ('W16A16_FP32_nonlinear', w16_ok), ('W8A8_N16_AV_F6', q_ok)]]
        dump_csv(a.out / 'experiment_comparison.csv', comparison)
        after = {str(p): stat_inventory(p) for p in (a.fp32_reference, a.w16_reference)}
        write_json(a.out / 'previous_experiment_inventory_after.json', after)
        if before != after:
            raise AssertionError('Previous experiment file metadata changed')
        verification = dict(integer_matmuls=len(integer_matmuls), int8_weight_matrices=len(weights),
                            forbidden_floating_nonlinear_ops_in_quantized_graph=forbidden_nl_ops,
                            fp32_reference_vs_official_max_abs=official_error, fp32_logits_vs_previous_max_abs=old_fp32_max_error,
                            previous_prediction_reads_before_fresh_inference=guard.before_completion_reads,
                            historical_prediction_files_read_only_after_completion=guard.after_completion_reads,
                            split_sample_requests=split_requests, testing_samples_used=0,
                            previous_experiment_all_file_sizes_and_mtimes_unchanged=True,
                            av_fractional_bits_all_6=all(fractions['layer_{:02d}/av/a'.format(i)] == 6 for i in range(1, 13)))
        write_json(a.out / 'verification.json', verification)
        result = dict(mode='validation' if not a.limit else 'partial validation pipeline check', samples=total,
                      calibration_samples=len(calibration), fp32_correct=int(ref_ok.sum()), fp32_accuracy=float(ref_ok.mean()),
                      previous_w16a16_correct=int(w16_ok.sum()), previous_w16a16_accuracy=float(w16_ok.mean()),
                      w8a8_n16_correct=int(q_ok.sum()), w8a8_n16_accuracy=float(q_ok.mean()),
                      accuracy_drop_vs_fp32_percentage_points=float((ref_ok.mean() - q_ok.mean()) * 100),
                      accuracy_drop_vs_w16_percentage_points=float((w16_ok.mean() - q_ok.mean()) * 100),
                      changed_predictions_vs_fp32=int((ref_ids != q_ids).sum()), correct_to_wrong=int((ref_ok & ~q_ok).sum()), wrong_to_correct=int((~ref_ok & q_ok).sum()),
                      matrix_activation_clipped_elements=sum(s['clipped'] for s in mat_stats.values()),
                      nonlinear_input_clipped_elements=sum(s['input_clipped'] for s in nl_stats.values()),
                      nonlinear_output_clipped_elements=sum(s['output_clipped'] for s in nl_stats.values()),
                      validation_seconds=time.perf_counter() - infer_start, elapsed_seconds=time.perf_counter() - started,
                      testing_samples_used=0, previous_experiments_unchanged=True, output_directory=str(a.out),
                      note='Integer numerical simulation, not FPGA timing. Training-only calibration; no retraining or testing samples.')
        write_json(a.out / 'result.json', result)
        report(a.out, result, rows, comparison, attention_stats)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


def report(out, result, stages, comparison, attention):
    lines = ['# KWT-1 W8A8 + N16，Softmax→AV 固定 F=6', '',
             '使用训练集 {} 条样本做无梯度标定，使用验证集 {} 条样本评估；testing 使用量为 0。每次推理均重新读取音频计算，旧预测只在新 logits 保存后用于对照。'.format(result['calibration_samples'], result['samples']), '',
             '| 模型 | 正确数 | 准确率 |', '|---|---:|---:|']
    lines += ['| {} | {} / {} | {:.6f}% |'.format(row['model'], row['correct'], row['samples'], row['accuracy'] * 100) for row in comparison]
    lines += ['', '比 FP32 下降 {:.6f} 个百分点；比上次 W16A16 下降 {:.6f} 个百分点。相对 FP32，{} 条预测改变，正确变错误 {} 条、错误变正确 {} 条。'.format(result['accuracy_drop_vs_fp32_percentage_points'], result['accuracy_drop_vs_w16_percentage_points'], result['changed_predictions_vs_fp32'], result['correct_to_wrong'], result['wrong_to_correct']), '',
              '## 数值实现', '',
              '- 所有 98 个矩阵乘的两个操作数为 signed int8，范围 [-127,127]，使用固定 2 的幂尺度、ties-to-even 舍入、饱和裁剪；乘法前扩展为 int64 做宽累加。',
              '- Softmax、GELU、LayerNorm 的输入输出为 signed int16。GELU 与 exp 查表离线生成，前向采用整数查表；LayerNorm 采用整数均值、方差、整数平方根及定点 gamma/beta。中间累加、乘除允许 int64；不是所有内部中间量都限制为16位。精确定义见 source/kwt_nonlinear_int16.py。',
              '- Softmax 首先产生 int16 Q15 概率，再量化成 int8、F=6、scale=1/64，作为 AV 左操作数。没有再次归一化行和。',
              '- MFCC、偏置、嵌入、残差相加和定点尺度之间的传输保留 FP32；本次量化范围是矩阵乘和上述三种非线性。',
              '- 除指定的 AV F=6、Softmax Q15、LayerNorm参数格式外，尺度仅由训练样本的浮点路径统计确定，未根据验证结果重新标定。', '',
              '## 每层误差（相对本次重新计算的 FP32）', '', '| 阶段 | MAE | RMSE | 最大绝对误差 | 相对L2 |', '|---|---:|---:|---:|---:|']
    lines += ['| {} | {:.6g} | {:.6g} | {:.6g} | {:.6g} |'.format(r['stage'], r['mae'], r['rmse'], r['max_abs_error'], r['relative_l2'] or 0.) for r in stages]
    lines += ['', '## AV 的 F=6 量化影响', '',
              '概率最小步长为 0.015625，舍入后小概率可能变成0，且每行概率之和未必等于1。下表记录实际进入AV矩阵乘的整数概率，按q/64还原后的统计。', '',
              '| 层 | 行和均值 | 行和最小值 | 行和最大值 | 零值比例 |', '|---|---:|---:|---:|---:|']
    lines += ['| {} | {:.6f} | {:.6f} | {:.6f} | {:.4f}% |'.format(i, s['row_sum_mean'], s['row_sum_min'], s['row_sum_max'], s['zero_rate'] * 100) for i, s in attention.items()]
    lines += ['', '## 保存结果', '',
              '- layer_outputs/fp32 与 layer_outputs/w8a8_n16：全部验证样本的15个阶段完整浮点输出，便于直接对照。',
              '- layer_outputs/n16_codes：12层最终LayerNorm输出的int16原始码；每层F位于quantization_config.json。',
              '- layer_outputs/av_int8_f6：12层实际进入AV的int8概率码；除以64得到定点实值。',
              '- layer_outputs_manifest.json：所有数组路径、shape、dtype；用numpy.load(path,mmap_mode="r")按需读取。',
              '- layer_comparison.csv 与 per_sample_layer_metrics.csv：逐层全局/逐样本误差；nonlinear_comparison.csv：48个非线性节点的误差与饱和数。',
              '- attention_av_f6_stats.csv：AV概率行和及零比例；activation_quantization_stats.csv：每个矩阵操作数的饱和数、零值、端点比例。',
              '- weights_int8.npz 与 weights_fp32.npz：量化权重和对应原参数；quantization_config.json：所有量化尺度与非线性参数。',
              '- first_sample_trace.npz：首个样本逐层输出、矩阵乘int8操作数、非线性int16输入输出。',
              '- fresh_inference_logits.npz、inference_complete.json：读取任何旧预测之前保存的新推理结果及哈希。',
              '- predictions.npz / .csv、changed_predictions.json：预测与混淆矩阵；experiment_comparison.csv：三种精度实验准确率对照。',
              '- verification.json、previous_experiment_inventory_before/after.json：没有testing样本、旧文件尺寸/修改时间未变、FP32对官方路径的一致性检查。',
              '- source/ 与 run_metadata.json：源码快照、版本、checkpoint与代码SHA256。', '',
              '本实验是CPU数值仿真，运行时间不能用作FPGA加速比。', '']
    (out / 'README_量化结果.md').write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    main()
