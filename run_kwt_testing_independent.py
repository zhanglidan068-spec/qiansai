"""Fresh FP32/W16A16 testing inference, with no dependency on prior experiments.

Only training samples calibrate the scales. Testing logits are committed before
the separate scorer reads their labels. Historical result directories are denied
by a Python file-open audit hook. Existing experiments are never changed.
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


ROOT = Path(__file__).absolute().parent


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, default=ROOT / 'keyword-transformer')
    p.add_argument('--data-dir', type=Path, default=ROOT / 'data2')
    p.add_argument('--out', type=Path)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--calibration-samples', type=int, default=256)
    p.add_argument('--activation-headroom', type=float, default=1.25)
    p.add_argument('--limit', type=int, default=0, help='0 = entire testing split')
    p.add_argument('--inference-only', action='store_true', help='Commit fresh logits; score separately later')
    a = p.parse_args()
    if a.batch_size < 1 or a.calibration_samples < 1 or a.limit < 0 or not math.isfinite(a.activation_headroom) or a.activation_headroom < 1:
        p.error('Invalid batch size, calibration count, limit or headroom')
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S_%f')
    a.out = (a.out or ROOT / 'testing_runs' / ('independent_testing_' + stamp)).absolute()
    a.repo, a.data_dir = a.repo.absolute(), a.data_dir.absolute()
    a.smoke = False
    return a


def normalized(path):
    return os.path.normcase(os.path.abspath(os.fspath(path)))


def inside(path, directory):
    return path == directory or path.startswith(directory + os.sep)


class HistoricalResultGuard:
    """Deny opening any other experiment; observe Python open/os.open events."""
    def __init__(self, out):
        self.allowed = normalized(out)
        self.roots = [normalized(ROOT / 'runs'), normalized(ROOT / 'testing_runs')]
        self.unexpected = []
        self.probing = False

    def hook(self, event, args):
        if event not in ('open', 'os.listdir', 'os.scandir') or not args or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = normalized(os.fsdecode(args[0]))
        if any(inside(path, root) for root in self.roots) and not inside(path, self.allowed):
            if not self.probing:
                self.unexpected.append(path)
            raise PermissionError('Independent inference forbids access to historical results: ' + path)

    def install(self):
        sys.addaudithook(self.hook)
        self.probing = True
        try:
            try:
                # A nonexistent probe is rejected by the hook before filesystem lookup.
                with open(str(ROOT / 'runs' / '__guard_probe__' / 'predictions.npz'), 'rb'):
                    pass
            except PermissionError:
                return
            raise AssertionError('Historical-result access guard did not reject probe')
        finally:
            self.probing = False

    def report(self):
        return dict(guard='Python open/listdir/scandir audit hook; not an OS filesystem sandbox', guard_probe_passed=True,
                    blocked_roots=self.roots, allowed_current_output=self.allowed,
                    unexpected_access_attempts=self.unexpected,
                    historical_result_reads_successful=0,
                    no_baseline_argument=True, no_previous_predictions_loaded=True)


class LayerError:
    def __init__(self):
        self.n = 0
        self.abs_sum = self.error2 = self.ref2 = self.quant2 = self.dot = self.maximum = 0.

    def add(self, np, reference, quantized):
        r, q = reference.astype(np.float64), quantized.astype(np.float64)
        e = q - r
        self.n += r.size
        self.abs_sum += float(np.abs(e).sum())
        self.error2 += float(np.square(e).sum())
        self.ref2 += float(np.square(r).sum())
        self.quant2 += float(np.square(q).sum())
        self.dot += float((r * q).sum())
        self.maximum = max(self.maximum, float(np.abs(e).max()))
        axes = tuple(range(1, r.ndim))
        r2, q2, e2 = [np.square(x).sum(axis=axes) for x in (r, q, e)]
        rel = np.divide(np.sqrt(e2), np.sqrt(r2), out=np.full(r.shape[0], np.nan), where=r2 != 0)
        denom = np.sqrt(r2 * q2)
        cosine = np.divide((r * q).sum(axis=axes), denom, out=np.full(r.shape[0], np.nan), where=denom != 0)
        return np.stack([np.abs(e).mean(axis=axes), np.sqrt(np.square(e).mean(axis=axes)), np.abs(e).max(axis=axes), rel, cosine], axis=1)

    def result(self):
        return dict(elements=self.n, mae=self.abs_sum / self.n, rmse=math.sqrt(self.error2 / self.n),
                    max_abs_error=self.maximum, relative_l2=math.sqrt(self.error2 / self.ref2) if self.ref2 else None,
                    cosine=self.dot / math.sqrt(self.ref2 * self.quant2) if self.ref2 * self.quant2 else None,
                    sqnr_db=10 * math.log10(self.ref2 / self.error2) if self.ref2 and self.error2 else None)


def dump_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8-sig') as stream:
        w = csv.DictWriter(stream, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    a = arguments()
    a.out.mkdir(parents=True, exist_ok=False)
    guard = HistoricalResultGuard(a.out)
    guard.install()
    print('OUTPUT_DIRECTORY=' + str(a.out), flush=True)
    start = time.perf_counter()
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(a.repo))
    import numpy as np
    import tensorflow as tf
    import tensorflow_addons as tfa
    from audit_kwt_splits import audit_splits
    from kwt_int16_graph import build_reference, build_quantized
    tf.compat.v1.disable_eager_execution()
    from kws_streaming.models import models as registry
    from kws_streaming.models import model_flags
    from kws_streaming.models.transformer_utils import KWSTransformer
    from kws_streaming.train import base_parser
    from kws_streaming.data import input_data
    if (tf.__version__, np.__version__, tfa.__version__) != ('2.4.0', '1.19.5', '0.12.1'):
        raise RuntimeError('Use the existing pinned .conda environment')
    random.seed(59185)
    np.random.seed(59185)
    tf.compat.v1.set_random_seed(59185)
    model_dir = a.repo / 'models_data_v2_12_labels' / 'kwt1'
    audit = audit_splits(a.data_dir, model_dir / 'flags.json')
    write_json(a.out / 'split_audit.json', audit)
    if not audit.get('passed'):
        raise AssertionError('Dataset split audit failed; see split_audit.json')
    flags = make_flags(a, model_dir, base_parser, registry.kws_transformer, model_flags)
    labels = (model_dir / 'labels.txt').read_text(encoding='utf-8').split()
    if labels != input_data.prepare_words_list(flags.wanted_words.split(','), flags.split_data):
        raise AssertionError('Label order differs from checkpoint')
    write_json(a.out / 'inference_flags.json', vars(flags))
    source = a.out / 'source'
    source.mkdir()
    for name in ('run_kwt_testing_independent.py', 'score_kwt_testing.py', 'audit_kwt_splits.py', 'kwt_int16_graph.py', 'run_kwt_fp32.py'):
        shutil.copyfile(ROOT / name, source / name)
    metadata = dict(split='testing', labels=labels, batch_size=a.batch_size,
                    observed_commit=git_revision(a.repo), python=sys.version, tensorflow=tf.__version__,
                    numpy=np.__version__, tensorflow_addons=tfa.__version__, device='CPU',
                    data_dir=str(a.data_dir), independent_from_previous_predictions=True,
                    calibration_split='training', calibration_samples=a.calibration_samples,
                    checkpoint_sha256={p.name: sha256(p) for p in sorted(model_dir.glob('best_weights.*'))},
                    source_sha256={p.name: sha256(p) for p in source.iterdir()},
                    runtime_note='TF 2.4.0 stable compatibility substitute for upstream rc1',
                    testing_note='Standard held-out split; checkpoint training history has not been independently audited')
    write_json(a.out / 'run_metadata.json', metadata)
    with tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(device_count={'GPU': 0})) as sess:
        tf.compat.v1.keras.backend.set_session(sess)
        tf.keras.backend.set_learning_phase(0)
        processor = input_data.AudioProcessor(flags)
        available = len(processor.data_index['testing'])
        total = min(available, a.limit) if a.limit else available
        candidates = processor.data_index['testing'][:total]
        calibration = processor.data_index['training'][:a.calibration_samples]
        if len(calibration) != a.calibration_samples or total == 0:
            raise ValueError('Insufficient testing/calibration data')

        def manifest(rows):
            return [dict(index=i, file=os.path.relpath(row['file'], str(a.data_dir)), source_word=row['label'],
                         label_id=int(processor.word_to_index[row['label']]),
                         label=labels[int(processor.word_to_index[row['label']])],
                         synthetic_zero_waveform=row['label'] == '_silence_') for i, row in enumerate(rows)]

        testing_manifest = manifest(candidates)
        calibration_manifest = manifest(calibration)
        write_json(a.out / 'testing_manifest.json', testing_manifest)
        write_json(a.out / 'sample_manifest.json', testing_manifest)
        write_json(a.out / 'calibration_manifest.json', calibration_manifest)
        # Verify membership of every real waveform. Synthetic silence uses a
        # training placeholder file which is replaced by zeros by AudioProcessor.
        official_test = set((a.data_dir / 'testing_list.txt').read_text(encoding='utf-8').splitlines())
        official_validation = set((a.data_dir / 'validation_list.txt').read_text(encoding='utf-8').splitlines())
        for row in testing_manifest:
            if not row['synthetic_zero_waveform'] and row['file'].replace('\\', '/') not in official_test:
                raise AssertionError('Real test waveform is not in official testing list')
        for row in calibration_manifest:
            relative = row['file'].replace('\\', '/')
            if not row['synthetic_zero_waveform'] and (relative in official_test or relative in official_validation):
                raise AssertionError('Calibration waveform belongs to held-out data')
        print('Fresh split: {} testing samples; {} training calibration samples'.format(total, len(calibration)), flush=True)
        model = registry.kws_transformer.model(flags)
        kwt = next(layer for layer in model.layers if isinstance(layer, KWSTransformer))
        official = build_probe(tf, model, kwt)
        reference, operands = build_reference(tf, model, kwt)
        sess.run(tf.compat.v1.global_variables_initializer())
        restore = model.load_weights(str(model_dir / 'best_weights'))
        restore.assert_existing_objects_matched()
        restore.expect_partial()
        export_weights(np, sess, kwt, model, a.out)

        def audio_batch(split, offset, count):
            audio, _ = processor.get_data(count, offset, flags, 0., 0., 0, split, 0., 0., sess)
            padded = np.zeros((a.batch_size, 16000), np.float32)
            padded[:count] = np.asarray(audio, np.float32)
            if count < a.batch_size:
                padded[count:] = padded[count - 1]
            return padded

        max_ops = {key: tf.reduce_max(tf.abs(value)) for key, value in operands.items()}
        maxima = {key: 0. for key in operands}
        for offset in range(0, len(calibration), a.batch_size):
            count = min(a.batch_size, len(calibration) - offset)
            observed = sess.run(max_ops, {model.inputs[0]: audio_batch('training', offset, count)})
            for key, value in observed.items():
                if not np.isfinite(value):
                    raise FloatingPointError('Nonfinite calibration operand: ' + key)
                maxima[key] = max(maxima[key], float(value))
            print('Training calibration {}/{}'.format(offset + count, len(calibration)), flush=True)
        fractions = {key: 0 if value == 0 else int(math.floor(math.log2(32767. / (value * (1. if key.endswith('/w') else a.activation_headroom)))))
                     for key, value in maxima.items()}
        write_json(a.out / 'quantization_config.json', dict(calibration_split='training', calibration_samples=len(calibration),
                   activation_headroom=a.activation_headroom, scheme='static symmetric signed int16, power-of-two scale',
                   formula='q=clip(rint(x*2^F),-32767,32767); int64 matmul; FP32 dequantize 2^(-Fleft-Fright)',
                   accumulator_bits=64, quantized_matmuls=98, nonlinear_dtype='float32',
                   operands={key: dict(fractional_bits=fractions[key], scale=2. ** -fractions[key], calibration_max_abs=value,
                                       shape=operands[key].shape.as_list()) for key, value in maxima.items()}))
        print('Building fresh W16A16 graph (several minutes of TensorFlow startup)', flush=True)
        variable_count = len(tf.compat.v1.global_variables())
        hybrid, hybrid_operands, quantized, clips = build_quantized(tf, model, kwt, fractions)
        if len(tf.compat.v1.global_variables()) != variable_count:
            raise AssertionError('Quantized graph introduced new weights')
        integer_ops = [op for op in tf.compat.v1.get_default_graph().get_operations()
                       if op.type in ('MatMul', 'BatchMatMul', 'BatchMatMulV2') and op.outputs[0].dtype == tf.int64]
        if len(integer_ops) != 98:
            raise AssertionError('Expected exactly 98 integer matrix multiplies')
        weights = sess.run({key: value for key, value in quantized.items() if key.endswith('/w')})
        np.savez_compressed(str(a.out / 'weights_int16.npz'), **{key.replace('/', '__'): value for key, value in weights.items()})
        print('Integer graph ready; writing testing outputs', flush=True)
        diagnostics = {key: dict(clipped=value, max_abs=tf.reduce_max(tf.abs(hybrid_operands[key]), axis=list(range(1, hybrid_operands[key].shape.ndims)))) for key, value in clips.items()}
        clipping = {key: dict(operand=key, elements=0, clipped=0, evaluation_max_abs=0., fractional_bits=fractions[key], scale=2. ** -fractions[key]) for key in clips}
        arrays, specs = {}, {}
        errors = {key: LayerError() for key in reference}
        for branch, tensors in [('fp32', reference), ('w16a16', hybrid)]:
            directory = a.out / 'layer_outputs' / branch
            directory.mkdir(parents=True)
            arrays[branch] = {}
            for key, tensor in tensors.items():
                shape = [total] + tensor.shape.as_list()[1:]
                path = directory / (key + '.npy')
                arrays[branch][key] = np.lib.format.open_memmap(str(path), mode='w+', dtype=np.float32, shape=tuple(shape))
                specs[branch + '/' + key] = dict(file=str(path.relative_to(a.out)), shape=shape, dtype='float32')
        write_json(a.out / 'layer_outputs_manifest.json', specs)
        ref_logits, quant_logits = [], []
        consistency = {key: 0. for key in reference if key in official}
        inference_start = time.perf_counter()
        with (a.out / 'per_sample_layer_metrics.csv').open('w', newline='', encoding='utf-8-sig') as stream:
            writer = csv.writer(stream)
            writer.writerow(['sample_index', 'stage', 'mae', 'rmse', 'max_abs_error', 'relative_l2', 'cosine'])
            for offset in range(0, total, a.batch_size):
                count = min(a.batch_size, total - offset)
                audio = audio_batch('testing', offset, count)
                fetched = sess.run(dict(fp32=reference, w16a16=hybrid, diagnostics=diagnostics,
                                        official={key: official[key] for key in consistency}), {model.inputs[0]: audio})
                for key in consistency:
                    r, o = fetched['fp32'][key][:count], fetched['official'][key][:count]
                    consistency[key] = max(consistency[key], float(np.abs(r - o).max()))
                    np.testing.assert_allclose(r, o, rtol=1e-5, atol=1e-5, err_msg=key)
                for key in reference:
                    r, q = fetched['fp32'][key][:count], fetched['w16a16'][key][:count]
                    if not np.isfinite(r).all() or not np.isfinite(q).all():
                        raise FloatingPointError('Nonfinite testing output: ' + key)
                    arrays['fp32'][key][offset:offset + count] = r
                    arrays['w16a16'][key][offset:offset + count] = q
                    for index, metrics in enumerate(errors[key].add(np, r, q)):
                        writer.writerow([offset + index, key] + metrics.tolist())
                for key, observed in fetched['diagnostics'].items():
                    if not np.isfinite(observed['max_abs'][:count]).all():
                        raise FloatingPointError('Nonfinite quantized operand: ' + key)
                    stat = clipping[key]
                    stat['elements'] += count * int(np.prod(hybrid_operands[key].shape.as_list()[1:]))
                    stat['clipped'] += int(observed['clipped'][:count].sum())
                    stat['evaluation_max_abs'] = max(stat['evaluation_max_abs'], float(observed['max_abs'][:count].max()))
                ref_logits.append(fetched['fp32']['logits'][:count])
                quant_logits.append(fetched['w16a16']['logits'][:count])
                if offset == 0:
                    np.savez_compressed(str(a.out / 'first_sample_trace.npz'), waveform=audio[:1],
                                        **{branch + '__' + key: value[:1] for branch in ('fp32', 'w16a16') for key, value in fetched[branch].items()})
                if offset == 0 or (offset // a.batch_size + 1) % 10 == 0 or offset + count == total:
                    print('Testing {}/{}; {:.1f}s'.format(offset + count, total, time.perf_counter() - inference_start), flush=True)
        for branch in arrays.values():
            for array in branch.values():
                array.flush()
                array._mmap.close()
        layer_rows = [dict(stage=key, **stat.result()) for key, stat in errors.items()]
        dump_csv(a.out / 'layer_comparison.csv', layer_rows)
        write_json(a.out / 'layer_comparison.json', layer_rows)
        for stat in clipping.values():
            stat['clipping_rate'] = stat['clipped'] / stat['elements']
        dump_csv(a.out / 'activation_quantization_stats.csv', list(clipping.values()))
        write_json(a.out / 'activation_quantization_stats.json', clipping)
        # This artifact deliberately contains no ground-truth labels.
        np.savez_compressed(str(a.out / 'inference_logits.npz'), fp32_logits=np.concatenate(ref_logits), w16a16_logits=np.concatenate(quant_logits))
        write_json(a.out / 'inference_complete.json', dict(samples=total, available_testing_samples=available,
                   complete_testing_split=total == available, logits_sha256=sha256(a.out / 'inference_logits.npz'),
                   manifest_sha256=sha256(a.out / 'testing_manifest.json'), elapsed_seconds=time.perf_counter() - start,
                   inference_seconds=time.perf_counter() - inference_start,
                   phase='fresh logits committed before separate scoring', contains_ground_truth_labels=False,
                   activation_clipped_elements=sum(s['clipped'] for s in clipping.values()), integer_matmuls=98,
                   fp32_reference_vs_official_max_abs=consistency))
        write_json(a.out / 'file_access_audit.json', guard.report())
        print('Fresh testing logits committed; historical result reads: 0', flush=True)
    if not a.inference_only:
        from score_kwt_testing import score_run
        result = score_run(a.out)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    write_json(a.out / 'file_access_audit.json', guard.report())


if __name__ == '__main__':
    main()
