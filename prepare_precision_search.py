"""Compute fresh train-calibration / validation MFCC caches, never testing data.

Only features are reused across precision trials; predictions are newly computed
by each candidate. No historical prediction file is read by this program.
"""
import argparse
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time

from run_kwt_fp32 import build_probe, export_weights, git_revision, make_flags, sha256, write_json

ROOT = Path(__file__).absolute().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--calibration-samples', type=int, default=256)
    parser.add_argument('--screen-per-class', type=int, default=32)
    a = parser.parse_args()
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S_%f')
    a.out = (a.out or ROOT / 'runs' / ('precision_search_' + stamp)).absolute()
    a.out.mkdir(parents=True, exist_ok=False)
    print('SEARCH_DIRECTORY=' + str(a.out), flush=True)
    start = time.perf_counter()
    a.repo, a.data_dir, a.smoke = ROOT / 'keyword-transformer', ROOT / 'data2', False
    old_list = ROOT / 'runs/validation_full/sample_manifest.json'
    manifest = json.loads(old_list.read_text(encoding='utf-8'))
    write_json(a.out / 'sample_manifest.json', manifest)
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(a.repo))
    import numpy as np
    import tensorflow as tf
    import tensorflow_addons as tfa
    tf.compat.v1.disable_eager_execution()
    from kws_streaming.models import models as registry
    from kws_streaming.models import model_flags
    from kws_streaming.models.transformer_utils import KWSTransformer
    from kws_streaming.train import base_parser
    from kws_streaming.data import input_data
    from kwt_int8_n16_graph import build_reference
    random.seed(59185)
    np.random.seed(59185)
    tf.compat.v1.set_random_seed(59185)
    model_dir = a.repo / 'models_data_v2_12_labels/kwt1'
    flags = make_flags(a, model_dir, base_parser, registry.kws_transformer, model_flags)
    labels = input_data.prepare_words_list(flags.wanted_words.split(','), flags.split_data)
    write_json(a.out / 'inference_flags.json', vars(flags))
    write_json(a.out / 'run_metadata.json', dict(purpose='Mixed precision train-only calibration and validation search',
        labels=labels, batch_size=a.batch_size, calibration_samples=a.calibration_samples,
        tensorflow=tf.__version__, numpy=np.__version__, tensorflow_addons=tfa.__version__, python=sys.version,
        observed_commit=git_revision(a.repo), checkpoint_sha256={p.name: sha256(p) for p in model_dir.glob('best_weights.*')},
        testing_samples_used=0, precision_target_max_drop_pp=2., accelerator_boundary='12 encoder blocks; patch and head FP32',
        area_limit_mm2=2.25, area_compliance_verified=False,
        feature_cache_note='Fresh MFCCs computed from audio once, reused as fixed inputs; candidate predictions recomputed'))
    source = a.out / 'source'
    source.mkdir()
    for name in ('prepare_precision_search.py', 'kwt_int8_n16_graph.py', 'run_kwt_fp32.py'):
        shutil.copyfile(ROOT / name, source / name)
    with tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(device_count={'GPU': 0})) as sess:
        tf.compat.v1.keras.backend.set_session(sess)
        tf.keras.backend.set_learning_phase(0)
        processor = input_data.AudioProcessor(flags)
        for row in manifest:
            if not row['synthetic_zero_waveform'] and input_data.which_set(row['file'], 10, 10) != 'validation':
                raise AssertionError('Validation manifest contains a non-validation waveform')
        processor.data_index['validation'] = [dict(label=r['source_word'], file=str(a.data_dir / r['file'])) for r in manifest]
        calibration = processor.data_index['training'][:a.calibration_samples]
        calibration_manifest = [dict(index=i, file=os.path.relpath(r['file'], str(a.data_dir)), source_word=r['label'],
            label_id=int(processor.word_to_index[r['label']]), synthetic_zero_waveform=r['label'] == '_silence_') for i, r in enumerate(calibration)]
        for row in calibration_manifest:
            if not row['synthetic_zero_waveform'] and input_data.which_set(row['file'], 10, 10) != 'training':
                raise AssertionError('Calibration waveform belongs to held-out data')
        write_json(a.out / 'calibration_manifest.json', calibration_manifest)
        model = registry.kws_transformer.model(flags)
        kwt = next(layer for layer in model.layers if isinstance(layer, KWSTransformer))
        reference, operands, nonlinear = build_reference(tf, model, kwt)
        probe = build_probe(tf, model, kwt)
        sess.run(tf.compat.v1.global_variables_initializer())
        status = model.load_weights(str(model_dir / 'best_weights'))
        status.assert_existing_objects_matched()
        status.expect_partial()
        export_weights(np, sess, kwt, model, a.out)
        range_ops = {key: tf.reduce_max(tf.abs(value)) for key, value in operands.items()}
        nl_ops = {key: tf.reduce_max(tf.abs(value)) for key, value in nonlinear.items()}
        maxima, nl_maxima = {k: 0. for k in operands}, {k: 0. for k in nonlinear}
        requests = dict(training=0, validation=0, testing=0)

        def get_batch(split, offset, count):
            if split not in ('training', 'validation'):
                raise RuntimeError('Testing samples are forbidden')
            requests[split] += count
            audio, targets = processor.get_data(count, offset, flags, 0., 0., 0, split, 0., 0., sess)
            padded = np.zeros((a.batch_size, 16000), np.float32)
            padded[:count] = np.asarray(audio, np.float32)
            if count < a.batch_size:
                padded[count:] = padded[count - 1]
            return padded, np.asarray(targets, np.int64)

        training_features = np.lib.format.open_memmap(str(a.out / 'calibration_mfcc.npy'), mode='w+', dtype=np.float32, shape=(len(calibration), 98, 40))
        for offset in range(0, len(calibration), a.batch_size):
            count = min(a.batch_size, len(calibration) - offset)
            audio, _ = get_batch('training', offset, count)
            found = sess.run(dict(mfcc=kwt.input, ranges=range_ops, nonlinear=nl_ops), {model.inputs[0]: audio})
            training_features[offset:offset + count] = found['mfcc'][:count]
            for output, target in [(found['ranges'], maxima), (found['nonlinear'], nl_maxima)]:
                for key, value in output.items():
                    if not np.isfinite(value):
                        raise FloatingPointError('Nonfinite calibration value: ' + key)
                    target[key] = max(target[key], float(value))
            print('Calibration feature cache {}/{}'.format(offset + count, len(calibration)), flush=True)
        training_features.flush()
        training_features._mmap.close()
        write_json(a.out / 'calibration_ranges.json', dict(matrix_operands=maxima, nonlinear=nl_maxima,
                   calibration_split='training', samples=len(calibration), activation_headroom=1.25))
        n = len(manifest)
        features = np.lib.format.open_memmap(str(a.out / 'validation_mfcc.npy'), mode='w+', dtype=np.float32, shape=(n, 98, 40))
        all_logits, targets_all = [], []
        max_error = 0.
        for offset in range(0, n, a.batch_size):
            count = min(a.batch_size, n - offset)
            audio, targets = get_batch('validation', offset, count)
            observed = sess.run(dict(mfcc=kwt.input, ref=reference['logits'], official=model.outputs[0]), {model.inputs[0]: audio})
            np.testing.assert_array_equal(targets, [r['label_id'] for r in manifest[offset:offset + count]])
            np.testing.assert_allclose(observed['ref'], observed['official'], rtol=1e-5, atol=1e-5)
            max_error = max(max_error, float(np.abs(observed['ref'] - observed['official']).max()))
            features[offset:offset + count] = observed['mfcc'][:count]
            all_logits.append(observed['official'][:count])
            targets_all.append(targets)
            if offset == 0 or (offset // a.batch_size + 1) % 20 == 0 or offset + count == n:
                print('Validation feature cache {}/{}'.format(offset + count, n), flush=True)
        features.flush()
        features._mmap.close()
        logits, targets = np.concatenate(all_logits), np.concatenate(targets_all)
        np.savez_compressed(str(a.out / 'fresh_fp32_reference.npz'), logits=logits, target_ids=targets)
        rng = np.random.RandomState(20261028)
        selected = []
        for class_id in range(len(labels)):
            indices = np.flatnonzero(targets == class_id)
            rng.shuffle(indices)
            selected.extend(indices[:a.screen_per_class].tolist())
        selected.sort()
        write_json(a.out / 'screening_indices.json', selected)
        minimum_correct = int(np.ceil(int((logits.argmax(1) == targets).sum()) - .02 * n - 1e-10))
        write_json(a.out / 'preparation_complete.json', dict(samples=n, calibration_samples=len(calibration),
            screening_samples=len(selected), fp32_correct=int((logits.argmax(1) == targets).sum()), fp32_accuracy=float((logits.argmax(1) == targets).mean()),
            min_correct_for_2pp_drop=minimum_correct, official_reference_max_abs_error=max_error,
            sample_requests=requests, elapsed_seconds=time.perf_counter() - start,
            feature_sha256=sha256(a.out / 'validation_mfcc.npy'), calibration_ranges_sha256=sha256(a.out / 'calibration_ranges.json')))
        print('PREPARATION_COMPLETE=' + str(a.out), flush=True)


if __name__ == '__main__':
    main()
