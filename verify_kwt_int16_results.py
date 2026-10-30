"""Read-only persisted-result checks; sample mmap pages, never rerun TensorFlow."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    run = parser.parse_args().run.absolute()
    def load(name):
        return json.loads((run / name).read_text(encoding='utf-8'))
    require((run / 'result.json').is_file(), 'Run is incomplete: missing result.json')
    result, metadata, manifest = load('result.json'), load('run_metadata.json'), load('layer_outputs_manifest.json')
    n = result['samples']
    require(n > 0, 'No evaluated samples')
    indices = sorted({0, min(31, n - 1), n - 1})
    stages = ['patch_output', 'encoder_input'] + ['layer_%02d_output' % i for i in range(1, 13)] + ['logits']
    require(set(manifest) == {b + '/' + s for b in ('fp32', 'w16a16') for s in stages}, 'Manifest stage set')
    require(len(list((run / 'layer_outputs').rglob('*.npy'))) == 30, 'Expected 30 output arrays')
    arrays = {}
    for key, spec in manifest.items():
        path = run / spec['file']
        array = arrays[key] = np.load(str(path), mmap_mode='r', allow_pickle=False)
        tail = [12] if key.endswith('/logits') else [98 if key.endswith('/patch_output') else 99, 64]
        require(list(array.shape) == spec['shape'] == [n] + tail, 'Shape mismatch: ' + key)
        require(str(array.dtype) == spec['dtype'] == 'float32', 'Dtype mismatch: ' + key)
        require(path.stat().st_size == array.offset + array.nbytes, 'File size mismatch: ' + key)
    selected, row_count = {}, 0
    with (run / 'per_sample_layer_metrics.csv').open(newline='', encoding='utf-8-sig') as stream:
        for row in csv.DictReader(stream):
            row_count += 1
            if int(row['sample_index']) in indices:
                selected[(int(row['sample_index']), row['stage'])] = row
    require(row_count == n * len(stages) and len(selected) == len(indices) * len(stages), 'Metrics row coverage')
    for i in indices:
        for stage in stages:
            r, q = [arrays[b + '/' + stage][i].astype(np.float64) for b in ('fp32', 'w16a16')]
            require(np.isfinite(r).all() and np.isfinite(q).all(), 'Nonfinite sampled output')
            e, r2, q2 = q - r, np.square(r).sum(), np.square(q).sum()
            values = [np.abs(e).mean(), np.sqrt(np.square(e).mean()), np.abs(e).max(), np.sqrt(np.square(e).sum() / r2) if r2 else np.nan, (r * q).sum() / np.sqrt(r2 * q2) if r2 * q2 else np.nan]
            columns = ['mae', 'rmse', 'max_abs_error', 'relative_l2', 'cosine']
            np.testing.assert_allclose(values, [float(selected[(i, stage)][k]) for k in columns], rtol=1e-10, atol=1e-12, equal_nan=True)
    with np.load(str(run / 'predictions.npz'), allow_pickle=False) as saved:
        r, q, targets = saved['fp32_logits'], saved['w16a16_logits'], saved['target_ids']
        require(r.shape == q.shape == (n, 12) and targets.shape == (n,), 'Prediction shape')
        require(np.isfinite(r).all() and np.isfinite(q).all(), 'Nonfinite logits')
        for b, logits in [('fp32', r), ('w16a16', q)]:
            np.testing.assert_array_equal(logits, arrays[b + '/logits'])
            confusion = np.zeros((12, 12), np.int64)
            np.add.at(confusion, (targets, logits.argmax(axis=1)), 1)
            np.testing.assert_array_equal(confusion, saved[b + '_confusion_matrix'])
    ri, qi = r.argmax(axis=1), q.argmax(axis=1)
    ro, qo = ri == targets, qi == targets
    counts = dict(fp32_correct=int(ro.sum()), w16a16_correct=int(qo.sum()), changed_predictions=int((ri != qi).sum()), correct_to_wrong=int((ro & ~qo).sum()), wrong_to_correct=int((~ro & qo).sum()))
    require(all(result[k] == v for k, v in counts.items()), 'Result prediction counts')
    np.testing.assert_allclose([result['fp32_accuracy'], result['w16a16_accuracy'], result['prediction_agreement']], [ro.mean(), qo.mean(), (ri == qi).mean()], rtol=0, atol=1e-14)
    with (run / 'predictions.csv').open(newline='', encoding='utf-8-sig') as stream:
        rows = list(csv.DictReader(stream))
    require(len(rows) == n and [x['index'] for x in load('changed_predictions.json')] == np.flatnonzero(ri != qi).tolist(), 'Prediction CSV/changes count')
    for i, row in enumerate(rows):
        require(int(row['index']) == i and row['fp32_prediction'] == metadata['labels'][ri[i]] and row['w16a16_prediction'] == metadata['labels'][qi[i]] and row['target'] == metadata['labels'][targets[i]] and row['fp32_correct'] == str(bool(ro[i])) and row['w16a16_correct'] == str(bool(qo[i])) and row['same_prediction'] == str(bool(ri[i] == qi[i])), 'Prediction CSV mismatch')
    before, after = load('previous_experiment_sha256_before.json'), load('previous_experiment_sha256_after.json')
    baseline = Path(metadata['baseline_run'])
    require(load('sample_manifest.json') == json.loads((baseline / 'sample_manifest.json').read_text(encoding='utf-8'))[:n], 'Validation manifests differ')
    np.testing.assert_array_equal(targets, [x['label_id'] for x in load('sample_manifest.json')])
    actual = {str(p.relative_to(baseline)): digest(p) for p in baseline.rglob('*') if p.is_file()}
    require(before == after == actual, 'Previous experiment hashes differ')
    source = {p.name: digest(p) for p in (run / 'source').iterdir() if p.is_file()}
    require(source == metadata['source_sha256'], 'Source snapshot hashes differ')
    expected = {'patch__w': (40, 64), 'head__w': (64, 12)}
    expected.update({'layer_%02d__%s__w' % (i, op): ((64, 256) if op == 'fc1' else (256, 64) if op == 'fc2' else (64, 64)) for i in range(1, 13) for op in ('q', 'k', 'v', 'o', 'fc1', 'fc2')})
    with np.load(str(run / 'weights_int16.npz'), allow_pickle=False) as weights:
        require(set(weights.files) == set(expected), 'Expected 74 weight matrices')
        for key, shape in expected.items():
            w = weights[key]
            require(w.dtype == np.int16 and w.shape == shape and np.all(w >= -32767), 'Invalid int16 weight: ' + key)
    print(json.dumps(dict(status='passed', run=str(run), samples=n, arrays=30, sampled_indices=indices, sampled_metric_rows=len(selected), weight_matrices=74, baseline_files_hashed=len(actual), source_files_hashed=len(source), counts=counts), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
