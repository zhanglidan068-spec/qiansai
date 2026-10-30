"""Second-stage per-output-channel W4/W6 validation search, no new calibration.

Feature caches, frozen weights and training activation ranges come from the
parent preparation. The only new calibration is a max over each weight column;
no validation value or label determines a scale. Historical trials stay intact.
"""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

from run_kwt_fp32 import make_flags, sha256, write_json
from run_kwt_int16 import dump_csv
from area_proxy import precision_accounting

ROOT = Path(__file__).absolute().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, required=True)
    a = parser.parse_args()
    parent = a.parent.absolute()
    run = parent / 'per_channel_search'
    run.mkdir(exist_ok=False)
    print('PER_CHANNEL_DIRECTORY=' + str(run), flush=True)
    meta = json.loads((parent / 'run_metadata.json').read_text(encoding='utf-8'))
    ranges = json.loads((parent / 'calibration_ranges.json').read_text(encoding='utf-8'))['matrix_operands']
    nlconfig = json.loads((parent / 'nonlinear_config.json').read_text(encoding='utf-8'))
    indices = json.loads((parent / 'screening_indices.json').read_text(encoding='utf-8'))
    source = run / 'source'
    source.mkdir()
    for name in ('search_precision_per_channel.py', 'kwt_mixed_pc_graph.py', 'kwt_nonlinear_int16.py',
                 'compact_gelu_int16.py', 'compact_softmax_int16.py', 'area_proxy.py'):
        shutil.copyfile(ROOT / name, source / name)
    write_json(run / 'source_sha256.json', {p.name: sha256(p) for p in source.iterdir()})
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
    sys.dont_write_bytecode = True
    args = argparse.Namespace(repo=ROOT / 'keyword-transformer', data_dir=ROOT / 'data2', out=run, smoke=False, batch_size=meta['batch_size'])
    sys.path.insert(0, str(args.repo))
    import numpy as np
    import tensorflow as tf
    tf.compat.v1.disable_eager_execution()
    tf.compat.v1.disable_control_flow_v2()
    from kws_streaming.models import models as registry
    from kws_streaming.models import model_flags
    from kws_streaming.models.transformer_utils import KWSTransformer
    from kws_streaming.train import base_parser
    from kwt_mixed_pc_graph import build_search_graph
    flags = make_flags(args, args.repo / 'models_data_v2_12_labels/kwt1', base_parser, registry.kws_transformer, model_flags)
    features = np.load(str(parent / 'validation_mfcc.npy'), mmap_mode='r', allow_pickle=False)
    with np.load(str(parent / 'fresh_fp32_reference.npz'), allow_pickle=False) as old:
        baseline, targets = old['logits'].copy(), old['target_ids'].copy()
    n, bs = len(targets), meta['batch_size']
    summaries, configs = [], {}
    with tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(device_count={'GPU': 0})) as sess:
        tf.compat.v1.keras.backend.set_session(sess)
        tf.keras.backend.set_learning_phase(0)
        model = registry.kws_transformer.model(flags)
        kwt = next(x for x in model.layers if isinstance(x, KWSTransformer))
        print('Building per-channel graph', flush=True)
        outputs, controls, records = build_search_graph(tf, model, kwt, ranges, nlconfig)
        sess.run(tf.compat.v1.global_variables_initializer())
        status = model.load_weights(str(args.repo / 'models_data_v2_12_labels/kwt1/best_weights'))
        status.assert_existing_objects_matched()
        status.expect_partial()
        original_weights = sess.run({k: v for k, v in records['operands'].items() if k.endswith('/w')})

        def resolve(p):
            feed = {controls['nonlinear_mode']: 1, controls['compact_gelu']: True}
            for key, control in controls['enabled'].items():
                feed[control] = key.startswith('layer_')
            widths, fractions = {}, {}
            for key, maximum in ranges.items():
                b = p['w'] if key.endswith('/w') else p['a']
                if key.endswith('/av/a'):
                    b = p['p']
                widths[key] = b
                if key.endswith('/w'):
                    maximum = np.abs(original_weights[key]).max(axis=0).astype(np.float64)
                    safe = np.where(maximum == 0, 1., maximum)
                    f = np.floor(np.log2(((1 << (b - 1)) - 1.) / safe)).astype(np.int32)
                    f[maximum == 0] = 0
                elif key.endswith('/av/a'):
                    f = p['pf']
                else:
                    f = 0 if maximum == 0 else int(math.floor(math.log2(((1 << (b - 1)) - 1.) / (maximum * 1.25))))
                fractions[key] = f.tolist() if isinstance(f, np.ndarray) else int(f)
                feed[controls['bits'][key]], feed[controls['F'][key]] = b, f
            return feed, widths, fractions

        def evaluate(p, scope, layers=False):
            config_name = p['name']
            configs[config_name] = p
            directory = run / (config_name + '__' + scope)
            directory.mkdir(exist_ok=False)
            selected = np.arange(n, dtype=np.int64) if scope != 'screen' else np.asarray(indices, np.int64)
            feed, widths, fractions = resolve(p)
            write_json(directory / 'precision_config.json', dict(policy=p, matrix_operand_bits=widths, fractional_bits=fractions,
                       nonlinear=nlconfig, weight_scale_granularity='per output channel', testing_samples_used=0,
                       accelerator_scope='encoder only; FP32 patch/head'))
            np.save(str(directory / 'sample_indices.npy'), selected)
            prediction, mmaps = [], {}
            if layers:
                output_dir = directory / 'layer_outputs'
                output_dir.mkdir()
                for key, value in outputs.items():
                    mmaps[key] = np.lib.format.open_memmap(str(output_dir / (key + '.npy')), mode='w+', dtype=np.float32,
                                    shape=(len(selected),) + tuple(value.shape.as_list()[1:]))
            start = time.perf_counter()
            for offset in range(0, len(selected), bs):
                batch = selected[offset:offset + bs]
                data = np.empty((bs, 98, 40), np.float32)
                data[:len(batch)] = features[batch]
                data[len(batch):] = data[len(batch) - 1]
                feed[kwt.input] = data
                got = sess.run(outputs if layers else outputs['logits'], feed)
                logits = got['logits'] if layers else got
                prediction.append(logits[:len(batch)])
                if layers:
                    for key, array in mmaps.items():
                        array[offset:offset + len(batch)] = got[key][:len(batch)]
                if scope != 'screen' and ((offset // bs + 1) % 30 == 0 or offset + len(batch) == len(selected)):
                    print('{}__{} {}/{}'.format(config_name, scope, offset + len(batch), len(selected)), flush=True)
            for array in mmaps.values():
                array.flush()
                array._mmap.close()
            logits = np.concatenate(prediction)
            if not np.isfinite(logits).all():
                raise FloatingPointError(config_name)
            correct = int((logits.argmax(1) == targets[selected]).sum())
            base_correct = int((baseline[selected].argmax(1) == targets[selected]).sum())
            drop = 100. * (base_correct - correct) / len(selected)
            accounting = precision_accounting(dict(default_weight_bits=p['w'], default_activation_bits=p['a'],
                av_probability_bits=p['p'], av_fractional_bits=p['pf'], matrix_operand_bits=widths))[1]
            result = dict(name=config_name, scope=scope, samples=len(selected), correct=correct, accuracy=correct / float(len(selected)),
                fp32_correct=base_correct, accuracy_drop_pp=drop, passes_2pp=drop <= 2. + 1e-12,
                is_full_validation=scope != 'screen', elapsed_seconds=time.perf_counter() - start,
                matrix_weight_kib=accounting['matrix_weight_kib'], arithmetic_work_proxy=accounting['relative_bit_product_work_vs_w8a8'],
                per_channel_scale_bytes=6912, area_compliance_verified=False, area_mm2=None, testing_samples_used=0)
            np.savez_compressed(str(directory / 'predictions.npz'), logits=logits, target_ids=targets[selected], baseline_logits=baseline[selected])
            write_json(directory / 'result.json', result)
            summaries.append(result)
            dump_csv(run / 'experiment_results.csv', summaries)
            write_json(run / 'experiment_results.json', summaries)
            print('{}__{}: {}/{} = {:.4f}%, drop {:.4f}pp'.format(config_name, scope, correct, len(selected), result['accuracy'] * 100, drop), flush=True)
            return result

        specs = [(4, 8, 8, 7), (4, 8, 10, 9), (4, 10, 10, 9), (4, 12, 10, 9), (4, 16, 12, 11),
                 (6, 8, 8, 7), (6, 8, 10, 9), (6, 10, 10, 9), (6, 12, 10, 9)]
        screened = []
        for w, act, prob, pf in specs:
            p = dict(name='pc_w{}a{}_p{}f{}'.format(w, act, prob, pf), w=w, a=act, p=prob, pf=pf)
            screened.append(evaluate(p, 'screen'))
        chosen = []
        for w in (4, 6):
            choices = [r for r in screened if configs[r['name']]['w'] == w and r['passes_2pp']]
            if choices:
                chosen.append(sorted(choices, key=lambda x: (-x['accuracy'], x['arithmetic_work_proxy']))[0]['name'])
        feasible = [r for r in screened if r['passes_2pp']]
        if feasible:
            efficient = sorted(feasible, key=lambda r: (r['arithmetic_work_proxy'], r['matrix_weight_kib'], -r['accuracy']))[0]['name']
            if efficient not in chosen:
                chosen.append(efficient)
        full = [evaluate(configs[name], 'full') for name in chosen]
        valid = [r for r in full if r['passes_2pp']]
        if not valid:
            raise RuntimeError('No full-validation per-channel candidate qualifies')
        best = sorted(valid, key=lambda x: (-x['accuracy'], x['matrix_weight_kib'], x['arithmetic_work_proxy']))[0]
        economical = sorted(valid, key=lambda x: (x['matrix_weight_kib'], x['arithmetic_work_proxy'], -x['accuracy']))[0]
        finals = {}
        for name in dict.fromkeys([best['name'], economical['name']]):
            p = configs[name]
            final = evaluate(p, 'final', layers=True)
            with np.load(str(run / (name + '__full') / 'predictions.npz')) as old, np.load(str(run / (name + '__final') / 'predictions.npz')) as new:
                np.testing.assert_array_equal(old['logits'], new['logits'])
            feed, widths, ff = resolve(p)
            values = sess.run({key: val for key, val in records['q_operands'].items() if key.startswith('layer_') and key.endswith('/w')}, feed)
            directory = run / (name + '__final')
            blob, fbytes, layout = bytearray(), bytearray(), {}
            for key, q in values.items():
                bits, offset, start_f = widths[key], len(blob), len(fbytes)
                accumulator = used = 0
                for code in q.ravel():
                    accumulator |= (int(code) & ((1 << bits) - 1)) << used
                    used += bits
                    while used >= 8:
                        blob.append(accumulator & 255)
                        accumulator >>= 8
                        used -= 8
                if used:
                    blob.append(accumulator & 255)
                fs = np.asarray(ff[key], np.int32)
                if np.any(fs < -128) or np.any(fs > 127):
                    raise OverflowError('Fractional-bit table exceeds int8')
                fbytes.extend(fs.astype(np.int8).tobytes())
                layout[key] = dict(bits=bits, shape=list(q.shape), byte_offset=offset, byte_length=len(blob)-offset,
                                   f_table_offset=start_f, f_table_length=len(fs), fractional_bits=ff[key])
            (directory / 'weights_packed.bin').write_bytes(blob)
            (directory / 'weight_channel_F_int8.bin').write_bytes(fbytes)
            write_json(directory / 'weights_manifest.json', dict(total_bytes=len(blob), channel_scale_bytes=len(fbytes), tensors=layout,
                       format='packed little-endian signed two-complement; per tensor byte alignment; per-column signed int8 F table'))
            np.savez_compressed(str(directory / 'weights_codes.npz'), **{key.replace('/', '__'): q.astype(np.int8) for key, q in values.items()})
            finals[name] = final
        summary = dict(parent=str(parent), samples=n, screening_samples=len(indices), screened_candidates=len(screened),
                       full_validation_candidates=full, accuracy_winner=best, efficient_winner=economical, finals=finals,
                       weight_scale_granularity='per output channel', channel_F_table_bytes=6912,
                       testing_samples_used=0, synthesized_area_mm2=None, area_compliance_verified=False,
                       note='All scaling uses fixed weight columns and parent training ranges; no validation calibration')
        write_json(run / 'search_result.json', summary)
        write_json(parent / 'per_channel_search_result.json', summary)
        print('PER_CHANNEL_SEARCH_COMPLETE=' + str(run), flush=True)


if __name__ == '__main__':
    main()
