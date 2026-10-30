"""Layer-wise mixed precision search using TRAIN calibration and VALIDATION only.

Requires prepare_precision_search.py output. MFCC input caches are fixed acoustic
features; every trial computes its logits anew. No testing audio is used.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

from run_kwt_fp32 import make_flags, sha256, write_json
from run_kwt_int16 import ErrorAccumulator, dump_csv
from area_proxy import precision_accounting

ROOT = Path(__file__).absolute().parent


def fraction(maximum, bits, headroom):
    return 0 if maximum == 0 else int(math.floor(math.log2(((1 << (bits - 1)) - 1.) / (maximum * headroom))))


def policy(name, w=8, a=8, p=10, f=None, **extra):
    return dict(name=name, default_weight_bits=w, default_activation_bits=a, av_probability_bits=p,
                av_fractional_bits=(p - 1 if f is None else f), nonlinear_mode=1,
                compact_nonlinear=True, software_boundaries=True, layer_overrides={},
                matrix_operand_bits={}, **extra)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--max-full', type=int, default=7)
    a = parser.parse_args()
    run = a.run.absolute()
    prep = json.loads((run / 'preparation_complete.json').read_text(encoding='utf-8'))
    ranges = json.loads((run / 'calibration_ranges.json').read_text(encoding='utf-8'))
    metadata = json.loads((run / 'run_metadata.json').read_text(encoding='utf-8'))
    manifest = json.loads((run / 'sample_manifest.json').read_text(encoding='utf-8'))
    selected = json.loads((run / 'screening_indices.json').read_text(encoding='utf-8'))
    source = run / 'search_source'
    source.mkdir(exist_ok=True)
    for name in ('search_kwt_precision.py', 'kwt_mixed_graph.py', 'kwt_nonlinear_int16.py',
                 'compact_gelu_int16.py', 'compact_softmax_int16.py', 'area_proxy.py'):
        shutil.copyfile(ROOT / name, source / name)
    write_json(run / 'search_source_sha256.json', {p.name: sha256(p) for p in source.iterdir()})
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
    sys.dont_write_bytecode = True
    args = argparse.Namespace(repo=ROOT / 'keyword-transformer', data_dir=ROOT / 'data2',
                              smoke=False, out=run, batch_size=metadata['batch_size'])
    sys.path.insert(0, str(args.repo))
    import numpy as np
    import tensorflow as tf
    tf.compat.v1.disable_eager_execution()
    tf.compat.v1.disable_control_flow_v2()
    from kws_streaming.models import models as registry
    from kws_streaming.models import model_flags
    from kws_streaming.models.transformer_utils import KWSTransformer
    from kws_streaming.train import base_parser
    from kwt_mixed_graph import build_search_graph
    flags = make_flags(args, args.repo / 'models_data_v2_12_labels/kwt1', base_parser, registry.kws_transformer, model_flags)
    nonlinear = {}
    for i in range(1, 13):
        for kind in ('softmax', 'ln1', 'gelu', 'ln2'):
            key = 'layer_{:02d}/{}'.format(i, kind)
            nonlinear[key] = dict(input_f=fraction(ranges['nonlinear'][key + '/input'], 16, 1.25),
                                 output_f=15 if kind == 'softmax' else fraction(ranges['nonlinear'][key + '/output'], 16, 1.25),
                                 gamma_f=14, beta_f=16, epsilon=1e-6)
    write_json(run / 'nonlinear_config.json', nonlinear)
    features = np.load(str(run / 'validation_mfcc.npy'), mmap_mode='r', allow_pickle=False)
    with np.load(str(run / 'fresh_fp32_reference.npz'), allow_pickle=False) as saved:
        baseline_logits, targets = saved['logits'].copy(), saved['target_ids'].copy()
    n, batch_size = len(targets), metadata['batch_size']
    if features.shape != (n, 98, 40) or n != len(manifest):
        raise ValueError('Feature/manifest shape mismatch')
    trials = run / 'trials'
    trials.mkdir(exist_ok=True)
    results, configs = [], {}
    start = time.perf_counter()
    with tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(device_count={'GPU': 0})) as sess:
        tf.compat.v1.keras.backend.set_session(sess)
        tf.keras.backend.set_learning_phase(0)
        model = registry.kws_transformer.model(flags)
        kwt = next(layer for layer in model.layers if isinstance(layer, KWSTransformer))
        print('Building one reusable mixed-precision search graph', flush=True)
        outputs, controls, records = build_search_graph(tf, model, kwt, ranges['matrix_operands'], nonlinear)
        sess.run(tf.compat.v1.global_variables_initializer())
        status = model.load_weights(str(args.repo / 'models_data_v2_12_labels/kwt1/best_weights'))
        status.assert_existing_objects_matched()
        status.expect_partial()
        print('Search graph ready; all trials reuse fixed training calibration ranges', flush=True)

        def resolve(p):
            widths, fractions = {}, {}
            for key, maximum in ranges['matrix_operands'].items():
                stage = key.split('/')[0]
                local = p.get('layer_overrides', {}).get(stage, {})
                bits = int(local.get('w', p['default_weight_bits']) if key.endswith('/w') else local.get('a', p['default_activation_bits']))
                if key.endswith('/av/a'):
                    bits = int(local.get('p', p['av_probability_bits']))
                bits = int(p.get('matrix_operand_bits', {}).get(key, bits))
                widths[key] = bits
                fractions[key] = int(local.get('pf', p['av_fractional_bits'])) if key.endswith('/av/a') else fraction(maximum, bits, 1. if key.endswith('/w') else 1.25)
            return widths, fractions

        def feed_policy(p):
            feed = {controls['nonlinear_mode']: int(p.get('nonlinear_mode', 1)),
                    controls['compact_gelu']: bool(p.get('compact_nonlinear', True))}
            for stage, control in controls['enabled'].items():
                enabled = stage.startswith('layer_')
                if not p.get('software_boundaries', True) and stage in ('patch', 'head'):
                    enabled = True
                if p.get('only_layer'):
                    enabled = stage == p['only_layer']
                if p.get('all_fp32'):
                    enabled = False
                feed[control] = enabled
            for op, control in controls.get('op_enabled', {}).items():
                feed[control] = op not in p.get('fp32_operations', [])
            widths, fractions = resolve(p)
            for key in widths:
                feed[controls['bits'][key]] = widths[key]
                feed[controls['F'][key]] = fractions[key]
            return feed, widths, fractions

        def feature_batch(indices):
            data = np.asarray(features[indices], np.float32)
            padded = np.empty((batch_size, 98, 40), np.float32)
            padded[:len(data)] = data
            padded[len(data):] = data[-1]
            return padded

        def accounting(p):
            widths, _ = resolve(p)
            return precision_accounting(dict(p, matrix_operand_bits=widths))[1]

        def evaluate(p, scope='screen', save_layers=False):
            configs[p['name']] = p
            ident = p['name'] + '__' + scope
            directory = trials / ident
            directory.mkdir(exist_ok=True)
            result_path = directory / 'result.json'
            if result_path.is_file() and not save_layers:
                cached = json.loads(result_path.read_text(encoding='utf-8'))
                if json.loads((directory / 'policy.json').read_text(encoding='utf-8')) != p:
                    raise RuntimeError('Existing trial name has a different policy')
                if not any(r['trial'] == ident for r in results):
                    results.append(cached)
                print('Resume completed ' + ident, flush=True)
                return cached
            indices = np.arange(n, dtype=np.int64) if scope in ('full', 'final') else np.asarray(selected, np.int64)
            feed, widths, fractions = feed_policy(p)
            write_json(directory / 'policy.json', p)
            write_json(directory / 'quantization.json', dict(operand_bits=widths, fractional_bits=fractions,
                       nonlinear=nonlinear, software_boundaries=p.get('software_boundaries', True),
                       floating_operations=['patch', 'classification_head', 'bias/residual/embedding transport'],
                       testing_samples_used=0))
            np.save(str(directory / 'sample_indices.npy'), indices)
            started = time.perf_counter()
            predictions, accumulated = [], {}
            layers = {}
            if save_layers:
                layer_dir = directory / 'layer_outputs'
                layer_dir.mkdir(exist_ok=False)
                for key, tensor in outputs.items():
                    shape = (len(indices),) + tuple(tensor.shape.as_list()[1:])
                    layers[key] = np.lib.format.open_memmap(str(layer_dir / (key + '.npy')), mode='w+', dtype=np.float32, shape=shape)
            for offset in range(0, len(indices), batch_size):
                batch_indices = indices[offset:offset + batch_size]
                feed[kwt.input] = feature_batch(batch_indices)
                if save_layers:
                    got = sess.run(outputs, feed)
                    logits = got['logits'][:len(batch_indices)]
                    for key, array in layers.items():
                        array[offset:offset + len(batch_indices)] = got[key][:len(batch_indices)]
                else:
                    logits = sess.run(outputs['logits'], feed)[:len(batch_indices)]
                if not np.isfinite(logits).all():
                    raise FloatingPointError('Nonfinite prediction ' + ident)
                predictions.append(logits)
                if scope != 'screen' and ((offset // batch_size + 1) % 30 == 0 or offset + len(batch_indices) == len(indices)):
                    print('{} {}/{}'.format(ident, offset + len(batch_indices), len(indices)), flush=True)
            for array in layers.values():
                array.flush()
                array._mmap.close()
            logits = np.concatenate(predictions)
            true = targets[indices]
            correct = int((logits.argmax(1) == true).sum())
            base_correct = int((baseline_logits[indices].argmax(1) == true).sum())
            drop = 100. * (base_correct - correct) / len(indices)
            stats = accounting(p)
            result = dict(trial=ident, name=p['name'], scope=scope, samples=len(indices), correct=correct,
                          accuracy=correct / float(len(indices)), fp32_correct=base_correct,
                          accuracy_drop_pp=drop, passes_2pp=drop <= 2. + 1e-12,
                          is_full_validation=scope in ('full', 'final'),
                          elapsed_seconds=time.perf_counter() - started,
                          matrix_weight_kib=stats['matrix_weight_kib'], arithmetic_work_proxy=stats['relative_bit_product_work_vs_w8a8'],
                          shared8bit_pass_proxy=stats['relative_ideal_8bit_chunk_mac_work'],
                          area_mm2=None, area_compliance_verified=False, testing_samples_used=0)
            np.savez_compressed(str(directory / 'predictions.npz'), logits=logits, target_ids=true, baseline_logits=baseline_logits[indices])
            write_json(directory / 'area_accounting.json', stats)
            write_json(result_path, result)
            results.append(result)
            dump_csv(run / 'experiment_results.csv', results)
            write_json(run / 'experiment_results.json', results)
            print('{}: {}/{} = {:.4f}%, drop {:.4f}pp; {:.1f}s'.format(ident, correct, len(indices), 100 * result['accuracy'], drop, result['elapsed_seconds']), flush=True)
            return result

        # First prove every FP32 bypass is equivalent to the official model on
        # the fixed features. This is a control, not a historical-answer replay.
        control = policy('fp32_bypass_control', all_fp32=True)
        evaluate(control)
        with np.load(str(trials / 'fp32_bypass_control__screen/predictions.npz')) as check:
            np.testing.assert_allclose(check['logits'], check['baseline_logits'], rtol=1e-5, atol=1e-5)

        # Layer-wise experiments: all other blocks remain genuine FP32.
        sensitivities = []
        for bits in (8, 10, 12, 16):
            for i in range(1, 13):
                stage = 'layer_{:02d}'.format(i)
                local = policy('single_{}_b{}'.format(stage, bits), w=bits, a=bits, p=bits,
                               f=6 if bits == 8 else bits - 1, only_layer=stage)
                row = evaluate(local)
                sensitivities.append(dict(layer=i, effective_bits=bits, **row))
                dump_csv(run / 'layer_sensitivity.csv', sensitivities)

        candidates = [policy('encoder_w8a8_p8f6_full_lut', p=8, f=6),
                      policy('encoder_w8a8_p8f6_compact', p=8, f=6),
                      policy('encoder_w16a16_p16', w=16, a=16, p=16)]
        candidates[0]['compact_nonlinear'] = False
        for abits, probability_options in [(8, [(8, 7), (10, 9), (12, 11), (16, 15)]),
                                           (10, [(8, 6), (8, 7), (10, 9), (12, 11)]),
                                           (12, [(8, 6), (8, 7), (10, 9), (12, 11)]),
                                           (16, [(8, 6), (10, 9), (16, 15)])]:
            for pbits, pf in probability_options:
                candidates.append(policy('encoder_w8a{}_p{}f{}'.format(abits, pbits, pf), w=8, a=abits, p=pbits, f=pf))
        candidates += [policy('encoder_w6a12_p10', w=6, a=12, p=10),
                       policy('encoder_w6a12_p12', w=6, a=12, p=12),
                       policy('encoder_w6a16_p12', w=6, a=16, p=12),
                       policy('encoder_w4a16_p16', w=4, a=16, p=16)]
        screened = []
        for p in candidates:
            screened.append(evaluate(p))
        # Sensitive layers receive additional bits; weights can stay at 8 bits.
        ranking = sorted([r for r in sensitivities if r['effective_bits'] == 8], key=lambda r: (-r['accuracy_drop_pp'], r['layer']))
        write_json(run / 'layer_sensitivity_ranking.json', ranking)
        for count in (1, 2, 3, 4, 6):
            p = policy('mixed_w8a8_top{}_a12_p10'.format(count), w=8, a=8, p=10)
            for row in ranking[:count]:
                p['layer_overrides']['layer_{:02d}'.format(row['layer'])] = dict(a=12, p=10, pf=9)
            screened.append(evaluate(p))
        for count in (1, 2, 3):
            p = policy('mixed_w6a12_top{}_w8_p12'.format(count), w=6, a=12, p=12)
            for row in ranking[:count]:
                p['layer_overrides']['layer_{:02d}'.format(row['layer'])] = dict(w=8)
            screened.append(evaluate(p))
        feasible = [r for r in screened if r['passes_2pp']]
        if not feasible:
            raise RuntimeError('No screening candidate meets target; inspect saved experiments')
        # Full verification covers the safest, most accurate, cheapest and F6
        # candidates. Screen results never count as full-validation proof.
        chosen = ['encoder_w16a16_p16']
        orders = [sorted(feasible, key=lambda r: (-r['accuracy'], r['matrix_weight_kib'], r['arithmetic_work_proxy'])),
                  sorted(feasible, key=lambda r: (r['matrix_weight_kib'], r['arithmetic_work_proxy'], -r['accuracy'])),
                  sorted(feasible, key=lambda r: (r['arithmetic_work_proxy'], r['matrix_weight_kib'], -r['accuracy']))]
        strict_f6 = [r for r in feasible if configs[r['name']]['av_fractional_bits'] == 6]
        if strict_f6:
            chosen.append(sorted(strict_f6, key=lambda r: (-r['accuracy'], r['arithmetic_work_proxy']))[0]['name'])
        for order in orders:
            for row in order[:2]:
                if row['name'] not in chosen:
                    chosen.append(row['name'])
                if len(chosen) >= a.max_full:
                    break
            if len(chosen) >= a.max_full:
                break
        full_results = [evaluate(configs[name], 'full') for name in chosen[:a.max_full]]
        valid = [r for r in full_results if r['passes_2pp']]
        if not valid:
            raise RuntimeError('Full validation found no passing candidate; all results preserved')
        # Report both the accuracy-oriented and storage/work-oriented winners.
        accuracy_winner = sorted(valid, key=lambda r: (-r['accuracy'], r['matrix_weight_kib'], r['arithmetic_work_proxy']))[0]
        efficient_winner = sorted(valid, key=lambda r: (r['matrix_weight_kib'], r['arithmetic_work_proxy'], -r['accuracy']))[0]
        selected_policy = configs[accuracy_winner['name']]
        final_result = evaluate(selected_policy, 'final', save_layers=True)
        full_prediction_path = trials / (accuracy_winner['name'] + '__full') / 'predictions.npz'
        with np.load(str(full_prediction_path)) as old, np.load(str(trials / (accuracy_winner['name'] + '__final') / 'predictions.npz')) as new:
            np.testing.assert_array_equal(old['logits'], new['logits'])
        feed, widths, fractions = feed_policy(selected_policy)
        quantized_weights = sess.run({key: val for key, val in records['q_operands'].items() if key.endswith('/w')}, feed)
        np.savez_compressed(str(run / 'best_weights_mixed.npz'), **{key.replace('/', '__'): value.astype(np.int8 if widths[key] <= 8 else np.int16) for key, value in quantized_weights.items() if key.startswith('layer_')})
        write_json(run / 'best_precision_config.json', dict(policy=selected_policy, operand_bits=widths, fractional_bits=fractions,
                   nonlinear=nonlinear, accelerator_scope='encoder only; patch and head FP32',
                   area_compliance_verified=False, required_area_limit_mm2=2.25))
        if efficient_winner['name'] != accuracy_winner['name']:
            efficient_final = evaluate(configs[efficient_winner['name']], 'final', save_layers=True)
        else:
            efficient_final = final_result
        exports = {}
        for role, selected_row in [('accuracy', accuracy_winner), ('efficiency', efficient_winner)]:
            p = configs[selected_row['name']]
            feed, wb, ff = feed_policy(p)
            weights = sess.run({k: v for k, v in records['q_operands'].items() if k.startswith('layer_') and k.endswith('/w')}, feed)
            buffer = bytearray()
            specs = {}
            for key, value in weights.items():
                bits = wb[key]
                offset = len(buffer)
                accumulator = used = 0
                for number in value.ravel():
                    accumulator |= (int(number) & ((1 << bits) - 1)) << used
                    used += bits
                    while used >= 8:
                        buffer.append(accumulator & 255)
                        accumulator >>= 8
                        used -= 8
                if used:
                    buffer.append(accumulator & 255)
                specs[key] = dict(byte_offset=offset, byte_length=len(buffer) - offset, bits=bits, fractional_bits=ff[key], shape=list(value.shape))
            export_dir = run / (role + '_export')
            export_dir.mkdir(exist_ok=False)
            (export_dir / 'weights_packed.bin').write_bytes(buffer)
            write_json(export_dir / 'weights_manifest.json', dict(format='little-endian packed two-complement signed integers, each tensor byte-aligned',
                       total_bytes=len(buffer), tensors=specs, policy=p, matrix_operand_bits=wb, fractional_bits=ff))
            exports[role] = dict(directory=str(export_dir.relative_to(run)), packed_weight_bytes=len(buffer), policy=p)
        summary = dict(samples=n, fp32_correct=prep['fp32_correct'], fp32_accuracy=prep['fp32_accuracy'],
                       minimum_correct=prep['min_correct_for_2pp_drop'], screening_samples=len(selected),
                       layer_sensitivity_experiments=len(sensitivities), screened_candidates=len(screened),
                       full_validation_candidates=full_results, accuracy_winner=accuracy_winner,
                       efficient_winner=efficient_winner, selected_final=final_result,
                       efficient_final=efficient_final, hardware_weight_exports=exports,
                       search_optimum_scope='Best among evaluated candidates; no exhaustive global optimum claim',
                       synthesized_area_mm2=None, area_compliance_verified=False, missing_area_evidence=['official cell library', 'official SRAM macro area table', 'accelerator RTL and frozen mapped netlist'],
                       testing_samples_used=0, elapsed_seconds=time.perf_counter() - start)
        write_json(run / 'search_result.json', summary)
        print('SEARCH_COMPLETE=' + str(run), flush=True)


if __name__ == '__main__':
    main()
