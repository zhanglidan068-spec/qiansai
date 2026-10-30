"""Compare every exported trainable KWT-1 tensor with its checkpoint value."""
import argparse
import json
import os
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import tensorflow as tf

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--run', type=Path, default=Path('runs/smoke'))
parser.add_argument('--checkpoint', default='keyword-transformer/models_data_v2_12_labels/kwt1/best_weights')
args = parser.parse_args()
prefix = 'layer_with_weights-0/'
mapping = {'pos_emb': prefix + 'pos_emb', 'class_emb': prefix + 'class_emb',
           'patch_kernel': prefix + 'patch_proj/kernel', 'patch_bias': prefix + 'patch_proj/bias',
           'head_kernel': 'layer_with_weights-1/layer_with_weights-0/kernel',
           'head_bias': 'layer_with_weights-1/layer_with_weights-0/bias'}
for index in range(12):
    source = prefix + 'enc_layers/{}/'.format(index)
    target = 'layer_{:02d}_'.format(index + 1)
    for name, relative, has_bias in [
            ('q', 'att/query_dense', False), ('k', 'att/key_dense', False),
            ('v', 'att/value_dense', False), ('o', 'att/combine_heads', True),
            ('fc1', 'ffn/layer_with_weights-0', True), ('fc2', 'ffn/layer_with_weights-1', True)]:
        mapping[target + name + '_kernel'] = source + relative + '/kernel'
        if has_bias:
            mapping[target + name + '_bias'] = source + relative + '/bias'
    for name, relative in [('ln1', 'layernorm1'), ('ln2', 'layernorm2')]:
        for parameter in ['gamma', 'beta']:
            mapping[target + name + '_' + parameter] = source + relative + '/' + parameter
reader = tf.train.load_checkpoint(args.checkpoint)
count = 0
with np.load(str(args.run / 'weights_fp32.npz')) as exported:
    assert set(exported.files) == set(mapping)
    for name, path in mapping.items():
        stored = reader.get_tensor(path + '/.ATTRIBUTES/VARIABLE_VALUE')
        actual = exported[name]
        np.testing.assert_array_equal(actual, stored, err_msg=name)
        count += actual.size
result = {'tensor_count': len(mapping), 'parameter_count': count,
          'all_values_exactly_equal': True, 'max_abs_error': 0.0,
          'note': 'Compared exported in-memory model weights against checkpoint tensors by semantic path.'}
(args.run / 'checkpoint_verification.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps(result, indent=2))
