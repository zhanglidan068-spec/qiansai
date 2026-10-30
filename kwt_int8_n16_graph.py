"""KWT W8A8 matrix products with signed INT16 nonlinear kernels.

The supplied model owns every trainable variable.  This module builds probes
without changing the official model or initializing/restoring any weights.
MFCC preprocessing, embeddings, bias and residual additions are FP32 at the
interfaces.  All 98 model matrix products accumulate quantized INT8 operands
in INT64.  Softmax, exact-GELU lookup and LayerNorm are delegated to the
integer implementations in ``kwt_nonlinear_int16``.

Matrix operand keys match the existing W16A16 experiment.  QK's /b is K before
transpose.  Softmax probabilities have INT16 F=15, then AV /a has INT8 F=6.
"""

import math
import operator


def _fractional_bit_count(value):
    if isinstance(value, bool):
        raise TypeError("fractional bits must be an integer, not bool")
    try:
        value = operator.index(value)
    except TypeError:
        raise TypeError("fractional bits must be an integer")
    if not -126 <= value <= 126:
        raise ValueError("fractional bits outside normal FP32 power-of-two range")
    return value


def _rounded_int8_units(tf, value, fractional_bits):
    fractional_bits = _fractional_bit_count(fractional_bits)
    return tf.math.rint(tf.cast(value, tf.float32) *
                        tf.constant(math.ldexp(1.0, fractional_bits), tf.float32))


def quantize_int8(tf, value, fractional_bits):
    """Symmetric INT8 [-127,127], power-of-two scale, nearest ties to even."""
    rounded = _rounded_int8_units(tf, value, fractional_bits)
    return tf.cast(tf.clip_by_value(rounded, -127.0, 127.0), tf.int8)


def _existing_layers(model, kwt):
    if not kwt.built or kwt.d_model != 64 or len(kwt.enc_layers) != 12:
        raise ValueError("Expected an already-built KWT-1: 12 layers, width 64")
    if kwt.distill_emb is not None or len(model.outputs) != 1:
        raise ValueError("Expected the non-distilled single-output KWT-1")
    head_container = model.layers[-1]
    if not hasattr(head_container, "layers") or len(head_container.layers) != 1:
        raise ValueError("Expected the official final Sequential with one Dense")
    head = head_container.layers[0]
    dense_layers = [kwt.patch_proj, head]
    for block in kwt.enc_layers:
        if block.att.num_heads != 1 or block.att.projection_dim != 64:
            raise ValueError("Expected one attention head of width 64")
        if len(block.ffn.layers) != 3:
            raise ValueError("Expected Dense/GELU/Dense feed-forward blocks")
        dense_layers.extend([block.att.query_dense, block.att.key_dense,
                             block.att.value_dense, block.att.combine_heads,
                             block.ffn.layers[0], block.ffn.layers[2]])
        if not block.layernorm1.built or not block.layernorm2.built:
            raise ValueError("LayerNorm variables must already exist")
        if abs(float(block.layernorm1.epsilon) - 1e-6) > 1e-15 or \
                abs(float(block.layernorm2.epsilon) - 1e-6) > 1e-15:
            raise ValueError("Expected official LayerNorm epsilon=1e-6")
    if any(not layer.built or not hasattr(layer, "kernel") for layer in dense_layers):
        raise ValueError("Every existing Dense layer must already be built")
    return head


def _build(tf, model, kwt, fractional_bits8, nonlinear_config):
    head = _existing_layers(model, kwt)
    quantized = fractional_bits8 is not None
    outputs, operands, q_operands, clips = {}, {}, {}, {}
    nonlinear_tensors, nonlinear_records = {}, {}
    if quantized:
        from kwt_nonlinear_int16 import fixed_softmax, fixed_gelu, fixed_layer_norm

    def operand(key, tensor):
        tensor = tf.convert_to_tensor(tensor, dtype=tf.float32)
        operands[key] = tensor
        if not quantized:
            return tensor
        if key not in fractional_bits8:
            raise KeyError("Missing INT8 fractional bits for " + key)
        fraction = _fractional_bit_count(fractional_bits8[key])
        if key.endswith("/av/a") and fraction != 6:
            raise ValueError("Softmax to AV INT8 /a must use F=6: " + key)
        rounded = _rounded_int8_units(tf, tensor, fraction)
        result = tf.cast(tf.clip_by_value(rounded, -127.0, 127.0), tf.int8)
        q_operands[key] = result
        if not key.endswith("/w"):
            rank = tensor.shape.rank
            if rank is None or rank < 2:
                raise ValueError("Activation operand requires known rank >=2: " + key)
            clipped = tf.logical_or(rounded < -127.0, rounded > 127.0)
            clips[key] = tf.reduce_sum(tf.cast(clipped, tf.int64),
                                       axis=list(range(1, rank)))
        return result

    def dequantize(accumulator, left_key, right_key):
        exponent = -(_fractional_bit_count(fractional_bits8[left_key]) +
                     _fractional_bit_count(fractional_bits8[right_key]))
        if not -126 <= exponent <= 126:
            raise ValueError("Combined matrix scale outside normal FP32 range")
        return tf.cast(accumulator, tf.float32) * \
            tf.constant(math.ldexp(1.0, exponent), tf.float32)

    def dense(prefix, value, layer):
        left_key, right_key = prefix + "/a", prefix + "/w"
        left, right = operand(left_key, value), operand(right_key, layer.kernel)
        if not quantized:
            return layer(left)
        shape = tf.shape(left)
        flat = tf.reshape(tf.cast(left, tf.int64), [-1, shape[-1]])
        accumulator = tf.matmul(flat, tf.cast(right, tf.int64))
        result = dequantize(accumulator, left_key, right_key)
        result = tf.reshape(result, tf.concat([shape[:-1], [layer.units]], axis=0))
        result.set_shape(value.shape.as_list()[:-1] + [layer.units])
        if layer.use_bias:
            result = tf.nn.bias_add(result, layer.bias)
        return result

    def matmul(prefix, a, b, transpose_b=False):
        left_key, right_key = prefix + "/a", prefix + "/b"
        left, right = operand(left_key, a), operand(right_key, b)
        if not quantized:
            return tf.matmul(left, right, transpose_b=transpose_b)
        accumulator = tf.matmul(tf.cast(left, tf.int64), tf.cast(right, tf.int64),
                                transpose_b=transpose_b)
        return dequantize(accumulator, left_key, right_key)

    def nonlinear(prefix, kind, value, layer=None):
        nonlinear_tensors[prefix + "/input"] = value
        if not quantized:
            if kind == "softmax":
                result = tf.nn.softmax(value, axis=-1)
            else:
                result = layer(value)
        else:
            if prefix not in nonlinear_config:
                raise KeyError("Missing nonlinear configuration for " + prefix)
            config = nonlinear_config[prefix]
            input_f = _fractional_bit_count(config["input_f"])
            output_f = _fractional_bit_count(config["output_f"])
            with tf.name_scope(prefix.replace("/", "_") + "_integer"):
                if kind == "softmax":
                    if output_f != 15:
                        raise ValueError("Signed INT16 Softmax output must use F=15")
                    record = fixed_softmax(tf, value, input_f, output_f)
                elif kind == "gelu":
                    record = fixed_gelu(tf, value, input_f, output_f)
                elif kind in ("ln1", "ln2"):
                    record = fixed_layer_norm(
                        tf, value, input_f, output_f, layer.gamma, layer.beta,
                        epsilon=float(config.get("epsilon", layer.epsilon)),
                        gamma_f=_fractional_bit_count(config.get("gamma_f", 14)),
                        beta_f=_fractional_bit_count(config.get("beta_f", 16)))
                else:
                    raise ValueError("Unsupported nonlinear kernel: " + kind)
            record = dict(record)
            record["pre_input"] = value
            required = ("input_int16", "output_int16", "output_dequant",
                        "input_clipped", "output_clipped")
            if any(key not in record for key in required):
                raise ValueError("Integer nonlinear helper omitted required record keys")
            if record["input_int16"].dtype != tf.int16 or \
                    record["output_int16"].dtype != tf.int16:
                raise TypeError("Nonlinear interfaces must carry signed INT16 codes")
            result = record["output_dequant"]
            result.set_shape(value.shape)
            nonlinear_records[prefix] = record
        nonlinear_tensors[prefix + "/output"] = result
        return result

    def attention(prefix, value, att):
        batch_size = tf.shape(value)[0]
        query = att.separate_heads(dense(prefix + "/q", value, att.query_dense), batch_size)
        key = att.separate_heads(dense(prefix + "/k", value, att.key_dense), batch_size)
        val = att.separate_heads(dense(prefix + "/v", value, att.value_dense), batch_size)
        scores = matmul(prefix + "/qk", query, key, transpose_b=True)
        # Head dimension is exactly 64; multiplying by 1/8 is exact in binary.
        probabilities = nonlinear(prefix + "/softmax", "softmax", scores * (1.0 / 8.0))
        attended = matmul(prefix + "/av", probabilities, val)
        attended = tf.transpose(attended, perm=[0, 2, 1, 3])
        attended = tf.reshape(attended, [batch_size, -1, att.embed_dim])
        return dense(prefix + "/o", attended, att.combine_heads)

    with tf.name_scope("kwt_w8a8_n16" if quantized else "kwt_fp32_nonlinear_reference"):
        value = dense("patch", kwt.input, kwt.patch_proj)
        outputs["patch_output"] = value
        token = tf.broadcast_to(kwt.class_emb, [tf.shape(value)[0], 1, kwt.d_model])
        value = tf.concat([token, value], axis=1) + kwt.pos_emb
        outputs["encoder_input"] = value
        for index, block in enumerate(kwt.enc_layers, 1):
            prefix = "layer_{:02d}".format(index)
            if block.prenorm:
                attn_input = nonlinear(prefix + "/ln1", "ln1", value, block.layernorm1)
                residual = value + attention(prefix, attn_input, block.att)
                ffn_input = nonlinear(prefix + "/ln2", "ln2", residual, block.layernorm2)
            else:
                attended = attention(prefix, value, block.att)
                residual = nonlinear(prefix + "/ln1", "ln1", value + attended, block.layernorm1)
                ffn_input = residual
            hidden = dense(prefix + "/fc1", ffn_input, block.ffn.layers[0])
            hidden = nonlinear(prefix + "/gelu", "gelu", hidden, block.ffn.layers[1])
            projected = dense(prefix + "/fc2", hidden, block.ffn.layers[2])
            if block.prenorm:
                value = residual + projected
            else:
                value = nonlinear(prefix + "/ln2", "ln2", residual + projected, block.layernorm2)
            outputs[prefix + "_output"] = value
        outputs["logits"] = dense("head", value[:, 0, :], head)
    if len(outputs) != 15 or len(operands) != 196 or len(nonlinear_tensors) != 96:
        raise AssertionError("Incomplete KWT stage, matrix-operand or nonlinear tracing")
    return outputs, operands, q_operands, clips, nonlinear_tensors, nonlinear_records


def build_reference(tf, model, kwt):
    """Return 15 FP32 stages, 196 operands and 96 nonlinear boundary tensors."""
    outputs, operands, _, _, nonlinear_tensors, _ = _build(tf, model, kwt, None, None)
    return outputs, operands, nonlinear_tensors


def build_quantized(tf, model, kwt, fractional_bits8, nonlinear_config):
    """Return stages, operands, INT8 codes, clip counts and 48 nonlinear records.

    ``fractional_bits8`` maps every matrix operand to its static F. Every AV /a
    is required to be F=6. ``nonlinear_config`` maps layer_XX/{softmax,ln1,
    gelu,ln2} to {input_f, output_f}; LayerNorm also accepts gamma_f=14,
    beta_f=16 and epsilon=1e-6.  Softmax output_f is required to be 15.

    A nonlinear record includes pre_input, input_int16, output_int16,
    output_dequant, input_clipped[B] and output_clipped[B], with optional
    helper diagnostics.  FP32 conversion is confined to kernel interfaces.
    """
    if fractional_bits8 is None or nonlinear_config is None:
        raise TypeError("Both matrix and nonlinear quantization configurations are required")
    outputs, operands, q_operands, clips, _, records = _build(
        tf, model, kwt, fractional_bits8, nonlinear_config)
    return outputs, operands, q_operands, clips, records
