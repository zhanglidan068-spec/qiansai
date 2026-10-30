"""Read-only FP32 reference and INT16/INT64-matmul graphs for KWT-1.

The supplied official model must already be built.  Both builders reuse its
existing variables; neither builder creates or initializes model weights.
Speech features are taken from ``kwt.input``, so the caller feeds the original
model input and retains the original, deterministic FP32 preprocessing graph.

Dense operands retain their natural shape, normally [B, N, C].  QK and AV
operands retain the official [B, heads, N, C] attention layout.  In particular,
``layer_XX/qk/b`` is K before its transpose; QK uses ``transpose_b=True``.
"""

import math
import operator


def _fractional_bit_count(value):
    """Accept Python/NumPy integral values without silently truncating floats."""
    if isinstance(value, bool):
        raise TypeError("fractional_bits must be an integer, not bool")
    try:
        result = operator.index(value)
    except TypeError:
        raise TypeError("fractional_bits must be an integer")
    # Every individual FP32 power-of-two scale must be finite and nonzero.
    if not -126 <= result <= 126:
        raise ValueError("fractional_bits must be between -126 and 126")
    return result


def _rounded_int16_units(tf, x, fractional_bits):
    fractional_bits = _fractional_bit_count(fractional_bits)
    multiplier = tf.constant(math.ldexp(1.0, fractional_bits), dtype=tf.float32)
    # TensorFlow rint uses round-to-nearest, ties-to-even.
    return tf.math.rint(tf.cast(x, tf.float32) * multiplier)


def _clip_cast_int16(tf, rounded):
    return tf.cast(tf.clip_by_value(rounded, -32767.0, 32767.0), tf.int16)


def quantize_int16(tf, x, fractional_bits):
    """Return ties-to-even, symmetric saturated INT16 values, scale=2**(-F).

    ``-32768`` is deliberately unused: both endpoints have magnitude 32767.
    This helper creates no variables and works in TF 2.4 graph or eager mode.
    """
    return _clip_cast_int16(tf, _rounded_int16_units(tf, x, fractional_bits))


def _existing_layers(model, kwt):
    """Validate the supported official, non-distilled single-head KWT-1."""
    if not kwt.built:
        raise ValueError("Build the official KWT model before creating probes")
    if kwt.d_model != 64 or len(kwt.enc_layers) != 12:
        raise ValueError("Expected KWT-1 with 12 encoder layers and width 64")
    if kwt.distill_emb is not None:
        raise ValueError("The INT16 probe requires the non-distilled KWT-1")
    if len(model.outputs) != 1:
        raise ValueError("Expected one classification output")
    head_container = model.layers[-1]
    if not hasattr(head_container, "layers") or len(head_container.layers) != 1:
        raise ValueError("Expected the official final Sequential with one Dense")
    head = head_container.layers[0]
    dense_layers = [kwt.patch_proj, head]
    for block in kwt.enc_layers:
        if block.att.num_heads != 1 or block.att.projection_dim != 64:
            raise ValueError("Expected the official single attention head of width 64")
        if len(block.ffn.layers) != 3:
            raise ValueError("Expected Dense/GELU/Dense feed-forward layers")
        dense_layers.extend([
            block.att.query_dense, block.att.key_dense, block.att.value_dense,
            block.att.combine_heads, block.ffn.layers[0], block.ffn.layers[2]])
        if not block.layernorm1.built or not block.layernorm2.built:
            raise ValueError("LayerNormalization weights must already exist")
    if any(not layer.built or not hasattr(layer, "kernel") for layer in dense_layers):
        raise ValueError("All Dense kernels must already exist")
    return head


def _build(tf, model, kwt, fractional_bits):
    head = _existing_layers(model, kwt)
    outputs = {}
    operands = {}
    quantized_operands = {}
    clipping_counts = {}
    quantized = fractional_bits is not None

    def operand(key, tensor):
        # Explicit Tensor views of existing variables keep weight ownership in
        # the official model and guarantee every public FP32 operand is a Tensor.
        tensor = tf.convert_to_tensor(tensor, dtype=tf.float32)
        operands[key] = tensor
        if not quantized:
            return tensor
        if key not in fractional_bits:
            raise KeyError("Missing fractional_bits for operand " + key)
        rounded = _rounded_int16_units(tf, tensor, fractional_bits[key])
        result = _clip_cast_int16(tf, rounded)
        quantized_operands[key] = result
        if not key.endswith("/w"):
            clipped = tf.logical_or(rounded < -32767.0, rounded > 32767.0)
            rank = tensor.shape.rank
            if rank is None or rank < 2:
                raise ValueError("Activation operand must have known rank >= 2: " + key)
            clipping_counts[key] = tf.reduce_sum(
                tf.cast(clipped, tf.int64), axis=list(range(1, rank)))
        return result

    def dequantize(accumulator, left_key, right_key):
        exponent = -(_fractional_bit_count(fractional_bits[left_key]) +
                     _fractional_bit_count(fractional_bits[right_key]))
        # Keep all non-integer arithmetic, including dequantization, in FP32.
        if not -126 <= exponent <= 126:
            raise ValueError("Combined operand scale is outside normal FP32 range: " +
                             left_key + ", " + right_key)
        scale = tf.constant(math.ldexp(1.0, exponent), dtype=tf.float32)
        return tf.cast(accumulator, tf.float32) * scale

    def dense(prefix, x, layer):
        left_key, right_key = prefix + "/a", prefix + "/w"
        left = operand(left_key, x)
        right = operand(right_key, layer.kernel)
        if not quantized:
            # The reference uses the original Dense implementation, including
            # its own FP32 kernel, bias and exact rank handling.
            return layer(left)
        shape = tf.shape(left)
        left_2d = tf.reshape(tf.cast(left, tf.int64), [-1, shape[-1]])
        accumulator = tf.matmul(left_2d, tf.cast(right, tf.int64))
        y = dequantize(accumulator, left_key, right_key)
        output_shape = tf.concat([shape[:-1], [layer.units]], axis=0)
        y = tf.reshape(y, output_shape)
        y.set_shape(x.shape.as_list()[:-1] + [layer.units])
        if layer.use_bias:
            y = tf.nn.bias_add(y, layer.bias)
        return y

    def matmul(prefix, a, b, transpose_b=False):
        left_key, right_key = prefix + "/a", prefix + "/b"
        left = operand(left_key, a)
        right = operand(right_key, b)
        if not quantized:
            return tf.matmul(left, right, transpose_b=transpose_b)
        accumulator = tf.matmul(tf.cast(left, tf.int64), tf.cast(right, tf.int64),
                                transpose_b=transpose_b)
        return dequantize(accumulator, left_key, right_key)

    def attention(prefix, x, att):
        batch_size = tf.shape(x)[0]
        query = dense(prefix + "/q", x, att.query_dense)
        key = dense(prefix + "/k", x, att.key_dense)
        value = dense(prefix + "/v", x, att.value_dense)
        query = att.separate_heads(query, batch_size)
        key = att.separate_heads(key, batch_size)
        value = att.separate_heads(value, batch_size)
        scores = matmul(prefix + "/qk", query, key, transpose_b=True)
        scaled_scores = scores / tf.math.sqrt(tf.cast(tf.shape(key)[-1], tf.float32))
        probabilities = tf.nn.softmax(scaled_scores, axis=-1)
        attended = matmul(prefix + "/av", probabilities, value)
        attended = tf.transpose(attended, perm=[0, 2, 1, 3])
        attended = tf.reshape(attended, [batch_size, -1, att.embed_dim])
        return dense(prefix + "/o", attended, att.combine_heads)

    with tf.name_scope("kwt_int16" if quantized else "kwt_fp32_reference"):
        x = dense("patch", kwt.input, kwt.patch_proj)
        outputs["patch_output"] = x
        token = tf.broadcast_to(kwt.class_emb, [tf.shape(x)[0], 1, kwt.d_model])
        x = tf.concat([token, x], axis=1) + kwt.pos_emb
        outputs["encoder_input"] = x

        for index, block in enumerate(kwt.enc_layers, 1):
            prefix = "layer_{:02d}".format(index)
            if block.prenorm:
                attn_input = block.layernorm1(x)
                residual = x + attention(prefix, attn_input, block.att)
                ffn_input = block.layernorm2(residual)
            else:
                attended = attention(prefix, x, block.att)
                residual = block.layernorm1(x + attended)
                ffn_input = residual
            hidden = dense(prefix + "/fc1", ffn_input, block.ffn.layers[0])
            hidden = block.ffn.layers[1](hidden)
            projected = dense(prefix + "/fc2", hidden, block.ffn.layers[2])
            if block.prenorm:
                x = residual + projected
            else:
                x = block.layernorm2(residual + projected)
            outputs[prefix + "_output"] = x

        outputs["logits"] = dense("head", x[:, 0, :], head)
    return outputs, operands, quantized_operands, clipping_counts


def build_reference(tf, model, kwt):
    """Return ``(outputs, FP32_operands)`` using the existing official layers.

    Outputs: patch_output, encoder_input, layer_01_output ... layer_12_output,
    logits.  Each encoder tensor is [B, 99, 64].  No augmentation or dropout is
    added; the caller must disable augmentation in the official input graph.
    """
    outputs, operands, _, _ = _build(tf, model, kwt, None)
    return outputs, operands


def build_quantized(tf, model, kwt, fractional_bits):
    """Return outputs, FP32 operands, INT16 operands and per-sample clipping.

    ``fractional_bits`` maps every operand key to an integer F.  Both operands
    of every matrix product are quantized to INT16, cast to INT64, accumulated
    with an INT64 matmul, and dequantized with 2**(-(F_left + F_right)).  Dense
    biases, embeddings, residuals, LayerNorm, GELU, attention scaling, and
    Softmax remain FP32.  Clipping is counted after rounding and before
    saturation for each activation operand, reducing every axis except batch.

    Keys: patch/{a,w}; layer_XX/{q,k,v,o,fc1,fc2}/{a,w};
    layer_XX/{qk,av}/{a,b}; head/{a,w}, for XX=01..12.  The two operand
    dictionaries have all 196 keys; clipping_counts has 122 activation keys.
    """
    if fractional_bits is None:
        raise TypeError("fractional_bits must be a mapping, not None")
    return _build(tf, model, kwt, fractional_bits)
