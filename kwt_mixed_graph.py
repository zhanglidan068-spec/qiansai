"""One TensorFlow graph for reproducible, layer-wise KWT precision searches.

Quantized matrix operands contain signed, symmetric fixed-point codes in an
INT64 transport tensor.  Their *effective* widths are the ``bits`` controls;
INT64 is needed only for exact product accumulation.  Setting a stage's
``enabled`` control to False selects genuine FP32 matmuls and nonlinear
operators.  No model variable is created, initialized, or updated here.
"""

import math

from kwt_int8_n16_graph import _existing_layers


def build_search_graph(tf, model, kwt, calibration_ranges, nonlinear_config,
                       dense_integer_matmul=None):
    """Return ``(outputs, controls, records)``; all controls are feedable.

    ``outputs`` has the same 15 keys as the existing quantized experiment.
    ``controls`` has maps ``enabled`` (patch/layer_01..12/head), ``op_enabled``
    (98 matrix operations), ``bits`` and ``F`` (196 operand keys), plus a
    scalar ``nonlinear_mode`` (0 FP32, 1 integer INT16) and its per-layer
    overrides in ``nonlinear_mode_by_layer``.  ``compact_gelu`` defaults True
    and selects the small shared interpolated GELU table; False reproduces
    the earlier full INT16-domain GELU lookup.  Defaults are W8A8/N16 with
    Softmax -> AV signed INT8 F=6.  Feed explicit F values for every trial.

    ``calibration_ranges`` maps operand keys to training-only max |x|; values
    may also be dicts with a ``max_abs``/``maximum``/``calibration_max_abs``.
    Integer nonlinear kernels retain their static training-calibrated config.
    ``dense_integer_matmul``, if supplied, receives (prefix, left, right) and
    returns an integer dot product before dequantization/bias. The default
    remains exact tf.matmul. QK and AV are unaffected by this optional hook.
    ``records`` exposes operands, bounded integer codes, clipping counts and
    selected nonlinear interfaces.  Dummy nonlinear integer interfaces are
    zeros when that layer selects FP32; ``used_int16`` identifies this case.
    """
    from kwt_nonlinear_int16 import fixed_softmax, fixed_gelu, fixed_layer_norm

    # TF 2.4 FuncGraph captures become expensive in a deeply nested search
    # graph. Classic Switch/Merge keeps trial controls cheap and predictable.
    tf.compat.v1.disable_control_flow_v2()
    head = _existing_layers(model, kwt)
    controls = {"enabled": {}, "op_enabled": {}, "bits": {}, "F": {},
                "nonlinear_mode_by_layer": {}}
    outputs, operands, q_operands, clips = {}, {}, {}, {}
    nonlinear_records, nonlinear_tensors = {}, {}
    stages = ["patch"] + ["layer_{:02d}".format(i) for i in range(1, 13)] + ["head"]

    def placeholder(dtype, value, name):
        return tf.compat.v1.placeholder_with_default(tf.constant(value, dtype),
                                                      shape=(), name=name)

    with tf.name_scope("mixed_precision_controls"):
        controls["nonlinear_mode"] = placeholder(tf.int32, 1, "nonlinear_mode")
        controls["compact_gelu"] = placeholder(tf.bool, True, "compact_gelu")
        for stage in stages:
            controls["enabled"][stage] = placeholder(tf.bool, True, stage + "_enabled")
            if stage.startswith("layer_"):
                controls["nonlinear_mode_by_layer"][stage] = \
                    tf.compat.v1.placeholder_with_default(
                        controls["nonlinear_mode"], shape=(), name=stage + "_nonlinear_mode")

    def max_abs(key):
        value = calibration_ranges[key]
        if isinstance(value, dict):
            for name in ("max_abs", "maximum", "calibration_max_abs", "observed_max_abs"):
                if name in value:
                    value = value[name]
                    break
        value = float(value)
        if not math.isfinite(value) or value < 0:
            raise ValueError("Invalid calibration range for " + key)
        return value

    def stage_of(prefix):
        return prefix.split("/")[0]

    def operation_enabled(prefix):
        if prefix not in controls["op_enabled"]:
            controls["op_enabled"][prefix] = placeholder(
                tf.bool, True, prefix.replace("/", "_") + "_matmul_enabled")
        return tf.logical_and(controls["enabled"][stage_of(prefix)],
                              controls["op_enabled"][prefix])

    def operand(key, tensor):
        tensor = tf.convert_to_tensor(tensor, dtype=tf.float32)
        operands[key] = tensor
        observed = max_abs(key)
        headroom = 1.0 if key.endswith("/w") else 1.25
        default_f = 0 if observed == 0 else int(math.floor(math.log2(127.0 / (observed * headroom))))
        if key.endswith("/av/a"):
            default_f = 6
        name = key.replace("/", "_")
        bits = controls["bits"][key] = placeholder(tf.int32, 8, name + "_bits")
        fraction = controls["F"][key] = placeholder(tf.int32, default_f, name + "_F")
        limit = tf.bitwise.left_shift(tf.constant(1, tf.int64),
                                     tf.cast(bits - 1, tf.int64)) - 1
        scale = tf.pow(tf.constant(2., tf.float32), tf.cast(fraction, tf.float32))
        rounded = tf.math.rint(tensor * scale)
        limit_float = tf.cast(limit, tf.float32)
        code = tf.cast(tf.clip_by_value(rounded, -limit_float, limit_float), tf.int64)
        q_operands[key] = code
        if not key.endswith("/w"):
            clips[key] = tf.reduce_sum(tf.cast(tf.logical_or(
                rounded < -limit_float, rounded > limit_float), tf.int64),
                axis=list(range(1, tensor.shape.rank)))
        return code

    def dequantize(accumulator, left_key, right_key):
        exponent = -(controls["F"][left_key] + controls["F"][right_key])
        scale = tf.pow(tf.constant(2., tf.float32), tf.cast(exponent, tf.float32))
        return tf.cast(accumulator, tf.float32) * scale

    def dense(prefix, value, layer):
        left_key, right_key = prefix + "/a", prefix + "/w"
        left, right = operand(left_key, value), operand(right_key, layer.kernel)
        enabled = operation_enabled(prefix)

        def integer_product():
            shape = tf.shape(left)
            flattened = tf.reshape(left, [-1, shape[-1]])
            accumulator = (tf.matmul(flattened, right) if dense_integer_matmul is None
                           else dense_integer_matmul(prefix, flattened, right))
            result = dequantize(accumulator, left_key, right_key)
            result = tf.reshape(result, tf.concat([shape[:-1], [layer.units]], axis=0))
            result.set_shape(value.shape.as_list()[:-1] + [layer.units])
            return tf.nn.bias_add(result, layer.bias) if layer.use_bias else result

        # The layers and their variables already exist. Calling .call directly
        # avoids Keras' recursive history traversal across the conditional DAG.
        result = tf.cond(enabled, integer_product, lambda: layer.call(value))
        result.set_shape(value.shape.as_list()[:-1] + [layer.units])
        return result

    def matmul(prefix, a, b, transpose_b=False):
        left_key, right_key = prefix + "/a", prefix + "/b"
        left, right = operand(left_key, a), operand(right_key, b)
        return tf.cond(operation_enabled(prefix), lambda: dequantize(
            tf.matmul(left, right, transpose_b=transpose_b), left_key, right_key),
            lambda: tf.matmul(a, b, transpose_b=transpose_b))

    def nonlinear(prefix, kind, value, layer=None):
        config = nonlinear_config[prefix]
        nonlinear_tensors[prefix + "/input"] = value
        stage = stage_of(prefix)
        enabled = tf.logical_and(controls["enabled"][stage], tf.not_equal(
            controls["nonlinear_mode_by_layer"][stage], 0))
        names = ("output_dequant", "input_int16", "output_int16",
                 "input_clipped", "output_clipped")

        def integer_kernel():
            if kind == "softmax":
                from compact_softmax_int16 import fixed_softmax as compact_softmax

                def softmax_record(function):
                    result = function(tf, value, int(config["input_f"]), int(config["output_f"]))
                    return tuple(result[name] for name in names)

                return tf.cond(controls["compact_gelu"],
                               lambda: softmax_record(compact_softmax),
                               lambda: softmax_record(fixed_softmax))
            elif kind == "gelu":
                from compact_gelu_int16 import fixed_gelu as compact_gelu

                def gelu_record(function):
                    result = function(tf, value, int(config["input_f"]), int(config["output_f"]))
                    return tuple(result[name] for name in names)

                return tf.cond(controls["compact_gelu"],
                               lambda: gelu_record(compact_gelu),
                               lambda: gelu_record(fixed_gelu))
            else:
                record = fixed_layer_norm(
                    tf, value, int(config["input_f"]), int(config["output_f"]),
                    layer.gamma, layer.beta, epsilon=float(config.get("epsilon", layer.epsilon)),
                    gamma_f=int(config.get("gamma_f", 14)), beta_f=int(config.get("beta_f", 16)))
            return tuple(record[name] for name in names)

        def floating_kernel():
            result = tf.nn.softmax(value, axis=-1) if kind == "softmax" else layer.call(value)
            zeros = tf.zeros_like(value, tf.int16)
            count_zeros = tf.zeros([tf.shape(value)[0]], tf.int64)
            return result, zeros, zeros, count_zeros, count_zeros

        chosen = tf.cond(enabled, integer_kernel, floating_kernel)
        record = dict(zip(names, chosen))
        record["pre_input"] = value
        record["used_int16"] = enabled
        nonlinear_records[prefix] = record
        result = record["output_dequant"]
        result.set_shape(value.shape)
        nonlinear_tensors[prefix + "/output"] = result
        return result

    def attention(prefix, value, att):
        batch_size = tf.shape(value)[0]
        query = att.separate_heads(dense(prefix + "/q", value, att.query_dense), batch_size)
        key = att.separate_heads(dense(prefix + "/k", value, att.key_dense), batch_size)
        val = att.separate_heads(dense(prefix + "/v", value, att.value_dense), batch_size)
        scores = matmul(prefix + "/qk", query, key, transpose_b=True)
        probabilities = nonlinear(prefix + "/softmax", "softmax", scores * (1. / 8.))
        attended = matmul(prefix + "/av", probabilities, val)
        attended = tf.transpose(attended, perm=[0, 2, 1, 3])
        attended = tf.reshape(attended, [batch_size, -1, att.embed_dim])
        return dense(prefix + "/o", attended, att.combine_heads)

    variables_before = tuple(tf.compat.v1.global_variables())
    with tf.name_scope("kwt_mixed_precision_search"):
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
            value = residual + projected if block.prenorm else nonlinear(
                prefix + "/ln2", "ln2", residual + projected, block.layernorm2)
            outputs[prefix + "_output"] = value
        outputs["logits"] = dense("head", value[:, 0, :], head)

    if tuple(tf.compat.v1.global_variables()) != variables_before:
        raise AssertionError("Precision-search graph created model variables")
    if (len(outputs), len(operands), len(controls["op_enabled"]), len(nonlinear_records)) != (15, 196, 98, 48):
        raise AssertionError("Incomplete KWT mixed-precision graph")
    controls["fractional_bits"] = controls["F"]
    controls["compact_nonlinear"] = controls["compact_gelu"]
    records = {"operands": operands, "q_operands": q_operands,
               "clipping_counts": clips, "nonlinear_records": nonlinear_records,
               "nonlinear_tensors": nonlinear_tensors,
               "code_container_dtype": "int64; effective width is controlled per operand"}
    return outputs, controls, records
