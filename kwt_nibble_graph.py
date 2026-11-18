"""Optional CPU oracle for progressive four-bit Dense multiplication.

The LUT executes only the arithmetic definition in software. It is not an RTL
implementation, and software wall time is not a hardware speedup measurement.
"""
import numpy as np

from kwt_mixed_graph import build_search_graph


class NibbleGraph:
    def __init__(self, tf, model, kwt, ranges, nonlinear, engine, batch_size=32):
        self.tf, self.engine, self.batch_size = tf, engine, int(batch_size)
        self.valid_samples = self.batch_size
        self.collect_stats = False
        self.native_enabled, self.counts = {}, {}
        self.measured_counts, self.native_call_counts = {}, {}
        self.product = tf.compat.v1.placeholder(tf.int32, [255, 255], name='nibble_product_lut')
        self.stage = tf.compat.v1.placeholder(tf.uint8, [255, 255], name='nibble_stage_lut')

        def hook(prefix, left, right):
            if not prefix.startswith('layer_'):
                return tf.matmul(left, right)
            enabled = tf.compat.v1.placeholder_with_default(False, (), name=prefix.replace('/', '_') + '_nibble_enabled')
            self.native_enabled[prefix] = enabled

            def native():
                def function(a, w, products, stages):
                    # Exclude duplicate padding from measured operation counts.
                    rows_per_sample = a.shape[0] // self.batch_size
                    valid_rows = rows_per_sample * self.valid_samples
                    result, counts = self.engine.matmul(a[:valid_rows], w, products, stages,
                                                        collect_stats=self.collect_stats)
                    if self.collect_stats:
                        self.measured_counts[prefix] = counts.copy()
                        self.native_call_counts[prefix] = self.native_call_counts.get(prefix, 0) + 1
                    if valid_rows < a.shape[0]:
                        padding = np.tile(result[-rows_per_sample:], (self.batch_size - self.valid_samples, 1))
                        result = np.concatenate([result, padding], axis=0)
                    return result, counts
                product, counts = tf.numpy_function(function, [left, right, self.product, self.stage],
                                                     [tf.int64, tf.int64], name=prefix.replace('/', '_') + '_oracle')
                product.set_shape([left.shape.as_list()[0], right.shape.as_list()[1]])
                counts.set_shape([4])
                return product, counts

            product, counts = tf.cond(enabled, native,
                                      lambda: (tf.matmul(left, right), tf.zeros([4], tf.int64)))
            self.counts[prefix] = counts
            return product

        self.outputs, self.controls, self.records = build_search_graph(
            tf, model, kwt, ranges, nonlinear, dense_integer_matmul=hook)

    def feed(self, quantization, product_lut, stage_lut, operations=(), fp32=False):
        controls = self.controls
        feed = {self.product: product_lut, self.stage: stage_lut,
                controls['nonlinear_mode']: 0 if fp32 else 1,
                controls['compact_gelu']: True}
        for name, control in controls['enabled'].items():
            feed[control] = name.startswith('layer_') and not fp32
        for key, value in quantization['operand_bits'].items():
            feed[controls['bits'][key]] = int(value)
            feed[controls['F'][key]] = int(quantization['fractional_bits'][key])
        active = set(operations)
        for op, control in self.native_enabled.items():
            feed[control] = op in active and not fp32
        return feed
