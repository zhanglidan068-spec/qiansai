"""Export common encoder parameters and two shared compact nonlinear tables.

Reads only weights_fp32.npz and nonlinear_config.json from an existing search
run. It never reads audio, labels, predictions or a testing split. Output files
are exclusive creations in a new directory. NumPy is the only dependency; no
TensorFlow graph or search process is touched.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import struct

import numpy as np

from compact_gelu_int16 import export_lut_tables as gelu_tables
from compact_gelu_int16 import implementation_metadata as gelu_metadata
from compact_softmax_int16 import export_lut_tables as softmax_tables
from compact_softmax_int16 import implementation_metadata as softmax_metadata


ROOT = Path(__file__).absolute().parent
LIMIT = 32767


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_exclusive(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def quantize_ln_parameter(value, fractional_bits):
    """Match fixed_layer_norm.affine_parameter: FP32 multiply, rint, clip."""
    if fractional_bits not in (14, 16):
        raise ValueError("This common payload requires gamma Q14 and beta Q16")
    value = np.asarray(value, dtype=np.float32)
    if not np.all(np.isfinite(value)):
        raise ValueError("Non-finite LayerNorm parameter")
    units = np.rint(np.multiply(
        value, np.float32(math.ldexp(1.0, fractional_bits)), dtype=np.float32))
    clips = int(np.count_nonzero((units < -LIMIT) | (units > LIMIT)))
    return np.clip(units, -LIMIT, LIMIT).astype("<i2"), clips


def self_test():
    cases = []
    for fraction in (14, 16):
        halves = np.asarray([0.5, 1.5, 2.5, -0.5, -1.5, -2.5], np.float32)
        q, clips = quantize_ln_parameter(halves * np.float32(2.0 ** -fraction), fraction)
        if q.tolist() != [0, 2, 2, 0, -2, -2] or clips != 0:
            raise AssertionError("LayerNorm nearest-even tie handling mismatch")
        q, clips = quantize_ln_parameter(
            np.asarray([32767.75, -32767.75], np.float32) * np.float32(2.0 ** -fraction), fraction)
        if q.tolist() != [32767, -32767] or clips != 2:
            raise AssertionError("LayerNorm symmetric saturation mismatch")
        cases.extend(["Q{} ties-to-even".format(fraction), "Q{} saturation".format(fraction)])
    if np.asarray([1, -1, 32767, -32767], dtype="<i2").tobytes() != struct.pack("<hhhh", 1, -1, 32767, -32767):
        raise AssertionError("Signed INT16 little-endian encoding mismatch")
    if np.asarray([1.0, -0.5], dtype="<f4").tobytes() != struct.pack("<ff", 1.0, -0.5):
        raise AssertionError("FP32 little-endian encoding mismatch")
    cases.extend(["INT16 little endian", "FP32 little endian"])
    return {"passed": True, "cases": cases, "test_count": len(cases)}


def export(run, out=None):
    tests = self_test()
    run = run.absolute()
    out = out.absolute() if out is not None else run / "common_hardware_payload"
    if out.exists():
        raise FileExistsError("Refusing to overwrite existing directory: " + str(out))
    if not out.parent.is_dir():
        raise FileNotFoundError("Output parent must already exist: " + str(out.parent))
    inputs = {"weights_fp32.npz": run / "weights_fp32.npz",
              "nonlinear_config.json": run / "nonlinear_config.json"}
    input_hashes = {name: sha256_file(path) for name, path in inputs.items()}
    nonlinear = json.loads(inputs["nonlinear_config.json"].read_text(encoding="utf-8"))
    entries, payload_parts, reference_arrays = [], [], []
    offset = 0

    def add(name, value, dtype, fractional_bits, category, source_key=None, clips=0):
        nonlocal offset
        value = np.ascontiguousarray(np.asarray(value, dtype=np.dtype(dtype)))
        if not np.all(np.isfinite(value)):
            raise ValueError("Non-finite payload entry: " + name)
        blob = value.tobytes(order="C")
        entries.append({
            "name": name, "category": category, "file": "encoder_common.bin",
            "offset_bytes": offset, "length_bytes": len(blob),
            "elements": int(value.size), "shape": list(value.shape),
            "dtype": np.dtype(dtype).str, "byte_order": "little",
            "fractional_bits": fractional_bits,
            "scale": None if fractional_bits is None else math.ldexp(1.0, -fractional_bits),
            "source_weight_key": source_key, "clipped_parameters": int(clips),
            "sha256": sha256_bytes(blob),
        })
        payload_parts.append(blob)
        reference_arrays.append(value.copy())
        offset += len(blob)

    with np.load(str(inputs["weights_fp32.npz"]), allow_pickle=False) as weights:
        for layer in range(1, 13):
            prefix = "layer_{:02d}".format(layer)
            for suffix, expected in (("o_bias", 64), ("fc1_bias", 256), ("fc2_bias", 64)):
                key = prefix + "_" + suffix
                source = weights[key]
                if source.dtype != np.float32 or source.shape != (expected,):
                    raise ValueError("Unexpected bias dtype/shape: " + key)
                add(prefix + "/" + suffix, source, "<f4", None, "encoder_FP32_bias", key)
            for kind in ("ln1", "ln2"):
                config = nonlinear[prefix + "/" + kind]
                if int(config["gamma_f"]) != 14 or int(config["beta_f"]) != 16:
                    raise ValueError("Expected LayerNorm gamma Q14/beta Q16: " + prefix + "/" + kind)
                for parameter, fraction in (("gamma", 14), ("beta", 16)):
                    key = prefix + "_" + kind + "_" + parameter
                    source = weights[key]
                    if source.dtype != np.float32 or source.shape != (64,):
                        raise ValueError("Unexpected LayerNorm dtype/shape: " + key)
                    quantized, clips = quantize_ln_parameter(source, fraction)
                    add(prefix + "/" + kind + "/" + parameter, quantized, "<i2",
                        fraction, "encoder_LayerNorm_INT16", key, clips)

    softmax_table = softmax_tables()["softmax_shared_compact_exp_q15"]
    gelu_table = gelu_tables()["gelu_shared_compact_q12"]
    if softmax_table.shape != (4096,) or gelu_table.shape != (1025,):
        raise AssertionError("Unexpected compact nonlinear ROM dimensions")
    add("shared/softmax_exp_q15", softmax_table, "<i2", 15, "shared_nonlinear_ROM")
    add("shared/gelu_q12", gelu_table, "<i2", 12, "shared_nonlinear_ROM")
    blob = b"".join(payload_parts)
    if len(entries) != 86 or len(blob) != 34818 or offset != len(blob):
        raise AssertionError("Common encoder payload size/count mismatch")
    if any(e["source_weight_key"] is not None and not e["source_weight_key"].startswith("layer_") for e in entries):
        raise AssertionError("A software-side parameter leaked into encoder payload")
    for entry, expected in zip(entries, reference_arrays):
        raw = blob[entry["offset_bytes"]:entry["offset_bytes"] + entry["length_bytes"]]
        decoded = np.frombuffer(raw, dtype=np.dtype(entry["dtype"])).reshape(entry["shape"])
        if sha256_bytes(raw) != entry["sha256"] or not np.array_equal(decoded, expected):
            raise AssertionError("Prewrite byte roundtrip mismatch: " + entry["name"])

    source_names = ["export_precision_common_payload.py", "kwt_nonlinear_int16.py",
                    "compact_gelu_int16.py", "compact_softmax_int16.py"]
    source_hashes = {name: sha256_file(ROOT / name) for name in source_names}
    manifest = {
        "purpose": "Common encoder bias, integer LayerNorm affine parameters and two shared compact nonlinear tables",
        "source_run": str(run), "testing_samples_read": 0,
        "input_sha256": input_hashes, "source_code_sha256": source_hashes,
        "binary": {"file": "encoder_common.bin", "bytes": len(blob),
                   "sha256": sha256_bytes(blob), "byte_order": "little", "padding_bytes": 0,
                   "sections": len(entries)},
        "totals": {"FP32_bias_elements": 4608, "FP32_bias_bytes": 18432,
                   "LayerNorm_INT16_elements": 3072, "LayerNorm_INT16_bytes": 6144,
                   "shared_ROM_bytes": 10242,
                   "LayerNorm_parameter_clips": sum(e["clipped_parameters"] for e in entries)},
        "quantization": {"affine_parameter_source": "kwt_nonlinear_int16.fixed_layer_norm.affine_parameter",
                         "algorithm": "FP32 multiply by exact power-of-two; nearest-even rint; clip [-32767,32767]; cast INT16",
                         "gamma_fractional_bits": 14, "beta_fractional_bits": 16,
                         "gamma_source_parameters_unchanged": True},
        "nonlinear_config": nonlinear,
        "shared_ROM_metadata": {"softmax": softmax_metadata(), "gelu": gelu_metadata()},
        "excluded": ["patch weights/bias", "classification head weights/bias", "position/class embeddings", "encoder matrix weights (candidate-specific)", "predictions", "labels", "audio"],
        "hardware_boundary": {"encoder_bias_fp32": True, "encoder_residual_fp32": True,
                              "FP32_arithmetic_and_converters_in_accelerator_area_required": True,
                              "E203_free_offload_assumed": False,
                              "area_mm2": None, "area_compliance_verified": False,
                              "all_integer_bias_residual_requires_new_numeric_validation": True},
        "self_test": tests, "entries": entries,
    }
    out.mkdir(exist_ok=False)
    with (out / "encoder_common.bin").open("xb") as handle:
        handle.write(blob)
    write_json_exclusive(out / "manifest.json", manifest)
    saved = (out / "encoder_common.bin").read_bytes()
    if saved != blob:
        raise AssertionError("Saved payload differs from in-memory reference")
    for entry in entries:
        section = saved[entry["offset_bytes"]:entry["offset_bytes"] + entry["length_bytes"]]
        if sha256_bytes(section) != entry["sha256"]:
            raise AssertionError("Saved section SHA256 mismatch")
    after = {name: sha256_file(path) for name, path in inputs.items()}
    if after != input_hashes:
        raise AssertionError("Source inputs changed during export")
    verification = {"passed": True, "self_tests": tests,
                    "binary_roundtrip_exact": True, "section_sha256_matches": len(entries),
                    "input_files_unchanged": True, "patch_head_excluded": True,
                    "LayerNorm_parameter_clips": manifest["totals"]["LayerNorm_parameter_clips"],
                    "binary_sha256": sha256_file(out / "encoder_common.bin"),
                    "manifest_sha256": sha256_file(out / "manifest.json"),
                    "binary_bytes": len(saved)}
    write_json_exclusive(out / "verification.json", verification)
    readme = """# 编码器公共硬件参数导出

`encoder_common.bin` 是不含文件头、不含填充的 little-endian 原始负载，共34,818字节、86段。

- 12层 O/FC1/FC2 FP32 bias：18,432字节，保持原权重比特。
- 24处 LayerNorm 的 gamma/beta：6,144字节，gamma int16 F=14、beta int16 F=16。
- 两共享非线性表：Softmax 4096×int16 Q15、GELU 1025×int16 Q12，共10,242字节。

`manifest.json` 记录每段offset、shape、little-endian dtype、F、scale、字节数、SHA256及源文件哈希；`verification.json` 记录舍入/饱和/字节序自测与86段落盘核验。

不含patch、分类头、嵌入、候选特定矩阵权重，也不读取音频、标签或历史预测。导出器拒绝覆盖已有目录。

注意：bias与residual在当前数值模型中仍为FP32。若在encoder硬件内部执行，必须计入FP32加法、定点/浮点转换的面积和周期，不能免费卸载给E203。若改为全整数接口，需要重新验证。此负载不构成ASIC面积合规证明。

读取单段示例：

```python
import json
import numpy as np
from pathlib import Path
root = Path(__file__).absolute().parent  # 示例脚本放在此导出目录
m = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
e = m['entries'][0]
with (root / e['file']).open('rb') as f:
    f.seek(e['offset_bytes'])
    a = np.frombuffer(f.read(e['length_bytes']), dtype=e['dtype']).reshape(e['shape'])
```
"""
    with (out / "README_公共硬件参数.md").open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(readme)
    return {"out": str(out), "bytes": len(blob), "sections": len(entries),
            "LayerNorm_parameter_clips": verification["LayerNorm_parameter_clips"],
            "sha256": verification["binary_sha256"], "verification_passed": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.run is None:
        if not args.self_test:
            parser.error("--run is required unless only --self-test is requested")
        print(json.dumps(self_test(), ensure_ascii=False, indent=2))
        return
    print(json.dumps(export(args.run, args.out), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
