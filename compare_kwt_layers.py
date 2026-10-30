"""Read one sample from saved FP32/W16A16 layer outputs without modifying files."""

import argparse
import json
import os
from pathlib import Path

import numpy as np


def load_output(run_dir, manifest, branch, stage):
    key = branch + "/" + stage
    if key not in manifest:
        available = sorted(k.split("/", 1)[1] for k in manifest if k.startswith(branch + "/"))
        raise ValueError("Unknown stage {!r}; available: {}".format(stage, ", ".join(available)))
    entry = manifest[key]
    relative = Path(entry["file"])
    if relative.is_absolute():
        raise ValueError("Manifest output path must be relative: {}".format(relative))
    output_path = Path(os.path.abspath(str(run_dir / relative)))
    if os.path.commonpath([str(run_dir), str(output_path)]) != str(run_dir):
        raise ValueError("Manifest output path leaves the experiment directory")
    array = np.load(str(output_path), mmap_mode="r", allow_pickle=False)
    if tuple(array.shape) != tuple(entry["shape"]):
        raise ValueError("{} shape {} disagrees with manifest {}".format(key, array.shape, entry["shape"]))
    if str(array.dtype) != entry["dtype"]:
        raise ValueError("{} dtype {} disagrees with manifest {}".format(key, array.dtype, entry["dtype"]))
    if array.ndim not in (2, 3):
        raise ValueError("{} must have [sample, channel] or [sample, token, channel] shape".format(key))
    return array


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path, help="Experiment directory")
    parser.add_argument("--sample-index", type=int, default=0, help="Zero-based sample index")
    parser.add_argument("--stage", default="layer_12_output", help="Stage name, e.g. layer_01_output or logits")
    parser.add_argument("--quant-branch", default="w16a16", help="Quantized branch: w16a16 or w8a8_n16")
    parser.add_argument("--token", type=int, default=0, help="Zero-based token index; token 0 is CLS")
    args = parser.parse_args()
    run_dir = Path(os.path.abspath(str(args.run)))

    try:
        with (run_dir / "layer_outputs_manifest.json").open("r", encoding="utf-8-sig") as handle:
            manifest = json.load(handle)
        fp32_all = load_output(run_dir, manifest, "fp32", args.stage)
        quant_all = load_output(run_dir, manifest, args.quant_branch, args.stage)
        if fp32_all.shape != quant_all.shape:
            raise ValueError("FP32 and W16A16 shapes differ")
        if not 0 <= args.sample_index < fp32_all.shape[0]:
            raise ValueError("--sample-index must be between 0 and {}".format(fp32_all.shape[0] - 1))
        token_count = fp32_all.shape[1] if fp32_all.ndim == 3 else 1
        if not 0 <= args.token < token_count:
            raise ValueError("--token must be between 0 and {} for {}".format(token_count - 1, args.stage))

        # Only materialize the chosen sample; the complete dataset stays memory mapped.
        fp32 = np.asarray(fp32_all[args.sample_index], dtype=np.float64)
        quant = np.asarray(quant_all[args.sample_index], dtype=np.float64)
        if not np.isfinite(fp32).all() or not np.isfinite(quant).all():
            raise ValueError("Selected sample contains NaN or Inf")
        difference = quant - fp32
        ref_norm = float(np.linalg.norm(fp32.ravel()))
        quant_norm = float(np.linalg.norm(quant.ravel()))
        error_norm = float(np.linalg.norm(difference.ravel()))
        relative_l2 = error_norm / ref_norm if ref_norm else (0.0 if not error_norm else float("inf"))
        if ref_norm and quant_norm:
            cosine = float(np.dot(fp32.ravel(), quant.ravel()) / (ref_norm * quant_norm))
            cosine = float(np.clip(cosine, -1.0, 1.0))
        else:
            cosine = 1.0 if ref_norm == quant_norm else 0.0

        print("Run: {}".format(run_dir))
        print("Stage: {} | stored shape: {} | sample: {}".format(args.stage, fp32_all.shape, args.sample_index))
        print("Whole-sample metrics (all tokens and channels):")
        print("  MAE         = {:.10g}".format(float(np.mean(np.abs(difference)))))
        print("  RMSE        = {:.10g}".format(float(np.sqrt(np.mean(difference ** 2)))))
        print("  maxabs      = {:.10g}".format(float(np.max(np.abs(difference)))))
        print("  relative L2 = {:.10g}".format(relative_l2))
        print("  cosine      = {:.10g}".format(cosine))
        if fp32.ndim == 2:
            ref_channels, quant_channels = fp32[args.token], quant[args.token]
            print("Token: {}{}".format(args.token, " (CLS)" if args.token == 0 else ""))
        else:
            ref_channels, quant_channels = fp32, quant
            print("Logits have no token dimension; --token must be 0.")
        print("{:>7} {:>16} {:>16} {:>16}".format("channel", "FP32", args.quant_branch, "quant - FP32"))
        for channel in range(min(8, ref_channels.shape[0])):
            reference = float(ref_channels[channel])
            fixed = float(quant_channels[channel])
            print("{:7d} {:16.9g} {:16.9g} {:16.9g}".format(channel, reference, fixed, fixed - reference))
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
