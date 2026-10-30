"""Read-only KWT-1 checkpoint inference and FP32 layer trace export.

Target: ARM-software/keyword-transformer @ 237e4097, Python 3.8 / TF 2.4.0.
The original repository and model directory are never written by this script.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path("keyword-transformer"))
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--split", choices=("validation", "testing"), default="validation")
    parser.add_argument("--limit", type=int, default=32, help="0 = entire selected split")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--out", type=Path, default=Path("runs/smoke"))
    parser.add_argument("--smoke", action="store_true", help="Use a zero waveform; no accuracy claim")
    args = parser.parse_args()
    if args.batch_size < 1 or args.limit < 0:
        parser.error("batch-size must be positive and limit must be nonnegative")
    if not args.smoke and args.data_dir is None:
        parser.error("Provide --data-dir, or use --smoke for the checkpoint loading check")
    return args


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_revision(repo):
    if not shutil.which("git"):
        return None
    result = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                            capture_output=True, text=True)
    revision = result.stdout.strip() if result.returncode == 0 else None
    if revision and not revision.startswith("237e4097"):
        raise RuntimeError("Wrong repository revision: " + revision + "; checkout 237e4097 first")
    return revision


def make_flags(args, model_dir, base_parser, kws_transformer, model_flags):
    parser = base_parser.base_parser()
    subparsers = parser.add_subparsers(dest="model_name")
    kws_transformer.model_parameters(subparsers.add_parser("kws_transformer"))
    flags = parser.parse_args(["kws_transformer"])
    vars(flags).update(json.loads((model_dir / "flags.json").read_text(encoding="utf-8")))
    expected = dict(model_name="kws_transformer", num_layers=12, heads=1,
                    d_model=64, mlp_dim=256, attention_type="time", preprocess="raw",
                    feature_type="mfcc_tf", sample_rate=16000, clip_duration_ms=1000,
                    dct_num_features=40)
    for key, value in expected.items():
        if getattr(flags, key, None) != value:
            raise ValueError("Unexpected KWT-1 setting: {}={!r}".format(key, getattr(flags, key, None)))
    flags.train = 0
    flags.training = False
    flags.data_url = ""
    flags.data_dir = "" if args.smoke else str(args.data_dir.absolute())
    flags.train_dir = str(args.out.absolute())
    flags.batch_size = args.batch_size
    flags.distill_teacher_json = ""
    flags.prenorm = False
    flags.approximate_gelu = False
    flags.return_softmax = 0
    flags.dropout1 = 0.0
    flags.use_spec_augment = 0
    flags.use_spec_cutout = 0
    flags.sp_resample = 0.0
    flags.sp_time_shift_ms = 0.0
    flags.time_shift_ms = 0.0
    flags.resample = 0.0
    flags.volume_resample = 0.0
    flags.background_frequency = 0.0
    flags.background_volume = 0.0
    flags.pick_deterministically = 1
    flags = model_flags.update_flags(flags)
    if (flags.desired_samples, flags.spectrogram_length, flags.label_count) != (16000, 98, 12):
        raise ValueError("Expected 16000 samples, 98 MFCC frames, 12 classes")
    return flags


def build_probe(tf, model, kwt):
    """Reuse the loaded official layers; also retain official CLS for comparison."""
    features = kwt.input
    x = kwt.patch_proj(features)
    token = tf.broadcast_to(kwt.class_emb, [tf.shape(x)[0], 1, kwt.d_model])
    x = tf.concat([token, x], axis=1) + kwt.pos_emb
    tensors = {"mfcc": features, "encoder_input": x}
    for index, block in enumerate(kwt.enc_layers, 1):
        x, _ = block(x, training=False)
        tensors["layer_{:02d}_output".format(index)] = x
    tensors["probe_cls"] = x[:, 0, :]
    tensors["official_cls"] = kwt.output
    tensors["probe_logits"] = model.layers[-1](tensors["probe_cls"])
    tensors["logits"] = model.outputs[0]
    return tensors


def export_weights(np, session, kwt, model, out):
    variables = {"pos_emb": kwt.pos_emb, "class_emb": kwt.class_emb,
                 "patch_kernel": kwt.patch_proj.kernel, "patch_bias": kwt.patch_proj.bias}
    for index, block in enumerate(kwt.enc_layers, 1):
        prefix = "layer_{:02d}_".format(index)
        for name, dense in (("q", block.att.query_dense), ("k", block.att.key_dense),
                            ("v", block.att.value_dense), ("o", block.att.combine_heads),
                            ("fc1", block.ffn.layers[0]), ("fc2", block.ffn.layers[2])):
            variables[prefix + name + "_kernel"] = dense.kernel
            if dense.use_bias:
                variables[prefix + name + "_bias"] = dense.bias
        for name, norm in (("ln1", block.layernorm1), ("ln2", block.layernorm2)):
            variables[prefix + name + "_gamma"] = norm.gamma
            variables[prefix + name + "_beta"] = norm.beta
    # In this fixed, non-distilled model the final Sequential is the linear head.
    head = model.layers[-1].layers[0]
    variables["head_kernel"] = head.kernel
    variables["head_bias"] = head.bias
    values = session.run(variables)
    np.savez_compressed(str(out / "weights_fp32.npz"), **values)
    write_json(out / "weights_manifest.json", {
        "dense_layout": "kernel[input_channel, output_channel]; y = x @ kernel + bias",
        "tensors": {key: {"shape": list(value.shape), "dtype": str(value.dtype)}
                    for key, value in values.items()}})


def main():
    args = arguments()
    args.repo = args.repo.absolute()
    args.out = args.out.absolute()
    model_dir = args.repo / "models_data_v2_12_labels" / "kwt1"
    if not (args.repo / "kws_streaming").is_dir():
        raise FileNotFoundError("--repo must point to the complete official repository")
    checkpoint = model_dir / "best_weights"
    files = sorted(model_dir.glob("best_weights.*"))
    if not (model_dir / "best_weights.index").is_file() or not list(model_dir.glob("best_weights.data-*")):
        raise FileNotFoundError("Missing best_weights.index / best_weights.data-* in " + str(model_dir))
    if (model_dir / "best_weights.index").read_bytes().startswith(b"version https://git-lfs"):
        raise RuntimeError("Checkpoint is a Git LFS pointer; download the actual weights first")
    revision = git_revision(args.repo)
    if revision is None:
        print("WARNING: git revision unavailable; manually verify repository commit 237e4097", flush=True)

    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(args.repo))
    import numpy as np
    import tensorflow as tf
    import tensorflow_addons as tfa
    tf.compat.v1.disable_eager_execution()
    # Import the model registry first to follow the upstream circular-import order.
    from kws_streaming.models import models as model_registry
    from kws_streaming.models import model_flags
    from kws_streaming.models.transformer_utils import KWSTransformer
    from kws_streaming.train import base_parser
    from kws_streaming.data import input_data
    kws_transformer = model_registry.kws_transformer

    if tf.__version__ != "2.4.0" or np.__version__ != "1.19.5" or tfa.__version__ != "0.12.1":
        raise RuntimeError("Use the pinned environment: TensorFlow 2.4.0, NumPy 1.19.5, Addons 0.12.1")
    random.seed(59185)
    np.random.seed(59185)
    tf.compat.v1.set_random_seed(59185)
    flags = make_flags(args, model_dir, base_parser, kws_transformer, model_flags)
    labels = input_data.prepare_words_list(flags.wanted_words.split(","), flags.split_data)
    checkpoint_labels = (model_dir / "labels.txt").read_text(encoding="utf-8").split()
    if labels != checkpoint_labels:
        raise ValueError("Label order does not match the checkpoint")
    if not args.smoke:
        if not args.data_dir.is_dir() or not list((args.data_dir / "yes").glob("*.wav")):
            raise FileNotFoundError("data-dir must directly contain yes/*.wav, no/*.wav, etc.")
        speech_dirs = [p for p in args.data_dir.iterdir()
                       if p.is_dir() and p.name != "_background_noise_" and next(p.glob("*.wav"), None)]
        if len(speech_dirs) != 35:
            raise ValueError("Use the complete Speech Commands v2 dataset (35 word directories), not only 10 words")

    args.out.mkdir(parents=True, exist_ok=True)
    metadata = {"expected_commit": "237e4097", "observed_commit": revision,
                "python": sys.version, "tensorflow": tf.__version__, "numpy": np.__version__,
                "tensorflow_addons": tfa.__version__, "device": "CPU", "batch_size": args.batch_size,
                "upstream_runtime": "tensorflow 2.4.0rc1",
                "runtime_note": "2.4.0 stable is a compatibility substitute; not an rc1 bit-exact claim",
                "checkpoint_sha256": {p.name: sha256(p) for p in files}, "labels": labels,
                "mode": "zero_waveform_smoke" if args.smoke else args.split}
    write_json(args.out / "run_metadata.json", metadata)
    write_json(args.out / "inference_flags.json", vars(flags))
    pip_freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True)
    (args.out / "environment_freeze.txt").write_text(pip_freeze.stdout, encoding="utf-8")
    started = time.perf_counter()
    config = tf.compat.v1.ConfigProto(device_count={"GPU": 0})
    with tf.compat.v1.Session(config=config) as session:
        tf.compat.v1.keras.backend.set_session(session)
        tf.keras.backend.set_learning_phase(0)
        processor = None if args.smoke else input_data.AudioProcessor(flags)
        model = kws_transformer.model(flags)
        kwts = [layer for layer in model.layers if isinstance(layer, KWSTransformer)]
        if len(kwts) != 1:
            raise RuntimeError("Expected exactly one time-domain KWT encoder")
        kwt = kwts[0]
        probe = build_probe(tf, model, kwt)
        session.run(tf.compat.v1.global_variables_initializer())
        status = model.load_weights(str(checkpoint))
        status.assert_existing_objects_matched()
        status.expect_partial()  # Extra checkpoint objects may be unused; all model variables must match.
        print("Checkpoint loaded; all current model objects matched.", flush=True)
        print("Input:", model.input_shape, "Output:", model.output_shape,
              "Parameters:", model.count_params(), flush=True)
        export_weights(np, session, kwt, model, args.out)

        candidates = [] if processor is None else processor.data_index[args.split]
        total = 1 if args.smoke else len(candidates)
        if not args.smoke and args.limit:
            total = min(total, args.limit)
        if total <= 0:
            raise ValueError("No evaluation samples")
        manifest = []
        for index, sample in enumerate(candidates[:total]):
            label_id = int(processor.word_to_index[sample["label"]])
            manifest.append({"index": index, "file": os.path.relpath(sample["file"], flags.data_dir),
                             "source_word": sample["label"], "label_id": label_id,
                             "label": labels[label_id],
                             "synthetic_zero_waveform": sample["label"] == "_silence_"})
        write_json(args.out / "sample_manifest.json", manifest)

        correct = 0
        confusion = np.zeros((12, 12), dtype=np.int64)
        all_logits = []
        all_targets = []
        trace_error = None
        with (args.out / "predictions.csv").open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.writer(stream)
            writer.writerow(["index", "source_file", "true_id", "true_label", "predicted_id", "predicted_label", "correct"])
            for offset in range(0, total, args.batch_size):
                count = min(args.batch_size, total - offset)
                if args.smoke:
                    audio = np.zeros((count, 16000), dtype=np.float32)
                    targets = np.full(count, -1, dtype=np.int64)
                else:
                    audio, targets = processor.get_data(count, offset, flags,
                                                       0.0, 0.0, 0, args.split, 0.0, 0.0, session)
                    audio = np.asarray(audio, dtype=np.float32)
                    targets = np.asarray(targets, dtype=np.int64)
                padded = np.zeros((args.batch_size, 16000), dtype=np.float32)
                padded[:count] = audio
                if offset == 0:
                    traced = session.run(probe, feed_dict={model.inputs[0]: padded})
                    np.testing.assert_allclose(traced["probe_cls"], traced["official_cls"], rtol=1e-5, atol=1e-5)
                    np.testing.assert_allclose(traced["probe_logits"], traced["logits"], rtol=1e-5, atol=1e-5)
                    trace_error = float(np.max(np.abs(traced["probe_cls"] - traced["official_cls"])))
                    for key, value in traced.items():
                        if not np.isfinite(value).all():
                            raise FloatingPointError("Non-finite tensor: " + key)
                    if traced["encoder_input"].shape[1:] != (99, 64):
                        raise ValueError("Unexpected encoder input shape")
                    first = {key: value[:1] for key, value in traced.items()}
                    first["waveform"] = audio[:1]
                    first["target_id"] = targets[:1]
                    np.savez_compressed(str(args.out / "first_sample_trace.npz"), **first)
                    write_json(args.out / "trace_shapes.json", {key: list(value.shape) for key, value in first.items()})
                    logits = traced["logits"][:count]
                    print("MFCC:", first["mfcc"].shape, "Encoder input:", first["encoder_input"].shape,
                          "Layer 12:", first["layer_12_output"].shape, flush=True)
                    print("Trace CLS vs official CLS max abs error:", trace_error, flush=True)
                else:
                    logits = session.run(model.outputs[0], feed_dict={model.inputs[0]: padded})[:count]
                if not np.isfinite(logits).all():
                    raise FloatingPointError("Non-finite logits")
                predictions = np.argmax(logits, axis=1)
                all_logits.append(logits)
                all_targets.append(targets)
                for row, (target, prediction) in enumerate(zip(targets, predictions)):
                    index = offset + row
                    matched = int(target == prediction)
                    source = "<synthetic_zero>" if args.smoke else manifest[index]["file"]
                    writer.writerow([index, source, "" if args.smoke else int(target),
                                     "" if args.smoke else labels[target], int(prediction), labels[prediction],
                                     "" if args.smoke else matched])
                    if not args.smoke:
                        correct += matched
                        confusion[target, prediction] += 1
                done = offset + count
                if args.smoke or offset == 0 or done == total or done % (args.batch_size * 10) == 0:
                    print("Processed {}/{}{}".format(done, total, "" if args.smoke else
                          "; accuracy so far {:.4%}".format(correct / done)), flush=True)

        np.savez_compressed(str(args.out / "predictions.npz"), logits=np.concatenate(all_logits),
                            target_ids=np.concatenate(all_targets), confusion_matrix=confusion)
        result = {"mode": metadata["mode"], "samples": total, "available_split_samples": len(candidates),
                  "correct": None if args.smoke else correct,
                  "accuracy": None if args.smoke else correct / total,
                  "trace_cls_max_abs_error": trace_error,
                  "elapsed_seconds": time.perf_counter() - started,
                  "note": "No accuracy is computed for the synthetic smoke input" if args.smoke else
                          "Accuracy includes silence and unknown. Partial runs are only pipeline checks."}
        write_json(args.out / "result.json", result)
        print(json.dumps(result, indent=2), flush=True)
        print("PASS: inference completed. Outputs:", args.out, flush=True)


if __name__ == "__main__":
    main()
