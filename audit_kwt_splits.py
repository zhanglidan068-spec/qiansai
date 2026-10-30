"""Audit local Speech Commands split membership without TensorFlow or predictions.

The membership function exactly mirrors kws_streaming.data.input_data.which_set.
Only WAV filenames, the two dataset split lists, and checkpoint flags are read.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import re


MAX_NUM_WAVS_PER_CLASS = 2 ** 27 - 1


def which_set(filename, validation_percentage, testing_percentage):
    base_name = str(filename).replace("\\", "/").rsplit("/", 1)[-1]
    hash_name = re.sub(r"_nohash_.*$", "", base_name)
    hashed = hashlib.sha1(hash_name.encode("utf-8")).hexdigest()
    percentage = ((int(hashed, 16) % (MAX_NUM_WAVS_PER_CLASS + 1)) *
                  (100.0 / MAX_NUM_WAVS_PER_CLASS))
    if percentage < validation_percentage:
        return "validation"
    if percentage < validation_percentage + testing_percentage:
        return "testing"
    return "training"


def _speaker(relative_path):
    return re.sub(r"_nohash_.*$", "", relative_path.rsplit("/", 1)[-1])


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_splits(data_dir, flags_path=None):
    """Return a JSON-serializable audit, reading no past experiment artifacts."""
    root = Path(os.path.abspath(str(data_dir)))
    if flags_path is None:
        flags_path = (Path(__file__).absolute().parent / "keyword-transformer" /
                      "models_data_v2_12_labels" / "kwt1" / "flags.json")
    flags_path = Path(os.path.abspath(str(flags_path)))
    flags = json.loads(flags_path.read_text(encoding="utf-8"))
    wanted_words = flags["wanted_words"].split(",")
    wanted_set = set(wanted_words)
    validation_percentage = float(flags["validation_percentage"])
    testing_percentage = float(flags["testing_percentage"])
    silence_percentage = float(flags["silence_percentage"])
    unknown_percentage = float(flags["unknown_percentage"])
    if not flags["split_data"]:
        raise ValueError("This audit expects the official automatic hash split.")

    raw = {key: set() for key in ("training", "validation", "testing")}
    word_counts = {key: Counter() for key in raw}
    for directory in os.scandir(str(root)):
        if not directory.is_dir() or directory.name == "_background_noise_":
            continue
        for item in os.scandir(directory.path):
            if item.is_file() and item.name.endswith(".wav"):
                relative = directory.name + "/" + item.name
                split = which_set(item.name, validation_percentage,
                                  testing_percentage)
                raw[split].add(relative)
                word_counts[split][directory.name.lower()] += 1

    list_sets = {}
    official_lists = {}
    for split in ("validation", "testing"):
        path = root / (split + "_list.txt")
        entries = [line.strip().replace("\\", "/")
                   for line in path.read_text(encoding="utf-8").splitlines()
                   if line.strip()]
        unique = set(entries)
        list_sets[split] = unique
        missing = sorted(relative for relative in unique
                         if not (root / relative).is_file())
        hash_mismatches = sorted(relative for relative in unique
                                 if which_set(relative, validation_percentage,
                                              testing_percentage) != split)
        official_lists[split] = {
            "path": str(path), "sha256": _sha256(path),
            "nonempty_lines": len(entries), "unique_files": len(unique),
            "duplicate_lines": len(entries) - len(unique),
            "speaker_count": len({_speaker(x) for x in unique}),
            "missing_wav_count": len(missing),
            "missing_wav_files": missing,
            "hash_mismatch_count": len(hash_mismatches),
            "hash_mismatch_files": hash_mismatches,
            "listed_but_not_in_local_hash_split": sorted(unique - raw[split]),
            "local_hash_split_but_not_listed": sorted(raw[split] - unique),
            "list_exactly_matches_local_hash_split": unique == raw[split],
        }

    speakers = {key: {_speaker(relative) for relative in paths}
                for key, paths in raw.items()}
    intersections = {}
    for left, right in (("training", "validation"), ("training", "testing"),
                        ("validation", "testing")):
        intersections[left + "_vs_" + right] = {
            "file_count": len(raw[left] & raw[right]),
            "speaker_count": len(speakers[left] & speakers[right]),
        }

    model_counts = {}
    for split in raw:
        keyword_count = sum(count for word, count in word_counts[split].items()
                            if word in wanted_set)
        unknown_available = len(raw[split]) - keyword_count
        unknown_count = min(unknown_available,
                            int(math.ceil(keyword_count * unknown_percentage / 100)))
        silence_count = int(math.ceil(keyword_count * silence_percentage / 100))
        model_counts[split] = {
            "raw_35_word_wav_count": len(raw[split]),
            "raw_speaker_count": len(speakers[split]),
            "wanted_keyword_wav_count": keyword_count,
            "unknown_pool_wav_count": unknown_available,
            "selected_unknown_count": unknown_count,
            "synthetic_silence_count": silence_count,
            "model_12_class_sample_count": keyword_count + unknown_count + silence_count,
            "per_word_raw_wav_count": dict(sorted(word_counts[split].items())),
        }

    checks = {
        "all_official_list_wavs_exist": all(
            value["missing_wav_count"] == 0 for value in official_lists.values()),
        "official_lists_have_no_duplicate_lines": all(
            value["duplicate_lines"] == 0 for value in official_lists.values()),
        "official_lists_exactly_match_upstream_hash_split": all(
            value["list_exactly_matches_local_hash_split"]
            for value in official_lists.values()),
        "all_raw_splits_file_disjoint": all(
            value["file_count"] == 0 for value in intersections.values()),
        "all_raw_splits_speaker_disjoint": all(
            value["speaker_count"] == 0 for value in intersections.values()),
    }
    return {
        "data_dir": str(root), "checkpoint_flags_path": str(flags_path),
        "checkpoint_flags_sha256": _sha256(flags_path),
        "membership_algorithm": "Upstream which_set: speaker prefix SHA1, modulo 2^27; divide by 2^27-1; validation first, then testing",
        "validation_percentage": validation_percentage,
        "testing_percentage": testing_percentage,
        "wanted_words": wanted_words,
        "unknown_percentage": unknown_percentage,
        "silence_percentage": silence_percentage,
        "official_lists": official_lists,
        "official_testing_vs_validation": {
            "file_intersection_count": len(list_sets["testing"] & list_sets["validation"]),
            "speaker_intersection_count": len(
                {_speaker(x) for x in list_sets["testing"]} &
                {_speaker(x) for x in list_sets["validation"]}),
        },
        "raw_split_intersections": intersections,
        "split_counts": model_counts,
        "checks": checks, "passed": all(checks.values()),
        "scope": "Checks the local filenames and speaker prefix partition. This does not independently audit how the checkpoint was trained or tuned.",
        "silence_note": "Upstream uses a training WAV path as a placeholder for silence in every split; evaluation replaces it with a zero waveform. Count synthetic silence separately when checking source file overlap.",
        "independence": "No prior run, prediction, logits, labels manifest, or quantization artifact is read.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path,
                        default=Path(__file__).absolute().parent / "data2")
    parser.add_argument("--flags", type=Path)
    parser.add_argument("--out", type=Path,
                        default=Path(__file__).absolute().parent / "testing_split_audit.json")
    args = parser.parse_args()
    result = audit_splits(args.data_dir, args.flags)
    # Exclusive creation preserves an earlier audit if the command is rerun.
    with args.out.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({"out": str(args.out), "passed": result["passed"],
                      "counts": {k: v["model_12_class_sample_count"]
                                 for k, v in result["split_counts"].items()}},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
