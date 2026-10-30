"""Locate checked-in regression metadata without depending on local reports.

The reference JSON files are byte-for-byte copies of immutable historical
results. New studies copy them into their own report bundles before replay.
Missing or changed fixtures are errors; this module never downloads, generates,
or substitutes a new expected result.
"""
import hashlib
import json
from pathlib import Path


REFERENCE_RESULTS = Path(__file__).absolute().parent / "reference_results"
REFERENCE_FILES = {
    "fixed300_20261029_v2": ("summary.json", "run_config.json"),
    "architecture_study_20261029_v1": ("summary.json", "run_config.json"),
    "architecture_study_ln64_20261029_v1": (
        "summary.json", "run_config.json", "historical_regression.json"),
}


def reference_report_path(reference_id):
    """Return a verified, versioned reference directory for a known study."""
    if reference_id not in REFERENCE_FILES:
        raise ValueError("unknown packaged regression reference: {!r}".format(reference_id))
    manifest_path = REFERENCE_RESULTS / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            "Packaged regression reference manifest is missing: {}. "
            "Restore hw_model/reference_results from the repository.".format(manifest_path))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or manifest.get("algorithm") != "sha256":
        raise ValueError("unsupported packaged regression reference manifest")
    files = manifest.get("files", {})
    directory = REFERENCE_RESULTS / reference_id
    for filename in REFERENCE_FILES[reference_id]:
        path = directory / filename
        if not path.is_file():
            raise FileNotFoundError(
                "Required packaged regression reference is missing: {}. "
                "Restore hw_model/reference_results from the repository; "
                "expected results are not generated automatically.".format(path))
        expected = files.get(reference_id + "/" + filename)
        if not isinstance(expected, dict):
            raise ValueError("packaged regression reference has no checksum: {}".format(path))
        content = path.read_bytes()
        if (len(content) != expected.get("bytes") or
                hashlib.sha256(content).hexdigest() != expected.get("sha256")):
            raise ValueError("packaged regression reference checksum mismatch: {}".format(path))
    return directory
