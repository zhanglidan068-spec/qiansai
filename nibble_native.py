"""ctypes CPU reference for nibble-product LUT matrix multiplication.

Public API::

    engine = ApproxDenseEngine(new_run / "native", threads=4)
    output, exit_counts = engine.matmul(a, w, product_lut, stage_lut)

``a`` and ``w`` contain symmetric INT8 qcodes transported as int64 arrays.
All LUTs use axes ``[activation + 127, weight + 127]``.  ``exit_counts[i]``
counts multiplication pairs ending after i+1 four-bit terms; it is zero
unless collect_stats=True.  Dot products accumulate exactly in int32, after
an overflow proof, and return int64.  No M*K*N broadcast tensor is built.

The runtime compiles only the adjacent nibble_matmul.cpp, stores its source
and build manifest in the caller's new output directory, and never reads
audio, labels, checkpoints, predictions, or earlier experiment artifacts.
"""

import ctypes
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np


DEFAULT_COMPILER = r"D:\Xilinx\Vivado\2020.2\tps\win64\msys64\mingw64\bin\g++.exe"
_DLL_DIRECTORY_HANDLES = []


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compile_backend(output_directory, compiler=None, library_path=None):
    """Build a native DLL once in a new run directory; return its path.

    Existing binaries are reused only if their manifest and source hash match.
    library_path permits a previously compiled *code-only* DLL to be copied
    into a new run; this does not use any model outputs or experiment data.
    """
    target = Path(output_directory).absolute()
    target.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).absolute().with_name("nibble_matmul.cpp")
    source_hash = _sha256(source)
    destination = target / ("nibble_matmul.dll" if os.name == "nt" else "nibble_matmul.so")
    manifest_path = target / "build_manifest.json"
    if destination.exists():
        if not manifest_path.exists():
            raise FileExistsError("Refusing to overwrite native library without a manifest: " + str(destination))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("source_sha256") != source_hash or manifest.get("library_sha256") != _sha256(destination):
            raise FileExistsError("Native library/source differs; use a fresh output directory")
        return destination

    compiler = str(compiler or os.environ.get("KWT_CXX", DEFAULT_COMPILER))
    flags = ["-O3", "-std=c++11", "-shared", "-fopenmp", "-march=native",
             "-static", "-static-libgcc", "-static-libstdc++"]
    snapshot = target / "nibble_matmul.cpp"
    if snapshot.exists() and _sha256(snapshot) != source_hash:
        raise FileExistsError("Refusing to overwrite a different source snapshot")
    if not snapshot.exists():
        shutil.copy2(str(source), str(snapshot))
    started = time.perf_counter()
    if library_path is not None:
        previous = Path(library_path).absolute()
        previous_manifest = previous.with_name("build_manifest.json")
        if not previous_manifest.exists():
            raise ValueError("Precompiled library needs its build_manifest.json")
        inherited = json.loads(previous_manifest.read_text(encoding="utf-8"))
        if inherited.get("source_sha256") != source_hash or inherited.get("library_sha256") != _sha256(previous):
            raise ValueError("Precompiled library/source hash mismatch")
        shutil.copy2(str(previous), str(destination))
        command = inherited.get("command")
        build_stdout = "Reused verified code-only native library: " + str(previous)
    else:
        command = [compiler] + flags + [str(snapshot), "-o", str(destination)]
        build_environment = os.environ.copy()
        # GCC's cc1plus subprocess also loads DLLs shipped beside g++.exe;
        # unlike Python's add_dll_directory, this PATH applies to child tools.
        build_environment["PATH"] = str(Path(compiler).parent) + os.pathsep + build_environment.get("PATH", "")
        try:
            result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, check=False, env=build_environment)
        except OSError as error:
            raise RuntimeError("Could not start local compiler; the outer Python invocation may need execution permission: " + str(error))
        build_stdout = result.stdout
        (target / "build_output.txt").write_text(build_stdout, encoding="utf-8")
        if result.returncode != 0:
            raise RuntimeError("Native compilation failed: " + build_stdout)
    manifest = {
        "source_sha256": source_hash,
        "library_sha256": _sha256(destination),
        "compiler": compiler,
        "command": command,
        "build_seconds": time.perf_counter() - started,
        "reused_library": str(library_path) if library_path else None,
        "lookup_axes": ["activation+127", "weight+127"],
        "accumulator": "int32 exact after K*max(abs(LUT)) overflow proof",
        "output_container": "int64",
        "stage_counts": "multiplication pairs ending after 1,2,3,4 terms",
        "hardware_speedup_claim": False,
    }
    (target / "build_output.txt").write_text(build_stdout, encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return destination


class ApproxDenseEngine:
    def __init__(self, output_directory, threads=4, compiler=None, library_path=None):
        if int(threads) != threads or threads < 1:
            raise ValueError("threads must be a positive integer")
        self.threads = int(threads)
        self.library_path = compile_backend(output_directory, compiler, library_path)
        compiler = str(compiler or os.environ.get("KWT_CXX", DEFAULT_COMPILER))
        if os.name == "nt" and hasattr(os, "add_dll_directory"):
            for directory in (str(self.library_path.parent), str(Path(compiler).parent)):
                if Path(directory).is_dir():
                    _DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(directory))
        self._dll = ctypes.CDLL(str(self.library_path))
        i64 = np.ctypeslib.ndpointer(dtype=np.int64, flags="C_CONTIGUOUS")
        i32 = np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")
        u8 = np.ctypeslib.ndpointer(dtype=np.uint8, flags="C_CONTIGUOUS")
        self._fn = self._dll.nibble_matmul
        self._fn.argtypes = [i64, i64, i32, u8, i64, i64] + [ctypes.c_int] * 5
        self._fn.restype = ctypes.c_int
        self._dll.nibble_backend_features.argtypes = []
        self._dll.nibble_backend_features.restype = ctypes.c_char_p
        self.features = self._dll.nibble_backend_features().decode("ascii")

    def matmul(self, a, w, product_lut, stage_lut, collect_stats=False):
        a = np.ascontiguousarray(a, dtype=np.int64)
        w = np.ascontiguousarray(w, dtype=np.int64)
        if a.ndim != 2 or w.ndim != 2 or a.shape[1] != w.shape[0]:
            raise ValueError("Expected a[M,K] and w[K,N]")
        if any(d > 2147483647 for d in a.shape + w.shape):
            raise ValueError("Matrix dimension exceeds native ABI")
        product_lut = np.ascontiguousarray(product_lut, dtype=np.int32)
        stage_lut = np.ascontiguousarray(stage_lut, dtype=np.uint8)
        if product_lut.shape != (255, 255) or stage_lut.shape != (255, 255):
            raise ValueError("Both LUTs must have shape [255,255]")
        bound = max(abs(int(product_lut.min())), abs(int(product_lut.max())))
        if a.shape[1] * bound > np.iinfo(np.int32).max:
            raise OverflowError("LUT products could overflow the signed int32 accumulator")
        output = np.empty((a.shape[0], w.shape[1]), dtype=np.int64)
        counts = np.zeros(4, dtype=np.int64)
        status = self._fn(a, w, product_lut, stage_lut, output, counts,
                          a.shape[0], a.shape[1], w.shape[1], self.threads,
                          int(bool(collect_stats)))
        if status:
            errors = {1: "invalid dimensions", 2: "qcode outside symmetric INT8 domain [-127,127]",
                      3: "observed stage_lut value outside 1..4"}
            raise ValueError(errors.get(status, "native error " + str(status)))
        return output, counts


def self_test_and_benchmark(output_directory, threads=4, library_path=None):
    """Small independent NumPy oracles plus representative FFN timings."""
    engine = ApproxDenseEngine(output_directory, threads=threads, library_path=library_path)
    rng = np.random.RandomState(20261030)
    domain = np.arange(-127, 128, dtype=np.int32)
    exact = domain[:, None] * domain[None, :]
    stages = np.full((255, 255), 4, dtype=np.uint8)
    checks = []
    for m, k, n in ((1, 1, 1), (3, 7, 11), (5, 64, 64), (2, 256, 256), (7, 99, 32)):
        a = rng.randint(-127, 128, (m, k)).astype(np.int64)
        w = rng.randint(-127, 128, (k, n)).astype(np.int64)
        output, counts = engine.matmul(a, w, exact, stages, True)
        np.testing.assert_array_equal(output, a.dot(w))
        np.testing.assert_array_equal(counts, [0, 0, 0, m * k * n])
        approximate = rng.randint(-16129, 16130, (255, 255)).astype(np.int32)
        exits = rng.randint(1, 5, (255, 255)).astype(np.uint8)
        output, counts = engine.matmul(a, w, approximate, exits, True)
        # Small oracle deliberately uses explicit K and has no native dependency.
        reference = np.zeros((m, n), dtype=np.int64)
        reference_counts = np.zeros(4, dtype=np.int64)
        for index in range(k):
            indices_a = a[:, index, None] + 127
            indices_w = w[index, None, :] + 127
            reference += approximate[indices_a, indices_w]
            reference_counts += np.bincount(exits[indices_a, indices_w].ravel(), minlength=5)[1:5]
        np.testing.assert_array_equal(output, reference)
        np.testing.assert_array_equal(counts, reference_counts)
        checks.append({"shape": [m, k, n], "exact_and_asymmetric_random_lut": "pass"})
    timings = []
    for m, k, n in ((1584, 64, 256), (1584, 256, 64)):
        a = rng.randint(-127, 128, (m, k)).astype(np.int64)
        w = rng.randint(-127, 128, (k, n)).astype(np.int64)
        engine.matmul(a, w, exact, stages, False)
        for stats in (False, True):
            started = time.perf_counter()
            repeats = 3
            for _ in range(repeats):
                output, counts = engine.matmul(a, w, exact, stages, stats)
            seconds = (time.perf_counter() - started) / repeats
            np.testing.assert_array_equal(output, a.dot(w))
            timings.append({"shape": [m, k, n], "collect_stats": stats,
                            "seconds": seconds, "integer_products_per_second": m * k * n / seconds})
    result = {"features": engine.features, "threads": threads, "oracle_checks": checks,
              "benchmarks": timings, "hardware_performance_claim": False}
    (Path(output_directory) / "self_test_benchmark.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--library-path")
    args = parser.parse_args()
    print(json.dumps(self_test_and_benchmark(args.output_directory, args.threads, args.library_path), indent=2))
