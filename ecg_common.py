"""
Shared helpers for the ECG-MPI-Inference pipeline.

This module keeps `sequential.py` and `mpi_inference.py` thin by centralizing
the glue code between the new scripts and the existing black-box ECG Edge AI
Engine, plus the Chapman dataset access and the result bookkeeping.

The engine is treated as a black box: we do NOT re-implement or refactor its
internal DSP / TFLite pipeline. We only make it importable without a `pip
install` step and expose a single predict() call.
"""

import csv
import importlib.util
import logging
import os
import sys
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parent
ENGINE_SRC_DIR = ROOT_DIR / "ecg-edge-ai-engine" / "src"
DEFAULT_DATA_DIR = ROOT_DIR / "data" / "chapman"
DEFAULT_RESULTS_DIR = ROOT_DIR / "results"

DEFAULT_SEQUENTIAL_CSV = DEFAULT_RESULTS_DIR / "sequential_predictions.csv"
DEFAULT_MPI_CSV = DEFAULT_RESULTS_DIR / "mpi_predictions.csv"

_ENGINE_LOADED = False


# ---------------------------------------------------------------------------
# ECG Edge AI Engine glue (importable without pip install)
# ---------------------------------------------------------------------------

def _load_engine_package() -> None:
    """Register ``ecg-edge-ai-engine/src`` as the ``ecg_edge_ai_engine`` package.

    The engine's ``setup.py`` maps the import name ``ecg_edge_ai_engine`` to the
    ``src/`` directory, so instead of requiring ``pip install -e .`` we register
    the same mapping at runtime. Relative imports inside the engine keep working
    because the module is registered under ``sys.modules``.
    """
    if not (ENGINE_SRC_DIR / "__init__.py").exists():
        raise RuntimeError(
            f"ECG Edge AI Engine source not found at: {ENGINE_SRC_DIR}"
        )

    spec = importlib.util.spec_from_file_location(
        "ecg_edge_ai_engine",
        ENGINE_SRC_DIR / "__init__.py",
        submodule_search_locations=[str(ENGINE_SRC_DIR)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["ecg_edge_ai_engine"] = module
    spec.loader.exec_module(module)


def get_engine():
    """Build one ECGEngine instance for the current process.

    Every MPI rank calls this separately, so each rank owns an independent
    engine instance / TFLite interpreter (no shared mutable state between
    processes). The result is cached as a module-level singleton.
    """
    global _ENGINE_LOADED
    if not _ENGINE_LOADED:
        # Keep TensorFlow / oneDNN chatter off the terminal of every rank.
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
        import warnings  # noqa: PLC0415

        warnings.filterwarnings(
            "ignore",
            message=r".*Interpreter is deprecated.*",
            module=r"tensorflow.*",
        )
        _load_engine_package()
        _quiet_engine_loggers()
        _ENGINE_LOADED = True

    from ecg_edge_ai_engine import ECGEngine  # noqa: PLC0415

    return ECGEngine()


def _quiet_engine_loggers() -> None:
    """Demote the engine's INFO logs to WARNING (one line per DSP step per
    record would otherwise flood the terminal on large datasets)."""
    for name, logger in logging.Logger.manager.loggerDict.items():
        if isinstance(logger, logging.Logger) and name.startswith("ecg_edge_ai_engine"):
            logger.setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Chapman dataset access (WFDB records, 3-lead I / II / III)
# ---------------------------------------------------------------------------

def discover_records(data_dir):
    """Return a deterministic (sorted) list of Chapman record paths.

    Each element is the record identifier without extension, which is how
    ``wfdb.rdrecord()`` expects it (it will look for ``<id>.hea`` and
    ``<id>.mat`` next to each other).
    """
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Chapman data directory not found: {data_dir}")

    hea_files = sorted(data_dir.rglob("*.hea"))
    return [p.with_suffix("") for p in hea_files]


def load_chapman_signal(record_path):
    """Load one Chapman record and return ``(signal, fs)``.

    ``signal`` is the first three leads (I, II, III) as a ``float32`` array of
    shape ``(N, 3)`` and ``fs`` is the native sampling rate (500 Hz). The engine
    resamples internally to its target rate (250 Hz).
    """
    import wfdb  # noqa: PLC0415

    record = wfdb.rdrecord(str(record_path), channels=[0, 1, 2])
    signal = record.p_signal.astype(np.float32)
    return signal, float(record.fs)


# ---------------------------------------------------------------------------
# Workload distribution (data parallelism across MPI ranks)
# ---------------------------------------------------------------------------

def split_workload(items, size, rank):
    """Deterministically partition ``items`` among ``size`` ranks.

    Every caller (rank) must pass the same ``items`` list. The list is split
    into contiguous chunks: ``base = n // size`` items each, and the remainder
    ``n % size`` items are handed to the first ranks one by one. No item is
    dropped and no item is processed twice, even when ``n`` is not divisible
    by ``size`` (e.g. 1003 files / 4 ranks -> 251, 251, 251, 250).
    """
    n = len(items)
    base, extra = divmod(n, size)

    start = rank * base + min(rank, extra)
    end = start + base + (1 if rank < extra else 0)

    return items[start:end]


# ---------------------------------------------------------------------------
# Result records & CSV persistence
# ---------------------------------------------------------------------------

BASE_FIELDS = ["file_name", "prediction", "confidence", "status", "error"]


def make_result_row(record_path, result, error=""):
    """Flatten one engine result (or an error) into a CSV-compatible row."""
    row = {
        "file_name": Path(record_path).stem,
        "prediction": "",
        "confidence": "",
        "status": "error" if error else "ok",
        "error": str(error),
    }
    if result is not None:
        row["prediction"] = result.get("prediction", "")
        row["confidence"] = result.get("confidence", "")
        for label, prob in (result.get("probabilities") or {}).items():
            row[f"prob_{label}"] = prob
    return row


def write_csv(rows, out_path):
    """Write prediction rows to ``out_path`` (directory is created if needed)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Union of all keys, preserving insertion order (rows may add prob_* cols).
    fieldnames = list(dict.fromkeys(k for r in rows for k in r))
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return out_path


def read_csv(path):
    """Load a prediction CSV back as a list of row dicts."""
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def compare_predictions(seq_path, mpi_path):
    """Compare two prediction CSVs using ``file_name`` as the key.

    Only records that succeeded (status == "ok") in BOTH files are compared.
    Returns a dict with counts plus the list of mismatching file names.
    """
    seq = {
        r["file_name"]: r
        for r in read_csv(seq_path)
        if r.get("status") == "ok"
    }
    mpi = {
        r["file_name"]: r
        for r in read_csv(mpi_path)
        if r.get("status") == "ok"
    }

    common = set(seq) & set(mpi)
    mismatches = [
        name for name in sorted(common)
        if seq[name].get("prediction") != mpi[name].get("prediction")
    ]

    return {
        "compared": len(common),
        "matched": len(common) - len(mismatches),
        "mismatches": mismatches,
        "sequential_only": len(set(seq) - set(mpi)),
        "mpi_only": len(set(mpi) - set(seq)),
    }