"""
Sequential baseline: process every Chapman ECG file one by one using the ECG
Edge AI Engine, then save the predictions as a CSV.

This script is the single-process reference used to:
  * establish the baseline execution time (T1),
  * validate that the MPI version produces identical classification results.

Usage:
    python sequential.py
    python sequential.py --limit 1000
"""

import argparse
import sys
import time

import ecg_common as common


def main():
    parser = argparse.ArgumentParser(
        description="Sequential ECG inference baseline (single process)."
    )
    parser.add_argument(
        "--data",
        default=str(common.DEFAULT_DATA_DIR),
        help=f"Chapman WFDB data directory (default: {common.DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N records (useful for quick experiments).",
    )
    parser.add_argument(
        "--out",
        default=str(common.DEFAULT_SEQUENTIAL_CSV),
        help=f"Output CSV path (default: {common.DEFAULT_SEQUENTIAL_CSV})",
    )
    args = parser.parse_args()

    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be a positive integer (or omitted)")

    records = common.discover_records(args.data)
    if args.limit is not None:
        records = records[: args.limit]
    if not records:
        print("No Chapman ECG records found. Aborting.")
        sys.exit(1)

    total = len(records)
    print(f"Sequential inference on {total} ECG records ...")
    print("Loading ECG Edge AI Engine (once) ...")

    # One engine instance for the whole process; model load is one-time and
    # excluded from the measured execution time.
    engine = common.get_engine()

    rows = []
    failed = 0
    start = time.perf_counter()

    for i, record in enumerate(records, 1):
        try:
            signal, fs = common.load_chapman_signal(record)
            result = engine.predict(signal, sampling_rate=fs)
            row = common.make_result_row(record, result)
        except Exception as exc:  # keep the experiment alive on bad files
            failed += 1
            row = common.make_result_row(record, None, error=exc)
        rows.append(row)

        if i % 50 == 0 or i == total:
            print(f"  [{i}/{total}] processed {row['file_name']} ...")

    elapsed = time.perf_counter() - start

    common.write_csv(rows, args.out)

    print("=" * 55)
    print("Sequential baseline finished.")
    print(f"Total Records : {total}")
    print(f"Failed        : {failed}")
    print(f"Saved         : {args.out}")
    print(f"Execution Time: {elapsed:.4f} seconds")


if __name__ == "__main__":
    main()