"""
MPI parallel inference: distribute the Chapman ECG workload across MPI ranks
(data parallelism), compute the local predictions on every rank, gather the
results back on rank 0, sort by file name and save the CSV.

Each rank runs the SAME inference algorithm on a DIFFERENT subset of files, so
the classification results must be identical to the sequential baseline.

Usage:
    mpiexec -n 4 python mpi_inference.py
    mpiexec -n 4 python mpi_inference.py --limit 1000
"""

import argparse
import sys

try:
    from mpi4py import MPI
except ImportError:
    print(
        "mpi4py is not installed. Install it together with an MPI runtime "
        "(Microsoft MPI / MS-MPI on Windows, OpenMPI on Linux), e.g.: "
        "pip install mpi4py",
        file=sys.stderr,
    )
    sys.exit(1)

import ecg_common as common


def main():
    parser = argparse.ArgumentParser(
        description="MPI parallel ECG inference (data parallelism)."
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
        default=str(common.DEFAULT_MPI_CSV),
        help=f"Output CSV path (default: {common.DEFAULT_MPI_CSV})",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip the consistency check against results/sequential_predictions.csv.",
    )
    args = parser.parse_args()

    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be a positive integer (or omitted)")

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # --- 1. Rank 0 finds the dataset and shares the file list with all ranks --
    if rank == 0:
        records = common.discover_records(args.data)
        if args.limit is not None:
            records = records[: args.limit]
        if not records:
            print("No Chapman ECG records found. Aborting.")
            comm.Abort(1)
        record_paths = [str(r) for r in records]
    else:
        record_paths = None

    # Broadcast: every rank receives the full, sorted identifier list.
    record_paths = comm.bcast(record_paths, root=0)
    total = len(record_paths)

    # --- 2. Deterministic workload partition (manual split by rank) ----------
    my_records = common.split_workload(record_paths, size, rank)

    if rank == 0:
        chunk_sizes = [len(common.split_workload(record_paths, size, r)) for r in range(size)]
        print(f"Dataset       : {total} records across {size} ranks -> chunks {chunk_sizes}")

    # Each rank owns its own engine instance / TFLite interpreter. The one-time
    # model load happens BEFORE the timer so it is excluded from the wall time,
    # matching the sequential baseline which also loads the model once.
    engine = common.get_engine()

    comm.Barrier()  # make sure every rank is ready before we start timing
    t_start = MPI.Wtime()

    # --- 3. Local inference ---------------------------------------------------
    local_rows = []
    local_failed = 0
    for record in my_records:
        try:
            signal, fs = common.load_chapman_signal(record)
            result = engine.predict(signal, sampling_rate=fs)
            row = common.make_result_row(record, result)
        except Exception as exc:  # keep the experiment alive on bad files
            local_failed += 1
            row = common.make_result_row(record, None, error=exc)
        local_rows.append(row)

    t_local = MPI.Wtime() - t_start
    comm.Barrier()
    elapsed = MPI.Wtime() - t_start

    print(f"[Rank {rank}/{size}] processed {len(my_records)} records in "
          f"{t_local:.4f}s (failed={local_failed})")

    # --- 4. Gather predictions on rank 0 --------------------------------------
    all_rows = comm.gather(local_rows, root=0)

    if rank == 0:
        rows = [row for chunk in all_rows for row in chunk]
        rows.sort(key=lambda r: r["file_name"])  # deterministic ordering
        common.write_csv(rows, args.out)

        print("=" * 55)
        print("MPI parallel inference finished.")
        print(f"MPI Processes : {size}")
        print(f"Total Records : {total}")
        print(f"Failed        : {sum(1 for r in rows if r['status'] == 'error')}")
        print(f"Saved         : {args.out}")
        print(f"Execution Time: {elapsed:.4f} seconds")

        # --- 5. Validation against the sequential baseline --------------------
        if not args.no_validate:
            seq_csv = common.DEFAULT_SEQUENTIAL_CSV
            if seq_csv.exists():
                stats = common.compare_predictions(seq_csv, args.out)
                consistency = (
                    100.0 * stats["matched"] / stats["compared"]
                    if stats["compared"]
                    else 0.0
                )
                print("=" * 55)
                print("Result validation vs sequential baseline.")
                print(f"Matching predictions : {stats['matched']} / {stats['compared']}")
                print(f"Prediction consistency: {consistency:.2f}%")
                if stats["mismatches"]:
                    print(
                        "Mismatched file(s)   : " + ", ".join(stats["mismatches"][:10])
                    )
                if stats["sequential_only"] or stats["mpi_only"]:
                    print(
                        f"Files only in one run: seq={stats['sequential_only']}, "
                        f"mpi={stats['mpi_only']}"
                    )
            else:
                print(
                    "Note: run `python sequential.py` first to enable the "
                    "sequential vs MPI consistency check."
                )

    MPI.Finalize()


if __name__ == "__main__":
    main()