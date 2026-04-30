#!/usr/bin/env python3
"""Configure and parse NCCL flight recorder dumps.

The NCCL flight recorder is a ring buffer of collective operations that
PyTorch maintains in-process. On timeout or crash, it dumps to a file
showing what each rank was doing — essential for diagnosing hangs in
tensor-parallel or expert-parallel vLLM deployments.

Usage:
    # Show env vars to enable flight recorder
    python3 nccl_flight_recorder.py setup

    # Parse a dump file
    python3 nccl_flight_recorder.py parse /tmp/nccl_trace_rank0.pkl

    # Parse all dumps in a directory
    python3 nccl_flight_recorder.py parse /tmp/nccl_dumps/
"""

import argparse
import os
import sys

FLIGHT_RECORDER_VARS = {
    "TORCH_FR_BUFFER_SIZE": ("1000", "Number of entries in ring buffer (was TORCH_NCCL_TRACE_BUFFER_SIZE before PyTorch 2.10)"),
    "TORCH_NCCL_DUMP_ON_TIMEOUT": ("1", "Dump buffer on NCCL timeout"),
    "TORCH_NCCL_ENABLE_TIMING": ("1", "Record per-op GPU timestamps"),
    "TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC": ("300", "Watchdog timeout (seconds)"),
    "TORCH_FR_CPP_STACK": ("0", "Include C++ stack traces (expensive; was TORCH_NCCL_TRACE_CPP_STACK before PyTorch 2.10)"),
}


def cmd_setup(args: argparse.Namespace) -> None:
    print("# NCCL Flight Recorder configuration")
    print("# Add these to your vLLM deployment env vars:")
    print()
    for var, (val, desc) in FLIGHT_RECORDER_VARS.items():
        print(f"# {desc}")
        print(f"export {var}={val}")
        print()
    print("# Dumps appear at: /tmp/nccl_trace_dump_<hostname>_rank<N>.pkl")
    print("# Trigger manually: kill -USR2 <vllm-pid>")


def cmd_parse(args: argparse.Namespace) -> None:
    import pickle

    paths = []
    target = args.path
    if os.path.isdir(target):
        for f in sorted(os.listdir(target)):
            if f.endswith(".pkl") and "nccl" in f.lower():
                paths.append(os.path.join(target, f))
        if not paths:
            print(f"No NCCL dump files (*.pkl) found in {target}", file=sys.stderr)
            sys.exit(1)
    elif os.path.isfile(target):
        paths.append(target)
    else:
        print(f"Not found: {target}", file=sys.stderr)
        sys.exit(1)

    for path in paths:
        print(f"\n{'='*60}")
        print(f"File: {path}")
        print(f"{'='*60}")

        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
        except Exception as e:
            print(f"  Failed to load: {e}", file=sys.stderr)
            continue

        if isinstance(data, dict):
            _parse_dict_format(data, args.last)
        elif isinstance(data, list):
            _parse_list_format(data, args.last)
        else:
            print(f"  Unknown format: {type(data)}")
            if hasattr(data, "__dict__"):
                for k, v in data.__dict__.items():
                    print(f"    {k}: {type(v)}")


def _flatten(val: object) -> str:
    """Flatten nested lists (e.g. [[4096, 1024]]) into a compact string."""
    if isinstance(val, list):
        while len(val) == 1 and isinstance(val[0], list):
            val = val[0]
        return "x".join(str(v) for v in val) if isinstance(val, list) else str(val)
    return str(val)


def _parse_dict_format(data: dict, last_n: int) -> None:
    for key in ["version", "pg_config", "pg_status"]:
        if key in data:
            print(f"\n  {key}: {data[key]}")

    entries = data.get("entries", data.get("collectives", []))
    if not entries:
        print("  No collective entries found")
        print(f"  Keys: {list(data.keys())}")
        return

    total = len(entries)
    show = entries[-last_n:] if last_n < total else entries
    print(f"\n  Collectives: {total} total, showing last {len(show)}")
    print(f"  {'idx':>5} {'op':>15} {'size':>12} {'dtype':>8} {'state':>10} {'time_ms':>10}")
    print(f"  {'-'*5} {'-'*15} {'-'*12} {'-'*8} {'-'*10} {'-'*10}")

    for entry in show:
        if isinstance(entry, dict):
            idx = entry.get("seq_id", entry.get("id", "?"))
            op = entry.get("profiling_name", entry.get("op", "?"))
            size = _flatten(entry.get("input_sizes", entry.get("size", "?")))
            dtype = _flatten(entry.get("input_dtypes", entry.get("dtype", "?")))
            state = entry.get("state", "?")
            duration = entry.get("duration_ms", entry.get("time_ms", "?"))
            print(f"  {str(idx):>5} {str(op):>15} {size:>12} {str(dtype):>8} {str(state):>10} {str(duration):>10}")
        else:
            print(f"  {entry}")


def _parse_list_format(data: list, last_n: int) -> None:
    total = len(data)
    show = data[-last_n:] if last_n < total else data
    print(f"\n  Entries: {total} total, showing last {len(show)}")
    for entry in show:
        print(f"  {entry}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="NCCL flight recorder setup and parsing"
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("setup", help="Show env vars to enable flight recorder")

    parse = sub.add_parser("parse", help="Parse a flight recorder dump")
    parse.add_argument("path", help="Path to .pkl dump file or directory")
    parse.add_argument(
        "--last", type=int, default=50, help="Show last N entries (default: 50)"
    )

    args = parser.parse_args()
    if args.command == "setup":
        cmd_setup(args)
    elif args.command == "parse":
        cmd_parse(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
