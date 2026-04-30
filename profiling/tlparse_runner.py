#!/usr/bin/env python3
"""Parse torch.compile trace logs into browsable HTML using tlparse.

tlparse is a Rust-based parser for TORCH_TRACE logs. It produces an HTML
report showing compilation steps, graph breaks, FX graphs, and inductor
output code — useful for understanding why torch.compile is slow or
producing suboptimal code.

Requires: pip install tlparse

Usage:
    # Collect traces from a pod
    oc exec POD -n NS -- ls /tmp/torch_trace/
    oc cp NS/POD:/tmp/torch_trace/ ./torch_traces/

    # Parse a trace file or directory of traces
    python3 tlparse_runner.py parse ./torch_traces/
    python3 tlparse_runner.py parse ./trace.log -o ./report/

    # Just check if tlparse is installed
    python3 tlparse_runner.py check
"""

import argparse
import os
import shutil
import subprocess
import sys


def _find_tlparse() -> str | None:
    return shutil.which("tlparse")


def _check_install() -> bool:
    path = _find_tlparse()
    if path:
        result = subprocess.run(
            [path, "--version"], capture_output=True, text=True
        )
        version = result.stdout.strip() or result.stderr.strip()
        print(f"tlparse found: {path}")
        print(f"  Version: {version}")
        return True
    else:
        print("tlparse not found.", file=sys.stderr)
        print("Install with: pip install tlparse", file=sys.stderr)
        return False


def cmd_check(args: argparse.Namespace) -> None:
    if not _check_install():
        sys.exit(1)


def _run_tlparse(tlparse: str, trace_file: str, output_dir: str, overwrite: bool) -> bool:
    cmd = [tlparse, trace_file, "-o", output_dir, "--no-browser"]
    if overwrite:
        cmd.append("--overwrite")

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)

    if result.returncode != 0:
        print(f"\ntlparse exited with code {result.returncode}", file=sys.stderr)
        return False

    index = os.path.join(output_dir, "index.html")
    if os.path.exists(index):
        size_mb = sum(
            os.path.getsize(os.path.join(dp, f))
            for dp, _, fns in os.walk(output_dir)
            for f in fns
        ) / (1024 * 1024)
        print(f"\nReport generated: {index} ({size_mb:.1f} MB)")
        print(f"Open in browser: file://{os.path.abspath(index)}")
    else:
        print(f"\nOutput directory: {output_dir}")
    return True


def cmd_parse(args: argparse.Namespace) -> None:
    tlparse = _find_tlparse()
    if not tlparse:
        print("tlparse not found. Install with: pip install tlparse", file=sys.stderr)
        sys.exit(1)

    target = args.trace_path

    if os.path.isfile(target):
        output_dir = args.output or os.path.join(os.path.dirname(target), "tlparse_output")
        if not _run_tlparse(tlparse, target, output_dir, args.overwrite):
            sys.exit(1)
    elif os.path.isdir(target):
        trace_files = sorted(
            os.path.join(target, f) for f in os.listdir(target)
            if not f.startswith(".") and f.endswith(".log")
        )
        if not trace_files:
            print(f"No .log trace files in {target}", file=sys.stderr)
            sys.exit(1)
        print(f"Found {len(trace_files)} trace files in {target}")
        for trace_file in trace_files:
            name = os.path.splitext(os.path.basename(trace_file))[0]
            output_dir = args.output or os.path.join(target, f"tlparse_{name}")
            if not _run_tlparse(tlparse, trace_file, output_dir, args.overwrite):
                sys.exit(1)
    else:
        print(f"Not found: {target}", file=sys.stderr)
        sys.exit(1)


def cmd_collect(args: argparse.Namespace) -> None:
    """Print commands to collect traces from a pod."""
    pod = args.pod
    ns = args.namespace
    trace_path = args.trace_path
    local_dir = args.local_dir

    print(f"# Collect torch.compile traces from {pod}")
    print(f"# 1. Check traces exist:")
    print(f"oc exec {pod} -n {ns} -- ls {trace_path}/")
    print()
    print(f"# 2. Copy to local machine:")
    print(f"oc cp {ns}/{pod}:{trace_path}/ {local_dir}")
    print()
    print(f"# 3. Parse into HTML:")
    print(f"python3 {__file__} {local_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse torch.compile trace logs with tlparse"
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("check", help="Check if tlparse is installed")

    parse = sub.add_parser("parse", help="Parse trace file or directory into HTML report")
    parse.add_argument("trace_path", help="Trace log file or directory containing TORCH_TRACE output")
    parse.add_argument("-o", "--output", help="Output directory for HTML report")
    parse.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing output"
    )

    collect = sub.add_parser("collect", help="Print commands to collect traces from a pod")
    collect.add_argument("pod", help="Pod name")
    collect.add_argument("-n", "--namespace", required=True, help="Namespace")
    collect.add_argument(
        "--trace-path", default="/tmp/torch_trace",
        help="Trace directory inside pod (default: /tmp/torch_trace)",
    )
    collect.add_argument(
        "--local-dir", default="./torch_traces",
        help="Local directory to copy traces to (default: ./torch_traces)",
    )

    args = parser.parse_args()

    if args.command == "check":
        cmd_check(args)
    elif args.command == "parse":
        cmd_parse(args)
    elif args.command == "collect":
        cmd_collect(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
