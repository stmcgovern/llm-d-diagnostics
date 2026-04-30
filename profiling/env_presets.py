#!/usr/bin/env python3
"""Named environment variable presets for debugging and profiling vLLM/PyTorch.

Usage:
    python3 env_presets.py list
    python3 env_presets.py show <preset>
    python3 env_presets.py apply <preset> <deployment> [-n namespace]

"apply" prints the oc/kubectl commands — it does not run them.
"""

import argparse
import sys
from typing import NamedTuple


class Preset(NamedTuple):
    description: str
    vars: dict[str, str]


PRESETS: dict[str, Preset] = {
    "nccl-debug": Preset(
        "NCCL verbose logging — shows topology, ring/tree selection, transport.",
        {
            "NCCL_DEBUG": "INFO",
            "NCCL_DEBUG_SUBSYS": "ALL",
        },
    ),
    "nccl-trace": Preset(
        "NCCL timing per collective — less noisy than full debug.",
        {
            "NCCL_DEBUG": "INFO",
            "NCCL_DEBUG_SUBSYS": "INIT,COLL",
        },
    ),
    "flight-recorder": Preset(
        "NCCL flight recorder — ring buffer of collective ops, dumps on timeout or signal.",
        {
            "TORCH_FR_BUFFER_SIZE": "1000",
            "TORCH_NCCL_DUMP_ON_TIMEOUT": "1",
            "TORCH_NCCL_ENABLE_TIMING": "1",
        },
    ),
    "torch-trace": Preset(
        "torch.compile trace logs — feed to tlparse for HTML visualization.",
        {
            "TORCH_TRACE": "/tmp/torch_trace",
            "TORCH_LOGS": "+dynamo,+inductor",
        },
    ),
    "nixl-verbose": Preset(
        "NIXL/UCX transport debug — shows handshake, transfer state, xfer stats.",
        {
            "NIXL_LOG_LEVEL": "DEBUG",
            "UCX_LOG_LEVEL": "info",
            "VLLM_LOGGING_LEVEL": "DEBUG",
        },
    ),
    "memory-debug": Preset(
        "PyTorch CUDA memory snapshot — expandable segments + history recording.",
        {
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        },
    ),
    "cuda-memcheck": Preset(
        "Disable CUDA caching allocator for cuda-memcheck/compute-sanitizer. Very slow.",
        {
            "PYTORCH_NO_CUDA_MEMORY_CACHING": "1",
        },
    ),
    "all-debug": Preset(
        "Kitchen sink — NCCL debug + flight recorder + torch trace. Very noisy.",
        {
            "NCCL_DEBUG": "INFO",
            "NCCL_DEBUG_SUBSYS": "INIT,COLL",
            "TORCH_FR_BUFFER_SIZE": "1000",
            "TORCH_NCCL_DUMP_ON_TIMEOUT": "1",
            "TORCH_NCCL_ENABLE_TIMING": "1",
            "TORCH_TRACE": "/tmp/torch_trace",
            "TORCH_LOGS": "+dynamo,+inductor",
            "VLLM_LOGGING_LEVEL": "DEBUG",
        },
    ),
}


def cmd_list(args: argparse.Namespace) -> None:
    for name, preset in PRESETS.items():
        print(f"  {name:20s} {preset.description}")


def cmd_show(args: argparse.Namespace) -> None:
    preset = PRESETS.get(args.preset)
    if not preset:
        print(f"Unknown preset: {args.preset}", file=sys.stderr)
        print(f"Available: {', '.join(PRESETS)}", file=sys.stderr)
        sys.exit(1)
    print(f"# {args.preset}: {preset.description}")
    for k, v in preset.vars.items():
        print(f"export {k}={v}")


def cmd_apply(args: argparse.Namespace) -> None:
    preset = PRESETS.get(args.preset)
    if not preset:
        print(f"Unknown preset: {args.preset}", file=sys.stderr)
        sys.exit(1)
    cli = "oc"
    ns_flag = f" -n {args.namespace}" if args.namespace else ""
    env_pairs = " ".join(f"{k}={v}" for k, v in preset.vars.items())
    print(f"# Apply '{args.preset}' to {args.deployment}")
    print(f"# {preset.description}")
    print(f"{cli} set env deployment/{args.deployment}{ns_flag} {env_pairs}")
    print()
    print("# To remove:")
    env_unset = " ".join(f"{k}-" for k in preset.vars)
    print(f"{cli} set env deployment/{args.deployment}{ns_flag} {env_unset}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Environment variable presets for PyTorch/vLLM debugging"
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("list", help="List available presets")

    show = sub.add_parser("show", help="Show env vars for a preset")
    show.add_argument("preset")

    apply_ = sub.add_parser("apply", help="Print oc set env commands")
    apply_.add_argument("preset")
    apply_.add_argument("deployment", help="Deployment name")
    apply_.add_argument("-n", "--namespace", help="Kubernetes namespace")

    args = parser.parse_args()
    if args.command == "list":
        cmd_list(args)
    elif args.command == "show":
        cmd_show(args)
    elif args.command == "apply":
        cmd_apply(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
