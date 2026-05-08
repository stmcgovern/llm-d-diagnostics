"""
Model sweep: deploy, measure, undeploy for each model in a JSON config.

Usage:
    python3 toolkit/sweep.py sweeps/kv-ratio.json

Config format:
    {
      "base": "clusters/rdu3-t4x5-phi3",
      "experiments": ["decompose", "latency", "throughput", "seqlen"],
      "models": [
        {"model": "microsoft/Phi-3-mini-4k-instruct"},
        {"model": "Qwen/Qwen2.5-1.5B-Instruct", "max_model_len": 4096}
      ]
    }

Each model gets deployed, measured, and torn down. Results land in
clusters/_sweep/<model-slug>/data/ for consumption by kv_sweep.py.
"""

import json
import os
import re
import signal
import subprocess
import sys


def parse_env_sh(path):
    """Extract export VAR=value pairs from an env.sh file."""
    env = {}
    with open(path) as f:
        for line in f:
            m = re.match(r'''export\s+(\w+)=["']?([^"'\n]*)["']?''', line.strip())
            if m:
                env[m.group(1)] = m.group(2)
    return env


def model_slug(model_id):
    return model_id.split("/")[-1].lower()


def write_env_sh(path, base_env, overrides):
    """Write an env.sh that merges base values with per-model overrides."""
    merged = {**base_env, **overrides}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("#!/bin/bash\n")
        for key, val in sorted(merged.items()):
            if "${" in val:
                f.write(f'export {key}="{val}"\n')
            else:
                f.write(f'export {key}="{val}"\n')


def run(cmd, label, dry_run=False):
    print(f"  [{label}]  {' '.join(cmd)}")
    if dry_run:
        return
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"{label} failed (exit {result.returncode})")


def main():
    dry_run = "--dry-run" in sys.argv
    args = [a for a in sys.argv[1:] if a != "--dry-run"]
    config_path = args[0]

    with open(config_path) as f:
        config = json.load(f)

    base_dir = config["base"]
    base_env = parse_env_sh(os.path.join(base_dir, "env.sh"))
    experiments = config.get("experiments", ["characterize"])
    models = config["models"]
    ns = base_env.get("NS", "llm-d")

    mode = "DRY RUN" if dry_run else "LIVE"
    print(f"Sweep ({mode}): {len(models)} models, {len(experiments)} experiments each")
    print(f"Base:  {base_dir} (namespace={ns})")
    print()

    current_cluster_dir = None

    def cleanup(signum=None, frame=None):
        if current_cluster_dir and not dry_run:
            print(f"\nCleaning up: undeploy {current_cluster_dir}")
            subprocess.run(
                ["scripts/undeploy.sh", current_cluster_dir],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        sys.exit(1)

    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    completed = []
    failed = []

    for i, entry in enumerate(models, 1):
        model_id = entry["model"]
        slug = model_slug(model_id)
        cluster_dir = os.path.join("clusters", "_sweep", slug)
        data_dir = os.path.join(cluster_dir, "data")

        print(f"{'='*60}")
        print(f"  [{i}/{len(models)}] {model_id}")
        print(f"  Output: {data_dir}")
        print(f"{'='*60}")

        overrides = {
            "MODEL": model_id,
            "DATA_DIR": data_dir,
        }
        env_map = {
            "max_model_len": "MAX_MODEL_LEN",
            "gpu_memory_utilization": "GPU_MEMORY_UTILIZATION",
            "dtype": "DTYPE",
        }
        for json_key, env_key in env_map.items():
            if json_key in entry:
                overrides[env_key] = str(entry[json_key])

        ns_val = base_env.get("NS", "llm-d")
        overrides["PREFILL_HOST"] = f"vllm-prefill-svc.{ns_val}.svc.cluster.local:8100"

        env_path = os.path.join(cluster_dir, "env.sh")
        write_env_sh(env_path, base_env, overrides)

        if dry_run:
            print(f"  Generated: {env_path}")
            with open(env_path) as f:
                for line in f:
                    if line.startswith("export MODEL=") or line.startswith("export DATA_DIR="):
                        print(f"    {line.rstrip()}")

        current_cluster_dir = cluster_dir

        try:
            run(["scripts/deploy.sh", cluster_dir], "deploy", dry_run)

            for exp in experiments:
                try:
                    run(["toolkit/run.sh", cluster_dir, exp], exp, dry_run)
                except RuntimeError as e:
                    print(f"  WARNING: {e}")

            run(["scripts/undeploy.sh", cluster_dir], "undeploy", dry_run)
            current_cluster_dir = None
            completed.append(model_id)
            print(f"  DONE: {slug}\n")

        except RuntimeError as e:
            print(f"  FAILED: {e}")
            failed.append((model_id, str(e)))
            if not dry_run:
                try:
                    run(["scripts/undeploy.sh", cluster_dir], "undeploy (cleanup)")
                except RuntimeError:
                    pass
            current_cluster_dir = None
            print()

    print(f"{'='*60}")
    print(f"  Sweep complete: {len(completed)} ok, {len(failed)} failed")
    if completed:
        print(f"  Completed: {', '.join(model_slug(m) for m in completed)}")
    if failed:
        print(f"  Failed:    {', '.join(model_slug(m) for m, _ in failed)}")
    print()
    print(f"  Analyze with:")
    print(f"    python3 advisor/kv_sweep.py clusters/_sweep/*/data/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
