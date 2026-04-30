#!/usr/bin/env python3
"""
Experiment 9: Model Load Time Decomposition

Measures cold-start time by killing a decode pod and parsing startup logs from
its replacement. Decomposes total startup into phases: pod scheduling, weight
loading, NIXL initialization, and model compilation/warmup.

Why this matters: cold-start time directly affects rolling update duration,
scale-up latency, and fault recovery time. Understanding which phase dominates
tells you where to invest (faster storage, model caching, pre-warming).

DESTRUCTIVE: This experiment kills decode pods. It verifies at least 2 decode
pods exist before proceeding so the deployment stays available.

Usage:
    python3 toolkit/exp9_model_load.py
    LOAD_RUNS=5 python3 toolkit/exp9_model_load.py

Env vars:
    NS              Kubernetes namespace (default: default)
    DATA_DIR        Output directory (default: data)
    LOAD_RUNS       Number of kill/measure cycles (default: 3)
    DECODE_SELECTOR Label selector for decode pods (default: app=vllm-decode)
    VLLM_CONTAINER  Container name for vLLM logs (default: vllm)
    STARTUP_TIMEOUT Max seconds to wait for pod Ready (default: 600)

Outputs:
    data/exp9-results.csv   Per-phase timing for each run.
"""

import os
import re
import subprocess
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(__file__))
from client import DATA_DIR, MODEL, NS, env, progress, write_run_info
from schemas import Exp9Row, ModelLoadPhase, TypedCSVWriter

LOAD_RUNS = int(env("LOAD_RUNS", "3"))
DECODE_SELECTOR = env("DECODE_SELECTOR", "app=vllm-decode")
VLLM_CONTAINER = env("VLLM_CONTAINER", "vllm")
STARTUP_TIMEOUT = int(env("STARTUP_TIMEOUT", "600"))



# ── Helpers ─────────────────────────────────────────────────────────────────

def oc(*args):
    """Run an oc command, return (stdout, stderr)."""
    cmd = ["oc", *args]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    return result.stdout.strip(), result.stderr.strip()


def oc_check(*args):
    """Run an oc command, return (stdout, stderr). Raise on failure."""
    cmd = ["oc", *args]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        raise RuntimeError(f"oc {' '.join(args[:3])} failed: {result.stderr.strip()[:200]}")
    return result.stdout.strip(), result.stderr.strip()


def get_running_pods():
    """Return list of Running decode pod names."""
    out, _ = oc_check("get", "pods", "-l", DECODE_SELECTOR, "-n", NS,
                      "-o", "jsonpath={range .items[*]}{.metadata.name} {.status.phase}{'\\n'}{end}")
    pods = []
    for line in out.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1] == "Running":
            pods.append(parts[0])
    return pods


def stream_logs(stop_event, captured):
    """Stream decode pod logs in background, appending lines to captured list."""
    cmd = ["oc", "logs", "-f", "-l", DECODE_SELECTOR, "-n", NS,
           "-c", VLLM_CONTAINER, "--since=1s"]
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        while not stop_event.is_set():
            line = proc.stdout.readline()
            if not line:
                if proc.poll() is not None:
                    break
                continue
            captured.append((time.time(), line.rstrip()))
        proc.terminate()
        proc.wait(timeout=5)
    except Exception:
        pass


def parse_phases(captured):
    """Parse captured log lines for vLLM startup milestones.

    Returns dict of phase_name -> (epoch_time, log_line).
    """
    markers = {
        "weight_load_start": re.compile(r"(?i)(loading\s+(model\s+)?weights|loading\s+weights)", re.IGNORECASE),
        "weight_load_end":   re.compile(r"(?i)(model\s+weights\s+loaded|weights?\s+loaded)", re.IGNORECASE),
        "model_load_start":  re.compile(r"(?i)loading\s+model\b", re.IGNORECASE),
        "nixl_ready":        re.compile(r"(?i)(nixl|nxl).*(initialized|registered|ready)", re.IGNORECASE),
        "compile_start":     re.compile(r"(?i)(compil|warmup|warming|graph\s+capture).*start", re.IGNORECASE),
        "compile_end":       re.compile(r"(?i)(compil|warmup|warming|graph\s+capture).*(done|complete|finish)", re.IGNORECASE),
    }
    found = {}
    for ts, line in captured:
        for name, pattern in markers.items():
            if name not in found and pattern.search(line):
                found[name] = (ts, line)
    return found


def wait_for_ready(old_pod, timeout):
    """Poll until a new Running/Ready decode pod appears. Return new pod name."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        pods = get_running_pods()
        new_pods = [p for p in pods if p != old_pod]
        if new_pods:
            # Check Ready condition
            out, _ = oc("get", "pod", new_pods[0], "-n", NS,
                        "-o", "jsonpath={.status.conditions[?(@.type=='Ready')].status}")
            if out == "True":
                return new_pods[0]
        time.sleep(5)
    raise TimeoutError(f"No new Ready decode pod after {timeout}s")


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    skip_confirm = "--yes" in sys.argv or "-y" in sys.argv

    print("=" * 60)
    print("  EXPERIMENT 9: MODEL LOAD TIME DECOMPOSITION")
    print("  WARNING: This is a DESTRUCTIVE experiment — kills pods.")
    print("=" * 60)
    print()

    if not skip_confirm:
        try:
            answer = input("  Type YES to continue, or Ctrl-C to abort: ")
        except (EOFError, KeyboardInterrupt):
            print("\n  Aborted.")
            sys.exit(1)
        if answer.strip() != "YES":
            print("  Aborted.")
            sys.exit(1)
        print()

    # Safety check: need at least 2 decode pods
    pods = get_running_pods()
    if len(pods) < 2:
        print(f"ABORT: Need at least 2 Running decode pods, found {len(pods)}.")
        print("The deployment must stay available during the test.")
        sys.exit(1)

    progress(f"Found {len(pods)} decode pods: {', '.join(pods)}")
    progress(f"Will run {LOAD_RUNS} kill/measure cycles")
    os.makedirs(DATA_DIR, exist_ok=True)

    csv_path = os.path.join(DATA_DIR, "exp9-results.csv")
    writer = TypedCSVWriter(csv_path, Exp9Row)
    write_run_info("exp9", extra={"load_runs": LOAD_RUNS, "model": MODEL,
                                  "decode_selector": DECODE_SELECTOR})

    for run in range(1, LOAD_RUNS + 1):
        progress(f"── Run {run}/{LOAD_RUNS} ──")
        pods = get_running_pods()
        if len(pods) < 2:
            progress(f"Only {len(pods)} pod(s) left, stopping early")
            break

        target = pods[0]
        progress(f"Killing pod {target}")

        # Start log capture before the kill
        stop_event = threading.Event()
        captured = []
        log_thread = threading.Thread(target=stream_logs, args=(stop_event, captured))
        log_thread.start()

        # Kill the pod
        t_delete = time.time()
        oc("delete", "pod", target, "-n", NS, "--grace-period=0")
        t_deleted = time.time()

        # Wait for replacement
        try:
            new_pod = wait_for_ready(target, STARTUP_TIMEOUT)
        except TimeoutError as e:
            progress(f"TIMEOUT: {e}")
            stop_event.set()
            log_thread.join(timeout=10)
            continue

        t_ready = time.time()
        wall_total_ms = int((t_ready - t_delete) * 1000)
        progress(f"New pod {new_pod} ready in {wall_total_ms}ms")

        # Give logs a moment to flush, then stop capture
        time.sleep(2)
        stop_event.set()
        log_thread.join(timeout=10)

        # Parse phases
        phases = parse_phases(captured)
        progress(f"Detected phases: {list(phases.keys()) or 'none'}")

        def write_phase(phase, duration_ms, log_line="",
                        _run=run, _target=target, _new_pod=new_pod,
                        _wall_total_ms=wall_total_ms):
            writer.write({
                "experiment": "exp9", "run": _run,
                "pod_deleted": _target, "pod_new": _new_pod,
                "phase": phase, "duration_ms": int(duration_ms),
                "wall_total_ms": _wall_total_ms,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "log_line": log_line[:200] if log_line else "",
            })

        # Pod delete phase
        write_phase(ModelLoadPhase.POD_DELETE, (t_deleted - t_delete) * 1000)

        # Weight load phase
        if "weight_load_start" in phases and "weight_load_end" in phases:
            dur = (phases["weight_load_end"][0] - phases["weight_load_start"][0]) * 1000
            write_phase(ModelLoadPhase.WEIGHT_LOAD, dur, phases["weight_load_end"][1])
        else:
            write_phase(ModelLoadPhase.WEIGHT_LOAD, 0, "not_detected")

        # NIXL init: from weight_load_end (or model_load_start) to nixl_ready
        if "nixl_ready" in phases:
            ref = phases.get("weight_load_end", phases.get("model_load_start"))
            if ref:
                dur = (phases["nixl_ready"][0] - ref[0]) * 1000
                write_phase(ModelLoadPhase.NIXL_INIT, max(0, dur), phases["nixl_ready"][1])
            else:
                write_phase(ModelLoadPhase.NIXL_INIT, 0, phases["nixl_ready"][1])
        else:
            write_phase(ModelLoadPhase.NIXL_INIT, 0, "not_detected")

        # Compile/warmup phase
        if "compile_start" in phases and "compile_end" in phases:
            dur = (phases["compile_end"][0] - phases["compile_start"][0]) * 1000
            write_phase(ModelLoadPhase.COMPILE, dur, phases["compile_end"][1])
        else:
            write_phase(ModelLoadPhase.COMPILE, 0, "not_detected")

        # Total from first log to ready
        if captured:
            first_log_ts = captured[0][0]
            total_startup_ms = (t_ready - first_log_ts) * 1000
            write_phase(ModelLoadPhase.TOTAL_STARTUP, total_startup_ms)
        else:
            write_phase(ModelLoadPhase.TOTAL_STARTUP, 0, "no_logs_captured")

        # Wall total
        write_phase(ModelLoadPhase.WALL_TOTAL, wall_total_ms)

    writer.close()
    progress(f"Results written to {csv_path}")


if __name__ == "__main__":
    main()
