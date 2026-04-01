#!/usr/bin/env python3
"""
Experiment 4: Fault Tolerance

Runs from OUTSIDE the cluster. Uses `oc exec` to send requests via test-client
and `oc` commands to kill/scale pods.

Sub-experiments:
    4a. Decode pod failure (consumer dies)
    4b. Prefill pod failure (producer dies)
    4d. Graceful degradation (3 GPU -> 2 GPU -> 3 GPU)
    4e. Failure under load (prefill dies while requests in flight)
    4f. Network partition (block NIXL between prefill and decode)
    4g. Slow network (tc netem latency + packet loss on prefill egress)

Usage:
    python3 scripts/toolkit/exp4_fault.py
    NS=my-ns python3 scripts/toolkit/exp4_fault.py

Additional env vars:
    TEST_CLIENT       Pod name for test client (default: test-client)
    PREFILL_DEPLOY    Prefill deployment name (default: vllm-prefill)
    DECODE1_DEPLOY    Decode-1 deployment name (default: vllm-decode)
    DECODE2_DEPLOY    Decode-2 deployment name (default: vllm-decode-2)
    PREFILL_SELECTOR  Label selector for prefill pods (default: app=vllm-prefill)
    DECODE1_SELECTOR  Label selector for decode-1 pods (default: app=vllm-decode)
    VLLM_CONTAINER    vLLM container name (default: vllm)
    SIDECAR_CONTAINER Sidecar container name (default: routing-sidecar)
    LOAD_REQUESTS     Concurrent requests for 4e (default: 8)
    NETPOLICY_FILE    Network partition manifest for 4f
                      (default: manifests/testing/90-networkpolicy-partition.yaml)
    NETEM_DELAY_MS    Added latency for 4g slow-network test (default: 100)
    NETEM_LOSS_PCT    Packet loss percentage for 4g (default: 10)

Outputs:
    data/exp4-results.csv   Timing and status data for all sub-experiments.
    data/exp4-logs/         Pod logs captured at key moments during each
                            fault scenario (pre-kill, post-recovery, during
                            partition). Shows NIXL reconnection, ZMQ discovery,
                            and sidecar error handling sequences.
"""

import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
import threading

sys.path.insert(0, os.path.dirname(__file__))
from client import (
    env, MODEL, NS, PREFILL_HOST, DATA_DIR,
    DISAGG_D1_URL, DISAGG_D2_URL,
    CSVWriter, progress, SIM, write_run_info,
)

TEST_CLIENT = env("TEST_CLIENT", "test-client")
VLLM_CONTAINER = env("VLLM_CONTAINER", "vllm")

# Use the same URLs as the other experiments (from client.py)
D1_URL = DISAGG_D1_URL  # e.g. https://vllm-decode-svc:8000/v1/completions
D2_URL = DISAGG_D2_URL  # e.g. https://vllm-decode-2-svc:8000/v1/completions

PREFILL_DEPLOY = env("PREFILL_DEPLOY", "vllm-prefill")
DECODE1_DEPLOY = env("DECODE1_DEPLOY", "vllm-decode")
DECODE2_DEPLOY = env("DECODE2_DEPLOY", "vllm-decode-2")
PREFILL_SELECTOR = env("PREFILL_SELECTOR", "app=vllm-prefill")
DECODE1_SELECTOR = env("DECODE1_SELECTOR", "app=vllm-decode")
LOAD_REQUESTS = int(env("LOAD_REQUESTS", "8"))
NETPOLICY_FILE = env("NETPOLICY_FILE", "manifests/testing/90-networkpolicy-partition.yaml")
SIDECAR_CONTAINER = env("SIDECAR_CONTAINER", "routing-sidecar")
NETEM_DELAY_MS = int(env("NETEM_DELAY_MS", "100"))
NETEM_LOSS_PCT = int(env("NETEM_LOSS_PCT", "10"))

FIELDS = [
    "experiment", "sub", "phase", "timestamp", "epoch_ms",
    "ttft_ms", "total_ms", "status_code", "note", "error",
]


def oc(*args):
    """Run an oc command and return stdout. Warns on non-zero exit."""
    cmd = ["oc"] + list(args)
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        progress(f"  WARNING: oc {' '.join(args[:3])} exited {result.returncode}: "
                 f"{result.stderr.strip()[:200]}")
    return result.stdout.strip(), result.stderr.strip()


def send_via_test_client(url, prompt="Hello world", max_tokens=10,
                         curl_timeout=10, subprocess_timeout=30):
    """Send a request via oc exec to the test-client pod.

    Returns parsed timing: (ttft_ms, total_ms, status_code, error).
    Uses curl with -w for timing since we're going through oc exec
    (can't use Python http.client from outside the cluster).
    """
    cmd = [
        "oc", "exec", TEST_CLIENT, "-n", NS, "--",
        "curl", "-sk", "--http1.1", "-m", str(curl_timeout),
        "-w", "|%{time_starttransfer}|%{time_total}|%{http_code}",
        "-H", "Content-Type: application/json",
        "-H", f"x-prefiller-host-port: {PREFILL_HOST}",
        url,
        "-d", json.dumps({
            "model": MODEL,
            "prompt": prompt,
            "max_tokens": max_tokens,
        }),
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True,
                                timeout=subprocess_timeout)
        output = result.stdout.strip()

        # Parse: body|ttft_s|total_s|http_code
        parts = output.rsplit("|", 3)
        if len(parts) >= 4:
            ttft_s = float(parts[-3])
            total_s = float(parts[-2])
            code = parts[-1].strip()
            return round(ttft_s * 1000, 1), round(total_s * 1000, 1), code, ""
        else:
            return 0, 0, "parse_error", output[:200]

    except subprocess.TimeoutExpired:
        return 0, 0, "timeout", "oc exec timed out"
    except Exception as e:
        return 0, 0, "error", str(e)


def wait_for_ready(deployment, timeout_s=600):
    """Wait for a deployment to have 1 ready replica."""
    sys.stderr.write(f"  Waiting for {deployment} to be ready")
    sys.stderr.flush()
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        out, _ = oc("get", "deployment", deployment, "-n", NS,
                     "-o", "jsonpath={.status.readyReplicas}")
        if out.strip() == "1":
            progress(" ready")
            return
        sys.stderr.write(".")
        sys.stderr.flush()
        time.sleep(5)
    progress(f" TIMEOUT after {timeout_s}s")
    raise TimeoutError(f"{deployment} not ready after {timeout_s}s")


def get_pod_name(selector):
    """Get the name of the first pod matching a label selector."""
    out, _ = oc("get", "pods", "-n", NS, "-l", selector,
                "-o", "jsonpath={.items[0].metadata.name}")
    return out.strip()


def timestamp():
    return time.strftime("%H:%M:%S")


def collect_cluster_info():
    """Gather cluster-level metadata via oc. Returns dict."""
    info = {}

    # oc version (client + server)
    out, _ = oc("version", "--output=json")
    if out:
        try:
            v = json.loads(out)
            info["oc_client"] = v.get("clientVersion", {}).get("gitVersion", "")
            info["server"] = v.get("serverVersion", {}).get("gitVersion", "")
        except (json.JSONDecodeError, ValueError):
            pass

    # Pod images (vLLM and sidecar versions)
    for selector, label in [(PREFILL_SELECTOR, "prefill"),
                            (DECODE1_SELECTOR, "decode1")]:
        out, _ = oc("get", "pods", "-n", NS, "-l", selector,
                     "-o", "jsonpath={.items[0].spec.containers[*].image}")
        if out:
            info[f"{label}_images"] = out.strip().split()

    # GPU info from node labels
    out, _ = oc("get", "nodes", "-o",
                "jsonpath={range .items[*]}{.metadata.name}={.metadata.labels.nvidia\\.com/gpu\\.product} ")
    if out:
        gpu_nodes = {}
        for item in out.strip().split():
            if "=" in item:
                node, gpu = item.split("=", 1)
                if gpu:
                    gpu_nodes[node] = gpu
        if gpu_nodes:
            info["gpu_nodes"] = gpu_nodes

    # vLLM/NIXL/NCCL environment variables from running pods.
    # These are the knobs that control fault tolerance and communication behavior.
    _INTERESTING_PREFIXES = (
        "VLLM_", "NIXL_", "NCCL_", "CUDA_", "TORCH_",
        "RDMA_", "UCX_", "GLOO_",
    )
    for selector, label in [(PREFILL_SELECTOR, "prefill"),
                            (DECODE1_SELECTOR, "decode1")]:
        pod = get_pod_name(selector)
        if not pod:
            continue
        out, _ = oc("exec", pod, "-c", VLLM_CONTAINER, "-n", NS,
                     "--", "env")
        if out:
            env_vars = {}
            for line in out.splitlines():
                if "=" in line:
                    k, _, v = line.partition("=")
                    if k.startswith(_INTERESTING_PREFIXES):
                        env_vars[k] = v
            if env_vars:
                info[f"{label}_env"] = env_vars

    return info


def capture_logs(pod, container, label, log_dir):
    """Capture pod logs to a file. Returns the filepath written.

    Args:
        pod: Pod name (or empty string to skip).
        container: Container name within the pod.
        label: Descriptive label for the filename (e.g., '4a-decode1-pre').
        log_dir: Directory to write log files.
    """
    if not pod:
        return None
    filepath = os.path.join(log_dir, f"{label}-{container}.log")
    out, err = oc("logs", pod, "-c", container, "-n", NS)
    if out:
        with open(filepath, "w") as f:
            f.write(out)
        return filepath
    return None


def capture_pod_logs(selector, containers, label, log_dir):
    """Capture logs from all containers of the first pod matching selector.

    Errors are logged but never propagate — log capture must not crash
    the experiment.
    """
    try:
        pod = get_pod_name(selector)
        if not pod:
            progress(f"  (no pod found for {selector}, skipping log capture)")
            return
        files = []
        for c in containers:
            path = capture_logs(pod, c, label, log_dir)
            if path:
                files.append(os.path.basename(path))
        if files:
            progress(f"  Captured: {', '.join(files)}")
    except Exception as e:
        progress(f"  WARNING: log capture failed for {label}: {e}")


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    log_dir = os.path.join(DATA_DIR, "exp4-logs")
    os.makedirs(log_dir, exist_ok=True)
    outfile = os.path.join(DATA_DIR, "exp4-results.csv")

    # Collect cluster metadata (exp4 runs locally with oc access)
    progress("  Collecting cluster metadata...")
    cluster_info = collect_cluster_info()
    write_run_info("exp4", {"load_requests": LOAD_REQUESTS, "cluster": cluster_info})

    writer = CSVWriter(outfile, FIELDS)

    progress("=== Experiment 4: Fault Tolerance ===")
    progress(f"  Model: {MODEL}")
    progress(f"  Namespace: {NS}")
    progress(f"  Output: {outfile}")
    progress(f"  Logs: {log_dir}")
    progress("")

    def epoch_ms():
        return int(time.time() * 1000)

    def record(sub, phase, ttft_ms, total_ms, status, note, error=""):
        writer.write({
            "experiment": "exp4", "sub": sub, "phase": phase,
            "timestamp": timestamp(), "epoch_ms": epoch_ms(),
            "ttft_ms": ttft_ms, "total_ms": total_ms,
            "status_code": status, "note": note, "error": error,
        })
        progress(f"    [{sub}/{phase}] code={status} ttft={ttft_ms}ms "
                 f"total={total_ms}ms {note}")

    # ── 4a: Decode pod failure ───────────────────────────
    progress("=== 4a: Decode pod failure ===")
    progress("")

    progress("  Pre-flight: verify both decode pods healthy")
    t, tot, code, err = send_via_test_client(D1_URL)
    record("4a", "pre-d1", t, tot, code, "decode-1 before kill", err)
    t, tot, code, err = send_via_test_client(D2_URL)
    record("4a", "pre-d2", t, tot, code, "decode-2 before kill", err)

    # Capture pre-kill logs from decode-1 (vllm + sidecar)
    capture_pod_logs(DECODE1_SELECTOR, [VLLM_CONTAINER, SIDECAR_CONTAINER],
                     "4a-decode1-pre", log_dir)

    progress("")
    progress("  Killing decode-1 pod...")
    decode1_pod = get_pod_name(DECODE1_SELECTOR)
    out, _ = oc("delete", "pod", decode1_pod, "-n", NS,
                "--grace-period=0", "--force")
    record("4a", "kill", 0, 0, "n/a", f"killed {decode1_pod}")
    progress(f"  {out}")
    progress("")

    progress("  Immediate request to decode-2 (should still work):")
    t, tot, code, err = send_via_test_client(D2_URL)
    record("4a", "during-d2", t, tot, code, "decode-2 while decode-1 dead", err)

    progress("  Immediate request to decode-1 (should fail or timeout):")
    t, tot, code, err = send_via_test_client(D1_URL)
    record("4a", "during-d1", t, tot, code, "decode-1 just killed", err)

    progress("")
    progress("  Waiting for decode-1 replacement...")
    recovery_start = time.time()
    wait_for_ready(DECODE1_DEPLOY)
    recovery_time = int(time.time() - recovery_start)
    progress(f"  Recovery time: {recovery_time}s")

    progress("  Post-recovery request to decode-1:")
    t, tot, code, err = send_via_test_client(D1_URL)
    record("4a", "post-d1", t, tot, code,
           f"decode-1 after recovery ({recovery_time}s)", err)

    # Check new engine_id
    new_decode1 = get_pod_name(DECODE1_SELECTOR)
    progress(f"  Old pod: {decode1_pod} -> New pod: {new_decode1}")
    engine_line = "(engine_id not found in logs)"
    out, _ = oc("logs", new_decode1, "-c", VLLM_CONTAINER, "-n", NS)
    for line in out.splitlines():
        if "engine_id:" in line:
            engine_line = line
    progress(f"  {engine_line}")

    # Capture post-recovery logs — shows NIXL handshake re-establishment
    capture_pod_logs(DECODE1_SELECTOR, [VLLM_CONTAINER, SIDECAR_CONTAINER],
                     "4a-decode1-post", log_dir)

    # Verify warm-up: send a second request to see if cold start clears
    progress("  Second request (should be at steady-state):")
    t, tot, code, err = send_via_test_client(D1_URL)
    record("4a", "post-d1-warm", t, tot, code,
           "decode-1 second request after recovery", err)
    progress("")

    # ── 4b: Prefill pod failure ──────────────────────────
    progress("=== 4b: Prefill pod failure ===")
    progress("")

    progress("  Pre-flight: verify disagg works")
    t, tot, code, err = send_via_test_client(D1_URL)
    record("4b", "pre", t, tot, code, "before prefill kill", err)

    # Capture pre-kill logs from prefill and both decode sidecars
    capture_pod_logs(PREFILL_SELECTOR, [VLLM_CONTAINER],
                     "4b-prefill-pre", log_dir)
    capture_pod_logs(DECODE1_SELECTOR, [SIDECAR_CONTAINER],
                     "4b-decode1-pre", log_dir)

    progress("")
    progress("  Killing prefill pod...")
    prefill_pod = get_pod_name(PREFILL_SELECTOR)
    out, _ = oc("delete", "pod", prefill_pod, "-n", NS,
                "--grace-period=0", "--force")
    record("4b", "kill", 0, 0, "n/a", f"killed {prefill_pod}")
    progress(f"  {out}")
    progress("")

    progress("  Immediate request through decode-1 sidecar (prefill down):")
    t, tot, code, err = send_via_test_client(D1_URL)
    record("4b", "during", t, tot, code, "prefill dead", err)

    progress("")
    progress("  Waiting for prefill replacement...")
    recovery_start = time.time()
    wait_for_ready(PREFILL_DEPLOY)
    recovery_time = int(time.time() - recovery_start)
    progress(f"  Recovery time: {recovery_time}s")

    new_prefill = get_pod_name(PREFILL_SELECTOR)
    progress(f"  Old pod: {prefill_pod} -> New pod: {new_prefill}")
    out, _ = oc("get", "pod", new_prefill, "-n", NS,
                "-o", "jsonpath={.status.podIP}")
    progress(f"  New prefill IP: {out}")

    progress("  Post-recovery requests:")
    t, tot, code, err = send_via_test_client(D1_URL)
    record("4b", "post-1", t, tot, code, "first request after prefill recovery", err)
    t, tot, code, err = send_via_test_client(D2_URL)
    record("4b", "post-2", t, tot, code, "decode-2 after prefill recovery", err)

    # Capture post-recovery logs — shows ZMQ discovery of new prefill
    capture_pod_logs(PREFILL_SELECTOR, [VLLM_CONTAINER],
                     "4b-prefill-post", log_dir)
    capture_pod_logs(DECODE1_SELECTOR, [SIDECAR_CONTAINER, VLLM_CONTAINER],
                     "4b-decode1-post", log_dir)

    # Second requests to check warm-up
    progress("  Second requests (steady-state check):")
    t, tot, code, err = send_via_test_client(D1_URL)
    record("4b", "post-1-warm", t, tot, code, "decode-1 second request", err)
    t, tot, code, err = send_via_test_client(D2_URL)
    record("4b", "post-2-warm", t, tot, code, "decode-2 second request", err)
    progress("")

    # ── 4d: Graceful degradation ─────────────────────────
    progress("=== 4d: Graceful degradation (3 GPU -> 2 GPU -> 3 GPU) ===")
    progress("")

    progress(f"  Scaling {DECODE2_DEPLOY} to 0...")
    oc("scale", "deployment", DECODE2_DEPLOY, "--replicas=0", "-n", NS)
    time.sleep(5)

    progress("  Sending 5 requests through decode-1 (should all work):")
    for i in range(1, 6):
        t, tot, code, err = send_via_test_client(D1_URL)
        record("4d", f"2gpu-{i}", t, tot, code,
               f"decode-1 only ({DECODE2_DEPLOY} scaled down)", err)

    progress("")
    progress(f"  Scaling {DECODE2_DEPLOY} back to 1...")
    oc("scale", "deployment", DECODE2_DEPLOY, "--replicas=1", "-n", NS)
    recovery_start = time.time()
    wait_for_ready(DECODE2_DEPLOY)
    recovery_time = int(time.time() - recovery_start)
    progress(f"  Scale-up time: {recovery_time}s")

    progress("  Request to restored decode-2:")
    t, tot, code, err = send_via_test_client(D2_URL)
    record("4d", "restored", t, tot, code,
           f"decode-2 after scale-up ({recovery_time}s)", err)

    # Second request to check warm-up
    t, tot, code, err = send_via_test_client(D2_URL)
    record("4d", "restored-warm", t, tot, code,
           "decode-2 second request after scale-up", err)
    progress("")

    # ── 4e: Failure under load ───────────────────────────
    progress(f"=== 4e: Failure under load ({LOAD_REQUESTS} concurrent requests) ===")
    progress("")

    progress("  Pre-flight: verify disagg works")
    t, tot, code, err = send_via_test_client(D1_URL)
    record("4e", "pre", t, tot, code, "before load test", err)

    progress(f"  Launching {LOAD_REQUESTS} concurrent requests to decode-1...")
    progress("  Killing prefill after 1s delay...")

    # Use max_tokens=50 and longer timeouts to keep requests in-flight
    # long enough to overlap with the kill.
    def send_load_request(url):
        return send_via_test_client(url,
            prompt="Write a detailed essay about distributed systems",
            max_tokens=50, curl_timeout=30, subprocess_timeout=60)

    kill_error = [None]

    def _kill_prefill_delayed():
        try:
            time.sleep(1)
            pod = get_pod_name(PREFILL_SELECTOR)
            if not pod:
                kill_error[0] = "no prefill pod found"
                progress("  WARNING: no prefill pod found to kill")
                return
            out, _ = oc("delete", "pod", pod, "-n", NS,
                         "--grace-period=0", "--force")
            record("4e", "kill", 0, 0, "n/a", f"killed {pod}")
            progress(f"  Prefill killed at {timestamp()}: {out}")
        except Exception as e:
            kill_error[0] = str(e)
            progress(f"  WARNING: kill thread failed: {e}")

    killer = threading.Thread(target=_kill_prefill_delayed)
    killer.start()

    # Send concurrent long requests (should be in-flight when prefill dies)
    with ThreadPoolExecutor(max_workers=LOAD_REQUESTS) as pool:
        futures = [pool.submit(send_load_request, D1_URL) for _ in range(LOAD_REQUESTS)]
        request_results = [f.result() for f in futures]

    killer.join()

    if kill_error[0]:
        progress(f"  WARNING: kill may not have succeeded: {kill_error[0]}")

    succeeded = 0
    failed = 0
    for i, (t, tot, code, err) in enumerate(request_results):
        status_label = "ok" if code == "200" else "fail"
        if code == "200":
            succeeded += 1
        else:
            failed += 1
        record("4e", f"during-{i+1}", t, tot, code,
               f"request {i+1}/{LOAD_REQUESTS} ({status_label})", err)

    progress(f"  Results: {succeeded}/{LOAD_REQUESTS} succeeded, "
             f"{failed}/{LOAD_REQUESTS} failed")

    # Capture sidecar logs — shows how it handled in-flight request failures
    capture_pod_logs(DECODE1_SELECTOR, [SIDECAR_CONTAINER],
                     "4e-decode1-during", log_dir)

    progress("")
    progress("  Waiting for prefill recovery...")
    recovery_start = time.time()
    wait_for_ready(PREFILL_DEPLOY)
    recovery_time = int(time.time() - recovery_start)
    progress(f"  Recovery time: {recovery_time}s")

    progress("  Post-recovery request:")
    t, tot, code, err = send_via_test_client(D1_URL)
    record("4e", "post", t, tot, code,
           f"after prefill recovery ({recovery_time}s)", err)
    progress("")

    # ── 4f: Network partition ────────────────────────────
    progress("=== 4f: Network partition (block NIXL between prefill and decode) ===")
    progress("")

    progress("  Pre-flight: verify disagg works")
    t, tot, code, err = send_via_test_client(D1_URL)
    record("4f", "pre", t, tot, code, "before partition", err)

    progress(f"  Applying network partition: {NETPOLICY_FILE}")
    out, err_str = oc("apply", "-f", NETPOLICY_FILE, "-n", NS)
    record("4f", "partition-on", 0, 0, "n/a", "network partition applied")
    progress(f"  {out}")
    if err_str:
        progress(f"  stderr: {err_str}")

    try:
        # Give the network policy a moment to take effect
        time.sleep(3)

        # Verify policy was applied
        out, _ = oc("get", "networkpolicy", "-n", NS,
                     "-o", "jsonpath={.items[*].metadata.name}")
        progress(f"  Active policies: {out}")

        progress("  Request during partition (should fail or timeout):")
        t, tot, code, err = send_via_test_client(D1_URL)
        record("4f", "during-1", t, tot, code, "partition active", err)

        progress("  Second request during partition:")
        t, tot, code, err = send_via_test_client(D1_URL)
        record("4f", "during-2", t, tot, code, "partition active (2nd)", err)

        progress("  Request to decode-2 during partition:")
        t, tot, code, err = send_via_test_client(D2_URL)
        record("4f", "during-d2", t, tot, code, "partition active (decode-2)", err)

    finally:
        # Always remove the partition — leaving it breaks NIXL traffic
        progress("")
        progress(f"  Removing network partition...")
        out, err_str = oc("delete", "-f", NETPOLICY_FILE, "-n", NS)
        record("4f", "partition-off", 0, 0, "n/a", "network partition removed")
        progress(f"  {out}")

    # Give the network a moment to recover
    time.sleep(5)

    progress("  Post-partition request to decode-1:")
    t, tot, code, err = send_via_test_client(D1_URL)
    record("4f", "post-1", t, tot, code, "partition removed", err)

    progress("  Second post-partition request:")
    t, tot, code, err = send_via_test_client(D1_URL)
    record("4f", "post-2", t, tot, code, "partition removed (steady-state)", err)

    progress("  Post-partition request to decode-2:")
    t, tot, code, err = send_via_test_client(D2_URL)
    record("4f", "post-d2", t, tot, code, "partition removed (decode-2)", err)

    # Capture post-partition logs — shows NIXL reconnection after partition removal
    capture_pod_logs(DECODE1_SELECTOR, [SIDECAR_CONTAINER, VLLM_CONTAINER],
                     "4f-decode1-post", log_dir)
    capture_pod_logs(PREFILL_SELECTOR, [VLLM_CONTAINER],
                     "4f-prefill-post", log_dir)
    progress("")

    # ── 4g: Slow network (tc netem) ─────────────────────────
    progress(f"=== 4g: Slow network (delay={NETEM_DELAY_MS}ms, loss={NETEM_LOSS_PCT}%) ===")
    progress("")

    # Check if tc is available in the prefill pod
    tc_check, _ = oc("exec", get_pod_name(PREFILL_SELECTOR), "-c", VLLM_CONTAINER,
                      "-n", NS, "--", "which", "tc")
    if not tc_check.strip():
        progress("  SKIPPED: tc (iproute2) not available in prefill container.")
        progress("  To enable 4g, install iproute2 in the vLLM image or add a")
        progress("  privileged debug sidecar with NET_ADMIN capability.")
        record("4g", "skip", 0, 0, "n/a", "tc not available in container")
    else:
        progress("  Pre-flight: verify disagg works")
        t, tot, code, err = send_via_test_client(D1_URL)
        record("4g", "pre", t, tot, code, "before network degradation", err)

        prefill_pod = get_pod_name(PREFILL_SELECTOR)

        # Apply netem: add latency + packet loss to the prefill pod's egress.
        # This degrades NIXL KV transfers without completely blocking them —
        # a harder failure mode than a clean partition.
        netem_cmd = (f"tc qdisc add dev eth0 root netem "
                     f"delay {NETEM_DELAY_MS}ms 20ms distribution normal "
                     f"loss {NETEM_LOSS_PCT}%")
        progress(f"  Applying: {netem_cmd}")
        out, err_str = oc("exec", prefill_pod, "-c", VLLM_CONTAINER, "-n", NS,
                          "--", "bash", "-c", netem_cmd)
        record("4g", "netem-on", 0, 0, "n/a",
               f"delay={NETEM_DELAY_MS}ms loss={NETEM_LOSS_PCT}%")
        if err_str:
            progress(f"  stderr: {err_str}")

        try:
            time.sleep(2)

            # Send requests under degraded network — measures how NIXL handles
            # slow/lossy connections vs clean failures
            progress("  Requests during network degradation:")
            for i in range(1, 6):
                t, tot, code, err = send_via_test_client(D1_URL, curl_timeout=30,
                                                          subprocess_timeout=60)
                record("4g", f"during-{i}", t, tot, code,
                       f"request {i}/5 under degraded network", err)

            progress("  Request to decode-2 (also affected by prefill degradation):")
            t, tot, code, err = send_via_test_client(D2_URL, curl_timeout=30,
                                                      subprocess_timeout=60)
            record("4g", "during-d2", t, tot, code,
                   "decode-2 under degraded network", err)

            # Capture logs during degradation
            capture_pod_logs(DECODE1_SELECTOR, [SIDECAR_CONTAINER],
                             "4g-decode1-during", log_dir)

        finally:
            # Always remove netem rules
            progress("")
            progress("  Removing network degradation...")
            oc("exec", prefill_pod, "-c", VLLM_CONTAINER, "-n", NS,
               "--", "tc", "qdisc", "del", "dev", "eth0", "root")
            record("4g", "netem-off", 0, 0, "n/a", "network degradation removed")

        time.sleep(2)

        progress("  Post-recovery requests:")
        t, tot, code, err = send_via_test_client(D1_URL)
        record("4g", "post-1", t, tot, code, "after removing degradation", err)
        t, tot, code, err = send_via_test_client(D1_URL)
        record("4g", "post-2", t, tot, code, "steady-state after degradation", err)

        capture_pod_logs(DECODE1_SELECTOR, [SIDECAR_CONTAINER, VLLM_CONTAINER],
                         "4g-decode1-post", log_dir)
    progress("")

    writer.close()
    progress("")
    progress(f"=== Experiment 4 Complete === ({outfile})")


if __name__ == "__main__":
    main()
