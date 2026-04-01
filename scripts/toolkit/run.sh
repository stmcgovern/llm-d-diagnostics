#!/bin/bash
# Run toolkit experiments in-cluster via oc exec.
#
# Usage:
#   ./scripts/toolkit/run.sh <cluster-dir> [exp1|exp1b|exp2|exp3|exp5|exp6|exp7|all|analyze]
#
# Prerequisites:
#   1. test-client pod is running (oc apply -f manifests/ -n <namespace>)
#   2. Toolkit scripts are copied to the pod (this script handles it)
#   3. Cluster env is at <cluster-dir>/env.sh
#
# The script copies the toolkit to the pod, runs the experiment inside
# the cluster (no port-forwarding), and copies results back.
#
# Examples:
#   ./scripts/toolkit/run.sh clusters/rdu3-t4x3 exp5
#   ./scripts/toolkit/run.sh clusters/rdu3-t4x3 all
#   ./scripts/toolkit/run.sh clusters/rdu3-t4x3 analyze   # runs locally

set -euo pipefail

CLUSTER_DIR="${1:?Usage: $0 <cluster-dir> [exp5|exp6|exp7|all|analyze]}"
EXPERIMENT="${2:-all}"
# Note: exp4 (fault tolerance) runs locally because it kills pods via `oc`.
# Run it directly: NS=<namespace> python3 scripts/toolkit/exp4_fault.py
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

if [ ! -f "$CLUSTER_DIR/env.sh" ]; then
    echo "ERROR: $CLUSTER_DIR/env.sh not found"
    exit 1
fi

source "$CLUSTER_DIR/env.sh"
mkdir -p "$DATA_DIR"

POD=test-client
REMOTE_DIR="/scripts/toolkit"

echo "Cluster:    $CLUSTER_DIR"
echo "Namespace:  $NS"
echo "Pod:        $POD"
echo ""

# ── Analysis runs locally (no cluster needed) ──────────────────────────────
if [ "$EXPERIMENT" = "analyze" ]; then
    echo "=== Running Analysis (local) ==="
    python3 "$SCRIPT_DIR/analyze.py" "$DATA_DIR"
    exit 0
fi

# ── Verify pod is ready ────────────────────────────────────────────────────
if ! oc get pod "$POD" -n "$NS" -o jsonpath='{.status.phase}' 2>/dev/null | grep -q Running; then
    echo "ERROR: Pod '$POD' is not running in namespace '$NS'"
    echo "  Deploy with: oc apply -f manifests/ -n $NS"
    exit 1
fi

# ── Copy toolkit to pod ────────────────────────────────────────────────────
echo "Copying toolkit to pod..."
oc exec "$POD" -n "$NS" -- mkdir -p "$REMOTE_DIR" "$REMOTE_DIR/data"
oc cp "$SCRIPT_DIR/" "$NS/$POD:$REMOTE_DIR/"
echo ""

# ── Build env vars for remote execution ─────────────────────────────────────
# Use in-cluster defaults from client.py; only override what env.sh sets.
REMOTE_ENV="MODEL=$MODEL NS=$NS DATA_DIR=$REMOTE_DIR/data PREFILL_HOST=$PREFILL_HOST"
[ -n "${SIM:-}" ] && REMOTE_ENV="$REMOTE_ENV SIM=$SIM"

run_remote() {
    local name="$1"
    local script="$2"
    echo "=== Running $name ==="
    oc exec "$POD" -n "$NS" -- env $REMOTE_ENV python3 "$REMOTE_DIR/$script"
    echo ""
}

# ── Run experiments ─────────────────────────────────────────────────────────
case "$EXPERIMENT" in
    exp1)
        run_remote "Experiment 1: Single-Request Latency" exp1_latency.py
        ;;
    exp1b)
        run_remote "Experiment 1b: Latency Decomposition" exp1b_decompose.py
        ;;
    exp2)
        run_remote "Experiment 2: Throughput Under Load" exp2_throughput.py
        ;;
    exp3)
        run_remote "Experiment 3: Prefill Isolation" exp3_isolation.py
        ;;
    exp5)
        run_remote "Experiment 5: Sequence Length Sweep" exp5_seqlen_sweep.py
        ;;
    exp6)
        run_remote "Experiment 6: Saturation Profiling" exp6_saturation.py
        ;;
    exp7)
        run_remote "Experiment 7: Mixed Workload" exp7_mixed_workload.py
        ;;
    all)
        run_remote "Experiment 1: Single-Request Latency" exp1_latency.py
        run_remote "Experiment 1b: Latency Decomposition" exp1b_decompose.py
        run_remote "Experiment 2: Throughput Under Load" exp2_throughput.py
        run_remote "Experiment 3: Prefill Isolation" exp3_isolation.py
        run_remote "Experiment 5: Sequence Length Sweep" exp5_seqlen_sweep.py
        run_remote "Experiment 6: Saturation Profiling" exp6_saturation.py
        run_remote "Experiment 7: Mixed Workload" exp7_mixed_workload.py
        ;;
    *)
        echo "Unknown experiment: $EXPERIMENT"
        echo "Options: exp1, exp1b, exp2, exp3, exp5, exp6, exp7, all, analyze"
        echo "Note: exp4 runs locally (kills pods). Run directly with:"
        echo "  NS=$NS python3 scripts/toolkit/exp4_fault.py"
        exit 1
        ;;
esac

# ── Copy results back ──────────────────────────────────────────────────────
echo "Copying results from pod..."
oc cp "$NS/$POD:$REMOTE_DIR/data/" "$DATA_DIR/"
echo "Results saved to $DATA_DIR/"
