#!/bin/bash
# Tear down the P/D topology.
#
# Usage:
#   ./scripts/undeploy.sh clusters/my-cluster
#   ./scripts/undeploy.sh clusters/my-cluster --keep-pvc
#
# Deletes all resources labeled app.kubernetes.io/part-of=vllm-disagg.
# By default also deletes the model-cache PVC. Use --keep-pvc to preserve
# the downloaded model weights across redeployments.

set -euo pipefail

CLUSTER_DIR="${1:?Usage: $0 <cluster-dir> [--keep-pvc]}"
KEEP_PVC=false
for arg in "${@:2}"; do
    [ "$arg" = "--keep-pvc" ] && KEEP_PVC=true
done

if [ ! -f "$CLUSTER_DIR/env.sh" ]; then
    echo "ERROR: $CLUSTER_DIR/env.sh not found"
    exit 1
fi

source "$CLUSTER_DIR/env.sh"

echo "Tearing down in namespace $NS..."

oc delete deployment -l app.kubernetes.io/part-of=vllm-disagg -n "$NS" --ignore-not-found
oc delete service -l app.kubernetes.io/part-of=vllm-disagg -n "$NS" --ignore-not-found
oc delete pdb -l app.kubernetes.io/part-of=vllm-disagg -n "$NS" --ignore-not-found
oc delete pod test-client -n "$NS" --ignore-not-found
oc delete rolebinding test-client-pod-reader -n "$NS" --ignore-not-found 2>/dev/null || true
oc delete role pod-reader -n "$NS" --ignore-not-found 2>/dev/null || true
oc delete configmap vllm-model-config -n "$NS" --ignore-not-found

if [ "$KEEP_PVC" = true ]; then
    echo "PVC kept (--keep-pvc). Model cache preserved for next deploy."
else
    oc delete pvc model-cache -n "$NS" --ignore-not-found
    echo "PVC deleted. Model will re-download on next deploy."
fi

echo "Done."
