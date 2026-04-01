---
name: assess-cluster
description: Run the llm-d diagnostics toolkit against a cluster and produce an assessment
allowed-tools: Read, Bash, Grep, Glob, Write, Edit, Agent
argument-hint: [cluster-name] [namespace]
---

# Assess Cluster

Run the llm-d diagnostics toolkit against a live cluster deployment and produce
a structured assessment (setup, performance data, recommendation).

**Arguments:** `$ARGUMENTS`
- First argument: cluster name (used for `clusters/<name>/` directory)
- Second argument (optional): Kubernetes namespace (default: from $NS or "default")

## Prerequisites Check

Before running experiments, verify the deployment is healthy:

```bash
# Check all pods are Running and Ready
oc get pods -n <namespace> -l app.kubernetes.io/part-of=vllm-disagg -o wide

# Verify endpoints resolve
oc get svc -n <namespace> | grep vllm

# Check GPU allocation
oc describe nodes | grep -A5 "nvidia.com/gpu"
```

If pods aren't ready, help the user debug before proceeding. Common issues:
- PVC not created (`oc apply -f manifests/00-model-cache-pvc.yaml`)
- Model not downloaded yet (check vllm container logs for download progress)
- GPU not available (check node allocatable resources)

## Workflow

### 1. Setup

Determine the cluster name and namespace. Create the results directory:

```bash
mkdir -p clusters/<cluster-name>/data
```

Gather system information for the assessment header:

```bash
# Node hardware
oc get nodes -o wide
oc describe nodes | grep -E "nvidia.com/gpu|kubernetes.io/hostname|Capacity:|Allocatable:"

# Component versions
oc get pods -n <namespace> -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{range .spec.containers[*]}{.image}{"\n"}{end}{end}'

# Network (CNI plugin)
oc get network.operator cluster -o jsonpath='{.spec.defaultNetwork.type}'
```

### 2. Copy toolkit and run experiments

```bash
# Copy toolkit to test-client pod
oc cp scripts/toolkit test-client:/scripts/toolkit -n <namespace>

# Experiments 1-3 run inside the pod
oc exec test-client -n <namespace> -- python3 /scripts/toolkit/exp1_latency.py
oc exec test-client -n <namespace> -- python3 /scripts/toolkit/exp1b_decompose.py
oc exec test-client -n <namespace> -- python3 /scripts/toolkit/exp2_throughput.py
oc exec test-client -n <namespace> -- python3 /scripts/toolkit/exp3_isolation.py

# Experiment 4 runs locally (it kills pods via oc)
NS=<namespace> python3 scripts/toolkit/exp4_fault.py
```

Run experiments sequentially. After each one completes, report what happened
(success/failure, any errors in stderr). If an experiment fails, diagnose and
help fix it before moving on.

**Important:** Exp 4 is destructive — it kills pods to test recovery. Confirm
with the user before running it.

### 3. Collect data

```bash
oc cp test-client:data/ clusters/<cluster-name>/data/ -n <namespace>
# Exp 4 data is written locally, move it too
cp data/exp4-results.csv clusters/<cluster-name>/data/ 2>/dev/null
```

### 4. Analyze

```bash
python3 scripts/toolkit/analyze.py clusters/<cluster-name>/data/
```

Read the full output. Note:
- Data quality issues (truncated responses, errors)
- Overhead numbers (disagg median - baseline median)
- Decomposition (T_sidecar, T_prefill_rt)
- Throughput ratios at each concurrency level
- Isolation ratio (>1 = disagg helps, <1 = disagg hurts)
- Fault tolerance: which scenarios recovered, which failed, recovery times

### 5. Write assessment

Create `clusters/<cluster-name>/ASSESSMENT.md` with three sections:

**Setup** — What's deployed. Hardware, model, topology, versions. Be specific
about GPU type, network (RDMA or not), number of prefill/decode instances.

**Performance** — The key numbers from the analysis. Don't dump raw stats,
pick the numbers that tell the story:
- Overhead per request (from exp1/1b)
- Where the overhead comes from (from exp1b decomposition)
- Whether throughput crosses over (from exp2)
- Whether isolation works (from exp3)
- What breaks and what recovers (from exp4)

**Assessment** — The recommendation. Should this deployment use P/D
disaggregation? Why or why not? What would need to change (bigger model,
more GPUs, RDMA, prefill redundancy) to make it worthwhile?

### 6. Compare with previous clusters

If other cluster assessments exist in `clusters/`, read them and note
what's different. Useful comparisons:
- Same model on different hardware
- Same hardware with different topology (more decode GPUs, RDMA vs TCP)
- Overhead budget differences (sidecar, prefill RT, NIXL)
- Recovery time differences (correlates with model size and GPU memory bandwidth)

## Elasticity Scenarios

When the user wants to study elasticity (scaling decode GPUs up/down):

### Scale-down test
```bash
# Record baseline performance at full scale
# Then scale down a decode deployment
oc scale deployment vllm-decode-2 --replicas=0 -n <namespace>

# Run exp1 or exp2 at reduced scale
oc exec test-client -n <namespace> -- python3 /scripts/toolkit/exp1_latency.py

# Observe: does remaining decode handle all traffic? latency impact?
```

### Scale-up test
```bash
# Restore the scaled-down deployment
oc scale deployment vllm-decode-2 --replicas=1 -n <namespace>

# Wait for pod ready
oc wait --for=condition=Ready pod -l app=vllm-decode-2 -n <namespace> --timeout=300s

# Time the recovery: how long until the new pod is handling traffic?
# Run exp1 immediately and note cold-start latency
```

### Adding decode capacity
If the cluster has more GPUs available, test adding a third decode:
```bash
# Copy and modify the decode-2 manifest for decode-3
# Update: deployment name, labels, service name
# Apply and measure throughput improvement
```

Record elasticity results in the assessment under a dedicated section.

## Fault Tolerance Deep Dives

Exp4 now includes automated tests for all major failure scenarios:

- **4a**: Decode pod failure (consumer dies)
- **4b**: Prefill pod failure (producer dies)
- **4d**: Graceful degradation (scale down/up)
- **4e**: Failure under load (kill prefill while requests in flight)
- **4f**: Network partition (block NIXL via NetworkPolicy, observe, remove)

Beyond exp4's automated tests, the user may want to explore:

### Prefill redundancy
```bash
# If multiple prefill GPUs are available:
# Deploy a second prefill, test failover
# Question: does the sidecar discover the backup prefill?
```

Always confirm destructive actions with the user before executing.
