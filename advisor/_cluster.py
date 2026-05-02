"""Thin oc/kubectl adapter for advisor commands.

Provides subprocess wrappers for ``oc`` and full-metadata pod discovery.
This is intentionally minimal — operational helpers only, no model profiles
or HuggingFace fetching (those live in plan.py and toolkit/scaling_model.py).
"""

import json
import subprocess


def oc(*args, timeout=60):
    """Run an oc command, raising RuntimeError on non-zero exit."""
    r = subprocess.run(
        ["oc"] + list(args), capture_output=True, text=True, timeout=timeout,
    )
    if r.returncode != 0:
        raise RuntimeError(f"oc {' '.join(args)}: {r.stderr.strip()[:200]}")
    return r.stdout.strip()


def oc_safe(*args, timeout=60):
    """Run an oc command, returning (stdout, stderr) without raising."""
    r = subprocess.run(
        ["oc"] + list(args), capture_output=True, text=True, timeout=timeout,
    )
    return r.stdout.strip(), r.stderr.strip()


def get_pods_full(namespace, label=None):
    """Get full pod metadata: name, ip, ready status, labels, image, args.

    Returns a list of dicts matching the schema used by diagnose/health/rebalance.
    """
    cmd = ["get", "pods", "-n", namespace, "-o", "json"]
    if label:
        cmd.extend(["-l", label])
    out = oc(*cmd)
    items = json.loads(out).get("items", [])
    result = []
    for p in items:
        name = p["metadata"]["name"]
        ip = p.get("status", {}).get("podIP", "")
        ready = any(
            c.get("type") == "Ready" and c.get("status") == "True"
            for c in p.get("status", {}).get("conditions", [])
        )
        containers = p["spec"].get("containers", [{}])
        result.append({
            "name": name,
            "ip": ip,
            "ready": ready,
            "labels": p["metadata"].get("labels", {}),
            "image": containers[0].get("image", ""),
            "args": str(containers[0].get("args", [])),
        })
    return result
