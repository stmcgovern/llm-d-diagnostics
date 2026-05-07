"""
Cloud GPU pricing tables and cost calculation.

Supports two modes:
  1. **Live fetch** from gpucost.org (call ``refresh_pricing()``).
  2. **Static fallback** compiled from provider pricing pages.

The static tables are always available; live data overwrites them when
a fetch succeeds.  ``get_price`` / ``get_cheapest`` transparently use
whichever data is current.
"""

import json
import sys
import urllib.request

# ── Static fallback tables ───────────────────────────────────────────────
# $/hour per GPU, on-demand pricing.  Sources: provider pricing pages,
# gpucost.org, getdeploying.com.  Snapshot: April 2026.

_STATIC_PRICING = {
    "aws": {
        "t4": 0.53,        # g4dn.xlarge
        "a10g": 1.21,      # g5.xlarge
        "a100_40": 3.67,   # p4d.24xlarge / 8 GPUs
        "a100_80": 4.10,   # p4de.24xlarge / 8 GPUs
        "h100": 6.98,      # p5.48xlarge / 8 GPUs
        "h200": 8.10,      # p5e
        "l40s": 1.83,      # g6.xlarge
    },
    "gcp": {
        "t4": 0.35,
        "a100_40": 2.93,
        "a100_80": 3.67,
        "h100": 5.67,
        "l40s": 1.41,
    },
    "azure": {
        "t4": 0.53,        # NC4as_T4_v3
        "a100_80": 3.40,   # ND96amsr_A100_v4
        "h100": 6.98,      # ND96isr_H100_v5
    },
    "runpod": {
        "t4": 0.25,
        "a100_40": 0.79,
        "a100_80": 1.19,
        "h100": 2.49,
        "h200": 3.99,
        "l40s": 0.69,
    },
    "lambda": {
        "a100_40": 1.10,
        "a100_80": 1.25,
        "h100": 1.99,
        "h200": 3.49,
    },
    "tensordock": {
        "t4": 0.19,
        "a100_40": 0.55,
        "a100_80": 0.80,
        "h100": 1.99,
        "l40s": 0.59,
    },
}

GPU_VRAM_GB = {
    "t4": 16, "a10g": 24, "l40s": 48,
    "a100_40": 40, "a100_80": 80,
    "h100": 80, "h200": 141,
}

GPU_TFLOPS_FP16 = {
    "t4": 65, "a10g": 125, "l40s": 362,
    "a100_40": 312, "a100_80": 312,
    "h100": 989, "h200": 989,
}

GPU_MEM_BW_GBS = {
    "t4": 320, "a10g": 600, "l40s": 864,
    "a100_40": 1555, "a100_80": 2039,
    "h100": 3350, "h200": 4800,
}

GPU_NIC_BW_GBPS = {
    "t4": 25, "a10g": 25, "l40s": 50,
    "a100_40": 200, "a100_80": 200,
    "h100": 400, "h200": 400,
}

# Active pricing table -- starts as a deep copy of static, refreshed by
# ``refresh_pricing()`` when live data is available.
PRICING: dict[str, dict[str, float]] = {
    p: dict(gpus) for p, gpus in _STATIC_PRICING.items()
}

_pricing_source = "static (April 2026)"


# ── Live pricing fetch ───────────────────────────────────────────────────

# GPU name mapping: gpucost.org uses names like "NVIDIA T4", "NVIDIA A100 80GB"
# while we use short keys like "t4", "a100_80".
_GPU_NAME_MAP = {
    "nvidia t4": "t4",
    "nvidia a10g": "a10g",
    "nvidia l40s": "l40s",
    "nvidia a100 40gb": "a100_40",
    "nvidia a100 80gb": "a100_80",
    "nvidia h100": "h100",
    "nvidia h100 80gb": "h100",
    "nvidia h200": "h200",
    "a100": "a100_80",
    "h100": "h100",
    "h200": "h200",
    "t4": "t4",
    "l40s": "l40s",
}

_PROVIDER_NAME_MAP = {
    "amazon web services": "aws",
    "aws": "aws",
    "google cloud": "gcp",
    "gcp": "gcp",
    "microsoft azure": "azure",
    "azure": "azure",
    "runpod": "runpod",
    "lambda": "lambda",
    "lambda labs": "lambda",
    "tensordock": "tensordock",
}


def refresh_pricing(timeout: int = 10) -> bool:
    """Fetch live GPU pricing from gpucost.org.

    Updates the module-level ``PRICING`` dict in place.  Returns True on
    success, False on any failure (network, parse, etc.).  On failure the
    static fallback remains active -- callers never see an error.

    >>> ok = refresh_pricing()
    >>> print(f"Using {'live' if ok else 'static'} pricing")
    """
    global _pricing_source
    try:
        url = "https://cloud-gpus.com/api/gpus"
        req = urllib.request.Request(url, headers={
            "User-Agent": "llm-d-diagnostics/1.0",
            "Accept": "application/json",
        })
        raw = urllib.request.urlopen(req, timeout=timeout).read()
        data = json.loads(raw)

        if not isinstance(data, list) or len(data) == 0:
            return False

        updated = 0
        for entry in data:
            gpu_raw = entry.get("gpu", entry.get("name", "")).strip().lower()
            provider_raw = entry.get("provider", entry.get("cloud", "")).strip().lower()
            price = entry.get("price", entry.get("price_per_hour", 0))

            gpu_key = _GPU_NAME_MAP.get(gpu_raw)
            provider_key = _PROVIDER_NAME_MAP.get(provider_raw)

            if gpu_key and provider_key and isinstance(price, (int, float)) and price > 0:
                if provider_key not in PRICING:
                    PRICING[provider_key] = {}
                PRICING[provider_key][gpu_key] = round(float(price), 4)
                updated += 1

        if updated > 0:
            _pricing_source = f"live (cloud-gpus.com, {updated} entries)"
            return True
        return False

    except Exception:
        return False


def pricing_source() -> str:
    """Return a human-readable string describing the active pricing data source."""
    return _pricing_source


# ── Core API ─────────────────────────────────────────────────────────────

def get_price(provider: str, gpu_type: str) -> float:
    """Get $/hr for a GPU type from a provider. Returns 0 if not found."""
    return PRICING.get(provider.lower(), {}).get(gpu_type.lower(), 0.0)


def get_cheapest(gpu_type: str) -> tuple[str, float]:
    """Find cheapest provider for a GPU type.

    Returns ("", 999.0) if no provider offers the GPU -- callers should
    check for an empty provider string to detect "not found".
    """
    best = ("", 999.0)
    for provider, gpus in PRICING.items():
        price = gpus.get(gpu_type.lower(), 0)
        if 0 < price < best[1]:
            best = (provider, price)
    return best


def cost_per_1k_requests(throughput_rps: float, gpu_count: int,
                          price_per_gpu_hr: float) -> float:
    """Compute $/1000 requests given throughput and GPU cost."""
    if throughput_rps <= 0:
        return float('inf')
    requests_per_hour = throughput_rps * 3600
    cost_per_hour = gpu_count * price_per_gpu_hr
    return round(cost_per_hour / requests_per_hour * 1000, 4)
