"""Scrape Prometheus /metrics from a vLLM server for NIXL and spec-decode stats.

Parses the Prometheus text exposition format and extracts NIXL KV connector
histograms/counters and speculative decoding counters into a JSON file.

Usage:
    python collect_metrics.py --port 8200 --output metrics.json
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from urllib.request import urlopen


def fetch_raw_metrics(port: int) -> str:
    url = f"http://localhost:{port}/metrics"
    try:
        return urlopen(url, timeout=10).read().decode()
    except Exception as e:
        print(f"Warning: could not fetch metrics from port {port}: {e}",
              file=sys.stderr)
        return ""


def parse_histogram(lines: list[str], metric_name: str) -> dict:
    """Extract bucket, count, and sum from a Prometheus histogram."""
    buckets = []
    total_count = 0
    total_sum = 0.0

    for line in lines:
        if line.startswith("#"):
            continue
        if line.startswith(f"{metric_name}_bucket"):
            m = re.search(r'le="([^"]+)"', line)
            val = float(line.rsplit(" ", 1)[-1])
            if m:
                buckets.append({"le": m.group(1), "count": val})
        elif line.startswith(f"{metric_name}_count"):
            total_count = float(line.rsplit(" ", 1)[-1])
        elif line.startswith(f"{metric_name}_sum"):
            total_sum = float(line.rsplit(" ", 1)[-1])

    result = {
        "count": total_count,
        "sum": total_sum,
    }
    if total_count > 0:
        result["mean"] = total_sum / total_count
    else:
        result["mean"] = 0.0

    return result


def parse_counter(lines: list[str], metric_name: str) -> float:
    """Extract the value of a Prometheus counter."""
    for line in lines:
        if line.startswith("#"):
            continue
        if (line.startswith(metric_name + "{") or
                line.startswith(metric_name + " ")):
            return float(line.rsplit(" ", 1)[-1])
    return 0.0


def collect(port: int) -> dict:
    raw = fetch_raw_metrics(port)
    if not raw:
        return {"error": "could not fetch metrics", "port": port}

    lines = raw.split("\n")

    metrics = {}

    nixl_histograms = {
        "xfer_time_seconds": "vllm:nixl_xfer_time_seconds",
        "post_time_seconds": "vllm:nixl_post_time_seconds",
        "bytes_transferred": "vllm:nixl_bytes_transferred",
        "num_descriptors": "vllm:nixl_num_descriptors",
    }
    for key, prom_name in nixl_histograms.items():
        data = parse_histogram(lines, prom_name)
        if data["count"] > 0:
            metrics[f"nixl_{key}"] = data

    nixl_counters = {
        "num_failed_transfers": "vllm:nixl_num_failed_transfers",
        "num_failed_notifications": "vllm:nixl_num_failed_notifications",
        "num_kv_expired_reqs": "vllm:nixl_num_kv_expired_reqs",
    }
    for key, prom_name in nixl_counters.items():
        val = parse_counter(lines, prom_name)
        if val > 0:
            metrics[f"nixl_{key}"] = val

    sd_counters = {
        "num_drafts": "vllm:spec_decode_num_drafts_total",
        "num_accepted": "vllm:spec_decode_num_accepted_tokens_total",
        "num_emitted": "vllm:spec_decode_num_emitted_tokens_total",
    }
    for key, prom_name in sd_counters.items():
        val = parse_counter(lines, prom_name)
        if val > 0:
            metrics[f"spec_decode_{key}"] = val

    if "spec_decode_num_drafts" in metrics and metrics["spec_decode_num_drafts"] > 0:
        n_drafts = metrics["spec_decode_num_drafts"]
        n_accepted = metrics.get("spec_decode_num_accepted", 0)
        metrics["spec_decode_acceptance_length"] = 1 + (n_accepted / n_drafts)

    if "nixl_xfer_time_seconds" in metrics and "nixl_bytes_transferred" in metrics:
        xfer = metrics["nixl_xfer_time_seconds"]
        bts = metrics["nixl_bytes_transferred"]
        if xfer["sum"] > 0:
            metrics["nixl_throughput_mb_per_s"] = round(
                (bts["sum"] / (1024 * 1024)) / xfer["sum"], 2
            )

    metrics["port"] = port
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Collect vLLM NIXL metrics")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    metrics = collect(args.port)

    with open(args.output, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Metrics saved to {args.output}")
    if "nixl_xfer_time_seconds" in metrics:
        xfer = metrics["nixl_xfer_time_seconds"]
        print(f"  NIXL transfers: {xfer['count']:.0f}, "
              f"mean xfer: {xfer['mean']*1000:.2f}ms")
    if "nixl_throughput_mb_per_s" in metrics:
        print(f"  NIXL throughput: {metrics['nixl_throughput_mb_per_s']} MB/s")
    if "spec_decode_acceptance_length" in metrics:
        print(f"  Spec decode acceptance length: "
              f"{metrics['spec_decode_acceptance_length']:.3f}")


if __name__ == "__main__":
    main()
