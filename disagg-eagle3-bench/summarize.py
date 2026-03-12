"""Parse benchmark JSON results and NIXL metrics, print comparison table.

Usage:
    python summarize.py <results_dir>

Expects results_dir to contain some/all of:
    config_a_baseline.json              (Baseline)
    config_b_baseline_eagle3.json       (Baseline + EAGLE3)
    config_c_disagg_1p1d.json           (DisAgg 1P1D)
    config_d_disagg_eagle3_1p1d.json    (DisAgg + EAGLE3 1P1D)
    config_{a..h}_metrics.json          (NIXL/spec-decode metrics)
"""

import json
import os
import sys


CONFIG_MAP = {
    "A": {
        "label": "Baseline",
        "bench_file": "config_a_baseline.json",
        "metrics_file": "config_a_metrics.json",
        "num_gpus": 1,
    },
    "B": {
        "label": "Baseline + EAGLE3",
        "bench_file": "config_b_baseline_eagle3.json",
        "metrics_file": "config_b_metrics.json",
        "num_gpus": 1,
    },
    "C": {
        "label": "DisAgg (1P1D)",
        "bench_file": "config_c_disagg_1p1d.json",
        "metrics_file": "config_c_metrics.json",
        "num_gpus": 2,
    },
    "D": {
        "label": "DisAgg + EAGLE3 (1P1D)",
        "bench_file": "config_d_disagg_eagle3_1p1d.json",
        "metrics_file": "config_d_metrics.json",
        "num_gpus": 2,
    },
    "E": {
        "label": "DisAgg (2P1D)",
        "bench_file": "config_e_disagg_2p1d.json",
        "metrics_file": "config_e_metrics.json",
        "num_gpus": 3,
    },
    "F": {
        "label": "DisAgg + EAGLE3 (2P1D)",
        "bench_file": "config_f_disagg_eagle3_2p1d.json",
        "metrics_file": "config_f_metrics.json",
        "num_gpus": 3,
    },
    "G": {
        "label": "DisAgg (1P2D)",
        "bench_file": "config_g_disagg_1p2d.json",
        "metrics_file": "config_g_metrics.json",
        "num_gpus": 3,
    },
    "H": {
        "label": "DisAgg + EAGLE3 (1P2D)",
        "bench_file": "config_h_disagg_eagle3_1p2d.json",
        "metrics_file": "config_h_metrics.json",
        "num_gpus": 3,
    },
}


def load_json(path: str) -> dict | None:
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def fmt(val, suffix="", decimals=2):
    if val is None:
        return "N/A"
    if isinstance(val, float):
        return f"{val:.{decimals}f}{suffix}"
    return f"{val}{suffix}"


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <results_dir>")
        sys.exit(1)

    results_dir = sys.argv[1]
    rows = []

    for config_key in ["A", "B", "C", "D", "E", "F", "G", "H"]:
        cfg = CONFIG_MAP[config_key]
        bench = load_json(os.path.join(results_dir, cfg["bench_file"]))
        metrics = load_json(os.path.join(results_dir, cfg["metrics_file"]))

        if bench is None:
            continue

        rps = bench.get("request_throughput")
        output_tps = bench.get("output_throughput")
        mean_ttft = bench.get("mean_ttft_ms")
        mean_tpot = bench.get("mean_tpot_ms")
        mean_e2e = bench.get("mean_e2el_ms")
        p99_ttft = None
        p99_e2e = None
        for p, v in bench.get("percentiles_ttft_ms", []):
            if p == 99:
                p99_ttft = v
        for p, v in bench.get("percentiles_e2el_ms", []):
            if p == 99:
                p99_e2e = v

        nixl_xfer_ms = None
        nixl_mb_s = None
        nixl_transfers = None
        acceptance_len = None

        if metrics:
            xfer = metrics.get("nixl_xfer_time_seconds")
            if xfer:
                nixl_xfer_ms = xfer.get("mean", 0) * 1000
                nixl_transfers = xfer.get("count", 0)
            nixl_mb_s = metrics.get("nixl_throughput_mb_per_s")
            acceptance_len = metrics.get("spec_decode_acceptance_length")

        num_gpus = cfg["num_gpus"]
        rps_per_gpu = rps / num_gpus if rps else None
        tps_per_gpu = output_tps / num_gpus if output_tps else None

        rows.append({
            "config": config_key,
            "label": cfg["label"],
            "num_gpus": num_gpus,
            "rps": rps,
            "rps_per_gpu": rps_per_gpu,
            "output_tps": output_tps,
            "tps_per_gpu": tps_per_gpu,
            "mean_ttft": mean_ttft,
            "p99_ttft": p99_ttft,
            "mean_tpot": mean_tpot,
            "mean_e2e": mean_e2e,
            "p99_e2e": p99_e2e,
            "nixl_xfer_ms": nixl_xfer_ms,
            "nixl_mb_s": nixl_mb_s,
            "nixl_transfers": nixl_transfers,
            "acceptance_len": acceptance_len,
        })

    if not rows:
        print("No results found.")
        sys.exit(1)

    # ── Print table ──────────────────────────────────────────────────
    print()
    print("=" * 120)
    print("DisAgg + EAGLE3 Benchmark Comparison")
    print("=" * 120)

    headers = [
        ("Config", 26),
        ("GPUs", 4),
        ("RPS", 8),
        ("RPS/GPU", 8),
        ("Out tok/s", 10),
        ("tok/s/GPU", 10),
        ("TTFT(ms)", 10),
        ("P99 TTFT", 10),
        ("TPOT(ms)", 10),
        ("NIXL xfer", 10),
        ("NIXL MB/s", 10),
        ("Accept Len", 10),
    ]

    header_line = " | ".join(h.ljust(w) for h, w in headers)
    print(header_line)
    print("-" * len(header_line))

    for row in rows:
        cols = [
            row["label"].ljust(26),
            str(row["num_gpus"]).ljust(4),
            fmt(row["rps"]).ljust(8),
            fmt(row["rps_per_gpu"]).ljust(8),
            fmt(row["output_tps"]).ljust(10),
            fmt(row["tps_per_gpu"]).ljust(10),
            fmt(row["mean_ttft"]).ljust(10),
            fmt(row["p99_ttft"]).ljust(10),
            fmt(row["mean_tpot"]).ljust(10),
            fmt(row["nixl_xfer_ms"], "ms").ljust(10),
            fmt(row["nixl_mb_s"]).ljust(10),
            fmt(row["acceptance_len"], decimals=3).ljust(10),
        ]
        print(" | ".join(cols))

    print("=" * 120)

    # ── Relative comparisons ─────────────────────────────────────────
    # (metric_label, key, higher_is_better)
    compare_metrics = [
        ("RPS", "rps", True),
        ("RPS/GPU", "rps_per_gpu", True),
        ("Out tok/s", "output_tps", True),
        ("tok/s/GPU", "tps_per_gpu", True),
        ("TTFT", "mean_ttft", False),
        ("TPOT", "mean_tpot", False),
    ]

    def fmt_delta(delta: float, higher_is_better: bool) -> str:
        sign = "+" if delta >= 0 else ""
        good = (delta > 0) == higher_is_better
        indicator = "\U0001f7e2" if good else "\U0001f534"
        return f"{indicator} {sign}{delta:.1f}%"

    def print_relative(base_config: str, base_label: str,
                       compare_configs: list[str] | None = None):
        base = next((r for r in rows if r["config"] == base_config), None)
        if not base or not base["rps"]:
            return
        print()
        print(f"Relative to {base_label}:")
        for row in rows:
            if row["config"] == base_config or row["rps"] is None:
                continue
            if compare_configs and row["config"] not in compare_configs:
                continue
            parts = []
            for metric, key, hib in compare_metrics:
                bv = base.get(key)
                rv = row.get(key)
                if bv and rv:
                    delta = ((rv - bv) / bv) * 100
                    parts.append(f"{fmt_delta(delta, hib)} {metric}")
            print(f"  {row['label']}: {', '.join(parts)}")

    if len(rows) >= 2:
        print_relative("A", "Baseline")
        print_relative("C", "DisAgg (1P1D)", compare_configs=["D"])
        print_relative("E", "DisAgg (2P1D)", compare_configs=["F"])
        print_relative("G", "DisAgg (1P2D)", compare_configs=["H"])

    # ── Save structured summary ──────────────────────────────────────
    summary_json_path = os.path.join(results_dir, "summary.json")
    with open(summary_json_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nStructured summary saved to {summary_json_path}")


if __name__ == "__main__":
    main()
