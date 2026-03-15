#!/usr/bin/env python3
"""Config sweep for Mamba P/D validation.

Runs run_lm_eval.py across multiple configs, temperatures, and repeats.
Continues on failure. Each run saves its own results in results/<timestamp>/.

Usage:
  python my_wip/mamba_hetero_tp_pd_test/sweep.py --gpus 0,1,2,3,4,5,6,7
  python my_wip/mamba_hetero_tp_pd_test/sweep.py --gpus 0,1,2,3 --configs 1p1d 2p2d
  python my_wip/mamba_hetero_tp_pd_test/sweep.py --gpus 0,1 --configs 1p1d --temps 0.0 --server-repeats 1
  python my_wip/mamba_hetero_tp_pd_test/sweep.py --gpus 0,1,2,3 --model Qwen/Qwen3-0.6B --wait-gpus
"""

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
RUN_LMEVAL = str(SCRIPT_DIR / "run_lm_eval.py")
DEFAULT_MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8"

GPU_REQUIREMENTS = {
    "standalone": 2,
    "1p1d": 2,
    "2p2d": 4,
    "4p4d": 8,
}


def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def get_gpu_utilization(gpu_ids: list[int]) -> dict[int, float]:
    """Query GPU utilization % for given GPU IDs via nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=index,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10)
        utils = {}
        for line in result.stdout.strip().split("\n"):
            parts = line.split(",")
            if len(parts) == 2:
                idx = int(parts[0].strip())
                util = float(parts[1].strip())
                if idx in gpu_ids:
                    utils[idx] = util
        return utils
    except Exception:
        return {}


def wait_for_idle_gpus(gpu_ids: list[int], idle_minutes: float = 5.0,
                       threshold: float = 5.0, poll_interval: float = 30.0):
    """Block until all GPUs have been below threshold% for idle_minutes."""
    idle_since = None
    idle_needed = idle_minutes * 60

    log(f"Waiting for GPUs {gpu_ids} to be idle "
        f"(<{threshold}% util for {idle_minutes} min)...")

    while True:
        utils = get_gpu_utilization(gpu_ids)
        if not utils:
            log("  Could not query GPU utilization, retrying...")
            idle_since = None
            time.sleep(poll_interval)
            continue

        all_idle = all(u < threshold for u in utils.values())
        max_util = max(utils.values()) if utils else 0

        if all_idle:
            if idle_since is None:
                idle_since = time.monotonic()
                log(f"  GPUs idle (max={max_util:.0f}%), "
                    f"waiting {idle_minutes} min to confirm...")
            elapsed = time.monotonic() - idle_since
            if elapsed >= idle_needed:
                log(f"  GPUs confirmed idle for {idle_minutes} min. Starting!")
                return
        else:
            if idle_since is not None:
                log(f"  GPU activity detected (max={max_util:.0f}%), "
                    f"resetting timer...")
            idle_since = None

        time.sleep(poll_interval)


def main():
    parser = argparse.ArgumentParser(description="Config sweep for Mamba P/D")
    parser.add_argument("--gpus", default=None,
                        help="Comma-separated GPU IDs (optional: "
                             "if omitted, each run uses chg to reserve)")
    parser.add_argument("--skip-reserve", action="store_true",
                        help="Use --gpus without chg reservation")
    parser.add_argument("--models", nargs="+", default=[DEFAULT_MODEL],
                        help=f"Model name(s) (default: {DEFAULT_MODEL})")
    parser.add_argument("--configs", nargs="+",
                        default=["1p1d", "2p2d", "4p4d"],
                        help="Configs to test (default: 1p1d 2p2d 4p4d)")
    parser.add_argument("--temps", nargs="+", type=float,
                        default=[0.0, 0.6, 0.8, 1.0],
                        help="Temperatures (default: 0.0 0.6 0.8 1.0)")
    parser.add_argument("--server-repeats", type=int, default=1,
                        help="Server restarts per config/temp (default: 1)")
    parser.add_argument("--eval-repeats", type=int, default=1,
                        help="lm_eval runs per server session, passed to "
                             "run_lm_eval.py (default: 1)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base seed (incremented per server-repeat)")
    args = parser.parse_args()

    total_server_runs = (len(args.models) * len(args.configs)
                         * len(args.temps) * args.server_repeats)
    total_evals = total_server_runs * args.eval_repeats

    log(f"Sweep: {len(args.models)} models x {len(args.configs)} configs "
        f"x {len(args.temps)} temps x {args.server_repeats} server-repeats"
        f" x {args.eval_repeats} eval-repeats"
        f" = {total_server_runs} server runs, {total_evals} total evals")
    log(f"  Models:  {args.models}")
    log(f"  Configs: {args.configs}")
    log(f"  Temps:   {args.temps}")
    log(f"  GPUs:    {args.gpus or 'chg auto-reserve per run'}")
    log(f"  Seed:    {args.seed} (incremented per server-repeat)")

    print()
    run_idx = 0
    for model in args.models:
        log(f"=== Model: {model} ===")
        for config in args.configs:
            for temp in args.temps:
                for rep in range(1, args.server_repeats + 1):
                    run_idx += 1
                    seed = args.seed + rep - 1
                    er_label = (f" eval-repeats={args.eval_repeats}"
                                if args.eval_repeats > 1 else "")
                    log(f"[{run_idx}/{total_server_runs}] "
                        f"{model.split('/')[-1]} {config} temp={temp} "
                        f"server-repeat={rep}/{args.server_repeats} "
                        f"seed={seed}{er_label}")

                    cmd = [
                        sys.executable, RUN_LMEVAL, config,
                        "--model", model,
                        "--eval-temperature", str(temp),
                        "--seed", str(seed),
                    ]
                    if args.eval_repeats > 1:
                        cmd += ["--eval-repeats", str(args.eval_repeats)]
                    if args.gpus:
                        cmd += ["--gpus", args.gpus]
                    if args.skip_reserve:
                        cmd += ["--skip-reserve"]

                    result = subprocess.run(cmd)
                    log(f"  -> exit_code={result.returncode}")
                    print()

    log(f"Sweep complete. {total_server_runs} server runs attempted "
        f"({total_evals} total evals).")
    log(f"Results in: {SCRIPT_DIR / 'results'}/")


if __name__ == "__main__":
    main()
