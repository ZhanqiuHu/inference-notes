#!/usr/bin/env python3
"""Hybrid SSM-FA P/D Disagg — lm_eval Accuracy Validation

Launches vLLM server(s) and runs lm_eval gsm8k 5-shot with automated
pass/fail assertions for accuracy, KV transfer hit rate, and error checking.

Modes:
  standalone  — Single vLLM server (no P/D)
  1p1d, 1p2d, 2p1d, 2p2d, etc. — P/D with proxy

Usage:
  python run_lm_eval.py standalone --model nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8
  python run_lm_eval.py 1p1d --model nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8
  python run_lm_eval.py 2p2d --model nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8 --num-prefill 2 --num-decode 2
  python run_lm_eval.py 1p1d --model nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8 --quick     # single prompt only
  python run_lm_eval.py 1p1d --model nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8 --limit 100 # subset

Results: my_wip/mamba_hetero_tp_pd_test/results/<timestamp>_<config>/
"""

import argparse
import atexit
import json
import os
import re
import signal
import socket
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

# ── Constants ────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent
VLLM_ROOT = SCRIPT_DIR.parent.parent
PROXY_SCRIPT = str(
    VLLM_ROOT / "tests/v1/kv_connector/nixl_integration/toy_proxy_server.py"
)

MODELS = {
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8": {
        "max_model_len": 8192,
        "block_size": 128,
        "gpu_mem_util": 0.8,
        "expected_gsm8k": 0.84,
        "hma": True,
    },
    "nvidia/Nemotron-H-8B-Base-8K": {
        "max_model_len": 8192,
        "block_size": 128,
        "gpu_mem_util": 0.8,
        "expected_gsm8k": None,
        "hma": True,
    },
    "Qwen/Qwen3-0.6B": {
        "max_model_len": 4096,
        "block_size": 128,
        "gpu_mem_util": 0.2,
        "expected_gsm8k": 0.41,
        "hma": False,
    },
    "Qwen/Qwen3.5-35B-A3B": {
        "max_model_len": 8192,
        "block_size": 128,
        "gpu_mem_util": 0.8,
        "expected_gsm8k": None,
        "hma": True,
    },
    "deepseek-ai/deepseek-vl2-tiny": {
        "max_model_len": 4096,
        "block_size": 128,
        "gpu_mem_util": 0.8,
        "expected_gsm8k": 0.19,
        "hma": False,
    },
}

TASK = "gsm8k"
NUM_FEWSHOT = 5
NUM_CONCURRENT = 100
FILTER = "exact_match,strict-match"
RTOL = 0.03
CACHE_HIT_RATE_THRESHOLD = 0.99

_child_procs: list[subprocess.Popen] = []
_reserved_gpu_ids: str | None = None
_cleanup_done = False

# ── Terminal colors ──────────────────────────────────────────────────────

_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_RED = "\033[31m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_BLUE = "\033[34m"
_MAGENTA = "\033[35m"
_CYAN = "\033[36m"

_LABEL_COLORS = {
    "P": _BLUE, "D": _MAGENTA, "PROXY": _CYAN, "S": _GREEN,
}

_NOISE_PATTERNS = (
    "Route: /", "Methods: ", "Loading safetensors",
    "Autotuning process", "autotuner.py",
    "Waiting for application startup", "Application startup complete",
    "Started server process", "compilation.py",
    "Enabled custom fusions", "non-default args",
)


def log(msg: str, color: str = ""):
    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    reset = _RESET if color else ""
    print(f"{_BOLD}[{ts}]{_RESET} {color}{msg}{reset}", flush=True)


# ── GPU detection and reservation ────────────────────────────────────────

def detect_total_gpus() -> int:
    """Detect total number of GPUs on the server via nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            count = len([l for l in result.stdout.strip().split("\n") if l.strip()])
            return count
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    try:
        import torch
        return torch.cuda.device_count()
    except ImportError:
        pass
    return 0


def detect_available_gpus() -> list[int]:
    try:
        result = subprocess.run(
            ["chg", "status", "--json"],
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode != 0:
            return []
        gpu_list = json.loads(result.stdout)
        available = []
        for g in gpu_list:
            status = ""
            for key in ("status", "Status", "state", "State"):
                if key in g:
                    status = str(g[key]).upper()
                    break
            if status == "AVAILABLE":
                for key in ("index", "id", "gpu_id", "gpu", "ID", "Index"):
                    if key in g:
                        available.append(int(g[key]))
                        break
        if available:
            log(f"Available GPUs (chg): {available}")
        else:
            log("No GPUs marked AVAILABLE in chg status")
        return available
    except (subprocess.TimeoutExpired, FileNotFoundError,
            json.JSONDecodeError) as e:
        log(f"chg status failed: {e}")
        return []


def reserve_gpus(gpu_ids: list[int]) -> bool:
    global _reserved_gpu_ids
    ids_str = ",".join(str(g) for g in gpu_ids)
    log(f"Reserving GPUs: {ids_str}")
    result = subprocess.run(
        ["chg", "reserve", "--gpu-ids", ids_str, "--duration", "4h"],
        capture_output=True, text=True, timeout=30,
    )
    if result.returncode == 0:
        _reserved_gpu_ids = ids_str
        log(f"GPUs {ids_str} reserved successfully")
        return True
    stderr = result.stderr.strip() or result.stdout.strip()
    log(f"chg reserve failed: {stderr}")
    return False


def release_gpus():
    global _reserved_gpu_ids
    if _reserved_gpu_ids:
        log(f"Releasing GPUs: {_reserved_gpu_ids}")
        subprocess.run(
            ["chg", "release", "--gpu-ids", _reserved_gpu_ids],
            capture_output=True, timeout=10,
        )
        _reserved_gpu_ids = None


def reserve_with_retry(num_gpus: int, gpu_ids_override: str | None = None,
                       max_retries: int = 10000,
                       retry_delay: float = 10.0) -> list[int]:
    for attempt in range(1, max_retries + 1):
        log(f"GPU reservation attempt {attempt}/{max_retries}")
        if gpu_ids_override:
            candidates = [int(x) for x in gpu_ids_override.split(",")]
        else:
            candidates = detect_available_gpus()

        if len(candidates) < num_gpus:
            log(f"Need {num_gpus} GPUs, found {len(candidates)}. "
                f"Retrying in {retry_delay}s...")
            time.sleep(retry_delay)
            continue

        gpus = candidates[:num_gpus]
        if reserve_gpus(gpus):
            return gpus

        log(f"Reservation failed. Retrying in {retry_delay}s...")
        time.sleep(retry_delay)

    log(f"FAIL: Could not reserve GPUs after {max_retries} attempts")
    sys.exit(1)


# ── Utilities ────────────────────────────────────────────────────────────

def git_info() -> tuple[str, str]:
    sha = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True, text=True, cwd=VLLM_ROOT,
    ).stdout.strip()
    branch = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, cwd=VLLM_ROOT,
    ).stdout.strip()
    return sha, branch


def _port_is_free(port: int) -> bool:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", port))
            return True
    except OSError:
        return False


def find_free_ports(n: int, max_retries: int = 5) -> list[int]:
    for attempt in range(max_retries):
        socks, ports = [], []
        for _ in range(n):
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(("", 0))
            ports.append(s.getsockname()[1])
            socks.append(s)
        for s in socks:
            s.close()
        if all(_port_is_free(p) for p in ports):
            return ports
        log(f"Port conflict on attempt {attempt + 1}, retrying...")
        time.sleep(0.5)
    raise RuntimeError(f"Could not find {n} free ports after {max_retries} attempts")


# ── Process management ───────────────────────────────────────────────────

def _kill_tree(pid: int):
    """Kill a process and all its descendants."""
    try:
        os.killpg(os.getpgid(pid), signal.SIGTERM)
        time.sleep(0.5)
        os.killpg(os.getpgid(pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        pass

    try:
        result = subprocess.run(
            ["ps", "--ppid", str(pid), "-o", "pid="],
            capture_output=True, text=True, timeout=5,
        )
        for line in result.stdout.strip().split("\n"):
            child = line.strip()
            if child:
                _kill_tree(int(child))
    except (subprocess.TimeoutExpired, ValueError):
        pass
    try:
        os.kill(pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


def cleanup():
    global _cleanup_done
    if _cleanup_done:
        return
    _cleanup_done = True
    log("Cleaning up servers...")
    for proc in _child_procs:
        _kill_tree(proc.pid)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
    _child_procs.clear()
    release_gpus()
    log("Cleanup done.")


def _colorize(line: str, label: str) -> str | None:
    lc = label.upper()
    base_lc = lc.rstrip("0123456789")
    label_color = _LABEL_COLORS.get(base_lc, _LABEL_COLORS.get(lc, ""))
    colored_prefix = f"{label_color}{_BOLD}[{label}]{_RESET}"

    if any(p in line for p in _NOISE_PATTERNS):
        return None

    if "ERROR" in line or "Traceback" in line:
        return f"  {colored_prefix} {_RED}{_BOLD}{line}{_RESET}"
    if "[MAMBA-LOG]" in line:
        return f"  {colored_prefix} {_CYAN}{line}{_RESET}"
    if "WARNING" in line:
        return f"  {colored_prefix} {_YELLOW}{line}{_RESET}"
    if "ready" in line.lower() or "DONE" in line:
        return f"  {colored_prefix} {_GREEN}{line}{_RESET}"
    return f"  {colored_prefix} {_DIM}{line}{_RESET}"


def start_process(cmd: list[str], env: dict, log_file: str,
                  label: str) -> subprocess.Popen:
    full_env = os.environ.copy()
    full_env.update(env)
    fh = open(log_file, "w")
    proc = subprocess.Popen(
        cmd, env=full_env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        start_new_session=True,
    )

    def _tee(proc, fh, label):
        try:
            for raw in iter(proc.stdout.readline, b""):
                line = raw.decode("utf-8", errors="replace")
                ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                fh.write(f"[{ts}] [{label}] {line}")
                fh.flush()
                colored = _colorize(line, label)
                if colored is not None:
                    print(colored, end="", flush=True)
        except (ValueError, OSError):
            pass
        finally:
            fh.close()

    threading.Thread(target=_tee, args=(proc, fh, label), daemon=True).start()
    _child_procs.append(proc)
    return proc


def wait_for_server(port: int, timeout: int = 600) -> bool:
    log(f"Waiting for server on port {port}...")
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = urlopen(f"http://localhost:{port}/v1/models", timeout=5)
            body = resp.read().decode()
            if '"id"' in body:
                elapsed = int(time.time() - start)
                log(f"Server on port {port} ready ({elapsed}s)")
                return True
        except (URLError, OSError, TimeoutError):
            pass
        time.sleep(5)
    log(f"TIMEOUT: Server on port {port} not ready after {timeout}s", _RED)
    return False


def parse_config(config: str) -> tuple[int, int]:
    """Parse P/D TP config like '1p1d' -> (p_tp=1, d_tp=1)."""
    config = config.lower()
    configs = {
        "1p1d": (1, 1), "1p2d": (1, 2), "2p1d": (2, 1),
        "1p4d": (1, 4), "4p1d": (4, 1), "2p2d": (2, 2),
        "4p4d": (4, 4),
    }
    if config in configs:
        return configs[config]
    sys.exit(f"Unknown config: {config}. Use e.g. 1p1d, 1p2d, 2p1d, 2p2d")


# ── Assertions / Validation ─────────────────────────────────────────────

def scrape_prometheus_metrics(port: int) -> dict[str, float]:
    """Scrape Prometheus metrics from a vLLM server's /metrics endpoint."""
    metrics = {}
    try:
        resp = urlopen(f"http://localhost:{port}/metrics", timeout=10)
        body = resp.read().decode()
        for line in body.split("\n"):
            if line.startswith("#") or not line.strip():
                continue
            # Format: metric_name{labels} value  or  metric_name value
            match = re.match(r'^(\S+?)(?:\{[^}]*\})?\s+([\d.eE+\-]+)', line)
            if match:
                name, val = match.group(1), match.group(2)
                try:
                    fval = float(val)
                    metrics[name] = metrics.get(name, 0.0) + fval
                except ValueError:
                    pass
    except (URLError, OSError, TimeoutError) as e:
        log(f"Failed to scrape metrics from port {port}: {e}", _YELLOW)
    return metrics


def check_cache_hit_rate(decoder_ports: list[int]) -> tuple[bool, float, str]:
    """Check external KV transfer cache hit rate across all decoder instances.

    Returns (passed, hit_rate, detail_msg).
    """
    total_queries = 0.0
    total_hits = 0.0

    for port in decoder_ports:
        metrics = scrape_prometheus_metrics(port)
        queries = metrics.get("vllm:external_prefix_cache_queries_total",
                              metrics.get("vllm:external_prefix_cache_queries", 0))
        hits = metrics.get("vllm:external_prefix_cache_hits_total",
                           metrics.get("vllm:external_prefix_cache_hits", 0))
        total_queries += queries
        total_hits += hits

    if total_queries == 0:
        return False, 0.0, "No external cache queries recorded"

    hit_rate = total_hits / total_queries
    passed = hit_rate >= CACHE_HIT_RATE_THRESHOLD
    detail = (f"hits={total_hits:.0f}, queries={total_queries:.0f}, "
              f"rate={hit_rate:.4f} (threshold={CACHE_HIT_RATE_THRESHOLD})")
    return passed, hit_rate, detail


def scan_logs_for_errors(results_dir: str, log_files: list[str]) -> tuple[bool, int, list[str]]:
    """Scan server log files for errors and write consolidated errors.txt.

    Looks for NIXL transfer errors, tracebacks, crashes, and general ERROR
    lines. Writes all matched lines to errors.txt in the results directory.

    Returns (passed, error_count, error_samples).
    """
    nixl_error_patterns = [
        "transfer_setup_failed",
        "NIXL_ERR_INVALID_PARAM",
        "length mismatch",
        "remote index out of range",
        "nixlInvalidParamError",
    ]
    general_error_patterns = [
        "Traceback (most recent call last)",
        "RuntimeError:",
        "AssertionError:",
        "CUDA error",
        "OutOfMemoryError",
        "NCCL error",
        "Segmentation fault",
        "Killed",
        "SIGKILL",
        "SIGABRT",
    ]
    all_patterns = nixl_error_patterns + general_error_patterns

    nixl_error_count = 0
    error_samples = []
    all_error_lines: list[str] = []
    in_traceback = False
    tb_buffer: list[str] = []
    tb_source = ""

    for log_file in log_files:
        path = os.path.join(results_dir, log_file)
        if not os.path.exists(path):
            continue
        with open(path) as f:
            for line in f:
                stripped = line.strip()

                if "Traceback (most recent call last)" in stripped:
                    in_traceback = True
                    tb_buffer = [f"[{log_file}] {stripped}"]
                    tb_source = log_file
                    continue

                if in_traceback:
                    tb_buffer.append(f"[{tb_source}] {stripped}")
                    if stripped and not stripped.startswith("File ") and \
                       not stripped.startswith("raise ") and \
                       not stripped.startswith("self.") and \
                       not stripped.startswith("return ") and \
                       "Traceback" not in stripped and \
                       len(tb_buffer) > 2:
                        all_error_lines.extend(tb_buffer)
                        all_error_lines.append("")
                        in_traceback = False
                        tb_buffer = []
                    continue

                for pattern in all_patterns:
                    if pattern in stripped:
                        entry = f"[{log_file}] {stripped[:300]}"
                        all_error_lines.append(entry)
                        if pattern in nixl_error_patterns:
                            nixl_error_count += 1
                        if len(error_samples) < 5:
                            error_samples.append(f"[{log_file}] {stripped[:200]}")
                        break

    if tb_buffer:
        all_error_lines.extend(tb_buffer)

    errors_path = os.path.join(results_dir, "errors.txt")
    with open(errors_path, "w") as f:
        if all_error_lines:
            f.write(f"# Errors extracted from server logs\n")
            f.write(f"# Generated: {datetime.now().isoformat()}\n")
            f.write(f"# Total error lines: {len(all_error_lines)}\n")
            f.write(f"# NIXL transfer errors: {nixl_error_count}\n\n")
            for line in all_error_lines:
                f.write(line + "\n")
        else:
            f.write("# No errors found in server logs.\n")
            f.write(f"# Scanned: {', '.join(log_files)}\n")

    total_errors = len([l for l in all_error_lines if l.strip()])
    if all_error_lines:
        log(f"Error log written: {errors_path} ({total_errors} lines)")

    passed = nixl_error_count == 0 and not any(
        any(p in l for p in general_error_patterns) for l in all_error_lines
    )
    return passed, total_errors, error_samples


def run_quick_sanity(eval_url: str, model_name: str,
                     seed: int = 42) -> tuple[bool, str]:
    """Run a single completion prompt to verify basic functionality.

    Shows the full 'prompt + completion' sentence so repetition bugs
    (e.g. Mamba state corruption causing "is is") are immediately visible.
    """
    prompt = "The capital of France is"
    payload = json.dumps({
        "model": model_name,
        "prompt": prompt,
        "max_tokens": 30,
        "temperature": 0.0,
        "seed": seed,
    }).encode()
    req = Request(
        f"{eval_url}/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        resp = urlopen(req, timeout=60)
        body = json.loads(resp.read().decode())
        text = body["choices"][0]["text"].strip()
        full_sentence = f"{prompt} {text}"
        log(f"  Prompt:     \"{prompt}\"")
        log(f"  Completion: \"{text}\"")
        log(f"  Full:       \"{full_sentence}\"")

        is_coherent = len(text) > 0 and not all(c in ' \n\t|.' for c in text)

        # Detect repeated words — strong signal of Mamba state corruption
        words = full_sentence.lower().split()
        repeated = [
            f"'{words[i]}' at pos {i}"
            for i in range(1, len(words))
            if words[i] == words[i - 1] and words[i] not in ("the", "a", "and")
        ]
        if repeated:
            log(f"  {_YELLOW}{_BOLD}WARNING: Repeated words detected: "
                f"{', '.join(repeated)}{_RESET}")
            log(f"  {_YELLOW}This may indicate Mamba state corruption "
                f"in P/D transfer{_RESET}")

        return is_coherent, text
    except Exception as e:
        log(f"Quick sanity FAILED: {e}", _RED)
        return False, str(e)


# ── lm_eval ──────────────────────────────────────────────────────────────

def run_lm_eval(base_url: str, model_name: str, log_file: str,
                limit: int | None = None,
                eval_temperature: float = 0.0,
                log_samples: bool = False,
                num_concurrent: int = 100) -> float | None:
    """Run lm_eval CLI and return the strict-match score (or None)."""
    model_args = (
        f"model={model_name},"
        f"base_url={base_url}/completions,"
        f"num_concurrent={num_concurrent},"
        f"tokenized_requests=False"
    )

    cmd = [
        sys.executable, "-m", "lm_eval",
        "--model", "local-completions",
        "--model_args", model_args,
        "--tasks", TASK,
        "--num_fewshot", str(NUM_FEWSHOT),
        "--output_path", log_file.replace(".log", ""),
        "--gen_kwargs", f"temperature={eval_temperature}",
    ]
    if limit is not None:
        cmd.extend(["--limit", str(limit)])
    if log_samples:
        cmd.append("--log_samples")

    log(f"Starting lm_eval: task={TASK}, fewshot={NUM_FEWSHOT}, "
        f"concurrent={num_concurrent}, limit={limit}, temp={eval_temperature}")
    log(f"Command: {' '.join(cmd)}")

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    _child_procs.append(proc)

    output_lines = []
    for raw in iter(proc.stdout.readline, b""):
        line = raw.decode("utf-8", errors="replace")
        output_lines.append(line)
        print(line, end="", flush=True)

    proc.wait()
    full_output = "".join(output_lines)

    with open(log_file, "w") as f:
        f.write(full_output)

    if proc.returncode != 0:
        log(f"lm_eval exited with code {proc.returncode}", _RED)
        return None

    # Parse the lm_eval results table. The format is:
    # |     |       |strict-match    |     5|exact_match|↑  |0.7104|±  |0.0125|
    for line in output_lines:
        if "strict-match" in line and "exact_match" in line:
            parts = [p.strip() for p in line.split("|")]
            for i, part in enumerate(parts):
                if part.startswith("0.") or part.startswith("1."):
                    try:
                        score = float(part)
                        if 0.0 <= score <= 1.0:
                            log(f"Parsed gsm8k strict-match score: {score}")
                            return score
                    except ValueError:
                        continue

    # Fallback: search for the old format "exact_match,strict-match|...|0.XXXX"
    match = re.search(
        r"exact_match,strict-match\|.*?\|.*?([\d.]+)",
        full_output,
    )
    if match:
        return float(match.group(1))

    log("Could not parse lm_eval score from output", _YELLOW)
    return None


# ── Results ──────────────────────────────────────────────────────────────

def print_summary(checks: list[dict]) -> bool:
    """Print assertion summary table with color-coded results."""
    all_passed = True
    for c in checks:
        if not c["passed"]:
            all_passed = False

    border_color = _GREEN if all_passed else _RED
    log("")
    log(f"{border_color}{'=' * 70}{_RESET}")
    log(f"{_BOLD}  RESULTS SUMMARY{_RESET}")
    log(f"{border_color}{'-' * 70}{_RESET}")
    log(f"  {'Check':<35} {'Value':>12}   Status")
    log(f"{border_color}{'-' * 70}{_RESET}")
    for c in checks:
        if c["passed"]:
            status_str = f"{_GREEN}{_BOLD} PASS {_RESET}"
            row_color = _GREEN
        else:
            status_str = f"{_RED}{_BOLD} FAIL {_RESET}"
            row_color = _RED
        val_str = c.get("value_str", str(c.get("value", "N/A")))
        log(f"  {row_color}{c['name']:<35}{_RESET} {val_str:>12}   {status_str}")
        if c.get("detail"):
            log(f"    {_DIM}{c['detail']}{_RESET}")
    log(f"{border_color}{'=' * 70}{_RESET}")
    if all_passed:
        log(f"  {_GREEN}{_BOLD}ALL CHECKS PASSED{_RESET}")
    else:
        log(f"  {_RED}{_BOLD}SOME CHECKS FAILED{_RESET}")
    log(f"{border_color}{'=' * 70}{_RESET}")
    return all_passed


def notify_slack(webhook_url: str, message: str):
    """Send a notification to Slack via incoming webhook."""
    try:
        payload = json.dumps({"text": message}).encode()
        req = Request(
            webhook_url, payload, {"Content-Type": "application/json"}
        )
        urlopen(req, timeout=10)
    except Exception as e:
        log(f"Slack notification failed: {e}")


def get_slack_webhook(cli_arg: str | None) -> str | None:
    """Resolve webhook URL: CLI arg > env var > ~/.slack_webhook_url file."""
    if cli_arg:
        return cli_arg
    env = os.environ.get("SLACK_WEBHOOK_URL")
    if env:
        return env
    path = os.path.expanduser("~/.slack_webhook_url")
    if os.path.isfile(path):
        with open(path) as f:
            url = f.read().strip()
        if url:
            return url
    return None


def write_results_json(results_dir: str, config: dict, checks: list[dict]):
    """Write machine-parseable results JSON."""
    output = {
        "timestamp": datetime.now().isoformat(),
        "config": config,
        "checks": [
            {
                "name": c["name"],
                "passed": c["passed"],
                "value": c.get("value"),
                "detail": c.get("detail", ""),
            }
            for c in checks
        ],
        "all_passed": all(c["passed"] for c in checks),
    }
    path = os.path.join(results_dir, "results.json")
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    log(f"Results JSON: {path}")


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "config",
        help="'standalone' or P/D TP config: 1p1d, 1p2d, 2p1d, 2p2d")
    parser.add_argument(
        "--model", required=True,
        choices=list(MODELS.keys()),
        help="Full model name")
    parser.add_argument(
        "--tp", type=int, default=None,
        help="TP size for standalone mode (default: auto)")
    parser.add_argument(
        "--num-prefill", type=int, default=1,
        help="Number of prefill instances (default: 1)")
    parser.add_argument(
        "--num-decode", type=int, default=1,
        help="Number of decode instances (default: 1)")
    parser.add_argument(
        "--gpus", default=None,
        help="Comma-separated GPU IDs")
    parser.add_argument(
        "--skip-reserve", action="store_true",
        help="Use --gpus without chg reservation")
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick sanity only: single prompt, no lm_eval")
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Run only N examples (default: full dataset)")
    parser.add_argument(
        "--eval-temperature", type=float, default=0.0,
        help="Temperature for lm_eval generation (default: 0.0 = greedy)")
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for vLLM server and API requests (default: 42)")
    parser.add_argument(
        "--num-concurrent", type=int, default=100,
        help="Number of concurrent lm_eval requests (default: 100)")
    parser.add_argument(
        "--log-samples", action="store_true",
        help="Save per-sample predictions to lm_eval output dir")
    parser.add_argument(
        "--slack-webhook", default=None,
        help="Slack webhook URL for notifications (also reads "
             "SLACK_WEBHOOK_URL env or ~/.slack_webhook_url)")
    parser.add_argument(
        "--skip-assertions", action="store_true",
        help="Don't fail on assertion checks (just print results)")
    args = parser.parse_args()

    # ── Resolve model ────────────────────────────────────────────────
    model_name = args.model
    model_cfg = MODELS[model_name]
    is_standalone = args.config.lower() == "standalone"

    # ── Determine limit ──────────────────────────────────────────────
    if args.quick:
        eval_limit = None  # no lm_eval
    elif args.limit is not None:
        eval_limit = args.limit
    else:
        eval_limit = None  # full dataset

    # ── Compute GPU requirements ─────────────────────────────────────
    if is_standalone:
        tp = args.tp or 2
        num_p, num_d = 0, 0
        p_tp, d_tp = None, None
        needed = tp
    else:
        p_tp, d_tp = parse_config(args.config)
        tp = None
        num_p = args.num_prefill
        num_d = args.num_decode
        needed = num_p * p_tp + num_d * d_tp

    # ── Validate GPU count against server capacity ───────────────────
    total_gpus = detect_total_gpus()
    if total_gpus > 0 and needed > total_gpus:
        sys.exit(
            f"ERROR: Test requires {needed} GPUs but server only has "
            f"{total_gpus}.\n"
            f"  Config: {args.config}, "
            + (f"TP={tp}" if is_standalone else
               f"P_TP={p_tp} x {num_p} + D_TP={d_tp} x {num_d}")
            + f"\n  Reduce TP size, instance count, or use a larger server."
        )

    # ── Reserve GPUs ─────────────────────────────────────────────────
    if args.skip_reserve:
        if not args.gpus:
            sys.exit("--skip-reserve requires --gpus")
        gpu_ids = [int(g) for g in args.gpus.split(",")]
        if len(gpu_ids) < needed:
            sys.exit(f"Need {needed} GPUs, got {len(gpu_ids)}")
    else:
        gpu_ids = reserve_with_retry(
            num_gpus=needed,
            gpu_ids_override=args.gpus,
        )

    atexit.register(cleanup)
    signal.signal(signal.SIGINT, lambda *_: (cleanup(), sys.exit(130)))
    signal.signal(signal.SIGTERM, lambda *_: (cleanup(), sys.exit(143)))

    sha, branch = git_info()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    model_short = model_name.split("/")[-1]
    temp_tag = f"_t{args.eval_temperature:.1f}".replace(".", "")
    chunk_tag = ""
    conc_tag = f"_c{args.num_concurrent}" if args.num_concurrent != 100 else ""
    if is_standalone:
        dir_name = (f"{timestamp}_lmeval_{model_short}_standalone"
                    f"_tp{tp}{temp_tag}{chunk_tag}{conc_tag}_{sha}")
    else:
        inst_suffix = ""
        if num_p > 1 or num_d > 1:
            inst_suffix = f"_{num_p}Px{num_d}D"
        dir_name = (f"{timestamp}_lmeval_{model_short}_{args.config}"
                    f"_ptp{p_tp}_dtp{d_tp}{inst_suffix}{temp_tag}{chunk_tag}{conc_tag}_{sha}")
    results_dir = str(SCRIPT_DIR / "results" / dir_name)
    os.makedirs(results_dir, exist_ok=True)

    # ── Print banner ─────────────────────────────────────────────────
    log("=" * 70)
    mode_str = "quick sanity" if args.quick else (
        f"lm_eval gsm8k {NUM_FEWSHOT}-shot (N={eval_limit or 'ALL'})")
    log(f"{mode_str} — {args.config.upper()}")
    log(f"  Branch: {branch} ({sha})")
    log(f"  Model:  {model_name}")
    log(f"  HMA:    {model_cfg['hma']}")
    if is_standalone:
        log(f"  TP={tp} GPUs={gpu_ids}")
    else:
        log(f"  P_TP={p_tp} x{num_p}, D_TP={d_tp} x{num_d}, GPUs={gpu_ids}")
    log(f"  Logs:   {results_dir}")
    log("=" * 70)

    hma_flag = "--no-disable-hybrid-kv-cache-manager" if model_cfg["hma"] else ""

    # Track decoder ports for metrics scraping
    decoder_ports: list[int] = []
    # Track log files for error scanning
    server_log_files: list[str] = []

    if is_standalone:
        # ── Standalone mode ──────────────────────────────────────────
        ports = find_free_ports(1)
        server_port = ports[0]

        cmd = [
            "vllm", "serve", model_name,
            "--enforce-eager",
            "--block-size", str(model_cfg["block_size"]),
            "--gpu-memory-utilization", str(model_cfg["gpu_mem_util"]),
            "--max-model-len", str(model_cfg["max_model_len"]),
            "--trust-remote-code",
            "--port", str(server_port),
            "--tensor-parallel-size", str(tp),
            "--seed", str(args.seed),
        ]
        if hma_flag:
            cmd.append(hma_flag)

        env = {
            "CUDA_VISIBLE_DEVICES": ",".join(str(g) for g in gpu_ids),
        }
        log(f"Starting standalone server on port {server_port} "
            f"(GPUs {gpu_ids}, TP={tp})")
        server_log_files.append("server.log")
        start_process(cmd, env,
                      os.path.join(results_dir, "server.log"), "S")

        if not wait_for_server(server_port):
            log("FAIL: Server did not start", _RED)
            sys.exit(1)

        eval_url = f"http://localhost:{server_port}/v1"

    else:
        # ── P/D mode (multi-instance) ────────────────────────────────
        # Allocate GPUs: first num_p*p_tp for prefillers, rest for decoders
        gpu_cursor = 0

        kv_config = json.dumps({
            "kv_connector": "NixlConnector", "kv_role": "kv_both",
        })

        base_args = [
            "vllm", "serve", model_name,
            "--enforce-eager",
            "--block-size", str(model_cfg["block_size"]),
            "--gpu-memory-utilization", str(model_cfg["gpu_mem_util"]),
            "--max-model-len", str(model_cfg["max_model_len"]),
            "--trust-remote-code",
            "--kv-transfer-config", kv_config,
            "--seed", str(args.seed),
        ]
        if hma_flag:
            base_args.append(hma_flag)

        # We need ports for: num_p prefillers + num_d decoders + 1 proxy
        # + side channel ports (1 per instance, spaced to avoid TP clash)
        num_ports = num_p + num_d + 1
        ports = find_free_ports(num_ports + num_p + num_d)

        p_ports = []
        d_ports = []
        proxy_port = ports[num_p + num_d]
        sc_base = ports[num_p + num_d + 1:]

        # ── Start prefill instances ──────────────────────────────────
        for i in range(num_p):
            inst_gpus = gpu_ids[gpu_cursor:gpu_cursor + p_tp]
            gpu_cursor += p_tp
            p_port = ports[i]
            p_ports.append(p_port)
            sc_port = sc_base[i]

            p_cmd = base_args + [
                "--port", str(p_port),
                "--tensor-parallel-size", str(p_tp),
            ]
            p_env = {
                "CUDA_VISIBLE_DEVICES": ",".join(str(g) for g in inst_gpus),
                "VLLM_KV_CACHE_LAYOUT": "HND",
                "VLLM_NIXL_SIDE_CHANNEL_PORT": str(sc_port),
            }
            label = f"P{i}" if num_p > 1 else "P"
            log_name = f"prefiller_{i}.log" if num_p > 1 else "prefiller.log"
            server_log_files.append(log_name)
            log(f"Starting prefiller {i} on port {p_port} "
                f"(GPUs {inst_gpus}, TP={p_tp})")
            start_process(p_cmd, p_env,
                          os.path.join(results_dir, log_name), label)

        # ── Start decode instances ───────────────────────────────────
        for i in range(num_d):
            inst_gpus = gpu_ids[gpu_cursor:gpu_cursor + d_tp]
            gpu_cursor += d_tp
            d_port = ports[num_p + i]
            d_ports.append(d_port)
            decoder_ports.append(d_port)
            sc_port = sc_base[num_p + i]

            d_cmd = base_args + [
                "--port", str(d_port),
                "--tensor-parallel-size", str(d_tp),
            ]
            d_env = {
                "CUDA_VISIBLE_DEVICES": ",".join(str(g) for g in inst_gpus),
                "VLLM_KV_CACHE_LAYOUT": "HND",
                "VLLM_NIXL_SIDE_CHANNEL_PORT": str(sc_port),
            }
            label = f"D{i}" if num_d > 1 else "D"
            log_name = f"decoder_{i}.log" if num_d > 1 else "decoder.log"
            server_log_files.append(log_name)
            log(f"Starting decoder {i} on port {d_port} "
                f"(GPUs {inst_gpus}, TP={d_tp})")
            start_process(d_cmd, d_env,
                          os.path.join(results_dir, log_name), label)

        # ── Wait for all instances ───────────────────────────────────
        for i, port in enumerate(p_ports):
            if not wait_for_server(port):
                log(f"FAIL: Prefiller {i} did not start", _RED)
                sys.exit(1)
        for i, port in enumerate(d_ports):
            if not wait_for_server(port):
                log(f"FAIL: Decoder {i} did not start", _RED)
                sys.exit(1)

        # ── Start proxy ──────────────────────────────────────────────
        proxy_cmd = [
            sys.executable, PROXY_SCRIPT,
            "--port", str(proxy_port),
            "--prefiller-ports", *[str(p) for p in p_ports],
            "--decoder-ports", *[str(p) for p in d_ports],
        ]
        server_log_files.append("proxy.log")
        log(f"Starting proxy on port {proxy_port} "
            f"(P ports={p_ports}, D ports={d_ports})")
        start_process(proxy_cmd, {},
                      os.path.join(results_dir, "proxy.log"), "PROXY")
        time.sleep(3)

        eval_url = f"http://localhost:{proxy_port}/v1"

    # ── Collect assertion results ────────────────────────────────────
    checks: list[dict] = []

    # ── Quick sanity check ───────────────────────────────────────────
    log("")
    log("=" * 70)
    log("Running quick sanity check (single prompt)...")
    log("=" * 70)
    sanity_ok, sanity_text = run_quick_sanity(eval_url, model_name,
                                              seed=args.seed)
    checks.append({
        "name": "Quick sanity (coherent output)",
        "passed": sanity_ok,
        "value": sanity_text[:80],
        "value_str": "OK" if sanity_ok else "FAIL",
    })

    if args.quick:
        log("--quick mode: skipping lm_eval")
    else:
        # ── Run lm_eval ──────────────────────────────────────────────
        log("")
        log("=" * 70)
        log(f"Running lm_eval gsm8k {NUM_FEWSHOT}-shot against {eval_url}")
        log("=" * 70)

        measured = run_lm_eval(
            eval_url, model_name,
            os.path.join(results_dir, "lm_eval_output.log"),
            limit=eval_limit,
            eval_temperature=args.eval_temperature,
            log_samples=args.log_samples,
            num_concurrent=args.num_concurrent,
        )
        expected = model_cfg["expected_gsm8k"]

        # Accuracy check
        if measured is not None:
            if expected is not None:
                diff = abs(measured - expected)
                acc_passed = diff <= RTOL
                checks.append({
                    "name": f"Accuracy ({FILTER})",
                    "passed": acc_passed,
                    "value": measured,
                    "value_str": f"{measured:.4f}",
                    "detail": f"expected={expected}, diff={diff:.4f}, "
                              f"rtol={RTOL}",
                })
            else:
                checks.append({
                    "name": f"Accuracy ({FILTER})",
                    "passed": True,
                    "value": measured,
                    "value_str": f"{measured:.4f}",
                    "detail": "No expected baseline (informational only)",
                })
        else:
            checks.append({
                "name": f"Accuracy ({FILTER})",
                "passed": False,
                "value": None,
                "value_str": "N/A",
                "detail": "lm_eval did not return a score",
            })

    # ── Cache hit rate check (P/D mode only) ─────────────────────────
    if decoder_ports:
        hit_passed, hit_rate, hit_detail = check_cache_hit_rate(decoder_ports)
        checks.append({
            "name": "External KV cache hit rate",
            "passed": hit_passed,
            "value": hit_rate,
            "value_str": f"{hit_rate:.2%}" if hit_rate > 0 else "N/A",
            "detail": hit_detail,
        })

    # ── Log error scan ───────────────────────────────────────────────
    err_passed, err_count, err_samples = scan_logs_for_errors(
        results_dir, server_log_files)
    checks.append({
        "name": "Transfer errors in logs",
        "passed": err_passed,
        "value": err_count,
        "value_str": str(err_count),
        "detail": "; ".join(err_samples) if err_samples else "",
    })

    # ── Print summary ────────────────────────────────────────────────
    all_passed = print_summary(checks)

    # ── Write results ────────────────────────────────────────────────
    run_config = {
        "mode": args.config,
        "model": model_name,
        "hma": model_cfg["hma"],
        "branch": branch,
        "sha": sha,
        "gpu_ids": gpu_ids,
        "total_gpus_on_server": total_gpus,
    }
    if is_standalone:
        run_config["tp"] = tp
    else:
        run_config.update({
            "p_tp": p_tp, "d_tp": d_tp,
            "num_prefill": num_p, "num_decode": num_d,
        })
    write_results_json(results_dir, run_config, checks)

    log(f"Results dir: {results_dir}")

    # ── Slack notification ────────────────────────────────────────────
    slack_url = get_slack_webhook(args.slack_webhook)
    if slack_url:
        status = "PASS" if all_passed else "FAIL"
        emoji = ":white_check_mark:" if all_passed else ":x:"
        short_model = model_name.split("/")[-1]
        lines = [
            f"{emoji} *[lm_eval]* `{args.config}` "
            f"`t={args.eval_temperature}` `c={args.num_concurrent}` "
            f"@ `{sha[:9]}`",
            f"Model: `{short_model}`",
        ]
        for c in checks:
            c_emoji = ":white_check_mark:" if c["passed"] else ":x:"
            lines.append(f"{c_emoji} {c['name']}: *{c['value_str']}*")
        lines.append(f"*{status}*")
        notify_slack(slack_url, "\n".join(lines))

    if not all_passed and not args.skip_assertions:
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        slack_url = get_slack_webhook(None)
        if slack_url:
            config_hint = sys.argv[1] if len(sys.argv) > 1 else "?"
            notify_slack(
                slack_url,
                f":boom: *[lm_eval] CRASHED* `{config_hint}`\n"
                f"```{type(exc).__name__}: {exc}```",
            )
        raise
    finally:
        cleanup()
