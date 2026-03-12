#!/usr/bin/env python3
"""DisAgg + EAGLE3 Performance Benchmark

Runs vLLM serving configurations and benchmarks throughput/latency:
  D) Baseline             - 1 server, no spec decode, no disagg
  A) Baseline + EAGLE3   - 1 server, EAGLE3, no disagg
  B) DisAgg (1P1D)       - 1P + 1D, NixlConnector, no spec decode
  C) DisAgg + EAGLE3 (1P1D) - 1P + 1D, NixlConnector + EAGLE3
  E) DisAgg (2P1D)       - 2P + 1D, no spec decode
  F) DisAgg (1P2D)       - 1P + 2D, no spec decode
  G) DisAgg + EAGLE3 (2P1D) - 2P + 1D + EAGLE3
  H) DisAgg + EAGLE3 (1P2D) - 1P + 2D + EAGLE3

Usage:
  python run_bench.py                        # auto-detect GPUs, run D A B C
  python run_bench.py --skip-reserve         # use CUDA_VISIBLE_DEVICES
  python run_bench.py --gpu-ids 0,1          # reserve specific GPUs
  python run_bench.py --configs A C          # run only configs A and C
  python run_bench.py --configs E F --gpu-ids 0,1,2  # multi-instance P/D (3 GPUs)
"""

import argparse
import atexit
import json
import os
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

SCRIPT_DIR = Path(__file__).resolve().parent
GIT_ROOT = subprocess.run(
    ["git", "rev-parse", "--show-toplevel"],
    capture_output=True, text=True, cwd=SCRIPT_DIR
).stdout.strip()

PROXY_SCRIPT = f"{GIT_ROOT}/tests/v1/kv_connector/nixl_integration/toy_proxy_server.py"
METRICS_SCRIPT = str(SCRIPT_DIR / "collect_metrics.py")
SUMMARIZE_SCRIPT = str(SCRIPT_DIR / "summarize.py")

# ── Tracked subprocesses ─────────────────────────────────────────────────
_child_procs: list[subprocess.Popen] = []
_reserved_gpu_ids: str | None = None


def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ── Process & port management ────────────────────────────────────────────

def find_free_port() -> int:
    """Find and return a free TCP port by briefly binding to port 0."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("", 0))
        return s.getsockname()[1]


def find_free_ports(n: int) -> list[int]:
    """Find N distinct free ports."""
    ports = []
    socks = []
    for _ in range(n):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("", 0))
        ports.append(s.getsockname()[1])
        socks.append(s)
    for s in socks:
        s.close()
    return ports


def _kill_proc_tree(pid: int):
    """Kill a process and all its descendants."""
    try:
        result = subprocess.run(
            ["ps", "--ppid", str(pid), "-o", "pid="],
            capture_output=True, text=True, timeout=5,
        )
        for line in result.stdout.strip().split("\n"):
            child_pid = line.strip()
            if child_pid:
                _kill_proc_tree(int(child_pid))
    except (subprocess.TimeoutExpired, ValueError):
        pass
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def cleanup_servers(gpu_ids: list[int] | None = None):
    """Kill all tracked child processes, their trees, and stale vllm procs."""
    log("Cleaning up servers...")

    for proc in _child_procs:
        _kill_proc_tree(proc.pid)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
    _child_procs.clear()

    for pattern in ["vllm serve", "toy_proxy_server", "EngineCore",
                     "vllm.entrypoints"]:
        subprocess.run(
            ["pkill", "-9", "-f", pattern],
            capture_output=True, timeout=5,
        )

    time.sleep(3)

    if gpu_ids:
        check_set = set(gpu_ids)
        for attempt in range(15):
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,memory.used",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode != 0:
                break
            all_freed = True
            for line in result.stdout.strip().split("\n"):
                parts = line.strip().split(",")
                if len(parts) == 2:
                    idx = int(parts[0].strip())
                    used_mb = int(parts[1].strip())
                    if idx in check_set and used_mb > 2000:
                        all_freed = False
            if all_freed:
                break
            log(f"  Waiting for GPU memory release... "
                f"(attempt {attempt + 1}/15)")
            time.sleep(2)

    log("Cleanup done.")


_gpu_lockfile: str | None = None


def _write_gpu_lockfile(gpu_ids: str):
    """Write reserved GPU IDs to a lockfile for crash recovery."""
    global _gpu_lockfile
    _gpu_lockfile = str(SCRIPT_DIR / ".gpu_lock")
    with open(_gpu_lockfile, "w") as f:
        f.write(gpu_ids)


def _remove_gpu_lockfile():
    global _gpu_lockfile
    if _gpu_lockfile and os.path.exists(_gpu_lockfile):
        os.remove(_gpu_lockfile)
        _gpu_lockfile = None


def release_gpus():
    if _reserved_gpu_ids:
        log(f"Releasing GPUs: {_reserved_gpu_ids}")
        subprocess.run(
            ["chg", "release", "--gpu-ids", _reserved_gpu_ids],
            capture_output=True, timeout=10,
        )
        _remove_gpu_lockfile()


_cleanup_done = False


def final_cleanup():
    global _cleanup_done
    if _cleanup_done:
        return
    _cleanup_done = True
    cleanup_servers()
    release_gpus()


def print_log_tail(log_file: str, n_lines: int = 30):
    """Print the last N lines of a log file for error diagnosis."""
    try:
        with open(log_file) as f:
            lines = f.readlines()
        tail = lines[-n_lines:] if len(lines) > n_lines else lines
        log(f"--- Last {len(tail)} lines of {os.path.basename(log_file)} ---")
        for line in tail:
            print(f"  | {line}", end="", flush=True)
        log(f"--- End of {os.path.basename(log_file)} ---")
    except FileNotFoundError:
        log(f"Log file not found: {log_file}")


def start_process(cmd: list[str], env: dict | None = None,
                  log_file: str | None = None,
                  label: str = "") -> subprocess.Popen:
    """Start a background process, track it, and tee output to log file + stdout."""
    full_env = os.environ.copy()
    if env:
        full_env.update(env)

    if log_file:
        fh = open(log_file, "w")
        proc = subprocess.Popen(
            cmd, env=full_env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        # Start a reader thread that tees to both log file and stdout
        import threading

        def _tee_reader(proc, fh, label):
            prefix = f"[{label}] " if label else ""
            try:
                for line in iter(proc.stdout.readline, b""):
                    text = line.decode("utf-8", errors="replace")
                    fh.write(text)
                    fh.flush()
                    print(f"  {prefix}{text}", end="", flush=True)
            except (ValueError, OSError):
                pass
            finally:
                fh.close()

        t = threading.Thread(target=_tee_reader, args=(proc, fh, label),
                             daemon=True)
        t.start()
    else:
        proc = subprocess.Popen(
            cmd, env=full_env,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    _child_procs.append(proc)
    return proc


def wait_for_server(port: int, deadline: int = 600) -> bool:
    """Wait until the vLLM server on `port` reports a loaded model."""
    log(f"Waiting for server on port {port}...")
    start = time.time()
    while time.time() - start < deadline:
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
    log(f"FAIL: Server on port {port} did not start within {deadline}s")
    return False


def collect_metrics(port: int, output_file: str):
    log(f"Collecting metrics from port {port}...")
    result = subprocess.run(
        [sys.executable, METRICS_SCRIPT, "--port", str(port),
         "--output", output_file],
        capture_output=True, text=True, timeout=30,
    )
    if result.returncode != 0:
        log(f"Warning: metric collection failed for port {port}")
    else:
        print(result.stdout, end="")


def run_benchmark(port: int, result_file: str, label: str, cfg: "BenchConfig"):
    log(f"Running benchmark: {label} (port={port}, "
        f"rate={cfg.request_rate}, prompts={cfg.num_prompts}, "
        f"dataset={cfg.dataset_name})")
    bench_log = os.path.join(cfg.results_dir, f"{label}_bench.log")
    cmd = [
        "vllm", "bench", "serve",
        "--port", str(port),
        "--model", cfg.model_name,
        "--dataset-name", cfg.dataset_name,
        "--num-prompts", str(cfg.num_prompts),
        "--request-rate", str(cfg.request_rate),
        "--save-result",
        "--result-dir", cfg.results_dir,
        "--result-filename", result_file,
    ]
    if cfg.dataset_name == "random":
        cmd += [
            "--random-input-len", str(cfg.random_input_len),
            "--random-output-len", str(cfg.random_output_len),
        ]
    elif cfg.dataset_name == "sharegpt":
        cmd += ["--dataset-path", cfg.dataset_path]
        if cfg.sharegpt_output_len:
            cmd += ["--sharegpt-output-len", str(cfg.sharegpt_output_len)]
    elif cfg.dataset_name == "hf":
        cmd += ["--dataset-path", cfg.hf_name]
        if cfg.hf_split:
            cmd += ["--hf-split", cfg.hf_split]
        if cfg.hf_subset:
            cmd += ["--hf-subset", cfg.hf_subset]
        if cfg.hf_output_len:
            cmd += ["--hf-output-len", str(cfg.hf_output_len)]
    if cfg.bench_max_concurrency:
        cmd += ["--max-concurrency", str(cfg.bench_max_concurrency)]
    # Tee benchmark output to both stdout and log file
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    with open(bench_log, "w") as fh:
        for line in iter(proc.stdout.readline, b""):
            text = line.decode("utf-8", errors="replace")
            fh.write(text)
            fh.flush()
            print(text, end="", flush=True)
    proc.wait()

    if proc.returncode != 0:
        log(f"Benchmark {label} exited with code {proc.returncode}")
        return False

    log(f"Benchmark {label} complete. Log: {bench_log}")
    return True


# ── Config dataclass ─────────────────────────────────────────────────────

class BenchConfig:
    def __init__(self, args):
        self.model_name = args.model_name
        self.sd_method = args.sd_method
        self.sd_model = args.sd_model
        self.num_spec_tokens = args.num_spec_tokens
        self.max_model_len = args.max_model_len
        self.gpu_memory_utilization = args.gpu_memory_utilization
        self.block_size = args.block_size
        self.num_prompts = args.num_prompts
        self.request_rate = args.request_rate
        self.random_input_len = args.random_input_len
        self.random_output_len = args.random_output_len
        self.gpu0 = args.gpu0
        self.gpu1 = args.gpu1
        self.gpu2 = getattr(args, "gpu2", None)
        self.results_dir = args.results_dir
        self.dataset_name = args.dataset_name
        self.dataset_path = getattr(args, "dataset_path", None)
        self.sharegpt_output_len = getattr(args, "sharegpt_output_len", None)
        self.hf_name = getattr(args, "hf_name", None)
        self.hf_split = getattr(args, "hf_split", None)
        self.hf_subset = getattr(args, "hf_subset", None)
        self.hf_output_len = getattr(args, "hf_output_len", None)
        self.bench_max_concurrency = getattr(args, "bench_max_concurrency", None)
        self.max_num_seqs = getattr(args, "max_num_seqs", None)
        self.enforce_eager = not getattr(args, "no_enforce_eager", False)

        self.prefill_spec_config = json.dumps({
            "method": self.sd_method,
            "model": self.sd_model,
            "num_speculative_tokens": 1,
            "max_model_len": self.max_model_len,
        })
        self.decode_spec_config = json.dumps({
            "method": self.sd_method,
            "model": self.sd_model,
            "num_speculative_tokens": self.num_spec_tokens,
            "max_model_len": self.max_model_len,
        })
        self.single_spec_config = self.decode_spec_config
        self.kv_config = json.dumps({
            "kv_connector": "NixlConnector",
            "kv_role": "kv_both",
        })

    def base_serve_args(self) -> list[str]:
        args = [
            "vllm", "serve", self.model_name,
            "--max-model-len", str(self.max_model_len),
            "--block-size", str(self.block_size),
            "--gpu-memory-utilization", str(self.gpu_memory_utilization),
            "--attention-backend", "FLASH_ATTN",
            "--no-enable-prefix-caching",
        ]
        if self.enforce_eager:
            args.append("--enforce-eager")
        if self.max_num_seqs:
            args.extend(["--max-num-seqs", str(self.max_num_seqs)])
        return args




# ── Config runners ───────────────────────────────────────────────────────

def run_config_a(cfg: BenchConfig) -> bool:
    """Config A: Baseline (single server, 1 GPU, no spec decode)."""
    port = find_free_port()

    log("")
    log("=" * 60)
    log(f"Config A: Baseline (GPU {cfg.gpu0}, port {port})")
    log("=" * 60)

    cmd = cfg.base_serve_args() + [
        "--port", str(port),
    ]
    server_log = os.path.join(cfg.results_dir, "servers_a.log")
    server_proc = start_process(
        cmd,
        env={"CUDA_VISIBLE_DEVICES": str(cfg.gpu0)},
        log_file=server_log,
        label="server-a",
    )

    time.sleep(3)
    if server_proc.poll() is not None:
        log(f"Config A: server process exited immediately "
            f"(code={server_proc.returncode})")
        print_log_tail(server_log)
        return False

    if not wait_for_server(port):
        log("Config A: server failed to start")
        print_log_tail(server_log)
        return False

    collect_metrics(port,
                    os.path.join(cfg.results_dir, "config_a_metrics_pre.json"))
    ok = run_benchmark(port, "config_a_baseline.json",
                       "config_a", cfg)
    collect_metrics(port,
                    os.path.join(cfg.results_dir, "config_a_metrics.json"))
    cleanup_servers(gpu_ids=[cfg.gpu0, cfg.gpu1])
    log("Config A complete.")
    return ok


def run_config_b(cfg: BenchConfig) -> bool:
    """Config B: Baseline + EAGLE3 (single server, 1 GPU)."""
    port = find_free_port()

    log("")
    log("=" * 60)
    log(f"Config B: Baseline + EAGLE3 (GPU {cfg.gpu0}, port {port})")
    log("=" * 60)

    cmd = cfg.base_serve_args() + [
        "--port", str(port),
        "--speculative-config", cfg.single_spec_config,
    ]
    server_log = os.path.join(cfg.results_dir, "servers_b.log")
    server_proc = start_process(
        cmd,
        env={"CUDA_VISIBLE_DEVICES": str(cfg.gpu0)},
        log_file=server_log,
        label="server-b",
    )

    time.sleep(3)
    if server_proc.poll() is not None:
        log(f"Config B: server process exited immediately "
            f"(code={server_proc.returncode})")
        print_log_tail(server_log)
        return False

    if not wait_for_server(port):
        log("Config B: server failed to start")
        print_log_tail(server_log)
        return False

    collect_metrics(port,
                    os.path.join(cfg.results_dir, "config_b_metrics_pre.json"))
    ok = run_benchmark(port, "config_b_baseline_eagle3.json",
                       "config_b", cfg)
    collect_metrics(port,
                    os.path.join(cfg.results_dir, "config_b_metrics.json"))
    cleanup_servers(gpu_ids=[cfg.gpu0, cfg.gpu1])
    log("Config B complete.")
    return ok


def _start_pd_servers(cfg: BenchConfig, label: str,
                      prefill_gpus: list[int] | None = None,
                      decode_gpus: list[int] | None = None,
                      prefill_spec: str | None = None,
                      decode_spec: str | None = None,
                      ) -> tuple[bool, list[int], list[int], int]:
    """Start prefill + decode + proxy servers for a disagg config.

    Supports multiple prefill and/or decode instances.
    Returns (success, prefill_ports, decode_ports, proxy_port).
    """
    if prefill_gpus is None:
        prefill_gpus = [cfg.gpu0]
    if decode_gpus is None:
        decode_gpus = [cfg.gpu1]

    n_p, n_d = len(prefill_gpus), len(decode_gpus)
    num_ports = n_p + n_d + 1 + n_p + n_d  # ports + proxy + side channels
    ports = find_free_ports(num_ports)
    idx = 0
    prefill_ports = ports[idx:idx + n_p]; idx += n_p
    decode_ports = ports[idx:idx + n_d]; idx += n_d
    proxy_port = ports[idx]; idx += 1
    sc_prefill = ports[idx:idx + n_p]; idx += n_p
    sc_decode = ports[idx:idx + n_d]; idx += n_d

    log(f"  {n_p}P + {n_d}D | prefill GPUs={prefill_gpus}, "
        f"decode GPUs={decode_gpus}, proxy={proxy_port}")

    procs_to_check = []

    for i, (gpu, port, sc) in enumerate(
            zip(prefill_gpus, prefill_ports, sc_prefill)):
        env = {
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "VLLM_KV_CACHE_LAYOUT": "HND",
            "UCX_NET_DEVICES": "all",
            "VLLM_NIXL_SIDE_CHANNEL_PORT": str(sc),
        }
        cmd = cfg.base_serve_args() + [
            "--port", str(port),
            "--kv-transfer-config", cfg.kv_config,
        ]
        if prefill_spec:
            cmd += ["--speculative-config", prefill_spec]
        suffix = f"_prefill{i}" if n_p > 1 else "_prefill"
        lf = os.path.join(cfg.results_dir, f"servers_{label}{suffix}.log")
        proc = start_process(cmd, env=env, log_file=lf,
                             label=f"{label}-prefill{i}")
        procs_to_check.append((proc, f"prefill{i}", lf, port))

    for i, (gpu, port, sc) in enumerate(
            zip(decode_gpus, decode_ports, sc_decode)):
        env = {
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "VLLM_KV_CACHE_LAYOUT": "HND",
            "UCX_NET_DEVICES": "all",
            "VLLM_NIXL_SIDE_CHANNEL_PORT": str(sc),
        }
        cmd = cfg.base_serve_args() + [
            "--port", str(port),
            "--kv-transfer-config", cfg.kv_config,
        ]
        if decode_spec:
            cmd += ["--speculative-config", decode_spec]
        suffix = f"_decode{i}" if n_d > 1 else "_decode"
        lf = os.path.join(cfg.results_dir, f"servers_{label}{suffix}.log")
        proc = start_process(cmd, env=env, log_file=lf,
                             label=f"{label}-decode{i}")
        procs_to_check.append((proc, f"decode{i}", lf, port))

    time.sleep(3)
    for proc, name, lf, _ in procs_to_check:
        if proc.poll() is not None:
            log(f"Config {label.upper()}: {name} server exited immediately "
                f"(code={proc.returncode})")
            print_log_tail(lf)
            return False, [], [], 0

    for _, name, lf, port in procs_to_check:
        if not wait_for_server(port):
            log(f"Config {label.upper()}: {name} server failed to start")
            print_log_tail(lf)
            return False, [], [], 0

    log(f"Starting proxy on port {proxy_port}...")
    proxy_cmd = [
        sys.executable, PROXY_SCRIPT,
        "--port", str(proxy_port),
        "--prefiller-hosts", *["localhost"] * n_p,
        "--prefiller-ports", *[str(p) for p in prefill_ports],
        "--decoder-hosts", *["localhost"] * n_d,
        "--decoder-ports", *[str(p) for p in decode_ports],
    ]
    start_process(
        proxy_cmd,
        log_file=os.path.join(cfg.results_dir, f"servers_{label}_proxy.log"),
        label=f"{label}-proxy",
    )
    time.sleep(5)
    return True, prefill_ports, decode_ports, proxy_port


def _run_disagg(cfg: BenchConfig, label: str, result_file: str,
                prefill_gpus: list[int], decode_gpus: list[int],
                prefill_spec: str | None = None,
                decode_spec: str | None = None) -> bool:
    """Generic runner for any P/D disagg configuration."""
    all_gpus = prefill_gpus + decode_gpus
    n_p, n_d = len(prefill_gpus), len(decode_gpus)
    tag = f"{n_p}P{n_d}D"
    eagle_tag = " + EAGLE3" if decode_spec else ""

    log("")
    log("=" * 60)
    log(f"Config {label.upper()}: DisAgg{eagle_tag} ({tag}) "
        f"GPUs P={prefill_gpus} D={decode_gpus}")
    log("=" * 60)

    ok, prefill_ports, decode_ports, proxy_port = _start_pd_servers(
        cfg, label,
        prefill_gpus=prefill_gpus,
        decode_gpus=decode_gpus,
        prefill_spec=prefill_spec,
        decode_spec=decode_spec,
    )
    if not ok:
        return False

    metrics_port = decode_ports[0]
    collect_metrics(metrics_port,
                    os.path.join(cfg.results_dir,
                                 f"config_{label}_metrics_pre.json"))
    ok = run_benchmark(proxy_port, result_file, f"config_{label}", cfg)
    collect_metrics(metrics_port,
                    os.path.join(cfg.results_dir,
                                 f"config_{label}_metrics.json"))
    cleanup_servers(gpu_ids=all_gpus)
    log(f"Config {label.upper()} complete.")
    return ok


def run_config_c(cfg: BenchConfig) -> bool:
    """Config C: DisAgg 1P1D, no spec decode."""
    return _run_disagg(cfg, "c", "config_c_disagg_1p1d.json",
                       prefill_gpus=[cfg.gpu0], decode_gpus=[cfg.gpu1])


def run_config_d(cfg: BenchConfig) -> bool:
    """Config D: DisAgg 1P1D + EAGLE3."""
    return _run_disagg(cfg, "d", "config_d_disagg_eagle3_1p1d.json",
                       prefill_gpus=[cfg.gpu0], decode_gpus=[cfg.gpu1],
                       prefill_spec=cfg.prefill_spec_config,
                       decode_spec=cfg.decode_spec_config)


def run_config_e(cfg: BenchConfig) -> bool:
    """Config E: DisAgg 2P1D, no spec decode."""
    return _run_disagg(cfg, "e", "config_e_disagg_2p1d.json",
                       prefill_gpus=[cfg.gpu0, cfg.gpu1],
                       decode_gpus=[cfg.gpu2])


def run_config_f(cfg: BenchConfig) -> bool:
    """Config F: DisAgg 2P1D + EAGLE3."""
    return _run_disagg(cfg, "f", "config_f_disagg_eagle3_2p1d.json",
                       prefill_gpus=[cfg.gpu0, cfg.gpu1],
                       decode_gpus=[cfg.gpu2],
                       prefill_spec=cfg.prefill_spec_config,
                       decode_spec=cfg.decode_spec_config)


def run_config_g(cfg: BenchConfig) -> bool:
    """Config G: DisAgg 1P2D, no spec decode."""
    return _run_disagg(cfg, "g", "config_g_disagg_1p2d.json",
                       prefill_gpus=[cfg.gpu0],
                       decode_gpus=[cfg.gpu1, cfg.gpu2])


def run_config_h(cfg: BenchConfig) -> bool:
    """Config H: DisAgg 1P2D + EAGLE3."""
    return _run_disagg(cfg, "h", "config_h_disagg_eagle3_1p2d.json",
                       prefill_gpus=[cfg.gpu0],
                       decode_gpus=[cfg.gpu1, cfg.gpu2],
                       prefill_spec=cfg.prefill_spec_config,
                       decode_spec=cfg.decode_spec_config)


# ── GPU detection & reservation ──────────────────────────────────────────

def _extract_gpu_id(gpu_obj: dict) -> int:
    """Extract GPU index from a chg status JSON object, trying common keys."""
    for key in ("index", "id", "gpu_id", "gpu", "ID", "Index", "GPU"):
        if key in gpu_obj:
            return int(gpu_obj[key])
    raise KeyError(
        f"Cannot find GPU ID key in chg JSON. Keys: {list(gpu_obj.keys())}"
    )


def _extract_gpu_status(gpu_obj: dict) -> str:
    """Extract status string from a chg status JSON object."""
    for key in ("status", "Status", "state", "State"):
        if key in gpu_obj:
            return str(gpu_obj[key]).upper()
    return ""


def detect_available_gpus() -> list[int]:
    """Detect available GPUs using chg status.

    Only uses chg — nvidia-smi cannot tell us about reservations.
    Returns an empty list on failure so the retry loop can keep trying.
    """
    try:
        result = subprocess.run(
            ["chg", "status", "--json"],
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode != 0:
            stderr = result.stderr.strip() or result.stdout.strip()
            log(f"chg status failed (exit {result.returncode}): {stderr}")
            return []

        gpu_list = json.loads(result.stdout)
        if gpu_list:
            log(f"chg JSON sample: {json.dumps(gpu_list[0])}")
        available = [
            _extract_gpu_id(g) for g in gpu_list
            if _extract_gpu_status(g) == "AVAILABLE"
        ]
        if available:
            log(f"Available GPUs (via chg): {available}")
        else:
            log("No GPUs marked AVAILABLE in chg status")
        return available

    except (subprocess.TimeoutExpired, FileNotFoundError,
            json.JSONDecodeError, KeyError) as e:
        log(f"chg status failed: {e}")
        return []


def reserve_gpus(gpu_ids: str) -> bool:
    global _reserved_gpu_ids
    log(f"Reserving GPUs: {gpu_ids}")
    result = subprocess.run(
        ["chg", "reserve", "--gpu-ids", gpu_ids, "--duration", "4h"],
        capture_output=True, text=True, timeout=30,
    )
    if result.returncode == 0:
        _reserved_gpu_ids = gpu_ids
        _write_gpu_lockfile(gpu_ids)
        log(f"GPUs {gpu_ids} reserved successfully")
        return True
    else:
        stderr = result.stderr.strip() or result.stdout.strip()
        log(f"chg reserve failed for GPUs {gpu_ids}: {stderr}")
        return False


def reserve_with_retry(gpu_ids_override: str | None = None,
                       num_gpus: int = 2,
                       max_retries: int = 10,
                       retry_delay: float = 10.0) -> list[int]:
    """Find available GPUs and reserve them, retrying on failure.

    Returns the list of reserved GPU IDs.
    """
    for attempt in range(1, max_retries + 1):
        log(f"GPU reservation attempt {attempt}/{max_retries}")

        if gpu_ids_override:
            candidates = [int(x) for x in gpu_ids_override.split(",")]
        else:
            candidates = detect_available_gpus()

        if len(candidates) < num_gpus:
            log(f"Only {len(candidates)} GPU(s) available: {candidates}. "
                f"Need {num_gpus}. Retrying in {retry_delay}s...")
            time.sleep(retry_delay)
            continue

        gpus = candidates[:num_gpus]
        target = ",".join(str(g) for g in gpus)

        if reserve_gpus(target):
            return gpus

        log(f"Reservation failed. Retrying in {retry_delay}s...")
        time.sleep(retry_delay)

    log(f"FAIL: Could not reserve GPUs after {max_retries} attempts")
    sys.exit(1)


# ── Main ─────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="DisAgg + EAGLE3 Performance Benchmark")
    parser.add_argument("--skip-reserve", action="store_true",
                        help="Skip GPU reservation, use CUDA_VISIBLE_DEVICES")
    parser.add_argument("--gpu-ids", type=str, default="",
                        help="Comma-separated GPU IDs to reserve (e.g. 0,1)")
    parser.add_argument("--configs", nargs="+",
                        default=["A", "B", "C", "D"],
                        help="Configs to run: A=baseline, B=baseline+EAGLE3, "
                             "C=disagg 1P1D, D=disagg+EAGLE3 1P1D, "
                             "E=disagg 2P1D, F=disagg+EAGLE3 2P1D, "
                             "G=disagg 1P2D, H=disagg+EAGLE3 1P2D")

    parser.add_argument("--model-name", type=str,
                        default=os.environ.get("MODEL_NAME",
                                               "meta-llama/Llama-3.1-8B-Instruct"))
    parser.add_argument("--sd-method", type=str,
                        default=os.environ.get("SD_METHOD", "eagle3"))
    parser.add_argument("--sd-model", type=str,
                        default=os.environ.get("SD_MODEL",
                                               "RedHatAI/Llama-3.1-8B-Instruct-speculator.eagle3"))
    parser.add_argument("--num-spec-tokens", type=int,
                        default=int(os.environ.get("NUM_SPEC_TOKENS", "3")))
    parser.add_argument("--max-model-len", type=int,
                        default=int(os.environ.get("MAX_MODEL_LEN", "16384")))
    parser.add_argument("--gpu-memory-utilization", type=float,
                        default=float(os.environ.get("GPU_MEMORY_UTILIZATION", "0.7")))
    parser.add_argument("--block-size", type=int,
                        default=int(os.environ.get("BLOCK_SIZE", "16")))
    parser.add_argument("--num-prompts", type=int,
                        default=int(os.environ.get("NUM_PROMPTS", "200")))
    parser.add_argument("--request-rate", type=str,
                        default=os.environ.get("REQUEST_RATE", "inf"))
    parser.add_argument("--random-input-len", type=int,
                        default=int(os.environ.get("RANDOM_INPUT_LEN", "1024")))
    parser.add_argument("--random-output-len", type=int,
                        default=int(os.environ.get("RANDOM_OUTPUT_LEN", "256")))
    parser.add_argument("--dataset-name", type=str, default="random",
                        choices=["random", "sharegpt", "hf"],
                        help="Dataset to benchmark with")
    parser.add_argument("--dataset-path", type=str, default=None,
                        help="Path to sharegpt dataset file")
    parser.add_argument("--sharegpt-output-len", type=int, default=None,
                        help="Max output length for sharegpt dataset")
    parser.add_argument("--hf-name", type=str, default=None,
                        help="HuggingFace dataset name (e.g. openai/gsm8k)")
    parser.add_argument("--hf-split", type=str, default=None,
                        help="HuggingFace dataset split")
    parser.add_argument("--hf-subset", type=str, default=None,
                        help="HuggingFace dataset subset")
    parser.add_argument("--hf-output-len", type=int, default=None,
                        help="Output length for HF dataset")
    parser.add_argument("--bench-max-concurrency", type=int, default=None,
                        help="Client-side: max in-flight requests from "
                             "benchmark load generator (None = unlimited)")
    parser.add_argument("--max-num-seqs", type=int, default=None,
                        help="Server-side: max sequences per scheduler "
                             "iteration (vLLM default: 256)")
    parser.add_argument("--no-enforce-eager", action="store_true",
                        help="Allow CUDA graphs (don't pass --enforce-eager "
                             "to vllm serve)")
    return parser.parse_args()


def main():
    args = parse_args()

    def _signal_handler(signum, _frame):
        log(f"\nCaught signal {signum}, cleaning up...")
        final_cleanup()
        sys.exit(128 + signum)

    atexit.register(final_cleanup)
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    # ── GPU setup ────────────────────────────────────────────────────
    configs_upper = [c.upper() for c in args.configs]
    gpu_needs = {"A": 1, "B": 1, "C": 2, "D": 2,
                 "E": 3, "F": 3, "G": 3, "H": 3}
    num_gpus_needed = max(gpu_needs.get(c, 2) for c in configs_upper)

    if args.skip_reserve:
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if not cvd:
            log("FAIL: --skip-reserve requires CUDA_VISIBLE_DEVICES to be set")
            sys.exit(1)
        gpus = [int(x) for x in cvd.split(",")]
        log(f"Using pre-reserved GPUs: {cvd}")
    else:
        gpus = reserve_with_retry(
            gpu_ids_override=args.gpu_ids or None,
            num_gpus=num_gpus_needed,
            max_retries=10,
            retry_delay=10.0,
        )

    # ── Results directory ────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode_tag = "cudagraph" if args.no_enforce_eager else "eager"
    results_dir = str(SCRIPT_DIR / "results" / f"{timestamp}_{mode_tag}")
    os.makedirs(results_dir, exist_ok=True)

    args.gpu0 = gpus[0]
    args.gpu1 = gpus[1] if len(gpus) > 1 else None
    args.gpu2 = gpus[2] if len(gpus) > 2 else None
    args.results_dir = results_dir

    cfg = BenchConfig(args)

    log("=" * 60)
    log("DisAgg + EAGLE3 Performance Benchmark")
    log("=" * 60)
    log(f"Model:         {cfg.model_name}")
    log(f"SD model:      {cfg.sd_model}")
    log(f"Spec tokens:   {cfg.num_spec_tokens}")
    log(f"GPUs:          {cfg.gpu0}, {cfg.gpu1}")
    log(f"Num prompts:   {cfg.num_prompts}")
    log(f"Request rate:  {cfg.request_rate}")
    log(f"CUDA graphs:   {'enabled' if not cfg.enforce_eager else 'disabled (eager)'}")
    if cfg.bench_max_concurrency:
        log(f"Bench concurr: {cfg.bench_max_concurrency} (client-side)")
    if cfg.max_num_seqs:
        log(f"Max num seqs:  {cfg.max_num_seqs} (server-side)")
    log(f"Dataset:       {cfg.dataset_name}")
    if cfg.dataset_name == "random":
        log(f"Input/Output:  {cfg.random_input_len}/{cfg.random_output_len}")
    elif cfg.dataset_name == "hf":
        log(f"HF dataset:    {cfg.hf_name} (split={cfg.hf_split})")
    elif cfg.dataset_name == "sharegpt":
        log(f"ShareGPT path: {cfg.dataset_path}")
    log(f"Results dir:   {results_dir}")
    log(f"Configs:       {' '.join(configs_upper)}")
    log("=" * 60)

    # ── Run configs ──────────────────────────────────────────────────
    runners = {
        "A": run_config_a, "B": run_config_b,
        "C": run_config_c, "D": run_config_d,
        "E": run_config_e, "F": run_config_f,
        "G": run_config_g, "H": run_config_h,
    }
    failed = []

    for c in configs_upper:
        runner = runners.get(c)
        if not runner:
            log(f"Unknown config: {c} (use A-H)")
            continue
        try:
            ok = runner(cfg)
            if not ok:
                failed.append(c)
                cleanup_servers(gpu_ids=[cfg.gpu0, cfg.gpu1])
        except Exception as e:
            log(f"Config {c} raised exception: {e}")
            failed.append(c)
            cleanup_servers(gpu_ids=[cfg.gpu0, cfg.gpu1])

    # ── Summary ──────────────────────────────────────────────────────
    log("")
    log("=" * 60)
    log("Generating summary...")
    log("=" * 60)

    summary_file = os.path.join(results_dir, "summary.txt")
    result = subprocess.run(
        [sys.executable, SUMMARIZE_SCRIPT, results_dir],
        capture_output=True, text=True, timeout=30,
    )
    print(result.stdout, end="")
    with open(summary_file, "w") as f:
        f.write(result.stdout)

    log("")
    if failed:
        log(f"WARNING: The following configs failed: {' '.join(failed)}")
    log(f"All benchmarks complete. Results in: {results_dir}")


if __name__ == "__main__":
    try:
        main()
    finally:
        final_cleanup()
