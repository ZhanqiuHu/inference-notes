# Disaggregated Serving + EAGLE3 Speculative Decoding Benchmark

Benchmarking the interaction between disaggregated prefill/decode serving and EAGLE3 speculative decoding in vLLM.

## Setup

| Parameter | Value |
|-----------|-------|
| **Hardware** | NVIDIA H200 (single node) |
| **Model** | Qwen/Qwen3-8B |
| **EAGLE3 Speculator** | RedHatAI/Qwen3-8B-speculator.eagle3 |
| **Dataset** | [philschmid/mt-bench](https://huggingface.co/datasets/philschmid/mt-bench) (80 unique prompts, avg ~76 input tokens, 256 max output tokens). Run 1: 80 prompts; Run 2: 400 prompts (5x repeated). |
| **vLLM Version** | 0.15.0rc2.dev (main branch, commit `c4e744dbd`) |
| **Attention Backend** | FLASH_ATTN |
| **Prefix Caching** | Disabled |
| **CUDA Graphs** | Run 1: Disabled (`--enforce-eager`); Run 2: Enabled |
| **KV Connector** | NixlConnector (for DisAgg configs) |

### Configurations

| Config | Description | GPUs |
|--------|-------------|------|
| **A: Baseline** | Single server, no spec decode, no disagg | 1 |
| **B: Baseline + EAGLE3** | Single server with EAGLE3 (3 spec tokens) | 1 |
| **C: DisAgg (1P1D)** | 1 prefill + 1 decode server, NIXL KV transfer, no spec decode | 2 |
| **D: DisAgg + EAGLE3 (1P1D)** | 1P + 1D with EAGLE3 (1 spec token on prefill, 3 on decode) | 2 |
| **E: DisAgg (2P1D)** | 2 prefill + 1 decode server | 3 |
| **F: DisAgg + EAGLE3 (2P1D)** | 2P + 1D with EAGLE3 | 3 |
| **G: DisAgg (1P2D)** | 1 prefill + 2 decode servers | 3 |
| **H: DisAgg + EAGLE3 (1P2D)** | 1P + 2D with EAGLE3 | 3 |

## Results

### Run 1: 80 prompts, 10 RPS, max concurrency 32, eager mode (Configs A-D only)

| Config | GPUs | RPS | RPS/GPU | Out tok/s | tok/s/GPU | TTFT (ms) | TPOT (ms) | NIXL xfer | Accept Rate | Accept Len |
|--------|------|-----|---------|-----------|-----------|-----------|-----------|-----------|-------------|------------|
| Baseline | 1 | 7.30 | 7.30 | 1869 | 1869 | 39.4 | 11.1 | — | — | — |
| Baseline + EAGLE3 | 1 | 8.09 | 8.09 | 2068 | 2068 | 52.9 | 6.9 | — | 41.2% | 2.24 |
| DisAgg (1P1D) | 2 | 7.19 | 3.60 | 1840 | 920 | 142.2 | 11.3 | 11.1ms | — | — |
| DisAgg + EAGLE3 (1P1D) | 2 | 7.83 | 3.92 | 2005 | 1003 | 173.1 | 7.4 | 11.1ms | 43.6% | 2.23 |

### Run 2: 400 prompts, 25 RPS, max concurrency 32, CUDA graphs enabled (all configs)

| Config | GPUs | RPS | RPS/GPU | Out tok/s | tok/s/GPU | TTFT (ms) | TPOT (ms) | NIXL xfer | NIXL MB/s | Accept Len |
|--------|------|-----|---------|-----------|-----------|-----------|-----------|-----------|-----------|------------|
| Baseline | 1 | 17.05 | 17.05 | 4366 | 4366 | 23.9 | 6.77 | — | — | — |
| Baseline + EAGLE3 | 1 | 21.49 | 21.49 | 5498 | 5498 | 47.3 | 5.12 | — | — | 2.221 |
| DisAgg (1P1D) | 2 | 17.29 | 8.65 | 4426 | 2213 | 78.5 | 6.48 | 6.46ms | 1906 | — |
| DisAgg + EAGLE3 (1P1D) | 2 | 21.54 | 10.77 | 5512 | 2756 | 106.1 | 4.99 | 10.95ms | 1156 | 2.236 |
| DisAgg (2P1D) | 3 | 16.86 | 5.62 | 4315 | 1438 | 114.5 | 6.49 | 5.39ms | 2284 | — |
| DisAgg + EAGLE3 (2P1D) | 3 | 21.31 | 7.10 | 5455 | 1818 | 125.7 | 4.93 | 6.04ms | 2097 | 2.233 |
| DisAgg (1P2D) | 3 | 18.72 | 6.24 | 4791 | 1597 | 72.9 | 5.97 | 5.79ms | 2090 | — |
| DisAgg + EAGLE3 (1P2D) | 3 | 23.38 | 7.79 | 5985 | 1995 | 87.2 | 4.03 | 8.99ms | 1381 | 2.222 |

**Relative to Baseline** (🟢 = favorable, 🔴 = unfavorable):

| Config | RPS | RPS/GPU | Out tok/s | tok/s/GPU | TTFT | TPOT |
|--------|-----|---------|-----------|-----------|------|------|
| Baseline + EAGLE3 | 🟢 +26.0% | 🟢 +26.0% | 🟢 +25.9% | 🟢 +25.9% | 🔴 +98.3% | 🟢 -24.5% |
| DisAgg (1P1D) | 🟢 +1.4% | 🔴 -49.3% | 🟢 +1.4% | 🔴 -49.3% | 🔴 +229% | 🟢 -4.3% |
| DisAgg + EAGLE3 (1P1D) | 🟢 +26.3% | 🔴 -36.8% | 🟢 +26.3% | 🔴 -36.9% | 🔴 +345% | 🟢 -26.4% |
| DisAgg (2P1D) | 🔴 -1.2% | 🔴 -67.1% | 🔴 -1.2% | 🔴 -67.1% | 🔴 +380% | 🟢 -4.1% |
| DisAgg + EAGLE3 (2P1D) | 🟢 +25.0% | 🔴 -58.3% | 🟢 +24.9% | 🔴 -58.4% | 🔴 +427% | 🟢 -27.2% |
| DisAgg (1P2D) | 🟢 +9.8% | 🔴 -63.4% | 🟢 +9.7% | 🔴 -63.4% | 🔴 +206% | 🟢 -11.9% |
| DisAgg + EAGLE3 (1P2D) | 🟢 +37.1% | 🔴 -54.3% | 🟢 +37.1% | 🔴 -54.3% | 🔴 +266% | 🟢 -40.5% |

**EAGLE3 uplift within each DisAgg topology:**

| Topology | RPS | RPS/GPU | Out tok/s | tok/s/GPU | TTFT | TPOT |
|----------|-----|---------|-----------|-----------|------|------|
| 1P1D | 🟢 +24.6% | 🟢 +24.6% | 🟢 +24.5% | 🟢 +24.5% | 🔴 +35.2% | 🟢 -23.1% |
| 2P1D | 🟢 +26.4% | 🟢 +26.4% | 🟢 +26.4% | 🟢 +26.4% | 🔴 +9.7% | 🟢 -24.0% |
| 1P2D | 🟢 +24.9% | 🟢 +24.9% | 🟢 +24.9% | 🟢 +24.9% | 🔴 +19.6% | 🟢 -32.5% |

## Limitations

- All disagg testing is single-node; cross-node NIXL performance may differ.
- MT-bench prompts average ~76 input tokens — relatively short for prefill-heavy workloads.
- Output length is capped at 256 tokens (vLLM default for HF datasets).
- Single benchmark run per configuration; no variance measured across runs.

## Reproduction

```bash
cd /path/to/vllm  # vLLM repo (main branch)

pip install -e .   # or: uv pip install -e .
pip install datasets

# Run 1: eager mode, configs A-D only
python run_bench.py \
  --model-name Qwen/Qwen3-8B \
  --sd-model RedHatAI/Qwen3-8B-speculator.eagle3 \
  --dataset-name hf \
  --hf-name philschmid/mt-bench \
  --num-prompts 80 \
  --request-rate 10 \
  --bench-max-concurrency 32

# Run 2: CUDA graphs, all configs
python run_bench.py \
  --model-name Qwen/Qwen3-8B \
  --sd-model RedHatAI/Qwen3-8B-speculator.eagle3 \
  --dataset-name hf \
  --hf-name philschmid/mt-bench \
  --num-prompts 400 \
  --request-rate 25 \
  --bench-max-concurrency 32 \
  --no-enforce-eager \
  --configs A B C D E F G H

# Summarize results
python summarize.py results/<timestamp>/
```

The script auto-detects available GPUs via `chg` and reserves 2 (or 3 for multi-instance configs). Use `--skip-reserve` with `CUDA_VISIBLE_DEVICES` set if not using `chg`.

## Files

- `run_bench.py` — Benchmark orchestrator (GPU reservation, server lifecycle, benchmarking)
- `summarize.py` — Results parser and comparison table generator
- `collect_metrics.py` — Prometheus metrics scraper for NIXL and spec-decode stats
- `results/` — Raw benchmark outputs (JSON)

---

*Co-authored with [Cursor](https://cursor.com) (Claude claude-4.6-opus).*
