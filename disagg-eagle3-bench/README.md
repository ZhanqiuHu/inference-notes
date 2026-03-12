# Does EAGLE3 Help Disaggregated Serving in vLLM?

**TL;DR**: On a single-node 1P1D (1 prefill + 1 decode) setup with Qwen3-8B, EAGLE3 speculative decoding improves throughput by ~9% and cuts TPOT by ~34% when added to disaggregated serving. However, disaggregation itself halves per-GPU efficiency compared to a single-GPU baseline, and significantly increases TTFT due to KV cache transfer overhead.

## Motivation

A recent discussion in the vLLM community reported that combining **disaggregated prefill/decode** (DisAgg) with **EAGLE3 speculative decoding** resulted in *worse* performance than plain disaggregation. This was surprising — EAGLE3 consistently helps in aggregated (single-server) mode, so why would it hurt when combined with DisAgg?

We set out to reproduce this and understand the interaction between these two features.

## Background

- **Disaggregated Serving (DisAgg)**: Separates the prefill and decode phases onto different GPUs, connected via [NIXL](https://github.com/ai-dynamo/nixl) for KV cache transfer. The idea is to independently scale prefill and decode capacity.
- **EAGLE3**: A speculative decoding method that uses a lightweight draft model to propose multiple tokens at once, which the target model verifies in a single forward pass. This reduces the number of decode iterations needed.

## Setup

| Parameter | Value |
|-----------|-------|
| **Hardware** | NVIDIA H200 (single node) |
| **Model** | Qwen/Qwen3-8B |
| **EAGLE3 Speculator** | RedHatAI/Qwen3-8B-speculator.eagle3 |
| **Dataset** | [philschmid/mt-bench](https://huggingface.co/datasets/philschmid/mt-bench) (80 prompts, avg ~76 input tokens, 256 output tokens) |
| **vLLM Version** | 0.15.0rc2.dev (main branch) |
| **Attention Backend** | FLASH_ATTN |
| **Prefix Caching** | Disabled |
| **CUDA Graphs** | Disabled (enforce_eager) |
| **KV Connector** | NixlConnector (for DisAgg configs) |

### Configurations

We benchmark 4 configurations:

| Config | Description | GPUs |
|--------|-------------|------|
| **Baseline** | Single server, no spec decode, no disagg | 1 |
| **Baseline + EAGLE3** | Single server with EAGLE3 (3 spec tokens) | 1 |
| **DisAgg (1P1D)** | 1 prefill + 1 decode server, NIXL transfer, no spec decode | 2 |
| **DisAgg + EAGLE3 (1P1D)** | 1P + 1D with EAGLE3 (1 spec token on prefill, 3 on decode) | 2 |

## Results

### Run 1: 80 prompts, 10 RPS, max concurrency 32

| Config | GPUs | RPS | RPS/GPU | Out tok/s | tok/s/GPU | TTFT (ms) | TPOT (ms) | NIXL xfer | Accept Len |
|--------|------|-----|---------|-----------|-----------|-----------|-----------|-----------|------------|
| Baseline | 1 | 7.30 | 7.30 | 1868 | 1868 | 39.4 | 11.1 | — | — |
| Baseline + EAGLE3 | 1 | 8.09 | 8.09 | 2068 | 2068 | 52.9 | 6.9 | — | 2.24 |
| DisAgg (1P1D) | 2 | 7.19 | 3.60 | 1840 | 920 | 142.2 | 11.3 | 11.1ms | — |
| DisAgg + EAGLE3 (1P1D) | 2 | 7.83 | 3.92 | 2005 | 1003 | 173.1 | 7.4 | 11.1ms | 2.23 |

**Relative to Baseline:**

| Config | RPS | RPS/GPU | Out tok/s | tok/s/GPU | TTFT | TPOT |
|--------|-----|---------|-----------|-----------|------|------|
| Baseline + EAGLE3 | +10.8% | +10.8% | +10.7% | +10.7% | +34.3% | **-38.1%** |
| DisAgg (1P1D) | -1.5% | **-50.7%** | -1.5% | **-50.8%** | **+261%** | +1.7% |
| DisAgg + EAGLE3 (1P1D) | +7.3% | **-46.4%** | +7.3% | **-46.3%** | **+340%** | **-33.1%** |

**Relative to DisAgg (1P1D)** — does adding EAGLE3 help within disagg?

| Config | RPS | Out tok/s | TTFT | TPOT |
|--------|-----|-----------|------|------|
| DisAgg + EAGLE3 (1P1D) | **+8.9%** | **+9.0%** | +21.8% | **-34.2%** |

<!-- ### Run 2: 400 prompts, 15 RPS, max concurrency 32

TODO: Add results from second run
-->

## Analysis

### EAGLE3 consistently helps throughput

In both aggregated and disaggregated modes, EAGLE3 delivers ~10% higher throughput and ~34-38% lower TPOT. The spec decode acceptance length is ~2.23 in both modes (acceptance rate ~41%), meaning EAGLE3's effectiveness is **not degraded** by disaggregation.

### Disaggregation halves per-GPU efficiency at this scale

With 1P1D on a single node, DisAgg uses 2 GPUs to achieve roughly the same total throughput as 1 GPU alone. The per-GPU efficiency (RPS/GPU, tok/s/GPU) is cut in half. This is expected at small scale — disagg's value proposition is independent scaling of prefill and decode workers across many nodes, not 1P1D on a single machine.

### TTFT is the main casualty

DisAgg adds significant TTFT overhead:
- **Baseline → DisAgg**: 39ms → 142ms (+261%)
- **Baseline + EAGLE3 → DisAgg + EAGLE3**: 53ms → 173ms (+227%)

The NIXL KV cache transfer is only ~11ms, so the bulk of the TTFT overhead comes from:
1. **Proxy routing** — requests pass through a Python proxy server
2. **Two-phase scheduling** — prefill and decode have separate schedulers
3. **NIXL handshake/descriptor setup** — beyond bulk data transfer
4. **Queueing** — P99 TTFT of ~860-878ms indicates burst queueing at the prefill server

### We did NOT reproduce the reported regression

In our tests, DisAgg + EAGLE3 was consistently **faster** than plain DisAgg (+8.9% RPS). We did not observe EAGLE3 hurting disaggregated performance. Possible reasons for the discrepancy with the original report:
- Different model / hardware / vLLM version
- Different load patterns or concurrency levels
- The regression may only appear under specific conditions we haven't tested

## Caveats

- **Small scale**: 1P1D on a single node is the minimal disagg setup. Results may differ with more prefill/decode workers or cross-node deployments.
- **Short prompts**: MT-bench has ~76 tokens average input — this doesn't stress the prefill side, making the disagg overhead proportionally large.
- **Fixed output length**: All responses are capped at 256 tokens. Longer generation would give EAGLE3 more room to amortize overhead.
- **No CUDA graphs**: We ran with `--enforce-eager`, which is slower than production settings. With CUDA graphs enabled, absolute numbers would improve.
- **Single run**: Results from a single benchmark run. Variance across runs is not yet characterized.

## Reproduction

```bash
# Reserve 2 GPUs (adjust for your setup)
python run_bench.py \
  --model-name Qwen/Qwen3-8B \
  --sd-model RedHatAI/Qwen3-8B-speculator.eagle3 \
  --dataset-name hf \
  --hf-name philschmid/mt-bench \
  --num-prompts 80 \
  --request-rate 10 \
  --bench-max-concurrency 32

# Summarize results
python summarize.py results/<timestamp>/
```

Requires: vLLM (main branch), NIXL, `datasets` Python package.

## Files

- `run_bench.py` — Main benchmark script (GPU reservation, server lifecycle, benchmarking)
- `summarize.py` — Results parser and comparison table generator
- `collect_metrics.py` — Prometheus metrics scraper for NIXL and spec-decode stats
- `results/` — Raw benchmark outputs (JSON + logs)
