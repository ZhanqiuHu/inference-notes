# SSM-FA (HMA) P/D Disaggregation Eval

Automated accuracy evaluation for hybrid SSM-FA models in vLLM P/D disaggregated serving.

Launches vLLM server(s), runs `lm_eval` GSM8K 5-shot, and reports pass/fail with assertions on accuracy, KV transfer hit rate, and error checks.

## TL;DR

Run a single P/D accuracy test (greedy, full GSM8K dataset, auto GPU reservation):

```bash
python run_lm_eval.py 2p2d \
  --model nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8 \
  --eval-temperature 0.0 \
  --num-concurrent 100
```

Run 3 eval repeats per server session (no server restart between evals):

```bash
python run_lm_eval.py 2p2d \
  --model nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8 \
  --eval-temperature 0.0 \
  --eval-repeats 3
```

Run a full sweep across configs and temperatures:

```bash
python sweep.py \
  --models nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8 \
  --configs standalone 1p1d 2p2d 4p4d \
  --temps 0.0 0.6 0.8 1.0 \
  --server-repeats 2 \   # restart vLLM servers 2 times per config/temp (seed incremented)
  --eval-repeats 3        # run lm_eval 3 times per server session → 6 total evals each
```

Results land in `results/<timestamp>_<config>/` with `results.json`, server logs, and `lm_eval` output.

## Table of Contents

- [Requirements](#requirements)
- [Scripts](#scripts)
  - [`run_lm_eval.py` — Single Run](#run_lm_evalpy--single-run)
    - [Flags](#flags)
    - [Supported Models](#supported-models)
    - [vLLM Serve Flags](#vllm-serve-flags-applied-automatically)
    - [GPU Requirements by Config](#gpu-requirements-by-config)
    - [Output](#output)
  - [`sweep.py` — Multi-Run Sweep](#sweeppy--multi-run-sweep)
    - [Sweep Flags](#flags-1)
- [Eval Repeats](#eval-repeats)
- [Slack Notifications](#slack-notifications)

## Requirements

- vLLM (dev install with NIXL connector support)
- `lm_eval` (`pip install lm_eval`)
- `canhazgpu` (`chg`) for GPU reservation (optional: use `--gpus` + `--skip-reserve` to bypass)
- H100 or equivalent GPUs

## Scripts

### `run_lm_eval.py` — Single Run

Launches vLLM server(s), optionally a proxy for P/D mode, runs `lm_eval`, and reports results.

**Place this script inside your vLLM repo**, e.g. at `my_wip/mamba_hetero_tp_pd_test/run_lm_eval.py`. It expects the proxy script at `tests/v1/kv_connector/nixl_integration/toy_proxy_server.py`.

```bash
# Standalone (no P/D, baseline)
python run_lm_eval.py standalone --model nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8

# P/D with TP=1 per engine (2 GPUs total)
python run_lm_eval.py 1p1d --model nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8

# P/D with TP=2 per engine (4 GPUs total)
python run_lm_eval.py 2p2d --model nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8

# P/D with TP=4 per engine (8 GPUs total)
python run_lm_eval.py 4p4d --model nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8

# Quick sanity check (single prompt, no lm_eval)
python run_lm_eval.py 2p2d --model nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8 --quick

# Subset of examples
python run_lm_eval.py 2p2d --model nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8 --limit 100
```

#### Flags

| Flag | Default | Description |
|------|---------|-------------|
| `config` (positional) | *required* | `standalone`, `1p1d`, `2p2d`, `4p4d`, or any `NpMd` pattern |
| `--model` | *required* | HuggingFace model name (see supported models below) |
| `--tp` | auto | TP size for standalone mode |
| `--num-prefill` | `1` | Number of prefill instances |
| `--num-decode` | `1` | Number of decode instances |
| `--gpus` | auto (via `chg`) | Comma-separated GPU IDs (e.g. `0,1,2,3`) |
| `--skip-reserve` | `False` | Use `--gpus` without `chg` reservation |
| `--quick` | `False` | Quick sanity only: single prompt, no `lm_eval` |
| `--limit` | `None` (full) | Run only N examples |
| `--eval-temperature` | `0.0` | Temperature for `lm_eval` generation (0.0 = greedy) |
| `--seed` | `42` | Random seed for vLLM server |
| `--num-concurrent` | `100` | Number of concurrent `lm_eval` requests |
| `--log-samples` | `False` | Save per-sample predictions to output dir |
| `--slack-webhook` | `None` | Slack webhook URL for notifications (also reads `SLACK_WEBHOOK_URL` env or `~/.slack_webhook_url`) |
| `--skip-assertions` | `False` | Don't fail on assertion checks (just print results) |
| `--eval-repeats` | `1` | Run `lm_eval` N times against the same server session (see [Eval Repeats](#eval-repeats)) |

#### Supported Models

| Model | HMA | max_model_len | gpu_mem_util | Expected GSM8K |
|-------|-----|---------------|-------------|----------------|
| `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8` | Yes | 8192 | 0.8 | 0.84 |
| `nvidia/Nemotron-H-8B-Base-8K` | Yes | 8192 | 0.8 | — |
| `Qwen/Qwen3.5-35B-A3B` | Yes | 8192 | 0.8 | — |
| `Qwen/Qwen3-0.6B` | No | 4096 | 0.2 | 0.41 |
| `deepseek-ai/deepseek-vl2-tiny` | No | 4096 | 0.8 | 0.19 |

#### vLLM Serve Flags (applied automatically)

```
--enforce-eager
--block-size 128
--trust-remote-code
--no-disable-hybrid-kv-cache-manager    (for HMA models)
--kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}'  (for P/D mode)
```

#### GPU Requirements by Config

| Config | GPUs | Layout |
|--------|------|--------|
| `standalone` | 2 | Single server, TP=2 |
| `1p1d` | 2 | P (TP=1, 1 GPU) + D (TP=1, 1 GPU) + proxy |
| `2p2d` | 4 | P (TP=2, 2 GPUs) + D (TP=2, 2 GPUs) + proxy |
| `4p4d` | 8 | P (TP=4, 4 GPUs) + D (TP=4, 4 GPUs) + proxy |

#### Output

Results are saved to `results/<timestamp>_<config>/` containing:
- `results.json` — machine-parseable results
- `lm_eval_output.log` — raw `lm_eval` output
- `prefiller.log`, `decoder.log`, `proxy.log` — server logs
- `errors.txt` — any captured errors

---

### `sweep.py` — Multi-Run Sweep

Orchestrates multiple `run_lm_eval.py` runs across configs, temperatures, and repeats.

```bash
# Default sweep: 3 configs × 4 temps, single server + single eval per combo
python sweep.py --models nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8

# Typical usage: both repeat types
python sweep.py \
  --models nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8 \
  --configs 2p2d 4p4d \
  --temps 0.0 0.6 0.8 1.0 \
  --server-repeats 2 \   # restart servers 2x per config/temp (catches startup variance)
  --eval-repeats 3        # run lm_eval 3x per server session (catches eval variance)
  # total evals per config/temp = 2 × 3 = 6

# Eval-only repeats (no server restarts)
python sweep.py \
  --models nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8 \
  --configs standalone 1p1d 2p2d 4p4d \
  --temps 0.0 \
  --eval-repeats 5        # 5 lm_eval runs per single server session

# Background with nohup
nohup python sweep.py \
  --models nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8 \
  --configs 2p2d 4p4d \
  --temps 0.0 0.6 0.8 1.0 \
  --server-repeats 2 \   # 2 server restarts
  --eval-repeats 3 \     # 3 evals per server → 6 total per config/temp
  > ~/nohup_sweep.log 2>&1 &
```

#### Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--models` | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8` | Model name(s), space-separated |
| `--configs` | `1p1d 2p2d 4p4d` | Configs to test |
| `--temps` | `0.0 0.6 0.8 1.0` | Temperatures to sweep |
| `--server-repeats` | `1` | Server restarts per config/temp (seed incremented per restart) |
| `--eval-repeats` | `1` | `lm_eval` runs per server session (passed through to `run_lm_eval.py`) |
| `--seed` | `42` | Base seed (incremented per server-repeat: 42, 43, 44, ...) |
| `--gpus` | auto (via `chg`) | Comma-separated GPU IDs |
| `--skip-reserve` | `False` | Bypass `chg` reservation |

---

## Eval Repeats

`--eval-repeats N` runs `lm_eval` N times against **the same running server session**, without restarting vLLM between evaluations. This is useful for measuring variance from the evaluation harness itself while holding the server state fixed.

**Behavior:**

| `--eval-repeats` | What happens |
|---|---|
| `1` (default) | Single eval, identical to previous behavior |
| `N > 1` | Server starts once, `lm_eval` runs N times. Each repeat gets its own log file (`lm_eval_output_r1.log`, `_r2.log`, ...). Per-repeat summary and Slack notification are sent. After all repeats, an aggregate summary (mean, std, range) is printed and sent to Slack. |

**Output structure with `--eval-repeats 3`:**

```
results/<timestamp>_<config>/
  server.log / prefiller.log / decoder.log / proxy.log   (shared)
  lm_eval_output_r1.log
  lm_eval_output_r2.log
  lm_eval_output_r3.log
  results.json              (contains all per-repeat scores + aggregate)
```

**`results.json` format with eval-repeats:**

```json
{
  "config": { ... },
  "eval_repeats": 3,
  "repeats": [
    { "repeat": 1, "score": 0.8432, "checks": [...], "all_passed": true },
    { "repeat": 2, "score": 0.8401, "checks": [...], "all_passed": true },
    { "repeat": 3, "score": 0.8464, "checks": [...], "all_passed": true }
  ],
  "aggregate": {
    "scores": [0.8432, 0.8401, 0.8464],
    "mean": 0.8432,
    "std": 0.0032,
    "min": 0.8401,
    "max": 0.8464,
    "n": 3
  }
}
```

**`sweep.py` interaction:**

- `--server-repeats`: controls how many times vLLM servers are restarted per config/temp (seed incremented each time)
- `--eval-repeats`: passed through to `run_lm_eval.py`, controls evals per server session

Total evaluations = `configs × temps × server-repeats × eval-repeats`.

---

## Slack Notifications (optional)

Slack notifications are **off by default** and silently skipped if not configured. To enable per-run notifications:

1. Create a [Slack Incoming Webhook](https://api.slack.com/messaging/webhooks)
2. Provide the URL via any of these (checked in order):
   - `--slack-webhook <URL>` flag on `run_lm_eval.py`
   - `SLACK_WEBHOOK_URL` environment variable
   - `~/.slack_webhook_url` file (one URL per line)

Notifications include pass/fail status, accuracy score, KV hit rate, and error count. Crashes are also reported.
