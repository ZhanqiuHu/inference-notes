# vLLM v0.13.0 Backport Triage Tool

Identifies bugfix PRs merged into vLLM `main` after v0.13.0 that are candidates for backporting to the RHAI inference server (which ships on v0.13.0).

## Problem

vLLM v0.13.0 was released 2025-12-18. Since then, thousands of PRs have been merged, many fixing bugs that also exist in v0.13.0. We need to find and backport the relevant fixes without pulling in new features.

## How It Works

Single script, four phases, each saving an intermediate CSV:

### Phase 1 — Fetch

Queries GitHub for all merged PRs matching bugfix patterns (`label:bug`, `[Bugfix]`/`[BugFix]` in title). Uses 7-day date windows to avoid GitHub's 1,000-result API limit. Also fetches all "Revert" PRs to detect reverted bugfixes.

### Phase 2 — Classify → `phase2_all_prs_v0.13.0.csv`

Quick-classifies each PR:

| Type | Criteria | Action |
|------|----------|--------|
| `runtime_bug` | Has `bug` label, or title matches `[Bugfix]`/`[BugFix]`/`[Fix]` | Proceed to Phase 3 (NVIDIA) |
| `unclear` | Has "fix" in title but no clear prefix | Proceed to Phase 3 (NVIDIA) |
| `not_bugfix` | Title starts with `[Feature]`, `[Perf]`, `[CI]`, `[Refactor]`, etc. | Skip |
| `platform_specific` | Has `rocm`/`tpu`/`cpu`/`xpu` label | Phase 3 (separate platform CSVs) |

PRs that were later reverted are flagged with `reverted_by`.

### Phase 3 — File-Level Filter

Processes both NVIDIA and platform-specific PRs:

**NVIDIA** → `phase3_candidates_v0.13.0.csv`:
- Fetches files, approvers, reviewers, merger, line counts (batched GraphQL)
- Checks if each file existed at the `v0.13.0` tag
- Detects subsystem (attention, engine, models, etc.)
- Assigns a priority score (0-100)

**Platform-specific** → `platform_{rocm,tpu,cpu,xpu,intel-gpu}_v0.13.0.csv`:
- Same file-level filter applied per platform
- One CSV per platform for independent review

Priority scoring: `runtime_bug`=+40, file coverage up to +30, core subsystem +20, models/quant +10.

### Phase 4 — Blame-Level Filter → `phase4_blame_v0.13.0.csv`

For each Phase 3 NVIDIA candidate:
- Parses the merge commit diff to find modified line numbers
- Runs `git blame` at the **parent commit** (`merge_sha~1`) for each modified line
- Checks if the commit that introduced each line predates the v0.13.0 tag date
- Computes a `blame_score` (0-100%) representing the fraction of modified lines that existed in v0.13.0

This avoids line-number-drift issues (where lines shift between the tag and the merge parent due to intervening commits).

Groups results by confidence:
- **High** (≥80%): bug very likely exists in v0.13.0
- **Medium** (1-79%): partially applicable
- **Low** (0%): likely fixes code introduced after v0.13.0

## Usage

```bash
cd /path/to/vllm
python find_backport_candidates.py
```

No flags. Runs all four phases sequentially. All output goes to stdout + `triage_v0.13.0.log`.

Prerequisites: `gh` (GitHub CLI, authenticated), `git`, Python 3.10+, inside a vllm repo with `v0.13.0` tag.

### Output Files

```
phase2_all_prs_v0.13.0.csv           — All PRs with classification
phase3_candidates_v0.13.0.csv        — NVIDIA candidates with priority, subsystem, approvers
phase4_blame_v0.13.0.csv             — Final NVIDIA results with blame score per PR
platform_rocm_v0.13.0.csv            — ROCm-specific bugfixes
platform_tpu_v0.13.0.csv             — TPU-specific bugfixes
platform_cpu_v0.13.0.csv             — CPU-specific bugfixes
platform_xpu_v0.13.0.csv             — XPU-specific bugfixes
platform_intel-gpu_v0.13.0.csv       — Intel GPU-specific bugfixes
triage_v0.13.0.log                   — Full readable log
```

### Key CSV Columns

Phase 3: `priority`, `pr`, `title`, `subsystems`, `approvers`, `merged_by`, `reverted_by`, `verdict`, ...

Phase 4 adds: `blame_score`, `blame_detail`

Platform CSVs add: `platform`

## Next Steps

1. Review the Phase 4 CSV — filter by blame score, subsystem, and priority
2. Check `reverted_by` column — skip PRs that were later reverted
3. Scope to RHAI's actual deployment (which models, features, quantization methods)
4. Create a GitHub milestone for confirmed backports
5. Use vLLM's built-in cherry-pick script: `.buildkite/scripts/cherry-pick-from-milestone.sh`

## Claude Suggested Further Filtering Strategies

### A. Cross-reference with later patch releases (low effort, high signal)

vLLM's patch releases (v0.14.1, v0.15.1, v0.17.1) each have cherry-pick milestones where maintainers manually selected the most important bugfixes. Cross-referencing gives a "pre-approved" shortlist (~50-80 PRs).

### B. Scope by RHAI deployment (low effort, biggest reduction)

Answering these questions could eliminate 50%+ of candidates:
- **Which models?** Skip fixes for models RHAI doesn't ship
- **Quantization methods?** FP8, GPTQ, AWQ, NVFP4 each have dedicated fixes
- **LoRA / speculative decoding / tool calling / disaggregated serving?** Skip if unused

### C. Severity signals from GitHub (medium effort)

PRs fixing issues with many reactions/comments are higher-impact. Keywords like "Critical" or "regression" signal severity.

### D. Group related PRs (low effort, review efficiency)

Some PRs are "fix the fix" chains. Grouping by file overlap lets reviewers handle related fixes together.
