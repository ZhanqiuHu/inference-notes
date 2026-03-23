# vLLM v0.13.0 Backport Triage Tool

Identifies bugfix PRs merged into vLLM `main` after v0.13.0 that are candidates for backporting to the RHAI inference server (which ships on v0.13.0).

## Problem

vLLM v0.13.0 was released 2025-12-18. Since then, thousands of PRs have been merged, many fixing bugs that also exist in v0.13.0. We need to find and backport the relevant fixes without pulling in new features.

## How It Works

Single script, three phases, each saving an intermediate CSV:

### Phase 1 — Fetch

Queries GitHub for all merged PRs matching bugfix patterns (`label:bug`, `[Bugfix]`/`[BugFix]` in title). Uses 7-day date windows to avoid GitHub's 1,000-result API limit.

### Phase 2 — Classify → `phase2_all_prs_v0.13.0.csv`

Quick-classifies each PR:

| Type | Criteria | Action |
|------|----------|--------|
| `runtime_bug` | Has `bug` label, or title matches `[Bugfix]`/`[BugFix]`/`[Fix]` | Proceed to Phase 3 |
| `unclear` | Has "fix" in title but no clear prefix | Proceed to Phase 3 |
| `not_bugfix` | Title starts with `[Feature]`, `[Perf]`, `[CI]`, `[Refactor]`, etc. | Skip |
| `platform_specific` | Has `rocm`/`tpu`/`cpu`/`xpu` label | Skip (NVIDIA-only) |

### Phase 3 — File-Level Filter → `phase3_candidates_v0.13.0.csv`

For each `runtime_bug`/`unclear` PR:
- Fetches files, reviewers, merger, line counts (batched GraphQL)
- Checks if each file existed at the `v0.13.0` tag
- Detects subsystem (attention, engine, models, etc.)
- Assigns a priority score (0-100)

Priority scoring: `runtime_bug`=+40, file coverage up to +30, core subsystem +20, models/quant +10.

## Usage

```bash
cd /path/to/vllm
python find_backport_candidates.py
```

No flags. Runs all three phases sequentially. All output goes to stdout + `triage_v0.13.0.log`.

Prerequisites: `gh` (GitHub CLI, authenticated), `git`, Python 3.10+, inside a vllm repo with `v0.13.0` tag.

### Output Files

```
phase2_all_prs_v0.13.0.csv       — All PRs with classification
phase3_candidates_v0.13.0.csv    — Filtered candidates with priority, subsystem, reviewers
triage_v0.13.0.log               — Full readable log
```

### CSV Columns (Phase 3)

`priority`, `verdict`, `skip_reason`, `type`, `pr`, `title`, `author`, `merged_by`, `reviewers`, `subsystems`, `merged_at`, `labels`, `additions`, `deletions`, `files_in_release`, `files_total`, `files_existing`, `files_new`, `merge_sha`, `url`

## Example Run Results (2026-03-23)

- **646** PRs fetched
- **129** skipped early (not bugfix / platform-specific)
- **419** candidates after file-level filter
- **98** skipped (files don't exist in v0.13.0)
- Top subsystems: models (94), kernels (83), worker (45), quantization (31), api/frontend (30), attention (27)

## Next Steps

1. Review the Phase 3 CSV — filter by subsystem and priority
2. Scope to RHAI's actual deployment (which models, features, quantization methods)
3. Create a GitHub milestone for confirmed backports
4. Use vLLM's built-in cherry-pick script: `.buildkite/scripts/cherry-pick-from-milestone.sh`

## Claude Suggested Further Filtering Strategies

### A. Cross-reference with later patch releases (low effort, high signal)

vLLM's patch releases (v0.14.1, v0.15.1, v0.17.1) each have cherry-pick milestones where maintainers manually selected the most important bugfixes. Cross-referencing gives a "pre-approved" shortlist (~50-80 PRs).

### B. Scope by RHAI deployment (low effort, biggest reduction)

Answering these questions could eliminate 50%+ of candidates:
- **Which models?** Skip fixes for models RHAI doesn't ship
- **Quantization methods?** FP8, GPTQ, AWQ, NVFP4 each have dedicated fixes
- **LoRA / speculative decoding / tool calling / disaggregated serving?** Skip if unused

### C. Blame-level filter (medium effort, higher precision)

Check whether the specific **lines** each PR modifies existed in v0.13.0 (not just the file). Uses `git blame` at the merge commit's parent to determine when each modified line was introduced. This would reduce false positives from the file-level filter. Currently not implemented due to a line-number-drift issue that needs to be resolved.

### D. Severity signals from GitHub (medium effort)

PRs fixing issues with many reactions/comments are higher-impact. Keywords like "Critical" or "regression" signal severity.

### E. Group related PRs (low effort, review efficiency)

Some PRs are "fix the fix" chains. Grouping by file overlap lets reviewers handle related fixes together.
