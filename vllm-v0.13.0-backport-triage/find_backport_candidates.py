#!/usr/bin/env python3
"""
Find bugfix PRs merged after vLLM v0.13.0 that are candidates for backporting.

Four phases:
  1. Fetch all merged bugfix PRs after the release date
  2. Quick-classify each PR (runtime_bug / not_bugfix / platform / unclear)
  3. File-level filter: check which PRs touch files that existed in v0.13.0
  4. Blame-level filter: check which specific lines existed in v0.13.0

Intermediate CSVs are saved after phases 2, 3, and 4.

Prerequisites: gh (GitHub CLI, authenticated), git, Python 3.10+, inside vllm repo.
Usage: python find_backport_candidates.py
"""

import csv
import json
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

TAG = "v0.13.0"
REPO = "vllm-project/vllm"
OWNER, REPO_NAME = REPO.split("/")
WORKERS = 10
OUTPUT_DIR = Path(__file__).parent
OUTPUT_PHASE2 = OUTPUT_DIR / f"phase2_all_prs_{TAG}.csv"
OUTPUT_PHASE3 = OUTPUT_DIR / f"phase3_candidates_{TAG}.csv"
OUTPUT_PHASE4 = OUTPUT_DIR / f"phase4_blame_{TAG}.csv"
OUTPUT_LOG = OUTPUT_DIR / f"triage_{TAG}.log"

# ── Classification rules ─────────────────────────────────────────────

NON_RUNTIME_PATTERNS = [
    r"^tests/", r"^\.github/", r"^\.buildkite/", r"^docs/",
    r"^Dockerfile", r"^docker/", r"^\.pre-commit",
    r"^CONTRIBUTING", r"^README", r"^LICENSE",
    r"\.md$", r"^Makefile$", r"^\.gitignore$",
]

BUGFIX_TITLE_PATTERNS = [
    r"\[Bug\s*[Ff]ix\]", r"\[Bug\]", r"\[Fix\]", r"\[Bigfix\]",
]

NOT_BUGFIX_TITLE_PATTERNS = [
    r"^\[Feature\]", r"^\[Feat\]", r"^\[Perf\]", r"^\[RFC\]",
    r"^\[Refactor\]", r"^\[Misc\]", r"^\[Model\]", r"^\[Kernel",
    r"^\[Core\]", r"^\[Frontend\]", r"^\[Distributed\]",
    r"^\[CI\]", r"^\[Build\]", r"^\[Docs?\]", r"^\[Mypy\]",
    r"^\[CI/Build\]", r"^\[CI\]\[Build\]", r"^\[CI\]\[Bugfix\]",
    r"^\[Docker\]", r"^\[Release\]", r"^\[Test\]",
    r"^\[XPU\]", r"^\[ROCm\]", r"^\[TPU\]", r"^\[CPU\]",
]

PLATFORM_SKIP_TAGS = ["rocm", "tpu", "cpu", "intel-gpu", "xpu"]

SUBSYSTEM_RULES = [
    ("attention",     [r"vllm/.*/attention/"]),
    ("scheduler",     [r"vllm/v1/core/", r"vllm/core/"]),
    ("engine",        [r"vllm/v1/engine/", r"vllm/engine/"]),
    ("api/frontend",  [r"vllm/entrypoints/"]),
    ("models",        [r"vllm/model_executor/models/"]),
    ("quantization",  [r"vllm/model_executor/layers/quantization/"]),
    ("sampling",      [r"vllm/v1/sample/", r"vllm/model_executor/layers/sampler"]),
    ("distributed",   [r"vllm/distributed/"]),
    ("lora",          [r"vllm/lora/"]),
    ("spec_decode",   [r"vllm/spec_decode/", r"vllm/v1/spec_decode/"]),
    ("multimodal",    [r"vllm/multimodal/"]),
    ("config",        [r"vllm/config"]),
    ("worker",        [r"vllm/v1/worker/", r"vllm/worker/"]),
    ("kernels",       [r"vllm/model_executor/layers/", r"csrc/"]),
]

SEARCH_QUERIES = [
    # Maintainer-confirmed bugs (via label), regardless of title
    "label:bug",
    # vLLM convention title prefixes — one query each to avoid OR + date filter bugs
    '"[Bugfix]" in:title',
    '"[BugFix]" in:title',
    '"[Bug Fix]" in:title',
]


# ── Logging ──────────────────────────────────────────────────────────

class TeeLog:
    def __init__(self, log_path):
        self.log_file = open(log_path, "w")
        self.stdout = sys.stdout

    def write(self, msg):
        self.stdout.write(msg)
        self.log_file.write(msg)

    def flush(self):
        self.stdout.flush()
        self.log_file.flush()

    def close(self):
        self.log_file.close()


# ── Helpers ──────────────────────────────────────────────────────────

def run(cmd, check=True):
    r = subprocess.run(cmd, capture_output=True, text=True, check=check, timeout=120)
    return r.stdout.strip()


def file_exists_at_tag(filepath):
    try:
        subprocess.run(
            ["git", "cat-file", "-e", f"{TAG}:{filepath}"],
            capture_output=True, check=True, timeout=10,
        )
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return False


def detect_subsystems(files):
    subsystems = set()
    for f in files:
        for name, patterns in SUBSYSTEM_RULES:
            if any(re.search(p, f) for p in patterns):
                subsystems.add(name)
    return subsystems


def is_non_runtime_file(filepath):
    return any(re.search(p, filepath) for p in NON_RUNTIME_PATTERNS)


def classify_pr_type(title, labels):
    label_names = {l.lower() for l in labels}
    if label_names & set(PLATFORM_SKIP_TAGS):
        return "platform_specific"
    if any(re.search(p, title, re.IGNORECASE) for p in NOT_BUGFIX_TITLE_PATTERNS):
        return "not_bugfix"
    if "bug" in label_names:
        return "runtime_bug"
    if any(re.search(p, title) for p in BUGFIX_TITLE_PATTERNS):
        return "runtime_bug"
    if re.search(r"\bfix\b", title, re.IGNORECASE):
        return "unclear"
    return "not_bugfix"


def compute_priority(pr_type, files_in_release, files_total, subsystems):
    score = 0
    if pr_type == "runtime_bug":
        score += 40
    elif pr_type == "unclear":
        score += 20
    if files_total > 0:
        score += int(files_in_release / files_total * 30)
    core = {"engine", "scheduler", "api/frontend", "attention", "sampling", "worker"}
    if subsystems & core:
        score += 20
    if subsystems & {"models", "quantization"}:
        score += 10
    return min(score, 100)


PHASE3_COLUMNS = [
    "priority", "pr", "title", "subsystems", "approvers", "merged_by",
    "verdict", "skip_reason", "type", "merged_at", "author", "reviewers",
    "labels", "additions", "deletions", "files_in_release", "files_total",
    "files_existing", "files_new", "merge_sha", "url",
]

PHASE4_COLUMNS = [
    "blame_score", "blame_detail", "priority", "pr", "title", "subsystems",
    "approvers", "merged_by", "merged_at", "type", "verdict", "skip_reason",
    "author", "reviewers", "labels", "additions", "deletions",
    "files_in_release", "files_total", "files_existing", "files_new",
    "merge_sha", "url",
]


def write_csv(rows, path, columns=None):
    if not rows:
        return
    fieldnames = columns or list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"  Saved: {path}")


# ── GitHub API ───────────────────────────────────────────────────────

def _date_windows(start_date, end_date):
    from datetime import date, timedelta
    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)
    step = timedelta(days=7)
    while start < end:
        window_end = min(start + step, end)
        # Exclusive start (except first window), inclusive end
        yield start.isoformat(), window_end.isoformat()
        start = window_end + timedelta(days=1)


def fetch_all_bugfix_prs(tag_date):
    today = datetime.now().strftime("%Y-%m-%d")
    windows = list(_date_windows(tag_date, today))
    all_prs = []
    seen_nums = set()

    for qi, search_q in enumerate(SEARCH_QUERIES):
        print(f"\n  Search {qi+1}/{len(SEARCH_QUERIES)}: {search_q}")
        for i, (w_start, w_end) in enumerate(windows):
            search = f"merged:{w_start}..{w_end} {search_q}"
            raw = run([
                "gh", "pr", "list", "--repo", REPO, "--state", "merged",
                "--limit", "1000", "--search", search,
                "--json", "number,title,mergedAt,mergeCommit,labels,author,mergedBy",
            ])
            batch = json.loads(raw) if raw else []
            new = [p for p in batch if p["number"] not in seen_nums]
            for p in new:
                seen_nums.add(p["number"])
            all_prs.extend(new)
            status = f"    {w_start}..{w_end}: {len(batch)} ({len(new)} new)"
            if len(batch) >= 1000:
                status += " ⚠️ HIT LIMIT"
            print(status)

    all_prs.sort(key=lambda p: p.get("mergedAt", ""))
    return all_prs


def _graphql_batch_details(pr_numbers):
    fragments = []
    for num in pr_numbers:
        fragments.append(f"""
    pr_{num}: pullRequest(number: {num}) {{
      additions deletions
      mergedBy {{ login }}
      files(first: 100) {{ nodes {{ path }} }}
      reviews(first: 50) {{ nodes {{ author {{ login }} state }} }}
    }}""")
    query = f"""{{ repository(owner: "{OWNER}", name: "{REPO_NAME}") {{ {"".join(fragments)} }} }}"""
    raw = run(["gh", "api", "graphql", "-f", f"query={query}"], check=False)
    if not raw:
        return {}
    data = json.loads(raw).get("data", {}).get("repository", {})
    results = {}
    for num in pr_numbers:
        pr_data = data.get(f"pr_{num}")
        if not pr_data:
            results[num] = {"files": [], "reviewers": [], "approvers": [],
                            "merged_by": "", "additions": 0, "deletions": 0}
            continue
        files = [n["path"] for n in (pr_data.get("files", {}).get("nodes") or [])]
        seen_rev, seen_app = set(), set()
        reviewers, approvers = [], []
        for r in (pr_data.get("reviews", {}).get("nodes") or []):
            login = (r.get("author") or {}).get("login", "")
            if not login:
                continue
            if login not in seen_rev:
                seen_rev.add(login)
                reviewers.append(login)
            if r.get("state") == "APPROVED" and login not in seen_app:
                seen_app.add(login)
                approvers.append(login)
        results[num] = {
            "files": files, "reviewers": reviewers, "approvers": approvers,
            "merged_by": (pr_data.get("mergedBy") or {}).get("login", ""),
            "additions": pr_data.get("additions", 0),
            "deletions": pr_data.get("deletions", 0),
        }
    return results


def fetch_all_pr_details(pr_numbers, batch_size=30):
    all_details = {}
    batches = [pr_numbers[i:i + batch_size] for i in range(0, len(pr_numbers), batch_size)]
    for i, batch in enumerate(batches):
        print(f"  GraphQL batch {i+1}/{len(batches)} ({len(batch)} PRs)...")
        all_details.update(_graphql_batch_details(batch))
    return all_details


# ── Phase 1: Fetch ───────────────────────────────────────────────────

def phase1_fetch(tag_date):
    print(f"{'=' * 70}")
    print(f"  PHASE 1: Fetch all merged bugfix PRs after {TAG} ({tag_date})")
    print(f"{'=' * 70}")

    prs = fetch_all_bugfix_prs(tag_date)
    print(f"\n  Total merged bugfix PRs after {TAG}: {len(prs)}\n")
    return prs


# ── Phase 2: Classify ────────────────────────────────────────────────

def phase2_classify(prs):
    print(f"{'=' * 70}")
    print(f"  PHASE 2: Quick classification ({len(prs)} PRs)")
    print(f"{'=' * 70}\n")

    counts = {"runtime_bug": 0, "not_bugfix": 0, "platform_specific": 0, "unclear": 0}
    rows = []

    for pr in prs:
        num = pr["number"]
        title = pr["title"]
        merged = (pr.get("mergedAt") or "")[:10]
        sha = (pr.get("mergeCommit") or {}).get("oid", "")
        label_names = [l.get("name", "") for l in (pr.get("labels") or [])]
        author = (pr.get("author") or {}).get("login", "")
        merged_by = (pr.get("mergedBy") or {}).get("login", "")

        pr_type = classify_pr_type(title, label_names)
        counts[pr_type] += 1

        marker = {"runtime_bug": "BUG", "not_bugfix": "-- ",
                   "platform_specific": "plt", "unclear": "?  "}[pr_type]
        print(f"  [{marker}] #{num:<7} {merged}  merged_by:@{merged_by:<16} {title[:50]}")

        skip_reasons = {
            "not_bugfix": "not a bugfix (Feature/Perf/CI/Refactor)",
            "platform_specific": "platform-specific (ROCm/TPU/CPU/XPU)",
        }
        rows.append({
            "pr": f"#{num}", "title": title, "type": pr_type,
            "skip_reason": skip_reasons.get(pr_type, ""),
            "author": author, "merged_by": merged_by,
            "merged_at": merged, "labels": ", ".join(label_names),
            "merge_sha": sha, "url": f"https://github.com/{REPO}/pull/{num}",
        })

    write_csv(rows, OUTPUT_PHASE2)

    rt, uc = counts['runtime_bug'], counts['unclear']
    print(f"\n  Classification:")
    print(f"    runtime_bug:       {rt:>4}  (confirmed bugfixes)")
    print(f"    unclear:           {uc:>4}  ('fix' in title, needs review)")
    print(f"    not_bugfix:        {counts['not_bugfix']:>4}  (Feature/Perf/CI/Refactor)")
    print(f"    platform_specific: {counts['platform_specific']:>4}  (not relevant to NVIDIA)")
    print(f"\n  Phase 3 will process: {rt + uc} PRs\n")

    return prs


# ── Phase 3: File-level filter ───────────────────────────────────────

def phase3_file_filter(prs):
    print(f"{'=' * 70}")
    print(f"  PHASE 3: File-level filter against {TAG}")
    print(f"{'=' * 70}\n")

    label_map = {}
    for pr in prs:
        label_map[pr["number"]] = [l.get("name", "") for l in (pr.get("labels") or [])]

    worth_checking = [
        pr for pr in prs
        if classify_pr_type(pr["title"], label_map[pr["number"]]) in ("runtime_bug", "unclear")
    ]
    skipped_early = len(prs) - len(worth_checking)
    print(f"  Skipping {skipped_early} non-bugfix PRs")
    print(f"  Checking {len(worth_checking)} PRs\n")

    pr_numbers = [pr["number"] for pr in worth_checking]
    print(f"  Fetching details via GraphQL...")
    all_details = fetch_all_pr_details(pr_numbers)

    print(f"\n  Checking file existence ({WORKERS} workers)...\n")
    file_results = {}
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        def _check(pr):
            d = all_details.get(pr["number"], {})
            files = d.get("files", [])
            return pr["number"], files, [f for f in files if file_exists_at_tag(f)], [f for f in files if not file_exists_at_tag(f)]
        futures = {pool.submit(_check, pr): pr for pr in worth_checking}
        for future in as_completed(futures):
            num, files, in_rel, new_f = future.result()
            file_results[num] = (files, in_rel, new_f)

    candidates = []
    skipped = []

    for pr in worth_checking:
        num = pr["number"]
        title = pr["title"]
        merged = (pr.get("mergedAt") or "")[:10]
        sha = (pr.get("mergeCommit") or {}).get("oid", "")
        labels = label_map[num]
        author = (pr.get("author") or {}).get("login", "")
        detail = all_details.get(num, {})
        files, in_release, new_files = file_results[num]
        runtime_only = [f for f in in_release if not is_non_runtime_file(f)]

        pr_type = classify_pr_type(title, labels)
        subsystems = detect_subsystems(files)
        priority = compute_priority(pr_type, len(in_release), len(files), subsystems)
        merged_by = detail.get("merged_by", "")
        reviewers = detail.get("reviewers", [])
        approvers = detail.get("approvers", [])

        if not files:
            verdict, skip_reason = "SKIP", "no files detected"
        elif not in_release:
            verdict, skip_reason = "SKIP", "all files are post-release"
        elif not runtime_only:
            verdict, skip_reason = "SKIP", "only touches tests/docs/CI files"
        else:
            verdict, skip_reason = "CANDIDATE", ""

        row = {
            "priority": priority, "verdict": verdict, "skip_reason": skip_reason,
            "type": pr_type, "pr": f"#{num}", "title": title,
            "author": author, "merged_by": merged_by,
            "approvers": "; ".join(approvers),
            "reviewers": "; ".join(reviewers),
            "subsystems": ", ".join(sorted(subsystems)),
            "merged_at": merged, "labels": ", ".join(labels),
            "additions": detail.get("additions", 0),
            "deletions": detail.get("deletions", 0),
            "files_in_release": len(in_release), "files_total": len(files),
            "files_existing": " | ".join(in_release),
            "files_new": " | ".join(new_files),
            "merge_sha": sha, "url": f"https://github.com/{REPO}/pull/{num}",
        }

        if verdict == "CANDIDATE":
            candidates.append(row)
        else:
            skipped.append(row)

    candidates.sort(key=lambda r: -r["priority"])
    write_csv(candidates + skipped, OUTPUT_PHASE3, PHASE3_COLUMNS)

    # Summary
    sub_counts: dict[str, int] = {}
    for r in candidates:
        for s in r["subsystems"].split(", "):
            if s.strip():
                sub_counts[s.strip()] = sub_counts.get(s.strip(), 0) + 1

    print(f"\n  Phase 3 results: {len(candidates)} candidates, {len(skipped)} skipped")
    if sub_counts:
        print(f"\n  Subsystem breakdown:")
        for name, count in sorted(sub_counts.items(), key=lambda x: -x[1]):
            print(f"    {name:<20} {count:>4}")
    print()

    return candidates


# ── Blame helpers ────────────────────────────────────────────────────

_TAG_EPOCH = None  # lazily resolved to int timestamp of TAG

def _get_tag_epoch():
    global _TAG_EPOCH
    if _TAG_EPOCH is None:
        ts = run(["git", "log", "--format=%ct", TAG, "-1"])
        _TAG_EPOCH = int(ts)
    return _TAG_EPOCH


def _get_diff_line_map(merge_sha):
    """Parse diff of a merge commit → {filepath: [line_numbers_in_parent]}."""
    try:
        diff = run(["git", "diff", f"{merge_sha}~1..{merge_sha}", "--unified=0"], check=False)
    except Exception:
        return {}
    if not diff:
        return {}
    result = {}
    current_file = None
    for line in diff.split("\n"):
        if line.startswith("--- a/"):
            current_file = line[6:]
        elif line.startswith("+++ b/"):
            current_file = line[6:]
        elif line.startswith("@@ ") and current_file:
            match = re.match(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", line)
            if match:
                old_start = int(match.group(1))
                old_count = int(match.group(2) or 1)
                if current_file not in result:
                    result[current_file] = []
                result[current_file].extend(range(old_start, old_start + old_count))
    return result


def _blame_at_parent(merge_sha, filepath, line_numbers):
    """
    Blame at merge_sha~1 (the parent) for the given lines.
    For each line, check if the commit that introduced it predates TAG.
    Returns (matched_count, total_count).
    """
    if not line_numbers:
        return 0, 0
    tag_epoch = _get_tag_epoch()
    try:
        blame_raw = run(
            ["git", "blame", "--porcelain", f"{merge_sha}~1", "--", filepath],
            check=True,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return 0, len(line_numbers)

    line_set = set(line_numbers)
    commit_for_line = {}
    commit_time = {}
    current_sha = None
    current_orig_line = None

    for bline in blame_raw.split("\n"):
        parts = bline.split()
        if (len(parts) >= 3 and len(parts[0]) == 40
                and all(c in "0123456789abcdef" for c in parts[0])):
            current_sha = parts[0]
            current_orig_line = int(parts[2])
            if current_orig_line in line_set:
                commit_for_line[current_orig_line] = current_sha
        elif bline.startswith("committer-time ") and current_sha:
            commit_time[current_sha] = int(bline.split()[1])

    matched = 0
    for ln in line_numbers:
        sha = commit_for_line.get(ln)
        if sha and commit_time.get(sha, float("inf")) <= tag_epoch:
            matched += 1
    return matched, len(line_numbers)


def compute_blame_score(merge_sha):
    """
    For a merge commit, compute what fraction of modified lines
    were introduced on or before TAG (i.e., exist in v0.13.0).
    """
    diff_map = _get_diff_line_map(merge_sha)
    if not diff_map:
        return 0.0, "no diff"

    total_matched = 0
    total_lines = 0
    for filepath, lines in diff_map.items():
        if not file_exists_at_tag(filepath):
            continue
        matched, count = _blame_at_parent(merge_sha, filepath, lines)
        total_matched += matched
        total_lines += count

    if total_lines == 0:
        return 0.0, "no blameable lines"
    score = total_matched / total_lines
    return score, f"{total_matched}/{total_lines} lines pre-{TAG}"


# ── Phase 4: Blame-level filter ──────────────────────────────────────

def phase4_blame(candidates):
    print(f"{'=' * 70}")
    print(f"  PHASE 4: Blame-level filter — line-level precision")
    print(f"  Checking {len(candidates)} candidates ({WORKERS} workers)")
    print(f"{'=' * 70}\n")

    results = {}
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        def _blame(row):
            sha = row.get("merge_sha", "")
            if not sha:
                return row["pr"], 0.0, "no merge SHA"
            score, detail = compute_blame_score(sha)
            return row["pr"], score, detail

        futures = {pool.submit(_blame, row): row for row in candidates}
        done = 0
        for future in as_completed(futures):
            pr, score, detail = future.result()
            results[pr] = (score, detail)
            done += 1
            if done % 50 == 0:
                print(f"  Progress: {done}/{len(candidates)}")

    print(f"  Done: {len(results)} PRs analyzed\n")

    for row in candidates:
        score, detail = results.get(row["pr"], (0.0, "not analyzed"))
        row["blame_score"] = f"{score:.0%}"
        row["blame_detail"] = detail

    candidates.sort(key=lambda r: (-float(r["blame_score"].rstrip("%")) / 100, -r["priority"]))
    write_csv(candidates, OUTPUT_PHASE4, PHASE4_COLUMNS)

    high = [r for r in candidates if float(r["blame_score"].rstrip("%")) >= 80]
    med = [r for r in candidates if 40 <= float(r["blame_score"].rstrip("%")) < 80]
    low = [r for r in candidates if float(r["blame_score"].rstrip("%")) < 40]

    def _print_group(label, rows):
        print(f"\n  {'=' * 90}")
        print(f"  {label}")
        print(f"  {'=' * 90}")
        for r in rows:
            who = r.get("approvers") or r.get("merged_by") or ""
            if ";" in who:
                who = who.split(";")[0].strip()
            print(
                f"  {r['blame_score']:>4} [{r['priority']:>3}] {r['pr']:<8} "
                f"[{r['subsystems']:<20}] @{who:<14} "
                f"{r['title'][:40]}"
            )

    _print_group(f"HIGH CONFIDENCE ({len(high)} PRs, >= 80% lines in {TAG})", high)
    _print_group(f"MEDIUM CONFIDENCE ({len(med)} PRs, 40-79%)", med)
    _print_group(f"LOW CONFIDENCE ({len(low)} PRs, < 40% — likely post-release code)", low)

    print(f"\n{'=' * 70}")
    print(f"  FINAL SUMMARY")
    print(f"  Candidates analyzed: {len(candidates)}")
    print(f"    High confidence:   {len(high)}  (bug exists in {TAG})")
    print(f"    Medium:            {len(med)}  (partially applicable)")
    print(f"    Low:               {len(low)}  (likely post-release)")
    print(f"{'=' * 70}")


# ── Main ─────────────────────────────────────────────────────────────

def main():
    tee = TeeLog(OUTPUT_LOG)
    sys.stdout = tee

    tag_date = run(["git", "log", "--format=%ai", TAG, "-1"]).split(" ")[0]
    print(f"Release {TAG} date: {tag_date}")
    print(f"Run time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Log file: {OUTPUT_LOG}\n")

    prs = phase1_fetch(tag_date)
    prs = phase2_classify(prs)
    candidates = phase3_file_filter(prs)
    phase4_blame(candidates)

    tee.close()
    sys.stdout = tee.stdout
    print(f"\nFull output saved to: {OUTPUT_LOG}")
    print(f"CSVs: {OUTPUT_PHASE2.name}, {OUTPUT_PHASE3.name}, {OUTPUT_PHASE4.name}")


if __name__ == "__main__":
    main()
