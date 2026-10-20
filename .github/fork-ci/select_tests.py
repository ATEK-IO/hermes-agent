#!/usr/bin/env python3
"""Select the tests a fork CI diff can affect.

The ATEK-IO fork of hermes-agent cannot run upstream's full matrix: the suite is
~5.5k test files sized for 32-core larger runners, which this account does not
have. So this fork runs the tests the diff can actually break, and says out loud
when a diff touches a surface no lane here can execute.

Outputs (written to $GITHUB_OUTPUT):
  tests    space-separated test paths (empty when nothing test-bearing changed)
  count    number of selected files
  untested comma-separated list of surfaces/files with no executing check
  summary  human-readable classification, one line per changed file

Design rule: never widen a selection silently and never report "no tests" as
"passed". A Python file with no matching test lands in `untested`, which fails
the `untested` job. A module imported by more test files than the budget allows
is reported as SAMPLED with both numbers.

Allocation: every source file's *name* matches are selected first (the strongest
signal: a test file named after the module), then import matches fill the
remaining budget round-robin across files, so one wide core module cannot starve
the others.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

# A selection larger than this stops being a diff-scoped run.
MAX_TESTS = 60
MAX_CONTENT_MATCHES = 40

# Every subprocess is bounded: a hang in git or rg must fail the job, not stall it.
CMD_TIMEOUT_S = 300

# Surfaces that upstream covers with lanes this fork cannot run.
UNCOVERED = [
    (re.compile(r"^(apps/|web/|src-tauri/)"), "desktop/frontend"),
    (re.compile(r"\.(ts|tsx|js|jsx|mjs|cjs)$"), "frontend (JS/TS)"),
    (re.compile(r"(^|/)(package\.json|package-lock\.json|tsconfig[^/]*\.json)$"), "frontend (JS/TS)"),
    (re.compile(r"^(nix/|flake\.(nix|lock)$)"), "nix flake"),
    (re.compile(r"^(Dockerfile|docker/|\.dockerignore)"), "docker image"),
    (re.compile(r"(^|/)(Cargo\.toml|Cargo\.lock)$|\.rs$"), "rust"),
    (re.compile(r"^(packaging|installers?)/"), "packaging"),
]

# Changed paths that need no executing check and are not a gap: prose, CI
# configuration, and the repository's own metadata.
IGNORED = [
    re.compile(r"\.(md|mdx|txt|png|jpg|jpeg|gif|svg|ico)$"),
    re.compile(r"^docs/"),
    re.compile(r"^(LICENSE|NOTICE|\.gitignore|\.editorconfig)"),
]

# Paths that ARE the CI itself, or its tooling. Nothing here is application
# code, so "no test references it" would be a false gap.
CI_CONFIG = (".github/", "scripts/ci/")

# Test trees this lane never runs: e2e drives real processes and takes far
# longer than a diff-scoped lane should.
SKIP_TEST_TREES = ("tests/e2e/", "tests/manual/", "tests/evals/")

# pending entry: (source path, name matches, import matches, total matches found)
Pending = tuple[str, list[str], list[str], int]


def sh(*args: str) -> str:
    return subprocess.run(
        args, check=True, capture_output=True, text=True, timeout=CMD_TIMEOUT_S
    ).stdout


# None = not probed yet, True/False = rg present. Probed by running it: the
# repository's own guard (tests/test_managed_runtime_resolution.py) rejects a
# bare PATH probe in Hermes-owned code, and this file is Hermes-owned.
_RG: bool | None = None


def have_rg() -> bool:
    global _RG
    if _RG is None:
        try:
            sh("rg", "--version")
            _RG = True
        except (OSError, subprocess.SubprocessError):
            _RG = False
    return _RG


def grep_test_files(needle: str) -> list[str]:
    """Test files containing `needle`, via rg when present and plain Python when not."""
    if have_rg():
        try:
            out = sh("rg", "-l", "--glob", "test_*.py", "--fixed-strings", needle, "tests")
            return [line.strip() for line in out.splitlines() if line.strip()]
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            return []
    hits: list[str] = []
    for path in Path("tests").rglob("test_*.py"):
        try:
            if needle in path.read_text(encoding="utf-8", errors="ignore"):
                hits.append(str(path))
        except OSError:
            continue
    return hits


def changed_files() -> list[str]:
    override = os.environ.get("FORK_CI_FILES_OVERRIDE", "").strip()
    if override:
        return [p for p in override.replace(",", "\n").split() if p]

    event = os.environ.get("EVENT_NAME", "pull_request")
    if event == "pull_request":
        base = os.environ["PR_BASE_SHA"].strip()
        rng = f"{base}...HEAD"
    else:
        before = os.environ.get("PUSH_BEFORE", "").strip()
        rng = f"{before}...HEAD" if before and set(before) != {"0"} else "HEAD~1...HEAD"
    out = sh("git", "diff", "--name-only", "--diff-filter=ACMRD", rng)
    return [line.strip() for line in out.splitlines() if line.strip()]


def is_test_file(path: str) -> bool:
    name = os.path.basename(path)
    return path.startswith("tests/") and (name.startswith("test_") or name.endswith("_test.py"))


def skipped_test_tree(path: str) -> bool:
    return path.startswith(SKIP_TEST_TREES)


def is_ci_config(path: str) -> bool:
    return path.startswith(CI_CONFIG)


def find_tests_for(path: str) -> tuple[list[str], list[str], int]:
    """Return (test files named after this module, capped import matches, total found)."""
    stem = re.sub(r"^_+", "", Path(path).stem)
    dotted = path[: -len(".py")].replace("/", ".")

    named = sorted(
        str(p) for p in Path("tests").rglob(f"*{stem}*.py")
        if p.name.startswith("test_") and not skipped_test_tree(str(p))
    )

    content: list[str] = []
    total = len(named)
    for needle in (dotted, Path(path).stem):
        if len(needle) < 4:
            continue
        seen_here = 0
        for line in grep_test_files(needle):
            line = line.strip()
            if not line or line in named or skipped_test_tree(line):
                continue
            seen_here += 1
            if line not in content:
                content.append(line)
        total += seen_here
        if len(content) >= MAX_CONTENT_MATCHES:
            break
    return named, content[:MAX_CONTENT_MATCHES], total


def kind_of(path: str) -> str:
    """Classify one changed path. Untestable surfaces return 'untested:<label>'."""
    if skipped_test_tree(path):
        return "skip"
    if is_test_file(path):
        return "test"
    if is_ci_config(path):
        return "ci"
    if path.endswith(".py"):
        return "python"
    for pattern, label in UNCOVERED:
        if pattern.search(path):
            return f"untested:{label} — no lane on this fork"
    if any(pattern.search(path) for pattern in IGNORED):
        return "ignore"
    return "other"


def collect(files: list[str]) -> tuple[list[str], list[str], list[str], list[Pending]]:
    """Split the diff into a test list, a gap list, a summary, and per-module work."""
    tests: list[str] = []
    untested: list[str] = []
    summary: list[str] = []
    pending: list[Pending] = []

    for path in files:
        kind = kind_of(path)
        if kind == "skip":
            summary.append(f"skip    {path}  (e2e/manual test tree — not run by this lane)")
        elif kind == "test":
            if path not in tests:
                tests.append(path)
            summary.append(f"test    {path}")
        elif kind == "ci":
            summary.append(f"ci      {path}  (CI configuration — no test lane applies)")
        elif kind == "python":
            named, content, total = find_tests_for(path)
            if not named and not content:
                untested.append(f"{path} (no test references it)")
                summary.append(f"NOTEST  {path}  (no test file found)")
            else:
                pending.append((path, named, content, total))
        elif kind.startswith("untested:"):
            label = kind.split(":", 1)[1]
            untested.append(f"{path} ({label})")
            summary.append(f"NOTEST  {path}  ({label})")
        elif kind == "ignore":
            summary.append(f"docs    {path}  (no executing check needed)")
        else:
            summary.append(f"other   {path}  (no test lane mapped — lint only)")
    return tests, untested, summary, pending


def allocate_named(pending: list[Pending], tests: list[str], summary: list[str]) -> None:
    """Pass 1: every module's name-matched tests, before any import match."""
    for path, named, _content, _total in pending:
        added = 0
        for test in named:
            if test in tests or len(tests) >= MAX_TESTS:
                continue
            tests.append(test)
            added += 1
        summary.append(f"python  {path}  -> {added} test file(s) named after it")


def allocate_imports(pending: list[Pending], tests: list[str]) -> None:
    """Pass 2: import matches, round-robin so one wide module cannot starve others."""
    cursor = {path: 0 for path, _n, _c, _t in pending}
    while len(tests) < MAX_TESTS:
        progressed = False
        for path, _named, content, _total in pending:
            if len(tests) >= MAX_TESTS:
                break
            i = cursor[path]
            while i < len(content) and content[i] in tests:
                i += 1
            if i < len(content):
                tests.append(content[i])
                i += 1
                progressed = True
            cursor[path] = i
        if not progressed:
            break


def report_samples(pending: list[Pending], tests: list[str], summary: list[str]) -> None:
    """Name every module whose tests did not all run, with both numbers."""
    for path, named, content, total in pending:
        discovered = len({t for t in named + content})
        total = max(total, discovered)
        got = len({t for t in named + content if t in tests})
        if got < total:
            summary.append(
                f"SAMPLED {path}  -> {got} of {total} referencing test file(s) run; "
                f"the rest do not run in this lane"
            )


def write_outputs(tests: list[str], untested: list[str], summary: list[str]) -> None:
    payload = {
        "tests": " ".join(tests),
        "count": str(len(tests)),
        "untested": ", ".join(untested),
        "summary": "\n".join(summary),
    }
    out_path = os.environ.get("GITHUB_OUTPUT")
    if not out_path:
        return
    with open(out_path, "a", encoding="utf-8") as fh:
        for key, value in payload.items():
            marker = f"__FORK_CI_{key.upper()}__"
            fh.write(f"{key}<<{marker}\n{value}\n{marker}\n")


def main() -> int:
    files = changed_files()
    tests, untested, summary, pending = collect(files)

    allocate_named(pending, tests, summary)
    allocate_imports(pending, tests)
    report_samples(pending, tests, summary)

    # Determinism: a stable order makes the CI log comparable across runs.
    tests = sorted(dict.fromkeys(tests))
    untested = list(dict.fromkeys(untested))

    print("\n".join(summary))
    print(f"\nselected {len(tests)} test file(s), {len(untested)} untested item(s)")
    write_outputs(tests, untested, summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
