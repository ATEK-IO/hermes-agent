# Fork CI — what runs here, and what an author must cite instead

This repository is the ATEK-IO fork of `NousResearch/hermes-agent`. Upstream CI
cannot run on it. Before this workflow existed, a PR on this fork could show
"2 runs, both success" while **zero test code executed** (ATE-7435).

## Why upstream's CI cannot run here

| workflow | state on this fork | effect |
|---|---|---|
| `ci.yaml` (orchestrator: 24 jobs — tests, lint, js-tests, rust, e2e) | `disabled_manually` | never runs |
| `docker.yml` | active | `build` requires `github.repository == 'NousResearch/hermes-agent'`; every real job is `skipped`, the run concludes `success` |
| `nix.yml` | active | classifier runs; `nix flake check` correctly skips a Python-only diff, but **queues forever** when the classifier says the flake is affected (any `.github/**` change does) |

`ci.yaml` cannot simply be re-enabled: its Python, JS, Rust and desktop lanes run
on `ubuntu-latest-32-core` / `windows-latest-32-core`, GitHub *larger-runner*
labels this account does not have (no self-hosted runners either). The only
`ci.yaml` run this fork ever produced (2026-10-16, run 38944034361) shows it:
both `Run tests` slices and the Windows job ended `cancelled`, never started.

## What `Fork CI` (`fork-ci.yml`) runs

Standard `ubuntu-latest` runners only.

Known leftover: in 35 runs on this fork, no job with a `*-32-core` label has ever
executed a step. 17 were skipped by their `if:`; the 7 that reached `cancelled`
carry 0 steps each — on the only full `ci.yaml` run (38944034361) both `Run
tests` slices sat exactly 24h and were cancelled with 0 steps, which is the
queue timeout, not a result. So whenever an upstream workflow decides that lane
applies (for example `nix flake check` on any `.github/**` change), that check
sits `queued` and the commit status stays `pending`. Cancel that run; it will not
start. It is not evidence of anything, passing or failing.

| job | what it does |
|---|---|
| `Scope (select affected tests)` | classifies the diff, selects the tests it can affect |
| `Python tests (affected)` | executes that selection through `scripts/run_tests.sh` |
| `Lint (ruff + ty)` | upstream's blocking lint lane, unmodified |
| `NOT TESTED: surface this fork cannot run` | **fails** when the diff touches a surface no lane here can execute |
| `Fork CI gate` | the one check to cite |

The selection is diff-scoped, not the full ~5.5k-file suite: test files named
after a changed module first, then test files that reference it, capped at 60.
When the cap binds, the scope job prints `SAMPLED … -> N of M` — the run proves N
files, not M.

## What an author cites, per surface

| the diff touches | cite |
|---|---|
| Python under the package, `tests/**` | the `Fork CI gate` check. It executed real tests at the PR head SHA |
| docs, `*.md`, `.github/**` | the `Fork CI gate` check; no test lane applies |
| `apps/**`, `web/**`, `src-tauri/**`, any `*.ts`/`*.tsx`/`*.js` | the `NOT TESTED` job fails. Cite a **local** run of the affected command at the head SHA, pasted into the PR description |
| `nix/**`, `flake.nix`, `flake.lock` | same: `NOT TESTED`. Cite `nix flake check` run locally at the head SHA |
| `Dockerfile`, `docker/**` | same: `NOT TESTED`. Cite a local `docker build` + the test command used |
| `Cargo.toml`, `*.rs` | same: `NOT TESTED`. Cite `cargo test` locally |
| a Python file no test references | the `NOT TESTED` job fails. Either add the test, or state in the PR why the file needs no test |

## The evidence rule for reviewers

A green `Fork CI gate` means: **the selected tests executed at this head SHA and
passed, and the diff touched nothing outside the covered surfaces.** It does not
mean the whole suite passed, and it is not evidence for a file listed under
`SAMPLED` beyond the files that ran.

A red `NOT TESTED: surface this fork cannot run` is **not** a code verdict. It
means no executed check exists for part of the diff. A reviewer may still merge,
but the local evidence must be in the PR description, naming the command, the
head SHA and the result.

Never cite upstream's `ci.yaml`, `docker.yml` or `nix.yml` "success" as test
evidence on this fork: none of them executes a test here.
