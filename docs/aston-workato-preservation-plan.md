# Preserve `aston-workato` Branch and Extraction Plan

Date recorded: 2026-07-13

## TL;DR

Keep `origin/aston-workato` intact. It is exactly 2 commits ahead and 0 commits behind `origin/main`; no branch rewrite, force-push, deletion, merge, cherry-pick, or archive action was performed while recording this plan.

Recommendation: retain the branch as the preservation source of truth now, then extract by cherry-picking both unique commits together into a review branch only if the implementation is needed in a maintained branch. Do not squash or manually copy the changes during extraction, because preserving commit identity and bundled notices is the safest path.

## Branch State

Comparison command:

```bash
git rev-list --left-right --count origin/main...origin/aston-workato
```

Result:

| Branch side | Count |
| --- | ---: |
| `origin/main` only | 0 |
| `origin/aston-workato` only | 2 |

Refs at the time of inspection:

| Ref | Commit |
| --- | --- |
| `origin/main` | `3428dfd95309a7f3c84fd93259ded0f810d1ff91` |
| merge base | `3428dfd95309a7f3c84fd93259ded0f810d1ff91` |
| `origin/aston-workato` | `3e0bf178b338b870aceb10a76d23655ebd915406` |

Unique commits on `origin/aston-workato`, oldest first:

| Commit | Date | Author | Subject |
| --- | --- | --- | --- |
| `1eee135049f216d80e2c07c140709d8aa252943f` | 2026-03-16 | `aston-AI-CoE` | Add cold start analysis docs, timing instrumentation, and KV cache snapshot plan |
| `3e0bf178b338b870aceb10a76d23655ebd915406` | 2026-03-17 | `aston-AI-CoE` | Eliminate session cold start via KV cache snapshot/restore |

## Changed Files

Overall diff from `origin/main` to `origin/aston-workato`: 18 files changed, 2364 insertions, 36 deletions.

| Path | Status | Additions | Deletions | Purpose |
| --- | --- | ---: | ---: | --- |
| `.gitignore` | Modified | 3 | 0 | Ignores generated `snapshot*/` directories. |
| `.python_bin` | Added | 1 | 0 | Records local Python launcher preference. |
| `client/src/audio-processor.ts` | Modified | 13 | 15 | Adjusts playback buffering around initial playback and underrun recovery. |
| `client/src/pages/Conversation/hooks/useModelParams.ts` | Modified | 1 | 1 | Changes the default voice prompt. |
| `docs/00_executive-summary-cold-start-elimination.md` | Added | 293 | 0 | Executive summary of cold-start elimination, measured results, and remaining work. |
| `docs/01_session-cold-start-timing-report.md` | Added | 130 | 0 | Timing report for session startup and prompt prefill. |
| `docs/02_kv-cache-prefill-and-snapshot-explainer.md` | Added | 154 | 0 | Explains KV cache prefill, snapshotting, restore flow, and limitations. |
| `docs/03_kv-cache-snapshot-restore-plan.md` | Added | 149 | 0 | Snapshot/restore implementation plan, state captured, and gotchas. |
| `docs/04_ttft-pipeline-analysis.md` | Added | 157 | 0 | Time-to-first-audio pipeline analysis and optimization opportunities. |
| `docs/05_kv-cache-snapshot-implementation-results.md` | Added | 213 | 0 | Implementation results, snapshot structure, memory budget, and findings. |
| `docs/06_pin-memory-audio-blitzing-root-cause.md` | Added | 102 | 0 | Root-cause analysis for audio blitzing related to pinned memory. |
| `docs/assets/hybrid-system-prompt-architecture.png` | Added | binary | binary | Architecture image used by the new docs. |
| `docs/build-prompt-kv-cache-snapshot.md` | Added | 413 | 0 | Build prompt and working notes for snapshot implementation. |
| `docs_review/review_chatgpt.md` | Added | 232 | 0 | External review notes. |
| `docs_review/review_claude.md` | Added | 127 | 0 | External review notes. |
| `moshi/moshi/generate_snapshot.py` | Added | 174 | 0 | Offline snapshot generation entry point. |
| `moshi/moshi/models/lm.py` | Modified | 56 | 3 | Adds timing instrumentation around prefill and model step behavior. |
| `moshi/moshi/server.py` | Modified | 146 | 17 | Adds snapshot loading, matching, restore path, timing logs, and scenario metadata endpoint. |

## KV-Cache and Cold-Start Learnings

Design-level findings from the branch documentation and file metadata:

1. The cold-start bottleneck is dominated by repeated prompt prefill before the first usable audio, not by steady-state token generation.
2. The proposed optimization precomputes the model streaming state for a `(voice prompt, text prompt)` pair and restores that KV-cache snapshot per session instead of replaying full prefill.
3. Snapshot restore must preserve enough streaming state to resume generation consistently while still applying runtime generation parameters after restore.
4. CPU RAM is the preferred storage tier for multiple ready-to-restore snapshots; disk-only restore is treated as too slow for low-latency session start.
5. The branch records that CUDA graphs can persist across restore, which matters because recapturing graphs would erase much of the latency win.
6. A separate audio-startup issue was traced to pinned CPU memory. The design learning is that transfer speed is not the only constraint; audio pacing and buffer behavior can dominate perceived quality.
7. Client-side playback buffering is part of the end-to-end time-to-first-audio budget. Server cold-start improvements still need browser playback validation.

These are extracted as design conclusions only. The source branch remains the canonical place for implementation details.

## `client/.env.local` Inspection

`client/.env.local` is tracked on both `origin/main` and `origin/aston-workato` with the same blob. It is unchanged by the two unique branch commits.

Redacted inspection found two Vite client configuration entries. The values are non-placeholder values, but the keys are client URL/path configuration rather than credential-shaped secrets. Because this file is already tracked on `main` and is not changed by `aston-workato`, there is no branch-specific credential remediation to perform as part of this preservation task.

Do not publish or quote the values in issues, pull requests, logs, or extraction notes. If repository policy later decides tracked Vite environment files are unacceptable, handle that as a separate main-branch hygiene change with secret scanning and history guidance.

## License Notice Preservation

The current repository includes these license notice files:

| Path |
| --- |
| `LICENSE-MIT` |
| `client/LICENSE` |
| `moshi/LICENSE.audiocraft` |
| `moshi/LICENSE.moshi` |

Any extraction must preserve these notices with the relevant copied or cherry-picked code. If snapshot artifacts, generated docs, or derived implementation files are moved to another repository, include the applicable root and subproject notices in the same change set and keep third-party attribution intact.

## Extraction Recommendation

Recommended path: retain `origin/aston-workato` now, then cherry-pick both unique commits together if the work needs to land elsewhere.

Why not merge immediately:

| Option | Tradeoff |
| --- | --- |
| Merge `aston-workato` to `main` | Preserves all work quickly, but brings experimental server/client behavior into `main` before validation. |
| Cherry-pick both commits to a review branch | Preserves commit authorship and reviewability while allowing validation before mainline changes. |
| Retain branch only | Safest immediate archive-prevention step; no runtime behavior changes. Requires a later extraction action before deleting or archiving the source. |

Use this sequence for extraction:

1. Confirm `origin/aston-workato` still resolves to `3e0bf178b338b870aceb10a76d23655ebd915406` or intentionally update this plan with the new branch head.
2. Create a new branch from the target repository default branch.
3. Cherry-pick `1eee135049f216d80e2c07c140709d8aa252943f` and `3e0bf178b338b870aceb10a76d23655ebd915406` in order.
4. Keep all license notice files listed above with the extracted code.
5. Do not copy `client/.env.local` values into any issue, PR description, fixture, generated log, or external repo.
6. Validate with the commands below and at least one manual browser/server startup check if runtime dependencies and model access are available.
7. Only after validation, decide whether to merge the extraction branch or keep `aston-workato` as an archival branch.

Suggested validation:

```bash
git rev-list --left-right --count origin/main...origin/aston-workato
git log --reverse --oneline origin/main..origin/aston-workato
cd client && npm run build
cd ../moshi && python -m compileall moshi
python -m moshi.server --help
```

Runtime validation, when model credentials and GPU are available:

```bash
python -m moshi.generate_snapshot --help
python -m moshi.server --snapshot-dir <snapshot-directory>
```

Rollback:

1. If extraction was only cherry-picked to a review branch, close the PR and delete only that review branch.
2. If extraction was merged and must be reverted, revert the merge commit or revert the two cherry-picked commits in reverse order.
3. Do not force-push or delete `origin/aston-workato` until a separate archive decision explicitly approves it.

## What Would Falsify This Plan

This recommendation should be revisited if any of the following are true:

1. `origin/aston-workato` gains additional unique commits.
2. `client/.env.local` is changed on the branch or is later classified as a real credential leak by secret scanning.
3. License files or third-party notices change upstream.
4. Runtime validation shows the snapshot restore path is unsafe, nondeterministic, or incompatible with supported deployment hardware.
