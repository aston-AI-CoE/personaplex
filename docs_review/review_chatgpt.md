# Review of Uploaded Reports Against 01 as Ground Truth

## Summary

Using **01_session-cold-start-timing-report.md** as the baseline source of truth, the overall assessment is:

- **01** is the strongest and most reliable report because it contains direct timing measurements.
- **02** is mostly a reasonable explainer derived from 01, but it includes several implementation and performance claims that are not established by 01.
- **03** is a useful speculative latency analysis, but most of its detailed numbers are not verified by 01.
- **04** is best treated as a proposed implementation plan, not a verified findings report.

The main pattern across 02–04 is a shift from **measured facts** to **inference and speculation**. That is acceptable as long as the documents are labeled clearly, but several claims are currently written too strongly.

---

## What 01 Actually Establishes

These are the most defensible claims from 01:

1. **Cold start is about 7.6 seconds** on the tested setup.
2. The dominant contributor is **135 sequential LM steps** across the prompt-building sequence.
3. The first few LM steps are much slower, after which steady-state settles around **~56 ms/step**.
4. File loading, tokenization, locking, reset, and post-reset work are negligible compared with prefill.
5. A **snapshot/restore approach** is suggested as a mitigation, but in 01 this remains a **proposal/conclusion**, not a demonstrated implementation result.

So 01 is strong on **timing decomposition**, but weaker on exact mechanism claims and exact restore-performance claims.

---

## Review of 02_kv-cache-prefill-and-snapshot-explainer.md

### Supported by 01

The following points are directionally supported:

- KV cache prefill is the main cause of the ~7.6s startup delay.
- Prefill consists of 135 sequential model steps.
- Reusing a prefilled state for the same prompt/config is a reasonable explanation for why snapshotting could help.

### Not Proven by 01

The following claims are **not established by 01** and should be softened or explicitly labeled as estimates/hypotheses:

- KV cache size being **~1.5 GB**.
- The exact count of **3,000 positions / 4 minutes / 12.5 Hz**.
- “Restore is just memcpy + set offset.”
- Snapshot creation being ~5 ms.
- Restore being ~5 ms.
- Total cold start becoming ~50 ms.
- The claimed **~150x speedup**.
- Detailed VRAM layout claims.
- The use of a dedicated GPU snapshot buffer as an established implementation detail.

### Likely Incorrect or Inconsistent

- The voice prompt filename appears inconsistent:
  - 01 uses **NATF0.pt**
  - 02 uses **NATM0.pt**

This should be corrected.

### Unverified Internal Mechanism Claim

02 also says that after restoring, the first forward pass runs without a CUDA graph and recaptures automatically on the second call. That is **not verified by 01** and is also in tension with 03, which argues the opposite.

### Verdict on 02

**Recommended label:** explanatory/derived doc.

Keep the conceptual explanation, but soften exact implementation and performance numbers unless benchmarked elsewhere.

---

## Review of 03_ttft-pipeline-analysis.md

### Supported by 01 in Direction

The following high-level idea is reasonable:

- If the 7.6s prefill is removed, TTFT would be dominated by the remaining audio/network/streaming path instead of model prefill.

That is a fair directional consequence of 01.

### Not Verifiable from 01

Most of 03 depends on information not established by 01, including:

- Client capture timing.
- Opus framing details.
- Network estimates.
- Mimi encode/decode timings.
- Playback buffer timing.
- Same-AZ vs cross-region TTFT breakdowns.
- The estimated **~310 ms to first audio**.

These may be reasonable engineering estimates, but they are not verified by 01.

### Important Contradiction with 01

03 argues that 01 is wrong about CUDA graph recreation and says graphs persist across sessions because `_LMGenState.reset()` only resets offsets/state markers.

This may be true, but with the provided evidence set it is still **not independently verified**. The actual code proof is not present in 01 itself.

Therefore:

- 01’s mechanism claim about CUDA graph recreation should not be treated as fully proven.
- 03’s rebuttal should also not be treated as fully proven without the actual referenced code or benchmark evidence.

### Verdict on 03

**Recommended label:** speculative latency analysis after hypothetical snapshot/prefix-cache.

It is useful for planning, but not verified content.

---

## Review of 04_kv-cache-snapshot-restore-plan.md

### Directionally Supported by 01

The core idea is reasonable:

- Snapshot/restore is a plausible way to remove the dominant prefill bottleneck identified in 01.

### Not Verifiable from 01

Most of 04 goes beyond 01 and should be treated as proposal/design rather than fact:

- Exact code locations and function names.
- Exact tensor shapes and memory calculations.
- Storage-tier restore-speed table.
- CPU RAM recommendations.
- Detailed server changes and control flow.
- Deep-copy strategy.
- Config-hash matching logic.
- Depformer exclusion details.
- Prefix-cache variant timings.

### Claim That Should Be Softened

04 says the 7.6s cold start is **entirely sequential LM forward passes**.

That is directionally very close, but “entirely” is a little too absolute. A better phrasing would be:

> The 7.6s cold start is **almost entirely dominated by sequential LM prefill steps**.

That wording is more faithful to 01.

### Verdict on 04

**Recommended label:** implementation proposal based on 01.

It should not be presented as a verification report.

---

## Biggest Issues to Fix Across the Set

### 1. Clearly Separate Measured Facts from Derived Conclusions

Recommended framing:

- **01** = measured timing report / baseline fact source
- **02** = conceptual explainer derived from 01
- **03** = estimated latency analysis under hypothetical optimization
- **04** = implementation/design proposal

### 2. Fix the Prompt Filename Mismatch

Change **NATM0.pt** to **NATF0.pt** if 01 is the baseline truth.

### 3. Resolve the CUDA Graph Claim Carefully

01 and 03 make conflicting claims about whether CUDA graphs are recreated every session.

With the provided evidence set, neither claim should be stated as fully verified unless accompanied by direct code citation or measurement evidence.

### 4. Downgrade Exact Restore Timings to Estimates

All exact restore numbers in 02 and 04 should be labeled as:

- estimate
- projection
- expected
- hypothetical benchmark target

unless actual timing data exists.

### 5. Downgrade 03’s TTFT Numbers to Pipeline Estimates

The numerical TTFT values in 03 should not be framed as measured facts unless end-to-end profiling exists.

---

## Recommended Final Assessment

### 01_session-cold-start-timing-report.md

**Assessment:** reliable baseline report.

This is the strongest document because it contains direct timing measurements and a clear decomposition of where the startup latency comes from.

### 02_kv-cache-prefill-and-snapshot-explainer.md

**Assessment:** good explainer, but partially speculative.

The conceptual narrative is useful and mostly aligned with 01, but several exact implementation and performance claims are not grounded by 01.

### 03_ttft-pipeline-analysis.md

**Assessment:** useful speculative model, not verified report.

Helpful for planning and prioritization, but most of the numeric pipeline details are not established by 01.

### 04_kv-cache-snapshot-restore-plan.md

**Assessment:** design proposal, not verified findings.

Valuable as an implementation direction, but should be explicitly labeled as a plan derived from 01 rather than a validated result.

---

## Bottom Line

The cleanest summary is:

- **01 tells you what actually happened.**
- **02 explains it, with some over-precise claims that should be softened.**
- **03 estimates what latency might look like after optimization.**
- **04 proposes how to build the optimization.**

That framing keeps the reports internally consistent and prevents speculative conclusions from being mistaken for measured truth.
