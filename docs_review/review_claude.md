# Review: KV Cache / Cold Start Documentation Verification

**Reviewed by:** Claude
**Date:** 2026-03-13
**Files reviewed:** `01_session-cold-start-timing-report.md`, `02_kv-cache-prefill-and-snapshot-explainer.md`, `03_ttft-pipeline-analysis.md`, `04_kv-cache-snapshot-restore-plan.md`

---

## Overview

Doc 01 is treated as ground truth (empirical timing measurements on A10G). Docs 02–04 are progressively more speculative. All math was independently verified programmatically.

---

## Doc 01 — Session Cold Start Timing Report

**Status: ✅ Internally consistent, with one self-contradiction**

All raw numbers check out:

| Check | Expected | Calculated | Pass? |
|---|---|---|---|
| Total prefill steps | 135 | 51+6+72+6 = 135 | ✅ |
| Voice ms/frame | 53.0ms | 2705 / 51 = 53.0ms | ✅ |
| Text ms/token | 55.7ms | 4011 / 72 = 55.7ms | ✅ |
| Prefill subtotal | 7602.8ms (raw log) | 2705+553+4011+334 = 7603ms | ✅ |

### ⚠️ Internal Contradiction in Conclusion 3

Doc 01 makes two mutually exclusive claims:

> *"CUDA graph warmup adds ~2.8s to the first session only"* — implies warmup is a one-time server startup cost.

> *"reset_streaming() creates new CUDAGraphed objects each session, re-triggering capture"* — implies it is a per-session cost.

**The second claim is mathematically impossible.** If step[0] = 2,362ms and step[1] = 489ms occurred inside `voice_prompt_prefill` every session, the voice prefill alone would require at minimum 5,595ms. The measured value is 2,705ms. The CUDA graph warmup must be happening at server startup, outside the measured session window. Doc 03 likely has this right.

---

## Doc 02 — KV Cache Prefill and Snapshot Explainer

**Status: ⚠️ One factual error, rest is well-founded**

### 🔴 Factual Error: Voice Prompt Filename

- **Doc 01 (ground truth) says:** `NATF0.pt`
- **Doc 02 says:** `NATM0.pt`

This is a direct contradiction with the ground truth. The filename should be corrected to `NATF0.pt`.

### ✅ Everything Else Checks Out

- Step sequence (1–135) matches doc 01's breakdown exactly
- Snapshot size derivation: 2×1×32×3000×128 × 2 bytes × 32 layers = **1.46 GB ≈ 1.5 GB** ✓
- VRAM layout math: 18.7 + 1.5 + 1.5 + 0.5 + 1.8 = 24.0 GB ✓

### ℹ️ Note on Projected Values

The ~5ms snapshot creation, ~5ms restore, and ~50ms total cold start are **projections, not measured values** — appropriate since the feature is not yet built. They are internally plausible and consistent with the architecture, but should remain labeled as estimates.

---

## Doc 03 — TTFT Pipeline Analysis

**Status: 🟡 Mostly sound, directly corrects a Doc 01 error, TTFT estimate slightly optimistic**

### ✅ Correct Identification of Doc 01's CUDA Graph Error

Doc 03 flags doc 01's per-session CUDA graph re-capture claim as wrong, citing `_LMGenState.reset()` code where `graphed_main`, `graphed_embeddings`, and `graphed_depth` fields are untouched by `reset()`. The math above confirms this correction is valid.

### 🟡 TTFT Estimate is Slightly Optimistic

Independent cross-check of the same pipeline components yields **~336ms**, versus doc 03's stated **~310ms**. The ~8% gap likely comes from rounding down per-frame processing and assuming a cleaner pipeline than reality. Not a material error, but worth flagging.

### ℹ️ Unverified Client-Side Timings

The following are **inferred from code inspection, not profiling**:

- 80ms mic capture buffer (from `useUserAudio.ts`)
- 240ms client playback buffer (3 frames × 80ms, from `audio-processor.ts`)
- ~3–5ms for `mimi.encode` / `mimi.decode`

These are reasonable estimates but have not been empirically measured.

### ✅ A100 Bandwidth Figure

The ~2.0 TB/s HBM bandwidth figure is accurate for the A100 SXM4 variant.

---

## Doc 04 — KV Cache Snapshot/Restore Plan

**Status: ✅ Solid, all derived numbers verified**

| Check | Doc Claims | Verified |
|---|---|---|
| Snapshot size | ~1.5 GB | 1.46 GB ✅ |
| Voice + silence1 savings | 3.25s | 3.26s ✅ |
| Remaining text prefill | ~4.4s | 4.34s ✅ |
| Offset after voice + silence1 | 57 | 51+6 = 57 ✅ |
| Per-layer snapshot bytes | ~47 MB | 49.1 MB ✅ (close) |

### ℹ️ CPU RAM Restore Estimate is Conservative

Doc 04 estimates ~150–200ms for CPU RAM → GPU restore. At PCIe 4.0 bandwidth (~32 GB/s), 1.5 GB would transfer in ~47ms at the physical layer. The 150–200ms range accounts for Python/CUDA synchronization overhead and is a safe, conservative estimate. Not a problem — worth noting if aiming to tighten the TTFT target.

### ✅ Clone-Before-Restore Gotcha Correctly Identified

The note about cloning the snapshot before calling `set_streaming_state_inplace()` (to avoid mutating the cached copy) is correct and important.

---

## Summary of Issues

| Issue | Document | Severity |
|---|---|---|
| Voice prompt filename: `NATM0.pt` should be `NATF0.pt` | Doc 02 | 🔴 Factual error — fix this |
| Conclusion 3 self-contradiction on CUDA graphs | Doc 01 | 🟡 Wrong interpretation — math disproves per-session claim |
| TTFT ~310ms estimate is ~8% optimistic (~336ms calculated) | Doc 03 | 🟡 Minor — acceptable for a projection |
| Snapshot/restore times (~5ms, ~50ms) are projections | Docs 02, 03 | ℹ️ Fine — label clearly as estimates |
| CPU RAM restore ~150–200ms is conservative (physics: ~47ms) | Doc 04 | ℹ️ Safe conservatism — acceptable |

---

## Verdict

The architecture, logic, and core numbers across all four documents are **sound**. The most important fix is correcting `NATM0.pt → NATF0.pt` in Doc 02. The most important clarification is removing or rewriting the second sentence of Doc 01's Conclusion 3, since it is mathematically inconsistent with the measured timing data in the same document.
