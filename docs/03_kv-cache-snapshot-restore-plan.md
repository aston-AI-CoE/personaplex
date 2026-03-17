# KV Cache Snapshot/Restore to Eliminate Cold Start

**Type:** Implementation plan
**Date:** March 12, 2026
**Related:** [01_session-cold-start-timing-report.md](01_session-cold-start-timing-report.md) | [02_kv-cache-prefill-and-snapshot-explainer.md](02_kv-cache-prefill-and-snapshot-explainer.md)

---

## Why This Works

The 7.6s cold start is entirely sequential LM forward passes filling the KV cache (135 steps at ~56ms each -- measured in [doc 01](01_session-cold-start-timing-report.md)). Non-prefill overhead is <5ms (0.07%). If we pre-compute this once and save the resulting KV cache state, we can restore it via `tensor.copy_()` instead of re-running all 135 steps.

The codebase already has the infrastructure for this:
- `streaming.py` lines 367-391: `save_streaming_state()` -- serializes all KV cache tensors + metadata to safetensors + JSON
- `streaming.py` lines 232-259: `load_streaming_state()` -- loads from disk
- `streaming.py` lines 393-403: `set_streaming_state_inplace()` -- copies loaded tensors into live streaming state via `tensor.copy_()`

---

## Snapshot Size (Calculated)

Main LM transformer KV cache (the bulk):
- Per attention layer: `(2, 1, 32, 3000, 128)` bfloat16 = ~49 MB (2 x 1 x 32 x 3000 x 128 x 2 bytes)
- 32 layers = **~1.5 GB** total
- Plus small tensors (cache, provided, offsets) = negligible

---

## Restore Speed by Storage Tier

| Strategy | Cold Start | Savings vs 7.6s | VRAM Cost | Status |
|----------|-----------|-----------------|-----------|--------|
| **GPU VRAM cache** (tensor.copy_) | ~50ms | 99.3% | +1.5 GB per cached config | Estimate (not tested) |
| **CPU RAM cache** (pinned, clone+copy) | **~525ms** | **93%** | 0 VRAM, +1.5 GB RAM | **Measured** — but causes audio blitzing |
| **CPU RAM cache** (unpinned, .clone()) | **~520ms** | **93%** | 0 VRAM, +1.5 GB RAM | **Measured (doc 05, final)** |
| **NVMe SSD** (disk to GPU) | 3,000-11,000ms | Unreliable | 0 | **Measured — not viable for multi-snapshot** |
| **EBS gp3** (network disk to GPU) | ~1500-3000ms | 60-80% | 0 | Estimate |

> **Update (March 16, 2026):** The original estimate of 150-200ms for CPU RAM restore did not account for two costs: (1) cloning pinned tensors (~210ms) because `set_streaming_state_inplace` pops keys from the input dict, and (2) per-tensor Python dispatch overhead across ~100 copy operations (~300ms for the actual CPU→GPU transfer). The NVMe tier is not viable for multiple snapshots — the OS page cache (30 GB system) cannot retain 10 × 1.46 GB simultaneously, causing page faults and 3-11s latencies.

**Updated recommendation:** CPU RAM cache with `.clone()` (unpinned). Pinned and unpinned have essentially identical restore performance (~525ms vs ~520ms), but pinned causes audio blitzing on memory-constrained instances (see [doc 06](06_pin-memory-audio-blitzing-root-cause.md)). 520ms cold start is a 14.7x improvement over 7.6s.

---

## Architecture

```mermaid
flowchart TB
    subgraph offline [Offline: Snapshot Generation]
        A[Load model] --> B[streaming_forever + warmup]
        B --> C[Set voice_prompt + text_prompt]
        C --> D["Run step_system_prompts_async (7.6s, ONE TIME)"]
        D --> E["save_streaming_state() to disk"]
    end

    subgraph startup [Server Startup]
        F[Load model weights to GPU] --> G[streaming_forever + warmup]
        G --> H["load_streaming_state() into CPU RAM"]
        H --> I[Hold snapshot in memory]
    end

    subgraph session [Per Session: Fast Restore]
        J[WebSocket connect] --> K[reset_streaming]
        K --> L["set_streaming_state_inplace(snapshot)"]
        L --> M["Send handshake (estimated ~200ms total)"]
    end

    E -.->|snapshot files on disk| H
    I -.->|CPU RAM tensors| L
```

---

## What State Is Captured

The snapshot captures the **full LMGen streaming state tree** after prefill (verified from `_flatten_streaming_state()` in `streaming.py` and `_LMGenState` in `lm.py`):

- `_LMGenState.cache` -- token history `[1, 17, max_delay+3]` (long)
- `_LMGenState.provided` -- teacher-forcing mask (bool)
- `_LMGenState.offset` -- step counter (= 135 after prefill)
- `lm_model.transformer.layers.{0..31}.self_attn.kv_cache.cache` -- the actual KV tensors `[2, 1, 32, 3000, 128]` (bfloat16)
- `lm_model.transformer.layers.{0..31}.self_attn.kv_cache.end_offset` -- position counters `[1]` (long)
- `lm_model.transformer.layers.{0..31}.self_attn.offset` -- attention offset `[1]` (long)
- `lm_model.transformer.layers.{0..31}.self_attn.offset_cpu` -- CPU-side offset (int, stored as metadata)
- `lm_model.transformer.offset` -- transformer-level offset `[1]` (long)

What is NOT captured (and does not need to be):
- CUDAGraphed objects (`CUDAGraphed.asdict()` returns `{}`) -- they persist across sessions from `warmup()`
- Mimi state -- `mimi.reset_streaming()` is called after prefill anyway
- `text_prompt_tokens`, `voice_prompt` -- instance attributes on LMGen, set separately before prefill
- Depformer state -- `depformer.set_streaming_propagate(False)` excludes it; depformer resets per step inside `depformer_step()`

---

## Implementation Steps

### Step 1: Snapshot generation script — ✅ Done

Created `moshi/moshi/generate_snapshot.py`:
- Uses sync `step_system_prompts()` (not async) — avoids needing an event loop
- Saves `snapshot.safetensors` + `snapshot_metadata.json` + `snapshot_config.json`
- Accepts `--label`, `--description`, `--icon` for frontend metadata

### Step 2: Server startup -- pre-load snapshot to CPU RAM — ✅ Done

In `ServerState.__init__()`:
- Scans `snapshot_dir` and subdirectories for snapshot files
- Loads each via `load_streaming_state()`, clones into heap RAM via `.clone()` (fix 1 used `pin_memory()` but caused audio blitzing — see [doc 06](06_pin-memory-audio-blitzing-root-cause.md))
- Indexes by `{voice_prompt}:{text_prompt_hash}` for O(1) lookup
- Multiple snapshots supported

### Step 3: Per-session fast restore — ✅ Done

In `handle_chat()`:
- Looks up snapshot by incoming `voice_prompt` + SHA-256 of `text_prompt`
- Clone from CPU RAM + `set_streaming_state_inplace()` (~520ms measured, unpinned)
- Falls back to full prefill on cache miss

### Step 4: Handle config matching — ✅ Done

- Cache key: `{voice_filename}:{sha256(text_prompt)}`
- Automatic fallback to full prefill if no matching snapshot exists
- `/api/scenarios` endpoint exposes available snapshots to the frontend

---

## Key Gotchas

1. **Snapshot is config-specific**: Different voice prompts or text prompts produce different KV caches. A snapshot for "NATF0.pt + medical intake" will not work for "NATF0.pt + legal briefing".

2. **Clone before restore**: `set_streaming_state_inplace` uses `tensor.copy_()` which overwrites the destination tensors in the live streaming state. The source snapshot in CPU RAM must not be mutated, so clone it before each restore. Alternatively, re-call `load_streaming_state()` each time (reads from disk, creates fresh tensors).

3. **Batch size must match**: Snapshot was generated with batch_size=1; restore must also use batch_size=1 (matching `streaming_forever(1)`).

4. **CUDA graphs unaffected**: CUDAGraphed objects are created during `streaming_forever()` + `warmup()` and persist. Snapshot restore only touches the data tensors inside the KV cache, not the graph structure. This is verified: `_LMGenState.reset()` does not touch `graphed_main`, `graphed_embeddings`, or `graphed_depth`.

5. **Depformer is excluded**: `depformer.set_streaming_propagate(False)` means depformer state is not part of the snapshot. This is correct -- depformer state is created fresh per step inside `depformer_step()` via `with lm_model.depformer.streaming(B)`.

---

## Prefix-Cache Variant (Same Voice, Different Text)

If all users share the same voice but have different text prompts:
- Save snapshot after voice_prompt + silence_1 only (offset = 57, i.e. 51 + 6)
- At session start, restore voice prefix, then run only text_prompt prefill + silence_2
- Saves ~3.26s (voice 2,705ms + silence_1 553ms), remaining cold start: ~4.35s for text prefill
- This is a middle ground if text prompts vary per user

For the stated use case (same voice AND same instruction for all users), the full snapshot eliminates everything.
