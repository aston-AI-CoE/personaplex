# KV Cache Snapshot Implementation Results

**Type:** Implementation report with empirical measurements
**Date:** March 16, 2026
**Hardware:** NVIDIA A10G (24GB VRAM), g5.2xlarge EC2, us-east-2 (same as doc 01)
**Model:** nvidia/personaplex-7b-v1 (7B params, bfloat16)
**Related:** [01](01_session-cold-start-timing-report.md) | [02](02_kv-cache-prefill-and-snapshot-explainer.md) | [03](03_kv-cache-snapshot-restore-plan.md) | [04](04_ttft-pipeline-analysis.md)

---

## Summary

KV cache snapshot/restore is implemented and measured. Cold start dropped from **7,621ms to ~520ms** — a **14.7x improvement**. The implementation follows the plan in [doc 03](03_kv-cache-snapshot-restore-plan.md) with minor adjustments based on empirical findings.

---

## What Was Built

### 1. `moshi/moshi/generate_snapshot.py` — Offline snapshot generation

Standalone script that generates a KV cache snapshot for a given (voice_prompt, text_prompt) pair:
- Loads model identically to `server.py`
- Runs the full prefill synchronously via `step_system_prompts()` (not the async version)
- Saves KV cache state to `snapshot.safetensors` + `snapshot_metadata.json` via the existing `save_streaming_state()` API
- Saves scenario metadata to `snapshot_config.json` (voice, text, hash, label, description, icon)

Usage:
```bash
python3.11 -m moshi.generate_snapshot \
    --voice-prompt /path/to/NATF0.pt \
    --text-prompt "Your system prompt here" \
    --output-dir ./snapshots/scenario_00 \
    --label "Medical Receptionist" \
    --icon "🏥"
```

### 2. `moshi/moshi/server.py` — Snapshot loading and restore

**`ServerState.__init__`**: Added `snapshot_dir` parameter. At startup, scans the directory (and subdirectories) for snapshot files. Each snapshot is:
1. Loaded from disk via `load_streaming_state()` (~50ms per snapshot)
2. Cloned into regular CPU heap RAM via `.clone()` (~4s per snapshot, one-time)
3. Indexed by `{voice_prompt}:{text_prompt_hash}` for fast lookup

Note: `pin_memory()` was initially used but causes audio blitzing — see [doc 06](06_pin-memory-audio-blitzing-root-cause.md).

**`handle_chat`**: Before the prefill section, looks up the incoming request's voice/text config against the snapshot cache:
- **Cache hit**: Clone from CPU heap RAM → restore via `set_streaming_state_inplace()` → ~525ms
- **Cache miss**: Falls back to full prefill → ~5,000-10,000ms (unchanged)

**`/api/scenarios`**: New endpoint returning available pre-cached scenarios as JSON, consumed by the frontend.

**`--snapshot-dir` CLI arg**: Points to the snapshot directory. Multiple snapshots supported via subdirectories.

### 3. `client/dist/index.html` — Demo UI with scenario cards

Added a vanilla JS overlay on the existing HF-distributed React app (no rebuild required):
- Fetches available scenarios from `/api/scenarios` at page load
- Each card shows the scenario label, description, voice name, and an "Instant" badge
- Clicking a card sets the voice/text prompt and auto-connects
- "Use a custom prompt" link dismisses the overlay to access the original React form

### 4. `if __name__ == "__main__"` guard on `server.py`

Added so `generate_snapshot.py` can import `ServerState` and `wrap_with_system_tags` without triggering the server's `main()`.

---

## Measured Results

### Per-scenario cold start (10 scenarios, 1 connection each)

| Scenario | Voice | Text tokens | Offset | Cold Start |
|----------|-------|-------------|--------|------------|
| Medical Receptionist | NATF0.pt | 113 | 177 | 607ms |
| IT Help Desk | NATM0.pt | 111 | 174 | 527ms |
| Real Estate Agent | VARF0.pt | 140 | 221 | 535ms |
| Financial Advisor | VARM0.pt | 116 | 172 | 526ms |
| Hotel Concierge | NATF1.pt | 116 | 177 | 526ms |
| Auto Service Advisor | NATM1.pt | 129 | 193 | 525ms |
| Fitness Coach | VARF1.pt | 111 | 174 | 519ms |
| Bank Customer Service | NATF2.pt | 141 | 205 | 526ms |
| Travel Consultant | VARM1.pt | 160 | 219 | 521ms |
| Insurance Claims Agent | NATM2.pt | 141 | 204 | 527ms |

**Mean: 534ms** across all 10 different scenarios, each tested once sequentially (simulating different users clicking different cards).

### Server-side timing breakdown (steady state)

| Phase | Time | Notes |
|-------|------|-------|
| streaming_reset | ~4ms | Zero KV caches, reset offsets |
| snapshot_clone | ~201ms | Clone 1.46 GB from CPU heap RAM |
| snapshot_restore | ~299ms | `set_streaming_state_inplace()` — CPU→GPU `tensor.copy_()` |
| mimi_post_reset | ~1ms | Reset audio codec |
| **Total cold start (server-side)** | **~507ms** | |
| + WebSocket overhead | ~13ms | Connection setup |
| **Total cold start (client-measured)** | **~520ms** | |

### Comparison to baseline

| Metric | Before (doc 01) | After (snapshot) | Improvement |
|--------|-----------------|------------------|-------------|
| Cold start | 7,621ms | 520ms | **14.7x faster** |
| Prefill forward passes | 135 steps × 56ms | 0 steps | Eliminated |
| Restore mechanism | N/A | clone + copy_() (unpinned) | ~507ms |
| Snapshot size (each) | N/A | 1.46 GB | As predicted in doc 02 |
| Snapshot load time (disk→CPU, per file) | N/A | ~50ms | |
| Pin memory time (per file, one-time) | N/A | ~12s | Done at startup (fix 1, replaced — see below) |
| Clone to heap RAM (per file, one-time) | N/A | ~4s | Done at startup (fix 2, current) |

**Fix 1 → Fix 2:** The initial implementation used `pin_memory()` to lock snapshot tensors in physical RAM for faster GPU transfers. This worked for cold start performance but caused audio blitzing during conversations when multiple snapshots were loaded (see [doc 06](06_pin-memory-audio-blitzing-root-cause.md)). Fix 2 replaced `pin_memory()` with `.clone()` into regular heap memory. Measured restore time is essentially identical: ~520ms unpinned vs ~525ms pinned. Audio is clean. Startup is also faster (~4s vs ~12s per snapshot).

### Snapshot generation timing (offline, per scenario)

| Phase | Time |
|-------|------|
| Model load (mimi + moshi + tokenizer) | ~6s |
| Warmup (CUDA graph creation) | ~3s |
| Prefill (voice + silence + text + silence) | 5,000–12,000ms (varies by prompt length) |
| Save to disk | ~3s |
| **Total per snapshot** | ~15–25s |

---

## Snapshot File Structure

```
snapshots/
├── scenario_00/
│   ├── snapshot.safetensors      # 1.46 GB — KV cache tensors
│   ├── snapshot_metadata.json    # ~3 KB — offset counters, non-tensor state
│   └── snapshot_config.json      # ~0.5 KB — voice, text, hash, label, icon
├── scenario_01/
│   ├── ...
└── scenario_09/
    └── ...
```

Each `snapshot_config.json` contains:
```json
{
  "voice_prompt": "NATF0.pt",
  "text_prompt": "You work for Dr. Martinez's...",
  "text_prompt_hash": "78c88e02ceb33d...",
  "num_text_tokens": 113,
  "offset_after_prefill": 177,
  "label": "Medical Receptionist",
  "description": "Schedule appointments and assist patients...",
  "icon": "🏥"
}
```

---

## Architecture Decisions and Findings

### 1. Unpinned CPU RAM is the right storage tier

Doc 03 estimated three tiers. Here are the measured results vs estimates:

| Tier | Doc 03 Estimate | Measured | Notes |
|------|----------------|----------|-------|
| GPU VRAM copy | ~50ms | Not tested | Would consume 1.5 GB VRAM per snapshot |
| CPU RAM (pinned) → GPU | ~150-200ms | **~525ms** (measured) but causes audio blitzing | See [doc 06](06_pin-memory-audio-blitzing-root-cause.md) |
| CPU RAM (unpinned, `.clone()`) → GPU | Not estimated | **~520ms** (measured), clean audio | **Recommended** |
| NVMe disk → GPU | ~500-1000ms | Not viable for multi-snapshot | OS page cache eviction under memory pressure |

The 525ms total is higher than the 150-200ms estimate from doc 03 because the estimate did not account for:
- **Clone cost (~210ms)**: `set_streaming_state_inplace()` pops keys from the state dict during restore, so the cached snapshot dict can't be reused directly. Each restore needs a fresh dict with cloned tensors.
- **Python/CUDA overhead**: The transfer isn't a single contiguous memcpy — it's ~100+ individual tensor copies (32 layers x 3 tensors each + metadata), each with Python dispatch overhead.

### 2. Disk-only approach does not work for multiple snapshots

We tested loading snapshots from disk on each request (no RAM cache). While individual disk reads are fast (~50ms from OS page cache), the OS page cache cannot retain 10 × 1.46 GB = 14.6 GB simultaneously on a 30 GB system. Later snapshots experience page faults, causing 3–11 second first-use penalties. **Conclusion: snapshots must be materialized in process memory, not mmap'd.**

### 3. `set_streaming_state_inplace` requires dict mutation (clone is necessary)

The restore API pops keys from the input dict as a safety check (verifies all keys are consumed). This means we cannot pass the cached dict directly — it would be empty after the first restore. A clone of the dict values is required for each session.

### 4. Snapshot is config-agnostic at restore time

The snapshot restore does not validate that the incoming voice/text prompt matches the snapshot. It simply writes the cached KV state. Config matching is handled by the cache lookup key (`voice:text_hash`). If no match is found, the server falls back to full prefill.

### 5. CUDA graphs persist across restore (confirmed)

As predicted in docs 02 and 03, CUDA graphs created during `warmup()` are not affected by snapshot restore. `set_streaming_state_inplace()` only touches data tensors, not the `CUDAGraphed` objects. The first `lm_gen.step()` after restore runs at full graphed speed (~56ms).

---

## Memory Budget (10 snapshots, A10G)

| Component | GPU VRAM | CPU RAM |
|-----------|----------|---------|
| Model weights (7B, bf16) | ~18.7 GB | — |
| Live KV cache | ~1.5 GB | — |
| CUDA graphs + overhead | ~3.8 GB | — |
| 10 snapshots (unpinned RAM) | — | ~14.6 GB |
| Python + OS | — | ~5 GB |
| **Total** | **~24 GB** (full) | **~20 GB** |

The A10G's 24 GB VRAM is fully utilized by model + KV cache + CUDA graphs. Snapshots reside entirely in CPU heap RAM (unpinned). System has 30 GB total RAM, so 10 snapshots fit with ~10 GB free. Pinned memory is not used (causes audio blitzing — see [doc 06](06_pin-memory-audio-blitzing-root-cause.md)).

---

## What's Not Implemented (Future Work)

1. **Pre-warm restore before user connects** (doc 02 "pre-warm flow"): The current implementation restores on WebSocket connect. The full pre-warm architecture (POST /api/prepare → background prefill → push notification → user connects to pre-restored session) is not yet built.

2. **Prefix-cache variant** (doc 03 "Same Voice, Different Text"): Saving a partial snapshot (voice + silence only) and running text prefill on demand. Useful when voice is shared but text varies per user.

3. **Config-matched multi-tenant**: The server currently loads all snapshots at startup. A production system would need per-tenant snapshot management, lifecycle (create/delete), and potentially lazy loading with LRU eviction.

4. **Playback buffer optimization** (doc 04): The 240ms client playback buffer is unchanged. Reducing to 2 frames (160ms) would save ~80ms from TTFT.
