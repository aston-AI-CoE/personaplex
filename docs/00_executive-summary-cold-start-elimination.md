# Cold Start Elimination for PersonaPlex Voice Sessions

**Author:** Aston
**Date:** March 12–16, 2026
**Status:** Implemented and validated
**Model:** nvidia/personaplex-7b-v1 (7B params, bfloat16)

**Instance:** AWS g5.2xlarge, us-east-2
**GPU:** NVIDIA A10G — 24 GB VRAM, PCIe Gen4 x8, driver 580.126.09
**CPU:** AMD EPYC 7R32 — 4 cores / 8 threads (2 threads per core)
**RAM:** 31 GB DDR4 (no swap)
**Storage:** NVMe SSD (EBS gp3)

---

## Executive Summary

PersonaPlex voice sessions had a 7.6-second cold start — the time between a user connecting and hearing the AI speak. This delay was caused by the model running 135 sequential forward passes through its 7B-parameter transformer to fill the KV cache with voice and text prompt context.

We eliminated this by pre-computing the KV cache offline (snapshot) and restoring it via memory copy on each session. Cold start dropped from 7,621 ms to 520 ms — a 14.7x improvement. The user now hears the AI in under 1 second.

The approach required two new files (snapshot generation script, HTML overlay) and modifications to the server's session startup path. A follow-up investigation uncovered that CUDA pinned memory causes audio stuttering at scale; the final implementation uses regular CPU heap memory with no performance penalty.

---

## 1. Problem Statement

Every time a user connects to a PersonaPlex voice session, they wait **7.6 seconds** of silence before the AI speaks. This delay repeats on every connection — even when using the same voice and text prompt.

---

## 2. Time Analysis

Instrumented the full session startup path with per-phase timing:

| Phase | Time | % of Total |
|---|---|---|
| Voice prompt file load | 1.2 ms | 0% |
| Text tokenization | 0.2 ms | 0% |
| Lock acquisition | 0.0 ms | 0% |
| Streaming reset | 3.8 ms | 0% |
| **Voice prompt prefill** | **2,705 ms** | **35%** |
| Silence padding 1 | 553 ms | 7% |
| **Text prompt prefill** | **4,011 ms** | **53%** |
| Silence padding 2 | 334 ms | 4% |
| Mimi post-reset | 1.1 ms | 0% |
| **Total cold start** | **7,621 ms** | **100%** |

**Finding:** The entire 7.6 seconds is spent on "prefill" — 135 sequential steps where the model reads the voice and text prompt one piece at a time. Each step runs the full 7-billion-parameter transformer network (~56 ms per step). File loading, network setup, and everything else is under 5 ms combined.

**Why it can't be parallelized:** Each step must see the output of all previous steps. Step 100 depends on steps 0–99. This is fundamental to how transformer attention works.

---

## 3. Technical Approach
### 3.1 — KV cache structure

The transformer model maintains a KV cache (Key-Value cache) — a ~1.5 GB buffer in GPU VRAM that stores attention keys and values across all 32 layers. During prefill, each forward pass reads all prior cache entries and writes a new one, encoding the voice identity and text instructions into the model's context. Steps cannot be parallelized — step N depends on the output of steps 0 through N-1 (causal attention).

The KV cache is allocated once at server startup and reused for every session.

### 3.2 — The snapshot idea

Instead of re-computing the KV cache on every session (7.6s), we:
1. Run the prefill once offline and save the resulting KV cache state (snapshot) to disk
2. At server startup, load the snapshot into CPU RAM
3. On each session, copy the snapshot into the live GPU KV cache (~520 ms)

This replaces 135 neural network forward passes with a memory copy.

The codebase already had the infrastructure: save_streaming_state(), load_streaming_state(), and set_streaming_state_inplace() in the streaming module. We wired them into the session startup flow.

### 3.3 — What the snapshot contains

The snapshot captures the full LMGen streaming state after prefill:
- Per-layer KV cache tensors: [2, 1, 32, 3000, 128] bfloat16 x 32 layers (~1.46 GB)
- Per-layer offset counters (end_offset, offset_cpu)
- LMGen state: token cache, provided mask, step offset

What is NOT captured (and doesn't need to be):
- CUDA graphs — pre-compiled GPU execution plans that persist across sessions from warmup(); excluded automatically (CUDAGraphed.asdict() returns {})
- Mimi codec state — the audio encoder/decoder is reset after prefill in both the original and snapshot paths
- Depformer state — the depth transformer's streaming is disabled (set_streaming_propagate(False)); it creates fresh state per step

### 3.4 — Where to store the snapshot

We evaluated four storage tiers for holding cached snapshots between sessions:

| Storage tier | Estimated restore time | Measured | Verdict |
|---|---|---|---|
| GPU VRAM (keep a copy on the GPU) | ~50 ms | Not tested | Too expensive — each copy uses 1.5 GB of the GPU's limited 24 GB |
| CPU RAM, pinned (locked physical pages) | ~150–200 ms | ~525 ms (measured), but causes audio stuttering | Rejected — see section 6 |
| CPU RAM, unpinned (regular memory) | Not estimated | **~520 ms (measured), clean audio** | **Selected** |
| NVMe disk (SSD) | ~500–1,000 ms | 3,000–11,000 ms | Rejected — OS file caching is unreliable at this data size |

"Pinned" memory means telling the OS to lock specific RAM pages so they can never be swapped or moved. This is normally faster for GPU transfers, but at multi-GB scale it caused problems (section 6).

---

## 4. Implementation (Doc 05)

### 4.1 — Files changed

**generate_snapshot.py** (new, 175 lines)
Offline script. Loads the model, sets voice + text prompt, runs the full 135-step prefill, then saves the resulting KV cache to disk. Produces three files:
- snapshot.safetensors — the KV cache tensors (~1.46 GB)
- snapshot_metadata.json — position counters and non-tensor state
- snapshot_config.json — which voice/text prompt this snapshot is for

**server.py** (modified)
- New --snapshot-dir CLI argument
- At startup: scans the snapshot directory, loads each snapshot into CPU RAM
- Per session: computes a lookup key from the incoming voice + text prompt (SHA-256 hash), finds the matching snapshot, copies it to GPU. If no match exists, falls back to full prefill transparently
- New /api/scenarios endpoint for the frontend to discover available snapshots

**index.html** (new overlay)
Lightweight vanilla JS overlay on the existing React app. Shows available snapshots as clickable cards. No React rebuild required.

### 4.2 — Snapshot generation

Generated 10 snapshots with varying prompt lengths to validate consistency:

| Snapshot | Text tokens | Total prefill steps | Prefill time |
|---|---|---|---|
| Snapshot 1 | 113 | 177 | 9.9s |
| Snapshot 2 | 111 | 174 | 9.5s |
| Snapshot 3 | 140 | 221 | 12.1s |
| Snapshot 4 | 116 | 172 | 9.4s |
| Snapshot 5 | 116 | 177 | 9.7s |
| Snapshot 6 | 129 | 193 | 10.6s |
| Snapshot 7 | 111 | 174 | 9.5s |
| Snapshot 8 | 141 | 205 | 11.3s |
| Snapshot 9 | 160 | 219 | 12.0s |
| Snapshot 10 | 141 | 204 | 11.2s |

Total prefill steps = voice frames + silence + text tokens + silence. Each step costs ~56 ms, so prefill time scales linearly. The output file is always 1.46 GB — the KV cache shape is fixed by the model architecture, regardless of prompt length.

---

## 5. Measured Results (Doc 05)

### 5.1 — Cold start across 10 different snapshots

Each snapshot tested once, sequentially (simulating 10 different users each picking a different scenario):

| Snapshot | Cold Start |
|---|---|
| Snapshot 1 | 607 ms |
| Snapshot 2 | 527 ms |
| Snapshot 3 | 535 ms |
| Snapshot 4 | 526 ms |
| Snapshot 5 | 526 ms |
| Snapshot 6 | 525 ms |
| Snapshot 7 | 519 ms |
| Snapshot 8 | 526 ms |
| Snapshot 9 | 521 ms |
| Snapshot 10 | 527 ms |
| **Mean** | **534 ms** |

### 5.2 — Where the 525 ms goes (server-side breakdown)

| Phase | Time | What it does |
|---|---|---|
| Streaming reset | 4 ms | Zeros the KV cache and resets offsets |
| Clone snapshot | ~201 ms | Makes a copy of the cached snapshot in CPU RAM (needed because the restore API consumes its input) |
| Restore to GPU | ~299 ms | Copies 1.46 GB from CPU RAM into GPU KV cache via tensor.copy_() |
| Mimi codec reset | 1 ms | Resets the audio encoder/decoder |
| **Server-side total** | **~507 ms** | |
| WebSocket overhead | ~13 ms | Connection setup |
| **Client-measured total** | **~520 ms** | |

### 5.3 — Before vs After

| Metric | Before | After | Change |
|---|---|---|---|
| Cold start | 7,621 ms | 520 ms | **14.7x faster** |
| Neural network steps per session | 135 at 56 ms each | 0 | Eliminated entirely |
| What happens instead | N/A | Memory copy (1.46 GB CPU → GPU) | ~300 ms |

---

## 6. Audio Blitzing — Bug Found and Fixed (Doc 06)

### 6.1 — Symptom

After the initial implementation, audio played normally for 20–30 seconds then began stuttering — short bursts of distorted audio. This happened during ALL sessions, even ones that didn't use snapshots.

### 6.2 — Debugging

Initially suspected:
- Client-side audio buffer settings — ruled out (byte-identical files between working and broken builds)
- The frontend rebuild process — ruled out (reverted to the known-good HuggingFace dist)
- Snapshot restore corrupting model state — ruled out (stuttering also happened on the full prefill path)

### 6.3 — Root cause: pinned memory

Isolated via A/B testing:

| Configuration | Pinned memory | Free RAM | Audio quality |
|---|---|---|---|
| 0 snapshots loaded | 0 GB | ~26 GB | Clean |
| 1 snapshot, pinned | 1.5 GB | ~24 GB | Clean |
| 5 snapshots, pinned | 7.3 GB | ~15 GB | **Stuttering** |
| 5 snapshots, unpinned | 0 GB pinned | ~18 GB | **Clean** |

The initial implementation used pin_memory() — a CUDA function that locks physical RAM pages so the GPU can read from them faster via direct memory access (DMA). This is a standard optimization for GPU data transfers.

However, locking 7+ GB of physical RAM on a 30 GB instance interfered with the GPU's DMA subsystem during real-time audio processing. The model must process one audio frame every 80 ms to keep up with real-time playback. Occasionally, the DMA contention caused a frame to take longer than 80 ms, which emptied the client's audio buffer, producing an audible glitch.

### 6.4 — The fix (fix 1 → fix 2)

**Fix 1** (initial): Load snapshots with pin_memory() — fast GPU transfers but causes stuttering.
**Fix 2** (final): Load snapshots with .clone() into regular heap memory — ~520 ms restore (measured, essentially identical to pinned), no stuttering, and faster startup (~4s vs ~12s per snapshot).

The key insight: pin_memory() is designed for short-lived transfer buffers, not multi-GB persistent caches.

---

## 7. End-to-End Time to First Audio (Doc 04)

After snapshot restore, there's still a pipeline before the user hears sound. Here's the full chain:

| Step | Time | What happens |
|---|---|---|
| Snapshot restore | ~520 ms | KV cache copied from CPU to GPU (measured, unpinned) |
| Mic capture + Opus encode | ~80 ms | Browser captures 80 ms of mic audio, encodes to Opus format (estimated) |
| Network to server | ~5 ms | WebSocket, same availability zone (estimated) |
| Server processes 1 frame | ~62 ms | Mimi encode (3 ms) + transformer step (56 ms) + Mimi decode (3 ms) (derived from measured step time) |
| Network to client | ~5 ms | (estimated) |
| Client Opus decode | ~3 ms | (estimated) |
| Playback buffer fill | ~240 ms | Client buffers 3 frames (240 ms) before playing to absorb timing jitter (from code) |
| **Estimated total TTFT** | **~835–865 ms** | |

The playback buffer (240 ms) is the largest remaining contributor after snapshot restore.

---

## 8. Scaling - In my server test, 10 snapshots

### 8.1 — Memory budget on this instance

| Component | GPU VRAM | CPU RAM |
|---|---|---|
| Model weights (7B, bf16) | ~18.7 GB | — |
| Live KV cache | ~1.5 GB | — |
| CUDA graphs + runtime overhead | ~3.8 GB | — |
| 10 cached snapshots (unpinned) | — | ~14.6 GB |
| Python interpreter + OS | — | ~5 GB |
| **Total used** | **~24 GB** | **~20 GB** |
| **Available** | **~0 GB** | **~10 GB** |

### 8.2 — Practical limits

- Each unique (voice, text prompt) pair requires its own snapshot — different prompts produce different KV cache contents
- This 30 GB instance fits ~10 snapshots with 10 GB headroom
- Disk: 1.46 GB per snapshot, 10 snapshots = ~15 GB
- Larger instances (64 GB+ RAM) can hold 20+ snapshots comfortably

---

## 9. Remaining Work

### 9.1 — Pre-warm restore (target: 0 ms user-perceived cold start)

Currently, snapshot restore happens when the user connects (525 ms in their path). The pre-warm architecture would restore the KV cache in the background before the user even receives the notification:

1. Backend assembles the briefing and determines the voice + text prompt
2. Background task restores the KV cache from the matching snapshot
3. Push notification sent to user
4. User taps → WebSocket connects → KV cache already warm → no restore needed

This moves the entire 525 ms out of the user's path.

### 9.2 — Prefix-cache variant (same voice, different text)

For scenarios where many users share the same voice but have different text prompts: save a partial snapshot after the voice portion only (~57 steps, ~3.3s), then run only the text prefill on demand. Reduces cold start from 7.6s to ~4.3s — not eliminated, but a meaningful improvement for the dynamic-text case.

### 9.3 — Client playback buffer optimization

The client currently buffers 3 audio frames (240 ms) before starting playback. This is conservative — with CUDA-graphed inference, frame timing jitter is minimal (~56 ms +/- 2 ms). Reducing to 2 frames (160 ms) would save ~80 ms off TTFT. Needs testing on variable network conditions.

---

## 10. Summary

| What | Result |
|---|---|
| Cold start | 7,621 ms → 520 ms (14.7x improvement) |
| Root cause | 135 sequential transformer forward passes to fill KV cache |
| Solution | Pre-compute KV cache offline, restore via memory copy per session |
| Storage | Regular CPU RAM, ~1.46 GB per snapshot |
| Audio quality | Clean (after replacing pin_memory with .clone()) |
| Scaling | ~10 snapshots on 30 GB instance, ~20+ on 64 GB |
