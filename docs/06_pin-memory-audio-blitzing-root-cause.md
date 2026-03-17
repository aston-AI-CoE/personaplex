# Pin Memory Causes Audio Blitzing — Root Cause and Fix

**Type:** Debugging report
**Date:** March 16, 2026
**Hardware:** NVIDIA A10G (24GB VRAM), g5.2xlarge EC2 (30 GB CPU RAM), us-east-2
**Related:** [05](05_kv-cache-snapshot-implementation-results.md)

---

## Summary

Loading KV cache snapshots with `tensor.pin_memory()` causes audio blitzing (stuttering/glitching) during voice conversations. The fix: use regular CPU RAM (`.clone()`) instead of pinned memory. Cold start performance is unchanged. Audio quality is clean.

---

## Symptom

Audio plays normally for 20-30 seconds, then begins blitzing — short bursts of distorted/stuttered audio that recur throughout the session. The blitzing occurs on both the snapshot restore path and the full prefill fallback path, as long as pinned snapshots are resident in memory.

---

## Root Cause

`pin_memory()` locks physical RAM pages via CUDA's host memory allocator (`cudaHostAlloc`). These pages cannot be paged out, moved, or reclaimed by the OS.

On this instance (30 GB total RAM):

| Configuration | Pinned | Free RAM | Audio |
|---------------|--------|----------|-------|
| 0 snapshots | 0 GB | ~26 GB | Clean |
| 1 snapshot (pinned) | 1.5 GB | ~24 GB | Clean |
| 5 snapshots (pinned) | 7.3 GB | ~15 GB | Blitzing |
| 10 snapshots (pinned) | 14.6 GB | ~5 GB | Blitzing |
| 5 snapshots (unpinned, `.clone()`) | 0 GB pinned | ~18 GB | Clean |

The blitzing occurs even on the full prefill path (no snapshot code executes during the conversation). The mere presence of pinned memory in the process is sufficient to cause it.

The likely mechanism: pinned memory competes with CUDA's DMA subsystem for PCIe bandwidth and memory controller resources. During real-time audio processing (encode + 7B transformer step + decode per 80ms frame), occasional DMA stalls cause frames to exceed their 80ms real-time budget. The client's tight playback buffer (80ms initial, 10ms overflow threshold) has no margin to absorb these stalls, resulting in buffer underruns that manifest as audible glitches.

---

## The Fix

Replace `pin_memory()` with `.clone()` when loading snapshots at startup.

Before (causes blitzing):
```python
snapshot = load_streaming_state(snapshot_path, metadata_path, device='cpu')
snapshot = {
    k: v.pin_memory() if isinstance(v, torch.Tensor) else v
    for k, v in snapshot.items()
}
```

After (clean audio):
```python
snapshot = load_streaming_state(snapshot_path, metadata_path, device='cpu')
snapshot = {
    k: v.clone() if isinstance(v, torch.Tensor) else v
    for k, v in snapshot.items()
}
```

The `.clone()` is still needed to copy tensors from the mmap'd safetensors file into regular heap memory (prevents OS page cache eviction issues discovered earlier — see doc 05).

---

## Performance Impact

| Metric | With pin_memory | Without pin_memory |
|--------|----------------|-------------------|
| Snapshot load time (per file) | ~12s | ~4s |
| 5-snapshot startup total | ~70s | ~40s |
| Cold start (clone + restore) | ~525ms | **~520ms (measured)** |
| Audio quality | Blitzing | Clean |

Measured unpinned cold start is ~520ms — essentially identical to pinned (~525ms). The theoretical overhead of unpinned transfers (intermediate staging buffer) is not measurable at this scale because per-tensor Python dispatch overhead (~100 individual copy operations) dominates over raw transfer speed.

---

## Design Implications

1. **`pin_memory()` should not be used for long-lived allocations** on memory-constrained instances. It is designed for short-lived data transfer buffers, not multi-GB persistent caches.

2. **Scaling snapshots on a 30 GB instance**: Without pinning, 10 snapshots (14.6 GB) in regular RAM leaves ~10 GB free. This should be safe since unpinned memory can be managed normally by the OS. Testing with 10 unpinned snapshots is recommended to confirm.

3. **Larger instances eliminate the constraint**: On a 64 GB instance (e.g., g5.4xlarge), even 10 pinned snapshots would leave ~45 GB free, likely avoiding the issue. However, unpinned is still the safer default.

---

## Debugging Timeline

The blitzing was initially misattributed to:
- Audio processor buffer settings (`audio-processor.ts`) — ruled out by diffing the working HF dist against the rebuild (buffer values were identical after applying `configure.sh` patches)
- The Vite rebuild process — ruled out by switching to the unmodified HF dist with `sed` patches
- Snapshot restore state correctness — ruled out by observing blitzing on the full prefill path too
- Memory pressure (insufficient free RAM) — partially correct, but the specific mechanism is pinned memory, not free RAM quantity

The root cause was isolated through A/B testing:
1. Zero snapshots, 26 GB free: clean
2. Five pinned snapshots, 15 GB free: blitzing
3. Five unpinned snapshots, 18 GB free: clean
