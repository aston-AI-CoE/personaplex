# Session Cold Start Timing Report

**Type:** Empirical timing report
**Date:** March 12, 2026
**Hardware:** NVIDIA A10G (24GB VRAM), g5.2xlarge EC2, us-east-2
**Model:** nvidia/personaplex-7b-v1 (7B params, bfloat16)
**Voice prompt:** NATF0.pt (pre-computed embeddings, 51 frames)
**Text prompt:** 72 tokens (medical office intake scenario)

---

## Summary

The session cold start -- the delay between WebSocket connect and handshake (audio begins) -- is **7.6 seconds**. It is caused entirely by KV cache prefill (sequential LM forward passes to prime the transformer with voice and text context). Model weight loading, file I/O, streaming reset, and lock acquisition contribute less than 5ms combined.

---

## Full Timing Breakdown

| Phase | Time | % of Total | Details |
|---|---|---|---|
| Voice prompt file load | 1.2ms | 0% | Already cached (.pt file) |
| Text tokenize | 0.2ms | 0% | sentencepiece, 72 tokens |
| Lock acquire | 0.0ms | 0% | No contention |
| Streaming reset | 3.8ms | 0% | Zero KV caches, reset offsets |
| **Voice prompt prefill** | **2,705ms** | **35%** | **51 frames at 53.0ms/frame** |
| Silence 1 | 553ms | 7% | 6 frames |
| **Text prompt prefill** | **4,011ms** | **53%** | **72 tokens at 55.7ms/token** |
| Silence 2 | 334ms | 4% | 6 frames |
| Mimi post-reset | 1.1ms | 0% | |
| **Total cold start** | **7,621ms** | **100%** | |

### Prefill cost breakdown

Total forward passes: 51 (voice) + 6 (silence) + 72 (text) + 6 (silence) = **135 sequential LM steps**

Each step runs:
1. `forward_codes()` -- full 7B transformer forward pass
2. `depformer_step()` -- depth transformer (8 codebook autoregressive)

Steady-state per-step cost: ~56ms (39ms transformer + 16ms depformer).

---

## Per-Step Granular Timing (During Initial Warmup)

These timings were captured during `warmup()` when the server process first launches (when `_step_timing_count` was 0/1/2), not during any user session. The server process stays running indefinitely after this. By the time any session connects, CUDA graphs are already captured and all steps run at steady-state speed.

| Step | Prepare | Transformer | Depformer | Total | Notes |
|---|---|---|---|---|---|
| step[0] | 0.9ms | 1,966ms | 395ms | 2,362ms | CUDA graph warmup (un-graphed) |
| step[1] | 0.4ms | 216ms | 273ms | 489ms | CUDA graph capture |
| step[2] | 0.4ms | 39ms | 16ms | 56ms | Steady-state (graphed, representative) |

The first two steps during `warmup()` incur ~2,850ms of one-time CUDA graph overhead. This happens once when the server process first launches and never again -- the CUDA graphs persist in memory for the lifetime of the process across all sessions. After graph capture, every step (including all session prefill steps) runs at ~56ms. This is confirmed by the session data: voice prefill averages 53.0ms/frame and text prefill averages 55.7ms/token, both consistent with graphed steady-state.

---

## Conclusions

1. **The bottleneck is prefill, not model loading.** Model weights are loaded once at server startup and remain in GPU memory. The per-session cost is entirely autoregressive forward passes to fill the transformer KV cache with voice prompt + text prompt context.

2. **Voice prompt and text prompt contribute roughly equally** to total prefill time (2.7s and 4.0s respectively), proportional to their step counts (51 and 72).

3. **CUDA graph warmup is a one-time cost when the server process first launches (~2.8s).** The `warmup()` method runs 4 dummy steps, during which CUDA graphs are created and captured. These graphs persist in memory for the lifetime of the process across all subsequent sessions because `reset_streaming()` only calls `_LMGenState.reset()`, which zeroes `offset` and `provided` but leaves the `graphed_main`, `graphed_embeddings`, and `graphed_depth` fields untouched. This is confirmed by the session prefill running at steady-state speed: 2,705ms / 51 frames = 53.0ms/frame, consistent with graphed step[2] timing of 56ms.

4. **A snapshot/restore approach** that caches the post-prefill streaming state (KV caches, offsets) and restores it via tensor `copy_()` would eliminate the 7.6s prefill entirely, reducing cold start to an estimated <200ms (see [04_kv-cache-snapshot-restore-plan.md](04_kv-cache-snapshot-restore-plan.md)).

---

## Architecture: Hybrid System Prompt

![Hybrid System Prompt Architecture](assets/hybrid-system-prompt-architecture.png)

The diagram shows the full prefill sequence (left of the dotted line) and generation (right). Everything under "Hybrid System Prompt" is the 7.6s cold start -- the 135 steps the model must process before it can start generating. The three input channels (User Audio, Agent Text, Agent Audio) are fed through Mimi (audio codec), the Temporal Transformer (7B main model), and the Depth Transformer (depformer) at each step.

---

## What You Experience (Step by Step)

This is what happens from your perspective, with measured times:

```
1. You start the server                          (one-time setup)
   +-- Model weights load into GPU
   +-- Warmup: 4 dummy steps to bake CUDA graphs    ~2.8s (never happens again)
   +-- Server is now idle, waiting for connections

2. You open the browser, pick a scenario, click connect
   +-- WebSocket connects to server

3. You wait...                                    <-- this is the 7.6s you feel
   |
   |  What's happening behind the scenes:
   |  +-- Load voice prompt file               1.2ms  (instant)
   |  +-- Tokenize text prompt                 0.2ms  (instant)
   |  +-- Zero out the KV cache                3.8ms  (instant)
   |  +-- Feed voice into model, 51 steps      2,705ms (2.7s)
   |  +-- Feed silence, 6 steps                  553ms (0.6s)
   |  +-- Feed text into model, 72 steps       4,011ms (4.0s)
   |  +-- Feed silence, 6 steps                  334ms (0.3s)
   |  +-- Total: 135 steps at ~56ms each       7,621ms
   |
4. Handshake arrives -- you hear sound             7.6s after clicking
```

The 7.6 seconds is entirely the model "reading" the voice and text prompts one step at a time (135 steps). Everything else is instant. If you disconnect and reconnect with the same or different scenario, you pay the same 7.6s again -- the model has to re-read the prompts from scratch each time.

Step 1 (starting the server) only happens once. Steps 2-4 repeat every time you start a new session.

---

## Raw Log Output

```
[TIMING] step[0]: prepare=0.92ms transformer=1966.30ms depformer=394.78ms total=2362.00ms
[TIMING] step[1]: prepare=0.36ms transformer=215.73ms depformer=273.18ms total=489.27ms
[TIMING] step[2]: prepare=0.38ms transformer=38.87ms depformer=16.41ms total=55.66ms
[TIMING] voice_prompt_load: 1.2ms
[TIMING] text_tokenize: 0.2ms (72 tokens)
[TIMING] lock_acquire: 0.0ms
[TIMING] streaming_reset: 3.8ms
[TIMING] voice_prompt_prefill: 2705.3ms (51 frames, 53.0ms/frame)
[TIMING] silence_1: 552.7ms (6 frames)
[TIMING] text_prompt_prefill: 4010.9ms (72 tokens, 55.7ms/token)
[TIMING] silence_2: 333.6ms (6 frames)
[TIMING] system_prompts_total: 7602.8ms
[TIMING] mimi_post_reset: 1.1ms
[TIMING] total_cold_start: 7621.1ms
```
