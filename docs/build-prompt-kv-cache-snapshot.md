# Build Prompt: KV Cache Snapshot/Restore for Cold Start Elimination

## Goal

Eliminate the 7.6-second session cold start in PersonaPlex by implementing KV cache snapshot/restore. Instead of running 135 sequential LM forward passes (prefill) every time a user connects, pre-compute the KV cache once, save it, and restore it via tensor copy on each session.

## Background (What You Need to Know)

### The problem
When a user connects via WebSocket, the server runs `step_system_prompts_async()` which feeds voice prompt (51 frames) + silence (6 frames) + text prompt (72 tokens) + silence (6 frames) = 135 sequential forward passes through a 7B transformer at ~56ms each = 7.6 seconds. The user waits this entire time before hearing anything.

### The solution
The codebase already has `save_streaming_state()`, `load_streaming_state()`, and `set_streaming_state_inplace()` in `moshi/moshi/modules/streaming.py`. We need to:
1. Run the prefill once and save the resulting KV cache state to disk
2. At server startup, load that snapshot into CPU RAM
3. On each session, copy the snapshot into the live KV cache instead of re-running prefill

### Key architecture facts
- CUDA graphs are created during `warmup()` and persist across sessions. `reset_streaming()` calls `_LMGenState.reset()` which only zeroes `offset` and `provided` -- it does NOT touch `graphed_main`, `graphed_embeddings`, or `graphed_depth`. So after snapshot restore, `lm_gen.step()` runs at full graphed speed (~56ms) immediately.
- Mimi state is reset after prefill (`mimi.reset_streaming()` on line 307 of server.py), so Mimi state is NOT part of the snapshot.
- The depformer uses `set_streaming_propagate(False)`, so it is excluded from the streaming state tree. This is correct -- depformer state is created fresh per step.
- `CUDAGraphed.asdict()` returns `{}`, so CUDAGraphed objects are automatically excluded from save/load.
- The snapshot is ~1.5 GB (32 layers x 49 MB per layer of KV cache in bfloat16).
- `set_streaming_state_inplace()` uses `tensor.copy_()` to overwrite the EXISTING KV cache tensors in GPU VRAM. No additional GPU memory is needed -- the snapshot sits in CPU RAM and copies into the already-allocated GPU tensors.

---

## Files to Modify

### 1. `moshi/moshi/server.py` (main changes)

### 2. Create new file: `moshi/moshi/generate_snapshot.py` (snapshot generation script)

---

## Detailed Instructions

### Task 1: Create `moshi/moshi/generate_snapshot.py`

Create a standalone script that generates a KV cache snapshot for a given (voice_prompt, text_prompt) pair. This script will be run once offline to produce the snapshot files.

The script should:

1. Accept command-line arguments:
   - `--voice-prompt` (path to voice prompt file, e.g. a `.pt` file)
   - `--text-prompt` (the text prompt string)
   - `--output-dir` (directory to save snapshot files, default `./snapshots`)
   - `--hf-repo` (HF repo for model weights, default `nvidia/personaplex-7b-v1`)
   - `--moshi-weight` (optional local path to moshi weights)
   - `--mimi-weight` (optional local path to mimi weights)
   - `--tokenizer` (optional local path to tokenizer)
   - `--device` (default `cuda`)

2. Load the model exactly like `main()` in `server.py` does (lines 458-485):
   ```python
   mimi = loaders.get_mimi(mimi_weight, device)
   other_mimi = loaders.get_mimi(mimi_weight, device)  # needed for voice prompt processing
   text_tokenizer = sentencepiece.SentencePieceProcessor(tokenizer_path)
   lm = loaders.get_moshi_lm(moshi_weight, device=device)
   lm.eval()
   ```

3. Create a `ServerState` and run warmup:
   ```python
   state = ServerState(mimi=mimi, other_mimi=other_mimi, text_tokenizer=text_tokenizer,
                       lm=lm, device=device, voice_prompt_dir=os.path.dirname(voice_prompt))
   state.warmup()
   ```

4. Set voice prompt and text prompt on lm_gen (mirror the logic in `handle_chat()` lines 167-180):
   ```python
   voice_prompt_path = args.voice_prompt
   if voice_prompt_path.endswith('.pt'):
       state.lm_gen.load_voice_prompt_embeddings(voice_prompt_path)
   else:
       state.lm_gen.load_voice_prompt(voice_prompt_path)
   
   state.lm_gen.text_prompt_tokens = text_tokenizer.encode(
       wrap_with_system_tags(args.text_prompt)
   ) if args.text_prompt else None
   ```

5. Reset streaming and run the full prefill:
   ```python
   state.mimi.reset_streaming()
   state.other_mimi.reset_streaming()
   state.lm_gen.reset_streaming()
   
   # Run prefill synchronously (use step_system_prompts, not the async version)
   state.lm_gen.step_system_prompts(state.mimi)
   
   # Reset mimi after prefill (same as server.py line 307)
   state.mimi.reset_streaming()
   ```

6. Save the LMGen streaming state:
   ```python
   import os, json, hashlib
   os.makedirs(args.output_dir, exist_ok=True)
   
   snapshot_path = os.path.join(args.output_dir, "snapshot.safetensors")
   metadata_path = os.path.join(args.output_dir, "snapshot_metadata.json")
   
   state.lm_gen.save_streaming_state(snapshot_path, metadata_path)
   ```

7. Save a config file so we know what this snapshot is for:
   ```python
   config = {
       "voice_prompt": os.path.basename(voice_prompt_path),
       "text_prompt": args.text_prompt,
       "text_prompt_hash": hashlib.sha256(args.text_prompt.encode()).hexdigest(),
       "num_text_tokens": len(state.lm_gen.text_prompt_tokens) if state.lm_gen.text_prompt_tokens else 0,
       "offset_after_prefill": state.lm_gen._streaming_state.offset,
   }
   config_path = os.path.join(args.output_dir, "snapshot_config.json")
   with open(config_path, "w") as f:
       json.dump(config, f, indent=2)
   ```

8. Print timing and file sizes:
   ```python
   snapshot_size = os.path.getsize(snapshot_path) / (1024**3)
   print(f"Snapshot saved to {args.output_dir}")
   print(f"  snapshot.safetensors: {snapshot_size:.2f} GB")
   print(f"  offset after prefill: {config['offset_after_prefill']}")
   print(f"  voice prompt: {config['voice_prompt']}")
   print(f"  text tokens: {config['num_text_tokens']}")
   ```

9. Wrap everything in `with torch.no_grad():` just like server.py line 509.

**Important:** Use `step_system_prompts` (sync version, line 1176 of lm.py), NOT `step_system_prompts_async`. The sync version exists and doesn't need an event loop.

---

### Task 2: Modify `ServerState.__init__()` in `server.py`

Add snapshot loading capability to the server.

1. Add a new parameter `snapshot_dir: str | None = None` to `ServerState.__init__()`.

2. After the existing `streaming_forever()` calls (line 118) and before the method ends, add snapshot loading:

```python
self._cached_snapshot = None
self._snapshot_config = None

if snapshot_dir is not None:
    import time as _time
    snapshot_path = os.path.join(snapshot_dir, "snapshot.safetensors")
    metadata_path = os.path.join(snapshot_dir, "snapshot_metadata.json")
    config_path = os.path.join(snapshot_dir, "snapshot_config.json")
    
    if os.path.exists(snapshot_path) and os.path.exists(metadata_path):
        t0 = _time.monotonic()
        from .modules.streaming import load_streaming_state
        self._cached_snapshot = load_streaming_state(snapshot_path, metadata_path, device='cpu')
        # Pin tensors in CPU RAM for faster GPU transfer
        self._cached_snapshot = {
            k: v.pin_memory() if isinstance(v, torch.Tensor) else v
            for k, v in self._cached_snapshot.items()
        }
        t_load = _time.monotonic() - t0
        logger.info(f"[SNAPSHOT] Loaded snapshot from {snapshot_dir} in {t_load*1000:.1f}ms")
        
        if os.path.exists(config_path):
            with open(config_path) as f:
                self._snapshot_config = json.load(f)
            logger.info(f"[SNAPSHOT] Config: voice={self._snapshot_config.get('voice_prompt')}, "
                        f"text_tokens={self._snapshot_config.get('num_text_tokens')}, "
                        f"offset={self._snapshot_config.get('offset_after_prefill')}")
    else:
        logger.warning(f"[SNAPSHOT] Snapshot files not found in {snapshot_dir}, will use full prefill")
```

3. Add `import json` at the top of server.py if not already present.

---

### Task 3: Modify `handle_chat()` in `server.py`

Replace the prefill section with snapshot restore when a matching snapshot is available.

Find this block (lines 281-309):
```python
t0 = time.monotonic()
self.mimi.reset_streaming()
self.other_mimi.reset_streaming()
self.lm_gen.reset_streaming()
t_reset = time.monotonic() - t0
logger.info(f"[TIMING] streaming_reset: {t_reset * 1000:.1f}ms")

# ... is_alive function ...

t0 = time.monotonic()
await self.lm_gen.step_system_prompts_async(self.mimi, is_alive=is_alive)
t_system_prompts = time.monotonic() - t0
logger.info(f"[TIMING] system_prompts_total: {t_system_prompts * 1000:.1f}ms")

t0 = time.monotonic()
self.mimi.reset_streaming()
t_mimi_reset = time.monotonic() - t0
logger.info(f"[TIMING] mimi_post_reset: {t_mimi_reset * 1000:.1f}ms")
```

Replace with:
```python
t0 = time.monotonic()
self.mimi.reset_streaming()
self.other_mimi.reset_streaming()
self.lm_gen.reset_streaming()
t_reset = time.monotonic() - t0
logger.info(f"[TIMING] streaming_reset: {t_reset * 1000:.1f}ms")

# ... is_alive function stays exactly where it is ...

if self._cached_snapshot is not None:
    # Fast path: restore from snapshot
    t0 = time.monotonic()
    snapshot_copy = {k: v.clone() if isinstance(v, torch.Tensor) else v
                     for k, v in self._cached_snapshot.items()}
    t_clone = time.monotonic() - t0
    logger.info(f"[TIMING] snapshot_clone: {t_clone * 1000:.1f}ms")
    
    t0 = time.monotonic()
    self.lm_gen.set_streaming_state_inplace(snapshot_copy)
    t_restore = time.monotonic() - t0
    logger.info(f"[TIMING] snapshot_restore: {t_restore * 1000:.1f}ms")
    
    t0 = time.monotonic()
    self.mimi.reset_streaming()
    t_mimi_reset = time.monotonic() - t0
    logger.info(f"[TIMING] mimi_post_reset: {t_mimi_reset * 1000:.1f}ms")
    
    logger.info(f"[TIMING] total_cold_start_snapshot: {(time.monotonic() - t_cold_start) * 1000:.1f}ms")
else:
    # Slow path: full prefill (fallback)
    t0 = time.monotonic()
    await self.lm_gen.step_system_prompts_async(self.mimi, is_alive=is_alive)
    t_system_prompts = time.monotonic() - t0
    logger.info(f"[TIMING] system_prompts_total: {t_system_prompts * 1000:.1f}ms")
    
    t0 = time.monotonic()
    self.mimi.reset_streaming()
    t_mimi_reset = time.monotonic() - t0
    logger.info(f"[TIMING] mimi_post_reset: {t_mimi_reset * 1000:.1f}ms")
```

**Important:** The `is_alive` async function definition (lines 288-298) must stay between `reset_streaming()` and the prefill/restore block. Do not move or remove it -- it's still needed for the fallback path.

---

### Task 4: Add `--snapshot-dir` CLI argument in `main()`

In the `main()` function, add a new argument after the `--voice-prompt-dir` argument (around line 411):

```python
parser.add_argument(
    "--snapshot-dir",
    type=str,
    default=None,
    help=(
        "Directory containing pre-computed KV cache snapshot files "
        "(snapshot.safetensors, snapshot_metadata.json, snapshot_config.json). "
        "If provided, sessions will restore from snapshot instead of running full prefill."
    )
)
```

Then pass it to `ServerState` (around line 475):
```python
state = ServerState(
    mimi=mimi,
    other_mimi=other_mimi,
    text_tokenizer=text_tokenizer,
    lm=lm,
    device=args.device,
    voice_prompt_dir=args.voice_prompt_dir,
    save_voice_prompt_embeddings=False,
    snapshot_dir=args.snapshot_dir,  # ADD THIS
)
```

---

## Testing Instructions

### Step 1: Generate a snapshot

First, find the voice prompt and text prompt used in testing. From the timing report:
- Voice prompt: `NATF0.pt` (in the voice_prompt_dir)
- Text prompt: the medical office intake scenario text

Run:
```bash
cd /opt/personaplex
python3.11 -m moshi.generate_snapshot \
    --voice-prompt /path/to/voices/NATF0.pt \
    --text-prompt "Your text prompt here" \
    --output-dir ./snapshots \
    --device cuda
```

This will take ~10 seconds (7.6s prefill + model load). It produces:
- `snapshots/snapshot.safetensors` (~1.5 GB)
- `snapshots/snapshot_metadata.json` (small JSON)
- `snapshots/snapshot_config.json` (config for matching)

### Step 2: Start the server with snapshot

```bash
python3.11 -m moshi.server \
    --host 0.0.0.0 \
    --ssl /path/to/ssl \
    --snapshot-dir ./snapshots
```

You should see in the logs:
```
[SNAPSHOT] Loaded snapshot from ./snapshots in XXXms
[SNAPSHOT] Config: voice=NATF0.pt, text_tokens=72, offset=135
```

### Step 3: Connect from browser and check timing

Connect with the SAME voice prompt and text prompt that was used to generate the snapshot. In the server logs, you should see:

```
[TIMING] streaming_reset: ~4ms
[TIMING] snapshot_clone: ~XXms
[TIMING] snapshot_restore: ~XXms
[TIMING] mimi_post_reset: ~1ms
[TIMING] total_cold_start_snapshot: ~XXXms    <-- should be WAY less than 7621ms
```

If you connect with a DIFFERENT voice/text prompt, the server uses the snapshot anyway (no config matching in this initial version). For correctness, the voice prompt and text prompt used at connection time should match what was used to generate the snapshot.

---

## What NOT to Change

- Do NOT modify `moshi/moshi/modules/streaming.py` -- the save/load/restore infrastructure already works.
- Do NOT modify `moshi/moshi/models/lm.py` -- the streaming state structure is correct as-is.
- Do NOT remove the existing timing instrumentation in `handle_chat()` or `step_system_prompts_async()`.
- Do NOT change the `warmup()` method -- CUDA graphs must still be created at startup.
- Do NOT try to save/restore Mimi state -- it's reset after prefill and not needed.

## Key Code References

### Existing save/load API (`moshi/moshi/modules/streaming.py`)

- `save_streaming_state(save_path, metadata_save_path)` -- line 367. Calls `_flatten_streaming_state()` to serialize all tensors to safetensors + metadata to JSON.
- `load_streaming_state(path, metadata_path, device='cpu')` -- line 232. Returns a flat dict of tensors + metadata.
- `set_streaming_state_inplace(state_dict)` -- line 393. Walks the streaming module tree and copies each tensor via `copy_()`. Raises `RuntimeError` if any keys in state_dict are not consumed (safety check).

### State that gets saved (`moshi/moshi/models/lm.py`)

`_LMGenState` dataclass (line 557):
```python
@dataclass
class _LMGenState:
    cache: torch.Tensor        # [B, 17, max_delay+3], long
    provided: torch.Tensor     # [B, 17, max_delay+3], bool
    initial: torch.Tensor      # [1, 17, 1], long
    graphed_main: CUDAGraphed  # excluded (asdict returns {})
    graphed_embeddings: CUDAGraphed  # excluded
    graphed_depth: CUDAGraphed       # excluded
    offset: int = 0            # saved as metadata
```

`_LMGenState.reset()` (line 566):
```python
def reset(self):
    self.offset = 0
    self.provided[:] = False
```
Note: `reset()` does NOT touch graphed_main/graphed_embeddings/graphed_depth. This is why CUDA graphs persist and snapshot restore works.

### KV cache structure (`moshi/moshi/modules/transformer.py`)

Each attention layer has a `RingKVCache` with:
- `cache`: tensor `[2, B, num_heads, capacity, dim_per_head]` = `[2, 1, 32, 3000, 128]` bfloat16
- `end_offset`: tensor `[1]` long

32 transformer layers x 49 MB each = ~1.5 GB total.

### Server flow (`moshi/moshi/server.py`)

Current session startup (lines 281-309):
1. `reset_streaming()` -- zeros KV caches (~4ms)
2. `step_system_prompts_async()` -- 135 LM steps (~7,600ms) <-- THIS IS WHAT WE SKIP
3. `mimi.reset_streaming()` -- reset audio codec (~1ms)
4. Send handshake byte `b"\x00"`

After our change:
1. `reset_streaming()` -- zeros KV caches (~4ms)
2. Clone snapshot from CPU RAM
3. `set_streaming_state_inplace()` -- copy into GPU KV cache
4. `mimi.reset_streaming()` -- reset audio codec (~1ms)
5. Send handshake byte `b"\x00"`

### Sync prefill method (`moshi/moshi/models/lm.py` line 1176)

```python
def step_system_prompts(self, mimi):
    self._step_voice_prompt(mimi)
    self._step_audio_silence()
    self._step_text_prompt()
    self._step_audio_silence()
```

Use this in `generate_snapshot.py` instead of the async version.
