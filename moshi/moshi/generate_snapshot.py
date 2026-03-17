# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

"""
Standalone script to generate a KV cache snapshot for cold-start elimination.

Run once offline to produce snapshot files, then pass --snapshot-dir to the
server so sessions restore from the snapshot instead of re-running prefill.

Usage:
    python -m moshi.generate_snapshot \
        --voice-prompt /path/to/voices/NATF0.pt \
        --text-prompt "Your system prompt text here" \
        --output-dir ./snapshots \
        --device cuda
"""

import argparse
import hashlib
import json
import os
import time

from huggingface_hub import hf_hub_download
import sentencepiece
import torch

from .models import loaders
from .server import ServerState, wrap_with_system_tags


def main():
    parser = argparse.ArgumentParser(
        description="Generate a KV cache snapshot for PersonaPlex cold-start elimination."
    )
    parser.add_argument("--voice-prompt", type=str, required=True,
                        help="Path to voice prompt file (e.g. a .pt embeddings file or audio file).")
    parser.add_argument("--text-prompt", type=str, default="",
                        help="The system text prompt string.")
    parser.add_argument("--output-dir", type=str, default="./snapshots",
                        help="Directory to save snapshot files.")
    parser.add_argument("--hf-repo", type=str, default=loaders.DEFAULT_REPO,
                        help="HF repo for model weights.")
    parser.add_argument("--moshi-weight", type=str, default=None,
                        help="Optional local path to moshi weights.")
    parser.add_argument("--mimi-weight", type=str, default=None,
                        help="Optional local path to mimi weights.")
    parser.add_argument("--tokenizer", type=str, default=None,
                        help="Optional local path to tokenizer.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--label", type=str, default="",
                        help="Human-readable label for this scenario (shown in demo UI).")
    parser.add_argument("--description", type=str, default="",
                        help="Short description of the scenario (shown in demo UI).")
    parser.add_argument("--icon", type=str, default="",
                        help="Emoji icon for the scenario card.")

    args = parser.parse_args()
    device = torch.device(args.device)

    t_total_start = time.monotonic()

    print("Loading mimi...")
    if args.mimi_weight is None:
        args.mimi_weight = hf_hub_download(args.hf_repo, loaders.MIMI_NAME)
    mimi = loaders.get_mimi(args.mimi_weight, device)
    other_mimi = loaders.get_mimi(args.mimi_weight, device)
    print("mimi loaded.")

    if args.tokenizer is None:
        args.tokenizer = hf_hub_download(args.hf_repo, loaders.TEXT_TOKENIZER_NAME)
    text_tokenizer = sentencepiece.SentencePieceProcessor(args.tokenizer)

    print("Loading moshi...")
    if args.moshi_weight is None:
        args.moshi_weight = hf_hub_download(args.hf_repo, loaders.MOSHI_NAME)
    lm = loaders.get_moshi_lm(args.moshi_weight, device=device)
    lm.eval()
    print("moshi loaded.")

    state = ServerState(
        mimi=mimi,
        other_mimi=other_mimi,
        text_tokenizer=text_tokenizer,
        lm=lm,
        device=device,
        voice_prompt_dir=os.path.dirname(args.voice_prompt),
        save_voice_prompt_embeddings=False,
    )

    print("Warming up (creating CUDA graphs)...")
    state.warmup()

    voice_prompt_path = args.voice_prompt
    if voice_prompt_path.endswith('.pt'):
        state.lm_gen.load_voice_prompt_embeddings(voice_prompt_path)
    else:
        state.lm_gen.load_voice_prompt(voice_prompt_path)

    state.lm_gen.text_prompt_tokens = (
        text_tokenizer.encode(wrap_with_system_tags(args.text_prompt))
        if args.text_prompt else None
    )

    state.mimi.reset_streaming()
    state.other_mimi.reset_streaming()
    state.lm_gen.reset_streaming()

    print("Running prefill (this takes ~8 seconds)...")
    t_prefill_start = time.monotonic()
    state.lm_gen.step_system_prompts(state.mimi)
    t_prefill = time.monotonic() - t_prefill_start
    print(f"Prefill completed in {t_prefill * 1000:.1f}ms")

    state.mimi.reset_streaming()

    os.makedirs(args.output_dir, exist_ok=True)

    snapshot_path = os.path.join(args.output_dir, "snapshot.safetensors")
    metadata_path = os.path.join(args.output_dir, "snapshot_metadata.json")

    print("Saving snapshot...")
    t_save_start = time.monotonic()
    state.lm_gen.save_streaming_state(snapshot_path, metadata_path)
    t_save = time.monotonic() - t_save_start
    print(f"Snapshot saved in {t_save * 1000:.1f}ms")

    num_text_tokens = len(state.lm_gen.text_prompt_tokens) if state.lm_gen.text_prompt_tokens else 0
    config = {
        "voice_prompt": os.path.basename(voice_prompt_path),
        "text_prompt": args.text_prompt,
        "text_prompt_hash": hashlib.sha256(args.text_prompt.encode()).hexdigest(),
        "num_text_tokens": num_text_tokens,
        "offset_after_prefill": state.lm_gen._streaming_state.offset,
        "label": args.label,
        "description": args.description,
        "icon": args.icon,
    }
    config_path = os.path.join(args.output_dir, "snapshot_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    snapshot_size = os.path.getsize(snapshot_path) / (1024**3)
    t_total = time.monotonic() - t_total_start

    print(f"\nSnapshot saved to {args.output_dir}")
    print(f"  snapshot.safetensors: {snapshot_size:.2f} GB")
    print(f"  offset after prefill: {config['offset_after_prefill']}")
    print(f"  voice prompt: {config['voice_prompt']}")
    print(f"  text tokens: {config['num_text_tokens']}")
    print(f"  total time: {t_total * 1000:.1f}ms")


if __name__ == "__main__":
    with torch.no_grad():
        main()
