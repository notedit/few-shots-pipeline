#!/usr/bin/env python3
"""Pre-download all models used in the pipeline."""

from __future__ import annotations

import sys


def main():
    print("=== Downloading pipeline models ===\n")

    # 1. Silero VAD
    print("[1/4] Silero VAD...")
    import torch
    torch.hub.load("snakers4/silero-vad", "silero_vad", trust_repo=True)
    print("  OK\n")

    # 2. faster-whisper large-v3
    print("[2/4] faster-whisper large-v3...")
    from faster_whisper import WhisperModel
    WhisperModel("large-v3", device="cpu", compute_type="int8")
    print("  OK\n")

    # 3. Qwen3-ASR
    print("[3/4] Qwen3-ASR-1.7B...")
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
    AutoProcessor.from_pretrained("Qwen/Qwen3-ASR-1.7B", trust_remote_code=True)
    AutoModelForSpeechSeq2Seq.from_pretrained("Qwen/Qwen3-ASR-1.7B", trust_remote_code=True)
    print("  OK\n")

    # 4. SpeechBrain denoising
    print("[4/4] SpeechBrain sepformer-dns4-16k-enhancement...")
    from speechbrain.inference.separation import SepformerSeparation
    SepformerSeparation.from_hparams(
        source="speechbrain/sepformer-dns4-16k-enhancement",
        savedir="pretrained_models/denoise",
    )
    print("  OK\n")

    print("=== All models downloaded ===")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
