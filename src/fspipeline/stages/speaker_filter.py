"""Stage 1b: Per-segment speaker similarity filtering.

Computes a speaker embedding for every VAD segment and compares it against
the reference (target) audio embedding.  Segments whose cosine similarity
falls below *similarity_threshold* are marked invalid and will be excluded
from downstream stages.

This stage is lighter-weight than full diarization: it never needs to load
audio via torchaudio/torchcodec – it uses the project's own load_audio()
helper which falls back to soundfile / PyAV.
"""

from __future__ import annotations

import numpy as np

from ..config import PipelineConfig
from ..models import PipelineContext
from ..utils.audio import load_audio
from .base import PipelineStage


class SpeakerFilterStage(PipelineStage):
    name = "speaker_filter"

    def __init__(self, config: PipelineConfig):
        super().__init__(config)
        self.cfg = config.speaker_extract  # reuse SpeakerExtractConfig fields

    def validate(self, ctx: PipelineContext) -> None:
        if not ctx.segments:
            raise ValueError("No segments found. Run vad_segment first.")
        if not ctx.reference_audio_path.exists():
            raise FileNotFoundError(
                f"Reference audio not found: {ctx.reference_audio_path}"
            )

    def _load_embedding_model(self):
        import torch
        from pyannote.audio import Inference, Model

        hf_token = self.cfg.hf_token or None
        self.logger.info(f"Loading embedding model: {self.cfg.embedding_model}")
        model_checkpoint = Model.from_pretrained(
            self.cfg.embedding_model,
            token=hf_token,
        )
        # Use CPU for embedding to avoid CUDA driver compatibility issues;
        # the model is small enough that CPU inference is fast.
        emb_model = Inference(model_checkpoint, window="whole")
        return emb_model

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        a, b = a.flatten(), b.flatten()
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

    def run(self, ctx: PipelineContext) -> PipelineContext:
        emb_model = self._load_embedding_model()

        # Reference embedding
        ref_waveform, ref_sr = load_audio(ctx.reference_audio_path, sample_rate=16000)
        ref_emb = np.array(
            emb_model({"waveform": ref_waveform, "sample_rate": ref_sr})
        ).flatten()
        self.logger.info(
            f"Reference embedding computed from {ctx.reference_audio_path}"
        )

        threshold = self.cfg.similarity_threshold
        valid_count = 0
        total = len(ctx.segments)

        for i, seg in enumerate(ctx.segments):
            waveform, sr = load_audio(seg.audio_path, sample_rate=16000)
            emb = np.array(
                emb_model({"waveform": waveform, "sample_rate": sr})
            ).flatten()
            score = self._cosine(ref_emb, emb)
            seg.speaker_score = score

            if score >= threshold:
                valid_count += 1
            else:
                seg.is_valid = False

            if (i + 1) % 50 == 0 or i == 0 or i == total - 1:
                self.logger.info(
                    f"[{i+1}/{total}] {seg.id}: score={score:.4f} "
                    f"valid_so_far={valid_count}"
                )

        total = len(ctx.segments)
        self.logger.info(
            f"Speaker filter done: {valid_count}/{total} segments kept "
            f"(threshold={threshold}, {valid_count/total*100:.1f}%)"
        )
        return ctx
