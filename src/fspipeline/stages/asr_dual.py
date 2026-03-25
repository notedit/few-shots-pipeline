"""Stage 4: ASR transcription.

Supports two modes controlled by AsrConfig.whisper_only:
  - whisper_only=True  (default for this run): faster-whisper only; transcript_final
    is set directly from Whisper output without dual-model similarity filtering.
  - whisper_only=False: dual-model (faster-whisper + Qwen3-ASR) with cross-model
    similarity filtering as originally designed.

Only *valid* segments (is_valid=True) are transcribed; segments already marked
invalid by SpeakerFilterStage are skipped.
"""

from __future__ import annotations

from ..config import PipelineConfig
from ..models import PipelineContext
from ..utils.text import text_similarity
from .base import PipelineStage


class DualAsrStage(PipelineStage):
    name = "asr_dual"

    def __init__(self, config: PipelineConfig):
        super().__init__(config)
        self.cfg = config.asr

    def validate(self, ctx: PipelineContext) -> None:
        if not ctx.segments:
            raise ValueError("No segments found. Run vad_segment first.")

    def _init_whisper(self):
        from faster_whisper import WhisperModel
        import torch

        device = self.cfg.whisper_device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        compute_type = self.cfg.whisper_compute_type
        if device == "cpu":
            compute_type = "int8"

        self.logger.info(f"Loading Whisper {self.cfg.whisper_model} on {device} ({compute_type})")
        return WhisperModel(
            self.cfg.whisper_model,
            device=device,
            compute_type=compute_type,
        )

    def _init_qwen(self):
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32

        self.logger.info(f"Loading Qwen3-ASR: {self.cfg.qwen_model}")
        processor = AutoProcessor.from_pretrained(
            self.cfg.qwen_model,
            trust_remote_code=True,
        )
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.cfg.qwen_model,
            torch_dtype=dtype,
            trust_remote_code=True,
        ).to(device)
        return model, processor, device

    def _transcribe_whisper(self, whisper_model, audio_path: str) -> str:
        language = self.cfg.language if self.cfg.language != "auto" else None
        segments, _ = whisper_model.transcribe(
            audio_path,
            language=language,
            beam_size=5,
        )
        return " ".join(seg.text.strip() for seg in segments)

    def _transcribe_qwen(self, model, processor, device, audio_path: str) -> str:
        import torch
        from ..utils.audio import load_audio
        from pathlib import Path

        waveform, sr = load_audio(Path(audio_path), sample_rate=16000)

        inputs = processor(
            waveform.squeeze(0).numpy(),
            sampling_rate=16000,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=448)
        transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return transcription.strip()

    def run(self, ctx: PipelineContext) -> PipelineContext:
        whisper_only = getattr(self.cfg, "whisper_only", True)

        whisper_model = self._init_whisper()
        qwen_model = qwen_processor = qwen_device = None
        if not whisper_only:
            qwen_model, qwen_processor, qwen_device = self._init_qwen()

        valid_segs = [s for s in ctx.segments if s.is_valid]
        self.logger.info(
            f"Transcribing {len(valid_segs)} valid segments "
            f"({'whisper-only' if whisper_only else 'dual-model'} mode)"
        )

        for i, seg in enumerate(valid_segs):
            audio_path = str(seg.audio_path)

            seg.transcript_whisper = self._transcribe_whisper(whisper_model, audio_path)

            if whisper_only:
                # Single-model mode: accept all Whisper results directly
                seg.transcript_final = seg.transcript_whisper
                seg.similarity_score = 1.0
                if (i + 1) % 10 == 0 or i == 0:
                    self.logger.info(
                        f"[{i+1}/{len(valid_segs)}] {seg.id}: {seg.transcript_whisper[:60]!r}"
                    )
            else:
                # Dual-model mode: cross-check with Qwen
                seg.transcript_qwen = self._transcribe_qwen(
                    qwen_model, qwen_processor, qwen_device, audio_path
                )
                seg.similarity_score = text_similarity(
                    seg.transcript_whisper, seg.transcript_qwen
                )
                if seg.similarity_score >= self.cfg.similarity_threshold:
                    seg.transcript_final = seg.transcript_whisper
                else:
                    seg.is_valid = False
                    self.logger.debug(
                        f"{seg.id}: dual-ASR similarity={seg.similarity_score:.3f} "
                        f"< {self.cfg.similarity_threshold} (rejected)"
                    )

        final_valid = sum(1 for s in ctx.segments if s.is_valid)
        self.logger.info(
            f"ASR complete: {final_valid}/{len(ctx.segments)} segments valid"
        )
        return ctx
