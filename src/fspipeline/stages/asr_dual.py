"""Stage 4: Dual ASR (faster-whisper + Qwen3-ASR) with similarity filtering."""

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

        device = self.cfg.whisper_device
        if device == "auto":
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"

        compute_type = self.cfg.whisper_compute_type
        if device == "cpu":
            compute_type = "int8"

        self.logger.info(f"Loading Whisper {self.cfg.whisper_model} on {device}")
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
        import torchaudio

        waveform, sr = torchaudio.load(audio_path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != 16000:
            waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)

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
        whisper_model = self._init_whisper()
        qwen_model, qwen_processor, qwen_device = self._init_qwen()

        valid_count = 0
        for seg in ctx.segments:
            audio_path = str(seg.audio_path)

            # Whisper transcription
            seg.transcript_whisper = self._transcribe_whisper(whisper_model, audio_path)

            # Qwen3-ASR transcription
            seg.transcript_qwen = self._transcribe_qwen(
                qwen_model, qwen_processor, qwen_device, audio_path
            )

            # Compute similarity
            seg.similarity_score = text_similarity(
                seg.transcript_whisper, seg.transcript_qwen
            )

            # Filter by threshold
            if seg.similarity_score >= self.cfg.similarity_threshold:
                seg.transcript_final = seg.transcript_whisper
                valid_count += 1
            else:
                seg.is_valid = False
                self.logger.debug(
                    f"{seg.id}: similarity={seg.similarity_score:.3f} < "
                    f"{self.cfg.similarity_threshold} (rejected)"
                )

        self.logger.info(
            f"ASR complete: {valid_count}/{len(ctx.segments)} segments passed "
            f"similarity threshold"
        )
        return ctx
