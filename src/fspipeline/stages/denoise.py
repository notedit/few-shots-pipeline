"""Stage 2: Speech enhancement using MossFormer2 via SpeechBrain."""

from __future__ import annotations

import torch
import torchaudio

from ..config import PipelineConfig
from ..models import PipelineContext
from ..utils.audio import save_audio
from .base import PipelineStage


class DenoiseStage(PipelineStage):
    name = "denoise"

    def __init__(self, config: PipelineConfig):
        super().__init__(config)
        self.cfg = config.denoise

    def validate(self, ctx: PipelineContext) -> None:
        audio_path = ctx.speaker_audio_path or ctx.full_audio_path
        if audio_path is None or not audio_path.exists():
            raise FileNotFoundError("No audio to denoise. Run previous stages first.")

    def run(self, ctx: PipelineContext) -> PipelineContext:
        from speechbrain.inference.separation import SepformerSeparation

        input_path = ctx.speaker_audio_path or ctx.full_audio_path
        output_path = ctx.output_dir / "denoised_audio.wav"

        self.logger.info(f"Loading denoising model: {self.cfg.model_source}")
        model = SepformerSeparation.from_hparams(
            source=self.cfg.model_source,
            savedir=str(ctx.output_dir / "pretrained_models" / "denoise"),
        )

        # Load audio
        waveform, sr = torchaudio.load(str(input_path))
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Process in chunks to control VRAM usage
        chunk_samples = int(self.cfg.chunk_seconds * sr)
        total_samples = waveform.shape[1]
        enhanced_chunks = []

        self.logger.info(
            f"Denoising {total_samples / sr:.1f}s audio in "
            f"{self.cfg.chunk_seconds}s chunks"
        )

        for start in range(0, total_samples, chunk_samples):
            end = min(start + chunk_samples, total_samples)
            chunk = waveform[:, start:end]

            with torch.no_grad():
                est_sources = model.separate_batch(chunk.unsqueeze(0))
                # Take the first (enhanced) source
                enhanced = est_sources[:, :, 0].squeeze(0)

            enhanced_chunks.append(enhanced)

        enhanced_audio = torch.cat(enhanced_chunks, dim=-1)
        if enhanced_audio.ndim == 1:
            enhanced_audio = enhanced_audio.unsqueeze(0)

        save_audio(enhanced_audio, output_path, sr)
        ctx.denoised_audio_path = output_path
        self.logger.info(f"Denoised audio saved to {output_path}")
        return ctx
