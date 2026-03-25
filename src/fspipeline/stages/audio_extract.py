"""Stage 0: Extract audio from video/audio using PyAV (no system ffmpeg required)."""

from __future__ import annotations

import av
import numpy as np
import soundfile as sf

from ..config import PipelineConfig
from ..models import PipelineContext
from .base import PipelineStage


class AudioExtractStage(PipelineStage):
    name = "audio_extract"

    def __init__(self, config: PipelineConfig):
        super().__init__(config)
        self.cfg = config.audio_extract

    def validate(self, ctx: PipelineContext) -> None:
        if not ctx.video_path.exists():
            raise FileNotFoundError(f"Video not found: {ctx.video_path}")

    def run(self, ctx: PipelineContext) -> PipelineContext:
        output_path = ctx.output_dir / "full_audio.wav"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        target_sr = self.cfg.sample_rate
        target_ch = self.cfg.channels

        self.logger.info(f"Extracting audio from {ctx.video_path} (PyAV)")

        container = av.open(str(ctx.video_path))
        audio_stream = next((s for s in container.streams if s.type == "audio"), None)
        if audio_stream is None:
            container.close()
            raise RuntimeError(f"No audio stream found in {ctx.video_path}")

        src_sr = audio_stream.rate

        # Decode all audio frames
        frames: list[np.ndarray] = []
        for frame in container.decode(audio_stream):
            arr = frame.to_ndarray().astype(np.float32)
            frames.append(arr)
        container.close()

        if not frames:
            raise RuntimeError(f"No audio frames decoded from {ctx.video_path}")

        # Concatenate → (channels, total_samples)
        audio = np.concatenate(frames, axis=-1)
        if audio.ndim == 1:
            audio = audio[np.newaxis, :]

        # Normalise if outside [-1, 1]
        peak = max(abs(audio.max()), abs(audio.min()))
        if peak > 1.0:
            audio = audio / peak

        # Downmix to target channels
        if target_ch == 1 and audio.shape[0] > 1:
            audio = audio.mean(axis=0, keepdims=True)

        # Resample if needed
        if src_sr != target_sr:
            import torch
            import torchaudio

            waveform = torch.from_numpy(audio)
            waveform = torchaudio.transforms.Resample(src_sr, target_sr)(waveform)
            audio = waveform.numpy()

        # Write WAV
        sf.write(str(output_path), audio.squeeze(), target_sr)

        ctx.full_audio_path = output_path
        self.logger.info(
            f"Audio extracted to {output_path} "
            f"({audio.shape[-1] / target_sr:.1f}s, {target_sr}Hz)"
        )
        return ctx
