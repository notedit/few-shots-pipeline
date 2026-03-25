"""Stage 0: Extract audio from video using FFmpeg."""

from __future__ import annotations

import ffmpeg

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

        self.logger.info(f"Extracting audio from {ctx.video_path}")
        (
            ffmpeg.input(str(ctx.video_path))
            .output(
                str(output_path),
                ac=self.cfg.channels,
                ar=self.cfg.sample_rate,
                format="wav",
            )
            .overwrite_output()
            .run(quiet=True)
        )

        ctx.full_audio_path = output_path
        self.logger.info(f"Audio extracted to {output_path}")
        return ctx
