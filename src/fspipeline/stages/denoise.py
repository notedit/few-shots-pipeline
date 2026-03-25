"""Stage 2: Speech enhancement using MossFormer2 via ClearerVoice-Studio.

Supports two modes, selected automatically:

  Full-audio mode (no segments yet):
    Called before VadSegmentStage.  Enhances ctx.full_audio_path in one
    pass and saves the result as output_dir/full_audio_denoised.wav.
    ctx.full_audio_path is updated to point at the new file so that
    downstream VadSegmentStage reads the clean audio.

  Per-segment mode (segments present):
    Called after VadSegmentStage / SpeakerFilterStage.  Enhances each
    is_valid segment and writes to output_dir/segments_denoised/.
    seg.audio_path is updated to point at the denoised file.

Model: MossFormer2_SE_48K (48 kHz).
Output is always peak-normalised to -1 dBFS to prevent clipping.
"""

from __future__ import annotations

import numpy as np
import soundfile as sf

from ..config import PipelineConfig
from ..models import PipelineContext
from .base import PipelineStage


class DenoiseStage(PipelineStage):
    name = "denoise"

    def __init__(self, config: PipelineConfig):
        super().__init__(config)
        self.cfg = config.denoise

    def validate(self, ctx: PipelineContext) -> None:
        valid_segs = [s for s in ctx.segments if s.is_valid]
        if valid_segs:
            # Per-segment mode — segments must have audio files
            missing = [s for s in valid_segs if not s.audio_path.exists()]
            if missing:
                raise FileNotFoundError(
                    f"{len(missing)} segment audio files not found, "
                    f"e.g. {missing[0].audio_path}"
                )
        else:
            # Full-audio mode — full_audio_path must exist
            if ctx.full_audio_path is None or not ctx.full_audio_path.exists():
                raise FileNotFoundError(
                    "No full_audio_path found. Run audio_extract first."
                )

    def _load_model(self):
        from clearvoice import ClearVoice

        model_name = getattr(self.cfg, "model_name", "MossFormer2_SE_48K")
        self.logger.info(f"Loading denoising model: {model_name}")
        return ClearVoice(task="speech_enhancement", model_names=[model_name])

    @staticmethod
    def _normalize(arr: np.ndarray) -> np.ndarray:
        """Peak-normalise to -1 dBFS (≈ 0.891) to prevent clipping."""
        arr = np.array(arr, dtype=np.float32).flatten()
        peak = np.abs(arr).max()
        if peak > 1e-6:
            arr = arr / peak * 0.891
        return arr

    def _denoise_file(self, model, input_path: str) -> np.ndarray:
        """Run MossFormer2 on a file path, return normalised float32 array."""
        out = model(input_path=input_path, online_write=False)
        if isinstance(out, dict):
            out = next(iter(out.values()))
        return self._normalize(out)

    # ------------------------------------------------------------------
    # Full-audio mode
    # ------------------------------------------------------------------
    def _run_full_audio(self, ctx: PipelineContext, model) -> PipelineContext:
        input_path = str(ctx.full_audio_path)
        out_path = ctx.output_dir / "full_audio_denoised.wav"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        self.logger.info(
            f"Full-audio denoise: {ctx.full_audio_path.name} → {out_path.name}"
        )
        enhanced = self._denoise_file(model, input_path)
        sf.write(str(out_path), enhanced, 48000)

        ctx.full_audio_path = out_path
        self.logger.info(
            f"Full-audio denoise complete → {out_path} "
            f"({len(enhanced)/48000:.1f}s)"
        )
        return ctx

    # ------------------------------------------------------------------
    # Per-segment mode
    # ------------------------------------------------------------------
    def _run_per_segment(self, ctx: PipelineContext, model) -> PipelineContext:
        denoised_dir = ctx.output_dir / "segments_denoised"
        denoised_dir.mkdir(parents=True, exist_ok=True)

        valid_segs = [s for s in ctx.segments if s.is_valid]
        self.logger.info(
            f"Per-segment denoise: {len(valid_segs)} segments → {denoised_dir}"
        )

        for i, seg in enumerate(valid_segs):
            enhanced = self._denoise_file(model, str(seg.audio_path))
            out_path = denoised_dir / seg.audio_path.name
            sf.write(str(out_path), enhanced, 48000)
            seg.audio_path = out_path

            if (i + 1) % 10 == 0 or i == 0 or i == len(valid_segs) - 1:
                self.logger.info(f"  [{i+1}/{len(valid_segs)}] {seg.id}")

        self.logger.info(
            f"Per-segment denoise complete: {len(valid_segs)} segments enhanced"
        )
        return ctx

    # ------------------------------------------------------------------
    def run(self, ctx: PipelineContext) -> PipelineContext:
        model = self._load_model()
        valid_segs = [s for s in ctx.segments if s.is_valid]
        if valid_segs:
            return self._run_per_segment(ctx, model)
        else:
            return self._run_full_audio(ctx, model)
