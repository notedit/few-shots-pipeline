"""Stage: Post-denoise VAD trim.

Re-runs VAD on each denoised segment to strip leading/trailing silence,
then re-pads to exactly *padding_ms* milliseconds on each side.

This is much lighter than the initial VadSegmentStage because:
  - Operates on short, already-cleaned segments (not the full audio)
  - Uses the silero-vad pip package (v6+) — no torch.hub network call

MossFormer2_SE_48K outputs 48 kHz audio.  Silero VAD only accepts
8 kHz or 16 kHz, so we downsample to 16 kHz for detection but trim and
re-save the original 48 kHz waveform.

Segments that contain no detectable speech after trim (e.g. pure noise)
are marked is_valid=False.
"""

from __future__ import annotations

import numpy as np
import soundfile as sf
import torch

from ..config import PipelineConfig
from ..models import PipelineContext
from .base import PipelineStage

# Silero VAD native rate for detection
_VAD_SR = 16_000


class VadTrimStage(PipelineStage):
    name = "vad_trim"

    def __init__(self, config: PipelineConfig):
        super().__init__(config)
        self.cfg = config.vad
        self.padding_ms: int = getattr(config.vad, "trim_padding_ms", 200)

    def validate(self, ctx: PipelineContext) -> None:
        valid = [s for s in ctx.segments if s.is_valid]
        if not valid:
            raise ValueError("No valid segments to trim. Run denoise first.")
        missing = [s for s in valid if not s.audio_path.exists()]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} denoised segment files not found, e.g. {missing[0].audio_path}"
            )

    def _load_vad(self):
        from silero_vad import load_silero_vad, get_speech_timestamps
        self.logger.info("Loading Silero VAD (pip v6)...")
        model = load_silero_vad()
        return model, get_speech_timestamps

    @staticmethod
    def _resample_to_16k(waveform: np.ndarray, src_sr: int) -> torch.Tensor:
        """Downsample a (N,) float32 numpy array to 16 kHz torch.Tensor."""
        if src_sr == _VAD_SR:
            return torch.from_numpy(waveform)
        import torchaudio.transforms as T
        wav_t = torch.from_numpy(waveform).unsqueeze(0)  # (1, N)
        resampler = T.Resample(orig_freq=src_sr, new_freq=_VAD_SR)
        return resampler(wav_t).squeeze(0)  # (N',)

    def run(self, ctx: PipelineContext) -> PipelineContext:
        vad_model, get_speech_timestamps = self._load_vad()

        trimmed_dir = ctx.output_dir / "segments_trimmed"
        trimmed_dir.mkdir(parents=True, exist_ok=True)

        valid_segs = [s for s in ctx.segments if s.is_valid]
        self.logger.info(
            f"VAD-trimming {len(valid_segs)} segments "
            f"(padding={self.padding_ms}ms) → {trimmed_dir}"
        )

        padding_ms = self.padding_ms
        trimmed_count = 0
        silence_count = 0

        for i, seg in enumerate(valid_segs):
            # Load denoised file at native sample rate (48 kHz for MossFormer2)
            data, src_sr = sf.read(str(seg.audio_path), dtype="float32")
            if data.ndim > 1:
                data = data.mean(axis=1)  # stereo → mono

            # Downsample to 16 kHz for VAD detection
            wav_16k = self._resample_to_16k(data, src_sr)

            # Run VAD at 16 kHz
            timestamps = get_speech_timestamps(
                wav_16k,
                vad_model,
                threshold=self.cfg.threshold,
                min_speech_duration_ms=self.cfg.min_speech_duration_ms,
                min_silence_duration_ms=self.cfg.min_silence_duration_ms,
                sampling_rate=_VAD_SR,
                return_seconds=False,  # sample indices at _VAD_SR
            )

            if not timestamps:
                seg.is_valid = False
                silence_count += 1
                self.logger.debug(f"  {seg.id}: no speech detected → marked invalid")
                continue

            # Speech window in original sample rate
            ratio = src_sr / _VAD_SR
            speech_start_orig = int(timestamps[0]["start"] * ratio)
            speech_end_orig = int(timestamps[-1]["end"] * ratio)

            # Padding in original sample rate
            pad_samples = int(padding_ms * src_sr / 1000)
            trim_start = max(0, speech_start_orig - pad_samples)
            trim_end = min(len(data), speech_end_orig + pad_samples)

            trimmed = data[trim_start:trim_end]

            # Save trimmed file (overwrite or new path)
            out_path = trimmed_dir / seg.audio_path.name
            sf.write(str(out_path), trimmed, src_sr)
            seg.audio_path = out_path
            trimmed_count += 1

            if (i + 1) % 10 == 0 or i == 0 or i == len(valid_segs) - 1:
                saved_ms = round(len(trimmed) / src_sr * 1000)
                orig_ms = round(len(data) / src_sr * 1000)
                self.logger.info(
                    f"  [{i+1}/{len(valid_segs)}] {seg.id}: "
                    f"{orig_ms}ms → {saved_ms}ms"
                )

        self.logger.info(
            f"VAD trim complete: {trimmed_count} trimmed, "
            f"{silence_count} silence-only (marked invalid)"
        )
        return ctx
