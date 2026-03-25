"""Stage 3: VAD-based segmentation with silence padding."""

from __future__ import annotations

import numpy as np

from ..config import PipelineConfig
from ..models import AudioSegment, PipelineContext
from ..utils.audio import load_audio, save_audio
from .base import PipelineStage


class VadSegmentStage(PipelineStage):
    name = "vad_segment"

    def __init__(self, config: PipelineConfig):
        super().__init__(config)
        self.cfg = config.vad

    def validate(self, ctx: PipelineContext) -> None:
        audio_path = ctx.denoised_audio_path or ctx.speaker_audio_path or ctx.full_audio_path
        if audio_path is None or not audio_path.exists():
            raise FileNotFoundError("No audio for VAD. Run previous stages first.")

    def run(self, ctx: PipelineContext) -> PipelineContext:
        input_path = ctx.denoised_audio_path or ctx.speaker_audio_path or ctx.full_audio_path
        segments_dir = ctx.output_dir / "segments"
        segments_dir.mkdir(parents=True, exist_ok=True)

        # Load Silero VAD (pip package v6 — no network call)
        self.logger.info("Loading Silero VAD model...")
        from silero_vad import load_silero_vad, get_speech_timestamps
        vad_model = load_silero_vad()

        # Load audio (already WAV from prior stages, 16kHz)
        waveform, sr = load_audio(input_path, sample_rate=16000)
        waveform_1d = waveform.squeeze(0)

        # Get speech timestamps
        speech_timestamps = get_speech_timestamps(
            waveform_1d,
            vad_model,
            threshold=self.cfg.threshold,
            min_speech_duration_ms=self.cfg.min_speech_duration_ms,
            min_silence_duration_ms=self.cfg.min_silence_duration_ms,
            sampling_rate=sr,
        )

        self.logger.info(f"Found {len(speech_timestamps)} speech segments")

        padding_samples = int(self.cfg.padding_ms * sr / 1000)
        segments = []

        for i, ts in enumerate(speech_timestamps):
            start_sample = ts["start"]
            end_sample = ts["end"]
            start_time = start_sample / sr
            end_time = end_sample / sr

            # Extract speech chunk
            speech_chunk = waveform_1d[start_sample:end_sample].numpy()

            # Add silence padding (zero-valued)
            padding = np.zeros(padding_samples, dtype=speech_chunk.dtype)
            padded = np.concatenate([padding, speech_chunk, padding])

            seg_id = f"seg_{i:05d}"
            seg_path = segments_dir / f"{seg_id}.wav"
            save_audio(padded, seg_path, sr)

            segment = AudioSegment(
                id=seg_id,
                audio_path=seg_path,
                start_time=start_time,
                end_time=end_time,
                sample_rate=sr,
            )
            segments.append(segment)

        ctx.segments = segments
        self.logger.info(f"Created {len(segments)} segments in {segments_dir}")
        return ctx
