"""Integration tests for pipeline stages using real audio fixtures.

These tests exercise the actual stage logic end-to-end without requiring
ML models that need downloading (speaker_extract, denoise, asr_dual,
llm_correct).  The stages tested here only need torchaudio / soundfile /
PyAV which are already installed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).parent.parent
SOURCE_AUDIO = REPO_ROOT / "source_audio.m4a"
TARGET_AUDIO = REPO_ROOT / "target_audio.mp3"


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _default_config(**overrides):
    from fspipeline.config import PipelineConfig

    cfg = PipelineConfig()
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def _make_ctx(output_dir: Path, video_path: Path = SOURCE_AUDIO,
              reference: Path = TARGET_AUDIO):
    from fspipeline.models import PipelineContext

    return PipelineContext(
        video_path=video_path,
        reference_audio_path=reference,
        output_dir=output_dir,
    )


# ──────────────────────────────────────────────────────────────────────────────
# AudioExtractStage
# ──────────────────────────────────────────────────────────────────────────────

class TestAudioExtractStage:
    def test_extracts_wav(self, tmp_path):
        from fspipeline.stages.audio_extract import AudioExtractStage

        cfg = _default_config()
        cfg.output_dir = str(tmp_path)
        ctx = _make_ctx(tmp_path)

        stage = AudioExtractStage(cfg)
        ctx = stage.execute(ctx)

        assert ctx.full_audio_path is not None
        assert ctx.full_audio_path.exists()
        assert ctx.full_audio_path.suffix == ".wav"

    def test_output_is_mono_16k(self, tmp_path):
        from fspipeline.stages.audio_extract import AudioExtractStage
        import soundfile as sf

        cfg = _default_config()
        cfg.output_dir = str(tmp_path)
        ctx = _make_ctx(tmp_path)

        stage = AudioExtractStage(cfg)
        ctx = stage.execute(ctx)

        info = sf.info(str(ctx.full_audio_path))
        assert info.samplerate == 16000
        assert info.channels == 1

    def test_output_duration_positive(self, tmp_path):
        from fspipeline.stages.audio_extract import AudioExtractStage
        from fspipeline.utils.audio import get_duration

        cfg = _default_config()
        cfg.output_dir = str(tmp_path)
        ctx = _make_ctx(tmp_path)

        stage = AudioExtractStage(cfg)
        ctx = stage.execute(ctx)

        dur = get_duration(ctx.full_audio_path)
        assert dur > 1.0

    def test_validate_missing_video(self, tmp_path):
        from fspipeline.stages.audio_extract import AudioExtractStage
        from fspipeline.models import PipelineContext

        cfg = _default_config()
        ctx = PipelineContext(
            video_path=Path("/nonexistent/video.mp4"),
            reference_audio_path=TARGET_AUDIO,
            output_dir=tmp_path,
        )
        stage = AudioExtractStage(cfg)
        with pytest.raises(FileNotFoundError):
            stage.execute(ctx)


# ──────────────────────────────────────────────────────────────────────────────
# VadSegmentStage  (uses Silero VAD — downloaded from torch hub on first run)
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.slow
class TestVadSegmentStage:
    """Requires network access on first run (Silero VAD ~2 MB download)."""

    def _prepare_audio(self, tmp_path: Path) -> Path:
        """Extract audio first so VadSegmentStage has a WAV to read."""
        from fspipeline.stages.audio_extract import AudioExtractStage
        from fspipeline.config import PipelineConfig

        cfg = PipelineConfig()
        cfg.output_dir = str(tmp_path)
        ctx = _make_ctx(tmp_path)

        stage = AudioExtractStage(cfg)
        ctx = stage.execute(ctx)
        return ctx

    def test_produces_segments(self, tmp_path):
        from fspipeline.stages.vad_segment import VadSegmentStage
        from fspipeline.config import PipelineConfig

        cfg = PipelineConfig()
        cfg.output_dir = str(tmp_path)

        ctx = self._prepare_audio(tmp_path)

        stage = VadSegmentStage(cfg)
        ctx = stage.execute(ctx)

        assert len(ctx.segments) > 0, "VAD should find at least one speech segment"

    def test_segments_have_valid_paths(self, tmp_path):
        from fspipeline.stages.vad_segment import VadSegmentStage
        from fspipeline.config import PipelineConfig

        cfg = PipelineConfig()
        cfg.output_dir = str(tmp_path)

        ctx = self._prepare_audio(tmp_path)

        stage = VadSegmentStage(cfg)
        ctx = stage.execute(ctx)

        for seg in ctx.segments:
            assert seg.audio_path.exists(), f"Segment file missing: {seg.audio_path}"
            assert seg.audio_path.stat().st_size > 0

    def test_segments_are_valid_by_default(self, tmp_path):
        from fspipeline.stages.vad_segment import VadSegmentStage
        from fspipeline.config import PipelineConfig

        cfg = PipelineConfig()
        cfg.output_dir = str(tmp_path)

        ctx = self._prepare_audio(tmp_path)

        stage = VadSegmentStage(cfg)
        ctx = stage.execute(ctx)

        for seg in ctx.segments:
            assert seg.is_valid, f"Segment {seg.id} should start as valid"

    def test_segment_timing_non_negative(self, tmp_path):
        from fspipeline.stages.vad_segment import VadSegmentStage
        from fspipeline.config import PipelineConfig

        cfg = PipelineConfig()
        cfg.output_dir = str(tmp_path)

        ctx = self._prepare_audio(tmp_path)

        stage = VadSegmentStage(cfg)
        ctx = stage.execute(ctx)

        for seg in ctx.segments:
            assert seg.start_time >= 0.0
            assert seg.end_time > seg.start_time

    def test_validate_raises_without_audio(self, tmp_path):
        from fspipeline.stages.vad_segment import VadSegmentStage
        from fspipeline.models import PipelineContext
        from fspipeline.config import PipelineConfig

        cfg = PipelineConfig()
        ctx = PipelineContext(
            video_path=SOURCE_AUDIO,
            reference_audio_path=TARGET_AUDIO,
            output_dir=tmp_path,
        )
        # No audio paths set → validate should raise
        stage = VadSegmentStage(cfg)
        with pytest.raises(FileNotFoundError):
            stage.execute(ctx)


# ──────────────────────────────────────────────────────────────────────────────
# Stage validate() edge cases
# ──────────────────────────────────────────────────────────────────────────────

class TestStageValidation:
    def test_denoise_validate_no_audio(self, tmp_path):
        from fspipeline.stages.denoise import DenoiseStage
        from fspipeline.models import PipelineContext
        from fspipeline.config import PipelineConfig

        cfg = PipelineConfig()
        ctx = PipelineContext(
            video_path=SOURCE_AUDIO,
            reference_audio_path=TARGET_AUDIO,
            output_dir=tmp_path,
        )
        stage = DenoiseStage(cfg)
        with pytest.raises(FileNotFoundError):
            stage.execute(ctx)

    def test_asr_validate_no_segments(self, tmp_path):
        from fspipeline.stages.asr_dual import DualAsrStage
        from fspipeline.models import PipelineContext
        from fspipeline.config import PipelineConfig

        cfg = PipelineConfig()
        ctx = PipelineContext(
            video_path=SOURCE_AUDIO,
            reference_audio_path=TARGET_AUDIO,
            output_dir=tmp_path,
        )
        stage = DualAsrStage(cfg)
        with pytest.raises(ValueError, match="No segments"):
            stage.execute(ctx)

    def test_llm_validate_no_api_key(self, tmp_path):
        from fspipeline.stages.llm_correct import LlmCorrectStage
        from fspipeline.models import PipelineContext, AudioSegment
        from fspipeline.config import PipelineConfig

        cfg = PipelineConfig()
        cfg.llm.api_key = ""  # explicitly empty
        ctx = PipelineContext(
            video_path=SOURCE_AUDIO,
            reference_audio_path=TARGET_AUDIO,
            output_dir=tmp_path,
        )
        # Add one valid segment
        seg = AudioSegment(
            id="seg_00001",
            audio_path=Path("x.wav"),
            start_time=0.0,
            end_time=1.0,
        )
        seg.transcript_final = "hello"
        ctx.segments.append(seg)

        stage = LlmCorrectStage(cfg)
        with pytest.raises(ValueError, match="api_key"):
            stage.execute(ctx)

    def test_llm_validate_no_valid_segments(self, tmp_path):
        from fspipeline.stages.llm_correct import LlmCorrectStage
        from fspipeline.models import PipelineContext, AudioSegment
        from fspipeline.config import PipelineConfig

        cfg = PipelineConfig()
        cfg.llm.api_key = "sk-test"  # provide key but no valid segments
        ctx = PipelineContext(
            video_path=SOURCE_AUDIO,
            reference_audio_path=TARGET_AUDIO,
            output_dir=tmp_path,
        )
        # Add an INVALID segment
        seg = AudioSegment(
            id="seg_00001",
            audio_path=Path("x.wav"),
            start_time=0.0,
            end_time=1.0,
        )
        seg.is_valid = False
        ctx.segments.append(seg)

        stage = LlmCorrectStage(cfg)
        with pytest.raises(ValueError, match="No valid segments"):
            stage.execute(ctx)
