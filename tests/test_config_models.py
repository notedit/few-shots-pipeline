"""Tests for fspipeline.config and fspipeline.models."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import yaml


# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

class TestPipelineConfigDefaults:
    def test_default_instantiation(self):
        from fspipeline.config import PipelineConfig

        cfg = PipelineConfig()
        assert cfg.audio_extract.sample_rate == 16000
        assert cfg.audio_extract.channels == 1
        assert cfg.vad.threshold == 0.5
        assert cfg.asr.whisper_model == "large-v3"
        assert cfg.output_dir == "output"

    def test_speaker_extract_defaults(self):
        from fspipeline.config import PipelineConfig

        cfg = PipelineConfig()
        assert "speaker-diarization" in cfg.speaker_extract.diarization_model
        assert 0.0 < cfg.speaker_extract.similarity_threshold < 1.0

    def test_denoise_defaults(self):
        from fspipeline.config import PipelineConfig

        cfg = PipelineConfig()
        assert cfg.denoise.chunk_seconds > 0

    def test_llm_defaults(self):
        from fspipeline.config import PipelineConfig

        cfg = PipelineConfig()
        assert "openrouter" in cfg.llm.base_url
        assert cfg.llm.batch_size > 0
        assert 0.0 <= cfg.llm.temperature <= 1.0


class TestLoadConfig:
    def test_load_nonexistent_returns_defaults(self):
        from fspipeline.config import load_config

        cfg = load_config(Path("/nonexistent/config.yaml"))
        assert cfg.output_dir == "output"

    def test_load_none_returns_defaults(self):
        from fspipeline.config import load_config

        cfg = load_config(None)
        assert isinstance(cfg.audio_extract.sample_rate, int)

    def test_load_custom_yaml(self, tmp_path):
        from fspipeline.config import load_config

        data = {
            "output_dir": "/tmp/my_output",
            "audio_extract": {"sample_rate": 22050, "channels": 1},
            "vad": {"threshold": 0.6},
        }
        p = tmp_path / "config.yaml"
        p.write_text(yaml.dump(data))

        cfg = load_config(p)
        assert cfg.output_dir == "/tmp/my_output"
        assert cfg.audio_extract.sample_rate == 22050
        assert cfg.vad.threshold == 0.6
        # Unspecified fields keep defaults
        assert cfg.asr.whisper_model == "large-v3"

    def test_load_default_yaml(self):
        """Load the repo's own config/default.yaml and check it parses cleanly."""
        from fspipeline.config import load_config

        cfg_path = Path(__file__).parent.parent / "config" / "default.yaml"
        if cfg_path.exists():
            cfg = load_config(cfg_path)
            assert cfg.audio_extract.sample_rate > 0

    def test_partial_override(self, tmp_path):
        from fspipeline.config import load_config

        data = {"llm": {"batch_size": 5}}
        p = tmp_path / "c.yaml"
        p.write_text(yaml.dump(data))

        cfg = load_config(p)
        assert cfg.llm.batch_size == 5
        # Other llm fields still default
        assert cfg.llm.temperature == 0.3


# ──────────────────────────────────────────────────────────────────────────────
# Models
# ──────────────────────────────────────────────────────────────────────────────

class TestAudioSegment:
    def _make_seg(self, seg_id="seg_00001", start=1.0, end=3.0):
        from fspipeline.models import AudioSegment

        return AudioSegment(
            id=seg_id,
            audio_path=Path(f"/tmp/{seg_id}.wav"),
            start_time=start,
            end_time=end,
        )

    def test_duration_property(self):
        seg = self._make_seg(start=1.0, end=3.5)
        assert abs(seg.duration - 2.5) < 1e-9

    def test_default_is_valid(self):
        seg = self._make_seg()
        assert seg.is_valid is True

    def test_to_dict_keys(self):
        seg = self._make_seg()
        d = seg.to_dict()
        expected_keys = {
            "id", "audio_path", "start_time", "end_time", "duration",
            "sample_rate", "speaker_score", "transcript_whisper",
            "transcript_qwen", "transcript_final", "similarity_score", "is_valid",
        }
        assert expected_keys == set(d.keys())

    def test_to_dict_values(self):
        seg = self._make_seg(seg_id="seg_00042", start=5.0, end=7.0)
        seg.transcript_final = "hello world"
        seg.similarity_score = 0.95
        d = seg.to_dict()
        assert d["id"] == "seg_00042"
        assert d["duration"] == 2.0
        assert d["transcript_final"] == "hello world"
        assert d["similarity_score"] == 0.95

    def test_audio_path_stored_as_string_in_dict(self):
        seg = self._make_seg()
        d = seg.to_dict()
        assert isinstance(d["audio_path"], str)

    def test_default_sample_rate(self):
        seg = self._make_seg()
        assert seg.sample_rate == 16000


class TestPipelineContext:
    def test_default_paths_are_none(self):
        from fspipeline.models import PipelineContext

        ctx = PipelineContext(
            video_path=Path("input.mp4"),
            reference_audio_path=Path("ref.wav"),
            output_dir=Path("output"),
        )
        assert ctx.full_audio_path is None
        assert ctx.speaker_audio_path is None
        assert ctx.denoised_audio_path is None

    def test_segments_default_empty(self):
        from fspipeline.models import PipelineContext

        ctx = PipelineContext(
            video_path=Path("v.mp4"),
            reference_audio_path=Path("r.wav"),
            output_dir=Path("out"),
        )
        assert ctx.segments == []

    def test_set_paths(self):
        from fspipeline.models import PipelineContext

        ctx = PipelineContext(
            video_path=Path("v.mp4"),
            reference_audio_path=Path("r.wav"),
            output_dir=Path("out"),
        )
        ctx.full_audio_path = Path("full.wav")
        assert ctx.full_audio_path == Path("full.wav")

    def test_segments_mutation(self):
        from fspipeline.models import AudioSegment, PipelineContext

        ctx = PipelineContext(
            video_path=Path("v.mp4"),
            reference_audio_path=Path("r.wav"),
            output_dir=Path("out"),
        )
        seg = AudioSegment(
            id="seg_00001",
            audio_path=Path("seg.wav"),
            start_time=0.0,
            end_time=1.0,
        )
        ctx.segments.append(seg)
        assert len(ctx.segments) == 1
        assert ctx.segments[0].id == "seg_00001"
