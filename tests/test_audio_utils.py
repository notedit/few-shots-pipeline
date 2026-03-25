"""Tests for fspipeline.utils.audio using the provided audio fixtures."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

# Absolute paths to the provided audio files (in repo root)
REPO_ROOT = Path(__file__).parent.parent
SOURCE_AUDIO = REPO_ROOT / "source_audio.m4a"   # ~74 MB  44 100 Hz stereo
TARGET_AUDIO = REPO_ROOT / "target_audio.mp3"   # ~81 KB  24 000 Hz mono


@pytest.fixture(scope="module")
def tmp_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("audio_tests")


# ──────────────────────────────────────────────────────────────────────────────
# load_audio
# ──────────────────────────────────────────────────────────────────────────────

class TestLoadAudio:
    """load_audio should return a (1, N) float32 tensor at the requested SR."""

    def test_load_mp3_default_sr(self):
        from fspipeline.utils.audio import load_audio

        waveform, sr = load_audio(TARGET_AUDIO)
        assert sr == 16000, f"expected sr=16000, got {sr}"
        assert waveform.ndim == 2
        assert waveform.shape[0] == 1, "expected mono (1 channel)"
        assert waveform.shape[1] > 0

    def test_load_mp3_custom_sr(self):
        from fspipeline.utils.audio import load_audio

        waveform, sr = load_audio(TARGET_AUDIO, sample_rate=22050)
        assert sr == 22050

    def test_load_m4a_default_sr(self):
        from fspipeline.utils.audio import load_audio

        waveform, sr = load_audio(SOURCE_AUDIO)
        assert sr == 16000
        assert waveform.ndim == 2
        assert waveform.shape[0] == 1

    def test_load_m4a_stereo_downmixed(self):
        """M4A is stereo — loader must downmix to mono."""
        from fspipeline.utils.audio import load_audio

        waveform, sr = load_audio(SOURCE_AUDIO)
        assert waveform.shape[0] == 1, "stereo M4A should be downmixed to mono"

    def test_load_wav(self, tmp_dir):
        """Round-trip: save a WAV then load it back.

        soundfile stores float32 WAV and clips values to [-1, 1], so we
        keep the test signal within that range to guarantee exact recovery.
        """
        from fspipeline.utils.audio import load_audio, save_audio

        # Use rand() → [0, 1) then shift to [-0.5, 0.5) to stay well within [-1, 1]
        original = torch.rand(1, 16000) - 0.5
        wav_path = tmp_dir / "test.wav"
        save_audio(original, wav_path, sample_rate=16000)

        loaded, sr = load_audio(wav_path)
        assert sr == 16000
        assert loaded.shape == (1, 16000)
        assert torch.allclose(original, loaded, atol=1e-3)

    def test_output_dtype(self):
        from fspipeline.utils.audio import load_audio

        waveform, _ = load_audio(TARGET_AUDIO)
        assert waveform.dtype == torch.float32

    def test_values_in_range(self):
        from fspipeline.utils.audio import load_audio

        waveform, _ = load_audio(TARGET_AUDIO)
        assert waveform.abs().max().item() <= 1.0 + 1e-4, "samples should be in [-1, 1]"


# ──────────────────────────────────────────────────────────────────────────────
# save_audio
# ──────────────────────────────────────────────────────────────────────────────

class TestSaveAudio:
    def test_save_tensor(self, tmp_dir):
        from fspipeline.utils.audio import save_audio

        waveform = torch.randn(1, 8000)
        path = tmp_dir / "out_tensor.wav"
        save_audio(waveform, path, sample_rate=8000)
        assert path.exists()
        assert path.stat().st_size > 0

    def test_save_numpy_1d(self, tmp_dir):
        from fspipeline.utils.audio import save_audio

        arr = np.random.randn(8000).astype(np.float32)
        path = tmp_dir / "out_numpy.wav"
        save_audio(arr, path, sample_rate=8000)
        assert path.exists()

    def test_save_numpy_2d(self, tmp_dir):
        from fspipeline.utils.audio import save_audio

        arr = np.random.randn(1, 8000).astype(np.float32)
        path = tmp_dir / "out_numpy2d.wav"
        save_audio(arr, path, sample_rate=8000)
        assert path.exists()

    def test_creates_parent_dir(self, tmp_dir):
        from fspipeline.utils.audio import save_audio

        deep = tmp_dir / "a" / "b" / "c" / "out.wav"
        save_audio(torch.zeros(1, 100), deep, sample_rate=16000)
        assert deep.exists()

    def test_round_trip_values(self, tmp_dir):
        from fspipeline.utils.audio import load_audio, save_audio

        original = torch.rand(1, 4000) * 2 - 1  # uniform [-1, 1]
        path = tmp_dir / "rt.wav"
        save_audio(original, path, sample_rate=16000)
        loaded, sr = load_audio(path)
        assert sr == 16000
        assert torch.allclose(original, loaded, atol=1e-3)


# ──────────────────────────────────────────────────────────────────────────────
# get_duration
# ──────────────────────────────────────────────────────────────────────────────

class TestGetDuration:
    def test_mp3_duration_positive(self):
        from fspipeline.utils.audio import get_duration

        dur = get_duration(TARGET_AUDIO)
        assert dur > 0

    def test_mp3_duration_approx(self):
        """target_audio.mp3 is ~20 s; just check it's a plausible positive value."""
        from fspipeline.utils.audio import get_duration

        dur = get_duration(TARGET_AUDIO)
        assert 1.0 < dur < 3600.0

    def test_m4a_duration_positive(self):
        from fspipeline.utils.audio import get_duration

        dur = get_duration(SOURCE_AUDIO)
        assert dur > 0

    def test_wav_duration(self, tmp_dir):
        from fspipeline.utils.audio import get_duration, save_audio

        waveform = torch.zeros(1, 16000)
        path = tmp_dir / "dur_test.wav"
        save_audio(waveform, path, sample_rate=16000)
        dur = get_duration(path)
        assert abs(dur - 1.0) < 0.05, f"expected ~1.0s, got {dur:.3f}s"
