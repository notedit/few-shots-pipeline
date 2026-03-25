"""Audio I/O utilities.

Uses soundfile for WAV files and PyAV for compressed formats (MP3, M4A, AAC, etc.)
to avoid dependency on a system-installed ffmpeg binary (torchaudio ≥2.9 requires
TorchCodec which in turn needs ffmpeg shared libraries that may not be available).
"""

from __future__ import annotations

from pathlib import Path

import av
import numpy as np
import soundfile as sf
import torch
import torchaudio


def _load_with_av(path: Path, target_sr: int) -> tuple[torch.Tensor, int]:
    """Decode any audio format with PyAV and return a mono float32 tensor."""
    container = av.open(str(path))
    audio_stream = next(s for s in container.streams if s.type == "audio")
    src_sr = audio_stream.rate
    src_channels = audio_stream.channels

    frames: list[np.ndarray] = []
    for frame in container.decode(audio_stream):
        arr = frame.to_ndarray()  # shape: (channels, samples) or (samples,)
        frames.append(arr)
    container.close()

    if not frames:
        raise RuntimeError(f"No audio frames decoded from {path}")

    # Stack along last axis → (channels, total_samples)
    audio = np.concatenate(frames, axis=-1).astype(np.float32)

    # Normalise int formats to [-1, 1] if needed
    if audio.dtype != np.float32 or audio.max() > 1.0 or audio.min() < -1.0:
        max_val = max(abs(audio.max()), abs(audio.min()), 1.0)
        audio = audio / max_val

    # Ensure shape (channels, samples)
    if audio.ndim == 1:
        audio = audio[np.newaxis, :]

    # Downmix to mono
    if audio.shape[0] > 1:
        audio = audio.mean(axis=0, keepdims=True)

    waveform = torch.from_numpy(audio)  # (1, samples)

    # Resample to target_sr
    if src_sr != target_sr:
        resampler = torchaudio.transforms.Resample(src_sr, target_sr)
        waveform = resampler(waveform)

    return waveform, target_sr


def _load_with_soundfile(path: Path, target_sr: int) -> tuple[torch.Tensor, int]:
    """Load WAV / FLAC / OGG via soundfile (no ffmpeg dependency)."""
    data, sr = sf.read(str(path), always_2d=False, dtype="float32")
    # data shape: (samples,) mono or (samples, channels)
    if data.ndim == 1:
        data = data[np.newaxis, :]  # (1, samples)
    else:
        data = data.T  # (channels, samples)
    if data.shape[0] > 1:
        data = data.mean(axis=0, keepdims=True)

    waveform = torch.from_numpy(data.copy())

    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        waveform = resampler(waveform)

    return waveform, target_sr


# Formats that soundfile can handle natively (no ffmpeg needed)
_SF_EXTENSIONS = {".wav", ".flac", ".ogg", ".aiff", ".aif"}


def load_audio(path: Path, sample_rate: int = 16000) -> tuple[torch.Tensor, int]:
    """Load audio file and resample to *sample_rate*.

    Returns ``(waveform, sample_rate)`` where *waveform* has shape ``(1, N)``.

    Supports WAV / FLAC / OGG via soundfile and any format supported by PyAV
    (MP3, M4A, AAC, OPUS, …).
    """
    suffix = Path(path).suffix.lower()
    if suffix in _SF_EXTENSIONS:
        return _load_with_soundfile(path, sample_rate)
    return _load_with_av(path, sample_rate)


def save_audio(waveform: torch.Tensor | np.ndarray, path: Path, sample_rate: int = 16000) -> None:
    """Save audio to WAV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(waveform, torch.Tensor):
        waveform = waveform.numpy()
    if waveform.ndim == 1:
        waveform = waveform[np.newaxis, :]
    sf.write(str(path), waveform.squeeze(), sample_rate)


def get_duration(path: Path) -> float:
    """Get audio duration in seconds.

    Falls back to PyAV for formats that soundfile cannot introspect.
    """
    suffix = Path(path).suffix.lower()
    if suffix in _SF_EXTENSIONS:
        info = sf.info(str(path))
        return info.duration

    # Use PyAV for compressed formats
    container = av.open(str(path))
    audio_stream = next(s for s in container.streams if s.type == "audio")
    duration_sec: float | None = None
    if audio_stream.duration is not None and audio_stream.time_base is not None:
        duration_sec = float(audio_stream.duration * audio_stream.time_base)
    elif container.duration is not None:
        duration_sec = float(container.duration) / 1_000_000  # microseconds → seconds
    container.close()
    if duration_sec is None:
        raise RuntimeError(f"Cannot determine duration of {path}")
    return duration_sec
