"""Audio I/O utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio


def load_audio(path: Path, sample_rate: int = 16000) -> tuple[torch.Tensor, int]:
    """Load audio file and resample to target sample rate. Returns (waveform, sr)."""
    waveform, sr = torchaudio.load(str(path))
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != sample_rate:
        resampler = torchaudio.transforms.Resample(sr, sample_rate)
        waveform = resampler(waveform)
    return waveform, sample_rate


def save_audio(waveform: torch.Tensor | np.ndarray, path: Path, sample_rate: int = 16000) -> None:
    """Save audio to WAV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(waveform, torch.Tensor):
        waveform = waveform.numpy()
    if waveform.ndim == 1:
        waveform = waveform[np.newaxis, :]
    sf.write(str(path), waveform.squeeze(), sample_rate)


def get_duration(path: Path) -> float:
    """Get audio duration in seconds."""
    info = sf.info(str(path))
    return info.duration
