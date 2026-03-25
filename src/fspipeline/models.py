"""Data models for the pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class AudioSegment:
    """Represents a single audio segment extracted from the pipeline."""

    id: str  # "seg_00001"
    audio_path: Path
    start_time: float
    end_time: float
    sample_rate: int = 16000
    speaker_score: float = 0.0
    transcript_whisper: str = ""
    transcript_qwen: str = ""
    transcript_final: str = ""
    similarity_score: float = 0.0
    is_valid: bool = True

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "audio_path": str(self.audio_path),
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "sample_rate": self.sample_rate,
            "speaker_score": self.speaker_score,
            "transcript_whisper": self.transcript_whisper,
            "transcript_qwen": self.transcript_qwen,
            "transcript_final": self.transcript_final,
            "similarity_score": self.similarity_score,
            "is_valid": self.is_valid,
        }


@dataclass
class PipelineContext:
    """Shared context passed through pipeline stages."""

    video_path: Path
    reference_audio_path: Path
    output_dir: Path
    full_audio_path: Path | None = None
    speaker_audio_path: Path | None = None
    denoised_audio_path: Path | None = None
    segments: list[AudioSegment] = field(default_factory=list)
