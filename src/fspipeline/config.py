"""Pipeline configuration with Pydantic validation."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class AudioExtractConfig(BaseModel):
    sample_rate: int = 16000
    channels: int = 1


class SpeakerExtractConfig(BaseModel):
    diarization_model: str = "pyannote/speaker-diarization-3.1"
    embedding_model: str = "pyannote/wespeaker-voxceleb-resnet34-LM"
    similarity_threshold: float = 0.40
    hf_token: str = ""  # set via config YAML or HF_TOKEN env var


class DenoiseConfig(BaseModel):
    model_name: str = "MossFormer2_SE_48K"
    # Legacy field kept for backward compatibility
    model_source: str = "speechbrain/sepformer-dns4-16k-enhancement"
    chunk_seconds: float = 30.0


class VadConfig(BaseModel):
    min_speech_duration_ms: int = 250
    min_silence_duration_ms: int = 150  # tightened: cuts faster on speaker change
    padding_ms: int = 250        # padding added during initial segmentation
    threshold: float = 0.5
    trim_padding_ms: int = 200   # padding kept after post-denoise VAD trim


class SpeakerTrimConfig(BaseModel):
    window_sec: float = 1.5       # sliding window length in seconds
    stride_sec: float = 0.5       # step between windows
    threshold: float = 0.45       # per-window cosine similarity to keep
    min_duration_sec: float = 1.5 # discard trimmed segment shorter than this
    padding_ms: int = 200         # silence padding around kept region


class AsrConfig(BaseModel):
    whisper_model: str = "large-v3"
    whisper_device: str = "auto"   # auto → cuda if available, else cpu
    whisper_compute_type: str = "float16"
    qwen_model: str = "Qwen/Qwen3-ASR-1.7B"
    language: str = "auto"
    similarity_threshold: float = 0.90
    whisper_only: bool = True  # skip Qwen; use Whisper results directly


class LlmConfig(BaseModel):
    base_url: str = "https://openrouter.ai/api/v1"
    model: str = "anthropic/claude-sonnet-4"
    api_key: str = ""
    batch_size: int = 20
    temperature: float = 0.3


class PipelineConfig(BaseModel):
    audio_extract: AudioExtractConfig = Field(default_factory=AudioExtractConfig)
    speaker_extract: SpeakerExtractConfig = Field(default_factory=SpeakerExtractConfig)
    speaker_trim: SpeakerTrimConfig = Field(default_factory=SpeakerTrimConfig)
    denoise: DenoiseConfig = Field(default_factory=DenoiseConfig)
    vad: VadConfig = Field(default_factory=VadConfig)
    asr: AsrConfig = Field(default_factory=AsrConfig)
    llm: LlmConfig = Field(default_factory=LlmConfig)
    output_dir: str = "output"


def load_config(config_path: Path | None = None) -> PipelineConfig:
    """Load configuration from YAML file, falling back to defaults."""
    if config_path and config_path.exists():
        with open(config_path) as f:
            data = yaml.safe_load(f) or {}
        return PipelineConfig(**data)
    return PipelineConfig()
