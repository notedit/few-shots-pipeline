"""Pipeline orchestrator."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from .config import PipelineConfig
from .models import PipelineContext
from .stages import (
    AudioExtractStage,
    SpeakerExtractStage,
    SpeakerFilterStage,
    SpeakerTrimStage,
    DenoiseStage,
    VadSegmentStage,
    VadTrimStage,
    DualAsrStage,
    LlmCorrectStage,
)
from .stages.base import PipelineStage

logger = logging.getLogger(__name__)

ALL_STAGES: list[type[PipelineStage]] = [
    AudioExtractStage,     # 0: m4a/mp4 → full_audio.wav
    DenoiseStage,          # 1: full-audio speech enhancement (MossFormer2)
    VadSegmentStage,       # 2: VAD on clean audio → segments/
    SpeakerFilterStage,    # 3: coarse per-segment speaker similarity filter
    VadTrimStage,          # 4: post-segment VAD trim → segments_trimmed/
    SpeakerTrimStage,      # 5: fine sliding-window speaker trim
    DualAsrStage,          # 6: ASR transcription
    LlmCorrectStage,       # 7: LLM correction
]

STAGE_MAP: dict[str, type[PipelineStage]] = {
    cls.name: cls for cls in ALL_STAGES  # type: ignore[misc]
}


def setup_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(output_dir / "pipeline.log"),
        ],
    )


def run_pipeline(
    video_path: Path,
    reference_audio_path: Path,
    config: PipelineConfig,
    start_stage: str | None = None,
) -> PipelineContext:
    """Run the full pipeline or from a specific stage."""
    output_dir = Path(config.output_dir)
    setup_logging(output_dir)

    ctx = PipelineContext(
        video_path=video_path,
        reference_audio_path=reference_audio_path,
        output_dir=output_dir,
    )

    stages = ALL_STAGES
    if start_stage:
        start_idx = next(
            (i for i, s in enumerate(ALL_STAGES) if s.name == start_stage),  # type: ignore[misc]
            None,
        )
        if start_idx is None:
            raise ValueError(f"Unknown stage: {start_stage}")
        stages = ALL_STAGES[start_idx:]

    for stage_cls in stages:
        stage = stage_cls(config)
        ctx = stage.execute(ctx)

    # Write output manifests
    _write_manifests(ctx)
    return ctx


def run_single_stage(
    stage_name: str,
    config: PipelineConfig,
    input_path: Path | None = None,
) -> PipelineContext:
    """Run a single stage (for debugging / partial reruns)."""
    if stage_name not in STAGE_MAP:
        raise ValueError(f"Unknown stage: {stage_name}. Available: {list(STAGE_MAP)}")

    output_dir = Path(config.output_dir)
    setup_logging(output_dir)

    ctx = PipelineContext(
        video_path=input_path or Path("."),
        reference_audio_path=Path("."),
        output_dir=output_dir,
    )
    # Set input paths based on stage expectations
    if input_path:
        if stage_name == "audio_extract":
            ctx.video_path = input_path
        elif stage_name == "speaker_extract":
            ctx.full_audio_path = input_path
        elif stage_name == "denoise":
            ctx.speaker_audio_path = input_path
        elif stage_name in ("vad_segment", "asr_dual"):
            ctx.denoised_audio_path = input_path

    stage = STAGE_MAP[stage_name](config)
    ctx = stage.execute(ctx)
    return ctx


def _write_manifests(ctx: PipelineContext) -> None:
    """Write manifest.json and manifest_rejected.json."""
    valid = [s.to_dict() for s in ctx.segments if s.is_valid]
    rejected = [s.to_dict() for s in ctx.segments if not s.is_valid]

    manifest_path = ctx.output_dir / "manifest.json"
    rejected_path = ctx.output_dir / "manifest_rejected.json"

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(valid, f, ensure_ascii=False, indent=2)

    with open(rejected_path, "w", encoding="utf-8") as f:
        json.dump(rejected, f, ensure_ascii=False, indent=2)

    logger.info(
        f"Manifest written: {len(valid)} valid, {len(rejected)} rejected segments"
    )
