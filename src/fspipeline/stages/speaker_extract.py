"""Stage 1: Speaker diarization + target speaker extraction."""

from __future__ import annotations

import torch
import torchaudio
import numpy as np
from pyannote.audio import Pipeline as PyannotePipeline
from pyannote.audio import Inference

from ..config import PipelineConfig
from ..models import PipelineContext
from ..utils.audio import load_audio, save_audio
from .base import PipelineStage


class SpeakerExtractStage(PipelineStage):
    name = "speaker_extract"

    def __init__(self, config: PipelineConfig):
        super().__init__(config)
        self.cfg = config.speaker_extract

    def validate(self, ctx: PipelineContext) -> None:
        if ctx.full_audio_path is None or not ctx.full_audio_path.exists():
            raise FileNotFoundError("Full audio not found. Run audio_extract first.")
        if not ctx.reference_audio_path.exists():
            raise FileNotFoundError(f"Reference audio not found: {ctx.reference_audio_path}")

    def run(self, ctx: PipelineContext) -> PipelineContext:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        hf_token = self.cfg.hf_token or None

        # Run diarization
        self.logger.info("Running speaker diarization...")
        diarization_pipeline = PyannotePipeline.from_pretrained(
            self.cfg.diarization_model,
            use_auth_token=hf_token,
        )
        diarization_pipeline.to(device)
        diarization = diarization_pipeline(str(ctx.full_audio_path))

        # Compute reference embedding
        self.logger.info("Computing reference speaker embedding...")
        embedding_model = Inference(
            self.cfg.embedding_model,
            window="whole",
            use_auth_token=hf_token,
        )
        embedding_model.to(device)
        ref_embedding = embedding_model(str(ctx.reference_audio_path))

        # Group segments by speaker label
        speaker_segments: dict[str, list[tuple[float, float]]] = {}
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            speaker_segments.setdefault(speaker, []).append((turn.start, turn.end))

        # Load full audio for extraction
        waveform, sr = load_audio(ctx.full_audio_path, sample_rate=16000)

        # Find best matching speaker by centroid cosine similarity
        best_speaker = None
        best_score = -1.0

        for speaker, segs in speaker_segments.items():
            embeddings = []
            for start, end in segs[:10]:  # Sample up to 10 segments for centroid
                start_sample = int(start * sr)
                end_sample = int(end * sr)
                chunk = waveform[:, start_sample:end_sample]
                if chunk.shape[1] < sr:  # Skip very short segments
                    continue
                # Save temp chunk for embedding
                tmp_path = ctx.output_dir / f"_tmp_spk_{speaker}.wav"
                save_audio(chunk, tmp_path, sr)
                emb = embedding_model(str(tmp_path))
                embeddings.append(emb)
                tmp_path.unlink(missing_ok=True)

            if not embeddings:
                continue

            centroid = np.mean(embeddings, axis=0)
            # Cosine similarity
            score = float(
                np.dot(centroid, ref_embedding)
                / (np.linalg.norm(centroid) * np.linalg.norm(ref_embedding) + 1e-8)
            )
            self.logger.info(f"Speaker {speaker}: similarity={score:.4f} ({len(segs)} segments)")

            if score > best_score:
                best_score = score
                best_speaker = speaker

        if best_speaker is None or best_score < self.cfg.similarity_threshold:
            raise RuntimeError(
                f"No speaker matched reference (best={best_score:.4f}, "
                f"threshold={self.cfg.similarity_threshold})"
            )

        self.logger.info(
            f"Target speaker: {best_speaker} (score={best_score:.4f})"
        )

        # Extract and concatenate target speaker segments
        target_segs = speaker_segments[best_speaker]
        chunks = []
        for start, end in sorted(target_segs):
            start_sample = int(start * sr)
            end_sample = int(end * sr)
            chunks.append(waveform[:, start_sample:end_sample])

        speaker_audio = torch.cat(chunks, dim=1)
        output_path = ctx.output_dir / "speaker_audio.wav"
        save_audio(speaker_audio, output_path, sr)

        ctx.speaker_audio_path = output_path
        self.logger.info(
            f"Extracted {len(target_segs)} segments, "
            f"total {speaker_audio.shape[1] / sr:.1f}s"
        )
        return ctx
