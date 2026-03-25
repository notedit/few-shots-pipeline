"""Stage: Sliding-window speaker trim.

After VadTrim, some segments still contain multiple speakers because the
global embedding average in SpeakerFilterStage can pass a segment where the
target speaker dominates but non-target speech exists at the edges.

This stage re-processes every valid segment with a fine-grained sliding
window, keeps only the contiguous region where per-window cosine similarity
to the reference speaker is above threshold, and discards short leftovers.

Algorithm
---------
1. Slide a window of `window_sec` seconds (stride `stride_sec`) over the
   segment audio (at 16 kHz for the embedding model).
2. Compute cosine similarity between each window's embedding and the
   pre-computed reference embedding.
3. Mark every window whose score ≥ threshold as "target".
4. Find the longest contiguous run of target windows.
5. Convert that run back to sample indices (original sample rate), add
   `padding_ms` on each side, clamp to file bounds, and write the result.
6. If the kept region is shorter than `min_duration_sec`, mark the segment
   invalid.

The reference embedding is computed once from ctx.reference_audio_path and
reused across all segments.
"""

from __future__ import annotations

import numpy as np
import soundfile as sf
import torch

from ..config import PipelineConfig
from ..models import PipelineContext
from .base import PipelineStage

_EMB_SR = 16_000  # wespeaker embedding model native sample rate


class SpeakerTrimStage(PipelineStage):
    name = "speaker_trim"

    def __init__(self, config: PipelineConfig):
        super().__init__(config)
        self.cfg = config.speaker_trim
        self.emb_cfg = config.speaker_extract

    def validate(self, ctx: PipelineContext) -> None:
        valid = [s for s in ctx.segments if s.is_valid]
        if not valid:
            raise ValueError("No valid segments to trim.")
        if not ctx.reference_audio_path.exists():
            raise FileNotFoundError(
                f"Reference audio not found: {ctx.reference_audio_path}"
            )

    # ------------------------------------------------------------------
    # Embedding helpers
    # ------------------------------------------------------------------
    def _load_embedding_model(self):
        from pyannote.audio import Inference, Model

        hf_token = self.emb_cfg.hf_token or None
        self.logger.info(f"Loading embedding model: {self.emb_cfg.embedding_model}")
        model_checkpoint = Model.from_pretrained(
            self.emb_cfg.embedding_model, token=hf_token
        )
        return Inference(model_checkpoint, window="whole")

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        a, b = a.flatten(), b.flatten()
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

    @staticmethod
    def _resample_to_16k(data: np.ndarray, src_sr: int) -> np.ndarray:
        if src_sr == _EMB_SR:
            return data
        import torchaudio.transforms as T
        wav = torch.from_numpy(data).unsqueeze(0)
        wav = T.Resample(src_sr, _EMB_SR)(wav)
        return wav.squeeze(0).numpy()

    def _embed_chunk(self, emb_model, chunk_16k: np.ndarray) -> np.ndarray:
        """Embed a (N,) float32 array at 16 kHz."""
        wav_t = torch.from_numpy(chunk_16k).unsqueeze(0)  # (1, N)
        return np.array(
            emb_model({"waveform": wav_t, "sample_rate": _EMB_SR})
        ).flatten()

    # ------------------------------------------------------------------
    # Core sliding-window logic
    # ------------------------------------------------------------------
    def _find_target_region(
        self,
        emb_model,
        ref_emb: np.ndarray,
        data_16k: np.ndarray,
    ) -> tuple[int, int] | None:
        """Return (start_sample, end_sample) of the longest target run, or None."""
        sr = _EMB_SR
        win_samples = int(self.cfg.window_sec * sr)
        stride_samples = int(self.cfg.stride_sec * sr)
        n = len(data_16k)

        if n < win_samples:
            # Segment shorter than one window — treat the whole thing as one window
            score = self._cosine(ref_emb, self._embed_chunk(emb_model, data_16k))
            if score >= self.cfg.threshold:
                return (0, n)
            return None

        scores: list[float] = []
        window_starts: list[int] = []
        start = 0
        while start + win_samples <= n:
            chunk = data_16k[start: start + win_samples]
            emb = self._embed_chunk(emb_model, chunk)
            scores.append(self._cosine(ref_emb, emb))
            window_starts.append(start)
            start += stride_samples

        # Boolean mask of target windows
        mask = [s >= self.cfg.threshold for s in scores]

        # Find longest contiguous True run
        best_start_idx = best_len = 0
        cur_start_idx = cur_len = 0
        for i, v in enumerate(mask):
            if v:
                if cur_len == 0:
                    cur_start_idx = i
                cur_len += 1
                if cur_len > best_len:
                    best_len = cur_len
                    best_start_idx = cur_start_idx
            else:
                cur_len = 0

        if best_len == 0:
            return None

        # Convert window indices → sample indices (original 16 kHz)
        region_start = window_starts[best_start_idx]
        last_win = best_start_idx + best_len - 1
        region_end = min(window_starts[last_win] + win_samples, n)
        return (region_start, region_end)

    # ------------------------------------------------------------------
    def run(self, ctx: PipelineContext) -> PipelineContext:
        emb_model = self._load_embedding_model()

        # Reference embedding (16 kHz)
        from ..utils.audio import load_audio
        ref_waveform, _ = load_audio(ctx.reference_audio_path, sample_rate=_EMB_SR)
        ref_emb = self._embed_chunk(emb_model, ref_waveform.squeeze(0).numpy())
        self.logger.info(
            f"Reference embedding computed from {ctx.reference_audio_path}"
        )

        trimmed_dir = ctx.output_dir / "segments_trimmed"
        trimmed_dir.mkdir(parents=True, exist_ok=True)

        valid_segs = [s for s in ctx.segments if s.is_valid]
        self.logger.info(
            f"Speaker-trimming {len(valid_segs)} segments "
            f"(win={self.cfg.window_sec}s, stride={self.cfg.stride_sec}s, "
            f"thr={self.cfg.threshold}) → {trimmed_dir}"
        )

        pad_samples_16k = int(self.cfg.padding_ms * _EMB_SR / 1000)
        min_samples_16k = int(self.cfg.min_duration_sec * _EMB_SR)
        kept = discarded = 0

        for i, seg in enumerate(valid_segs):
            # Load at native sr (may be 16kHz from VAD segs or 48kHz from denoised)
            data, src_sr = sf.read(str(seg.audio_path), dtype="float32")
            if data.ndim > 1:
                data = data.mean(axis=1)

            # Downsample to 16 kHz for embedding
            data_16k = self._resample_to_16k(data, src_sr)

            region = self._find_target_region(emb_model, ref_emb, data_16k)

            if region is None:
                seg.is_valid = False
                discarded += 1
                self.logger.debug(f"  {seg.id}: no target region → discarded")
                continue

            r_start_16k, r_end_16k = region
            region_len_16k = r_end_16k - r_start_16k

            if region_len_16k < min_samples_16k:
                seg.is_valid = False
                discarded += 1
                self.logger.debug(
                    f"  {seg.id}: target region {region_len_16k/16000*1000:.0f}ms "
                    f"< min {self.cfg.min_duration_sec*1000:.0f}ms → discarded"
                )
                continue

            # Add padding in 16 kHz space, clamp
            r_start_16k = max(0, r_start_16k - pad_samples_16k)
            r_end_16k = min(len(data_16k), r_end_16k + pad_samples_16k)

            # Map back to original sample rate
            ratio = src_sr / _EMB_SR
            orig_start = int(r_start_16k * ratio)
            orig_end = min(len(data), int(r_end_16k * ratio))

            trimmed = data[orig_start:orig_end]

            out_path = trimmed_dir / seg.audio_path.name
            sf.write(str(out_path), trimmed, src_sr)
            seg.audio_path = out_path
            kept += 1

            if (i + 1) % 10 == 0 or i == 0 or i == len(valid_segs) - 1:
                orig_ms = round(len(data) / src_sr * 1000)
                kept_ms = round(len(trimmed) / src_sr * 1000)
                self.logger.info(
                    f"  [{i+1}/{len(valid_segs)}] {seg.id}: "
                    f"{orig_ms}ms → {kept_ms}ms "
                    f"(region {region[0]/16000*1000:.0f}–{region[1]/16000*1000:.0f}ms)"
                )

        self.logger.info(
            f"Speaker trim complete: {kept} kept, {discarded} discarded"
        )
        return ctx
