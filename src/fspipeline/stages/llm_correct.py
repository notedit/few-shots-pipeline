"""Stage 5: LLM-based transcript correction via OpenRouter."""

from __future__ import annotations

import json

from openai import OpenAI

from ..config import PipelineConfig
from ..models import AudioSegment, PipelineContext
from .base import PipelineStage

SYSTEM_PROMPT = """You are a multilingual speech transcript corrector. You will receive a batch of ASR transcriptions with surrounding context. Fix:
1. Homophones / misheard words based on context
2. Punctuation and sentence boundaries
3. Obvious ASR errors

Rules:
- Preserve the original language (do not translate)
- Keep the meaning unchanged
- Return ONLY a JSON array of corrected strings, one per input segment, in the same order
- Do not add explanations"""


class LlmCorrectStage(PipelineStage):
    name = "llm_correct"

    def __init__(self, config: PipelineConfig):
        super().__init__(config)
        self.cfg = config.llm

    def validate(self, ctx: PipelineContext) -> None:
        valid = [s for s in ctx.segments if s.is_valid]
        if not valid:
            raise ValueError("No valid segments to correct.")
        if not self.cfg.api_key:
            raise ValueError("LLM api_key not configured. Set llm.api_key in config.")

    def _build_prompt(self, batch: list[AudioSegment], all_segments: list[AudioSegment]) -> str:
        """Build prompt with context from surrounding segments."""
        lines = []
        for seg in batch:
            # Find index in full list for context
            idx = next(i for i, s in enumerate(all_segments) if s.id == seg.id)
            prev_text = all_segments[idx - 1].transcript_final if idx > 0 else ""
            next_text = (
                all_segments[idx + 1].transcript_final
                if idx < len(all_segments) - 1
                else ""
            )
            lines.append(
                f"[prev: {prev_text}] {seg.transcript_final} [next: {next_text}]"
            )
        return "\n".join(f"{i+1}. {line}" for i, line in enumerate(lines))

    def run(self, ctx: PipelineContext) -> PipelineContext:
        client = OpenAI(
            base_url=self.cfg.base_url,
            api_key=self.cfg.api_key,
        )

        valid_segments = [s for s in ctx.segments if s.is_valid]
        batch_size = self.cfg.batch_size

        self.logger.info(
            f"Correcting {len(valid_segments)} segments in batches of {batch_size}"
        )

        for batch_start in range(0, len(valid_segments), batch_size):
            batch = valid_segments[batch_start : batch_start + batch_size]
            prompt = self._build_prompt(batch, ctx.segments)

            try:
                response = client.chat.completions.create(
                    model=self.cfg.model,
                    temperature=self.cfg.temperature,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                )

                content = response.choices[0].message.content.strip()
                # Parse JSON array from response
                # Handle potential markdown code blocks
                if content.startswith("```"):
                    content = content.split("\n", 1)[1].rsplit("```", 1)[0].strip()

                corrections = json.loads(content)

                for seg, corrected in zip(batch, corrections):
                    seg.transcript_final = str(corrected).strip()

                self.logger.info(
                    f"Corrected batch {batch_start // batch_size + 1} "
                    f"({len(batch)} segments)"
                )

            except Exception as e:
                self.logger.warning(
                    f"LLM correction failed for batch starting at {batch_start}: {e}. "
                    f"Keeping original transcripts."
                )

        return ctx
