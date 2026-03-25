"""Text normalization and similarity utilities."""

from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher


def normalize_text(text: str) -> str:
    """Normalize text for comparison: lowercase, remove punctuation, collapse whitespace."""
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    # Remove punctuation (keep CJK characters, letters, digits, spaces)
    text = re.sub(r"[^\w\s\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def text_similarity(text1: str, text2: str) -> float:
    """Compute normalized text similarity using SequenceMatcher."""
    t1 = normalize_text(text1)
    t2 = normalize_text(text2)
    if not t1 and not t2:
        return 1.0
    if not t1 or not t2:
        return 0.0
    return SequenceMatcher(None, t1, t2).ratio()
