"""Tests for fspipeline.utils.text"""

from __future__ import annotations

import pytest

from fspipeline.utils.text import normalize_text, text_similarity


class TestNormalizeText:
    def test_lowercase(self):
        assert normalize_text("Hello World") == "hello world"

    def test_punctuation_removed(self):
        assert normalize_text("Hello, World!") == "hello world"

    def test_whitespace_collapsed(self):
        assert normalize_text("hello   world") == "hello world"

    def test_leading_trailing_stripped(self):
        assert normalize_text("  hello  ") == "hello"

    def test_cjk_preserved(self):
        result = normalize_text("你好世界")
        assert "你好世界" in result

    def test_unicode_nfkc(self):
        # Full-width ASCII should be normalised to half-width
        result = normalize_text("ＨｅＬＬｏ")
        assert result == "hello"

    def test_empty_string(self):
        assert normalize_text("") == ""

    def test_digits_preserved(self):
        result = normalize_text("abc 123")
        assert "123" in result

    def test_mixed_language(self):
        result = normalize_text("Hello 你好 World!")
        assert "hello" in result
        assert "你好" in result


class TestTextSimilarity:
    def test_identical_strings(self):
        assert text_similarity("hello world", "hello world") == 1.0

    def test_completely_different(self):
        score = text_similarity("abc", "xyz")
        assert 0.0 <= score < 0.5

    def test_partial_overlap(self):
        score = text_similarity("hello world", "hello python")
        assert 0.0 < score < 1.0

    def test_both_empty(self):
        assert text_similarity("", "") == 1.0

    def test_one_empty(self):
        assert text_similarity("hello", "") == 0.0
        assert text_similarity("", "hello") == 0.0

    def test_case_insensitive(self):
        # Normalisation makes comparison case-insensitive
        assert text_similarity("Hello World", "hello world") == 1.0

    def test_punctuation_ignored(self):
        assert text_similarity("hello, world!", "hello world") == 1.0

    def test_symmetry(self):
        a, b = "foo bar baz", "foo qux"
        assert text_similarity(a, b) == text_similarity(b, a)

    def test_cjk_similarity(self):
        score = text_similarity("你好世界", "你好世界")
        assert score == 1.0

    def test_score_range(self):
        for t1, t2 in [("abc", "abcd"), ("x", "xyz"), ("the quick brown fox", "quick")]:
            s = text_similarity(t1, t2)
            assert 0.0 <= s <= 1.0, f"score out of range for ({t1!r}, {t2!r}): {s}"
