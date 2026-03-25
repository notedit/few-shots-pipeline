"""Pytest configuration and shared fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slow: marks tests that require model downloads or heavy computation",
    )
