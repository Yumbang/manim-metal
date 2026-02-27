"""Shared test fixtures for manim-metal tests."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _patch_manim():
    """Ensure manim-metal patches are applied before every test."""
    import manim_metal  # noqa: F401
