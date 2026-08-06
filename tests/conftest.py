from __future__ import annotations

import unittest

import pytest


def _jax_available() -> bool:
    try:
        import jax  # noqa: F401

        return True
    except ImportError:
        return False


HAS_JAX = _jax_available()

requires_jax = pytest.mark.skipif(
    not HAS_JAX,
    reason="JAX not installed; pip install -e '.[ml]'",
)

skip_unless_jax = unittest.skipUnless(
    HAS_JAX,
    "JAX not installed; pip install -e '.[ml]'",
)


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "ml: tests requiring optional JAX/ML dependencies",
    )
