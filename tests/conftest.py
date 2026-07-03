from __future__ import annotations

import pytest


def _jax_available() -> bool:
    try:
        import jax  # noqa: F401

        return True
    except ImportError:
        return False


requires_jax = pytest.mark.skipif(
    not _jax_available(),
    reason="JAX not installed; pip install -e '.[ml]'",
)


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "ml: tests requiring optional JAX/ML dependencies",
    )
