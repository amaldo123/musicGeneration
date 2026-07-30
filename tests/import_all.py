"""Import every executable Python surface so import failures are fatal in CI."""

from __future__ import annotations

import importlib
import pkgutil

import aimusic


def import_all_executable_modules() -> tuple[str, ...]:
    """Import all package modules and the repository's root UI entrypoint."""
    module_names = tuple(
        sorted(
            module.name
            for module in pkgutil.walk_packages(
                aimusic.__path__,
                prefix=f"{aimusic.__name__}.",
            )
        )
    )
    for module_name in module_names:
        importlib.import_module(module_name)
    importlib.import_module("ui")
    return (*module_names, "ui")


if __name__ == "__main__":
    imported = import_all_executable_modules()
    print(f"Imported {len(imported)} executable modules.")
