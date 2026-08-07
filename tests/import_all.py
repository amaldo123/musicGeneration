"""Import every executable Python surface so import failures are fatal in CI."""

from __future__ import annotations

import importlib
import pkgutil

import aimusic


OPTIONAL_MODULE_PREFIXES = ("aimusic.ml.",)


def import_all_executable_modules() -> tuple[str, ...]:
    """Import executable surfaces available from the base installation."""
    module_names = tuple(
        sorted(
            module.name
            for module in pkgutil.walk_packages(
                aimusic.__path__,
                prefix=f"{aimusic.__name__}.",
            )
            if not module.name.startswith(OPTIONAL_MODULE_PREFIXES)
        )
    )
    for module_name in module_names:
        importlib.import_module(module_name)
    importlib.import_module("ui")
    return (*module_names, "ui")


if __name__ == "__main__":
    imported = import_all_executable_modules()
    print(f"Imported {len(imported)} executable modules.")
