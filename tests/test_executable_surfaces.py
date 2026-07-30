from __future__ import annotations

import unittest

from tests.import_all import import_all_executable_modules


class TestExecutableSurfaces(unittest.TestCase):
    def test_every_executable_module_imports(self) -> None:
        imported = import_all_executable_modules()

        self.assertIn("aimusic.app.cli", imported)
        self.assertIn("aimusic.app.main", imported)
        self.assertIn("ui", imported)


if __name__ == "__main__":
    unittest.main()
