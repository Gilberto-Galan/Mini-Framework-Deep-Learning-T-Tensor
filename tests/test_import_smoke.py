"""Smoke test basico para el paquete Python de T-Tensor."""

import unittest


class TestImportSmoke(unittest.TestCase):
    def test_import_ttensor(self) -> None:
        try:
            import ttensor  # noqa: F401
        except Exception as exc:
            self.fail(f"No se pudo importar ttensor: {exc}")


if __name__ == "__main__":
    unittest.main()
