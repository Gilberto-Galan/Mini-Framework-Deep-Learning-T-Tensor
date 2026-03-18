"""API Python de T-Tensor.

Intenta importar el modulo nativo ``_ttensor`` dentro del paquete.
Si no existe (por ejemplo al ejecutar desde el repo), busca un .pyd
compilado en rutas comunes de build de CMake.
"""

from __future__ import annotations

import importlib
from importlib import util as importlib_util
from pathlib import Path
import sys


def _load_native_module():
	try:
		return importlib.import_module("ttensor._ttensor")
	except ModuleNotFoundError:
		package_dir = Path(__file__).resolve().parent
		repo_root = package_dir.parent

		candidate_dirs = [
			repo_root / "build" / "Release",
			repo_root / "build" / "x64" / "Release",
			repo_root / "build-vs18" / "Release",
			repo_root / "build-vs18" / "x64" / "Release",
		]

		for candidate_dir in candidate_dirs:
			if not candidate_dir.exists():
				continue

			matches = sorted(candidate_dir.glob("_ttensor*.pyd"))
			if not matches:
				continue

			module_path = matches[0]
			spec = importlib_util.spec_from_file_location("ttensor._ttensor", module_path)
			if spec is None or spec.loader is None:
				continue

			module = importlib_util.module_from_spec(spec)
			spec.loader.exec_module(module)
			sys.modules["ttensor._ttensor"] = module
			return module

		raise ModuleNotFoundError(
			"No se encontro 'ttensor._ttensor'. Compila e instala el modulo nativo con:\n"
			"  cmake -S . -B build-vs18 -G \"Visual Studio 18 2026\" -A x64 "
			"-DCMAKE_CUDA_FLAGS=\"--allow-unsupported-compiler\" "
			"-Dpybind11_DIR=\"<ruta>/pybind11/share/cmake/pybind11\"\n"
			"  cmake --build build-vs18 --config Release\n"
			"  python -m pip install -e .\n"
		)


_native = _load_native_module()

for _name in dir(_native):
	if not _name.startswith("_"):
		globals()[_name] = getattr(_native, _name)

__all__ = [name for name in globals() if not name.startswith("_")]
