from pathlib import Path
import subprocess
import sys

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


def read_long_description() -> str:
	for candidate in ("README.md", "readme.md"):
		path = Path(candidate)
		if path.exists():
			return path.read_text(encoding="utf-8")
	return "T-Tensor: Mini-framework estilo PyTorch con autograd y aceleracion CUDA."


class CMakeExtension(Extension):
	def __init__(self, name: str, sourcedir: str = ".") -> None:
		super().__init__(name, sources=[])
		self.sourcedir = str(Path(sourcedir).resolve())


class CMakeBuild(build_ext):
	def build_extension(self, ext: Extension) -> None:
		ext_fullpath = Path(self.get_ext_fullpath(ext.name)).resolve()
		extdir = ext_fullpath.parent
		cfg = "Debug" if self.debug else "Release"

		build_temp = Path(self.build_temp) / ext.name
		build_temp.mkdir(parents=True, exist_ok=True)

		cmake_args = [
			f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{cfg.upper()}={extdir}",
			f"-DPython3_EXECUTABLE={sys.executable}",
			f"-DCMAKE_BUILD_TYPE={cfg}",
		]

		# En builds aislados (python -m build), CMake a veces no encuentra pybind11.
		# Intentamos resolver su cmake dir con el propio interprete activo.
		try:
			pybind11_cmake_dir = subprocess.check_output(
				[sys.executable, "-m", "pybind11", "--cmakedir"],
				text=True,
			).strip()
			if pybind11_cmake_dir:
				cmake_args.append(f"-Dpybind11_DIR={pybind11_cmake_dir}")
		except Exception:
			# Si no se puede resolver, dejamos que CMake intente su busqueda normal.
			pass

		if sys.platform.startswith("win"):
			cmake_args.append("-DCMAKE_CUDA_FLAGS=-allow-unsupported-compiler")
		build_args = ["--config", cfg, "--target", "ttensor_python"]

		subprocess.run(["cmake", ext.sourcedir, *cmake_args], cwd=build_temp, check=True)
		subprocess.run(["cmake", "--build", ".", *build_args], cwd=build_temp, check=True)


setup(
	name="ttensor",
	version="0.1.0",
	description="Mini-framework con tensores, optimizadores y aceleracion CUDA",
	long_description=read_long_description(),
	long_description_content_type="text/markdown",
	author="Gilberto Galán",
	license="MIT",
	keywords=["deep-learning", "autograd", "cuda", "neural-networks", "tensor"],
	classifiers=[
		"Development Status :: 3 - Alpha",
		"Intended Audience :: Developers",
		"Intended Audience :: Science/Research",
		"License :: OSI Approved :: MIT License",
		"Programming Language :: Python :: 3",
		"Programming Language :: Python :: 3.12",
		"Programming Language :: C++",
		"Topic :: Scientific/Engineering :: Artificial Intelligence",
		"Operating System :: Microsoft :: Windows",
	],
	python_requires=">=3.12",
	install_requires=[
		"pybind11>=2.11",
	],
	ext_modules=[CMakeExtension("ttensor")],
	data_files=[("", ["ttensor_cuda_path.pth"])],
	cmdclass={"build_ext": CMakeBuild},
	zip_safe=False,
)
