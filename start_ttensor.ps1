param(
    [switch]$TestImport,
    [switch]$BuildRelease,
    [switch]$ConfigureOnly,
    [string]$Generator = "Visual Studio 18 2026",
    [string]$Arch = "x64",
    [string]$BuildDir = "build",
    [string]$CudaBin = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin",
    [string]$Pybind11Dir = ""
)

$ErrorActionPreference = "Stop"

# Siempre trabaja desde la raiz del proyecto
Set-Location -Path $PSScriptRoot

$venvActivate = Join-Path $PSScriptRoot "venv\Scripts\Activate.ps1"
if (-not (Test-Path $venvActivate)) {
    throw "No se encontro el entorno virtual en venv\Scripts\Activate.ps1"
}

# Activar entorno virtual
. $venvActivate

# Inyectar CUDA al PATH solo para esta sesion
if (-not (Test-Path $CudaBin)) {
    Write-Warning "No se encontro CUDA bin en: $CudaBin"
} elseif (-not ($env:PATH -like "*$CudaBin*")) {
    $env:PATH = "$CudaBin;$env:PATH"
}

Write-Host "[T-Tensor] venv activo: $($env:VIRTUAL_ENV)"
Write-Host "[T-Tensor] CUDA bin: $CudaBin"
Write-Host "[T-Tensor] Python: $(python --version)"

if (-not $Pybind11Dir) {
    $Pybind11Dir = Join-Path $PSScriptRoot "venv\Lib\site-packages\pybind11\share\cmake\pybind11"
}

if ($BuildRelease -or $ConfigureOnly) {
    if (-not (Test-Path $Pybind11Dir)) {
        throw "No se encontro pybind11_DIR en: $Pybind11Dir"
    }

    Write-Host "[T-Tensor] Configurando CMake con '$Generator'..."
    cmake -S . -B $BuildDir -G $Generator -A $Arch -DCMAKE_CUDA_FLAGS="--allow-unsupported-compiler" -Dpybind11_DIR="$Pybind11Dir"
    if ($LASTEXITCODE -ne 0) {
        throw "Fallo la configuracion de CMake."
    }

    if ($BuildRelease) {
        Write-Host "[T-Tensor] Compilando Release..."
        cmake --build $BuildDir --config Release
        if ($LASTEXITCODE -ne 0) {
            throw "Fallo la compilacion Release."
        }
    }
}

if ($TestImport) {
    python -c "import os; os.add_dll_directory(r'$CudaBin'); import ttensor; print('ttensor import OK')"
    if ($LASTEXITCODE -ne 0) {
        throw "Fallo la validacion de import de ttensor."
    }
    Write-Host "[T-Tensor] Import validado correctamente."
}

Write-Host "[T-Tensor] Sesion lista."
