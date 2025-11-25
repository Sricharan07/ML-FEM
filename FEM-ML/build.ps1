# FEM-ML Build Script for Windows (PowerShell)
# Usage: .\build.ps1 [Release|Debug]

param(
    [string]$BuildType = "Release",
    [ValidateSet("Win32", "x64")]
    [string]$Architecture = "x64",
    [string]$MfemDir = "",
    [string]$Pybind11Dir = "",
    [string]$PythonExe = "",
    [bool]$EnableGpu = $false
)

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "    FEM-ML Build Script" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

# Check for CMake
if (-not (Get-Command cmake -ErrorAction SilentlyContinue)) {
    Write-Host "Error: CMake not found. Please install CMake." -ForegroundColor Red
    exit 1
}

Write-Host "Build Type: $BuildType" -ForegroundColor Green
Write-Host "Target Architecture: $Architecture" -ForegroundColor Green

# Create build directory and ensure cache matches target architecture
$buildDir = "build"
$cacheFile = Join-Path $buildDir "CMakeCache.txt"
if (Test-Path $cacheFile) {
    $currentPlatform = ""
    $platformMatch = Select-String -Path $cacheFile -Pattern "CMAKE_GENERATOR_PLATFORM:INTERNAL=(.*)$" -ErrorAction SilentlyContinue
    if ($platformMatch -and $platformMatch.Matches.Count -gt 0) {
        $currentPlatform = $platformMatch.Matches[0].Groups[1].Value.Trim()
    }
    if ($currentPlatform -ne $Architecture) {
        Write-Host "Cleaning build directory (was '$currentPlatform', needs '$Architecture')..." -ForegroundColor Yellow
        Remove-Item -Recurse -Force $buildDir
    }
}
if (-not (Test-Path $buildDir)) {
    Write-Host "Creating build directory..." -ForegroundColor Yellow
    New-Item -ItemType Directory -Path $buildDir | Out-Null
}

# Configure
if (-not $MfemDir) {
    $defaultMfem = Join-Path $PSScriptRoot "..\\mfem-master"
    if (Test-Path $defaultMfem) {
        $MfemDir = (Resolve-Path $defaultMfem).Path
    }
}
if ($MfemDir) {
    Write-Host "Using MFEM directory: $MfemDir" -ForegroundColor Green
}

if (-not $PythonExe) {
    try {
        $pythonProbe = python -c "import sys; print(sys.executable)" 2>$null
        if ($LASTEXITCODE -eq 0) {
            $candidate = $pythonProbe.Trim()
            if (Test-Path $candidate) {
                $PythonExe = $candidate
            }
        }
    }
    catch {
        # ignore
    }
}
if ($PythonExe) {
    Write-Host "Using Python executable: $PythonExe" -ForegroundColor Green
}

if (-not $Pybind11Dir) {
    $pybindQuery = "import pathlib, pybind11; print(pathlib.Path(pybind11.__file__).parent / 'share' / 'cmake' / 'pybind11')"
    try {
        $probe = python -c $pybindQuery 2>$null
        if ($LASTEXITCODE -eq 0) {
            $candidate = $probe.Trim()
            if (Test-Path $candidate) {
                $Pybind11Dir = $candidate
            }
        }
    }
    catch {
        # Ignore detection failures
    }
}
if ($Pybind11Dir) {
    Write-Host "Using pybind11 directory: $Pybind11Dir" -ForegroundColor Green
}

$gpuOption = if ($EnableGpu) { "ON" } else { "OFF" }

$cmakeArgs = @(
    "-S", ".",
    "-B", $buildDir,
    "-A", $Architecture,
    "-DCMAKE_BUILD_TYPE=$BuildType",
    "-DFEMML_BUILD_EXAMPLES=ON",
    "-DFEMML_BUILD_PYTHON=ON",
    "-DFEMML_BUILD_TESTS=OFF",
    "-DFEMML_USE_GPU=$gpuOption"
)
if ($MfemDir) {
    $cmakeArgs += "-DMFEM_DIR=$MfemDir"
}
if ($Pybind11Dir) {
    $cmakeArgs += "-Dpybind11_DIR=$Pybind11Dir"
}
if ($PythonExe) {
    $cmakeArgs += "-DPython3_EXECUTABLE=$PythonExe"
}

Write-Host "`nConfiguring project..." -ForegroundColor Yellow
cmake @cmakeArgs

if ($LASTEXITCODE -ne 0) {
    Write-Host "`nConfiguration failed!" -ForegroundColor Red
    exit 1
}

# Build
Write-Host "`nBuilding project..." -ForegroundColor Yellow
cmake --build build --config $BuildType

if ($LASTEXITCODE -ne 0) {
    Write-Host "`nBuild failed!" -ForegroundColor Red
    exit 1
}

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "    Build Complete!" -ForegroundColor Green
Write-Host "========================================`n" -ForegroundColor Cyan

Write-Host "Executable: build\bin\$BuildType\femml_solver.exe" -ForegroundColor Green
Write-Host "Python module: build\src\python\femml.pyd`n" -ForegroundColor Green

Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "  1. Run example: cd examples\simple_cube && python run_example.py"
Write-Host "  2. Launch GUI: python src\python\gui\main_window.py"
Write-Host "  3. Run tests: cd build && ctest`n"
