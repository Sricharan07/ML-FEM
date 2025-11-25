@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2017\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
cd /d D:\Research-work\Examples\FEM-ML
rmdir /s /q build 2>nul
mkdir build
cd build
cmake -G "Ninja" ^
  -DCMAKE_MAKE_PROGRAM=D:\Research-work\Examples\FEM-ML\build\ninja.exe ^
  -DFEMML_USE_GPU=ON ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DPython3_EXECUTABLE=C:/Users/onlys/AppData/Local/Programs/Python/Python311/python.exe ^
  -Dpybind11_DIR=C:/Users/onlys/AppData/Local/Programs/Python/Python311/Lib/site-packages/pybind11/share/cmake/pybind11 ^
  ..
if %ERRORLEVEL% EQU 0 (
    echo.
    echo =====================================
    echo Configuration successful!
    echo Now run: cmake --build build
    echo =====================================
) else (
    echo.
    echo =====================================
    echo Configuration failed!
    echo =====================================
)
