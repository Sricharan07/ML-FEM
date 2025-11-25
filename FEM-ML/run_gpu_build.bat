@echo off
setlocal
call "C:\Program Files (x86)\Microsoft Visual Studio\2017\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if %errorlevel% neq 0 exit /b %errorlevel%
set "PATH=C:\Users\onlys\AppData\Local\Programs\Python\Python311\Scripts;%PATH%"
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DFEMML_BUILD_EXAMPLES=ON -DFEMML_BUILD_PYTHON=ON -DFEMML_BUILD_TESTS=OFF -DFEMML_USE_GPU=ON -DMFEM_DIR=D:/Research-work/Examples/mfem-master -Dpybind11_DIR=C:/Users/onlys/AppData/Local/Programs/Python/Python311/Lib/site-packages/pybind11/share/cmake/pybind11 -DPython3_EXECUTABLE=C:/Users/onlys/AppData/Local/Programs/Python/Python311/python.exe -DCMAKE_CUDA_COMPILER="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0/bin/nvcc.exe" -DCMAKE_CUDA_FLAGS=--allow-unsupported-compiler -DCMAKE_CUDA_ARCHITECTURES=86
if %errorlevel% neq 0 exit /b %errorlevel%
cmake --build build --config Release
exit /b %errorlevel%
