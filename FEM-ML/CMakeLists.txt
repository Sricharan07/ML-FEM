cmake_minimum_required(VERSION 3.12)
project(FEM-ML VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Options
option(FEMML_USE_GPU "Enable GPU acceleration with CUDA" OFF)
option(FEMML_USE_MFEM "Use MFEM library" ON)
option(FEMML_USE_NEURAL "Enable neural network support" ON)
option(FEMML_BUILD_PYTHON "Build Python bindings" ON)
option(FEMML_BUILD_GUI "Build GUI application" ON)
option(FEMML_BUILD_TESTS "Build tests" ON)
option(FEMML_BUILD_EXAMPLES "Build examples" ON)

# Enable CUDA if requested
if(FEMML_USE_GPU)
    find_package(CUDAToolkit QUIET)
    if(CUDAToolkit_FOUND)
        enable_language(CUDA)
        message(STATUS "CUDA Toolkit found: ${CUDAToolkit_VERSION}")
        message(STATUS "CUDA Libraries: ${CUDAToolkit_LIBRARY_DIR}")
        set(CMAKE_CUDA_STANDARD 14)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
        # Respect user-provided architectures but fall back to a sensible default
        if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
            message(STATUS "Using default CUDA architectures: 60 61 70 75 80 86")
            set(CMAKE_CUDA_ARCHITECTURES 60 61 70 75 80 86)
        else()
            message(STATUS "Using user-specified CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
        endif()
    else()
        message(WARNING "CUDA Toolkit not found, GPU acceleration disabled")
        set(FEMML_USE_GPU OFF)
    endif()
endif()

# Find packages
if(FEMML_USE_MFEM)
    set(FEMML_MFEM_DIR "${CMAKE_SOURCE_DIR}/../mfem-master" CACHE PATH "Path to MFEM sources or install")
    if(NOT DEFINED MFEM_DIR OR MFEM_DIR STREQUAL "MFEM_DIR-NOTFOUND")
        set(MFEM_DIR "${FEMML_MFEM_DIR}")
    endif()
    set(_MFEM_MANUAL_DIR "${MFEM_DIR}")
    find_package(MFEM QUIET)
    if(MFEM_FOUND)
        message(STATUS "MFEM found via find_package")
    elseif(_MFEM_MANUAL_DIR AND EXISTS "${_MFEM_MANUAL_DIR}")
        message(STATUS "MFEM not found via find_package, using manual include path: ${_MFEM_MANUAL_DIR}")
        set(MFEM_INCLUDE_DIRS "${_MFEM_MANUAL_DIR}")
        set(MFEM_LIBRARIES "")
    else()
        message(WARNING "MFEM not found and manual directory missing; disabling MFEM-dependent features")
        set(FEMML_USE_MFEM OFF)
        set(MFEM_INCLUDE_DIRS "")
        set(MFEM_LIBRARIES "")
    endif()
endif()

if(FEMML_USE_NEURAL)
    # ONNX Runtime for NN inference
    find_package(onnxruntime QUIET)
    if(NOT onnxruntime_FOUND)
        message(STATUS "ONNX Runtime not found, neural network support will be limited")
    endif()
endif()

if(FEMML_BUILD_PYTHON)
    find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
    find_package(pybind11 QUIET)
    if(NOT pybind11_FOUND)
        message(STATUS "pybind11 not found, Python bindings disabled")
        set(FEMML_BUILD_PYTHON OFF)
    endif()
endif()

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${MFEM_INCLUDE_DIRS}
)

# Core library
add_subdirectory(src/core)

# Python bindings
if(FEMML_BUILD_PYTHON)
    add_subdirectory(src/python)
endif()

# Examples
if(FEMML_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# Tests
if(FEMML_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Installation
install(DIRECTORY include/ DESTINATION include/femml)
install(DIRECTORY src/python/ DESTINATION lib/python/femml
        FILES_MATCHING PATTERN "*.py")
