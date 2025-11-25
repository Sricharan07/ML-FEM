# Core C++ library sources
set(FEMML_CORE_SOURCES
    mesh/Mesh.cpp
    mesh/AbaqusImporter.cpp
    element/HexElement.cpp
    element/TetElement.cpp
    material/LinearElastic.cpp
    material/NeuralMaterial.cpp
    solver/ExplicitSolver.cpp
    integration/CentralDifference.cpp
    io/ResultsWriter.cpp
)

set(FEMML_CORE_HEADERS
    ${CMAKE_SOURCE_DIR}/include/mesh/Mesh.hpp
    ${CMAKE_SOURCE_DIR}/include/mesh/AbaqusImporter.hpp
    ${CMAKE_SOURCE_DIR}/include/element/Element.hpp
    ${CMAKE_SOURCE_DIR}/include/element/HexElement.hpp
    ${CMAKE_SOURCE_DIR}/include/element/TetElement.hpp
    ${CMAKE_SOURCE_DIR}/include/material/Material.hpp
    ${CMAKE_SOURCE_DIR}/include/material/LinearElastic.hpp
    ${CMAKE_SOURCE_DIR}/include/material/NeuralMaterial.hpp
    ${CMAKE_SOURCE_DIR}/include/solver/ExplicitSolver.hpp
    ${CMAKE_SOURCE_DIR}/include/integration/TimeIntegrator.hpp
    ${CMAKE_SOURCE_DIR}/include/integration/CentralDifference.hpp
    ${CMAKE_SOURCE_DIR}/include/io/ResultsWriter.hpp
)

# Add GPU sources if CUDA is enabled
if(FEMML_USE_GPU AND CUDAToolkit_FOUND)
    list(APPEND FEMML_CORE_SOURCES solver/GPUExplicitSolver.cu)
    list(APPEND FEMML_CORE_HEADERS ${CMAKE_SOURCE_DIR}/include/solver/GPUExplicitSolver.hpp)
    
    message(STATUS "GPU support enabled - compiling CUDA kernels")
endif()

# Create core library
add_library(femml_core ${FEMML_CORE_SOURCES} ${FEMML_CORE_HEADERS})

target_include_directories(femml_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(femml_core PUBLIC
    ${MFEM_LIBRARIES}
)

# Link CUDA libraries if GPU is enabled
if(FEMML_USE_GPU AND CUDAToolkit_FOUND)
    target_link_libraries(femml_core PUBLIC CUDA::cudart)
    target_compile_definitions(femml_core PUBLIC FEMML_USE_GPU)
endif()

# Create standalone solver executable
add_executable(femml_solver main.cpp)
target_link_libraries(femml_solver femml_core)

# Install
install(TARGETS femml_core femml_solver
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin)
