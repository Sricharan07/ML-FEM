# Python bindings with pybind11

if(NOT pybind11_FOUND)
    message(STATUS "pybind11 not found, skipping Python bindings")
    return()
endif()

# Create Python module
pybind11_add_module(femml api/femml_bindings.cpp)

target_link_libraries(femml PRIVATE femml_core)

target_include_directories(femml PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

# Install Python module
install(TARGETS femml
        LIBRARY DESTINATION lib/python/femml)
