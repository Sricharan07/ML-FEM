# Examples

if(NOT FEMML_BUILD_EXAMPLES)
    return()
endif()

message(STATUS "Configuring examples")

# Simple cube example
add_executable(example_cube simple_cube/main.cpp)
target_link_libraries(example_cube femml_core)

install(TARGETS example_cube DESTINATION bin/examples)
install(FILES simple_cube/cube.inp simple_cube/config.inp
        DESTINATION share/femml/examples/simple_cube)
