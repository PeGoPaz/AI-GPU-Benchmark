# Native half of AI-GPU-Benchmark: a high-rate NVML power sampler.
#
# Optional by design. The app is Python and runs without this; when the
# module is built and importable, telemetry gains a millisecond-scale power
# trace that a 250 ms Python poll cannot provide.
#
#   pip install ./native                       — the Python module, via scikit-build-core
#   cmake -S native -B build -DPOWER_BUILD_PYTHON=OFF
#   cmake --build build && ctest --test-dir build   — the C++ tests alone, no Python needed

cmake_minimum_required(VERSION 3.25)

# The single source of the version: pyproject.toml reads it from this line.
project(nvml_sampler VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(POWER_BUILD_PYTHON "Build the nvml_sampler Python module" ON)
option(POWER_WARNINGS_AS_ERRORS "Treat compiler warnings as errors (CI)" OFF)

# Warnings for our own targets only — third-party headers come in as SYSTEM.
add_library(power_warnings INTERFACE)
target_compile_options(power_warnings INTERFACE
    -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wsign-conversion
    $<$<BOOL:${POWER_WARNINGS_AS_ERRORS}>:-Werror>)

if(POWER_BUILD_PYTHON)
    find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module)
    find_package(pybind11 CONFIG REQUIRED)

    pybind11_add_module(nvml_sampler src/bindings.cpp)
    target_link_libraries(nvml_sampler PRIVATE power_warnings)
    target_compile_definitions(nvml_sampler PRIVATE
        NVML_SAMPLER_VERSION="${PROJECT_VERSION}"
        NVML_SAMPLER_BUILD_TYPE="${CMAKE_BUILD_TYPE}")

    # Straight into site-packages, so the import name is just nvml_sampler.
    install(TARGETS nvml_sampler LIBRARY DESTINATION .)
endif()
