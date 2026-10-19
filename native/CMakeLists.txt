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
set(POWER_SANITIZE "" CACHE STRING
    "Sanitizers to build everything with, e.g. address,undefined or thread")

# Applied to every target, GoogleTest included: ThreadSanitizer in particular
# reports false races when only part of the program is instrumented.
if(POWER_SANITIZE)
    if(MSVC)
        # MSVC has AddressSanitizer and nothing else.
        if(NOT POWER_SANITIZE STREQUAL "address")
            message(FATAL_ERROR "MSVC supports POWER_SANITIZE=address only")
        endif()
        add_compile_options(/fsanitize=address)
    else()
        add_compile_options(-fsanitize=${POWER_SANITIZE} -fno-omit-frame-pointer
                            -fno-sanitize-recover=all)
        add_link_options(-fsanitize=${POWER_SANITIZE})
    endif()
endif()

# Warnings for our own targets only — third-party headers come in as SYSTEM.
add_library(power_warnings INTERFACE)
if(MSVC)
    # Without this MSVC reports __cplusplus as 199711 whatever the standard,
    # and build_info() would claim C++98.
    add_compile_options(/Zc:__cplusplus)
    # /permissive- for standard conformance, /utf-8 because the sources are.
    target_compile_options(power_warnings INTERFACE
        /W4 /permissive- /utf-8 $<$<BOOL:${POWER_WARNINGS_AS_ERRORS}>:/WX>)
    # GoogleTest otherwise links the static CRT while everything else links
    # the DLL one, and the test executable fails to link.
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
else()
    target_compile_options(power_warnings INTERFACE
        -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wsign-conversion
        $<$<BOOL:${POWER_WARNINGS_AS_ERRORS}>:-Werror>)
endif()

# The sampler itself: plain C++, no Python. Both the module and the tests
# link it, which is what keeps it testable without an interpreter.
find_package(Threads REQUIRED)

add_library(power_core STATIC src/dynlib.cpp src/nvml_api.cpp src/sampler.cpp)
target_include_directories(power_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(power_core PUBLIC cxx_std_17)
# dlopen lives in libdl on glibc older than 2.34; empty where it is in libc.
target_link_libraries(power_core PUBLIC ${CMAKE_DL_LIBS} Threads::Threads PRIVATE power_warnings)
if(WIN32)
    # timeBeginPeriod, for the sampler's 1 ms timer resolution.
    target_link_libraries(power_core PUBLIC winmm)
endif()

if(POWER_BUILD_PYTHON)
    find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module)
    find_package(pybind11 CONFIG REQUIRED)

    pybind11_add_module(nvml_sampler src/bindings.cpp)
    target_link_libraries(nvml_sampler PRIVATE power_core power_warnings)
    target_compile_definitions(nvml_sampler PRIVATE
        NVML_SAMPLER_VERSION="${PROJECT_VERSION}"
        NVML_SAMPLER_BUILD_TYPE="${CMAKE_BUILD_TYPE}")

    # Straight into site-packages, so the import name is just nvml_sampler.
    install(TARGETS nvml_sampler LIBRARY DESTINATION .)
endif()

# Defines BUILD_TESTING (ON by default). pip builds turn it off in
# pyproject.toml, so installing the module never downloads GoogleTest.
include(CTest)
if(BUILD_TESTING)
    include(FetchContent)
    # Pinned by version and hash, like every other tool in this repository.
    FetchContent_Declare(googletest
        URL https://github.com/google/googletest/archive/refs/tags/v1.17.0.tar.gz
        URL_HASH SHA256=65fab701d9829d38cb77c14acdc431d2108bfdbf8979e40eb8ae567edf10b27c
        SYSTEM)
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)

    # A fake libnvidia-ml, loaded by path through the real dlopen code. The
    # incomplete variant lacks an entry point, for the missing-symbol path.
    foreach(variant fake_nvml fake_nvml_incomplete)
        add_library(${variant} SHARED tests/fake_nvml.cpp)
        target_include_directories(${variant} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
        target_link_libraries(${variant} PRIVATE power_warnings)
    endforeach()
    target_compile_definitions(fake_nvml_incomplete PRIVATE FAKE_NVML_INCOMPLETE)

    add_executable(power_tests
        tests/test_dynlib.cpp
        tests/test_ring_buffer.cpp
        tests/test_nvml_api.cpp
        tests/test_sampler.cpp)
    target_link_libraries(power_tests PRIVATE power_core power_warnings GTest::gtest_main)
    target_compile_definitions(power_tests PRIVATE
        FAKE_NVML_PATH="$<TARGET_FILE:fake_nvml>"
        FAKE_NVML_INCOMPLETE_PATH="$<TARGET_FILE:fake_nvml_incomplete>")
    add_dependencies(power_tests fake_nvml fake_nvml_incomplete)

    include(GoogleTest)
    gtest_discover_tests(power_tests)
endif()
