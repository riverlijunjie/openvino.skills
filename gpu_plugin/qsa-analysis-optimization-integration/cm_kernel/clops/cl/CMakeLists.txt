cmake_minimum_required(VERSION 3.23)
project(csrc)

find_package(pybind11 REQUIRED)

# ── Source files ──────────────────────────────────────────────────────────────
# ops.cpp : tensor helpers
# cl.cpp  : OpenCL/CM kernel API, profiling, tensor memory management
FILE(GLOB CPP_SRCS ./*.cpp)

pybind11_add_module(csrc ${CPP_SRCS})
target_compile_features(csrc PRIVATE cxx_std_17)

find_package(OpenMP)
if(OPENMP_FOUND)
    target_compile_options(csrc PRIVATE ${OpenMP_CXX_FLAGS})
    target_link_libraries(csrc PRIVATE ${OpenMP_CXX_LIBRARIES})
endif()

if(MSVC)
    target_compile_options(csrc PRIVATE /arch:AVX2)
else()
    target_compile_options(csrc PRIVATE -march=core-avx2)
endif()

# ── OpenCL ────────────────────────────────────────────────────────────────────
target_include_directories(csrc PRIVATE ../../include)
# On Linux the ICD loader's headers/libs live in default system search paths,
# so bare "OpenCL" resolves via -lOpenCL with no extra path needed. Windows has
# no such default location (e.g. a Khronos OpenCL-SDK install here), so this
# opt-in env var lets a Windows setup point at one without touching the
# unmodified Linux path. MSBuild's per-project IncludePath/LibraryPath does
# not inherit the process INCLUDE/LIB env vars by default, so this must be
# wired in as explicit target directories, not left to the environment.
if(DEFINED ENV{OPENCL_SDK_ROOT})
    target_include_directories(csrc PRIVATE "$ENV{OPENCL_SDK_ROOT}/include")
    target_link_directories(csrc PRIVATE "$ENV{OPENCL_SDK_ROOT}/lib")
endif()
target_link_libraries(csrc PRIVATE OpenCL)

target_compile_definitions(csrc PRIVATE CL_TARGET_OPENCL_VERSION=300)

set_target_properties(csrc
    PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/../"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/../"
)
# Multi-config generators (e.g. Visual Studio, used on Windows since no Ninja/
# Makefiles toolchain is available there) append a per-configuration
# subdirectory to LIBRARY/RUNTIME_OUTPUT_DIRECTORY unless the *_<CONFIG>
# variant is also set; Windows extension modules (.pyd) are DLLs and are
# placed by RUNTIME_OUTPUT_DIRECTORY, not LIBRARY_OUTPUT_DIRECTORY. Pin both
# so the built module always lands next to this CMakeLists.txt (clops/cl/),
# matching the single-config Linux layout that clops/cl/__init__.py expects.
foreach(qsa_cfg ${CMAKE_CONFIGURATION_TYPES})
    string(TOUPPER ${qsa_cfg} qsa_cfg_upper)
    set_target_properties(csrc PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY_${qsa_cfg_upper} "${CMAKE_BINARY_DIR}/../"
        RUNTIME_OUTPUT_DIRECTORY_${qsa_cfg_upper} "${CMAKE_BINARY_DIR}/../")
endforeach()

target_compile_definitions(csrc PRIVATE PACKAGE_NAME=csrc)
