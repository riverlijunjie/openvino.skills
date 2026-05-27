cmake_minimum_required(VERSION 3.16)
project(gemm_f16_test LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenCL
find_package(OpenCL REQUIRED)

add_executable(test_gemm_f16 test_gemm_f16.cpp)
target_link_libraries(test_gemm_f16 PRIVATE OpenCL::OpenCL)
target_compile_options(test_gemm_f16 PRIVATE -O2)

# Copy kernel file to build directory
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/gemm_f16.cl
               ${CMAKE_CURRENT_BINARY_DIR}/gemm_f16.cl COPYONLY)
