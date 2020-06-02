cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(sdf_cuda LANGUAGES CXX CUDA)

include(CTest)

add_library(sdf_lib STATIC
        includes/compute_sdf.h
        src/compute_sdf.cu
        )

target_compile_features(sdf_lib PUBLIC cxx_std_11)

set_target_properties(sdf_lib
                      PROPERTIES CUDA_SEPARABLE_COMPILATION ON
                      )

include_directories(utils)
include_directories(includes)
add_subdirectory(3rd_party)

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

add_executable(sdf_test app/test_sdf.cpp)

set_target_properties(sdf_test PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
 set_target_properties(sdf_test PROPERTIES CXX_STANDARD 11 CXX_STANDARD_REQUIRED YES)

include_directories(sdf_test ${EIGEN3_INCLUDE_DIR})
target_link_libraries(sdf_test PRIVATE sdf_lib
                               polyscope)
