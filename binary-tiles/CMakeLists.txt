cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(LargestSquareWavefrontTests LANGUAGES CXX CUDA)

find_package(GTest REQUIRED)

include_directories(${GTEST_INCLUDE_DIRS})

add_executable(test_wavefront
    test_wavefront.cu
    wavefront.cu   # your wavefront kernel code
    cpu_reference.h           # optional file with CPU code
)

target_link_libraries(test_wavefront 
    PRIVATE 
      GTest::gtest
      GTest::gtest_main
      cuda
      # ... other libs if needed
)

set_target_properties(test_wavefront 
    PROPERTIES
      CUDA_SEPARABLE_COMPILATION ON
      CXX_STANDARD 14
      CUDA_ARCHITECTURES "90"
)

enable_testing()
add_test(NAME largestSquareWavefrontTests
         COMMAND test_wavefront)
