cmake_minimum_required(VERSION 3.6)
project(fft)

set(CMAKE_CXX_STANDARD 17)

add_library(fft STATIC fft.cpp scalar_impl.cpp sse_impl.cpp avx_impl.cpp)
target_compile_options(fft PRIVATE -mavx2 -mfma -fno-fast-math)

add_library(testing testing.cpp)
target_link_libraries(testing fft)

add_executable(run run.cpp)
target_link_libraries(run testing)

add_executable(run_all run_all.cpp)
target_link_libraries(run_all testing pthread)
