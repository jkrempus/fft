cmake_minimum_required(VERSION 3.6)
project(fft)

set(CMAKE_CXX_STANDARD 17)

add_library(fft STATIC fft.cpp)
target_compile_options(fft PRIVATE -mavx2 -mfma)

add_executable(run test.cpp)
target_link_libraries(run fft)
