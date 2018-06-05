cmake_minimum_required(VERSION 3.6)
project(fft)

add_library(fft STATIC fft.cpp)
target_compile_options(fft PRIVATE -std=c++17 -mavx2 -mfma)

add_executable(run test.cpp)
target_link_libraries(run fft)
