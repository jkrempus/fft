cmake_minimum_required(VERSION 3.6)
project(fft)

set(CMAKE_CXX_STANDARD 17)

add_library(fft STATIC fft.cpp)
target_compile_options(fft PRIVATE -mavx2 -mfma)

add_library(testing testing.cpp)
target_link_libraries(testing fft)

add_executable(run run.cpp)
target_link_libraries(run testing)
