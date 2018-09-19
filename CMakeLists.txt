cmake_minimum_required(VERSION 3.6)
project(fft)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXE_LINKER_FLAGS -static)

option(ENABLE_SCALAR "Enable the scalar implementation" ON)
if(ENABLE_SCALAR)
  set(IMPL_FILES ${IMPL_FILES} scalar_impl.cpp)
  set(IMPL_DEFINITIONS ${IMPL_DEFINITIONS} AFFT_SCALAR_ENABLED=1)
endif()

option(ENABLE_SSE2 "Enable the SSE2 implementation" ON)
if(ENABLE_SSE2)
  set(IMPL_FILES ${IMPL_FILES} sse2_impl.cpp)
  set(IMPL_DEFINITIONS ${IMPL_DEFINITIONS} AFFT_SSE2_ENABLED=1)
  set_source_files_properties(sse2_impl.cpp PROPERTIES COMPILE_FLAGS -msse2)
endif()

option(ENABLE_AVX2 "Enable the AVX2 implementation" ON)
if(ENABLE_AVX2)
  set(IMPL_FILES ${IMPL_FILES} avx2_impl.cpp)
  set(IMPL_DEFINITIONS ${IMPL_DEFINITIONS} AFFT_AVX2_ENABLED=1)
  set_source_files_properties(
    avx2_impl.cpp PROPERTIES COMPILE_FLAGS "-mavx2 -mfma")
endif()

option(ENABLE_AVX512F "Enable the AVX512F implementation" ON)
if(ENABLE_AVX512F)
  set(IMPL_FILES ${IMPL_FILES} avx512f_impl.cpp)
  set(IMPL_DEFINITIONS ${IMPL_DEFINITIONS} AFFT_AVX512F_ENABLED=1)
  set_source_files_properties(
    avx512f_impl.cpp PROPERTIES COMPILE_FLAGS "-mavx512f -mfma")
endif()

option(ENABLE_NEON "Enable the NEON implementation" OFF)
if(ENABLE_NEON)
  set(IMPL_FILES ${IMPL_FILES} neon_impl.cpp)
  set(IMPL_DEFINITIONS ${IMPL_DEFINITIONS} AFFT_NEON_ENABLED=1)
  set_source_files_properties(
    neon_impl.cpp PROPERTIES COMPILE_FLAGS -mfpu=neon-vfpv4)
endif()

add_library(fft STATIC fft.cpp ${IMPL_FILES})
target_compile_options(fft PRIVATE -fno-fast-math)
target_compile_definitions(fft PRIVATE ${IMPL_DEFINITIONS})

add_library(testing testing.cpp)
target_link_libraries(testing fft)

add_executable(run run.cpp)
target_link_libraries(run testing)

add_executable(run_all run_all.cpp)
target_link_libraries(run_all testing pthread)
