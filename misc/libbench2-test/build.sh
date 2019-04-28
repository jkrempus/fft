#!/bin/bash

FFTW_SOURCE_DIR=$1
FFTW_BUILD_DIR=$2
AFFT_LIB=$3
CXX=${CXX:-c++}

$CXX doit.cpp -o doit \
  $FFTW_BUILD_DIR/libbench2/libbench2.a -I $FFTW_SOURCE_DIR/libbench2/ \
  -I $FFTW_BUILD_DIR \
  $AFFT_LIB -I ../..
