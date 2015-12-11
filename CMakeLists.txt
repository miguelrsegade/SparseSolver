cmake_minimum_required (VERSION 2.6)
project (pssolver)

set (CUDA_HOST_COMPILER /usr/bin/gcc-4.9)
find_package(CUDA REQUIRED)

set (CMAKE_CXX_STANDARD 11)

set (EXECUTABLE_OUTPUT_PATH "${CMAKE_SOURCE_DIR}/build/bin/")
set (LIBRARY_OUTPUT_PATH "${CMAKE_SOURCE_DIR}/build/lib/")

subdirs ( src tests )

