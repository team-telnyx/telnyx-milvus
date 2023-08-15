# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.23.1 FATAL_ERROR)

project(faiss
  VERSION 1.7.4
  DESCRIPTION "A library for efficient similarity search and clustering of dense vectors."
  HOMEPAGE_URL "https://github.com/facebookresearch/faiss"
  LANGUAGES CXX)
include(GNUInstallDirs)

include(CheckLanguage)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse3 -mavx -mf16c -mavx2 -mavx512f -mavx512dq -mavx512bw")
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

# Valid values are "generic", "avx2".
option(FAISS_OPT_LEVEL "" "generic")
option(FAISS_ENABLE_GPU "Enable support for GPU indexes." ON)
option(FAISS_ENABLE_PYTHON "Build Python extension." ON)
option(FAISS_ENABLE_C_API "Build C API." OFF)


set(ROCM_PATH "/opt/rocm-5.3.0") 
# Find HIP
list(APPEND CMAKE_PREFIX_PATH /opt/rocm-5.3.0/hip /opt/rocm-5.3.0)
find_package(HIP REQUIRED)

# // TODO: HADI testing the RDNA Architecture:
#set(HIP_ARCHITECTURE "gfx906")
#set(HIP_FLAGS "-march=${HIP_ARCHITECTURE}")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${HIP_FLAGS}")
# ..............................................................

if(FAISS_ENABLE_GPU)
  add_compile_definitions(__HIP_PLATFORM_AMD__)
  # add_compile_options(--offload-arch=gfx906)
  set(CMAKE_HIP_ARCHITECTURES "gfx908") # 
  find_package(hipblas REQUIRED)
endif()

add_subdirectory(faiss)
add_subdirectory(SPTAG)

if(FAISS_ENABLE_GPU)
  add_subdirectory(faiss/gpu)
endif()

# TODO: HADI
# if(FAISS_ENABLE_PYTHON)
#   add_subdirectory(faiss/python)
# endif()

if(FAISS_ENABLE_C_API)
  add_subdirectory(c_api)
endif()

# TODO: HADI
# add_subdirectory(demos)
# add_subdirectory(benchs)
# add_subdirectory(tutorial/cpp)

# CTest must be included in the top level to enable `make test` target.
# include(CTest)
# if(BUILD_TESTING)
#   add_subdirectory(tests)

#   if(FAISS_ENABLE_GPU)
#     add_subdirectory(faiss/gpu/test)
#   endif()
# endif()
