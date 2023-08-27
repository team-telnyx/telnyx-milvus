# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

file(GLOB FAISS_HEADERS "*.h" "impl/*.h" "utils/*.h" "gpu/*.h" "gpu/impl/*.h" "gpu/impl/*.cuh" "gpu/utils/*.h" "gpu/utils/*.cuh" "gpu/utils/nvidia/*.h" "gpu/utils/blockselect/*.cuh" "gpu/utils/warpselect/*.cuh")
file(GLOB FAISS_SRC "*.cpp" "impl/*.cpp" "utils/*.cpp")

# Exclude files containing "avx512" in their names
list(FILTER FAISS_SRC EXCLUDE REGEX ".*avx512.*")
list(FILTER FAISS_HEADERS EXCLUDE REGEX ".*avx512.*")

# Export FAISS_HEADERS variable to parent scope.
set(FAISS_HEADERS ${FAISS_HEADERS} PARENT_SCOPE)

add_library(faiss SHARED ${FAISS_SRC})

add_library(faiss_avx2 ${FAISS_SRC})
if(NOT FAISS_OPT_LEVEL STREQUAL "avx2")
  set_target_properties(faiss_avx2 PROPERTIES EXCLUDE_FROM_ALL TRUE)
endif()
if(NOT WIN32)
  target_compile_options(faiss_avx2 PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-mavx2 -mfma -mf16c -mpopcnt>)

  # target_compile_options(faiss_avx2 PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-msse3 -mavx2 -mfma -mf16c -mpopcnt>)
else()
  # MSVC enables FMA with /arch:AVX2; no separate flags for F16C, POPCNT
  # Ref. FMA (under /arch:AVX2): https://docs.microsoft.com/en-us/cpp/build/reference/arch-x64
  # Ref. F16C (2nd paragraph): https://walbourn.github.io/directxmath-avx2/
  # Ref. POPCNT: https://docs.microsoft.com/en-us/cpp/intrinsics/popcnt16-popcnt-popcnt64
  target_compile_options(faiss_avx2 PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/arch:AVX2>)
  # we need bigobj for the swig wrapper
  add_compile_options(/bigobj)
endif()

# Handle `#include <faiss/foo.h>`.
target_include_directories(faiss PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>)
# Handle `#include <faiss/foo.h>`.
target_include_directories(faiss_avx2 PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>)

set_target_properties(faiss PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  WINDOWS_EXPORT_ALL_SYMBOLS ON
)
set_target_properties(faiss_avx2 PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  WINDOWS_EXPORT_ALL_SYMBOLS ON
)

if(WIN32)
  target_compile_definitions(faiss PRIVATE FAISS_MAIN_LIB)
  target_compile_definitions(faiss_avx2 PRIVATE FAISS_MAIN_LIB)
endif()

target_compile_definitions(faiss PRIVATE FINTEGER=int)
target_compile_definitions(faiss PRIVATE FAISS_USE_FLOAT16=FALSE)
target_compile_definitions(faiss_avx2 PRIVATE FINTEGER=int)
target_compile_definitions(faiss_avx2 PRIVATE FAISS_USE_FLOAT16=FALSE)

find_package(OpenMP REQUIRED)
target_link_libraries(faiss PRIVATE OpenMP::OpenMP_CXX)
target_link_libraries(faiss_avx2 PRIVATE OpenMP::OpenMP_CXX)

find_package(MKL)
if(MKL_FOUND)
  target_link_libraries(faiss PUBLIC ${MKL_LIBRARIES})
  target_link_libraries(faiss_avx2 PUBLIC ${MKL_LIBRARIES})
else()
  find_package(rocBLAS REQUIRED)
  target_link_libraries(faiss PUBLIC roc::rocblas)
  target_link_libraries(faiss_avx2 PUBLIC roc::rocblas)

  #find_package(LAPACK REQUIRED)
  set(LAPACK_LIB "/usr/lib/x86_64-linux-gnu")
  set(LAPACK_INCLUDE_DIR "/usr/include")
  include_directories(${LAPACK_INCLUDE_DIR})

  target_link_libraries(faiss PUBLIC ${LAPACK_LIB}) 
  target_link_libraries(faiss_avx2 PUBLIC ${LAPACK_LIB}) 
endif()

add_subdirectory(gpu/test)

install(TARGETS faiss
  EXPORT faiss-targets
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
if(FAISS_OPT_LEVEL STREQUAL "avx2")
  install(TARGETS faiss_avx2
    EXPORT faiss-targets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  )
endif()

foreach(header ${FAISS_HEADERS})
  get_filename_component(dir ${header} DIRECTORY )
  install(FILES ${header}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/faiss/${dir}
  )
endforeach()

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
  "${PROJECT_BINARY_DIR}/cmake/faiss-config-version.cmake"
  VERSION ${CMAKE_PROJECT_VERSION}
  COMPATIBILITY AnyNewerVersion
)

configure_file(${PROJECT_SOURCE_DIR}/cmake/faiss-config.cmake.in
  ${PROJECT_BINARY_DIR}/cmake/faiss-config.cmake
  COPYONLY
)
install(FILES ${PROJECT_BINARY_DIR}/cmake/faiss-config.cmake
  ${PROJECT_BINARY_DIR}/cmake/faiss-config-version.cmake
  DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/faiss
)

install(EXPORT faiss-targets
  DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/faiss
)
