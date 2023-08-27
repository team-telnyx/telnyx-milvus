cmake_minimum_required(VERSION 3.12)
project(swigfaiss)

find_package(SWIG REQUIRED)
include(UseSWIG)

set(CMAKE_CXX_STANDARD 11)

# Add your NVCC and CUDA-related settings here if needed
# set(CUDA_VERSION 11)  # Example

# Check if NVCC is available
if(NOT DEFINED NVCC)
    set(NVCC "")  # Set your NVCC compiler here
endif()

if(NOT "${NVCC}" STREQUAL "")
    set(SWIGFLAGS "${SWIGFLAGS} -DGPU_WRAPPER")
endif()

# Swig wrapper
set_source_files_properties(swigfaiss.i PROPERTIES CPLUSPLUS ON)
swig_add_library(swigfaiss LANGUAGE python SOURCES swigfaiss.i)
swig_link_libraries(swigfaiss ${PYTHON_LIBRARIES})

# Build targets
add_library(_swigfaiss MODULE swigfaiss.cpp)
target_link_libraries(_swigfaiss PRIVATE faiss)
set_target_properties(_swigfaiss PROPERTIES PREFIX "_")

# Build extension module
add_custom_command(
    TARGET _swigfaiss
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
    ${CMAKE_CURRENT_BINARY_DIR}/_swigfaiss.so
    ${CMAKE_CURRENT_SOURCE_DIR}/_swigfaiss.so
)

# Install target
install(TARGETS _swigfaiss DESTINATION ${CMAKE_CURRENT_SOURCE_DIR})

# Clean target
add_custom_target(clean-all
    COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/clean-all.cmake
)

