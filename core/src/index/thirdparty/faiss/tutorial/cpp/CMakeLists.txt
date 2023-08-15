cmake_minimum_required(VERSION 3.0)
project(FaissExamples)

# Add the directory containing libfaiss.a to the search path
# link_directories(${CMAKE_SOURCE_DIR}/../../)

# Compiler flags and options
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

# List of CPU and GPU targets
set(CPU_TARGETS 1-Flat 2-IVFFlat 3-IVFPQ 9-BinaryFlat)
set(GPU_TARGETS 4-GPU 5-Multiple-GPUs)

find_library(FAISS_LIB libfaiss.a)
if(NOT FAISS_LIB)
        message(FATAL_ERROR "faiss library not found.")
endif()
     
# Compile CPU targets
foreach(target ${CPU_TARGETS})
    add_executable(${target} ${target}.cpp)
    target_link_libraries(${target} PRIVATE ${FAISS_LIB})
    target_include_directories(${target} PRIVATE "${CMAKE_SOURCE_DIR}/../../")
endforeach()

# Compile GPU targets
foreach(target ${GPU_TARGETS})
    add_executable(${target} ${target}.cpp)
    target_link_libraries(${target} PRIVATE faiss)
    target_include_directories(${target} PRIVATE "${CMAKE_SOURCE_DIR}/../../")
endforeach()

# Clean targets
add_custom_target(clean-all
    COMMAND ${CMAKE_COMMAND} -P ${CMAKE_SOURCE_DIR}/clean-all.cmake
)

# Set default target to cpu
add_custom_target(default cpu)

# Set up the 'all' target to build both CPU and GPU targets
# add_custom_target(all)
# add_dependencies(all cpu gpu)

# Set up the 'clean' target to remove all compiled targets
# add_custom_target(clean)
# add_dependencies(clean clean-all)
