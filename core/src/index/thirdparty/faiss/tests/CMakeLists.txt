cmake_minimum_required(VERSION 3.12)
project(TestProject)

# Add compiler flags if needed
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

# Add include directories
include_directories(gtest/include ..)

# Download and build gtest
include(ExternalProject)
ExternalProject_Add(
    gtest_external
    PREFIX ${CMAKE_BINARY_DIR}/gtest
    URL /home/hadi.sharifi/ws/v1.14.0.tar.gz #https://github.com/google/googletest/archive/release-1.8.0.tar.gz
    # URL_HASH SHA256=8ad598c73ad796e0d8280b082cebd82a630d73e73cd3c70057938a6501bba5d7
    TIMEOUT 60
    CMAKE_ARGS -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER} -DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}
    INSTALL_COMMAND ""
)

# Build the tests
file(GLOB TESTS_SRC "*.cpp")
add_executable(tests ${TESTS_SRC})
add_dependencies(tests gtest_external)
target_link_libraries(tests ${CMAKE_BINARY_DIR}/gtest/src/gtest_external-build/googlemock/gtest/libgtest.a ${CMAKE_BINARY_DIR}/gtest/src/gtest_external-build/googlemock/gtest/libgtest_main.a)
target_link_libraries(tests ${CMAKE_BINARY_DIR}/libfaiss.a)  # Assuming libfaiss.a is in the build directory
target_link_libraries(tests ${LDFLAGS} ${LIBS})

# Add custom target to download and build gtest
add_custom_target(download_gtest DEPENDS gtest_external)

# Specify the executable to run the tests
add_custom_target(run_tests COMMAND ${CMAKE_BINARY_DIR}/tests)

# Clean target
add_custom_target(clean_tests
    COMMAND ${CMAKE_COMMAND} -E remove -f ${CMAKE_BINARY_DIR}/tests
    COMMAND ${CMAKE_COMMAND} -E remove -f ${CMAKE_BINARY_DIR}/*.o
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_BINARY_DIR}/gtest
)

# Group targets
add_dependencies(run_tests tests)
add_dependencies(clean_tests tests)
