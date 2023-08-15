#!/bin/bash

BUILD_TYPE="Debug"
BUILD_UNITTEST="OFF"
INSTALL_PREFIX=$(pwd)/cmake_build
MAKE_CLEAN="OFF"
PROFILING="OFF"
FAISS_WITH_MKL="ON"

while getopts "p:d:t:uhrcgm" arg
do
        case $arg in
             t)
                BUILD_TYPE=$OPTARG # BUILD_TYPE
                ;;
             u)
                echo "Build and run unittest cases" ;
                BUILD_UNITTEST="ON";
                ;;
             p)
                INSTALL_PREFIX=$OPTARG
                ;;
             r)
                if [[ -d cmake_build ]]; then
                    rm ./cmake_build -r
                    MAKE_CLEAN="ON"
                fi
                ;;
             g)
                PROFILING="ON"
                ;;
             m)
                FAISS_WITH_MKL="ON"
                ;;
             h) # help
                echo "

parameter:
-t: build type(default: Debug)
-u: building unit test options(default: OFF)
-p: install prefix(default: $(pwd)/knowhere)
-r: remove previous build directory(default: OFF)
-g: profiling(default: OFF)
-m: build faiss with MKL(default: OFF)

usage:
./build.sh -t \${BUILD_TYPE} [-u] [-h] [-g] [-r] [-c] [-m]
                "
                exit 0
                ;;
             ?)
                echo "unknown argument"
        exit 1
        ;;
        esac
done

if [[ ! -d cmake_build ]]; then
	mkdir cmake_build
	MAKE_CLEAN="ON"
fi

cd cmake_build

HIP_COMPILER=/opt/rocm-5.3.0/bin/hipcc

if [[ ${MAKE_CLEAN} == "ON" ]]; then
    CMAKE_CMD="cmake -DBUILD_UNIT_TEST=${BUILD_UNITTEST} \
    -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX}
    -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
    -DCMAKE_PREFIX_PATH=${HIP_ROOT_DIR} \
    -DMILVUS_ENABLE_PROFILING=${PROFILING} \
    -DFAISS_WITH_MKL=${FAISS_WITH_MKL} \
    ../"
    echo ${CMAKE_CMD}

    ${CMAKE_CMD}
    make clean
fi

make -j 8 || exit 1

make install || exit 1
