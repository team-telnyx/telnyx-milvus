# Build avx2 version
CXXFLAGS="-mavx2 -mf16c" ./configure --with-hip=$HIP_ROOT --with-hip-arch="$HIP_ARCH"
make -j $CPU_COUNT
make -C python _swigfaiss_avx2.so
make clean

# Build vanilla version (no avx)
./configure --with-hip=$HIP_ROOT --with-hip-arch="$HIP_ARCH"
make -j $CPU_COUNT
make -C python _swigfaiss.so

make -C python build

cd python

$PYTHON setup.py install --single-version-externally-managed --record=record.txt
