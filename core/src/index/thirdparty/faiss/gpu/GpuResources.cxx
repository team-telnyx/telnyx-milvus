/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */


#include <faiss/gpu/GpuResources.h>
#include <faiss/gpu/utils/DeviceUtils.h>

namespace faiss { namespace gpu {

GpuResources::~GpuResources() {
}

hipblasHandle_t
GpuResources::getBlasHandleCurrentDevice() {
  return getBlasHandle(getCurrentDevice());
}

hipStream_t
GpuResources::getDefaultStreamCurrentDevice() {
  return getDefaultStream(getCurrentDevice());
}

std::vector<hipStream_t>
GpuResources::getAlternateStreamsCurrentDevice() {
  return getAlternateStreams(getCurrentDevice());
}

DeviceMemory&
GpuResources::getMemoryManagerCurrentDevice() {
  return getMemoryManager(getCurrentDevice());
}

hipStream_t
GpuResources::getAsyncCopyStreamCurrentDevice() {
  return getAsyncCopyStream(getCurrentDevice());
}

void
GpuResources::syncDefaultStream(int device) {
  HIP_VERIFY(hipStreamSynchronize(getDefaultStream(device)));
}

void
GpuResources::syncDefaultStreamCurrentDevice() {
  syncDefaultStream(getCurrentDevice());
}

} } // namespace
