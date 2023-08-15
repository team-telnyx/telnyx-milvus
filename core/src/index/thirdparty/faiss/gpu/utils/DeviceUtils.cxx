/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */


#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/gpu/utils/DeviceDefs.cuh>
#include <faiss/impl/FaissAssert.h>
#include <mutex>
#include <unordered_map>
// #include <cuda_profiler_api.h>

namespace faiss { namespace gpu {

int getCurrentDevice() {
  int dev = -1;
  HIP_VERIFY(hipGetDevice(&dev));
  FAISS_ASSERT(dev != -1);

  return dev;
}

void setCurrentDevice(int device) {
  HIP_VERIFY(hipSetDevice(device));
}

int getNumDevices() {
  int numDev = -1;
  hipError_t err = hipGetDeviceCount(&numDev);
  if (hipErrorNoDevice == err) {
    numDev = 0;
  } else {
    HIP_VERIFY(err);
  }
  FAISS_ASSERT(numDev != -1);

  return numDev;
}

void profilerStart() {
  HIP_VERIFY(hipProfilerStart());
}

void profilerStop() {
  HIP_VERIFY(hipProfilerStop());
}

void synchronizeAllDevices() {
  for (int i = 0; i < getNumDevices(); ++i) {
    DeviceScope scope(i);

    HIP_VERIFY(hipDeviceSynchronize());
  }
}

const hipDeviceProp_t& getDeviceProperties(int device) {
  static std::mutex mutex;
  static std::unordered_map<int, hipDeviceProp_t> properties;

  std::lock_guard<std::mutex> guard(mutex);

  auto it = properties.find(device);
  if (it == properties.end()) {
    hipDeviceProp_t prop;
    HIP_VERIFY(hipGetDeviceProperties(&prop, device));

    properties[device] = prop;
    it = properties.find(device);
  }

  return it->second;
}

const hipDeviceProp_t& getCurrentDeviceProperties() {
  return getDeviceProperties(getCurrentDevice());
}

int getMaxThreads(int device) {
  return getDeviceProperties(device).maxThreadsPerBlock;
}

int getMaxThreadsCurrentDevice() {
  return getMaxThreads(getCurrentDevice());
}

size_t getMaxSharedMemPerBlock(int device) {
  return getDeviceProperties(device).sharedMemPerBlock;
}

size_t getMaxSharedMemPerBlockCurrentDevice() {
  return getMaxSharedMemPerBlock(getCurrentDevice());
}

int getDeviceForAddress(const void* p) {
  if (!p) {
    return -1;
  }

  hipPointerAttribute_t att;
  hipError_t err = hipPointerGetAttributes(&att, p);
  FAISS_ASSERT_FMT(err == hipSuccess ||
                   err == hipErrorInvalidValue,
                   "unknown error %d", (int) err);

  if (err == hipErrorInvalidValue) {
    // Make sure the current thread error status has been reset
    err = hipGetLastError();
    FAISS_ASSERT_FMT(err == hipErrorInvalidValue,
                     "unknown error %d", (int) err);
    return -1;
  } else if (att.memoryType == hipMemoryTypeHost) {
    return -1;
  } else {
    return att.device;
  }
}

bool getFullUnifiedMemSupport(int device) {
  const auto& prop = getDeviceProperties(device);
  return (prop.major >= 6);
}

bool getFullUnifiedMemSupportCurrentDevice() {
  return getFullUnifiedMemSupport(getCurrentDevice());
}

bool getTensorCoreSupport(int device) {
  const auto& prop = getDeviceProperties(device);
  return (prop.major >= 7);
}

bool getTensorCoreSupportCurrentDevice() {
  return getTensorCoreSupport(getCurrentDevice());
}

int getMaxKSelection() {
  // Don't use the device at the moment, just base this based on the CUDA SDK
  // that we were compiled with
  return GPU_MAX_SELECTION_K;
}

DeviceScope::DeviceScope(int device) {
  prevDevice_ = getCurrentDevice();

  if (prevDevice_ != device) {
    setCurrentDevice(device);
  } else {
    prevDevice_ = -1;
  }
}

DeviceScope::~DeviceScope() {
  if (prevDevice_ != -1) {
    setCurrentDevice(prevDevice_);
  }
}

HipblasHandleScope::HipblasHandleScope() {
  auto blasStatus = hipblasCreate(&blasHandle_);
  FAISS_ASSERT(blasStatus == HIPBLAS_STATUS_SUCCESS);
}

HipblasHandleScope::~HipblasHandleScope() {
  auto blasStatus = hipblasDestroy(blasHandle_);
  FAISS_ASSERT(blasStatus == HIPBLAS_STATUS_SUCCESS);
}

HipEvent::HipEvent(hipStream_t stream)
    : event_(0) {
  HIP_VERIFY(hipEventCreateWithFlags(&event_, hipEventDisableTiming));
  HIP_VERIFY(hipEventRecord(event_, stream));
}

HipEvent::HipEvent(HipEvent&& event) noexcept
    : event_(std::move(event.event_)) {
  event.event_ = 0;
}

HipEvent::~HipEvent() {
  if (event_) {
    HIP_VERIFY(hipEventDestroy(event_));
  }
}

HipEvent&
HipEvent::operator=(HipEvent&& event) noexcept {
  event_ = std::move(event.event_);
  event.event_ = 0;

  return *this;
}

void
HipEvent::streamWaitOnEvent(hipStream_t stream) {
  HIP_VERIFY(hipStreamWaitEvent(stream, event_, 0));
}

void
HipEvent::cpuWaitOnEvent() {
  HIP_VERIFY(hipEventSynchronize(event_));
}

} } // namespace
