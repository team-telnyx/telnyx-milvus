/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */


#include <faiss/gpu/utils/MemorySpace.h>
#include <faiss/impl/FaissAssert.h>
#include <hip/hip_runtime.h>

namespace faiss { namespace gpu {

/// Allocates HIP memory for a given memory space
void allocMemorySpaceV(MemorySpace space, void** p, size_t size) {
  switch (space) {
    case MemorySpace::Device:
    {
      auto err = hipMalloc(p, size);

      // Throw if we fail to allocate
      FAISS_THROW_IF_NOT_FMT(
        err == hipSuccess,
        "failed to hipMalloc %zu bytes (error %d %s)",
        size, (int) err, hipGetErrorString(err));
    }
    break;
    case MemorySpace::Unified:
    {
#ifdef FAISS_UNIFIED_MEM
      auto err = hipMallocManaged(p, size);

      // Throw if we fail to allocate
      FAISS_THROW_IF_NOT_FMT(
        err == hipSuccess,
        "failed to hipMallocManaged %zu bytes (error %d %s)",
        size, (int) err, hipGetErrorString(err));
#else
      FAISS_THROW_MSG("Attempting to allocate via hipMallocManaged "
                      "without CUDA 8+ support");
#endif
    }
    break;
    case MemorySpace::HostPinned:
    {
      auto err = hipHostAlloc(p, size, hipHostMallocDefault);

      // Throw if we fail to allocate
      FAISS_THROW_IF_NOT_FMT(
        err == hipSuccess,
        "failed to hipHostAlloc %zu bytes (error %d %s)",
        size, (int) err, hipGetErrorString(err));
    }
    break;
    default:
      FAISS_ASSERT_FMT(false, "unknown MemorySpace %d", (int) space);
      break;
  }
}

// We'll allow allocation to fail, but free should always succeed and be a
// fatal error if it doesn't free
void freeMemorySpace(MemorySpace space, void* p) {
  switch (space) {
    case MemorySpace::Device:
    case MemorySpace::Unified:
    {
      auto err = hipFree(p);
      FAISS_ASSERT_FMT(err == hipSuccess,
                       "Failed to hipFree pointer %p (error %d %s)",
                       p, (int) err, hipGetErrorString(err));
    }
    break;
    case MemorySpace::HostPinned:
    {
      auto err = hipFreeHost(p);
      FAISS_ASSERT_FMT(err == hipSuccess,
                       "Failed to hipFreeHost pointer %p (error %d %s)",
                       p, (int) err, hipGetErrorString(err));
    }
    break;
    default:
      FAISS_ASSERT_FMT(false, "unknown MemorySpace %d", (int) space);
      break;
  }
}

} }
