// Copyright (C) 2019-2020 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License.

#ifdef MILVUS_GPU_VERSION

#pragma once

#include <string>

#include <hip/hip_runtime.h>
#include <rocm_smi/rocm_smi.h>

#include "utils/Status.h"

namespace milvus {
namespace server {

extern const int CUDA_MIN_VERSION;
extern const float GPU_MIN_COMPUTE_CAPACITY;
extern const char* NVIDIA_MIN_DRIVER_VERSION;

class GpuChecker {
 private:
    static std::string
    RsmiErrorString(rsmi_status_t error_no);

    static std::string
    HipErrorString(hipError_t error_no);

 private:
    static Status
    GetGpuComputeCapacity(hipDevice_t device, int& major, int& minor);

    static Status
    GetGpuAmdDriverVersion(std::string& version);

    static Status
    GetGpuHipDriverVersion(int& version);

    static Status
    GetGpuHipRuntimeVersion(int& version);

 public:
    static Status
    CheckGpuEnvironment();
};

}  // namespace server
}  // namespace milvus
#endif
