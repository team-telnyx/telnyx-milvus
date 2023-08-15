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
#include "server/init/GpuChecker.h"

#include <iostream>
#include <set>
#include <vector>

#include <fiu-local.h>

#include "config/Config.h"
#include "utils/Log.h"

#include "rocm_smi/rocm_smi.h"

namespace milvus {
namespace server {

namespace {
std::string
ConvertHipVersion(int version) {
    return std::to_string(version / 1000) + "." + std::to_string((version % 100) / 10);
}
}  // namespace

const int CUDA_MIN_VERSION = 10000;  // 10.0
const float GPU_MIN_COMPUTE_CAPACITY = 6.0;
const char* NVIDIA_MIN_DRIVER_VERSION = "418.00";

std::string
GpuChecker::RsmiErrorString(rsmi_status_t error_no) {
    return "code: " + std::to_string(error_no) + ", message: " + RsmiErrorString(error_no);
}

std::string
GpuChecker::HipErrorString(hipError_t error_no) {
    return "code: " + std::to_string(error_no) + ", message: " + hipGetErrorString(error_no);
}

Status
GpuChecker::GetGpuComputeCapacity(hipDevice_t device, int& major, int& minor) {
        int compute_capability[2] = { 0, 0 };
    hipError_t hip_result = hipDeviceGetAttribute(&compute_capability[0], hipDeviceAttributeComputeCapabilityMajor, device);
    if (hip_result != hipSuccess) {
        return Status(SERVER_UNEXPECTED_ERROR, "Failed to get GPU compute capability major version");
    }

    hip_result = hipDeviceGetAttribute(&compute_capability[1], hipDeviceAttributeComputeCapabilityMinor, device);
    if (hip_result != hipSuccess) {
        return Status(SERVER_UNEXPECTED_ERROR, "Failed to get GPU compute capability minor version");
    }

    major = compute_capability[0];
    minor = compute_capability[1];

    return Status::OK();
}

Status
GpuChecker::GetGpuAmdDriverVersion(std::string& version) {
    std::array<char, 128> buffer;
    std::string command = "rocm-smi --showdriverversion";

    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(command.c_str(), "r"), pclose);
    if (!pipe) {
        return Status(SERVER_UNEXPECTED_ERROR, "Failed to execute rocm-smi command");
    }

    std::string driverVersion;
    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
        driverVersion += buffer.data();
    }

    if (driverVersion.empty()) {
        return Status(SERVER_UNEXPECTED_ERROR, "Failed to retrieve GPU driver version");
    }

    version = driverVersion;
    return Status::OK();
}

Status
GpuChecker::GetGpuHipDriverVersion(int& version) {
    auto hip_code = hipDriverGetVersion(&version);
    if (hipSuccess != hip_code) {
        std::string error_msg = "Check hip driver version failed. " + HipErrorString(hip_code);
        return Status(SERVER_UNEXPECTED_ERROR, error_msg);
    }
    return Status::OK();
}

Status
GpuChecker::GetGpuHipRuntimeVersion(int& version) {
    auto hip_code = hipRuntimeGetVersion(&version);
    if (hipSuccess != hip_code) {
        std::string error_msg = "Check hip runtime version failed. " + HipErrorString(hip_code);
        return Status(SERVER_UNEXPECTED_ERROR, error_msg);
    }
    return Status::OK();
}

Status
GpuChecker::CheckGpuEnvironment() {
    std::string err_msg;

    auto& config = Config::GetInstance();
    bool gpu_enable = true;
    auto status = config.GetGpuResourceConfigEnable(gpu_enable);
    if (!status.ok()) {
        err_msg = "Cannot check if GPUs are enable from configuration. " + status.message();
        LOG_SERVER_FATAL_ << err_msg;
        return Status(SERVER_UNEXPECTED_ERROR, err_msg);
    }
    if (!gpu_enable) {
        return Status::OK();
    }

    std::vector<int64_t> build_gpus;
    status = config.GetGpuResourceConfigBuildIndexResources(build_gpus);
    if (!status.ok()) {
        err_msg = "Get GPU resources of building index failed. " + status.message();
        LOG_SERVER_FATAL_ << err_msg;
        return Status(SERVER_UNEXPECTED_ERROR, err_msg);
    }

    std::vector<int64_t> search_gpus;
    status = config.GetGpuResourceConfigSearchResources(search_gpus);
    if (!status.ok()) {
        err_msg = "Get GPU resources of search failed. " + status.message();
        LOG_SERVER_FATAL_ << err_msg;
        return Status(SERVER_UNEXPECTED_ERROR, err_msg);
    }

    std::set<int64_t> gpu_sets(build_gpus.begin(), build_gpus.end());
    gpu_sets.insert(search_gpus.begin(), search_gpus.end());

    hipError_t hip_result = hipInit(0);
    if (hip_result != hipSuccess) {
        std::string err_msg = "HIP initialization failed";
        LOG_SERVER_FATAL_ << err_msg;
        std::cerr << err_msg << std::endl;
        return Status(SERVER_UNEXPECTED_ERROR, err_msg);
    }

    /* Check nvidia driver version */
    std::string nvidia_version;
    status = GetGpuAmdDriverVersion(nvidia_version);
    fiu_do_on("GpuChecker.CheckGpuEnvironment.get_nvidia_driver_fail", status = Status(SERVER_UNEXPECTED_ERROR, ""));
    if (!status.ok()) {
        err_msg = " Check nvidia driver failed. " + status.message();
        LOG_SERVER_FATAL_ << err_msg;
        return Status(SERVER_UNEXPECTED_ERROR, err_msg);
    }

    fiu_do_on("GpuChecker.CheckGpuEnvironment.nvidia_driver_too_slow",
              nvidia_version = std::to_string(std::stof(NVIDIA_MIN_DRIVER_VERSION) - 1));
    if (nvidia_version.compare(NVIDIA_MIN_DRIVER_VERSION) < 0) {
        err_msg = "Nvidia driver version " + std::string(nvidia_version) + " is slower than " +
                  std::string(NVIDIA_MIN_DRIVER_VERSION);
        LOG_SERVER_FATAL_ << err_msg;
        std::cerr << err_msg << std::endl;
        return Status(SERVER_UNEXPECTED_ERROR, err_msg);
    }

    /* Check Cuda version */
    int cuda_driver_version = 0;
    status = GetGpuHipDriverVersion(cuda_driver_version);
    fiu_do_on("GpuChecker.CheckGpuEnvironment.cuda_driver_fail", status = Status(SERVER_UNEXPECTED_ERROR, ""));
    if (!status.ok()) {
        err_msg = " Check Cuda driver failed. " + status.message();
        LOG_SERVER_FATAL_ << err_msg;
        return Status(SERVER_UNEXPECTED_ERROR, err_msg);
    }
    fiu_do_on("GpuChecker.CheckGpuEnvironment.cuda_driver_too_slow", cuda_driver_version = CUDA_MIN_VERSION - 1);
    if (cuda_driver_version < CUDA_MIN_VERSION) {
        err_msg = "Cuda driver version is " + ConvertHipVersion(cuda_driver_version) +
                  ", slower than minimum required version " + ConvertHipVersion(CUDA_MIN_VERSION);
        LOG_SERVER_FATAL_ << err_msg;
        std::cerr << err_msg << std::endl;
        return Status(SERVER_UNEXPECTED_ERROR, err_msg);
    }

    int cuda_runtime_version = 0;
    status = GetGpuHipRuntimeVersion(cuda_runtime_version);
    fiu_do_on("GpuChecker.CheckGpuEnvironment.cuda_runtime_driver_fail", status = Status(SERVER_UNEXPECTED_ERROR, ""));
    if (!status.ok()) {
        err_msg = " Check Cuda runtime driver failed. " + status.message();
        LOG_SERVER_FATAL_ << err_msg;
        return Status(SERVER_UNEXPECTED_ERROR, err_msg);
    }
    fiu_do_on("GpuChecker.CheckGpuEnvironment.cuda_runtime_driver_too_slow",
              cuda_runtime_version = CUDA_MIN_VERSION - 1);
    if (cuda_runtime_version < CUDA_MIN_VERSION) {
        err_msg = "Cuda runtime version is " + ConvertHipVersion(cuda_runtime_version) +
                  ", slow than minimum required version " + ConvertHipVersion(CUDA_MIN_VERSION);
        LOG_SERVER_FATAL_ << err_msg;
        std::cerr << err_msg << std::endl;
        return Status(SERVER_UNEXPECTED_ERROR, err_msg);
    }

    /* Compute capacity */
    uint32_t device_count = 0;
    for (int i = 0; ; ++i) {
        hip_result = hipSetDevice(i);
        if (hip_result == hipErrorInvalidDevice) {
            break;  // No more devices found
        } else if (hip_result != hipSuccess) {
            std::string err_msg = "Failed to set device " + std::to_string(i);
            LOG_SERVER_FATAL_ << err_msg;
            return Status(SERVER_UNEXPECTED_ERROR, err_msg);
        }

        ++device_count;
    }

    fiu_do_on("GpuChecker.CheckGpuEnvironment.nvml_device_count_zero", device_count = 0);
    if (device_count == 0) {
        err_msg = "GPU count is zero. Make sure there are available GPUs in host machine";
        LOG_SERVER_FATAL_ << err_msg;
        return Status(SERVER_UNEXPECTED_ERROR, err_msg);
    }

    constexpr int MAX_DEVICE_NAME_LENGTH = 256;
    char device_name[MAX_DEVICE_NAME_LENGTH];
    int major, minor;
    for (uint32_t i = 0; i < device_count; i++) {
        if (gpu_sets.find(i) == gpu_sets.end()) {
            continue;
        }

        hipDevice_t device;
        hipError_t hip_result = hipDeviceGet(&device, i);
        if (hip_result != hipSuccess) {
            err_msg = "Obtain GPU " + std::to_string(i) + " handle failed.";
            LOG_SERVER_FATAL_ << err_msg;
            return Status(SERVER_UNEXPECTED_ERROR, err_msg);
        }

        memset(device_name, 0, MAX_DEVICE_NAME_LENGTH);
        rsmi_status_t rocm_smi_result = rsmi_dev_name_get(device, device_name, MAX_DEVICE_NAME_LENGTH);
        if (rocm_smi_result != RSMI_STATUS_SUCCESS) {
            err_msg = "Obtain GPU " + std::to_string(i) + " name failed.";
            LOG_SERVER_FATAL_ << err_msg;
            return Status(SERVER_UNEXPECTED_ERROR, err_msg);
        }

        major = 0;
        minor = 0;
        Status status = GetGpuComputeCapacity(device, major, minor);
        if (!status.ok()) {
            err_msg = "Obtain GPU " + std::to_string(i) + " compute capacity failed. " + status.message();
            LOG_SERVER_FATAL_ << err_msg;
            std::cerr << err_msg << std::endl;
            return Status(SERVER_UNEXPECTED_ERROR, err_msg);
        }
        float cc = major + minor / 1.0f;
        if (cc < GPU_MIN_COMPUTE_CAPACITY) {
            err_msg = "GPU " + std::to_string(i) + " compute capability " + std::to_string(cc) +
                    " is too weak. Required least GPU compute capability is " +
                    std::to_string(GPU_MIN_COMPUTE_CAPACITY);
            LOG_SERVER_FATAL_ << err_msg;
            std::cerr << err_msg << std::endl;
            return Status(SERVER_UNEXPECTED_ERROR, err_msg);
        }

        LOG_SERVER_INFO_ << "GPU" << i << ": name=" << device_name << ", compute capacity=" << cc;
    }

    rsmi_status_t rocm_smi_result = rsmi_shut_down();
    if (rocm_smi_result != RSMI_STATUS_SUCCESS) {
        std::string err_msg = "ROCm SMI shutdown failed.";
        LOG_SERVER_FATAL_ << err_msg;
        return Status(SERVER_UNEXPECTED_ERROR, err_msg);
    }

    return Status::OK();
    std::cout << "Nvidia driver version: " << nvidia_version << "\n"
              << "CUDA Driver Version / Runtime Version : " << ConvertHipVersion(cuda_driver_version) << " / "
              << ConvertHipVersion(cuda_runtime_version) << std::endl;

    return Status::OK();
}

}  // namespace server
}  // namespace milvus
#endif
