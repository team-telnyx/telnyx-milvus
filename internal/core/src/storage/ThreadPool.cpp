// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "ThreadPool.h"

namespace milvus {

void
ThreadPool::Init() {
    for (int i = 0; i < threads_.size(); i++) {
        threads_[i] = std::thread(Worker(this, i));
    }
}

void
ThreadPool::ShutDown() {
    LOG_SEGCORE_INFO_ << "Start shutting down " << name_;
    shutdown_ = true;
    condition_lock_.notify_all();
    for (int i = 0; i < threads_.size(); i++) {
        if (threads_[i].joinable()) {
            threads_[i].join();
        }
    }
    LOG_SEGCORE_INFO_ << "Finish shutting down " << name_;
}
};  // namespace milvus
