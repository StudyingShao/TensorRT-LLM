/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION &
 * AFFILIATES. All rights reserved. SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

/// pytorch version should be greater than 2.1.0
#include "torch/csrc/distributed/c10d/FileStore.hpp"
#include "torch/csrc/distributed/c10d/ProcessGroup.hpp"
#include "torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp"

#include <pybind11/pybind11.h>

class HackGroupNCCL : public c10d::ProcessGroupNCCL
{
public:
    void hack_broadcastUniqueNCCLID(ncclUniqueId* ncclID, bool isSingleP2POp, const std::string& devicesKey, int p2pRank)
    {
        broadcastUniqueNCCLID(ncclID, isSingleP2POp, devicesKey, p2pRank);
    }
};

extern HackGroupNCCL* global_hack_group_nccl_ptr;
extern int global_rank;
extern int global_size;