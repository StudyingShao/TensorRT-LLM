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

#include "nccl_inherit_utils.h"

int global_rank;
int global_size;
ncclUniqueId global_ncclID;

void nccl_get_info(int rank, int size, std::vector<uint8_t>& vec)
{
    global_rank = rank;
    global_size = size;
    std::memcpy(&global_ncclID, vec.data(), NCCL_UNIQUE_ID_BYTES);
}

std::vector<uint8_t> nccl_create_uniqueId()
{
    ncclUniqueId ncclID;
    ncclGetUniqueId(&ncclID);

    auto vec = std::vector<uint8_t>(
        reinterpret_cast<uint8_t*>(&ncclID), reinterpret_cast<uint8_t*>(&ncclID) + NCCL_UNIQUE_ID_BYTES);

    return vec;
}

PYBIND11_MODULE(libhackNCCL, m)
{
    m.def("nccl_get_info", &nccl_get_info);
    m.def("nccl_create_uniqueId", &nccl_create_uniqueId);
}