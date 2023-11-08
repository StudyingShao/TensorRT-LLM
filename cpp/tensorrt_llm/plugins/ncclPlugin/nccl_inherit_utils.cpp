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

HackGroupNCCL* global_hack_group_nccl_ptr;
int global_rank;
int global_size;

void getProcessGroupNCCL(c10d::ProcessGroupNCCL& process_group_nccl)
{
    c10d::ProcessGroupNCCL* process_group_nccl_ptr = &process_group_nccl;
    HackGroupNCCL* hack_group_nccl_ptr = (HackGroupNCCL*) process_group_nccl_ptr;

    global_hack_group_nccl_ptr = hack_group_nccl_ptr;
}

void getProcessGroupNCCL_list(pybind11::list hack_list, int rank, int size)
{
  for(auto process_group_nccl : hack_list)
  {
      c10d::ProcessGroupNCCL* process_group_nccl_ptr = reinterpret_cast<c10d::ProcessGroupNCCL*>(&process_group_nccl);
      HackGroupNCCL* hack_group_nccl_ptr = (HackGroupNCCL*) process_group_nccl_ptr;

      global_hack_group_nccl_ptr = hack_group_nccl_ptr;
      global_rank = rank;
      global_size = size;
  }
}

PYBIND11_MODULE(libhackNCCL, m)
{
    m.def("getProcessGroupNCCL", &getProcessGroupNCCL);
    m.def("getProcessGroupNCCL_list", &getProcessGroupNCCL_list);
}