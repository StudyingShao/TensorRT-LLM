/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "tensorrt_llm/kernels/weightOnlyBatchedGemv/kernelDispatcher.h"

namespace tensorrt_llm
{
namespace kernels
{
namespace weight_only
{
INSTANTIATE_WEIGHT_ONLY_CUDA_DISPATCHERS(
    KernelType::BF16Int4Groupwise, BF16DetailsA, Int4DetailsW, ColumnMajorInterleaved, true, 64);
// KTile=128 for Ada w4a8
INSTANTIATE_WEIGHT_ONLY_CUDA_DISPATCHERS(
    KernelType::BF16Int4Groupwise, BF16DetailsA, Int4DetailsW, ColumnMajorInterleaved, true, 128);
} // namespace weight_only
} // namespace kernels
} // namespace tensorrt_llm
