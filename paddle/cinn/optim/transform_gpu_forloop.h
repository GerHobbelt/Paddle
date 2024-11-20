// Copyright (c) 2021 CINN Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once
#include <algorithm>
#include <unordered_set>
#include <utility>

#include "paddle/cinn/ir/ir.h"
#include "paddle/cinn/ir/lowered_func.h"
#include "paddle/cinn/poly/isl_utils.h"
#include "paddle/cinn/poly/stage.h"

namespace cinn {
namespace optim {

void OptimizeExprGPU(Expr* expr);
/*
  // replace 'for' loop to gpu 'block/thread'
  // update buffer index to save memory size.
  // re-compute buffer size.
*/

/**
 * Remove the GPU block/thread-bound For loops, add IfThenElse guards if needed.
 *
 * It's usually safe to remove bound loops, because when launching the kernel,
 * we are expected to choose dim sizes that match the extents of these loops.
 * However, there are cases where we cannot simply remove a loop, but need to
 * add an IfThenElse as guard:
 *   1) if the loop doesn't start from 0.
 *   2) if we cannot prove that the loop's extent is always equal to or greater
 *      than the corresponding dim size.
 *
 * Example 1:
 *   # assume blockDim.x == 256
 *   thread_bind[threadIdx.x] for (k, 0, 256):
 *     ScheduleBlock(A)
 * =>
 *   ScheduleBlock(A)
 *
 * Example 2:
 *   # assume gridDim.x == 8
 *   thread_bind[blockIdx.x] for (k, 2, min(S0, 8)):
 *     ScheduleBlock(A)
 * =>
 *   if (blockIdx.x >= 2 && blockIdx.x < min(S0, 8)):
 *     ScheduleBlock(A)
 *
 * @param fn The LoweredFunc to process.
 */
void RemoveGpuForLoops(ir::LoweredFunc fn);

/**
 * Add __syncthreads() to shared memory producer.
 */
void CudaSyncThreadsDropIfThenElse(ir::LoweredFunc fn);

}  // namespace optim
}  // namespace cinn
