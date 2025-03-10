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

#include "paddle/cinn/optim/optimize.h"

#include "paddle/cinn/ir/ir_printer.h"
#include "paddle/cinn/ir/schedule/ir_schedule_util.h"
#include "paddle/cinn/ir/utils/ir_copy.h"
#include "paddle/cinn/optim/call_arg_list_to_pod_value.h"
#include "paddle/cinn/optim/cast_bool_to_int8.h"
#include "paddle/cinn/optim/eliminate_broadcast_in_forloop.h"
#include "paddle/cinn/optim/eliminate_invariant_loop.h"
#include "paddle/cinn/optim/extern_call_process.h"
#include "paddle/cinn/optim/fold_cinn_call_arguments.h"
#include "paddle/cinn/optim/if_fusion.h"
#include "paddle/cinn/optim/insert_debug_log_callee.h"
#include "paddle/cinn/optim/ir_simplify.h"
#include "paddle/cinn/optim/lower_function_call_bind_vars.h"
#include "paddle/cinn/optim/lower_intrin.h"
#include "paddle/cinn/optim/map_extern_call.h"
#include "paddle/cinn/optim/rearrange_load_instruction.h"
#include "paddle/cinn/optim/remove_schedule_block.h"
#include "paddle/cinn/optim/replace_const_param_to_integer.h"
#include "paddle/cinn/optim/replace_cross_block_reduction.h"
#include "paddle/cinn/optim/replace_cross_thread_reduction.h"
#include "paddle/cinn/optim/trans_buffer_with_dynamic_shape.h"
#include "paddle/cinn/optim/transform_gpu_forloop.h"
#include "paddle/cinn/optim/transform_polyfor_to_for.h"
#include "paddle/cinn/optim/unroll_loops.h"
#include "paddle/cinn/optim/vectorize_for_trans.h"
#include "paddle/cinn/optim/vectorize_loops.h"

namespace cinn {
namespace optim {

ir::LoweredFunc Optimize(ir::LoweredFunc fn,
                         Target target,
                         bool runtime_debug_info,
                         bool remove_gpu_for_loops) {
  PADDLE_ENFORCE_EQ(
      fn.defined(),
      true,
      ::common::errors::InvalidArgument(
          "Expected expression 'fn' to be defined, but it is undefined."));

  auto copied = ir::ir_utils::IRCopy(fn);

  ReplaceConstParamToInteger(&copied->body);
  // Simplify already contains CastSimplify
  Simplify(&copied->body);
  EliminateInvariantLoop(&copied->body);
  VLOG(4) << "After Optimize EliminateInvariantLoop:" << copied;
  ReplaceCrossThreadReduction(copied);
  VLOG(4) << "After Optimize ReplaceCrossThreadReduction:" << copied;
  ReplaceCrossBlockReduction(copied);
  VLOG(4) << "After Optimize ReplaceCrossBlockReduction:" << copied;

  cinn::common::DefaultDeviceTarget().arch.Match(
      [&](std::variant<common::UnknownArch, common::X86Arch, common::ARMArch>) {
      },
      [&](common::NVGPUArch) {
#ifdef CINN_WITH_CUDA
        ir::SetCudaAxisInfo(copied);
        if (remove_gpu_for_loops) {
          RemoveGpuForLoops(copied);
        }
        CudaSyncThreadsDropIfThenElse(copied);
    // CudaTransBufferWithDynamicShape(&copied);
#endif
      },
      [&](common::HygonDCUArchHIP) {
#ifdef CINN_WITH_HIP
        ir::SetCudaAxisInfo(copied);
        if (remove_gpu_for_loops) {
          RemoveGpuForLoops(copied);
        }
        CudaSyncThreadsDropIfThenElse(copied);
    // CudaTransBufferWithDynamicShape(&copied);
#endif
      });

  SimplifyBlocks(&copied->body);
  VLOG(4) << "After SimplifyBlocks:" << copied;

  MapExternCall(&copied->body, target);
  VLOG(10) << "After Optimize MapExternCall:" << copied;

  ExternCallMultiOutputShallowStore(&copied->body);
  VLOG(10) << "After Optimize ExternCallMultiOutputShallowStore:" << copied;
  // Simplify already contains CastSimplify
  Simplify(&copied->body);
  VLOG(10) << "After Optimize Simplify:" << copied;

  // TODO(liangshuhao): this pass may unexpectedly remove schedule blocks, and
  // it actually doesn't contribute to performance, so temporarily disabled.
  // IfFusion(&copied->body);

  VectorizeForTrans(&copied->body);
  VLOG(10) << "After Optimize vectorize" << copied;

  Simplify(&copied->body);
  VLOG(10) << "After Optimize Simplify" << copied;

  RemoveScheduleBlock(&copied->body);
  VLOG(10) << "After RemoveScheduleBlock:" << copied;

  LowerIntrin(&copied->body, target);
  VLOG(10) << "After LowerIntrin:" << copied;

  return copied;
}

}  // namespace optim
}  // namespace cinn
