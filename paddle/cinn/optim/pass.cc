// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/cinn/optim/pass.h"

#include "paddle/cinn/ir/ir_printer.h"
#include "paddle/cinn/ir/stmt_visitors.h"

namespace cinn {
namespace optim {

bool ApplyFunctionPass(FunctionPass* pass, ir::LoweredFunc f) {
  return pass->RunOnFunction(f);
}

bool ApplyBlockPass(BlockPass* pass, ir::stmt::BlockRef func_body) {
  VLOG(3) << "Before ApplyBlockPass: [" << pass->name()
          << "] on block: " << func_body;
  bool changed = false;
  std::vector<ir::stmt::StmtRef> new_stmts = func_body->stmts();
  for (ir::stmt::StmtRef inner_stmt : new_stmts) {
    std::vector<ir::stmt::BlockRef> inner_blocks = inner_stmt->block_fields();
    for (ir::stmt::BlockRef inner_block : inner_blocks) {
      changed = ApplyBlockPass(pass, inner_block) || changed;
    }
    inner_stmt->set_block_fields(inner_blocks);
  }
  func_body->set_stmts(new_stmts);
  changed = pass->RunOnBlock(func_body) || changed;
  VLOG(3) << "After ApplyBlockPass: [" << pass->name()
          << "] on block: " << func_body;
  return changed;
}

bool ApplyStatementPass(StatementPass* pass, ir::stmt::BlockRef func_body) {
  bool changed = false;
  ir::stmt::Mutate(
      func_body,
      [&](ir::stmt::StmtRef stmt) {},
      [&](ir::stmt::StmtRef stmt) {
        if (pass->RunOnStmt(stmt)) {
          changed = true;
        }
      });
  return changed;
}

}  // namespace optim
}  // namespace cinn
