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

#pragma once

#include <string>

#include "paddle/cinn/ir/ir.h"
#include "paddle/cinn/ir/lowered_func.h"
#include "paddle/cinn/ir/stmt.h"

namespace cinn {
namespace optim {

enum PassKind { PK_FUNC, PK_BLOCK, PK_STMT, PK_EXPR };

class Pass {
 public:
  explicit Pass(PassKind kind, const std::string& name)
      : kind_(kind), name_(name) {}
  virtual ~Pass() {}

  PassKind kind() const { return kind_; }
  const std::string& name() const { return name_; }

 private:
  PassKind kind_;
  std::string name_;
};

class FunctionPass : public Pass {
 public:
  explicit FunctionPass(const std::string& name) : Pass(PK_FUNC, name) {}

  virtual bool RunOnFunction(ir::LoweredFunc f) = 0;
};

class BlockPass : public Pass {
 public:
  explicit BlockPass(const std::string& name) : Pass(PK_BLOCK, name) {}
  virtual bool RunOnBlock(ir::stmt::BlockRef block) = 0;
};

class StatementPass : public Pass {
 public:
  explicit StatementPass(const std::string& name) : Pass(PK_STMT, name) {}
  virtual bool RunOnStmt(ir::stmt::StmtRef stmt) = 0;
};

bool ApplyFunctionPass(FunctionPass* pass, ir::LoweredFunc f);
// post order traverse apply block pass on function body
bool ApplyBlockPass(BlockPass* pass, ir::stmt::BlockRef func_body);
// post order traverse apply statement pass on function body
bool ApplyStatementPass(StatementPass* pass, ir::stmt::BlockRef func_body);

// TODO(hongqing-work): add manager for pass

}  // namespace optim
}  // namespace cinn
