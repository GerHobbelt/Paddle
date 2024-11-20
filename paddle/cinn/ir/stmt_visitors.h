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

#include <functional>

#include "paddle/cinn/ir/ir.h"
#include "paddle/cinn/ir/stmt.h"

namespace cinn {
namespace ir {
namespace stmt {

// Defines utilities for walking statements.
void Visit(const BlockRef &block,
           const std::function<void(const StmtRef &)> &pre_callback,
           const std::function<void(const StmtRef &)> &post_callback);

void Visit(const StmtRef &stmt,
           const std::function<void(const StmtRef &)> &pre_callback,
           const std::function<void(const StmtRef &)> &post_callback);

void Mutate(BlockRef block,
            const std::function<void(StmtRef)> &pre_callback,
            const std::function<void(StmtRef)> &post_callback);

void Mutate(StmtRef stmt,
            const std::function<void(StmtRef)> &pre_callback,
            const std::function<void(StmtRef)> &post_callback);

#define CINN_CHECK_STMT_DEFINED(stmt)                                       \
  PADDLE_ENFORCE_EQ(                                                        \
      stmt.defined(),                                                       \
      true,                                                                 \
      ::common::errors::Unavailable("The statement is not defined. Please " \
                                    "provide a valid statement."));

template <typename StmtRetTy = void,
          typename BlockRetTy = void,
          typename... Args>
class StmtVisitor {
 public:
  virtual StmtRetTy VisitStmt(const StmtRef &stmt, Args... args) {
    CINN_CHECK_STMT_DEFINED(stmt)
    switch (stmt->stmt_type()) {
#define __(stmt__)                                \
  case ir::StmtNodeTy::stmt__:                    \
    return VisitStmt(stmt.as<stmt__>(), args...); \
    break;

      NODETY_FORALL_STMT(__)

      default:
        PADDLE_THROW(::common::errors::InvalidArgument(
            "Deadcode, not supported StmtNodeTy"));
#undef __
    }
  }

  // Default implementation for visiting block with void return type.
  virtual BlockRetTy VisitBlock(const BlockRef &block, Args... args) {
    for (const StmtRef &inner_stmt : block->stmts()) {
      VisitStmt(inner_stmt);
    }
    return BlockRetTy();
  }

 protected:
#define __(stmt__) \
  virtual StmtRetTy VisitStmt(const stmt__ &stmt, Args... args) = 0;
  NODETY_FORALL_STMT(__)
#undef __
};

template <typename StmtRetTy = void,
          typename BlockRetTy = void,
          typename... Args>
class StmtMutator {
 public:
  virtual StmtRetTy VisitStmt(StmtRef stmt, Args... args) {
    CINN_CHECK_STMT_DEFINED(stmt)
    switch (stmt->stmt_type()) {
#define __(stmt__)                                \
  case ir::StmtNodeTy::stmt__:                    \
    return VisitStmt(stmt.as<stmt__>(), args...); \
    break;

      NODETY_FORALL_STMT(__)

      default:
        PADDLE_THROW(::common::errors::InvalidArgument(
            "Deadcode, not supported StmtNodeTy"));
#undef __
    }
  }

  // Default implementation for visiting block with void return type.
  virtual BlockRetTy VisitBlock(BlockRef block, Args... args) {
    std::vector<StmtRef> new_stmts = block->stmts();
    for (StmtRef inner_stmt : new_stmts) {
      VisitStmt(inner_stmt);
    }
    block->set_stmts(new_stmts);
    return BlockRetTy();
  }

 protected:
#define __(stmt__) virtual StmtRetTy VisitStmt(stmt__ stmt, Args... args) = 0;
  NODETY_FORALL_STMT(__)
#undef __
};

#undef CINN_CHECK_STMT_DEFINED

}  // namespace stmt
}  // namespace ir
}  // namespace cinn
