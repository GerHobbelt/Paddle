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

#include "paddle/cinn/ir/stmt_visitors.h"

namespace cinn {
namespace ir {
namespace stmt {
void Visit(const BlockRef &block,
           const std::function<void(const StmtRef &)> &pre_callback,
           const std::function<void(const StmtRef &)> &post_callback) {
  for (const StmtRef &inner_stmt : block->stmts()) {
    Visit(inner_stmt, pre_callback, post_callback);
  }
}

void Visit(const StmtRef &stmt,
           const std::function<void(const StmtRef &)> &pre_callback,
           const std::function<void(const StmtRef &)> &post_callback) {
  pre_callback(stmt);
  for (const BlockRef &inner_block : stmt->block_fields()) {
    Visit(inner_block, pre_callback, post_callback);
  }
  post_callback(stmt);
}

void Mutate(BlockRef block,
            const std::function<void(StmtRef)> &pre_callback,
            const std::function<void(StmtRef)> &post_callback) {
  // No need to deep copy stmts or blocks as the inner Exprs would not be
  // changed inplacely.
  std::vector<StmtRef> stmts = block->stmts();
  for (StmtRef inner_stmt : stmts) {
    Mutate(inner_stmt, pre_callback, post_callback);
  }
  block->set_stmts(std::move(stmts));
}

void Mutate(StmtRef stmt,
            const std::function<void(StmtRef)> &pre_callback,
            const std::function<void(StmtRef)> &post_callback) {
  pre_callback(stmt);
  std::vector<BlockRef> new_blocks;
  for (const BlockRef &inner_block : stmt->block_fields()) {
    BlockRef new_inner_block = inner_block;
    Mutate(new_inner_block, pre_callback, post_callback);
    new_blocks.emplace_back(new_inner_block);
  }
  stmt->set_block_fields(std::move(new_blocks));
  post_callback(stmt);
}
}  // namespace stmt
}  // namespace ir
}  // namespace cinn
