// Copyright (c) 2024 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/optim/longlong2int.h"
#include "paddle/cinn/ir/ir_mutator.h"
#include "paddle/cinn/ir/ir_printer.h"
#include "paddle/cinn/ir/ir_utils.h"
#include "paddle/cinn/ir/ir_visitor.h"

namespace cinn {
namespace optim {

class CheckOverflow : public ir::IRVisitor {
 public:
  bool is_overflow(Expr* expr) {
    ir::IRVisitor::Visit(expr);
    return is_overflow_;
  }

 private:
  void Visit(const ir::For* for_op) override {
    if (!for_op->extent.is_constant()) is_overflow_ = true;
    if (!for_op->extent.type().is_index_type()) is_overflow_ = true;
    if (curr_product_ > INT_MAX) is_overflow_ = true;

    if (is_overflow_) return;

    curr_product_ *= for_op->extent.as_int64();
    ir::IRVisitor::Visit(&for_op->body);
    curr_product_ /= for_op->extent.as_int64();
  }
  void Visit(const ir::ScheduleBlock* op) override {
    ir::IRVisitor::Visit(&(op->body));
  }
  void Visit(const ir::ScheduleBlockRealize* op) override {
    ir::IRVisitor::Visit(&(op->schedule_block));
  }
  void Visit(const ir::Block* op) {
    for (auto& expr : op->stmts) {
      ir::IRVisitor::Visit(&expr);
    }
  }
  void Visit(const ir::IfThenElse* op) {
    ir::IRVisitor::Visit(&(op->true_case));
    if (op->false_case.defined()) ir::IRVisitor::Visit(&(op->false_case));
  }
  int64_t curr_product_ = 1;
  bool is_overflow_ = false;
};

class CastLonglong2Int : public ir::IRMutator<> {
 public:
  void operator()(Expr* expr) { ir::IRMutator<>::Visit(expr, expr); }

 private:
  void Visit(const ir::_Tensor_* op, Expr* expr) override {
    auto node = expr->As<ir::_Tensor_>();
    std::for_each(node->shape.begin(),
                  node->shape.end(),
                  [&](cinn::ir::Expr& e) { e->convert_int64_to_int32(); });
    CastBufferMeta(node->buffer);
  }
  void Visit(const ir::Load* op, Expr* expr) override {
    auto node = expr->As<ir::Load>();
    std::for_each(node->indices.begin(),
                  node->indices.end(),
                  [&](cinn::ir::Expr& e) { e->convert_int64_to_int32(); });

    ir::IRMutator<>::Visit(&node->tensor, &node->tensor);
  }
  void Visit(const ir::Store* op, Expr* expr) override {
    auto node = expr->As<ir::Store>();
    std::for_each(node->indices.begin(),
                  node->indices.end(),
                  [&](cinn::ir::Expr& e) { e->convert_int64_to_int32(); });
    ir::IRMutator<>::Visit(&node->value, &node->value);
    ir::IRMutator<>::Visit(&node->tensor, &node->tensor);
  }
  void Visit(const ir::For* op, Expr* expr) override {
    auto node = expr->As<ir::For>();
    CastVarWithBound(node->loop_var);
    node->min->convert_int64_to_int32();
    node->extent->convert_int64_to_int32();
    ir::IRMutator<>::Visit(&node->body, &node->body);
  }
  void Visit(const ir::ScheduleBlock* op, Expr* expr) override {
    auto* node = expr->As<ir::ScheduleBlock>();

    std::for_each(node->iter_vars.begin(),
                  node->iter_vars.end(),
                  [&](cinn::ir::Var& v) { CastVarWithBound(v); });

    for (auto& buffer_range : node->read_buffers) {
      if (auto range = buffer_range.As<ir::_BufferRange_>()) {
        std::for_each(range->ranges.begin(),
                      range->ranges.end(),
                      [&](cinn::ir::Var& v) { CastVarWithBound(v); });
        auto bf = range->buffer.as_buffer_ref();
        CastBufferMeta(bf);
      }
    }

    for (auto& buffer_range : node->write_buffers) {
      if (auto range = buffer_range.As<ir::_BufferRange_>()) {
        std::for_each(range->ranges.begin(),
                      range->ranges.end(),
                      [&](cinn::ir::Var& v) { CastVarWithBound(v); });
        auto bf = range->buffer.as_buffer_ref();
        CastBufferMeta(bf);
      }
    }
    ir::IRMutator<>::Visit(&(node->body), &(node->body));
  }

  void Visit(const ir::ScheduleBlockRealize* op, Expr* expr) override {
    auto* node = expr->As<ir::ScheduleBlockRealize>();

    std::for_each(node->iter_values.begin(),
                  node->iter_values.end(),
                  [&](cinn::ir::Expr& e) { e->convert_int64_to_int32(); });
    ir::IRMutator<>::Visit(&node->schedule_block, &node->schedule_block);
  }

  void CastVarWithBound(cinn::ir::Var& var) {  // NOLINT
    if (!var.defined()) return;
    var->convert_int64_to_int32();
    auto lb = var->lower_bound;
    auto ub = var->upper_bound;
    if (lb.defined()) lb->convert_int64_to_int32();
    if (ub.defined()) ub->convert_int64_to_int32();
  }
  void CastBufferMeta(cinn::ir::Buffer& bf) {  // NOLINT
    if (!bf.defined()) return;
    std::for_each(bf->shape.begin(), bf->shape.end(), [&](cinn::ir::Expr& e) {
      e->convert_int64_to_int32();
    });
    std::for_each(bf->strides.begin(),
                  bf->strides.end(),
                  [&](cinn::ir::Expr& e) { e->convert_int64_to_int32(); });
    bf->elem_offset->convert_int64_to_int32();
  }
};

void TryCastLonglong2Int(Expr* expr) {
  VLOG(6) << "Before TryCastLonglong2Int, Expr = \n" << *expr;
  CheckOverflow check_overflow;
  if (!check_overflow.is_overflow(expr)) {
    CastLonglong2Int narrow;
    narrow(expr);
  }
  VLOG(6) << "After TryCastLonglong2Int, Expr = \n" << *expr;
}
}  // namespace optim
}  // namespace cinn
