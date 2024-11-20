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

#pragma once
#include "paddle/cinn/common/simplify_corner_case.h"
#include <optional>
#include <stack>
#include <unordered_map>
#include <vector>
#include "paddle/cinn/common/integer_set.h"
#include "paddle/cinn/ir/op/ir_operators.h"
namespace cinn {
namespace common {
// S0 / (S1 * S2) * S1 * S2 + S4 % (S1 * S2) ==> S0
// s.t. (S4 - S0) % (S1 * S2) == 0
static std::pair<bool, ir::IndexExpr> DivMulAddModCornerCase(
    const ir::IndexExpr& lhs, const ir::IndexExpr& rhs) {
  auto lhsMul = lhs.As<ir::Mul>();
  auto rhsMod = rhs.As<ir::Mod>();
  if (!lhsMul || !rhsMod) return std::make_pair(false, ir::IndexExpr());

  // Why inner is lhs of Mul? beacuse we sort by expr length, and the length of
  // inner is longer in this case.
  auto inner = lhsMul->a().as_index();
  auto mult_outer = lhsMul->b().as_index();

  // Calculate the outer multiplier
  while (true) {
    auto mulPtr = inner.As<ir::Mul>();
    if (mulPtr) {
      inner = mulPtr->a().as_index();
      mult_outer = mulPtr->b().as_index() * mult_outer.as_index();
    } else {
      break;
    }
  }

  // Check if the inner expression is a div
  auto innerDiv = inner.As<ir::Div>();
  if (!innerDiv) return std::make_pair(false, ir::IndexExpr());
  if (innerDiv->b().as_index() == rhsMod->b().as_index() &&
      innerDiv->b().as_index() == mult_outer &&
      ProveDivisible(rhsMod->a().as_index() - innerDiv->a().as_index(),
                     mult_outer)) {
    return std::make_pair(true, innerDiv->a().as_index());
  }
  return std::make_pair(false, ir::IndexExpr());
}

// (S0 + S1 - (S0 + S1) % S2) % S2 == 0
// (S0 + S1 - (S0 + S1) % S2) / S2 == (S0 + S1) / S2
static std::pair<bool, ir::IndexExpr> SubModCornerCase(const ir::IndexExpr& lhs,
                                                       const ir::IndexExpr& rhs,
                                                       bool isDiv) {
  auto flatten = GetFlattenExprs<ir::Add>(lhs);

  for (int64_t i = 0, e = flatten.size(); i < e; ++i) {
    // Check if negation
    ir::IndexExpr beforeNegation = flatten[i];
    auto isNeg = IsNegatedIndexExpr(flatten[i], beforeNegation);

    // Check if the negation term is a mod
    auto innerMod = beforeNegation.As<ir::Mod>();
    if (!innerMod) continue;
    if (innerMod->b() != rhs) continue;

    // Check if the sum of all other terms is equal to the lhs of mod
    auto diff = ir::IndexExpr(0);
    for (int64_t j = 0; j < e; ++j)
      if (i != j) diff = diff + flatten[j];
    diff = isNeg ? diff - innerMod->a().as_index()
                 : diff + innerMod->a().as_index();
    if (IsZero(diff)) {
      if (!isDiv) return std::make_pair(true, ir::IndexExpr(0));
      return isNeg ? std::make_pair(true, innerMod->a().as_index() / rhs)
                   : std::make_pair(true, -(innerMod->a().as_index() / rhs));
    }
  }
  return std::make_pair(false, ir::IndexExpr());
}

std::optional<ir::IndexExpr> SimplifyCornerCase(const ir::IndexExpr& expr) {
  switch (expr.node_type()) {
    case ir::IrNodeTy::IntImm:
      [[fallthrough]];
    case ir::IrNodeTy::_Var_:
      return expr;
    case ir::IrNodeTy::Add:
      return SimplifyAddCornerCase(expr->operand(0).as_index(),
                                   expr->operand(1).as_index());
    case ir::IrNodeTy::Mul:
      return SimplifyMulCornerCase(expr->operand(0).as_index(),
                                   expr->operand(1).as_index());
    case ir::IrNodeTy::Div:
      return SimplifyDivCornerCase(expr->operand(0).as_index(),
                                   expr->operand(1).as_index());
    case ir::IrNodeTy::Mod:
      return SimplifyModCornerCase(expr->operand(0).as_index(),
                                   expr->operand(1).as_index());
  }
}

std::optional<ir::IndexExpr> SimplifyAddCornerCase(const ir::IndexExpr& lhs,
                                                   const ir::IndexExpr& rhs) {
  auto ret = DivMulAddModCornerCase(lhs, rhs);
  if (ret.first) return ret.second;

  // Add other corner cases
  return std::nullopt;
}

std::optional<ir::IndexExpr> SimplifyMulCornerCase(const ir::IndexExpr& lhs,
                                                   const ir::IndexExpr& rhs) {
  // Add other corner cases
  return std::nullopt;
}

std::optional<ir::IndexExpr> SimplifyDivCornerCase(const ir::IndexExpr& lhs,
                                                   const ir::IndexExpr& rhs) {
  auto ret = SubModCornerCase(lhs, rhs, true);
  if (ret.first) return ret.second;
  // Add other corner cases
  return std::nullopt;
}

std::optional<ir::IndexExpr> SimplifyModCornerCase(const ir::IndexExpr& lhs,
                                                   const ir::IndexExpr& rhs) {
  auto ret = SubModCornerCase(lhs, rhs, false);
  if (ret.first) return ret.second;

  // Add other corner cases
  return std::nullopt;
}

}  // namespace common
}  // namespace cinn
