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
#include "paddle/cinn/common/simplify_special_pattern.h"
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
std::optional<ir::IndexExpr> DivMulAddModCornerCase(const ir::IndexExpr& lhs,
                                                    const ir::IndexExpr& rhs) {
  auto lhsMul = lhs.As<ir::Mul>();
  auto rhsMod = rhs.As<ir::Mod>();
  if (!lhsMul || !rhsMod) return std::nullopt;

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
  if (!innerDiv) return std::nullopt;
  if (innerDiv->b().as_index() == rhsMod->b().as_index() &&
      innerDiv->b().as_index() == mult_outer &&
      ProveDivisible(rhsMod->a().as_index() - innerDiv->a().as_index(),
                     mult_outer)) {
    return innerDiv->a().as_index();
  }
  return std::nullopt;
}

// (S0 * 8 + S1 * 2 + S2) + (S1 * 2 + S2) * (-1) ===> 0
std::optional<ir::IndexExpr> AddMulCornerCase(
    const ir::IndexExpr& lhs,
    const ir::IndexExpr& rhs,
    const ir::IndexExpr& scale = ir::IndexExpr(1)) {
  auto rhsMul = rhs.As<ir::Mul>();
  if (!rhsMul) return std::nullopt;
  if (!rhsMul->b().is_constant()) return std::nullopt;

  auto scale_ = scale * rhsMul->b().as_index();
  auto flatten = GetFlattenExprs<ir::Add>(rhsMul->a());
  std::optional<ir::IndexExpr> resOpt;
  ir::IndexExpr res = lhs;
  for (const auto& expr : flatten) {
    if (auto innerMul = expr.As<ir::Mul>()) {
      if (!innerMul->b().is_constant()) return std::nullopt;
      auto resOpt = AddMulCornerCase(res, expr, scale_);
      if (!resOpt.has_value())
        return std::nullopt;
      else
        res = resOpt.value();
    } else {
      if (!IsSumPartialBySymbol(res, expr)) return std::nullopt;
    }
  }

  for (const auto& expr : flatten) {
    if (expr.As<ir::Mul>()) continue;
    if (expr.is_constant()) {
      res = res + expr * scale_;
      continue;
    }
    res = SimplifySymbolicAdd(res, expr, scale_);
  }
  return res;
}

// (S0 + S1 - (S0 + S1) % S2) % S2 == 0
// (S0 + S1 - (S0 + S1) % S2) / S2 == (S0 + S1) / S2
std::optional<ir::IndexExpr> SubModCornerCase(const ir::IndexExpr& lhs,
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
    if (!ProveDivisible(innerMod->b(), rhs)) continue;

    // Check if the sum of all other terms is equal to the lhs of mod
    auto diff = ir::IndexExpr(0);
    for (int64_t j = 0; j < e; ++j)
      if (i != j) diff = diff + flatten[j];
    diff = isNeg ? diff - innerMod->a().as_index()
                 : diff + innerMod->a().as_index();
    if (IsZero(diff)) {
      if (!isDiv) return ir::IndexExpr(0);
      return isNeg ? innerMod->a().as_index() / rhs
                   : -(innerMod->a().as_index() / rhs);
    }

    // For simplify mod case: ((S0 * 256 + S1) % 512 - S1) % 32 == 0
    if (!isDiv) {
      auto diffBeforeNegation = diff;
      auto isDiffNeg = IsNegatedIndexExpr(diff, diffBeforeNegation);
      if (isDiffNeg) diff = diffBeforeNegation;
      auto flatten_diff = GetFlattenExprs<ir::Add>(diff);
      bool isDivisible = true;
      for (const auto& expr : flatten_diff) {
        if (!isDivisible) break;
        if (!ProveDivisible(expr, rhs)) isDivisible = false;
      }
      if (isDivisible) return ir::IndexExpr(0);
    }
  }
  return std::nullopt;
}

// (S0 + S1) / (S0 + S1) == 1
// (S0 + S1) % (S0 + S1) == 0
std::optional<ir::IndexExpr> MultiArgsDivAndMod(const ir::IndexExpr& lhs,
                                                const ir::IndexExpr& rhs,
                                                bool isDiv) {
  // TODO(liujinnan): Dealing with multiple relationships.
  if (lhs == rhs) {
    return isDiv ? ir::IndexExpr(1) : ir::IndexExpr(0);
  }
  return std::nullopt;
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
  if (auto res = DivMulAddModCornerCase(lhs, rhs)) return res.value();
  if (auto res = AddMulCornerCase(lhs, rhs)) return res.value();
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
  if (auto res = SubModCornerCase(lhs, rhs, true)) return res.value();
  if (auto res = MultiArgsDivAndMod(lhs, rhs, true)) return res.value();
  // Add other corner cases
  return std::nullopt;
}

std::optional<ir::IndexExpr> SimplifyModCornerCase(const ir::IndexExpr& lhs,
                                                   const ir::IndexExpr& rhs) {
  if (auto res = SubModCornerCase(lhs, rhs, false)) return res.value();
  // Add other corner cases
  if (auto res = MultiArgsDivAndMod(lhs, rhs, false)) return res.value();
  return std::nullopt;
}

}  // namespace common
}  // namespace cinn
