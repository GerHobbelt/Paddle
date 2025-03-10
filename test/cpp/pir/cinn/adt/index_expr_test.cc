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

#include <glog/logging.h>
#include <gtest/gtest.h>
#include "paddle/cinn/common/integer_set.h"
#include "paddle/cinn/common/simplify_special_pattern.h"
#include "paddle/cinn/ir/ir.h"
#include "paddle/cinn/ir/ir_base.h"
#include "paddle/cinn/ir/ir_mutator.h"
#include "paddle/cinn/ir/op/ir_operators.h"

namespace cinn {
namespace common {
TEST(IndexExpr, IndexExpr_0) {
  ir::IndexExpr a(14);
  ir::IndexExpr b(7);
  Expr d(6);
  ir::Expr c0 = a + b;
  ir::Expr c1 = a - b;
  ir::Expr c2 = a * b;
  ir::Expr c3 = a / b;
  ir::Expr c4 = a % b;

  ir::Expr c5 = a / d.as_index();
  ir::Expr c6 = a % d.as_index();

  EXPECT_EQ(c0, Expr(21));
  EXPECT_EQ(c1, Expr(7));
  EXPECT_EQ(c2, Expr(98));
  EXPECT_EQ(c3, Expr(2));
  EXPECT_EQ(c4, Expr(0));
  EXPECT_EQ(c5, Expr(2));
  EXPECT_EQ(c6, Expr(2));
}

TEST(IndexExpr, IndexExpr_1) {
  auto S4 =
      ir::Var(ir::Expr(static_cast<int64_t>(1)), ir::Expr(INT32_MAX), "S4");
  auto S5 =
      ir::Var(ir::Expr(static_cast<int64_t>(1)), ir::Expr(INT32_MAX), "S5");
  auto S6 =
      ir::Var(ir::Expr(static_cast<int64_t>(1)), ir::Expr(INT32_MAX), "S6");
  auto S7 =
      ir::Var(ir::Expr(static_cast<int64_t>(1)), ir::Expr(INT32_MAX), "S7");

  ir::IndexExpr e1 = (S5 * ((S4 * (S5 * (S6 * S7))) / S5));
  ir::IndexExpr e2 = (S4 * (S5 * (S6 * S7))) / S5;
  ir::IndexExpr e3 = (S4 * S5) / S5;

  ir::IndexExpr e4 = (S4 * (S5 * (S6 * S7)) + S5) / S5;
  ir::IndexExpr e5 = (S4 * (S5 * (S6 * S7)) + 2 * S5) / S5;

  ir::IndexExpr e6 = (S4 * (S5 * (S6 * S7)) + S5 / S6) / S5;
  ir::IndexExpr e7 = (S4 * (S5 * (S6 * S7)) + 2 * S5 / S6) / S5;

  EXPECT_EQ(e1.as_index().Normalize(), ir::IndexExpr((S6 * S7) * S4 * S5));
  EXPECT_EQ(e2.as_index().Normalize(), ir::IndexExpr((S6 * S7) * S4));
  EXPECT_EQ(e3.as_index().Normalize(), ir::IndexExpr(S4));
  EXPECT_EQ(e4.as_index().Normalize(), ir::IndexExpr(((S6 * S7) * S4) + 1));
  EXPECT_EQ(e5.as_index().Normalize(), ir::IndexExpr(((S6 * S7) * S4) + 2));
  EXPECT_EQ(e6.as_index().Normalize(),
            ir::IndexExpr(((S6 * S7) * S4) + (1 / S6)));
  EXPECT_EQ(e7.as_index().Normalize(),
            ir::IndexExpr(((S6 * S7) * S4) + (2 / S6)));
}

TEST(IndexExpr, IndexExpr_2) {
  auto S4 =
      ir::Var(ir::Expr(static_cast<int64_t>(1)), ir::Expr(INT32_MAX), "S4");
  auto S5 =
      ir::Var(ir::Expr(static_cast<int64_t>(1)), ir::Expr(INT32_MAX), "S5");
  auto S6 =
      ir::Var(ir::Expr(static_cast<int64_t>(1)), ir::Expr(INT32_MAX), "S6");
  auto S7 =
      ir::Var(ir::Expr(static_cast<int64_t>(1)), ir::Expr(INT32_MAX), "S7");

  ir::Expr q1 = S4;
  ir::Expr q2 = S4;

  ir::Expr q3 = S4 + S5;
  ir::Expr q4 = S5 + S4;

  ir::Expr q5 = S4 * 2 + S5 / 4;
  ir::Expr q6 = S5 / 4 + S4 * 2;

  ir::Expr q7 = S4 + S5 + S6;
  ir::Expr q8 = S5 + (S4 + S6);

  ir::Expr q9 = S4 + (S5 + S7 / 4 + S6 * 2);
  ir::Expr q10 = S5 + (S4 + S6 * 2 + S7 / 4);

  ir::Expr q11 = (S7 + S5) + (S4 + S6);
  ir::Expr q12 = (S4 + S5) + (S6 + S7);

  ir::Expr q13 = (S4 + S5) * 3 + (S6 / 2 + S7) * 2;
  ir::Expr q14 = (S6 / 2 + S7) * 2 + (S4 + S5) * 3;

  ir::Expr q15 = (S4 + S5 * 2) * 3 + (S6 / 2 + S7) * 2;
  ir::Expr q16 = (S6 / 2 + S7) * 2 + (S4 + S5 * 2) * 3;

  ir::Expr q17 = (S4 + S5 * 2) * 3 + (S6 / 2 + S7) * 2 + S4;
  ir::Expr q18 = (S6 / 2 + S7) * 2 + (S4 + S5 * 2) * 3 + S4;

  ir::Expr q19 = (S4 + S5 * 2) * 3 + (S6 / 2 + S7) * 2 + S4;
  ir::Expr q20 = (S6 / 2 + S7) * 2 + (S4 + S5 * 2) * 3 + S5;

  EXPECT_EQ(q1.as_index().Normalize(), q2.as_index().Normalize());
  EXPECT_EQ(q3.as_index().Normalize(), q4.as_index().Normalize());
  EXPECT_EQ(q5.as_index().Normalize(), q6.as_index().Normalize());
  EXPECT_EQ(q7.as_index().Normalize(), q8.as_index().Normalize());
  EXPECT_EQ(q9.as_index().Normalize(), q10.as_index().Normalize());
  EXPECT_EQ(q11.as_index().Normalize(), q12.as_index().Normalize());
  EXPECT_EQ(q13.as_index().Normalize(), q14.as_index().Normalize());
  EXPECT_EQ(q15.as_index().Normalize(), q16.as_index().Normalize());
  EXPECT_EQ(q17.as_index().Normalize(), q18.as_index().Normalize());
  EXPECT_NE(q19.as_index().Normalize(), q20.as_index().Normalize());
}

TEST(IndexExpr, IndexExpr_3) {
  auto S4 =
      ir::Var(ir::Expr(static_cast<int64_t>(1)), ir::Expr(INT32_MAX), "S4");
  auto S5 =
      ir::Var(ir::Expr(static_cast<int64_t>(1)), ir::Expr(INT32_MAX), "S5");
  auto S6 =
      ir::Var(ir::Expr(static_cast<int64_t>(1)), ir::Expr(INT32_MAX), "S6");
  auto S7 =
      ir::Var(ir::Expr(static_cast<int64_t>(1)), ir::Expr(INT32_MAX), "S7");

  // `Add` corner cases
  ir::Expr q1 = S4 / S5 * S5 + S4 % S5;
  ir::Expr q2 = (S4 + S5) / S6 * S6 + (S4 + S5) % S6;
  ir::Expr q3 = S4 / (S5 + S6) * (S5 + S6) + S4 % (S5 + S6);
  ir::Expr q4 = (S4 + S5) / (S6 + S7) * (S6 + S7) + (S4 + S5) % (S6 + S7);
  ir::Expr q5 = (S4 + S5) / 5 * 5 + (S4 + S5) * 11 % 5;
  ir::Expr q14 = (S4 + S5) / (S6 * S7) * S6 * S7 + (S4 + S5) % (S6 * S7);
  ir::Expr q15 =
      (S4 * 256 + S5 + S6 * 1024) % 25088 / 512 * 512 + (S4 * 256 + S5) % 512;
  ir::Expr q16 =
      ((S4 * 256 + S5) / S6 / S7 * S7 + (S4 * 256 + S5) / S6 % S7) * S6 +
      (S4 * 256 + S5) % S6;

  // `Div` corner cases
  ir::Expr q6 = (S4 % S5 - S4) / S5;
  ir::Expr q7 = (S4 - S4 % S5) / S5;
  ir::Expr q8 = ((S4 + S5) % S6 - S4 - S5) / S6;
  ir::Expr q9 = (S4 + S5 - (S4 + S5) % S6) / S6;

  // `Mod` corner cases
  ir::Expr q10 = (S4 % S5 - S4) % S5;
  ir::Expr q11 = (S4 - S4 % S5) % S5;
  ir::Expr q12 = ((S4 + S5) % S6 - S4 - S5) % S6;
  ir::Expr q13 = (S4 + S5 - (S4 + S5) % S6) % S6;

  EXPECT_EQ(q1.as_index().Normalize(), ir::IndexExpr(S4));
  EXPECT_EQ(q2.as_index().Normalize(), ir::IndexExpr(S4 + S5));
  EXPECT_EQ(q3.as_index().Normalize(), ir::IndexExpr(S4));
  EXPECT_EQ(q4.as_index().Normalize(), ir::IndexExpr(S4 + S5));
  EXPECT_EQ(q5.as_index().Normalize(), ir::IndexExpr(S4 + S5));
  EXPECT_EQ(q6.as_index().Normalize(), ir::IndexExpr((S4 / S5) * (-1)));
  EXPECT_EQ(q7.as_index().Normalize(), ir::IndexExpr(S4 / S5));
  EXPECT_EQ(q8.as_index().Normalize(), ir::IndexExpr(((S4 + S5) / S6) * (-1)));
  EXPECT_EQ(q9.as_index().Normalize(), ir::IndexExpr((S4 + S5) / S6));
  EXPECT_EQ(q10.as_index().Normalize(), ir::IndexExpr(0));
  EXPECT_EQ(q11.as_index().Normalize(), ir::IndexExpr(0));
  EXPECT_EQ(q12.as_index().Normalize(), ir::IndexExpr(0));
  EXPECT_EQ(q13.as_index().Normalize(), ir::IndexExpr(0));
  EXPECT_EQ(q14.as_index().Normalize(), ir::IndexExpr(S4 + S5));
  EXPECT_EQ(q15.as_index().Normalize(),
            ir::IndexExpr((S4 * 256 + S5 + S6 * 1024)) % 25088);
  EXPECT_EQ(q16.as_index().Normalize(), ir::IndexExpr(S4 * 256 + S5));
}
}  // namespace common
}  // namespace cinn
