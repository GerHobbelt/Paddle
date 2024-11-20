/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <glog/logging.h>

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "paddle/common/ddim.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/mixed_vector.h"
#include "paddle/utils/test_macros.h"

namespace paddle {
namespace framework {

/*
 * LoD is short for Level of Details.
 *
 * - in a level, each element indicates relative offset of the lower level
 * - the first element should be 0 and that indicates that this sequence start
 * from 0
 * - each sequence's begin and end(no-inclusive) is level[id, id+1]
 *
 * For example:
 *    3-level LoD stores
 *
 *    0 2 3
 *    0 2 4 7
 *    0 2 5 7 10 12 15 20
 */
using LoD = std::vector<phi::Vector<size_t>>;

std::string LoDToString(const LoD& lod);

TEST_API bool operator==(const LoD& a, const LoD& b);

/*
 * Check whether this lod's format is valid.
 *
 * ATTENTION:
 *   - Empty lod is treated as valid.
 *
 * It will check two things:
 *
 *  1. all the offsets in a level should be non-descending.
 *  2. there should be more than 2 offsets existing in each level.
 *  3. the higher level's last offset should equals the lower level's size-1.
 *  4. the first offset(the begin offset) of each level should be 0.
 *  5. the lowest level's last offset should equals `tensor_height` if
 * tensor_height>0.
 */

TEST_API bool CheckLoD(const LoD& in, int tensor_height = -1);

TEST_API LoD ConvertToOffsetBasedLoD(const LoD& length_lod);

/*
 * Serialize/Deserialize phi::DenseTensor to std::ostream
 * You can pass ofstream or ostringstream to serialize to file
 * or to a in memory string. GPU tensor will be copied to CPU.
 */
void SerializeToStream(std::ostream& os,
                       const phi::DenseTensor& tensor,
                       const phi::DeviceContext& dev_ctx);
void DeserializeFromStream(std::istream& is,
                           phi::DenseTensor* tensor,
                           const phi::DeviceContext& dev_ctx);
void DeserializeFromStream(std::istream& is,
                           phi::DenseTensor* tensor,
                           const phi::DeviceContext& dev_ctx,
                           const size_t& seek,
                           const std::vector<int64_t>& shape);

void SerializeToStream(std::ostream& os, const phi::DenseTensor& tensor);

void DeserializeFromStream(std::istream& os, phi::DenseTensor* tensor);

}  // namespace framework
}  // namespace paddle
