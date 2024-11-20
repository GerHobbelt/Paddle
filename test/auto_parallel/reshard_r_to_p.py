# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

import numpy as np

import paddle
import paddle.distributed as dist
from paddle.framework import core


class TestReshardRToP:
    def __init__(self):
        self._shape = eval(os.getenv("shape"))
        self._dtype = os.getenv("dtype")
        self._seeds = eval(os.getenv("seeds"))
        self._backend = os.getenv("backend")
        self._mesh = dist.ProcessMesh([0, 1], dim_names=["x"])

    def run_test_case(self):
        if self._backend == "cpu":
            paddle.set_device("cpu")
            place = paddle.CPUPlace()
        elif self._backend == "gpu":
            place = paddle.CUDAPlace(dist.get_rank())

        dev_ctx = core.DeviceContext.create(place)
        a = paddle.ones(self._shape)

        input_tensor = dist.shard_tensor(a, self._mesh, [dist.Replicate()])
        # TODO(liyurui): here due to reshard is static graph logic, dist_attr must be call `_set_partial_dims` for Partial. it should be removed when reshard updated.
        out = dist.reshard(
            input_tensor, self._mesh, [dist.Partial(dist.ReduceType.kRedSum)]
        )

        if dist.get_rank() == 0:
            np.testing.assert_equal(
                out._local_value().numpy(), input_tensor.numpy()
            )
        else:
            zeros = paddle.zeros(self._shape)
            np.testing.assert_equal(out._local_value().numpy(), zeros.numpy())

        assert np.equal(out.shape, input_tensor.shape).all()
        assert np.equal(out._local_shape, input_tensor._local_shape).all()


if __name__ == '__main__':
    TestReshardRToP().run_test_case()
