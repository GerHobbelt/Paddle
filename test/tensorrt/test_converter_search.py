# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

import unittest

import numpy as np
from tensorrt_test_base import TensorRTBaseTest

import paddle


class TestArgmaxTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.argmax
        self.api_args = {
            "x": np.random.randn(2, 3).astype(np.float32),
            "axis": -1,
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [1, 3]}
        self.max_shape = {"x": [5, 3]}

    def test_trt_result(self):
        self.check_trt_result()


class TestTopkCase1TRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.topk
        self.api_args = {
            "x": np.random.randn(2, 3).astype("float32"),
            "k": 1,
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [1, 3]}
        self.max_shape = {"x": [5, 3]}

    def test_trt_result(self):
        self.check_trt_result()


class TestTopkCase2TRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.topk
        self.api_args = {
            "x": np.random.randn(2).astype("int64"),
            "k": 1,
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [1]}
        self.max_shape = {"x": [5]}

    def test_trt_result(self):
        self.check_trt_result()


class TestTopkCase3TRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.topk
        self.api_args = {
            "x": np.random.randn(2).astype("int64"),
            "k": 1,
            "axis": 0,
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [1]}
        self.max_shape = {"x": [5]}

    def test_trt_result(self):
        self.check_trt_result()


class TestTopkCase4TRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.topk
        self.api_args = {
            "x": np.array([[1, 4, 5, 7], [2, 6, 2, 5]]).astype("int64"),
            "k": np.array([1]).astype("int64"),
            "axis": -1,
        }
        self.program_config = {"feed_list": ["x", "k"]}
        self.min_shape = {}
        self.max_shape = {}

    def test_trt_result(self):
        self.check_trt_result()


if __name__ == '__main__':
    unittest.main()
