#   Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

import paddle.distributed as dist
from paddle.distributed import fleet
from paddle.nn import Layer


class ParallelOptimizer:
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.is_initialized = False

    def __getattr__(self, item):
        return getattr(self.optimizer, item)

    def parallelize(self, level, parallized_parameters):
        assert self.optimizer is not None
        if self.is_initialized:
            return
        # 1.replace optimizer parameters
        self.optimizer._parameter_list = parallized_parameters

        # 2.wrap with shard_optimizer
        mesh = fleet.auto.get_mesh()
        if level == "os":
            self.optimizer = dist.shard_optimizer(
                self.optimizer, dist.ShardingStage1(mesh)
            )
        elif level == "os_g":
            self.optimizer = dist.shard_optimizer(
                self.optimizer, dist.ShardingStage2(mesh)
            )
        elif level == "p_g_os":
            self.optimizer = dist.shard_optimizer(
                self.optimizer, dist.ShardingStage3(mesh)
            )
        else:
            self.optimizer = dist.shard_optimizer(self.optimizer)
        self.is_initialized = True


class ParallelBase(Layer):
    def __init__(self, model, optimizer=None):
        super().__init__()
        self.pp_parallelizer = None
        self.tp_parallelizer = None
        self.sharding_parallelizer = None
        self.level = None

        if isinstance(model, ParallelBase):
            self.pp_parallelizer = model.pp_parallelizer
            self.tp_parallelizer = model.tp_parallelizer
            self.sharding_parallelizer = model.sharding_parallelizer

            self.model = model.model
            self.optimizer = (
                ParallelOptimizer(optimizer)
                if model.optimizer.optimizer is None
                else model.optimizer
            )
        else:
            self.model = model
            self.optimizer = ParallelOptimizer(optimizer)

        self.is_parallelized = False

    def parallelize_model_and_optimizer(self):
        if self.pp_parallelizer is not None:
            assert callable(self.pp_parallelizer)
            self.model = self.pp_parallelizer(self.model)

        if self.tp_parallelizer is not None:
            assert callable(self.tp_parallelizer)
            self.model = self.tp_parallelizer(self.model)

        if self.sharding_parallelizer is not None:
            assert callable(self.sharding_parallelizer)
            self.model = self.sharding_parallelizer(self.model)

        assert isinstance(self.optimizer, ParallelOptimizer)
        assert not self.optimizer.is_initialized
        self.optimizer.parallelize(self.level, self.model.parameters())

    def forward(self, *args):
        if not self.is_parallelized:
            self.parallelize_model_and_optimizer()
            self.is_parallelized = True
        return self.model(*args)
