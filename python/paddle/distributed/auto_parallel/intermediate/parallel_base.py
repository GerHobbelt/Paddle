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

import paddle
import paddle.distributed as dist
from paddle import pir
from paddle.base.framework import (
    in_dygraph_mode,
    in_pir_mode,
)
from paddle.distributed import fleet
from paddle.nn import Layer
from paddle.optimizer import Optimizer


def is_tensor(tensor):
    if in_dygraph_mode():
        return isinstance(tensor, paddle.Tensor)
    elif in_pir_mode():
        return isinstance(tensor, pir.Value)
    else:
        raise RuntimeError(
            "PipelineParallel are only supported in dynamic or pir mode."
        )


class ParallelOptimizer:
    def __init__(self, optimizer, level=None):
        self.level = None
        self.optimizer = None

        if isinstance(optimizer, ParallelOptimizer):
            self.optimizer = optimizer.optimizer
            self.level = optimizer.level
        else:
            assert isinstance(optimizer, Optimizer)
            self.optimizer = optimizer
            assert level in ("os", "os_g", "p_g_os", None)
            self.level = level

        self.is_initialized = False

    def parallelize(self, parallelized_parameters):
        assert self.optimizer is not None
        if self.is_initialized:
            return self.optimizer
        # 1.replace optimizer parameters
        self.optimizer._parameter_list = parallelized_parameters
        if isinstance(parallelized_parameters[0], dict):
            self.optimizer._param_groups = []
            for param_group in self.parallelized_parameters:
                self.optimizer._add_param_group(param_group.copy())
        else:
            self.optimizer._param_groups = self.optimizer._parameter_list
        # 2.wrap with shard_optimizer
        mesh = fleet.auto.get_mesh()
        if self.level == "os":
            self.optimizer = dist.shard_optimizer(
                self.optimizer, dist.ShardingStage1(mesh)
            )
        elif self.level == "os_g":
            self.optimizer = dist.shard_optimizer(
                self.optimizer, dist.ShardingStage2(mesh)
            )
        elif self.level == "p_g_os":
            self.optimizer = dist.shard_optimizer(
                self.optimizer, dist.ShardingStage3(mesh)
            )
        else:
            self.optimizer = dist.shard_optimizer(self.optimizer)
        self.is_initialized = True

        return self.optimizer


class ParallelModel:
    def __init__(self, model):
        super().__init__()
        self.pp_parallelizer = None
        self.tp_parallelizer = None
        self.sharding_parallelizer = None
        self.model = None

        if isinstance(model, ParallelModel):
            self.pp_parallelizer = model.pp_parallelizer
            self.tp_parallelizer = model.tp_parallelizer
            self.sharding_parallelizer = model.sharding_parallelizer
            self.model = model.model
        else:
            assert isinstance(model, Layer)
            self.model = model

        self.is_parallelized = False

    def get_mesh(self, pp_idx=0):
        mesh = fleet.auto.get_mesh()
        if "pp" in mesh.dim_names:
            mesh = mesh.get_mesh_with_dim("pp", pp_idx)
        return mesh

    def parallelize_model(self):
        assert self.model is not None
        if self.is_parallelized:
            return self.model

        if self.pp_parallelizer is not None:
            assert callable(self.pp_parallelizer)
            self.model = self.pp_parallelizer(self.model)

        if self.tp_parallelizer is not None:
            assert callable(self.tp_parallelizer)
            self.model = self.tp_parallelizer(self.model)

        if self.sharding_parallelizer is not None:
            assert callable(self.sharding_parallelizer)
            self.model = self.sharding_parallelizer(self.model)

        self.is_parallelized = True

        return self.model


def parallelize_model_and_optimizer(model, optimizer=None):
    assert isinstance(model, ParallelModel)
    parallelized_model = model.parallelize_model()
    parallelized_optimizer = None
    if optimizer is not None:
        assert isinstance(optimizer, ParallelOptimizer)
        parallelized_optimizer = optimizer.parallelize(
            parallelized_model.parameters()
        )

    return parallelized_model, parallelized_optimizer
