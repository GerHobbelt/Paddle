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


from paddle.distributed import fleet

from .parallel_base import ParallelModel, ParallelOptimizer


class ShardedDataParallel(ParallelModel):
    """
    ShardedDataParallel converts a single card model to a distrubuted data parallel model

    Args:
        model (paddle.nn.Layer): A single card model to be distributed.
        optimizer (paddle.optimizer.Optimizer): an optimizer to be distributed.
        level (str): Zero stage, can be the following values:
            0: no sharding (pure dp)
            1: Zero Stage1
            2: Zero Stage2
            3: Zero Stage3
            Default: None, which means optimizer is replicated among all process.
        offload (bool): whether enable cpu offload strategy, not implemented currently.
        exclude_layer (list): Specify which layers do not use the zero stage strategy, not implemented currently.
    """

    def __init__(
        self,
        model,
        offload=False,
        exclude_layer=None,
    ):
        super().__init__(model)
        assert offload is False
        assert exclude_layer is None

        self.sharding_parallelizer = self.sharding_parallelizer_func

    def sharding_parallelizer_func(self, model):
        return model


def sharded_data_parallel(
    model, optimizer=None, level=None, offload=False, exclude_layer=None
):
    """
    sharded_data_parallel converts model and optimizer to distributed and supports set zero stage1/2/3

    Args:
        model (paddle.nn.Layer): A single card model to be distributed
        optimizer (paddle.optimizer.Optimizer): an optimizer to be distributed
        level (str): Zero stage, can be the following values:
            0: no sharding (pure dp)
            1: Zero Stage1
            2: Zero Stage2
            3: Zero Stage3
            Default: None, which means optimizer is replicated among all process.
        offload (bool): whether enable cpu offload strategy, not implemented currently.
        exclude_layer (list): Specify which layers do not use the zero stage strategy, not implemented currently.

    Returns:
        ShardedDataParallel: a distributed model
        ParallelOptimizer: a distributed optimizer
    """
    sdp_model = ShardedDataParallel(model, offload, exclude_layer)
    if optimizer is not None:
        optimizer = ParallelOptimizer(optimizer, level)

    # check global_mesh
    mesh = fleet.auto.get_mesh()
    assert (
        mesh is not None
    ), "global mesh must not be None, please call fleet.auto.set_mesh(global_mesh) firstly"
    assert (
        "dp" in mesh.dim_names
    ), "dp must in the mesh dim_names when use sharded_data_parallel"
    return sdp_model, optimizer
