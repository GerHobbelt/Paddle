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
import logging
import re

import paddle
import paddle.distributed as dist

from .parallel_base import ParallelModel, ParallelOptimizer


class PlanBase:
    def apply(self, param, process_mesh, shard_weight, shard_bias):
        raise NotImplementedError("Don't call the PlanBase directly.")


class ColWiseParallel(PlanBase):
    """
    Col wise parallel plan.
    Will try to split weight on the second dim and the bias on the first dim.
    This api is designed for paddle.nn.Linear or paddle.nn.Embedding.
    If any other instance of paddle.nn.Layer is passed,
    this plan will try to split `layer.weight` and `layer.bias` if it has.

    Note: `layer.weight` should have two dims.
    Note: `layer.bias` should have one dim.
    """

    def __init__(self):
        super().__init__()

    def apply(self, layer, process_mesh, shard_weight=True, shard_bias=True):
        """
        With calling of this function, parameters will be marked as split and turn in to shard_tensor.
        :param layer: paddle.nn.Layer, layer to be split
        :param process_mesh: dist.ProcessMesh, process_mesh where the split will work on
        :param shard_weight: BOOL, whether shard the weight or not
        :param shard_bias: BOOL, whether shard the weight or not
        :return: no return, the shard will happen on the origin layer
        """
        index = process_mesh.dim_names.index('mp')  # get the axis for the split
        size = len(process_mesh.shape)
        placement = [dist.Replicate() for _ in range(size)]
        assert isinstance(layer, paddle.nn.Layer)
        if not isinstance(layer, (paddle.nn.Linear, paddle.nn.Embedding)):
            logging.warning(
                f"ColWiseParallel is designed to handle Linear and Embedding. "
                f"But got {layer.__class__.__name__}. "
                f"Will try to shard weight and bias if the layer contains one."
            )
        if (
            hasattr(layer, "weight")
            and layer.weight is not None
            and shard_weight
        ):
            placement[index] = dist.Shard(1)
            assert len(layer.weight.shape) == 2
            layer.weight = dist.shard_tensor(
                layer.weight,
                process_mesh,
                placement,
            )
        if hasattr(layer, "bias") and layer.bias is not None and shard_bias:
            placement[index] = dist.Shard(0)
            assert len(layer.bias.shape) == 1
            layer.bias = dist.shard_tensor(layer.bias, process_mesh, placement)


class RowWiseParallel(PlanBase):
    """
    Row wise parallel plan.
    Will try to split weight on the first dim.
    This api is designed for paddle.nn.Linear or paddle.nn.Embedding.
    If any other instance of paddle.nn.Layer is passed, this plan will try to split `layer.weight` if it has.

    Note: `layer.weight` should have two dims.
    """

    def __init__(self):
        super().__init__()

    def apply(self, layer, process_mesh, shard_weight=True, shard_bias=False):
        """
        With calling of this function, parameters will be marked as split and turn in to shard_tensor.
        :param layer: paddle.nn.Layer, layer to be split
        :param process_mesh: dist.ProcessMesh, process_mesh where the split will work on
        :param shard_weight: BOOL, whether shard the weight or not
        :param shard_bias: BOOL, whether shard the weight or not
        :return: no return, the shard will happen on the origin layer
        """
        index = process_mesh.dim_names.index('mp')  # get the axis for the split
        size = len(process_mesh.shape)
        placement = [dist.Replicate() for _ in range(size)]
        placement[index] = dist.Shard(0)
        assert isinstance(layer, paddle.nn.Layer)
        if not isinstance(layer, (paddle.nn.Linear, paddle.nn.Embedding)):
            logging.warning(
                f"RowWiseParallel is designed to handle Linear and Embedding. "
                f"But got {layer.__class__.__name__}. "
                f"Will try to shard weight if the layer contains one."
            )
        if (
            hasattr(layer, "weight")
            and layer.weight is not None
            and shard_weight
        ):
            assert len(layer.weight.shape) == 2
            layer.weight = dist.shard_tensor(
                layer.weight,
                process_mesh,
                placement,
            )


class TensorParallel(ParallelModel):
    def __init__(self, model, parallelize_plan=None):
        super().__init__(model)
        if parallelize_plan is not None:
            assert isinstance(parallelize_plan, dict)
            for key, plan in parallelize_plan.items():
                assert isinstance(
                    key, str
                ), "The key of the parallelize plan should be a string."
                assert isinstance(
                    plan, PlanBase
                ), "The value the the parallelize plan should be a instance of PlanBase."

            self.global_mesh = dist.auto_parallel.get_mesh()
            self.parallelize_plan = parallelize_plan
            self.tp_parallelizer = self.tensor_parallelizer_fn

    def get_mesh(self):
        # TODO(yaliu): fit pp
        # Get local mesh for current pp.
        assert "mp" in self.global_mesh.dim_names
        if "pp" in self.global_mesh.dim_names:
            assert (
                self.global_mesh.get_dim_size("pp") == 1
            ), "Not support pp with mp for now."
            mesh = self.global_mesh.get_mesh_with_dim("pp")[0]
        else:
            mesh = self.global_mesh
        assert len(mesh.shape) in [1, 2]
        return mesh

    def match_layer(self, name):
        # Match the layer to a plan.
        # Will return the plan if the layer hits one, otherwise return None.
        for key, plan in self.parallelize_plan.items():
            shard_weight = True
            shard_bias = True
            # Find some plan for specific parameter, such as
            # "lm_head.weight": ColWiseParallel()
            # Only support weight or bias.
            if key.endswith(".weight"):
                key = key.replace(".weight", "")
                shard_bias = False
            elif key.endswith(".bias"):
                key = key.replace(".bias", "")
                shard_weight = False
            re_find = re.match(key, name)
            if key == name or (re_find is not None and re_find.string == name):
                return plan, shard_weight, shard_bias
        return None, None, None

    def tensor_parallelizer_fn(self, model):
        if self.parallelize_plan is None:
            return
        for name, layer in model.named_sublayers():
            if len(layer.sublayers()) == 0:
                plan, shard_weight, shard_bias = self.match_layer(name)
                if plan is not None:
                    plan.apply(layer, self.get_mesh(), shard_weight, shard_bias)
        return model


def tensor_parallel(model, parallelize_plan=None, optimizer=None):
    """
    Tensor parallel.
    :param model: paddle.nn.Layer, the model to be shard into tensor parallel.
    :param parallelize_plan: Dict, the plan to shard the layer.
    :param optimizer: paddle.optimizer.Optimizer, the optimizer.
    :return:
        model: model after sharding
        optimizer: optimizer after sharding

    NOTE: the plan should be a dict maps layer name or parameter name to a split_plan,
    which will be used to split the layer or the parameter. The name can be written in regular format.

    An example for the plan is:
    ```
    plan = {
        "llama.embed_tokens": ColWiseParallel(),
        "llama.layers.*.self_attn.q_proj": ColWiseParallel(),
        "llama.layers.*.self_attn.k_proj": ColWiseParallel(),
        "llama.layers.*.self_attn.v_proj": ColWiseParallel(),
        "llama.layers.*.self_attn.o_proj": RowWiseParallel(),
        "llama.layers.*.mlp.gate_proj": ColWiseParallel(),
        "llama.layers.*.mlp.up_proj": ColWiseParallel(),
        "llama.layers.*.mlp.down_proj": RowWiseParallel(),
        "lm_head.weight": ColWiseParallel(),
    }
    ```
    """
    if parallelize_plan is None:
        # Do nothing if no plan.
        logging.warning(
            "No parallelize plan, tensor parallel won't do anything."
        )
        return model, optimizer

    global_mesh = dist.auto_parallel.get_mesh()

    assert (
        global_mesh is not None
    ), "global mesh must not be None, please call fleet.auto.set_mesh(global_mesh) firstly"
    assert (
        "mp" in global_mesh.dim_names
    ), "mp must in the mesh dim_names when use tensor_parallel"

    model = TensorParallel(model, parallelize_plan)
    if optimizer is not None:
        optimizer = ParallelOptimizer(optimizer)

    return model, optimizer
