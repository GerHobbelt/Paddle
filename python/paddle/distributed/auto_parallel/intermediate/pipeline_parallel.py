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

import itertools
import re
from collections import OrderedDict
from enum import Enum

import paddle.distributed as dist
from paddle.distributed import fleet
from paddle.distributed.utils.log_utils import get_logger

from .parallel_base import ParallelModel, ParallelOptimizer, is_tensor

logger = get_logger("INFO", __name__)


class SplitPoint(Enum):
    BEGINNING = 0
    END = 1


class PipelineParallel(ParallelModel):
    def __init__(self, model, split_spec):
        super().__init__(model)
        self.split_spec = split_spec
        self.pp_parallelizer = self.pipeline_parallel_fn

    def pipeline_parallel_fn(self, model):
        mesh = fleet.auto.get_mesh()
        pipeline_stage_num = mesh.get_dim_size("pp")
        assert len(self.split_spec) == pipeline_stage_num - 1

        name_to_layer = {}
        for layer_name, layer in model.named_sublayers():
            name_to_layer[layer_name] = layer

        def get_layer_by_name(name):
            assert (
                name in name_to_layer
            ), f"layer name:{name} not in the model, please check the split_spec"
            return name_to_layer[name]

        def forward_post_hook(layer, input, output):
            pipeline_stage_index = layer.pipeline_stage_index
            split_point = layer.split_point
            assert split_point == SplitPoint.END
            # reshard to next pipeline stage
            if isinstance(output, (dict, OrderedDict)):
                for key, tensor in output.items():
                    assert is_tensor(tensor)
                    output[key] = dist.reshard(
                        tensor,
                        self.get_mesh(pipeline_stage_index + 1),
                        tensor.placements,
                    )
            elif isinstance(output, (list, tuple)):
                for i in range(len(output)):
                    assert is_tensor(output[i])
                    output[i] = dist.reshard(
                        output[i],
                        self.get_mesh(pipeline_stage_index + 1),
                        output[i].placements,
                    )
            elif is_tensor(output):
                output = dist.reshard(
                    output,
                    self.get_mesh(pipeline_stage_index + 1),
                    output.placements,
                )
            else:
                raise ValueError(
                    f"output should be a dict of tensors or list of tensors or tensor, but {type(output)}"
                )
            return output

        def forward_pre_hook(layer, input):
            assert split_point == SplitPoint.BEGINNING
            # TODO(deepllz): support in the future
            return input

        # step1: set every layer's own pipeline_stage_index
        split_layer_names = list(self.split_spec.keys())
        sublayer_names = [name for name, _ in model.named_sublayers()]
        # Mark which layer is the next pipeline stage
        pipline_layer_mark = [0 for _ in range(len(sublayer_names))]
        for split_layer_name in split_layer_names:
            split_point = self.split_spec[split_layer_name]
            index = sublayer_names.index(split_layer_name)
            if split_point == SplitPoint.END:
                is_valid = False
                for i in range(index + 1, len(sublayer_names)):
                    if not sublayer_names[i].startswith(split_layer_name):
                        pipline_layer_mark[i] = 1
                        is_valid = True
                        break
                assert (
                    is_valid
                ), f"the last layer:{split_layer_name} must not be SplitPoint.END, please check the split_spec"
            else:
                raise NotImplementedError(
                    "SplitPoint.BEGINNING is not supported currently"
                )
                pipline_layer_mark[index] = 1
        # the inclusiveSum of pipline_layer_mark is the pipeline stage index
        pipline_stage_index = list(itertools.accumulate(pipline_layer_mark))
        for index, (name, layer) in enumerate(model.named_sublayers()):
            layer.pipeline_stage_index = pipline_stage_index[index]

        # step2: insert reshard
        for name in split_layer_names:
            layer = get_layer_by_name(name)
            split_point = self.split_spec[name]
            layer.split_point = split_point
            if split_point == SplitPoint.END:
                layer.register_forward_post_hook(forward_post_hook)
            else:
                raise NotImplementedError(
                    "SplitPoint.BEGINNING is not supported currently"
                )
                layer.register_forward_pre_hook(forward_pre_hook)

        return model


def pipeline_parallel(model, optimizer, split_spec, mesh=None, dimension=None):
    """
    pipeline_parallel converts model and optimizer to pipelined distributed model

    Args:
        model (paddle.nn.Layer): A single card model to be distributed
        optimizer (paddle.optimizer.Optimizer): An optimizer to be distributed
        split_spec (OrderedDict|dict|str): The pipeline parallel split point.
            if split_spec is a string, such as "llama.layer", Then the layer with same prefix a will be divided equally according to the size of pipeline degree.
            if split_spec is a OrderedDict|dict, key is the layer name, and the value is the split position that can be SplitPoint.BEGINNING or SplitPoint.END, the order of the keys is the order of the pipeline stage.
            NOTE: dict is also ordered after python3.7, so use dict at this time.
        the order of the keys is the order of the pipeline stage
        mesh (ProcessMesh): A ProcessMesh Object.
        dimension (int|str): The mesh dimension to pipeline the model.

    Returns:
        PipelineParallel: a distributed model
        ParallelOptimizer: a distributed optimizer
    """
    if mesh is None:
        mesh = fleet.auto.get_mesh()
        assert (
            mesh is not None
        ), "global mesh must not be None, please call fleet.auto.set_mesh(global_mesh) firstly"
        assert (
            "pp" in mesh.dim_names
        ), "pp must in the mesh dim_names when use pipeline_parallel"
    else:
        assert NotImplementedError(
            "Specifying a custom mesh is not supported currently"
        )

    if isinstance(split_spec, str):
        # match layer_name with split_spec following by a dot and numbers and no other characters
        # such as split_spec = "llama.layer", then llama.layer.0 is matched, llama.layer.0.mlp is not matched
        pattern = rf"{split_spec}\.\d+$"
        matched_layer_name = [
            name
            for name, _ in model.named_sublayers()
            if re.match(pattern, name)
        ]

        pp_size = mesh.get_dim_size("pp")
        layer_num = len(matched_layer_name)
        assert (
            layer_num > 0
        ), "No layer match the split_spec, please check its correctness"
        assert (
            layer_num % pp_size == 0
        ), f"The number of layers must be divisible by the pp size, but got {layer_num} and {pp_size}"
        layers_per_rank = layer_num // pp_size
        split_spec_dict = OrderedDict(
            [
                (
                    f"{split_spec}.{i * layers_per_rank - 1}",
                    SplitPoint.END,
                )
                for i in range(1, pp_size)
            ]
        )
    else:
        split_spec_dict = split_spec

    logger.info(f"split_spec_dict: {split_spec_dict}")

    model = PipelineParallel(model, split_spec_dict)
    if optimizer is not None:
        optimizer = ParallelOptimizer(optimizer)

    return model, optimizer
