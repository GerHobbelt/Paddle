# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import paddle
import paddle.distributed as dist
from paddle.nn import Layer

if TYPE_CHECKING:
    from paddle.distributed import Placement
    from paddle.distributed.auto_parallel.process_mesh import ProcessMesh


class LocalLayer(Layer):
    """
    The `LocalLayer` class is a specialized `Layer` for managing distributed tensors during
    forward and backward passes in a parallelized training environment. It converts distributed tensors
    to local tensors for computation and then back to distributed tensors as output, ensuring seamless
    integration with distributed parallelism frameworks.

    Args:
        out_dist_attrs (list[tuple[ProcessMesh, list[Placement]]]):
            A list where each entry is a tuple containing the `ProcessMesh` and the list of `Placement`
            attributes for the corresponding output tensors. These attributes define the distribution
            strategy for the outputs.

    Examples:
        .. code-block:: python

            import paddle
            import paddle.distributed as dist
            from paddle import nn

            class CustomLayer(LocalLayer):
                def __init__(self, mesh):
                    super().__init__(
                        out_dist_attrs=[(mesh, [dist.Partial(dist.ReduceType.kRedSum)])]
                    )
                    self.fc = nn.Linear(16, 8)

                def forward(self, x):
                    return self.fc(x)

            # doctest: +REQUIRES(env:DISTRIBUTED)
            mesh = dist.ProcessMesh([0, 1], dim_names=["x"])
            custom_layer = CustomLayer(mesh)
            input_tensor = dist.auto_parallel.api.dtensor_from_local(
                paddle.randn([4, 16]), mesh, [dist.Replicate()]
            )

            output_tensor = custom_layer(input_tensor)
            print(output_tensor)
    """

    def __init__(
        self, out_dist_attrs: list[tuple[ProcessMesh, list[Placement]]]
    ):
        super().__init__()
        self.out_dist_attrs = out_dist_attrs

    def __call__(self, *inputs: Any, **kwargs: Any) -> Any:
        """
        Overrides the base `Layer`'s `__call__` method. Transforms distributed tensors to local tensors
        before computation, invokes the parent class's `__call__` method, and then transforms the
        outputs back to distributed tensors based on the specified distribution attributes.
        """
        inputs = list(inputs)
        for idx in range(len(inputs)):
            if inputs[idx].is_dist():
                inputs[idx] = dist.auto_parallel.api.dtensor_to_local(
                    inputs[idx]
                )
        outputs = Layer.__call__(self, *inputs, **kwargs)
        list_outs = paddle.utils.flatten(outputs)
        for idx in range(len(list_outs)):
            list_outs[idx] = dist.auto_parallel.api.dtensor_from_local(
                list_outs[idx],
                self.out_dist_attrs[idx][0],
                self.out_dist_attrs[idx][1],
            )
        return paddle.utils.pack_sequence_as(outputs, list_outs)
