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

from .parallel_base import parallelize_model_and_optimizer
from .pipeline_parallel import pipeline_parallel
from .sharded_data_parallel import sharded_data_parallel
from .tensor_parallel import tensor_parallel


def parallelize(
    model, optimizer, mesh=None, dp_config=None, mp_config=None, pp_config=None
):
    # TODO(yaliu): global mesh and split axis support
    if pp_config is not None:
        assert isinstance(pp_config, dict)
        model, optimizer = pipeline_parallel(
            model, optimizer, pp_config.get('split_spec')
        )
    if mp_config is not None:
        assert isinstance(mp_config, dict)
        model, optimizer = tensor_parallel(
            model, optimizer, mp_config.get('parallelize_plan')
        )
    if dp_config is not None:
        assert isinstance(dp_config, dict)
        model, optimizer = sharded_data_parallel(
            model,
            optimizer,
            level=dp_config.get('sharding_level'),
            offload=bool(dp_config.get('offload')),
            exclude_layer=dp_config.get('exclude_layer'),
        )
    model, optimizer = parallelize_model_and_optimizer(model, optimizer)
    return model, optimizer
