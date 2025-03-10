# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved
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


import tensorrt as trt

from paddle.tensorrt.register import converter_registry


@converter_registry.register("pd_op.pool2d", trt_version="8.x")
def pool2d_converter(network, paddle_op, inputs):
    input_tensor = inputs[0]

    input_shape = paddle_op.operands()[0].source().shape
    input_dims = len(input_shape)

    global_pooling = paddle_op.attrs().get("global_pooling", False)
    pool_type = paddle_op.attrs().get("pooling_type")
    strides = paddle_op.attrs().get("strides")
    paddings = paddle_op.attrs().get("paddings")
    exclusive = paddle_op.attrs().get("exclusive", True)
    ceil_mode = paddle_op.attrs().get("ceil_mode", False)
    adaptive = paddle_op.attrs().get("adaptive", False)
    padding_algorithm = paddle_op.attrs().get("padding_algorithm", "EXPLICIT")

    if not paddle_op.attrs().get("kernel_size") and len(inputs) == 2:
        full_int_op = paddle_op.operands()[1].source().get_defining_op()
        if full_int_op.name() == "pd_op.full_int_array":
            kernel_size = full_int_op.attrs().get("value")
        else:
            raise Exception(
                "The defining op of kernel size must be pd_op.full_int_array"
            )
    else:
        kernel_size = paddle_op.attrs().get("kernel_size")

    nv_pool_type = trt.PoolingType.MAX
    reduce_operation = trt.ReduceOperation.MAX
    if pool_type == "max":
        nv_pool_type = trt.PoolingType.MAX
        reduce_operation = trt.ReduceOperation.MAX
    elif pool_type == "avg":
        nv_pool_type = trt.PoolingType.AVERAGE
        reduce_operation = trt.ReduceOperation.AVG

    if global_pooling or adaptive:
        paddings = [0] * len(paddings)

    if padding_algorithm == "VALID":
        paddings = [0] * len(paddings)

    nv_paddings = trt.DimsHW(paddings[0], paddings[1])
    nv_ksize = trt.DimsHW(kernel_size[0], kernel_size[1])
    nv_strides = trt.DimsHW(strides[0], strides[1])

    layer = None
    g_pre_pad = trt.DimsHW(0, 0)
    g_post_pad = trt.DimsHW(0, 0)

    if (
        input_shape[input_dims - 2] > 0
        and input_shape[input_dims - 2] - kernel_size[0] + 2 * paddings[0] < 0
    ):
        g_post_pad.h = strides[0] - 1
    if (
        input_shape[input_dims - 1] > 0
        and input_shape[input_dims - 1] - kernel_size[1] + 2 * paddings[1] < 0
    ):
        g_post_pad.w = strides[1] - 1

    real_paddings = paddings.copy()
    for i in range(2):
        copy_pad = paddings[i]
        real_paddings.insert(2 * i + 1, copy_pad)

    if padding_algorithm == "SAME":
        for i in range(2):
            copy_pad = paddings[2 * i]
            paddings.insert(2 * i + 1, copy_pad)

        for i in range(2):
            out_size = (input_shape[2 + i] + strides[i] - 1) // strides[i]
            pad_sum = max(
                (out_size - 1) * strides[i]
                + kernel_size[i]
                - input_shape[2 + i],
                0,
            )
            pad_0 = pad_sum // 2
            pad_1 = pad_sum - pad_0
            paddings[2 * i] = pad_0
            paddings[2 * i + 1] = pad_1
        real_paddings = paddings.copy()

    paddings = [paddings[i] for i in range(len(paddings)) if i % 2 == 0]

    if padding_algorithm == "VALID":
        read_paddings = [0] * len(real_paddings)

    if not adaptive and not global_pooling and not ceil_mode:
        if padding_algorithm != "SAME" and (
            (g_post_pad.h > 0 and input_shape[input_dims - 2] > 0)
            or (g_post_pad.w > 0 and input_shape[input_dims - 1] > 0)
        ):
            pad_layer = network.add_padding_nd(
                input=input_tensor,
                pre_padding=tuple(g_pre_pad),
                post_padding=tuple(g_post_pad),
            )
            input_tensor = pad_layer.get_output(0)
        pooling_layer = network.add_pooling_nd(
            input=input_tensor, type=nv_pool_type, window_size=nv_ksize
        )
        pooling_layer.stride_nd = nv_strides
        pooling_layer.padding_nd = nv_paddings
        pooling_layer.average_count_excludes_padding = exclusive
        if padding_algorithm == "SAME":
            pooling_layer.padding_mode = trt.PaddingMode.SAME_UPPER

        layer = pooling_layer
    elif not adaptive and not global_pooling and ceil_mode:
        pooling_layer = network.add_pooling_nd(
            input=input_tensor, type=nv_pool_type, window_size=nv_ksize
        )
        pooling_layer.stride_nd = nv_strides
        pooling_layer.padding_nd = nv_paddings
        pooling_layer.average_count_excludes_padding = exclusive
        if padding_algorithm == "SAME":
            pooling_layer.padding_mode = trt.PaddingMode.SAME_UPPER
        else:
            pooling_layer.padding_mode = trt.PaddingMode.EXPLICIT_ROUND_UP
        layer = pooling_layer
    elif global_pooling and not adaptive:
        reduce_axes = (1 << (input_dims - 2)) | (1 << (input_dims - 1))
        reduce_layer = network.add_reduce(
            input=input_tensor,
            op=reduce_operation,
            axes=reduce_axes,
            keep_dims=True,
        )
        layer = reduce_layer
    else:
        raise NotImplementedError(
            "The combination of attributes is not supported yet."
        )

    output_tensor = layer.get_output(0)
    return output_tensor
