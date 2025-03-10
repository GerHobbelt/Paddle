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


import paddle

try:
    import tensorrt as trt
except Exception as e:
    pass
from paddle import pir


def map_dtype(pd_dtype):
    version_list = get_trt_version_list()
    if pd_dtype == "FLOAT32":
        return trt.float32
    elif pd_dtype == "FLOAT16":
        return trt.float16
    elif pd_dtype == "INT32":
        return trt.int32
    elif pd_dtype == "INT8":
        return trt.int8
    elif pd_dtype == "BOOL":
        return trt.bool
    # trt version<10.0 not support int64,so convert int64 to int32
    elif pd_dtype == "INT64":
        return trt.int64 if version_list[0] >= 10 else trt.int32
    # Add other dtype mappings as needed
    else:
        raise TypeError(f"Unsupported dtype: {pd_dtype}")


def run_pir_pass(program, partition_mode=False):
    pm = pir.PassManager(opt_level=4)
    pm.enable_print_statistics()
    paddle.base.libpaddle.pir.infer_symbolic_shape_pass(pm, program)
    passes = [
        {'trt_op_marker_pass': {}},
    ]
    if partition_mode:
        passes = [{'trt_sub_graph_extract_pass': {}}]

    for pass_item in passes:
        for pass_name, pass_attr in pass_item.items():
            pm.add_pass(pass_name, pass_attr)
    pm.run(program)
    return program


def forbid_op_lower_trt(program, disabled_ops):
    if isinstance(disabled_ops, str):
        disabled_ops = [disabled_ops]
    for op in program.global_block().ops:
        if op.name() in disabled_ops:
            op.set_bool_attr("__l_trt__", False)


def enforce_op_lower_trt(program, op_name):
    for op in program.global_block().ops:
        if op.name() == op_name:
            op.set_bool_attr("__l_trt__", True)


def predict_program(program, feed_data, fetch_var_list, scope=None):
    with paddle.pir_utils.IrGuard():
        with paddle.static.program_guard(program):
            place = paddle.CUDAPlace(0)
            executor = paddle.static.Executor(place)
            output = executor.run(
                program,
                feed=feed_data,
                fetch_list=fetch_var_list,
                scope=scope,
            )
            return output


def warmup_shape_infer(program, min_shape_feed, max_shape_feed, scope=None):
    paddle.framework.set_flags({"FLAGS_enable_collect_shape": True})
    with paddle.pir_utils.IrGuard():
        with paddle.static.program_guard(program):
            executor = paddle.static.Executor()
            # Run the program with input_data
            for _ in range(1):
                executor.run(program, feed=min_shape_feed, scope=scope)

            # Run the program with input_data_max_shape (fake max_shape input)
            for _ in range(1):
                executor.run(program, feed=max_shape_feed, scope=scope)

            exe_program, _, _ = (
                executor._executor_cache.get_pir_program_and_executor(
                    program,
                    feed=max_shape_feed,
                    fetch_list=None,
                    feed_var_name='feed',
                    fetch_var_name='fetch',
                    place=paddle.framework._current_expected_place_(),
                    scope=scope,
                    plan=None,
                )
            )
    paddle.framework.set_flags({"FLAGS_enable_collect_shape": False})
    return exe_program


def get_trt_version_list():
    version = trt.__version__
    return list(map(int, version.split('.')))


# Adding marker labels to builtin ops facilitates convert processing, but they ultimately do not enter the TensorRT subgraph.
def mark_buitlin_op(program):
    for op in program.global_block().ops:
        if op.name() == "builtin.split":
            defining_op = op.operands()[0].source().get_defining_op()
            if defining_op is not None:
                if (
                    defining_op.has_attr("__l_trt__")
                    and defining_op.attrs()["__l_trt__"]
                ):
                    op.set_bool_attr("__l_trt__", True)
        if op.name() == "builtin.combine":
            defining_op = op.results()[0].all_used_ops()[0]
            if defining_op is not None:
                if (
                    defining_op.has_attr("__l_trt__")
                    and defining_op.attrs()["__l_trt__"]
                ):
                    op.set_bool_attr("__l_trt__", True)


def weight_to_tensor(network, paddle_value, trt_tensor, use_op_name):
    # the following op needn't cast trt.Weight to ITensor, because the layer need weight as input
    forbid_cast_op = [
        "pd_op.depthwise_conv2d",
        "pd_op.conv2d",
        "pd_op.conv2d_transpose",
        "pd_op.batch_norm",
        "pd_op.batch_norm_",
        "pd_op.layer_norm",
        "pd_op.depthwise_conv2d_transpose",
    ]
    if use_op_name in forbid_cast_op:
        return trt_tensor
    input_shape = paddle_value.shape
    if type(trt_tensor) == trt.Weights:
        return network.add_constant(input_shape, trt_tensor).get_output(0)
    return trt_tensor


def zero_dims_to_one_dims(network, trt_tensor):
    if trt_tensor is None:
        return None
    if type(trt_tensor) == trt.Weights:
        return trt_tensor
    if len(trt_tensor.shape) != 0:
        return trt_tensor
    shuffle_layer = network.add_shuffle(trt_tensor)
    shuffle_layer.reshape_dims = (1,)
    return shuffle_layer.get_output(0)
