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

import logging

import paddle
from paddle.base import core

OpRole = core.op_proto_and_checker_maker.OpRole

from paddle.autograd import backward_utils

from ..auto_parallel.static.utils import (
    get_logger,
)
from .pass_base import PassBase, register_pass

logger = get_logger(logging.INFO)


@register_pass("auto_parallel_recompute_pir")
class AutoParallelRecomputePIRPass(PassBase):
    def __init__(self):
        super().__init__()

    def _check_self(self):
        return True

    def _check_conflict(self, other_pass):
        return True

    def get_fwd_bwd_ops(self, program):
        fwd_ops = []
        bwd_ops = []
        for op in program.global_block().ops:
            if op.op_role == int(OpRole.Forward):
                fwd_ops.append(op)
            elif op.op_role == int(OpRole.Backward):
                bwd_ops.append(op)
        assert len(fwd_ops) and len(bwd_ops)
        return fwd_ops, bwd_ops

    def get_first_bwd_used_op(self, fwd_op, bwd_ops):
        # Find the first user op of the op result in backward op list.
        first_op = bwd_ops[-1]
        for res in fwd_op.results():
            for user_op in res.all_used_ops():
                if user_op in bwd_ops and first_op.id() >= user_op.id():
                    first_op = user_op
        return first_op

    def is_seed_used_by_dropout(self, seed_op):
        # Ensure that the random operator has the same output in backward recompute.
        if seed_op.name() != "seed":
            return False
        seed_value = seed_op.results()[0]
        dropout_ops = ["pd_op.dropout", "pd_op.fused_dropout_add"]
        return any(
            True
            for used_op in seed_value.all_used_ops()
            if used_op.name() in dropout_ops
        )

    def get_segments(self, program):
        # `fwd_recompute_id` indicates the ID assigned to the segment for
        # which the OP requires recompute.
        # A segment comprises all OPs within a program, ranging from the OP
        # with the minimum index to the OP with the maximum index, and all
        # these operations share the same `fwd_recompute_id`.
        segment_beg = {}
        segment_end = {}
        max_op_id = len(program.global_block().ops)
        for idx, op in enumerate(program.global_block().ops):
            if not op.has_attr("fwd_recompute_id"):
                continue
            rc_id = op.attrs()["fwd_recompute_id"]
            if rc_id not in segment_beg:
                segment_beg[rc_id] = max_op_id
                segment_end[rc_id] = 0
            segment_beg[rc_id] = min(segment_beg[rc_id], idx)
            segment_end[rc_id] = max(segment_end[rc_id], idx)

        segments = {}
        idx = 0
        assert len(segment_beg.keys()) == len(segment_end.keys())
        for segment_id, beg_id in segment_beg.items():
            assert segment_id in segment_end.keys()
            end_id = segment_end[segment_id]
            assert beg_id <= end_id
            segment = []
            for p_id in range(beg_id, end_id - 1):
                segment.append(p_id)
            segments[idx] = segment
            idx += 1
        return segments

    def _apply_single_impl(self, main_program, startup_program, context=None):
        segments = self.get_segments(main_program)
        if len(segments) == 0:
            logger.info("No segments found in PIR recompite pass.")
            return

        fwd_ops, bwd_ops = self.get_fwd_bwd_ops(main_program)

        input_value = main_program.list_vars()
        value_map = paddle.pir.IrMapping()
        for val in input_value:
            value_map.add(val, val)

        for rc_id, segment in segments.items():
            first_bwd_used_op = bwd_ops[-1]
            for idx in segment:
                op = main_program.global_block().ops[idx]
                bwd_used_op = self.get_first_bwd_used_op(op, bwd_ops)
                if first_bwd_used_op.id() > bwd_used_op.id():
                    first_bwd_used_op = bwd_used_op

            ori_segment_outputs = backward_utils.ValueSet()
            paddle.pir.set_insertion_point(first_bwd_used_op)

            for idx in segment:
                op = main_program.global_block().ops[idx]
                ori_segment_outputs.update(op.results())

                if self.is_seed_used_by_dropout(op):
                    continue

                rc_op = op.clone(
                    value_map, paddle.pir.CloneOptions(False, True, True)
                )
                rc_op.set_int_attr("bwd_recompute_id", rc_id)

                if first_bwd_used_op.has_attr('op_role'):
                    rc_op.set_int_attr("op_role", first_bwd_used_op.op_role)

                if first_bwd_used_op.has_attr('chunk_id'):
                    rc_op.set_int_attr("chunk_id", first_bwd_used_op.chunk_id)

            for ori_value in ori_segment_outputs:
                rc_value = value_map.look_up(ori_value)
                ori_value.replace_grad_users_with(rc_value, set(bwd_ops))
