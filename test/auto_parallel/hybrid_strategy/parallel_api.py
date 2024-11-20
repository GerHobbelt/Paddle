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
import os
import random
from functools import reduce

import numpy as np
from single_llama_model import LlamaForCausalLM, LlamaPretrainingCriterion

import paddle
import paddle.distributed as dist
from paddle import LazyGuard
from paddle.distributed.auto_parallel.intermediate.parallel_base import (
    parallelize_model_and_optimizer,
)
from paddle.distributed.auto_parallel.intermediate.sharded_data_parallel import (
    sharded_data_parallel,
)
from paddle.distributed.auto_parallel.intermediate.tensor_parallel import (
    ColWiseParallel,
    RowWiseParallel,
    tensor_parallel,
)
from paddle.io import BatchSampler, DataLoader, Dataset


def is_pp_enable():
    global_mesh = dist.auto_parallel.get_mesh()
    return "pp" in global_mesh.dim_names


def get_mesh(pp_idx=None):
    global_mesh = dist.auto_parallel.get_mesh()
    assert global_mesh is not None, "global_mesh is not initialized!"
    if pp_idx is None:
        return global_mesh
    if is_pp_enable():
        mesh = global_mesh.get_mesh_with_dim("pp")[pp_idx]
        return mesh
    else:
        return global_mesh


class Config:
    vocab_size = 32000
    hidden_size = 4096
    intermediate_size = 11008
    seq_length = 2048
    num_hidden_layers = 2
    num_attention_heads = 32
    rms_norm_eps = 1e-6
    sequence_parallel = False
    use_lazy_init = False


class RandomDataset(Dataset):
    def __init__(self, seq_len, num_samples=100):
        super().__init__()
        self.seq_len = seq_len
        self.num_samples = num_samples

    def __getitem__(self, index):
        input = np.random.uniform(size=[self.seq_len]).astype("int64")
        label = (np.random.uniform(size=[self.seq_len]) * 10).astype("int64")
        return input, label

    def __len__(self):
        return self.num_samples


def create_optimizer(model, lr_scheduler):
    decay_parameters = [
        p.name
        for n, p in model.named_parameters()
        if not any(nd in n for nd in ["bias", "norm"])
    ]

    def apply_decay_param_fun(x):
        return x in decay_parameters

    # test global_clip in auto_parallel
    if os.getenv("use_param_group") == "true":
        param_group = {}
        param_group["params"] = list(model.parameters())
        param_group["weight_decay"] = 0.01
        param_group["grad_clip"] = paddle.nn.ClipGradByGlobalNorm(1.0)
        optimizer = paddle.optimizer.adamw.AdamW(
            learning_rate=lr_scheduler,
            apply_decay_param_fun=apply_decay_param_fun,
            parameters=[param_group],
        )
    else:
        optimizer = paddle.optimizer.adamw.AdamW(
            learning_rate=lr_scheduler,
            apply_decay_param_fun=apply_decay_param_fun,
            parameters=model.parameters(),
            weight_decay=0.01,
            grad_clip=paddle.nn.ClipGradByGlobalNorm(1.0),
        )
    return optimizer


class TestParallelAPI:
    def __init__(self):
        self.config = Config()
        self.dp = int(os.getenv("dp"))
        self.mp = int(os.getenv("mp"))
        self.pp = int(os.getenv("pp"))
        if os.getenv("use_sp") == "true":
            self.config.sequence_parallel = True
        if os.getenv("use_lazy_init") == "true":
            self.config.use_lazy_init = True
        self.gradient_accumulation_steps = int(os.getenv("acc_step"))

        self.amp = False
        self.amp_dtype = "float16"
        self.amp_level = "O1"
        self.amp_master_grad = False
        if os.getenv("amp") == "true":
            self.amp = True
        if os.getenv("amp_dtype") in ["float16", "bfloat16"]:
            self.amp_dtype = os.getenv("amp_dtype")
        if os.getenv("amp_level") in ["O0", "O1", "O2"]:
            self.amp_level = os.getenv("amp_level")
        if os.getenv("amp_master_grad") == "true":
            self.amp_master_grad = True
        self.sharding_stage_to_level = [None, "os", "os_g", "p_g_os"]
        self.level = self.sharding_stage_to_level[
            int(os.getenv("sharding_stage", 0))
        ]

        seed = int(os.getenv("seed", 2024))
        np.random.seed(seed)
        random.seed(seed)
        paddle.seed(seed)
        self.init_dist_env()

    def init_dist_env(self):
        mesh_dims = [("dp", self.dp), ("pp", self.pp), ("mp", self.mp)]
        if self.pp * self.mp == 1:
            mesh_dims = [("dp", self.dp)]
        dim_names = [mesh_dim[0] for mesh_dim in mesh_dims]
        mesh_shape = [mesh_dim[1] for mesh_dim in mesh_dims]
        mesh_arr = np.arange(
            0, reduce(lambda x, y: x * y, mesh_shape, 1)
        ).reshape(mesh_shape)
        global_mesh = dist.ProcessMesh(mesh_arr, dim_names)
        dist.auto_parallel.set_mesh(global_mesh)

    def check_mp(self, layer):
        if self.mp == 1:
            return
        for name, sub_layer in layer.named_sublayers():
            if len(sub_layer.sublayers()) == 0:
                if 'q_proj' in name or 'k_proj' in name or 'v_proj' in name:
                    assert sub_layer.weight.placements == [
                        dist.Replicate(),
                        dist.Shard(1),
                    ]
                    assert sub_layer.bias.placements == [
                        dist.Replicate(),
                        dist.Shard(0),
                    ]
                if (
                    'gate_proj' in name
                    or 'up_proj' in name
                    or 'embed_tokens' in name
                    or 'lm_head' in name
                ):
                    assert sub_layer.weight.placements == [
                        dist.Replicate(),
                        dist.Shard(1),
                    ]
                if 'o_proj' in name:
                    assert sub_layer.weight.placements == [
                        dist.Replicate(),
                        dist.Shard(0),
                    ]
                    assert sub_layer.bias.placements is None
                if 'down_proj' in name:
                    assert sub_layer.weight.placements == [
                        dist.Replicate(),
                        dist.Shard(0),
                    ]

    def parallel_model(self, layer, optimizer=None):
        if self.dp > 1:
            layer, optimizer = sharded_data_parallel(
                layer, optimizer, self.level
            )
        if self.mp > 1:
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
            layer, optimizer = tensor_parallel(layer, plan, optimizer)
        layer, optimizer = parallelize_model_and_optimizer(layer, optimizer)
        self.check_mp(layer)
        if optimizer is None:
            return layer
        return layer, optimizer

    def run_llama(self, to_static=0):
        if self.config.use_lazy_init:
            with LazyGuard():
                model = LlamaForCausalLM(self.config)
        else:
            model = LlamaForCausalLM(self.config)

        lr_scheduler = paddle.optimizer.lr.LinearWarmup(
            learning_rate=0.0001, warmup_steps=2, start_lr=0, end_lr=0.0001
        )
        optimizer = create_optimizer(model, lr_scheduler)
        model, optimizer = self.parallel_model(model, optimizer)

        criterion = LlamaPretrainingCriterion(self.config)
        criterion = self.parallel_model(criterion)

        if self.config.use_lazy_init:
            for param in model.parameters():
                assert not param._is_initialized()
                param.initialize()

        if self.amp and not to_static:
            model, optimizer = paddle.amp.decorate(
                models=model,
                optimizers=optimizer,
                level=self.amp_level,
                dtype=self.amp_dtype,
                master_grad=self.amp_master_grad,
            )

        train_dataset = RandomDataset(self.config.seq_length)
        train_sampler = BatchSampler(
            train_dataset,
            batch_size=2,
            shuffle=True,
            drop_last=True,
        )
        train_dataloader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            num_workers=0,
        )

        if self.pp == 1:
            meshes = [get_mesh(0)]
        elif self.pp > 1:
            meshes = [get_mesh(0), get_mesh(-1)]
        else:
            raise ValueError("pp should be greater or equal to 1")

        dist_loader = dist.shard_dataloader(
            dataloader=train_dataloader,
            meshes=meshes,
            shard_dims="dp",
        )

        global_step = 1
        tr_loss = float(0)

        if not to_static:
            model.train()
            scaler = None
            if self.amp and self.amp_dtype == "float16":
                scaler = paddle.amp.GradScaler(init_loss_scaling=1024)
                scaler = dist.shard_scaler(scaler)

            for step, inputs in enumerate(dist_loader()):
                input_ids, labels = inputs
                custom_black_list = [
                    "reduce_sum",
                    "c_softmax_with_cross_entropy",
                ]
                custom_white_list = []
                if self.amp_level == "O2":
                    custom_white_list.extend(
                        ["lookup_table", "lookup_table_v2"]
                    )
                with paddle.amp.auto_cast(
                    self.amp,
                    custom_black_list=set(custom_black_list),
                    custom_white_list=set(custom_white_list),
                    level=self.amp_level,
                    dtype=self.amp_dtype,
                ):
                    logits = model(input_ids)
                    tr_loss_step = criterion(logits, labels)

                if self.gradient_accumulation_steps > 1:
                    tr_loss_step /= self.gradient_accumulation_steps
                if scaler is not None:
                    scaler.scale(tr_loss_step).backward()
                else:
                    tr_loss_step.backward()
                tr_loss += tr_loss_step

                if global_step % self.gradient_accumulation_steps == 0:
                    logging.info(
                        f"step: {global_step // self.gradient_accumulation_steps}  loss: {tr_loss.numpy()}"
                    )
                    if scaler is not None:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.clear_grad()
                    lr_scheduler.step()
                    tr_loss = 0

                global_step += 1
                if global_step // self.gradient_accumulation_steps >= 10:
                    break
        else:
            strategy = dist.Strategy()
            if self.gradient_accumulation_steps > 1:
                strategy.pipeline.accumulate_steps = (
                    self.gradient_accumulation_steps
                )

            if self.amp:
                amp = strategy.amp
                amp.enable = self.amp
                amp.dtype = self.amp_dtype
                amp.level = self.amp_level.lower()
                if self.amp_master_grad:
                    amp.use_master_grad = True

            dist_model = dist.to_static(
                model,
                dist_loader,
                criterion,
                optimizer,
                strategy=strategy,
            )

            dist_model.train()
            for step, inputs in enumerate(dist_loader()):
                input_ids, labels = inputs
                loss = dist_model(input_ids, labels)
                logging.info(step, loss)

                if step >= 10:
                    break

    def run_test_cases(self):
        self.run_llama(to_static=0)
        # self.run_llama(to_static=1)


if __name__ == '__main__':
    TestParallelAPI().run_test_cases()
