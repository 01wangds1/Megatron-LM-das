# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from functools import partial

import torch
from megatron.core import parallel_state as mpu
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.utils import get_attr_wrapped_model
from pretrain_gpt import get_timers, loss_func, stimer

from .data_schedule_utils import broadcast_scalars, pad_packed_batch_before_cp_slice
from .runtime import get_cp_group


def _prepare_shard(batch, keys, config, cp_group, tp_size):
    """Zero out physical padding, then slice each document in the runtime CP group."""
    logical = batch["cu_seqlens"]
    physical = batch["cu_seqlens_padded"]
    positions = torch.arange(int(physical[-1]), dtype=torch.int32, device=physical.device)
    docs = torch.searchsorted(physical[1:].contiguous(), positions, right=True)
    padding = positions >= (physical[:-1] + logical.diff())[docs]
    batch["padding_mask"] = padding
    for key in keys:
        if key != "padding_mask":
            batch[key] = batch[key].masked_fill(padding, 0)
    pad_packed_batch_before_cp_slice(batch, config, cp_group.size(), tp_size)
    if cp_group.size() > 1:
        import transformer_engine_torch as tex

        physical = batch["cu_seqlens_padded"]
        index = tex.thd_get_partitioned_indices(
            physical, batch["padding_mask"].numel(), cp_group.size(), cp_group.rank()
        )
        for key in keys:
            batch[key] = batch[key].index_select(0, index)


def get_batch(data_iterator, config, vp_stage):
    tp = mpu.get_tensor_model_parallel_group()
    source = tp.rank() == 0
    device = torch.cuda.current_device()
    fields = {"padding_mask": torch.bool}
    if mpu.is_pipeline_first_stage(ignore_virtual=False, vp_stage=vp_stage):
        fields.update(tokens=torch.long, position_ids=torch.long)
    if mpu.is_pipeline_last_stage(ignore_virtual=False, vp_stage=vp_stage):
        fields.update(labels=torch.long, loss_mask=torch.float32)

    dynamic = config.dynamic_context_parallel
    header = [0] * 4
    batch = {}
    if source:
        # RerunDataIterator can replay this dict. Only modify a per-call view.
        batch = dict(next(data_iterator))
        group = get_cp_group(int(batch["local_cp_size"])) if dynamic else mpu.get_context_parallel_group()
        _prepare_shard(batch, fields, config, group, tp.size())
        header = [group.size(), batch["padding_mask"].numel(), batch["cu_seqlens"].numel(),
                  int(batch["max_seqlen"])]
    cp_size, shard_length, n_offsets, max_seqlen = broadcast_scalars(
        header, tp, device, dtype=torch.int64
    )

    src = mpu.get_tensor_model_parallel_src_rank()
    for key, dtype in {**fields, "cu_seqlens": torch.int32, "cu_seqlens_padded": torch.int32}.items():
        length = shard_length if key in fields else n_offsets
        if not source:
            batch[key] = torch.empty(length, device=device, dtype=dtype)
        if tp.size() > 1:
            torch.distributed.broadcast(batch[key], src, group=tp)
        if key in fields:
            batch[key] = batch[key].view(1, shard_length)

    logical, physical = batch["cu_seqlens"], batch["cu_seqlens_padded"]
    params = PackedSeqParams(
        qkv_format="thd", cu_seqlens_q=logical, cu_seqlens_kv=logical,
        cu_seqlens_q_padded=physical, cu_seqlens_kv_padded=physical,
        max_seqlen_q=max_seqlen, max_seqlen_kv=max_seqlen,
        local_cp_size=cp_size if dynamic else None,
        cp_group=get_cp_group(cp_size) if dynamic else None,
        pad_between_seqs=not torch.equal(logical, physical),
    )
    return batch, params


def forward_step(data_iterator, model, return_schedule_plan=False, microbatch_id=None):
    """Use the target model and loss with the selected microbatch's layout."""
    timer = get_timers()("batch-generator", log_level=2)
    timer.start()
    with stimer(bdata=True):
        batch, params = get_batch(
            data_iterator, get_attr_wrapped_model(model, "config"), get_attr_wrapped_model(model, "vp_stage")
        )
    timer.stop()
    with stimer:
        output = model(
            batch.get("tokens"), batch.get("position_ids"), None,
            labels=batch.get("labels"), loss_mask=batch.get("loss_mask"),
            packed_seq_params=params, padding_mask=batch["padding_mask"],
        )
    return output, partial(loss_func, batch.get("loss_mask"), model=model)
