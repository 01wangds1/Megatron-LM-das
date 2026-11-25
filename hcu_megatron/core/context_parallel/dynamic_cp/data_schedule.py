# Copyright (c) 2025-2026, NVIDIA CORPORATION. All rights reserved.
# Adapted from Megatron-LM PR #9061.

from itertools import groupby

import torch

from megatron.core import parallel_state as mpu

from .data_schedule_utils import (
    align_sample_id_groups,
    broadcast_scalars,
    build_packed_microbatches,
    create_data_iterator,
    get_global_seqlens_and_ids,
    get_global_real_seqlen_stats,
    get_packed_sequence_alignment,
    get_packed_sequence_length,
    next_dynamic_cp_group_packing_aware,
    reroute_samples_to_dcp_ranks,
)
from .runtime import valid_cp_sizes


def _dynamic_schedule(lengths, pool_size, capacity, min_size, vp_multiple, pad):
    sizes = valid_cp_sizes(pool_size)
    remaining = lengths
    steps = []
    while remaining:
        _, remaining, _, rank_samples = next_dynamic_cp_group_packing_aware(
            remaining, pool_size, capacity, min_cp_size=min_size,
            cp_group_sizes=sizes, pad_sequences=pad,
        )
        # Group width alone identifies the communicator on each rank. Put larger
        # groups first so their starts are multiples of the pre-created width.
        groups = [list(group) for _, group in groupby(rank_samples)]
        groups.sort(key=len, reverse=True)
        steps.append([ids for group in groups for ids in group])
    if vp_multiple > 1:
        steps = align_sample_id_groups(
            steps, vp_multiple, cp_group_sizes=sizes, sample_lengths=dict(lengths),
            max_seq_len_per_rank=capacity, pad_sequences=pad,
        )
    return steps


def _static_schedule(lengths, dp_size, cp_size, capacity, vp_multiple, pad):
    packs = []
    pack_length = 0
    for sample_id, length in lengths:
        length = get_packed_sequence_length(length, cp_size, pad)
        limit = cp_size * capacity
        if length > limit:
            raise ValueError(f"Padded sample of {length} tokens exceeds static CP capacity {limit}")
        if not packs or pack_length + length > limit:
            packs.append([])
            pack_length = 0
        packs[-1].append(sample_id)
        pack_length += length

    # Each DP replica needs equal numbers of forwards, in complete VPP groups.
    missing = -len(packs) % (dp_size * vp_multiple)
    for pack in reversed(packs[:]):
        while missing and len(pack) > 1:
            packs.append([pack.pop()])
            missing -= 1
        if not missing:
            break
    if missing:
        raise ValueError("Too few source samples to align packed microbatches for DP and VPP")
    return [
        [pack for pack in packs[start:start + dp_size] for _ in range(cp_size)]
        for start in range(0, len(packs), dp_size)
    ]


def wrap_data_iterator(data_iterator, config, num_microbatches):
    """Return the packed iterator(s), forward count, logical token sum and squared-length sum."""
    dp = mpu.get_data_parallel_group()
    pool = mpu.get_data_parallel_group(with_context_parallel=True)
    tp = mpu.get_tensor_model_parallel_group()
    pp = mpu.get_pipeline_model_parallel_group()
    device = torch.cuda.current_device()
    dynamic = config.sequence_packing_scheduler == "default_dynamic_cp"
    vpp_size = config.virtual_pipeline_model_parallel_size or 1
    vp_multiple = config.microbatch_group_size_per_vp_stage if vpp_size > 1 else 1
    cp_size = pool.size() // dp.size()
    capacity = config.max_seqlen_per_dp_cp_rank
    alignment, pad = get_packed_sequence_alignment(config, tp.size())
    if pad:
        capacity -= capacity % alignment
        if capacity < alignment:
            raise ValueError(f"Per-rank packing capacity must fit alignment {alignment}")

    vp_needs_data = None
    if vpp_size > 1:
        if data_iterator is not None:
            data_iterator = next((it for it in data_iterator if it is not None), None)
        vp_needs_data = [
            (pp.rank() == 0 and stage == 0)
            or (pp.rank() == pp.size() - 1 and stage == vpp_size - 1)
            for stage in range(vpp_size)
        ]

    fetched = []
    exhausted = torch.zeros(1, dtype=torch.int32, device=device)
    if tp.rank() == 0:
        if data_iterator is None:
            raise ValueError("Packed datasets must be built on TP rank zero of every PP stage")
        try:
            for _ in range(num_microbatches):
                fetched.append(next(data_iterator))
        except StopIteration:
            exhausted.fill_(1)
    for group in (tp, pool, pp):
        if group.size() > 1:
            torch.distributed.all_reduce(exhausted, op=torch.distributed.ReduceOp.MAX, group=group)
    if exhausted.item():
        raise StopIteration

    samples = None
    count = total = squares = None
    if tp.rank() == 0:
        batch = [sample for microbatch in fetched for sample in microbatch]
        lengths, local_ids, offsets = get_global_seqlens_and_ids(batch, dp)
        if dynamic:
            steps = _dynamic_schedule(lengths, pool.size(), capacity,
                                      config.min_dynamic_context_parallel_size, vp_multiple, pad)
        else:
            steps = _static_schedule(lengths, dp.size(), cp_size, capacity, vp_multiple, pad)

        keys = {"original_seq_len", "padded_seq_len"}
        if pp.rank() == 0:
            keys.update(("tokens", "position_ids"))
        if pp.rank() == pp.size() - 1:
            keys.update(("labels", "loss_mask"))
        batch = [{k: sample[k] for k in keys} for sample in batch]
        selected = reroute_samples_to_dcp_ranks(
            batch, local_ids, lengths, steps, offsets, dp, pool, sample_keys=keys,
        )
        samples = build_packed_microbatches(selected, steps, pool.rank(), device, dynamic)
        count = len(steps)
        total, squares = get_global_real_seqlen_stats(batch, dp, device)

    count, total, squares = broadcast_scalars([count, total, squares], tp, device, torch.float64)
    packed = create_data_iterator(samples, tp, config, vp_needs_data, dynamic)
    return packed, int(count), total, squares
