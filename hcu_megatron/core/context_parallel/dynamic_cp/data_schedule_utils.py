# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from math import lcm
from typing import Dict, List, Optional, Sequence, Tuple, TypeVar

import torch

from .runtime import valid_cp_sizes
from megatron.core.rerun_state_machine import RerunDataIterator

_DYNAMIC_CP_WORKLOAD_CAP_DELTA = 0.05

# Dynamic CP rerouting currently owns the text-only GPT/SFT sample schema.
# Multimodal metadata needs explicit element-layout and routing semantics
# before it can be added here.
_REROUTE_KEY_ORDER = (
    "tokens",
    "labels",
    "loss_mask",
    "position_ids",
    "original_seq_len",
    "padded_seq_len",
)
_REROUTE_KEY_SET = frozenset(_REROUTE_KEY_ORDER)
_REROUTE_SCALAR_KEYS = frozenset(("original_seq_len", "padded_seq_len"))
_SequenceLength = TypeVar('_SequenceLength', int, torch.Tensor)


def get_packed_sequence_alignment(config, tp_size: int) -> tuple[int, bool]:
    """Return CP-local alignment and whether post-pack padding is enabled.

    The user-facing alignment is in pre-SP units. MXFP8's quantization
    block constraint applies after SP and must therefore include the TP size.
    """
    requested = getattr(config, 'pad_packed_seq_alignment', None)
    mxfp8 = (
        getattr(config, 'fp8', None) is not None and getattr(config, 'fp8_recipe', None) == 'mxfp8'
    )
    sp_size = tp_size if getattr(config, 'sequence_parallel', False) else 1
    alignment = lcm(2 * sp_size, 32 * sp_size if mxfp8 else 1)
    if requested not in (None, 'max'):
        if int(requested) <= 0:
            raise ValueError("pad_packed_seq_alignment must be positive")
        alignment = lcm(alignment, int(requested))
    return alignment, requested is not None or mxfp8


def get_packed_sequence_length(
    seq_len: _SequenceLength, cp_size: int, pad_sequences: bool
) -> _SequenceLength:
    """Return the physical length used by eager per-sequence zigzag padding."""
    alignment = 2 * cp_size if pad_sequences and cp_size > 1 else 1
    return seq_len + (-seq_len % alignment)


def pad_packed_batch_before_cp_slice(batch: Dict, config, cp_size: int, tp_size: int) -> None:
    """Pad global THD data and metadata before constructing any CP layout.

    Preserve compact logical boundaries and existing physical gaps. Align each
    physical sequence for zigzag CP, then append a masked dummy sequence to
    reach the requested CP-local alignment.
    """
    alignment, enabled = get_packed_sequence_alignment(config, tp_size)
    if not enabled:
        return
    logical = batch['cu_seqlens']
    physical = batch['cu_seqlens_padded']
    lengths = physical.diff()
    padded_lengths = get_packed_sequence_length(lengths, cp_size, pad_sequences=True)
    global_length = int(padded_lengths.sum().item())
    local_length = (global_length + cp_size - 1) // cp_size
    limit = getattr(config, 'max_seqlen_per_dp_cp_rank', None)
    if getattr(config, 'pad_packed_seq_alignment', None) == 'max':
        if limit is None or limit % alignment:
            raise ValueError(
                f"Packed padding target must be a multiple of {alignment}, got {limit}"
            )
        target = limit
    else:
        target = local_length + (-local_length % alignment)
    if target < local_length or (limit is not None and target > limit):
        raise ValueError(
            f"Padded CP-local length {target} (input {local_length}) exceeds "
            f"max_seqlen_per_dp_cp_rank={limit}; increase the limit or reduce packing."
        )

    padded_boundaries = torch.cat((physical[:1], padded_lengths.cumsum(0, dtype=physical.dtype)))
    positions = torch.arange(global_length, device=physical.device)
    seq_ids = torch.searchsorted(padded_boundaries[1:], positions, right=True)
    offsets = positions - padded_boundaries[seq_ids]
    source_indices = physical[seq_ids] + offsets
    within_source = offsets < lengths[seq_ids]
    source_indices = source_indices.clamp(max=int(physical[-1].item()) - 1).long()
    tail = target * cp_size - global_length
    for key in ('tokens', 'labels', 'loss_mask', 'position_ids', 'padding_mask'):
        tensor = batch.get(key)
        if tensor is None:
            continue
        fill = key == 'padding_mask'
        padded = tensor.index_select(0, source_indices).masked_fill(~within_source, fill)
        batch[key] = torch.cat((padded, tensor.new_full((tail,), fill))) if tail else padded

    if tail:
        logical = torch.cat((logical, logical[-1:] + tail))
        padded_boundaries = torch.cat((padded_boundaries, padded_boundaries[-1:] + tail))
    batch['cu_seqlens'] = logical
    batch['cu_seqlens_padded'] = padded_boundaries
    batch['max_seqlen'] = padded_boundaries.diff().max().to(torch.int32)


def get_global_seqlens_and_ids(batch, dp_group):
    """
    Gathers the sequence lengths of all subsamples from all DP ranks and calculates global IDs.
    """
    subsample_seqlens = torch.cat([sample["padded_seq_len"] for sample in batch]).to(
        dtype=torch.int32, device=torch.cuda.current_device(), non_blocking=True
    )
    # Collect the number of subsamples from all ranks
    num_local_subsamples = subsample_seqlens.shape[0]
    local_len = torch.tensor([num_local_subsamples], dtype=torch.int32).cuda()
    dp_subsample_count = [torch.zeros_like(local_len) for _ in range(dp_group.size())]
    torch.distributed.all_gather(dp_subsample_count, local_len, group=dp_group)

    # Find the max number of subsamples across all ranks and pad subsample_seqlens to max length
    dp_subsample_counts = torch.stack(dp_subsample_count, dim=0).cpu().view(-1)
    max_sub_samples = int(dp_subsample_counts.max().item())
    if max_sub_samples == 0:
        raise ValueError("Packed global batch contains no sequences with supervised targets")

    if num_local_subsamples < max_sub_samples:
        subsample_seqlens_padded = torch.cat(
            [
                subsample_seqlens,
                torch.zeros(max_sub_samples - num_local_subsamples, dtype=torch.int32).cuda(),
            ],
            dim=0,
        )
    else:
        subsample_seqlens_padded = subsample_seqlens

    # Gather the subsample_seqlens from all ranks
    seqlens_gathered = [torch.empty_like(subsample_seqlens_padded) for _ in range(dp_group.size())]
    torch.distributed.all_gather(seqlens_gathered, subsample_seqlens_padded, group=dp_group)

    # Trim each seqlens_gathered to the length of the correct sample
    for dp_rank, seqlen in enumerate(seqlens_gathered):
        seqlens_gathered[dp_rank] = seqlen[: dp_subsample_counts[dp_rank]]

    seqlens_gathered = torch.cat(seqlens_gathered, dim=0)
    seqlens_gathered = seqlens_gathered.cpu().tolist()

    # Calculate the offsets to assign unique global ID to each subsample.
    csum = torch.cumsum(dp_subsample_counts, dim=0, dtype=torch.int32)
    offsets = torch.cat([torch.zeros(1, dtype=torch.int32), csum], dim=0)

    # Calculate global ID for each subsample
    dp_rank = dp_group.rank()
    global_ids = torch.arange(len(seqlens_gathered), dtype=torch.int32).cuda()

    # Create a list of (global_id, seqlen) tuples for scheduling
    global_id_seqlens = [(i, seqlens_gathered[i]) for i in range(len(global_ids))]

    # Get the global IDs locally present on this rank
    start_idx = offsets[dp_rank]
    end_idx = offsets[dp_rank + 1]

    global_ids_this_rank = global_ids[start_idx:end_idx]

    return global_id_seqlens, global_ids_this_rank, offsets


def _pack_sequences(
    samples: List,
    padded_lengths: torch.Tensor,
    original_lengths: torch.Tensor,
    local_cp_size: Optional[torch.Tensor],
    dev: torch.device,
) -> Dict[str, torch.Tensor]:
    """Pack multiple samples into a single packed sample."""

    def _pack_tensors(tensors):
        return torch.cat([t.reshape(-1) for t in tensors], dim=0)

    new_sample = {}
    for key in ('tokens', 'labels', 'loss_mask', 'position_ids'):
        if key in samples[0]:
            new_sample[key] = _pack_tensors([sample[key] for sample in samples])

    padded_lengths = padded_lengths.to(device=dev, dtype=torch.int32, non_blocking=True).reshape(-1)
    cu_seqlens_padded = torch.empty(padded_lengths.numel() + 1, device=dev, dtype=torch.int32)
    cu_seqlens_padded[0] = 0
    cu_seqlens_padded[1:] = torch.cumsum(padded_lengths, dim=0)
    max_seqlen = torch.max(padded_lengths).to(dtype=torch.int32)

    new_sample["cu_seqlens_padded"] = cu_seqlens_padded
    new_sample["max_seqlen"] = max_seqlen

    original_lengths = original_lengths.to(
        device=dev, dtype=torch.int32, non_blocking=True
    ).reshape(-1)
    cu_seqlens = torch.empty(original_lengths.numel() + 1, device=dev, dtype=torch.int32)
    cu_seqlens[0] = 0
    cu_seqlens[1:] = torch.cumsum(original_lengths, dim=0).reshape(-1)
    new_sample["cu_seqlens"] = cu_seqlens
    if local_cp_size is not None:
        new_sample["local_cp_size"] = local_cp_size

    return new_sample


def broadcast_tensor(item, src_rank, group) -> None:
    """Broadcast a tensor from src_rank to all ranks in the group."""
    if item is not None:
        torch.distributed.broadcast(item, src_rank, group=group)


def broadcast_scalars(values: List, group, dev, dtype=torch.float32) -> List:
    """
    Broadcast scalar values from rank 0 to all ranks in the group.

    Args:
        values: List of scalar values to broadcast (only used on rank 0).
        group: The process group to broadcast within.
        dev: The device to use for the tensor.
        dtype: The data type for the tensor.

    Returns:
        List of broadcasted values.
    """
    if group.size() <= 1:
        return values

    src_rank = torch.distributed.get_process_group_ranks(group)[0]
    num_values = len(values)

    if group.rank() == 0:
        info_to_broadcast = torch.tensor(values, dtype=dtype, device=dev)
    else:
        info_to_broadcast = torch.zeros(num_values, dtype=dtype, device=dev)

    broadcast_tensor(info_to_broadcast, src_rank, group)

    if group.rank() != 0:
        values = info_to_broadcast.cpu().tolist()

    return values


def create_data_iterator(
    new_samples, tp_group, config, vpp_needs_data=None, is_dynamic_cp: bool = False
):
    """Build packed-data iterators for the local pipeline stage.

    Every virtual stage gets an independent ``RerunDataIterator``. Stages at
    the model boundary, and stages containing MTP, receive the data fields;
    all other virtual stages receive only packed-sequence metadata.
    """
    if (
        config.virtual_pipeline_model_parallel_size is not None
        and config.virtual_pipeline_model_parallel_size > 1
    ):
        vpp_size = config.virtual_pipeline_model_parallel_size
        if tp_group.rank() == 0:
            metadata_keys = ["max_seqlen", "cu_seqlens", "cu_seqlens_padded"]
            if is_dynamic_cp:
                metadata_keys.append("local_cp_size")
            new_data_iterator = []
            for vp_stage in range(vpp_size):
                if vpp_needs_data is not None and vpp_needs_data[vp_stage]:
                    samples = [dict(sample) for sample in new_samples]
                else:
                    # Keep dictionaries independent: CP batch preparation mutates
                    # sequence-valued entries in-place.
                    samples = [
                        {key: sample[key] for key in metadata_keys if key in sample}
                        for sample in new_samples
                    ]
                new_data_iterator.append(RerunDataIterator(iter(samples)))
        else:
            new_data_iterator = [None for _ in range(vpp_size)]
    else:
        new_data_iterator = RerunDataIterator(iter(new_samples)) if tp_group.rank() == 0 else None

    return new_data_iterator


def reroute_samples_to_dcp_ranks(
    batch,
    global_ids_this_rank,
    global_id_seqlens,
    sample_id_groups,
    offsets,
    dp_group,
    dp_cp_group,
    sample_keys=None,
):
    """
    Reroutes the sub-samples to the correct rank after scheduling.

    Each CP lane gathers the samples from its DP group, then keeps only the
    samples assigned to its DPxCP rank. Gathering within ``dp_group`` avoids
    collecting the identical input held by every CP sibling and avoids the
    fully connected P2P transport created by NCCL all-to-all.

    The gather is issued one data key at a time to bound temporary memory to
    one global field. Selected slices are cloned before the next key so the
    full gather buffer can be released.

    This is intentionally a text-only contract. Multimodal fields are rejected
    until their per-token, per-sample, or per-media routing layout is defined.
    """

    dcp_rank = dp_cp_group.rank()
    dp_rank = dp_group.rank()
    dp_size = dp_group.size()

    batch_keys = set(batch[0]) if batch else set(sample_keys or _REROUTE_KEY_ORDER)
    unsupported_keys = batch_keys - _REROUTE_KEY_SET
    assert not unsupported_keys, (
        "Dynamic CP reroute currently supports only the text sample schema; "
        f"cannot reroute unsupported sample keys {sorted(unsupported_keys)}. "
        "extend _REROUTE_KEY_ORDER and classify their element layout."
    )
    for sample_idx, sample in enumerate(batch[1:], start=1):
        sample_keys = set(sample)
        assert sample_keys == batch_keys, (
            f"Sample {sample_idx} keys {sorted(sample_keys)} do not match sample 0 keys "
            f"{sorted(batch_keys)}."
        )
    data_keys = [key for key in _REROUTE_KEY_ORDER if key in batch_keys]

    offset_values = [int(value) for value in offsets.tolist()]
    assert (
        len(offset_values) == dp_size + 1
    ), f"Expected {dp_size + 1} DP offsets, got {len(offset_values)}."
    local_ids = [int(gid) for gid in global_ids_this_rank.tolist()]
    expected_local_ids = list(range(offset_values[dp_rank], offset_values[dp_rank + 1]))
    assert (
        local_ids == expected_local_ids
    ), f"Local sample IDs {local_ids} do not match DP-rank range {expected_local_ids}."
    assert len(batch) == len(
        local_ids
    ), f"Local batch size {len(batch)} does not match sample-ID count {len(local_ids)}."

    recv_ids = sorted(
        {gid for sample_id_group in sample_id_groups for gid in sample_id_group[dcp_rank]}
    )
    recv_samples = {gid: {key: None for key in data_keys} for gid in recv_ids}
    seq_len_by_gid = dict(global_id_seqlens)

    def _build_layout(is_scalar):
        sample_numels = {
            gid: 1 if is_scalar else int(seq_len_by_gid[gid]) for gid in range(offset_values[-1])
        }
        rank_numels = [
            sum(sample_numels[gid] for gid in range(offset_values[rank], offset_values[rank + 1]))
            for rank in range(dp_size)
        ]
        max_rank_numel = max(rank_numels)

        sample_slices = {}
        for source_rank in range(dp_size):
            cursor = source_rank * max_rank_numel
            for gid in range(offset_values[source_rank], offset_values[source_rank + 1]):
                sample_numel = sample_numels[gid]
                sample_slices[gid] = (cursor, sample_numel)
                cursor += sample_numel

        return rank_numels, max_rank_numel, sample_slices

    layouts = {False: _build_layout(is_scalar=False), True: _build_layout(is_scalar=True)}

    for key in data_keys:
        rank_numels, max_rank_numel, sample_slices = layouts[key in _REROUTE_SCALAR_KEYS]

        local_tensor = (
            torch.cat(
                [
                    sample[key]
                    .to(
                        device=torch.cuda.current_device(),
                        non_blocking=True,
                        dtype=torch.int64 if key in _REROUTE_SCALAR_KEYS else sample[key].dtype,
                    )
                    .reshape(-1)
                    for sample in batch
                ],
                dim=0,
            )
            if batch
            else torch.empty(
                0,
                device=torch.cuda.current_device(),
                dtype=torch.float32 if key == 'loss_mask' else torch.int64,
            )
        )
        assert (
            local_tensor.numel() == rank_numels[dp_rank]
        ), f"Packed {key} has {local_tensor.numel()} elements, expected {rank_numels[dp_rank]}."

        if local_tensor.numel() < max_rank_numel:
            gather_input = local_tensor.new_zeros(max_rank_numel)
            gather_input[: local_tensor.numel()].copy_(local_tensor)
        else:
            gather_input = local_tensor.contiguous()

        if dp_size == 1:
            gathered_tensor = gather_input
        else:
            gathered_tensor = local_tensor.new_empty(dp_size * max_rank_numel)
            torch.distributed.all_gather_into_tensor(gathered_tensor, gather_input, group=dp_group)

        for gid in recv_ids:
            start, sample_numel = sample_slices[gid]
            recv_samples[gid][key] = gathered_tensor[start : start + sample_numel].clone()

    return recv_samples


def build_packed_microbatches(
    samples_this_rank_with_id: Dict[int, Dict[str, torch.Tensor]],
    sample_id_groups: List[List[List[int]]],
    dcp_rank: int,
    dev: torch.device,
    is_dynamic_cp: bool = False,
) -> List[Dict[str, torch.Tensor]]:
    """Build this rank's packed samples and attach its runtime CP size."""
    num_micro_batches = len(sample_id_groups)
    seg_starts: List[int] = [0]
    original_lens_tensors = []
    padded_lens_tensors = []

    grouped_samples = [
        [
            samples_this_rank_with_id[sub_sample_id]
            for sub_sample_id in sample_id_groups[i][dcp_rank]
        ]
        for i in range(num_micro_batches)
    ]

    local_cp_sizes = None
    if is_dynamic_cp:
        local_cp_sizes = []
        for i in range(num_micro_batches):
            sample_ids_this_rank = sample_id_groups[i][dcp_rank]
            assert sample_ids_this_rank, "Dynamic CP must assign at least one sample to every rank"
            representative_id = sample_ids_this_rank[0]
            local_cp_sizes.append(
                sum(representative_id in rank_ids for rank_ids in sample_id_groups[i])
            )
        local_cp_sizes = torch.tensor(local_cp_sizes, dtype=torch.int32, device=dev)

    for i in range(num_micro_batches):
        samples = grouped_samples[i]
        seg_starts.append(seg_starts[-1] + len(samples))
        original_lens_tensors.extend([s["original_seq_len"].reshape(-1) for s in samples])
        padded_lens_tensors.extend([s["padded_seq_len"].reshape(-1) for s in samples])

    padded_lens_all_gpu = torch.cat(padded_lens_tensors, dim=0).to(dtype=torch.int32)
    original_lens_all_gpu = torch.cat(original_lens_tensors, dim=0).to(dtype=torch.int32)

    new_samples: List[Dict[str, torch.Tensor]] = []
    for i in range(num_micro_batches):
        samples = grouped_samples[i]
        lens_padded = padded_lens_all_gpu[seg_starts[i] : seg_starts[i + 1]]
        lens_original = original_lens_all_gpu[seg_starts[i] : seg_starts[i + 1]]
        local_cp_size = local_cp_sizes[i] if is_dynamic_cp else None
        new_sample = _pack_sequences(samples, lens_padded, lens_original, local_cp_size, dev)
        new_samples.append(new_sample)

    return new_samples


def get_global_real_seqlen_stats(batch, dp_group, dev) -> Tuple[float, float]:
    """Sum the real (unpadded) sub-sample lengths of the global batch and their squares.

    Args:
        batch: This DP rank's sub-samples from the collated source microbatches.
        dp_group: The DP group whose padded lengths were gathered for scheduling.
        dev: Device for the reduction.

    Returns:
        ``(sum_i L_i, sum_i L_i**2)`` over the real lengths ``L_i`` of every sub-sample in
        ``dp_group``, as expected by ``num_floating_point_operations``.
    """
    seqlens = [sample["original_seq_len"].reshape(-1) for sample in batch]
    seqlens = (torch.cat(seqlens) if seqlens else torch.zeros(0)).to(
        device=dev, dtype=torch.float64
    )
    stats = torch.stack((seqlens.sum(), seqlens.square().sum()))
    if dp_group.size() > 1:
        torch.distributed.all_reduce(stats, group=dp_group)
    seqlen_sum, seqlen_squared_sum = stats.tolist()
    return seqlen_sum, seqlen_squared_sum


# =============================================================================
# Dynamic CP scheduling algorithms (used by DefaultDynamicCPScheduler)
# =============================================================================


def next_dynamic_cp_group_packing_aware(
    sample_seqlens: List[Tuple[int, int]],
    total_gpus: int,
    max_seq_len_per_rank: int,
    min_cp_size: int = 1,
    cp_group_sizes: Optional[Sequence[int]] = None,
    pad_sequences: bool = False,
) -> Tuple[List[List[int]], List[Tuple[int, int]], List[float], List[List[int]]]:
    """Form one DCP microbatch with packing-aware CP group selection.

    This differs from the legacy DCP scheduler in two ways:
    1. Short sequences may use a larger CP group than their minimum required
       CP size when that lowers the critical-path rank workload.
    2. Candidate placements are bounded by ``tall * max_seq_len_per_rank``,
       the per-rank workload upper bound for packing sequences no longer than
       the local tallest sequence in the microbatch.

    The scheduler keeps the legacy invariant that each returned microbatch has
    no empty DPxCP rank after the fill step. For non-power-of-two DPxCP layouts,
    it falls back to the full DPxCP group if power-of-two expansion cannot fill
    every rank.
    """
    valid_group_sizes: List[int] = valid_cp_sizes(total_gpus)
    if cp_group_sizes is None:
        cp_group_sizes = valid_group_sizes
    else:
        cp_group_sizes = sorted(set(cp_group_sizes))
        unsupported_group_sizes = set(cp_group_sizes) - set(valid_group_sizes)
        if unsupported_group_sizes:
            raise ValueError(
                f"Unsupported Dynamic CP group sizes {sorted(unsupported_group_sizes)} for "
                f"DPxCP size {total_gpus}; expected a subset of {valid_group_sizes}"
            )
    if min_cp_size not in cp_group_sizes:
        raise ValueError(
            f"min_cp_size={min_cp_size} is not available; configured Dynamic CP group "
            f"sizes are {list(cp_group_sizes)}"
        )
    enabled_cp_sizes = tuple(size for size in cp_group_sizes if size >= min_cp_size)

    if not sample_seqlens:
        return (
            [[] for _ in range(total_gpus)],
            [],
            [0.0 for _ in range(total_gpus)],
            [[] for _ in range(total_gpus)],
        )

    def cp_min_fn(seq_len: int) -> int:
        for size in enabled_cp_sizes:
            if per_rank_length(seq_len, size) <= max_seq_len_per_rank:
                return size
        raise ValueError(
            f"Sequence length {seq_len}, including CP padding, exceeds the capacity of "
            f"configured CP groups {enabled_cp_sizes} at {max_seq_len_per_rank} tokens per rank"
        )

    def per_rank_length(seq_len: int, cp_size: int) -> float:
        return get_packed_sequence_length(seq_len, cp_size, pad_sequences) / cp_size

    def workload(seq_len: int, cp_size: int) -> float:
        return (seq_len * seq_len) / cp_size

    sample_seqlens = sorted(sample_seqlens, key=lambda x: x[1], reverse=True)
    local_tall = sample_seqlens[0][1]
    cap = float(local_tall) * float(max_seq_len_per_rank) * (1.0 + _DYNAMIC_CP_WORKLOAD_CAP_DELTA)

    micro_batches: List[List[int]] = [[] for _ in range(total_gpus)]
    exec_times: List[float] = [0.0 for _ in range(total_gpus)]
    sample_ids_per_gpu: List[List[int]] = [[] for _ in range(total_gpus)]
    packing_sequence_len: Dict[int, float] = {}

    gpu_group_id: List[Optional[int]] = [None] * total_gpus
    group_members: Dict[int, List[int]] = {}
    group_size: Dict[int, int] = {}
    next_gid = 0

    sample_id, seq_len = sample_seqlens[0]
    cp_size = cp_min_fn(seq_len)
    assert cp_size <= total_gpus, (
        f"Sequence length {seq_len} requires CP size {cp_size}, "
        f"but only {total_gpus} DPxCP ranks are available."
    )
    group_id = next_gid
    next_gid += 1
    members = list(range(cp_size))
    group_members[group_id] = members
    group_size[group_id] = cp_size
    packing_sequence_len[group_id] = per_rank_length(seq_len, cp_size)
    per_gpu_cost = workload(seq_len, cp_size)
    for rank in members:
        gpu_group_id[rank] = group_id
        micro_batches[rank].append(seq_len)
        exec_times[rank] += per_gpu_cost
        sample_ids_per_gpu[rank].append(sample_id)

    leftovers: List[Tuple[int, int]] = []
    for sample_id, seq_len in sample_seqlens[1:]:
        min_needed = cp_min_fn(seq_len)
        best = None

        for cp_size in enabled_cp_sizes:
            if cp_size < min_needed or per_rank_length(seq_len, cp_size) > max_seq_len_per_rank:
                continue
            per_gpu_cost = workload(seq_len, cp_size)

            for group_id, size in list(group_size.items()):
                if size != cp_size:
                    continue
                if (
                    packing_sequence_len.get(group_id, 0) + per_rank_length(seq_len, cp_size)
                    > max_seq_len_per_rank
                ):
                    continue
                members = group_members[group_id]
                member_set = set(members)
                projected_max = max(
                    time + per_gpu_cost if rank in member_set else time
                    for rank, time in enumerate(exec_times)
                )
                if projected_max <= cap and (best is None or projected_max < best[0]):
                    best = (projected_max, cp_size, "add", group_id, None)

            free_ranks = [
                rank
                for rank, assigned_group_id in enumerate(gpu_group_id)
                if assigned_group_id is None
            ]
            if len(free_ranks) >= cp_size:
                chosen_members = sorted(free_ranks, key=lambda rank: exec_times[rank])[:cp_size]
                chosen_set = set(chosen_members)
                projected_max = max(
                    time + per_gpu_cost if rank in chosen_set else time
                    for rank, time in enumerate(exec_times)
                )
                if projected_max <= cap and (best is None or projected_max < best[0]):
                    best = (projected_max, cp_size, "new", None, chosen_members)

        if best is None:
            leftovers.append((sample_id, seq_len))
            continue

        _, selected_cp_size, action, group_id, chosen_members = best
        per_gpu_cost = workload(seq_len, selected_cp_size)
        if action == "add":
            members = group_members[group_id]
            packing_sequence_len[group_id] += per_rank_length(seq_len, selected_cp_size)
            for rank in members:
                micro_batches[rank].append(seq_len)
                exec_times[rank] += per_gpu_cost
                sample_ids_per_gpu[rank].append(sample_id)
        else:
            group_id = next_gid
            next_gid += 1
            group_members[group_id] = chosen_members
            group_size[group_id] = selected_cp_size
            packing_sequence_len[group_id] = per_rank_length(seq_len, selected_cp_size)
            for rank in chosen_members:
                gpu_group_id[rank] = group_id
                micro_batches[rank].append(seq_len)
                exec_times[rank] += per_gpu_cost
                sample_ids_per_gpu[rank].append(sample_id)

    def fill_empty_gpus_once() -> bool:
        nonlocal micro_batches, exec_times, sample_ids_per_gpu

        empty_ranks = [rank for rank, micro_batch in enumerate(micro_batches) if not micro_batch]
        if not empty_ranks:
            return False
        assert all(
            not micro_batches[rank] for rank in range(empty_ranks[0], total_gpus)
        ), "fill_empty_gpus_once assumes empty ranks are contiguous at the tail"

        existing_group_sizes = set(group_size.values())
        if not existing_group_sizes:
            return False
        min_group_size = min(existing_group_sizes)
        current_group_index = enabled_cp_sizes.index(min_group_size)
        if current_group_index + 1 >= len(enabled_cp_sizes):
            return False
        next_group_size = enabled_cp_sizes[current_group_index + 1]

        for group_id, size in list(group_size.items()):
            if size != min_group_size:
                continue

            members = group_members[group_id]
            if (
                sum(
                    per_rank_length(length, next_group_size) for length in micro_batches[members[0]]
                )
                > max_seq_len_per_rank
            ):
                continue
            needed_count = next_group_size - min_group_size
            group_start_rank = members[0]
            group_end_rank = members[-1]
            empty_rank = empty_ranks[0]
            if group_end_rank + 1 > empty_rank or group_end_rank + needed_count >= total_gpus:
                continue

            work_to_push = micro_batches[group_end_rank + 1 : empty_rank]
            exec_times_to_push = exec_times[group_end_rank + 1 : empty_rank]
            sample_ids_to_push = sample_ids_per_gpu[group_end_rank + 1 : empty_rank]
            # Expanding across a gap shifts every intervening group to the
            # right. A sparse configured size family (for example
            # 1, 2, 14) can otherwise choose an expansion that fits by itself
            # but pushes an existing group past the end of the DPxCP domain.
            if group_end_rank + needed_count + len(work_to_push) >= total_gpus:
                continue

            new_micro_batches: List[List[int]] = [[] for _ in range(total_gpus)]
            new_exec_times: List[float] = [0.0 for _ in range(total_gpus)]
            new_sample_ids_per_gpu: List[List[int]] = [[] for _ in range(total_gpus)]

            for rank in range(group_start_rank):
                new_micro_batches[rank] = micro_batches[rank]
                new_exec_times[rank] = exec_times[rank]
                new_sample_ids_per_gpu[rank] = sample_ids_per_gpu[rank]

            for rank in range(group_start_rank, group_end_rank + needed_count + 1):
                new_micro_batches[rank] = list(micro_batches[group_end_rank])
                new_exec_times[rank] = sum(
                    workload(length, next_group_size) for length in micro_batches[group_end_rank]
                )
                new_sample_ids_per_gpu[rank] = list(sample_ids_per_gpu[group_end_rank])

            for idx, work in enumerate(work_to_push):
                target_rank = group_end_rank + needed_count + 1 + idx
                new_micro_batches[target_rank] = work
                new_exec_times[target_rank] = exec_times_to_push[idx]
                new_sample_ids_per_gpu[target_rank] = sample_ids_to_push[idx]

            group_size[group_id] = next_group_size
            group_members[group_id] = list(
                range(group_start_rank, group_end_rank + needed_count + 1)
            )
            for other_group_id in list(group_size.keys()):
                if other_group_id == group_id:
                    continue
                if min(group_members[other_group_id]) > group_end_rank:
                    group_members[other_group_id] = [
                        rank + needed_count for rank in group_members[other_group_id]
                    ]

            micro_batches = new_micro_batches
            exec_times = new_exec_times
            sample_ids_per_gpu = new_sample_ids_per_gpu
            return True

        return False

    def fill_with_full_dpxcp_group() -> None:
        nonlocal micro_batches, exec_times, sample_ids_per_gpu, leftovers

        if total_gpus not in enabled_cp_sizes:
            raise ValueError(
                "Cannot fill DPxCP ranks with configured Dynamic CP group sizes "
                f"{enabled_cp_sizes}; full-domain group {total_gpus} is unavailable"
            )

        selected: List[Tuple[int, int]] = []
        next_leftovers: List[Tuple[int, int]] = []
        packed_sequence_len = 0.0

        for sample_id, seq_len in sample_seqlens:
            per_rank_len = per_rank_length(seq_len, total_gpus)
            if packed_sequence_len + per_rank_len <= max_seq_len_per_rank:
                selected.append((sample_id, seq_len))
                packed_sequence_len += per_rank_len
            else:
                next_leftovers.append((sample_id, seq_len))

        assert selected, (
            "At least one sequence should fit in the full DPxCP group; "
            "try to increase 'max-seqlen-per-dp-cp-rank'."
        )

        selected_ids = [sample_id for sample_id, _ in selected]
        selected_lens = [seq_len for _, seq_len in selected]
        per_rank_work = sum(workload(seq_len, total_gpus) for _, seq_len in selected)

        micro_batches = [list(selected_lens) for _ in range(total_gpus)]
        exec_times = [per_rank_work for _ in range(total_gpus)]
        sample_ids_per_gpu = [list(selected_ids) for _ in range(total_gpus)]
        leftovers = next_leftovers

    while any(not micro_batch for micro_batch in micro_batches):
        if not fill_empty_gpus_once():
            fill_with_full_dpxcp_group()
            break

    return micro_batches, leftovers, exec_times, sample_ids_per_gpu


def align_sample_id_groups(
    sample_id_groups: List,
    microbatch_group_size_per_vp_stage: int,
    *,
    cp_group_sizes: Optional[Sequence[int]] = None,
    sample_lengths: Optional[Dict[int, int]] = None,
    max_seq_len_per_rank: Optional[int] = None,
    pad_sequences: bool = False,
) -> List:
    """Split microbatches for VPP without dropping samples or inventing CP groups.

    Expand only through the configured group family, falling back to the full
    DPxCP group if smaller groups cannot fill a non-power-of-two domain. A full
    domain group can itself be split when it contains multiple packed samples.
    Every loop either adds a microbatch/rank or terminates with an explicit error.
    """
    multiple = int(microbatch_group_size_per_vp_stage)
    if multiple < 1:
        raise ValueError("microbatch_group_size_per_vp_stage must be positive")
    if not sample_id_groups:
        return sample_id_groups
    total_ranks = len(sample_id_groups[0])
    valid_sizes = tuple(
        sorted(cp_group_sizes or valid_cp_sizes(total_ranks))
    )

    def fits(ids, size):
        if sample_lengths is None or max_seq_len_per_rank is None:
            return True
        return (
            sum(get_packed_sequence_length(sample_lengths[sid], size, pad_sequences) for sid in ids)
            <= size * max_seq_len_per_rank
        )

    def fill(groups):
        groups = list(groups)
        assigned = sum(size for _, size in groups)
        while assigned < total_ranks:
            # Expanding the smallest group first preserves power-of-two alignment.
            groups.sort(key=lambda group: group[1], reverse=True)
            for i in range(len(groups) - 1, -1, -1):
                ids, size = groups[i]
                larger_sizes = [candidate for candidate in valid_sizes if candidate > size]
                if not larger_sizes:
                    continue
                next_size = larger_sizes[0]
                if next_size - size <= total_ranks - assigned and fits(ids, next_size):
                    groups[i] = (ids, next_size)
                    assigned += next_size - size
                    break
            else:
                # E.g. two CP2 groups and two empty ranks in a DPxCP=6 domain.
                # CP4 does not exist there, so merge into the supported full group.
                ids = [sid for group_ids, _ in groups for sid in group_ids]
                if total_ranks not in valid_sizes or not fits(ids, total_ranks):
                    return None
                groups = [(ids, total_ranks)]
                assigned = total_ranks
        groups.sort(key=lambda group: group[1], reverse=True)
        return [list(ids) for ids, size in groups for _ in range(size)]

    def split_group(microbatch):
        groups = []
        rank = 0
        while rank < total_ranks:
            ids = microbatch[rank]
            if not ids:
                raise ValueError("VPP alignment requires nonempty input ranks")
            end = rank + 1
            while end < total_ranks and microbatch[end] == ids:
                end += 1
            groups.append((ids, end - rank))
            rank = end

        if len(groups) == 1:
            ids, size = groups[0]
            if len(ids) < 2:
                return None, None
            mid = len(ids) // 2
            return fill([(ids[:mid], size)]), fill([(ids[mid:], size)])

        # Prefer balanced rank counts, but try other boundaries if padding
        # prevents a particular pair of expanded batches from fitting.
        cuts = sorted(
            range(1, len(groups)),
            key=lambda cut: abs(2 * sum(size for _, size in groups[:cut]) - total_ranks),
        )
        for cut in cuts:
            left, right = fill(groups[:cut]), fill(groups[cut:])
            if left is not None and right is not None:
                return left, right
        return None, None

    remainder = (-len(sample_id_groups)) % multiple
    for _ in range(remainder):
        for i in range(len(sample_id_groups) - 1, -1, -1):
            left, right = split_group(sample_id_groups[i])
            if left is not None and right is not None:
                sample_id_groups[i] = left
                sample_id_groups.append(right)
                break
        else:
            raise ValueError(
                "Cannot align packed microbatches for VPP: no microbatch can be split "
                "within the configured CP groups and padded capacity. Increase the "
                "number of input samples or reduce microbatch_group_size_per_vp_stage."
            )
    return sample_id_groups
