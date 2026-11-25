# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from copy import copy
from datetime import timedelta
from functools import wraps
from inspect import signature
from weakref import WeakValueDictionary

from .causal_padding import causal_padding_forward_wrapper


# Megatron owns the communicators in its parallel-state group list. Keep only
# lookup references so an in-process restart can tear down the old generation.
_groups = WeakValueDictionary()
_installed = None


def valid_cp_sizes(pool_size):
    """Every power-of-two divisor of the DP x CP domain plus the full domain."""
    if pool_size < 1:
        raise ValueError("Dynamic CP requires a nonempty DP x CP domain")
    sizes = []
    size = 1
    while size < pool_size and pool_size % size == 0:
        sizes.append(size)
        size *= 2
    return sizes + [pool_size]


def get_cp_group(size):
    """Return this rank's pre-created group, including the real CP=1 group."""
    if size not in _groups:
        raise ValueError(f"No runtime CP group of size {size}; available sizes: {sorted(_groups)}")
    return _groups[size]


def _initialize_groups(original):
    import torch
    from hcu_megatron.core.parallel_state import get_parallel_group_ranks
    from megatron.core import parallel_state
    from megatron.training import get_args

    @wraps(original)
    def initialize(*args, **kwargs):
        # DAS records every decoder DP x CP lane, including those on other PP
        # stages and TP ranks. Use the rank order from this initialization.
        lanes = get_parallel_group_ranks()["DATA_PARALLEL_GROUP_WITH_CP"]
        start = len(lanes)
        _groups.clear()
        result = original(*args, **kwargs)
        pool = parallel_state.get_data_parallel_group(
            with_context_parallel=True, with_gtp_remat=False
        )
        pool_size = pool.size()
        sizes = valid_cp_sizes(pool_size)
        min_size = get_args().min_dynamic_context_parallel_size
        if min_size not in sizes:
            raise ValueError(f"Minimum dynamic CP size {min_size} must be one of {sizes}")
        if not lanes[start:]:
            raise RuntimeError("DAS did not record any decoder DP x CP groups")
        rank = torch.distributed.get_rank()
        timeout = timedelta(minutes=get_args().distributed_timeout_minutes)
        local_groups = {pool_size: pool}
        for ranks in lanes[start:]:
            for size in sizes:
                if size < min_size or size == pool_size:
                    continue
                for offset in range(0, pool_size, size):
                    members = ranks[offset : offset + size]
                    group = parallel_state.create_group(
                        members, timeout=timeout, group_desc=f"DYNAMIC_CONTEXT_GROUP_{size}"
                    )
                    if rank in members:
                        local_groups[size] = group
        # RCCL/NCCL creates communicators lazily. Initialize them in the same
        # order on every lane before ranks can run different microbatch plans.
        for size in sizes:
            if size >= min_size:
                torch.distributed.barrier(
                    group=local_groups[size], device_ids=[torch.cuda.current_device()]
                )
                torch.cuda.synchronize()
        _groups.update(local_groups)
        return result

    return initialize


def _attention_with_runtime_rope(original):
    parameters = signature(original)

    @wraps(original)
    def forward(self, *args, **kwargs):
        if "packed_seq_params" in kwargs:
            params = kwargs["packed_seq_params"]
        else:
            params = parameters.bind(self, *args, **kwargs).arguments.get("packed_seq_params")
        if params is None or params.local_cp_size is None:
            return original(self, *args, **kwargs)
        # DAS RoPE reads self.pg_collection.cp. Give only this attention layer
        # a view for this call; parameter-gradient reductions keep their groups.
        previous = self.pg_collection
        groups = copy(previous)
        groups.cp = params.cp_group
        self.pg_collection = groups
        try:
            return original(self, *args, **kwargs)
        finally:
            self.pg_collection = previous

    return forward


def _te_with_runtime_group(original, attention_class):
    import torch

    @wraps(original)
    def forward(
        self, query, key, value, attention_mask, attn_mask_type,
        attention_bias=None, packed_seq_params=None, num_splits=None,
    ):
        if packed_seq_params is None or packed_seq_params.local_cp_size is None:
            return original(
                self, query, key, value, attention_mask, attn_mask_type,
                attention_bias, packed_seq_params, num_splits,
            )
        group = packed_seq_params.cp_group
        if group is None or group.size() != packed_seq_params.local_cp_size:
            raise ValueError("PackedSeqParams must carry the scheduled runtime CP group")
        prior_group = self.cp_group
        prior_ranks = self.cp_global_ranks
        prior_stream = self.cp_stream
        if group.size() == 1:
            # Keep a real singleton for RoPE, but TE v0.19.2 expects None to
            # select the CP-off backend.
            packed_seq_params = copy(packed_seq_params)
            packed_seq_params.cp_group = None
        elif attention_class.cp_stream is None:
            # Constructor-time CP can be 1 even when runtime CP is greater.
            attention_class.cp_stream = torch.cuda.Stream()
        try:
            return original(
                self, query, key, value, attention_mask, attn_mask_type,
                attention_bias, packed_seq_params, num_splits,
            )
        finally:
            self.set_context_parallel_group(prior_group, prior_ranks, prior_stream, self.cp_comm_type)

    return forward


def install_runtime(dynamic):
    """Install after the DAS adaptor has registered its v0.19.2 model patches."""
    global _installed
    if _installed is not None:
        if _installed != dynamic:
            raise RuntimeError("Restart the process to change dynamic CP mode")
        return

    from megatron.core import parallel_state
    from megatron.core.extensions.transformer_engine import TEDotProductAttention
    from megatron.core.transformer.attention import Attention

    te_forward = causal_padding_forward_wrapper(TEDotProductAttention.forward)
    if dynamic:
        if parallel_state.model_parallel_is_initialized():
            raise RuntimeError("Install dynamic CP before model-parallel initialization")
        parallel_state.initialize_model_parallel = _initialize_groups(
            parallel_state.initialize_model_parallel
        )
        Attention.forward = _attention_with_runtime_rope(Attention.forward)
        te_forward = _te_with_runtime_group(te_forward, TEDotProductAttention)
    TEDotProductAttention.forward = te_forward
    _installed = dynamic
