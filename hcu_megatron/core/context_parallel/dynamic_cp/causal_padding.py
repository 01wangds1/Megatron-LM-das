# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from copy import copy
from functools import wraps


def causal_padding_forward_wrapper(original):
    import torch

    if not torch.version.hip:
        return original
    try:
        from transformer_engine.pytorch.quantization import FP8GlobalStateManager
    except ImportError:
        try:
            from transformer_engine.pytorch.fp8 import FP8GlobalStateManager
        except ImportError:
            return original

    def can_use_physical_lengths(layer, query, key, value, mask, mask_type, bias, params):
        if (
            params is None
            or params.qkv_format != "thd"
            or params.pad_between_seqs is not True
            or getattr(mask_type, "name", None) not in ("causal", "padding_causal")
            or mask is not None
            or bias is not None
            or getattr(layer, "attention_type", None) != "self"
            or getattr(layer, "attention_dropout", None) != 0
            or getattr(layer, "softmax_type", None) != "vanilla"
            or getattr(layer, "window_size", None) not in (None, (-1, 0))
            or getattr(layer, "cp_comm_type", None) not in (None, "p2p")
            or getattr(layer, "return_max_logit", False)
        ):
            return False
        config = layer.config
        if any(
            getattr(config, option, None)
            for option in (
                "fp8", "fp4", "enable_cuda_graph", "qk_clip", "log_max_attention_logit",
                "num_moe_experts", "mtp_num_layers", "window_size", "attn_logit_softcapping",
                "multi_latent_attention",
            )
        ) or getattr(config, "cuda_graph_impl", "none") not in (None, "none"):
            return False
        if any(
            type(tensor) is not torch.Tensor
            or tensor.dtype != torch.bfloat16
            or not tensor.is_cuda
            or tensor.ndim != 3
            for tensor in (query, key, value)
        ):
            return False
        if (
            query.device != key.device
            or query.device != value.device
            or query.shape[0] != key.shape[0]
            or query.shape[0] != value.shape[0]
        ):
            return False
        if (
            FP8GlobalStateManager.is_fp8_enabled()
            or FP8GlobalStateManager.is_fp8_calibration()
            or torch.cuda.is_current_stream_capturing()
        ):
            return False
        logical = params.cu_seqlens_q
        physical = params.cu_seqlens_q_padded
        if (
            logical is not params.cu_seqlens_kv
            or physical is not params.cu_seqlens_kv_padded
            or logical is physical
        ):
            return False
        if any(
            type(offsets) is not torch.Tensor
            or offsets.dtype != torch.int32
            or offsets.device != query.device
            or offsets.ndim != 1
            or not offsets.is_contiguous()
            for offsets in (logical, physical)
        ):
            return False
        return logical.shape == physical.shape and logical.numel() >= 2

    @wraps(original)
    def forward(
        self, query, key, value, attention_mask, attn_mask_type,
        attention_bias=None, packed_seq_params=None, num_splits=None,
    ):
        if can_use_physical_lengths(
            self, query, key, value, attention_mask, attn_mask_type,
            attention_bias, packed_seq_params,
        ):
            packed_seq_params = copy(packed_seq_params)
            packed_seq_params.cu_seqlens_q = packed_seq_params.cu_seqlens_q_padded
            packed_seq_params.cu_seqlens_kv = packed_seq_params.cu_seqlens_kv_padded
            packed_seq_params.pad_between_seqs = False
        return original(
            self, query, key, value, attention_mask, attn_mask_type,
            attention_bias=attention_bias, packed_seq_params=packed_seq_params,
            num_splits=num_splits,
        )

    return forward
