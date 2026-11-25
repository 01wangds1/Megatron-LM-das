# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import os
import sys


def add_arguments(parser):
    group = parser.add_argument_group("Dynamic context parallelism")
    group.add_argument("--dynamic-context-parallel", action="store_true",
                       help="Schedule the CP size of each packed microbatch over DP x CP.")
    group.add_argument("--min-dynamic-context-parallel-size", type=int, default=1)
    group.add_argument("--sequence-packing-scheduler", choices=("default_dynamic_cp", "dp_balanced"))
    group.add_argument("--use-varlen-dataset", action="store_true")
    group.add_argument("--varlen-mock-dataset-config-json")
    group.add_argument("--sft-mock-dataset-config-json")
    group.add_argument("--pad-packed-seq-alignment", metavar="N|max",
                       help="Align the CP-local packed length or pad to the per-rank capacity.")
    return parser


def configure_packing(args):
    """Resolve the opt-in after Megatron has calculated the DP and eval sizes."""
    env = os.environ.get("HCU_DYNAMIC_CONTEXT_PARALLEL")
    if env not in (None, "0", "1"):
        raise ValueError("HCU_DYNAMIC_CONTEXT_PARALLEL must be 0 or 1")
    if env == "0" and args.dynamic_context_parallel:
        raise ValueError("--dynamic-context-parallel conflicts with HCU_DYNAMIC_CONTEXT_PARALLEL=0")
    args.dynamic_context_parallel |= env == "1"
    if args.sequence_packing_scheduler is None:
        if args.dynamic_context_parallel:
            args.sequence_packing_scheduler = "default_dynamic_cp"
        elif args.use_varlen_dataset:
            args.sequence_packing_scheduler = "dp_balanced"
        else:
            return False
    if (args.sequence_packing_scheduler == "default_dynamic_cp") != args.dynamic_context_parallel:
        raise ValueError("default_dynamic_cp requires dynamic CP and dp_balanced requires static CP")
    if args.hybrid_context_parallel:
        raise ValueError("Select only one of --hybrid-context-parallel and sequence packing")
    if args.sft == args.use_varlen_dataset:
        raise ValueError("Packed training requires exactly one of --use-varlen-dataset and --sft")
    if args.dataloader_type != "single" or not args.calculate_per_token_loss:
        raise ValueError("Packed training requires --dataloader-type single and --calculate-per-token-loss")
    if args.schedule_method != "vanilla" or args.pipe_sp_splits != 1:
        raise ValueError("Packed microbatches require the vanilla Megatron schedule")
    backend = getattr(args.attention_backend, "name", args.attention_backend)
    if (args.cuda_graph_impl != "none" or args.transformer_impl != "transformer_engine"
            or backend == "local"):
        raise ValueError("Packed training requires eager Transformer Engine THD attention")
    if args.gtp_weight_remat_size > 1 or args.expert_gtp_weight_remat_size > 1:
        raise ValueError("Packed training requires the data sampler and CP pool to use the same DP grid; "
                         "GTP weight rematerialization is unsupported")

    # These v0.19.2 consumers use fixed-size batches, a static CP group, or a
    # different gradient/update schedule. Catch them before group creation.
    incompatible = (
        "use_megatron_fsdp", "use_torch_fsdp2", "enable_vocab_parallel",
        "enable_dynamic_grad_comp", "overlap_moe_expert_parallel_comm",
        "experimental_attention_variant", "mtp_num_layers",
        "num_microbatches_with_partial_activation_checkpoints", "full_validation",
        "fused_single_qkv_rope", "fim_data", "reset_position_ids", "reset_attention_mask",
        "perform_rl_step",
    )
    if args.dynamic_context_parallel:
        incompatible += ("hierarchical_context_parallel_sizes", "multi_latent_attention",
                         "num_experts", "deterministic_mode")
    selected = [name for name in incompatible if getattr(args, name, None)]
    if selected:
        raise ValueError("Unsupported with packed training: " +
                         ", ".join("--" + name.replace("_", "-") for name in selected))
    pool = args.context_parallel_size
    if args.dynamic_context_parallel:
        from .runtime import valid_cp_sizes

        pool *= args.data_parallel_size
        sizes = valid_cp_sizes(pool)
        if args.min_dynamic_context_parallel_size not in sizes:
            raise ValueError(f"--min-dynamic-context-parallel-size must be one of {sizes}")
    capacity = args.max_seqlen_per_dp_cp_rank
    if capacity is None or capacity <= 0 or capacity * pool < args.seq_length:
        raise ValueError(f"Set --max-seqlen-per-dp-cp-rank so {pool} ranks can fit --seq-length")
    if args.pad_packed_seq_alignment not in (None, "max"):
        args.pad_packed_seq_alignment = int(args.pad_packed_seq_alignment)
        if args.pad_packed_seq_alignment < 1:
            raise ValueError("--pad-packed-seq-alignment must be positive or max")
    if args.eval_iters == 0 and not args.eval_interval:
        # v0.19.2 divides by this interval when sizing datasets and restoring
        # sample counts, even when evaluation is disabled.
        args.eval_interval = sys.maxsize
    # Document boundaries in cu_seqlens supply the causal mask to TE.
    args.create_attention_mask_in_dataloader = False
    args.variable_seq_lengths = True
    return True
