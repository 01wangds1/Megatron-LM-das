# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import sys
import time
import typing
from pathlib import Path

from .arguments import add_arguments, configure_packing


def main():
    started = time.time()
    root = Path(__file__).resolve().parents[4]
    # The two checkouts both contain a top-level pretrain_gpt and gpt_builders.
    # Resolve DAS scripts first and Megatron imports from its pinned submodule.
    sys.path[:0] = [str(root), str(root / "3rdparty" / "Megatron-LM")]
    # The pinned GPT builder imports typing.override on the DAS Python 3.10 image.
    if not hasattr(typing, "override"):
        from typing_extensions import override

        typing.override = override
    import pretrain_gpt as gpt

    def extra_args(parser):
        if gpt.has_nvidia_modelopt:
            parser = gpt.add_modelopt_args(parser)
        return add_arguments(parser)

    gpt.set_startup_timestamps(program_start=started, main_entry=time.time())
    args = gpt.parse_and_validate_args(
        extra_args_provider=extra_args, args_defaults={"tokenizer_type": "GPT2BPETokenizer"}
    )
    packed = configure_packing(args)
    if gpt.has_nvidia_modelopt:
        gpt.maybe_enable_modelopt(args)
    if gpt.has_nvidia_modelopt and getattr(args, "modelopt_enabled", False):
        model_cfg = gpt.gpt_config_from_args(args, model_config_cls=gpt.ModelOptModelConfig)
    else:
        model_cfg = gpt.gpt_config_from_args(args)

    forward_step = gpt.forward_step
    provider = gpt.train_valid_test_datasets_provider
    if packed:
        from .batch import forward_step
        from .dataset import install_data_loader, make_dataset_provider
        from .pipeline import install_pipeline
        from .runtime import install_runtime

        # The target dataclass predates these flags. The model, P2P code and
        # packing schedule all consume the same transformer config instance.
        config = model_cfg.transformer
        for name in ("sequence_packing_scheduler", "dynamic_context_parallel",
                     "min_dynamic_context_parallel_size", "pad_packed_seq_alignment",
                     "variable_seq_lengths"):
            setattr(config, name, getattr(args, name))
        install_runtime(dynamic=args.dynamic_context_parallel)
        install_data_loader()
        install_pipeline()
        provider = make_dataset_provider(gpt.core_gpt_dataset_config_from_args)
        gpt.print_rank_0(
            f"GPT sequence packing: {args.sequence_packing_scheduler}, "
            f"per-rank capacity={args.max_seqlen_per_dp_cp_rank} tokens"
        )

    provider.is_distributed = True
    train, store = gpt.inprocess_restart.maybe_wrap_for_inprocess_restart(gpt.pretrain)
    train(
        gpt.pretrain_cfg_container_from_args(args, model_cfg),
        provider,
        gpt.ModelType.encoder_or_decoder,
        forward_step,
        store=store,
        get_embedding_ranks=gpt.get_embedding_ranks,
    )


if __name__ == "__main__":
    main()
