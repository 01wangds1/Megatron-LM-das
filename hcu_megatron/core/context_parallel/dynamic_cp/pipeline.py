# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from functools import wraps
import time

import torch
from megatron.core import parallel_state
from megatron.core.num_microbatches_calculator import get_num_microbatches
from megatron.core.pipeline_parallel.utils import is_pp_last_stage
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.rerun_state_machine import RerunMode, get_rerun_state_machine
from megatron.core.utils import get_attr_wrapped_model
from megatron.training import ft_integration, get_args, get_timers, print_rank_0
from megatron.training import training

from .data_schedule import wrap_data_iterator


def _wrap_train_step(train_step, save_stats):
    @wraps(train_step)
    def packed_train_step(forward_step_func, data_iterator, model, optimizer,
                          opt_param_scheduler, config, forward_backward_func, *args, **kwargs):
        def run_schedule(*, data_iterator, num_microbatches, **schedule_kwargs):
            # Read through the raw RerunDataIterator inside the rerun loop, so a
            # replay or checkpoint restore reconstructs the same packed batch.
            packed, count, tokens, squares = wrap_data_iterator(
                data_iterator, config, num_microbatches
            )
            save_stats(tokens, squares)
            losses = forward_backward_func(
                data_iterator=packed, num_microbatches=count, **schedule_kwargs
            )
            if not losses:
                return losses
            # A whole SFT step can have zero supervised tokens. The target's
            # two-value reporting branch divides unconditionally; pass its
            # scalar branch the already-reduced, token-weighted means.
            totals = {}
            _accumulate_loss_sums(
                totals, losses, parallel_state.get_data_parallel_group(with_context_parallel=True)
            )
            return [_token_means(totals)]

        # Update the optimizer and consumed-sample counters using the original
        # sample count. Only the number of pipeline forward/backward calls changes.
        return train_step(
            forward_step_func, data_iterator, model, optimizer, opt_param_scheduler,
            config, run_schedule, *args, **kwargs
        )

    return packed_train_step


def _skip_raw_batch(data_iterator):
    """Fast-forward the TP0 data owner over one original global batch."""
    source = data_iterator
    if isinstance(source, list):
        source = next((it for it in source if it is not None), None)
    state = get_rerun_state_machine()
    while state.should_run_forward_backward(data_iterator):
        if source is not None:
            for _ in range(get_num_microbatches()):
                next(source)


def _accumulate_loss_sums(totals, losses, group):
    """Sum local loss numerators and denominators over the fixed replica domain."""
    if not losses:
        return
    for key in losses[0]:
        entries = torch.stack([batch[key].reshape(-1) for batch in losses])
        pair = entries.float().sum(dim=0)
        if pair.numel() == 1:
            pair = torch.cat((pair, pair.new_tensor([len(losses)])))
        elif pair.numel() != 2:
            raise ValueError(f"Expected a scalar or (loss sum, token count) for {key}")
        torch.distributed.all_reduce(pair, group=group)
        if key in totals:
            totals[key] += pair
        else:
            totals[key] = pair


def _token_means(totals):
    return {key: pair[0] / pair[1].clamp(min=1) for key, pair in totals.items()}


@torch.no_grad()
def _evaluate(forward_step_func, data_iterator, model, process_non_loss_data_func, config,
              verbose=False, non_loss_data_func=None, eval_iters=None, pg_collection=None,
              p2p_communicator=None):
    args = get_args()
    timers = get_timers()
    state = get_rerun_state_machine()
    old_mode = state.get_mode()
    original_timers = config.timers
    groups = get_attr_wrapped_model(model[0], "pg_collection")
    if groups is None:
        groups = ProcessGroupCollection.use_mpu_process_groups()
    reduction_group = getattr(groups, "dp_cp_gtp_remat", None) or groups.dp_cp
    run = training.get_forward_backward_func(schedule_pg_collection=pg_collection)
    raw_count = args.eval_global_batch_size // (
        args.eval_micro_batch_size * args.data_parallel_size * args.gtp_weight_remat_size
    )

    def run_batch(collect_non_loss_data=False):
        try:
            packed, count, _, _ = wrap_data_iterator(data_iterator, config, raw_count)
        except StopIteration:
            # The scheduler synchronizes exhaustion before any rank enters F/B.
            return None
        ft_integration.on_eval_step_start()
        try:
            return run(
                forward_step_func=forward_step_func, data_iterator=packed, model=model,
                num_microbatches=count, seq_length=args.seq_length,
                micro_batch_size=args.eval_micro_batch_size,
                decoder_seq_length=args.decoder_seq_length, forward_only=True,
                collect_non_loss_data=collect_non_loss_data, pg_collection=pg_collection,
                p2p_communicator=p2p_communicator,
            )
        finally:
            ft_integration.on_eval_step_end()

    timers("evaluate", log_level=0).start(barrier=True)
    for chunk in model:
        chunk.eval()
    config.timers = None
    state.set_mode(RerunMode.DISABLED)
    totals = {}
    collected = None
    try:
        steps = args.eval_iters if eval_iters is None else eval_iters
        for step in range(steps):
            if verbose:
                print_rank_0(f"Evaluating iter {step + 1}/{steps}")
            losses = run_batch()
            if losses is None:
                break
            if is_pp_last_stage(groups.pp):
                _accumulate_loss_sums(totals, losses, reduction_group)
            args.consumed_valid_samples += args.eval_global_batch_size
            if args.empty_unused_memory_level >= 1:
                torch.cuda.empty_cache()
            if args.exit_duration_in_mins:
                elapsed = (time.time() - training._TRAIN_START_TIME) / 60
                done = torch.tensor(int(elapsed > args.exit_duration_in_mins), device="cuda")
                torch.distributed.all_reduce(done, op=torch.distributed.ReduceOp.MAX)
                if done.item():
                    return None, None, True
        if non_loss_data_func is not None:
            collected = non_loss_data_func(model)
        elif process_non_loss_data_func is not None:
            # All stages must enter the schedule even when only one logs its output.
            collected = run_batch(collect_non_loss_data=True)
    finally:
        config.timers = original_timers
        state.set_mode(old_mode)
        for chunk in model:
            chunk.train()
        timers("evaluate").stop()
        timers.log(["evaluate"])
    return _token_means(totals), collected, False


def install_pipeline():
    """Install once after DAS has registered its training functions."""
    from hcu_megatron.patch_utils import MegatronPatchesManager
    from hcu_megatron.training import training as das_training

    # DAS retains the original v0.19.2 step when registering its EDGC variant.
    # The vanilla step accepts the same scheduler and handles an unset save interval.
    original_step = MegatronPatchesManager.patches_info[
        "megatron.training.training.train_step"
    ].orig_func_or_cls
    iteration_stats = (None, None)

    def save_stats(tokens, squares):
        nonlocal iteration_stats
        iteration_stats = (tokens, squares)

    def consume_stats():
        nonlocal iteration_stats
        result, iteration_stats = iteration_stats, (None, None)
        return result

    packed_step = _wrap_train_step(original_step, save_stats)
    # HCU train() and Megatron evaluate_and_print_results() each retain
    # module-local function references from before the adaptor was installed.
    for module in (training, das_training):
        module.train_step = packed_step
        module.dummy_train_step = _skip_raw_batch
        module.evaluate = _evaluate
        module.consume_seqlen_stats_in_iteration = consume_stats
