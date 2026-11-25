#!/usr/bin/env bash
# Qwen3-4B packed training on v0.19.2 DAS.
set -eo pipefail

die() { printf '%s\n' "$*" >&2; exit 2; }

usage() {
    cat <<'EOF'
Usage: bash train_qwen3_4B_dynamic_cp.sh [HOSTFILE NNODES] [options]

  DATA_PATH=/data/train.jsonl TOKENIZER_MODEL_PATH=/models/Qwen3-4B bash train_qwen3_4B_dynamic_cp.sh
  --data_path=PATH               Text records for pretraining or messages JSONL for SFT
  --tokenizer_path=PATH          Tokenizer directory or Hugging Face model ID
  --training_mode=pretrain|sft   SFT also accepts a directory containing train.jsonl
  --launch_backend=torchrun|mpirun     Override LAUNCHER
  --checkpoint_path=DIR          Save Megatron checkpoints (LOAD_PATH resumes one)
  --launch_with_binding=SCRIPT   Optional MPI rank binding script
  --train_iters=N                Training steps (default 100)
  --dry-run                      Print commands without starting MPI or training

Direct launch defaults to single-node torchrun. With HOSTFILE, run once from
the first host (one hostname or IP per line). LAUNCHER=mpirun starts one worker
per device, the hostfile default. LAUNCHER=torchrun starts one agent per host.
To launch torchrun separately on each node, set NNODES, NODE_RANK, MASTER_ADDR,
MASTER_PORT and GPUS_PER_NODE. GPUS_PER_NODE defaults to 8.
HCU_DYNAMIC_CONTEXT_PARALLEL=1 enables dynamic CP (default), 0 uses static packing.
EOF
}

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/../.." && pwd)
hostfile=
worker=0
if [[ ${1:-} == --worker ]]; then worker=1; shift; fi
if [[ -n ${OMPI_COMM_WORLD_RANK:-} ]]; then worker=1; fi
if (( $# )) && [[ $1 != -* ]]; then
    (( $# >= 2 )) || die 'Expected HOSTFILE NNODES, or MASTER_ADDR MASTER_PORT for an MPI worker'
    if (( worker )); then
        # The original examples/qwen3/run_qwen.sh supplies HOST PORT per MPI process.
        MASTER_ADDR=$1
        MASTER_PORT=$2
    else
        hostfile=$1
        NNODES=$2
        [[ -r $hostfile ]] || die "Cannot read hostfile: $hostfile"
    fi
    shift 2
fi

for option in "$@"; do
    case "$option" in
        -h|--help) usage; exit 0 ;;
        --data_path=*) DATA_PATH=${option#*=} ;;
        --tokenizer_path=*) TOKENIZER_MODEL_PATH=${option#*=} ;;
        --launch_backend=*) LAUNCHER=${option#*=} ;;
        --launch_with_binding=*) LAUNCH_WITH_BINDING=${option#*=} ;;
        --checkpoint_path=*) CHECKPOINT_PATH=${option#*=} ;;
        --training_mode=*) TRAINING_MODE=${option#*=} ;;
        --train_iters=*) TRAIN_ITERS=${option#*=} ;;
        --profiling=) ;;  # The stock run_qwen.sh always passes this option.
        --dry-run) DRY_RUN=1 ;;
        *) die "Unknown option: $option (see --help)" ;;
    esac
done

cp_dir="$repo_root/hcu_megatron/core/context_parallel"
for source_file in __init__.py dynamic_cp/__init__.py dynamic_cp/pretrain_gpt.py; do
    [[ -f "$cp_dir/$source_file" ]] || die "Missing $cp_dir/$source_file. Sync the full hcu_megatron/core/context_parallel directory to this node."
done

: "${DATA_PATH:?Set DATA_PATH or --data_path=PATH}"
: "${TOKENIZER_MODEL_PATH:?Set TOKENIZER_MODEL_PATH or --tokenizer_path=PATH}"
: "${DRY_RUN:=0}" "${TRAINING_MODE:=pretrain}" "${NNODES:=1}" "${GPUS_PER_NODE:=8}" "${MASTER_PORT:=11452}"
[[ $DRY_RUN == 0 || $DRY_RUN == 1 ]] || die 'DRY_RUN must be 0 or 1'
[[ $NNODES =~ ^[1-9][0-9]*$ && $GPUS_PER_NODE =~ ^[1-9][0-9]*$ ]] ||
    die 'NNODES and GPUS_PER_NODE must be positive integers'

default_backend=torchrun
if [[ -n $hostfile ]]; then
    mapfile -t hosts < <(awk '{sub(/#.*/, ""); sub(/\r$/, ""); if (NF) print $1}' "$hostfile")
    (( ${#hosts[@]} >= NNODES )) || die "Hostfile contains fewer than $NNODES hosts"
    : "${MASTER_ADDR:=${hosts[0]}}"
    default_backend=mpirun
elif (( worker )); then
    default_backend=mpirun
fi
export MEGATRON_LAUNCH_BACKEND=${LAUNCHER:-${MEGATRON_LAUNCH_BACKEND:-$default_backend}}
case $MEGATRON_LAUNCH_BACKEND in
    torchrun|mpirun) ;;
    *) die 'LAUNCHER must be torchrun or mpirun' ;;
esac
if [[ -z $hostfile ]] && (( ! worker && NNODES > 1 )); then
    [[ $MEGATRON_LAUNCH_BACKEND == torchrun && -n ${MASTER_ADDR:-} && -n ${NODE_RANK:-} ]] ||
        die 'Supply HOSTFILE NNODES, or set MASTER_ADDR and NODE_RANK on each node for torchrun'
fi
if (( ! worker )); then : "${MASTER_ADDR:=127.0.0.1}"; fi

if (( ! worker )) && [[ -n $hostfile || $MEGATRON_LAUNCH_BACKEND == mpirun ]]; then
    # Dispatch this same script once per DCU, or once per host for torchrun.
    unset NODE_RANK  # Each remote process receives its own Open MPI rank.
    if [[ $MEGATRON_LAUNCH_BACKEND == torchrun ]]; then slots=1; else slots=$GPUS_PER_NODE; fi
    mpi=(mpirun --allow-run-as-root --bind-to none --map-by "ppr:$slots:node"
         -np "$((NNODES * slots))" --wdir "$repo_root")
    if [[ -n $hostfile ]]; then
        host_slots=()
        for ((i = 0; i < NNODES; i++)); do host_slots+=("${hosts[i]}:$slots"); done
        host_list=$(IFS=,; echo "${host_slots[*]}")
        mpi+=(--host "$host_list")
    fi
    if [[ -n ${SSH_PORT:-} ]]; then mpi+=(--mca plm_rsh_args "-p $SSH_PORT"); fi
    for name in PATH LD_LIBRARY_PATH PYTHONPATH DTK_ENV NCCL_ENV LAUNCH_WITH_BINDING \
        MASTER_ADDR MASTER_PORT NNODES GPUS_PER_NODE HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES \
        IB_TYPE NCCL_SOCKET_IFNAME GLOO_SOCKET_IFNAME NCCL_IB_HCA GPU_MAX_HW_QUEUES \
        CUDA_DEVICE_MAX_CONNECTIONS DATA_PATH TOKENIZER_MODEL_PATH TRAINING_MODE \
        SFT_PROMPT_FORMAT TRAIN_ITERS LOAD_PATH CHECKPOINT_PATH SAVE_INTERVAL \
        HCU_DYNAMIC_CONTEXT_PARALLEL TP PP CP MBS GBS SEQ_LEN MAX_POSITION_EMBEDDINGS \
        MAX_SEQLEN_PER_DP_CP_RANK MIN_DYNAMIC_CONTEXT_PARALLEL_SIZE NUM_WORKERS PYTHON TORCHRUN \
        TORCH_FORCE_WEIGHTS_ONLY_LOAD TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD; do
        if [[ -v $name ]]; then export "$name"; mpi+=(-x "$name"); fi
    done
    remote=(bash "examples/qwen3/${BASH_SOURCE[0]##*/}" --worker "--launch_backend=$MEGATRON_LAUNCH_BACKEND")
    if [[ $DRY_RUN == 0 ]]; then exec "${mpi[@]}" "${remote[@]}"; fi
    printf 'MPI dispatch: '; printf '%q ' "${mpi[@]}" "${remote[@]}"; printf '\n'
    # Render rank zero's training command locally for a hostfile dry run.
    export OMPI_COMM_WORLD_RANK=0 OMPI_COMM_WORLD_LOCAL_RANK=0 OMPI_COMM_WORLD_SIZE=$((NNODES * slots))
fi

# The stock network environment derives its own paths from the invoking script.
# Preserve the caller's Python paths and set the actual DAS paths below.
source_default_network_env() {
    local PYTHONPATH=${PYTHONPATH:-} CURRENT_DIR MEGATRON_PATH CUDA_DEVICE_MAX_CONNECTIONS
    source "$repo_root/requirements/env.sh"
}
if [[ $DRY_RUN == 0 ]]; then
    if [[ -n ${DTK_ENV:-} ]]; then source "$DTK_ENV"; fi
    if [[ -z ${NCCL_ENV:-} || $NCCL_ENV -ef "$repo_root/requirements/env.sh" ]]; then
        source_default_network_env || die "Failed to initialize $repo_root/requirements/env.sh"
    else
        source "$NCCL_ENV"
    fi
fi
set -euo pipefail
cd -- "$repo_root"
export PYTHONPATH="$repo_root:$repo_root/3rdparty/Megatron-LM:$repo_root/3rdparty/Megatron-Bridge/src${PYTHONPATH:+:$PYTHONPATH}"
export GPU_MAX_HW_QUEUES="${GPU_MAX_HW_QUEUES:-4}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export HCU_DYNAMIC_CONTEXT_PARALLEL="${HCU_DYNAMIC_CONTEXT_PARALLEL:-1}"

: "${TP:=1}" "${PP:=1}" "${CP:=2}" "${MBS:=1}" "${GBS:=32}" "${SEQ_LEN:=4096}"
: "${MAX_POSITION_EMBEDDINGS:=40960}" "${MIN_DYNAMIC_CONTEXT_PARALLEL_SIZE:=1}"
: "${TRAIN_ITERS:=100}" "${NUM_WORKERS:=2}"
: "${SAVE_INTERVAL:=50}"
for key in TP PP CP MBS GBS SEQ_LEN MAX_POSITION_EMBEDDINGS MIN_DYNAMIC_CONTEXT_PARALLEL_SIZE \
           TRAIN_ITERS SAVE_INTERVAL MASTER_PORT; do
    [[ ${!key} =~ ^[1-9][0-9]*$ ]] || die "$key must be a positive integer"
done
[[ $NUM_WORKERS =~ ^(0|[1-9][0-9]*)$ ]] || die 'NUM_WORKERS must be a nonnegative integer'
(( MASTER_PORT <= 65535 )) || die 'MASTER_PORT must be between 1 and 65535'
(( MAX_POSITION_EMBEDDINGS >= SEQ_LEN )) || die 'MAX_POSITION_EMBEDDINGS must cover SEQ_LEN'
: "${MAX_SEQLEN_PER_DP_CP_RANK:=$(( (SEQ_LEN + CP - 1) / CP ))}"
[[ $MAX_SEQLEN_PER_DP_CP_RANK =~ ^[1-9][0-9]*$ ]] || die 'MAX_SEQLEN_PER_DP_CP_RANK must be positive'

module=hcu_megatron.core.context_parallel.dynamic_cp.pretrain_gpt
MPI_DISTRIBUTED_ARGS=()
case "$MEGATRON_LAUNCH_BACKEND" in
    torchrun)
        if (( NNODES > 1 )); then
            : "${MASTER_ADDR:?Set MASTER_ADDR for multi-node torchrun}"
            NODE_RANK=${NODE_RANK:-${OMPI_COMM_WORLD_RANK:-${PMI_RANK:-}}}
            : "${NODE_RANK:?Set NODE_RANK for multi-node torchrun}"
        fi
        : "${MASTER_ADDR:=127.0.0.1}" "${NODE_RANK:=0}"
        [[ $NODE_RANK =~ ^(0|[1-9][0-9]*)$ ]] && (( NODE_RANK < NNODES )) || die 'NODE_RANK must be in [0, NNODES)'
        world_size=$((NNODES * GPUS_PER_NODE))
        launcher=("${TORCHRUN:-torchrun}" --nnodes "$NNODES" --node_rank "$NODE_RANK"
                  --master_addr "$MASTER_ADDR" --master_port "$MASTER_PORT"
                  --nproc_per_node "$GPUS_PER_NODE" --module "$module")
        ;;
    mpirun)
        : "${OMPI_COMM_WORLD_RANK:?Start mpirun with one process per GPU}"
        : "${OMPI_COMM_WORLD_LOCAL_RANK:?Missing MPI local rank}" "${OMPI_COMM_WORLD_SIZE:?Missing MPI world size}"
        : "${MASTER_ADDR:?Pass master host and port, or set MASTER_ADDR}"
        world_size=$OMPI_COMM_WORLD_SIZE
        [[ $world_size =~ ^[1-9][0-9]*$ ]] || die 'Invalid MPI world size'
        export RANK=$OMPI_COMM_WORLD_RANK LOCAL_RANK=$OMPI_COMM_WORLD_LOCAL_RANK WORLD_SIZE=$world_size
        MPI_DISTRIBUTED_ARGS=(--rank "$RANK" --local-rank "$LOCAL_RANK" --world-size "$WORLD_SIZE"
                              --dist-url "tcp://$MASTER_ADDR:$MASTER_PORT")
        launcher=()
        if [[ -n ${LAUNCH_WITH_BINDING:-} ]]; then launcher+=(bash "$LAUNCH_WITH_BINDING" "$LOCAL_RANK"); fi
        launcher+=("${PYTHON:-python}" -u -m "$module")
        ;;
    *) die 'launch_backend must be torchrun or mpirun' ;;
esac
(( world_size % (TP * PP * CP) == 0 )) || die 'World size must be divisible by TP * PP * CP'
dp=$(( world_size / (TP * PP * CP) ))
(( GBS % (MBS * dp) == 0 )) || die 'GBS must be divisible by MBS * DP'

case "$HCU_DYNAMIC_CONTEXT_PARALLEL" in
    1) scheduler=default_dynamic_cp; pool=$((dp * CP)) ;;
    0) scheduler=dp_balanced; pool=$CP ;;
    *) die 'HCU_DYNAMIC_CONTEXT_PARALLEL must be 0 or 1' ;;
esac
(( pool * MAX_SEQLEN_PER_DP_CP_RANK >= SEQ_LEN )) || die 'Increase MAX_SEQLEN_PER_DP_CP_RANK to accommodate SEQ_LEN'

GPT_MODEL_ARGS=(
    --num-layers 36 --hidden-size 2560 --ffn-hidden-size 9728
    --num-attention-heads 32 --kv-channels 128 --num-query-groups 8 --group-query-attention
    --normalization RMSNorm --swiglu --qk-layernorm --disable-bias-linear
    --position-embedding-type rope --rotary-base 1000000
    --seq-length "$SEQ_LEN" --max-position-embeddings "$MAX_POSITION_EMBEDDINGS"
)
TRAINING_ARGS=(
    --transformer-impl transformer_engine --use-mcore-models --use-flash-attn --bf16
    --micro-batch-size "$MBS" --global-batch-size "$GBS" --train-iters "$TRAIN_ITERS"
    --lr 3e-5 --min-lr 3e-6 --lr-decay-style cosine --lr-warmup-iters "$(( TRAIN_ITERS > 1 ))"
    --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.95 --init-method-std 0.02 --clip-grad 1.0
    --attention-dropout 0 --hidden-dropout 0 --use-distributed-optimizer --overlap-grad-reduce
    --schedule-method vanilla --calculate-per-token-loss --dataloader-type single --num-workers "$NUM_WORKERS"
    --no-create-attention-mask-in-dataloader --ckpt-format torch --log-throughput --log-interval 1 --eval-iters 0
)
MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size "$TP" --pipeline-model-parallel-size "$PP" --context-parallel-size "$CP"
    --sequence-packing-scheduler "$scheduler" --max-seqlen-per-dp-cp-rank "$MAX_SEQLEN_PER_DP_CP_RANK"
)
if (( TP > 1 )); then MODEL_PARALLEL_ARGS+=(--sequence-parallel); fi
if [[ $HCU_DYNAMIC_CONTEXT_PARALLEL == 1 ]]; then
    MODEL_PARALLEL_ARGS+=(--min-dynamic-context-parallel-size "$MIN_DYNAMIC_CONTEXT_PARALLEL_SIZE")
fi

DATA_ARGS=(--tokenizer-model "$TOKENIZER_MODEL_PATH")
case "$TRAINING_MODE" in
    pretrain)
        DATA_ARGS+=(--use-varlen-dataset --tokenizer-type HuggingFaceTokenizer --data-path "$DATA_PATH" --split 100,0,0)
        ;;
    sft)
        if [[ -d $DATA_PATH ]]; then DATA_PATH=${DATA_PATH%/}/train.jsonl; fi
        DATA_ARGS+=(--sft --sft-tokenizer-prompt-format "${SFT_PROMPT_FORMAT:-default}"
                    --tokenizer-type SFTTokenizer --train-data-path "$DATA_PATH")
        ;;
    *) die 'training_mode must be pretrain or sft' ;;
esac
if [[ -n ${LOAD_PATH:-} ]]; then TRAINING_ARGS+=(--load "$LOAD_PATH"); fi
if [[ -n ${CHECKPOINT_PATH:-} ]]; then TRAINING_ARGS+=(--save "$CHECKPOINT_PATH" --save-interval "$SAVE_INTERVAL"); fi

command=("${launcher[@]}" "${GPT_MODEL_ARGS[@]}" "${TRAINING_ARGS[@]}"
         "${MODEL_PARALLEL_ARGS[@]}" "${DATA_ARGS[@]}" "${MPI_DISTRIBUTED_ARGS[@]}")
printf 'Qwen3-4B %s: %s, %s, TP=%s PP=%s CP=%s DP=%s, rank capacity=%s\n' \
    "$TRAINING_MODE" "$MEGATRON_LAUNCH_BACKEND" "$scheduler" "$TP" "$PP" "$CP" "$dp" "$MAX_SEQLEN_PER_DP_CP_RANK"
if [[ $DRY_RUN == 1 ]]; then
    printf 'Dry run (environment files skipped): '
    printf '%q ' "${command[@]}"
    printf '\n'
else
    exec "${command[@]}"
fi
