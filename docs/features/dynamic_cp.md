# Dynamic CP 动态上下文并行

## 简介

动态上下文并行（Dynamic Context Parallelism，简称 Dynamic CP 或 DCP）面向长度差异较大的变长序列训练。在每个训练步中，它根据样本长度、计算负载和单卡 token 容量重新打包数据，并为不同的 packed microbatch 选择合适的 CP 规模，以减少负载不均和不必要的上下文并行通信。

当前 DAS 实现面向 Megatron Core v0.19.2，适用于采用 Transformer Engine THD attention 的 dense GPT 预训练和文本 SFT。通过独立的训练模块入口启用，使用方式见[模块 README](../../hcu_megatron/core/context_parallel/dynamic_cp/README.md)。

## 实现原理

### 如何动态选择 CP 组

1. **预建通信组。** 在固定 TP、PP 位置的 DP×CP 资源池中，初始化时创建多个候选 CP 组，运行时选择已有组。
2. **按训练步重新调度。** 读取一个原始 global batch，根据样本长度、估计的 Attention 工作量和每 rank 容量分配样本，生成 packed microbatch 及其 CP 组。
3. **携带微批次元数据执行。** 使用 THD（token、head、head dimension）布局和 `cu_seqlens` 等序列边界元数据；RoPE、Attention 和 CP 分片使用当前微批次对应的组。
4. **对齐训练调度。** 打包后微批次数可能变化，训练调度器使用实际微批次数，并保证各 rank 的前反向次数对齐。优化器更新、学习率和 checkpoint 的样本计数仍按原始 global batch 计算；梯度按固定 DP×CP 域内的有效监督 token 数归一化。

![同一 DP×CP 资源池内选择不同规模的 CP 组]

以 TP=1、PP=1、总卡数为 8 的配置为例，即使初始化设置 `--context-parallel-size 2`，动态调度也可以从 1、2、4、8 中选择运行时 CP 规模。初始 CP 值用于确定基础并行拓扑，不是动态 CP 的上限；TP、PP 的模型切分保持不变，也不需要随每个微批次重新切分模型权重。

图中分组仅说明选择方式，实际分组由容量和负载共同决定，并不保证所有短样本都使用 CP=1。

## 使用方式

以下命令在已能运行 v0.19.2 DAS 的 Linux/DCU 训练容器中执行，工作目录为 **Megatron-LM-das 仓库根目录**。向训练节点同步代码时，需要包含完整的 `hcu_megatron/core/context_parallel/` 目录及其 `__init__.py`，避免旧版同名 Python 包遮蔽当前实现。

### 单机预训练

使用仓库提供的 [Qwen3-4B 启动脚本](../../examples/qwen3/train_qwen3_4B_dynamic_cp.sh)。`DATA_PATH` 为变长文本 JSONL 文件，每行格式如下：

```json
{"text": "一段训练文本"}
```

```bash
launcher=examples/qwen3/train_qwen3_4B_dynamic_cp.sh
export TOKENIZER_MODEL_PATH=/models/Qwen3-4B
export DATA_PATH=/data/dcp_varlen_train.jsonl

# 先预览生成的命令，不启动训练
bash "$launcher" --dry-run

# 单机 8 卡，开启动态 CP
HCU_DYNAMIC_CONTEXT_PARALLEL=1 bash "$launcher" --train_iters=100
```

脚本默认 TP=1、PP=1、CP=2，单机 DP=4；序列上限为 4096，单 rank 容量为 2048 tokens，global batch 为 32 个原始样本。可以通过环境变量覆盖这些配置，例如：

```bash
TP=1 PP=1 CP=4 GBS=32 SEQ_LEN=4096 MAX_SEQLEN_PER_DP_CP_RANK=1024 \
  HCU_DYNAMIC_CONTEXT_PARALLEL=1 bash "$launcher" --train_iters=100
```

`TOKENIZER_MODEL_PATH` 指向 Qwen3-4B 的 Hugging Face tokenizer，模型默认随机初始化。恢复已有 Megatron checkpoint 时设置 `LOAD_PATH`；保存时设置 `CHECKPOINT_PATH` 和 `SAVE_INTERVAL`。脚本采用 `torch` checkpoint 格式，`--train_iters` 表示恢复后要达到的总训练步数。

### 静态 CP 对照

使用相同数据和并行配置，将环境开关设为 `0`：

```bash
HCU_DYNAMIC_CONTEXT_PARALLEL=0 bash "$launcher" --train_iters=100
```

此时仍使用变长数据和 packing，但调度器变为 `dp_balanced`，各微批次使用固定 CP。默认配置中固定 CP=2。启动日志中的 `GPT sequence packing: default_dynamic_cp` 和 `GPT sequence packing: dp_balanced` 分别表示动态与静态 packing 路径。

对照时应保持样本、tokenizer、初始化权重、global batch、TP/PP、容量和精度等条件一致。关闭 DCP 并不等于关闭 packing。DCP 开关在启动时解析，切换动态与静态模式需要重新启动训练进程。

### 文本 SFT

SFT 数据使用包含 `messages` 的 JSONL，例如：

```json
{"messages": [{"role": "user", "content": "什么是上下文并行？"}, {"role": "assistant", "content": "上下文并行将长序列分到多个设备协同计算。"}]}
```

```bash
HCU_DYNAMIC_CONTEXT_PARALLEL=1 bash "$launcher" \
  --training_mode=sft --data_path=/data/sft/train.jsonl --train_iters=100
```

`--data_path` 也可以传入目录，脚本读取其中的 `train.jsonl`。模型和数据的相对路径从 DAS 仓库根目录解析。

### 多机启动

准备 hostfile，每行一个节点主机名或 IP，第一行为主节点。各节点需能通过 Open MPI/SSH 互访，并保持仓库、数据、tokenizer 和 checkpoint 路径一致。下面以两台机器、每台 8 卡为例，在主节点执行一次：

```bash
# 默认由 Open MPI 每卡启动一个训练进程
bash "$launcher" /shared/hostfile 2 --train_iters=100

# 也可由 Open MPI 在每个节点启动一个 torchrun
LAUNCHER=torchrun bash "$launcher" /shared/hostfile 2 --train_iters=100
```

非默认 SSH 端口使用 `SSH_PORT`。脚本加载 `requirements/env.sh`，也可以通过 `DTK_ENV` 和 `NCCL_ENV` 指定平台及通信环境脚本。单机默认直接使用 `torchrun`。

### 参数说明

启动脚本使用下列环境变量控制配置，其默认值仅适用于该脚本：

| 环境变量 | 默认值 | 说明 |
| --- | --- | --- |
| `HCU_DYNAMIC_CONTEXT_PARALLEL` | `1` | `1` 启用 DCP，`0` 使用静态 packing |
| `GPUS_PER_NODE` | `8` | 每节点设备数 |
| `TP` / `PP` / `CP` | `1` / `1` / `2` | 初始化并行拓扑 |
| `MBS` / `GBS` | `1` / `32` | 初始 micro batch size 与原始 global batch size；打包后的实际微批次数由调度结果决定 |
| `SEQ_LEN` | `4096` | 输入序列长度上限 |
| `MAX_SEQLEN_PER_DP_CP_RANK` | `ceil(SEQ_LEN / CP)` | 每 rank 的物理 token 容量，默认配置为 2048，包含 padding |
| `MIN_DYNAMIC_CONTEXT_PARALLEL_SIZE` | `1` | 动态 CP 候选组规模下界，必须是合法候选值 |
| `LOAD_PATH` / `CHECKPOINT_PATH` | 未设置 | 恢复 / 保存 Megatron checkpoint 的目录 |
| `SAVE_INTERVAL` | `50` | 设置保存目录后，每隔多少步保存 checkpoint |

也可以直接通过 `torchrun --module hcu_megatron.core.context_parallel.dynamic_cp.pretrain_gpt` 启动。此时使用以下 Megatron CLI 参数；完整的四卡合成数据示例见[模块 README](../../hcu_megatron/core/context_parallel/dynamic_cp/README.md#启动)。

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `--dynamic-context-parallel` | bool | `False` | 显式开启 DCP；也可用环境变量 `HCU_DYNAMIC_CONTEXT_PARALLEL=1` 开启 |
| `--sequence-packing-scheduler` | str | 自动选择 | 开启 DCP 时为 `default_dynamic_cp`；关闭 DCP 且使用变长数据时为 `dp_balanced` |
| `--min-dynamic-context-parallel-size` | int | `1` | 动态 CP 组规模下界 |
| `--max-seqlen-per-dp-cp-rank` | int | 需显式设置 | 每 rank 的物理 token 容量，必须为正数 |
| `--use-varlen-dataset` | bool | `False` | 使用变长文本数据入口，与 `--sft` 二选一 |
| `--pad-packed-seq-alignment` | int 或 `max` | 未设置 | 设置 packed 长度的额外对齐；`max` 补齐至每 rank 容量，仍须满足后端对齐要求 |

脚本选项与 Megatron CLI 参数属于不同层级。脚本接受 `--data_path=...`、`--tokenizer_path=...`、`--training_mode=...`、`--train_iters=...` 等自身选项，不透传任意 Megatron 参数。直接使用 CLI 开关时应清除 `HCU_DYNAMIC_CONTEXT_PARALLEL` 或将其设为 `1`；环境变量为 `0` 时同时传入 `--dynamic-context-parallel` 会报冲突。

原有训练仍可使用仓库根目录的 `pretrain_gpt.py`。本文的 DCP 配置需要上述独立模块入口，不能只给原入口增加 DCP 参数。

## 兼容性限制

以下限制针对当前 DAS v0.19.2 适配实现，具体校验见 [arguments.py](../../hcu_megatron/core/context_parallel/dynamic_cp/arguments.py)。

### 数据与并行拓扑

- 使用 `--use-varlen-dataset` 或 `--sft`，两者互斥；同时要求 `--dataloader-type single` 和 `--calculate-per-token-loss`。当前样本重分发面向文本 GPT/SFT 数据，多模态图像、视频及其元数据尚未接入。
- 资源池限定在同一 TP、PP 位置的 DP×CP ranks。候选 CP 规模为能够整除资源池大小的 2 的幂，加上完整资源池；例如资源池为 8 时可选 1、2、4、8，为 6 时可选 1、2、6。
- `--min-dynamic-context-parallel-size` 必须属于候选集合。每 rank 容量必须大于 0，动态模式要求 `DP × CP × 容量 >= seq_length`；静态 packing 要求 `CP × 容量 >= seq_length`。这只是必要容量检查，实际分包还需满足 padding 和后端对齐要求。
- 启动脚本要求总卡数能够被 `TP × PP × CP` 整除，`GBS` 能够被 `MBS × DP` 整除。不能用打包后的微批次数替代原始 batch 配置。
- 容量是 token 预算，不是显存自动探测结果；应根据模型规模、精度和激活显存设置。`--pad-packed-seq-alignment max` 还要求容量满足实际对齐粒度。

### 模型、算子与调度

| 类别 | 当前支持范围或限制 |
| --- | --- |
| 模型与 Attention | DCP 面向 dense GPT，使用 eager Transformer Engine THD attention；不支持 local attention backend、MoE、MLA、MTP 和实验性 Attention 变体 |
| 流水线 | 要求 `--schedule-method vanilla`、`--pipe-sp-splits 1`；可走标准 PP/VPP 路径，VPP 需要足够样本组成对齐的微批组；不支持 Seq1F1B、RiPipe 等自定义 DAS 调度 |
| CP 组合 | packing 与旧 `--hybrid-context-parallel` 互斥；DCP 不支持 `--hierarchical-context-parallel-sizes` |
| 执行与分片 | 不支持 CUDA Graph、Megatron FSDP、Torch FSDP2、GTP 权重重计算，以及 `--enable-vocab-parallel`、`--enable-dynamic-grad-comp`、`--overlap-moe-expert-parallel-comm` |
| 其他组合 | 不支持 `--fused-single-qkv-rope`、`--num-microbatches-with-partial-activation-checkpoints`、FIM、`--reset-position-ids`、`--reset-attention-mask`、RL step；DCP 不支持 `--deterministic-mode` |
| 评估 | 常规按 `--eval-iters` 计步的评估共用打包逻辑；不支持 `--full-validation`，Qwen3-4B 示例默认关闭评估 |

动态 CP 不改变模型计算目标，但打包顺序、归约顺序和 Attention 执行形状会变化，因此不应要求与静态路径逐位一致；应检查有效 token 的 loss、梯度及训练曲线是否在合理数值误差内。

## 性能说明

DCP 的收益主要来自减少长短样本之间的负载差异、减少短样本不必要的 CP 通信，以及改善微批次安排带来的等待。长度分布差异较大、静态 CP 为少量长样本配置得较大时，更有机会获得收益。若样本长度接近、CP=1 已能容纳数据，额外调度和数据重分发也可能使训练变慢。



