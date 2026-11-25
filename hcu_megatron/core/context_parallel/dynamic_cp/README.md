<!-- Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->

# DAS v0.19.2 的 Dynamic Context Parallelism

本目录为目标子模块 `4b4acac9` 适配了参考 checkout `2cb820e0` 中的 GPT 动态 CP 训练路径。每个训练步先读取一个原始 global batch，根据文档长度和单 rank token 容量打包，然后在固定 TP、PP 位置的 DP×CP 资源池中为各个 pack 选择 CP 组。所有 rank 的前反向次数对齐。优化器更新、学习率和 checkpoint 的样本计数使用原始 global batch，梯度按固定 DP×CP 域的有效监督 token 数归一化。[上游 PR #9061](https://github.com/NVIDIA/Megatron-LM/pull/9061)

## 启动

在已能运行 v0.19.2 DAS 的 Linux/DCU 训练容器中，从 **Megatron-LM-das 根目录**执行。Qwen3-4B 的模型参数、数据配置和进程分发都在同一个[启动脚本](../../../../examples/qwen3/train_qwen3_4B_dynamic_cp.sh)中。

向训练节点同步代码时，请复制整个 `hcu_megatron/core/context_parallel/` 目录，包括父目录的 `__init__.py` 和 `dynamic_cp/` 子包。父目录的包标记保证新仓库排在搜索路径首位时，Python 会选中这份实现。

```bash
launcher=examples/qwen3/train_qwen3_4B_dynamic_cp.sh
export TOKENIZER_MODEL_PATH=/models/Qwen3-4B
export DATA_PATH=/data/dcp_varlen_train.jsonl

bash "$launcher" --dry-run                 # 打印训练命令
bash "$launcher" --train_iters=100         # 单机 8 卡，开启动态 CP
HCU_DYNAMIC_CONTEXT_PARALLEL=0 bash "$launcher" --train_iters=100  # 相同数据，静态 CP 对照
```

**模型和数据。** 模型结构沿用 `examples/qwen3/train_qwen3_4B.sh`，共 36 层，hidden size 为 2560，32 个注意力头、8 个 KV 组，每头 128 维。`TOKENIZER_MODEL_PATH` 填训练容器内的 Qwen3-4B Hugging Face tokenizer 目录。模型默认随机初始化，如需读入已有 Megatron checkpoint，另设 `LOAD_PATH`。需要保存时设 `CHECKPOINT_PATH` 和 `SAVE_INTERVAL`。

checkpoint 沿用 4B 示例的 `torch` 格式。`LOAD_PATH` 用于恢复训练进度，`--train_iters` 表示恢复后要达到的总步数。若 PyTorch 报 `Weights only load failed`，只在恢复自己生成或已确认来源可信的 checkpoint 时设置 `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`，以读取其中的训练状态。

预训练的 `DATA_PATH` 填变长文本 JSONL 文件，每行例如 `{"text":"一段训练文本"}`。工作区的 `dcp_varlen_train.jsonl` 就是这个格式，挂载到容器后填写实际路径。也可用 `--data_path=...` 和 `--tokenizer_path=...` 传入路径。对话 SFT 用 `--training_mode=sft --data_path=/data/sft/train.jsonl`，每行包含 `messages`；传目录时读取其中的 `train.jsonl`。相对的模型和数据路径从 DAS 仓库根目录解析。

**并行参数。** 默认每节点 8 卡，TP=1、PP=1、CP=2，单机时 DP=4。序列上限为 4096，单 rank 容量默认为 2048 tokens，global batch 为 32 个样本。可用 `GPUS_PER_NODE`、`TP`、`PP`、`CP`、`GBS`、`SEQ_LEN` 和 `MAX_SEQLEN_PER_DP_CP_RANK` 覆盖。启动日志会打印 `GPT sequence packing: default_dynamic_cp` 或 `dp_balanced`，分别对应动态和静态调度。

**多机启动。** 准备一个 hostfile，每行填写一个训练节点的主机名或 IP，第一行是主节点。确保各节点能通过 Open MPI/SSH 互访，仓库、数据、tokenizer 和 checkpoint 在各节点的路径一致。两台机器、每台 8 卡时，在主节点执行一次下面的命令。

```bash
bash "$launcher" /shared/hostfile 2 --train_iters=100
```

hostfile 模式默认由 Open MPI 每卡启动一个训练进程。若希望每节点启动一个 `torchrun`，可在同一命令前加 `LAUNCHER=torchrun`。SSH 使用非默认端口时设 `SSH_PORT`。脚本在各节点加载仓库的 `requirements/env.sh`。可通过 `DTK_ENV` 指定平台初始化脚本，通过 `NCCL_ENV` 指定站点通信环境脚本。单机默认直接使用 `torchrun`。

平台环境沿用示例的 `CUDA_DEVICE_MAX_CONNECTIONS=1`。在 CUDA 中，它把每个设备的并发计算和拷贝工作队列连接数设为 1。Megatron 在部分 TP/CP 配置中会检查这个值。动态 CP 的开关由 `HCU_DYNAMIC_CONTEXT_PARALLEL` 单独控制。[CUDA 环境变量说明](https://docs.nvidia.com/cuda/archive/13.2.0/cuda-programming-guide/05-appendices/environment-variables.html#cuda-device-max-connections)

下例直接调用模块入口，显式指定合成样本的长度分布。它使用 4 卡、TP=1、PP=1、静态 CP=2 和 DP=2，运行 5 步。

```bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
export HCU_DYNAMIC_CONTEXT_PARALLEL=1
torchrun --standalone --nproc-per-node=4 \
  --module hcu_megatron.core.context_parallel.dynamic_cp.pretrain_gpt \
  --schedule-method vanilla --transformer-impl transformer_engine \
  --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 \
  --context-parallel-size 2 --max-seqlen-per-dp-cp-rank 512 \
  --min-dynamic-context-parallel-size 1 \
  --seq-length 1024 --max-position-embeddings 1024 \
  --micro-batch-size 1 --global-batch-size 16 \
  --num-layers 4 --hidden-size 256 --num-attention-heads 8 \
  --position-embedding-type rope --bf16 --attention-dropout 0 --hidden-dropout 0 \
  --no-gradient-accumulation-fusion --lr 0.00015 --lr-decay-style constant \
  --calculate-per-token-loss --dataloader-type single --use-varlen-dataset \
  --no-create-attention-mask-in-dataloader --mock-data --split 100,0,0 \
  --varlen-mock-dataset-config-json '{"mode":"distribution","type":"lognormal","min_seq_len":32,"max_seq_len":1024,"mean_seq_len":256,"lognormal_sigma":1.1}' \
  --tokenizer-type NullTokenizer --vocab-size 4096 --num-workers 0 \
  --train-iters 5 --log-interval 1 --eval-iters 0
```

| 场景 | 开关及数据选项 |
| --- | --- |
| 动态 CP | `HCU_DYNAMIC_CONTEXT_PARALLEL=1`，或清除该环境变量后传 `--dynamic-context-parallel`。调度器默认为 `default_dynamic_cp`。 |
| 同数据的静态 packing 对照 | 将上述命令的环境开关置为 `0`，保留变长数据参数，可显式传 `--sequence-packing-scheduler dp_balanced`。各 pack 使用固定 CP=2。 |
| 原有 DAS 训练 | 使用原根目录的 `pretrain_gpt.py`。新模块入口在开关关闭且未选择变长数据或 packing 调度器时，也调用原有 provider 和 forward step。 |

真实数据可将示例中的 mock、mock 长度配置和 NullTokenizer 选项替换为自己的数据路径及 tokenizer 参数。`--use-varlen-dataset` 支持带 `text` 的文本记录，以及 OpenAI messages、ShareGPT、Alpaca 格式；对话数据需要具有 `tokenize_conversation` 接口的 tokenizer。已有 Megatron SFT 消息数据可用 `--sft` 配合动态 CP 或显式 packing 调度器。两种数据入口互斥。训练和常规按 `--eval-iters` 计步的评估共用打包逻辑。

`--context-parallel-size` 确定初始化拓扑。`--max-seqlen-per-dp-cp-rank` 以单 rank 的物理 token 数为单位，调度会计入分片所需的 padding。动态 CP 可选组宽包括能够整除 DP×CP 的 2 的幂，以及完整资源池；`--min-dynamic-context-parallel-size` 进一步限定下界。标准 PP/VPP 路径使用实际微批次数与变长 P2P，VPP 还需要足够的样本组成对齐后的微批组。

本次动态路径面向使用 Transformer Engine THD attention 的 dense GPT。v0.19.2 中的 MoE、MLA、MTP、GTP 权重重计算、自定义 DAS 流水线、FSDP、CUDA Graph 和 full-validation 涉及不同的进程组、批次或梯度语义，入口会检查这些组合。HIP 的 BF16 因果 attention 保留参考实现对文档尾部 padding 的局部兼容处理。

## 改动文件

Python 实现和本文档位于 `hcu_megatron/core/context_parallel/dynamic_cp/`。启动脚本位于仓库的 `examples/qwen3/`。

| 文件 | 职责 |
| --- | --- |
| `../__init__.py` | 标记 `context_parallel` 父包，避免旧版同名包遮蔽当前目录 |
| `__init__.py` | 定义可选模块包 |
| `pretrain_gpt.py` | torchrun 入口，加载目标 DAS、选择训练路径 |
| `examples/qwen3/train_qwen3_4B_dynamic_cp.sh` | Qwen3-4B 模型、训练数据、checkpoint、静态或动态 CP，以及单机和多机启动 |
| `arguments.py` | 解析 CLI 和环境开关，校验调度参数 |
| `dataset.py` | 变长文本、SFT、mock 数据和 data loader 接入 |
| `data_schedule.py` | 每步打包调度、PP/VPP 数据所有权、静态 packing 基线 |
| `data_schedule_utils.py` | 组宽选择、样本重分发、THD 打包与对齐 |
| `batch.py` | CP 分片、TP 广播、PackedSeqParams 和 GPT forward |
| `runtime.py` | 预建 CP 组，向 RoPE 和 TE attention 传递当前微批的组 |
| `causal_padding.py` | HIP/BF16 因果 attention 的物理长度处理 |
| `pipeline.py` | 训练和评估的实际微批次数、loss 汇总与 token 统计 |
| `README.md` | 启动方式、适用范围与文件清单 |

## 验证范围

已进行 Python 语法检查、入口和目标接口核对、开关检查及独立调度计算。调度计算检查样本覆盖、每 rank 容量、子组对齐和 VPP 微批数。多卡通信、前反向数值及吞吐尚待在实际 DCU/HIP 训练栈验证。

合并后的启动脚本已通过 Bash 语法检查和命令预览，覆盖单机预训练、静态 SFT，以及两机 hostfile 下的 MPI worker 和 torchrun 两种启动方式。
