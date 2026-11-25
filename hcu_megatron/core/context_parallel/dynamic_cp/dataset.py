# Copyright (c) 2025-2026, NVIDIA CORPORATION. All rights reserved.

import csv
import json
import os
from functools import wraps
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch

from megatron.core import parallel_state as mpu
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.training import get_args
from megatron.training.datasets.sft_dataset import IGNORE_INDEX, SFTDataset, SFTLowLevelDataset

# Field-name synonyms (probed in order; first non-empty wins).
_INSTRUCTION_FIELDS: Tuple[str, ...] = (
    "instruction", "prompt", "query", "question",
)
_OUTPUT_FIELDS: Tuple[str, ...] = (
    "output", "response", "completion", "answer",
)
# Supplementary user-turn context: Stanford Alpaca's "input", Dolly's "context".
_EXTRA_INPUT_FIELDS: Tuple[str, ...] = ("input", "context")

# ShareGPT "from" value -> chat-template "role". Unknown values fall back to
# "user" so downstream tokenization does not crash on unfamiliar speakers.
_SHAREGPT_ROLE_MAP: Dict[str, str] = {
    "human": "user",
    "user": "user",
    "gpt": "assistant",
    "assistant": "assistant",
    "model": "assistant",
    "chatgpt": "assistant",
    "bing": "assistant",
    "bard": "assistant",
    "system": "system",
    "tool": "tool",
    "function": "tool",
    "observation": "tool",
}


def _looks_like_hf_id(path: str) -> bool:
    """Heuristic: does ``path`` look like an ``owner/repo`` HF dataset id?

    True iff ``path`` contains ``/``, is not an absolute/relative file path,
    and does not exist on the local filesystem.
    """
    if not path:
        return False
    if os.path.exists(path):
        return False
    if path.startswith(("/", "./", "../")):
        return False
    return "/" in path


def _first_present(
    sample: Dict[str, Any], fields: Iterable[str]
) -> Optional[str]:
    """Return the first non-empty string value among the given fields, or None."""
    for f in fields:
        v = sample.get(f)
        if v in (None, ""):
            continue
        if not isinstance(v, str):
            raise ValueError(
                f"VarlenDataset: field '{f}' must be a string, "
                f"got {type(v).__name__}."
            )
        return v
    return None


def _ensure_str_content(content: Any, where: str) -> str:
    """Validate that a turn's content is a plain string (reject multi-modal lists)."""
    if content is None:
        return ""
    if not isinstance(content, str):
        raise ValueError(
            f"VarlenDataset: {where} content must be a string, "
            f"got {type(content).__name__}. Multi-modal datasets (e.g. "
            "content as a list of image/text parts) are not supported."
        )
    return content


def _alpaca_to_messages(sample: Dict[str, Any]) -> List[Dict[str, str]]:
    """Convert an Alpaca/Dolly-style sample to a 3-turn messages list."""
    instruction = _first_present(sample, _INSTRUCTION_FIELDS) or ""
    extra_input = _first_present(sample, _EXTRA_INPUT_FIELDS) or ""
    output = _first_present(sample, _OUTPUT_FIELDS) or ""
    user_content = (
        f"{instruction}\n\n{extra_input}" if extra_input else instruction
    )
    return [
        {"role": "system", "content": ""},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": output},
    ]


def _sharegpt_to_messages(sample: Dict[str, Any]) -> List[Dict[str, str]]:
    """Convert a ShareGPT ``conversations`` sample to a messages list.

    Prepends an empty ``system`` turn unless the conversation already starts
    with one, so ``SFTDataset._split_conversations`` treats the sample as a
    single conversation.
    """
    conv = sample.get("conversations") or []
    out: List[Dict[str, str]] = []
    first_speaker = (conv[0].get("from") or "").lower() if conv else ""
    if first_speaker != "system":
        out.append({"role": "system", "content": ""})
    for turn in conv:
        speaker = (turn.get("from") or "").lower()
        role = _SHAREGPT_ROLE_MAP.get(speaker, "user")
        content = _ensure_str_content(turn.get("value"), f"sharegpt turn role={role}")
        out.append({"role": role, "content": content})
    return out


def _messages_passthrough(sample: Dict[str, Any]) -> List[Dict[str, str]]:
    """Pass through an OpenAI ``messages`` sample, ensuring a leading system turn.

    Strips any keys other than ``role``/``content`` (e.g. ``name``,
    ``tool_calls``) since they are not part of the chat-template input
    expected by SFTTokenizer.
    """
    raw = list(sample.get("messages") or [])
    if raw and raw[0].get("role") != "system":
        raw = [{"role": "system", "content": ""}] + raw
    out: List[Dict[str, str]] = []
    for m in raw:
        role = m.get("role") or "user"
        content = _ensure_str_content(m.get("content"), f"messages turn role={role}")
        out.append({"role": role, "content": content})
    return out


def _raw_text_loader(sample: Dict[str, Any]) -> str:
    """Return the ``text`` column unchanged for pretrain-style packed runs.

    Unlike the SFT schemas this returns a plain string (no messages list).
    :class:`VarlenDataset.__getitem__` dispatches on the return type to pick
    a tokenization path that skips chat templating and prompt masking.
    """
    text = sample.get("text") or ""
    if not isinstance(text, str):
        raise ValueError(
            f"VarlenDataset (pretrain-text schema): 'text' must be a string, "
            f"got {type(text).__name__}."
        )
    return text


def _select_converter(
    column_names: List[str],
) -> Tuple[Callable[[Dict[str, Any]], Any], str]:
    """Pick a sample converter based on dataset column names.

    Priority (most explicit first): openai-messages > sharegpt > alpaca/dolly
    > pretrain-text. ``pretrain-text`` is the fallback for datasets that
    only carry a single ``text`` column (e.g. Dolma / OLMo midtraining
    corpora) — long-context pretraining packed through the same THD path
    as SFT.
    """
    cols = set(column_names)
    if "messages" in cols:
        return _messages_passthrough, "openai-messages"
    if "conversations" in cols:
        return _sharegpt_to_messages, "sharegpt"
    has_instr = any(f in cols for f in _INSTRUCTION_FIELDS)
    has_out = any(f in cols for f in _OUTPUT_FIELDS)
    if has_instr and has_out:
        return _alpaca_to_messages, "alpaca"
    if "text" in cols:
        return _raw_text_loader, "pretrain-text"
    raise ValueError(
        "VarlenDataset cannot infer schema from columns "
        f"{sorted(cols)}. Supported schemas: "
        f"alpaca/dolly ({'|'.join(_INSTRUCTION_FIELDS)} + "
        f"{'|'.join(_OUTPUT_FIELDS)} [+ optional {'|'.join(_EXTRA_INPUT_FIELDS)}]), "
        "sharegpt (conversations), openai-messages (messages), "
        "pretrain-text (text)."
    )


class VarlenLowLevelDataset(SFTLowLevelDataset):
    """Low-level loader: HF Hub repo / local parquet / local jsonl, normalized.

    Dataset path interpretation:

      * HF Hub repo id (e.g. ``Yukang/LongAlpaca-12k``) — contains ``/`` and
        does not exist on the local filesystem; loaded via
        ``datasets.load_dataset(path, split="train")``.
      * Local ``.parquet`` — loaded via
        ``datasets.load_dataset("parquet", data_files=path, split="all")``;
        parquet's footer schema makes chunked loading safe.
      * Otherwise local jsonl/json — parsed line-by-line with the stdlib
        ``json`` module and built with ``Dataset.from_list``. We avoid
        ``datasets.load_dataset("json", ...)`` for local files because its
        pyarrow-based JSON reader infers schema per parallel chunk and fails
        with ``CastError`` when the union of fields varies between rows
        (e.g. LongAlpaca-12k); ``from_list`` unifies the schema over the
        whole file in one pass.

    A per-sample converter is selected once at construction time based on
    column names and applied at access time. The instruction-tuning schemas
    convert to a messages list; the ``pretrain-text`` fallback returns the raw
    string instead.
    """

    def __init__(self, dataset_path: str) -> None:
        try:
            from datasets import Dataset, load_dataset
        except ImportError as exc:
            raise ImportError(
                "VarlenDataset requires the `datasets` library "
                "(pip install datasets)."
            ) from exc

        if _looks_like_hf_id(dataset_path):
            self.dataset = load_dataset(dataset_path, split="train")
        elif dataset_path.endswith(".parquet"):
            self.dataset = load_dataset(
                "parquet", data_files=dataset_path, split="all"
            )
        else:
            # Parse the jsonl with the stdlib and build the table in one
            # pyarrow pass: datasets.load_dataset("json", ...) infers schema
            # per parallel chunk and fails with CastError when the union of
            # fields varies between rows (e.g. LongAlpaca-12k); building from
            # the full record list sidesteps that without a pandas dependency.
            with open(dataset_path, encoding="utf-8") as f:
                if dataset_path.endswith(".json"):
                    records = json.load(f)
                    if isinstance(records, dict):
                        records = [records]
                else:
                    records = [json.loads(line) for line in f if line.strip()]
            self.dataset = Dataset.from_list(records)

        self._converter, self._schema_name = _select_converter(
            list(self.dataset.column_names)
        )

    @property
    def schema_name(self) -> str:
        """Detected schema name: ``alpaca`` / ``sharegpt`` / ``openai-messages`` /
        ``pretrain-text`` (the raw ``text``-column fallback)."""
        return self._schema_name

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        return self._converter(self.dataset[idx])


class PackedTextDataset(SFTDataset):
    """Expose each document separately so the schedule can place it on its own CP group."""

    @staticmethod
    def build_low_level_dataset(dataset_path, config):
        if config.sft:
            return SFTLowLevelDataset(dataset_path)
        return VarlenLowLevelDataset(dataset_path)

    def __len__(self):
        if len(self.indices) == 0:
            return 0
        return self.num_samples if self.num_samples is not None else len(self.indices)

    def _sample(self, ids, targets, max_length):
        eod = self.config.tokenizer.eod
        if eod is None:
            raise ValueError("Variable-length packing requires a tokenizer EOD token")
        ids = list(ids)
        targets = list(targets)
        if not ids:
            ids, targets = [eod], [eod]
        if len(ids) > max_length + 1:
            ids, targets = ids[:max_length] + [eod], targets[:max_length] + [eod]
        elif ids[-1] != eod or len(ids) == 1:
            if len(ids) == max_length + 1:
                ids[-1], targets[-1] = eod, eod
            else:
                ids.append(eod)
                targets.append(eod)
        length = len(ids) - 1
        physical = length + (-length % self.config.packing_alignment)
        # Zero is always a legal embedding/vocabulary index. Mask only padding
        # positions and ignored targets, even when EOS is also the pad token.
        tokens = torch.zeros(physical, dtype=torch.long)
        tokens[:length] = torch.as_tensor(ids[:-1], dtype=torch.long)
        labels = torch.zeros_like(tokens)
        targets = torch.as_tensor(targets[1:], dtype=torch.long)
        supervised = targets != IGNORE_INDEX
        labels[:length] = targets.masked_fill(~supervised, 0)
        loss_mask = torch.zeros(physical, dtype=torch.float32)
        loss_mask[:length] = supervised
        return {
            "tokens": tokens, "labels": labels, "loss_mask": loss_mask,
            "position_ids": torch.arange(physical, dtype=torch.long).clamp_max_(length - 1),
            "original_seq_len": torch.tensor([length], dtype=torch.int32),
            "padded_seq_len": torch.tensor([physical], dtype=torch.int32),
        }

    def __getitem__(self, idx):
        item = self.dataset[int(self.indices[idx % len(self.indices)])]
        limit = self.config.sequence_length
        tokenizer = self.config.tokenizer
        if isinstance(item, str):
            ids = tokenizer.tokenize(item)
            return self._sample(ids, ids, limit)

        conversations = self._split_conversations(item) if self.config.sft else [item]
        documents = []
        for conversation in conversations:
            ids, targets = tokenizer.tokenize_conversation(
                conversation, return_target=True, add_generation_prompt=False,
            )
            sample = self._sample(ids.tolist(), targets.tolist(), limit)
            documents.append(sample)
            limit -= int(sample["original_seq_len"].item())
            if limit <= 0:
                break
        if not documents:
            raise ValueError(f"Empty conversation list in dataset row {idx}")
        return {"documents": documents}


class MockPackedTextDataset(PackedTextDataset):
    @staticmethod
    def build_low_level_dataset(dataset_path, config):
        spec = config.mock_lengths_spec
        if spec is None:
            options = {
                "mode": "distribution", "type": "lognormal",
                "min_seq_len": max(2, config.sequence_length // 2),
                "max_seq_len": max(2, config.sequence_length),
                "mean_seq_len": max(2, 3 * config.sequence_length // 4), "lognormal_sigma": 1.1,
            }
        elif os.path.isfile(spec):
            with open(spec, encoding="utf-8") as source:
                options = json.load(source)
        else:
            options = json.loads(spec)
        if options["mode"] == "file":
            lengths = []
            with open(options["path"], newline="", encoding="utf-8") as source:
                for row in csv.reader(source):
                    for cell in row:
                        try:
                            lengths.append(int(float(cell)))
                        except ValueError:
                            continue  # CSV headers
            if not lengths or min(lengths) < 1:
                raise ValueError("Mock length file must contain positive sequence lengths")
            return np.asarray(lengths)
        if options["mode"] != "distribution" or options["type"] != "lognormal":
            raise ValueError("Mock lengths require file or lognormal distribution mode")
        sigma = options["lognormal_sigma"]
        mu = np.log(options["mean_seq_len"]) - sigma ** 2 / 2
        lengths = np.random.RandomState(0).lognormal(mu, sigma, size=1_000_000)
        return np.clip(lengths, options["min_seq_len"], options["max_seq_len"]).astype(np.int64)

    def __getitem__(self, idx):
        length = int(self.dataset[self.indices[idx % len(self.indices)]])
        # The reference length includes the terminal EOD. Keep generated ids
        # valid with small synthetic vocabularies as well.
        length = min(max(2, length), self.config.sequence_length + 1)
        ids = np.arange(1, length) % self.config.tokenizer.vocab_size
        tokens = [*ids.tolist(), self.config.tokenizer.eod]
        return self._sample(tokens, tokens, self.config.sequence_length)


def make_dataset_provider(build_config):
    """Use the target GPT tokenizer, split and blended-dataset configuration."""
    def provider(train_val_test_num_samples, vp_stage=None):
        args = get_args()
        config = build_config(args)
        config.sft = args.sft
        config.mock_lengths_spec = (
            args.sft_mock_dataset_config_json if args.sft else args.varlen_mock_dataset_config_json
        )
        # GPT's fast path skips split indices entirely. These datasets index
        # source documents directly and therefore require the builder's indices.
        config.fast_cache_load = False
        cp = config.context_parallel_size
        if args.dynamic_context_parallel:
            cp *= config.data_parallel_size
        config.packing_alignment = (2 * cp if cp > 1 else 1) * max(1, config.sequence_parallel_size)
        return BlendedMegatronDatasetBuilder(
            MockPackedTextDataset if args.mock_data else PackedTextDataset,
            train_val_test_num_samples,
            lambda: mpu.get_tensor_model_parallel_rank() == 0,
            config,
        ).build()

    provider.is_distributed = True
    return provider


def _collate_documents(items):
    documents = []
    for item in items:
        if "documents" in item:
            documents.extend(item["documents"])
        else:
            documents.append({k: v for k, v in item.items() if k != "dataset_id"})
    return documents


def install_data_loader():
    """Retain the target sampler, consumed-sample offset, worker and eval settings."""
    from megatron.training import training
    from megatron.training.datasets import data_samplers

    original = data_samplers.build_pretraining_data_loader

    @wraps(original)
    def build_loader(dataset, consumed_samples):
        if dataset is None or len(dataset) == 0:
            return None
        loader = original(dataset, consumed_samples)
        loader.collate_fn = _collate_documents
        loader.generator = torch.Generator().manual_seed(torch.initial_seed())
        return loader

    data_samplers.build_pretraining_data_loader = build_loader
    training.build_pretraining_data_loader = build_loader
