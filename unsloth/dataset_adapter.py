# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Universal Dataset Adapter for LLM and Diffusion Training.

Automatically detects, validates, cleans, and adapts ANY input dataset
(HuggingFace Dataset/DatasetDict, Hub repo ID, local JSON/JSONL/CSV/Parquet/TXT,
pandas DataFrame, list of dicts) into the canonical representation required
by the selected training method:
- SFT (Supervised Fine-Tuning / Masked Denoising SFT)
- DPO (Direct Preference Optimization)
- ORPO (Odds Ratio Preference Optimization)
- GRPO (Group Relative Policy Optimization)
- KTO (Kahneman-Tversky Optimization)
- Pretraining / Continued Pretraining

Adheres to OOP, SOLID, KISS, DRY, and BDUF principles.
"""

from __future__ import annotations
import os
import json
import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

__all__ = [
    "DatasetFormat",
    "TrainingMethod",
    "UniversalDatasetAdapter",
    "DatasetSchemaDetector",
    "DatasetCleaner",
]


class DatasetFormat(str, Enum):
    """Recognized dataset formats across the open-source ecosystem."""
    CHATML = "chatml"                      # {"messages": [{"role": ..., "content": ...}]}
    SHAREGPT = "sharegpt"                  # {"conversations": [{"from": ..., "value": ...}]}
    ALPACA = "alpaca"                      # {"instruction": ..., "input": ..., "output": ...}
    PROMPT_COMPLETION = "prompt_completion"# {"prompt": ..., "completion": ...}
    PREFERENCE_TRIPLET = "dpo_triplet"     # {"prompt": ..., "chosen": ..., "rejected": ...}
    PREFERENCE_CONVERSATIONAL = "dpo_conv" # {"chosen": [...], "rejected": [...]}
    KTO_LABELED = "kto_labeled"            # {"prompt": ..., "completion": ..., "label": bool}
    GRPO_PROMPTS = "grpo_prompts"          # {"prompt": ...} or {"question": ...}
    RAW_TEXT = "raw_text"                  # {"text": ...} or {"content": ...}
    MULTIMODAL = "multimodal"              # includes image/video paths/tensors
    UNKNOWN = "unknown"


class TrainingMethod(str, Enum):
    """Supported training methodologies."""
    SFT = "sft"
    DPO = "dpo"
    ORPO = "orpo"
    GRPO = "grpo"
    KTO = "kto"
    PRETRAIN = "pretrain"


class DatasetSchemaDetector:
    """Introspects dataset structure to identify format and column mappings."""

    @staticmethod
    def detect(sample_record: Dict[str, Any]) -> Tuple[DatasetFormat, Dict[str, str]]:
        """
        Inspect a sample record and return detected format + column mapping.
        """
        keys = set(k.lower() for k in sample_record.keys())
        original_keys = {k.lower(): k for k in sample_record.keys()}

        # 1. Check for Multimodal
        has_media = any(k in keys for k in ("image", "images", "video", "videos", "pixel_values"))

        # 2. Preference Conversational: chosen and rejected containing lists of messages
        if "chosen" in keys and "rejected" in keys:
            chosen_val = sample_record.get(original_keys["chosen"])
            if isinstance(chosen_val, list) and len(chosen_val) > 0 and isinstance(chosen_val[0], dict):
                fmt = DatasetFormat.PREFERENCE_CONVERSATIONAL
                mapping = {"chosen": original_keys["chosen"], "rejected": original_keys["rejected"]}
                if "prompt" in keys:
                    mapping["prompt"] = original_keys["prompt"]
                return fmt, mapping

            # Preference Triplet: prompt, chosen, rejected (or text strings)
            fmt = DatasetFormat.PREFERENCE_TRIPLET
            mapping = {"chosen": original_keys["chosen"], "rejected": original_keys["rejected"]}
            if "prompt" in keys:
                mapping["prompt"] = original_keys["prompt"]
            elif "instruction" in keys:
                mapping["prompt"] = original_keys["instruction"]
            elif "question" in keys:
                mapping["prompt"] = original_keys["question"]
            else:
                mapping["prompt"] = None
            return fmt, mapping

        # 3. KTO Labeled: prompt/completion with label (bool/binary)
        if ("label" in keys or "is_desirable" in keys) and ("completion" in keys or "output" in keys or "text" in keys):
            lbl_key = original_keys.get("label") or original_keys.get("is_desirable")
            comp_key = (
                original_keys.get("completion")
                or original_keys.get("output")
                or original_keys.get("response")
                or original_keys.get("text")
            )
            prompt_key = (
                original_keys.get("prompt")
                or original_keys.get("instruction")
                or original_keys.get("question")
            )
            return DatasetFormat.KTO_LABELED, {"prompt": prompt_key, "completion": comp_key, "label": lbl_key}

        # 4. ChatML / OpenAI Messages
        if "messages" in keys:
            val = sample_record.get(original_keys["messages"])
            if isinstance(val, list):
                fmt = DatasetFormat.MULTIMODAL if has_media else DatasetFormat.CHATML
                return fmt, {"messages": original_keys["messages"]}

        # 5. ShareGPT
        if "conversations" in keys or "conversation" in keys:
            conv_key = original_keys.get("conversations") or original_keys.get("conversation")
            val = sample_record.get(conv_key)
            if isinstance(val, list):
                fmt = DatasetFormat.MULTIMODAL if has_media else DatasetFormat.SHAREGPT
                return fmt, {"conversations": conv_key}

        # 6. Alpaca: instruction, [input], output
        if "instruction" in keys and ("output" in keys or "response" in keys):
            out_key = original_keys.get("output") or original_keys.get("response")
            inp_key = original_keys.get("input", None)
            return DatasetFormat.ALPACA, {
                "instruction": original_keys["instruction"],
                "input": inp_key,
                "output": out_key,
            }

        # 7. Prompt / Completion
        prompt_candidates = ("prompt", "question", "input", "query", "problem", "src")
        comp_candidates = ("completion", "output", "response", "answer", "target", "tgt")

        found_prompt = next((original_keys[c] for c in prompt_candidates if c in keys), None)
        found_comp = next((original_keys[c] for c in comp_candidates if c in keys), None)

        if found_prompt and found_comp:
            fmt = DatasetFormat.MULTIMODAL if has_media else DatasetFormat.PROMPT_COMPLETION
            return fmt, {"prompt": found_prompt, "completion": found_comp}

        # 8. GRPO Prompts (only prompt/question present)
        if found_prompt and not found_comp:
            return DatasetFormat.GRPO_PROMPTS, {"prompt": found_prompt}

        # 9. Raw Text / Pretraining
        text_candidates = ("text", "content", "raw_text", "body", "passage")
        found_text = next((original_keys[c] for c in text_candidates if c in keys), None)
        if found_text:
            return DatasetFormat.RAW_TEXT, {"text": found_text}

        return DatasetFormat.UNKNOWN, {}


class DatasetCleaner:
    """Sanitizes strings and filters out corrupted or empty entries."""

    @staticmethod
    def clean_text(text: Any) -> str:
        if text is None:
            return ""
        if not isinstance(text, str):
            text = str(text)
        return text.strip()

    @staticmethod
    def is_valid_message(msg: Any) -> bool:
        if not isinstance(msg, dict):
            return False
        role = msg.get("role") or msg.get("from")
        content = msg.get("content") or msg.get("value")
        return bool(role and content)


class UniversalDatasetAdapter:
    """
    Universal adapter capable of transforming ANY input dataset into the canonical
    format needed for training discrete diffusion or autoregressive models.
    """

    def __init__(
        self,
        tokenizer: Any,
        target_method: Union[str, TrainingMethod] = TrainingMethod.SFT,
        max_length: int = 2048,
        system_prompt: Optional[str] = None,
        canvas_block_size: int = 256,
    ):
        self.tokenizer = tokenizer
        self.target_method = TrainingMethod(str(target_method).lower())
        self.max_length = max_length
        self.system_prompt = system_prompt
        self.canvas_block_size = canvas_block_size

    def load_raw_dataset(self, data: Any) -> Any:
        """Load dataset from various sources (HF Dataset, Hub ID, local path, dict, list)."""
        from datasets import Dataset, DatasetDict, load_dataset

        if isinstance(data, (Dataset, DatasetDict)):
            return data

        if isinstance(data, str):
            # Check if local file
            if os.path.isfile(data):
                ext = os.path.splitext(data)[-1].lower()
                if ext in (".json", ".jsonl"):
                    return load_dataset("json", data_files=data, split="train")
                elif ext in (".csv", ".tsv"):
                    delimiter = "\t" if ext == ".tsv" else ","
                    return load_dataset("csv", data_files=data, delimiter=delimiter, split="train")
                elif ext == ".parquet":
                    return load_dataset("parquet", data_files=data, split="train")
                elif ext == ".txt":
                    return load_dataset("text", data_files=data, split="train")
                else:
                    raise ValueError(f"Unsupported file format: {ext}")
            else:
                # Assume Hugging Face Hub dataset repo ID
                logger.info(f"Loading dataset from HuggingFace Hub: {data}")
                return load_dataset(data, split="train")

        if isinstance(data, list):
            return Dataset.from_list(data)

        if isinstance(data, dict):
            return Dataset.from_dict(data)

        # Pandas DataFrame support
        try:
            import pandas as pd
            if isinstance(data, pd.DataFrame):
                return Dataset.from_pandas(data)
        except ImportError:
            pass

        raise TypeError(f"Cannot load dataset of type {type(data)}")

    def adapt(
        self,
        dataset: Any,
        formatting_func: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        num_proc: Optional[int] = None,
        batched: bool = False,
    ) -> Any:
        """
        Main entry point: adapts any dataset to the target training method.
        """
        raw_ds = self.load_raw_dataset(dataset)

        # If custom formatting_func provided, apply it first
        if formatting_func is not None:
            logger.info("Applying user-provided formatting_func before automatic adaptation.")
            raw_ds = raw_ds.map(formatting_func, batched=batched)

        # Sample first record to detect format
        sample = raw_ds[0] if len(raw_ds) > 0 else {}
        detected_format, col_mapping = DatasetSchemaDetector.detect(sample)
        logger.info(
            f"UniversalDatasetAdapter: Detected format='{detected_format.value}' "
            f"for target_method='{self.target_method.value}'"
        )

        # Route to appropriate transformation based on target method
        if self.target_method == TrainingMethod.SFT:
            return self._adapt_for_sft(raw_ds, detected_format, col_mapping, num_proc)
        elif self.target_method in (TrainingMethod.DPO, TrainingMethod.ORPO):
            return self._adapt_for_preference(raw_ds, detected_format, col_mapping, num_proc)
        elif self.target_method == TrainingMethod.GRPO:
            return self._adapt_for_grpo(raw_ds, detected_format, col_mapping, num_proc)
        elif self.target_method == TrainingMethod.KTO:
            return self._adapt_for_kto(raw_ds, detected_format, col_mapping, num_proc)
        elif self.target_method == TrainingMethod.PRETRAIN:
            return self._adapt_for_pretrain(raw_ds, detected_format, col_mapping, num_proc)
        else:
            raise NotImplementedError(f"Unsupported training method: {self.target_method}")

    # -----------------------------------------------------------------
    # Adapters for Specific Methods
    # -----------------------------------------------------------------

    def _adapt_for_sft(
        self, dataset: Any, fmt: DatasetFormat, mapping: Dict[str, str], num_proc: Optional[int]
    ) -> Any:
        """Transforms data into tokenized input_ids + prompt_length for diffusion SFT."""
        tokenizer = self.tokenizer

        def sft_transform(example):
            messages = self._to_messages(example, fmt, mapping)
            if not messages:
                return {"input_ids": [], "prompt_length": 0}

            # Separate prompt messages from final assistant response
            # Prompt = all messages up to the last user turn; Response = final assistant turn
            if messages[-1]["role"] == "assistant":
                prompt_messages = messages[:-1]
                response_text = messages[-1]["content"]
            else:
                prompt_messages = messages
                response_text = ""

            # Apply chat template
            if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
                try:
                    # Tokenize prompt only with add_generation_prompt=True
                    prompt_ids = tokenizer.apply_chat_template(
                        prompt_messages,
                        add_generation_prompt=True,
                        tokenize=True,
                    )
                    # Full sequence
                    full_ids = tokenizer.apply_chat_template(
                        messages,
                        add_generation_prompt=False,
                        tokenize=True,
                    )
                except Exception:
                    # Fallback to plain text concatenation
                    prompt_str = self._format_messages_plain(prompt_messages)
                    full_str = prompt_str + "\n" + response_text
                    prompt_ids = tokenizer.encode(prompt_str, add_special_tokens=True)
                    full_ids = tokenizer.encode(full_str, add_special_tokens=True)
            else:
                prompt_str = self._format_messages_plain(prompt_messages)
                full_str = prompt_str + "\n" + response_text
                prompt_ids = tokenizer.encode(prompt_str, add_special_tokens=True)
                full_ids = tokenizer.encode(full_str, add_special_tokens=True)

            prompt_len = len(prompt_ids)
            if len(full_ids) > self.max_length:
                full_ids = full_ids[: self.max_length]
                prompt_len = min(prompt_len, len(full_ids))

            return {
                "input_ids": full_ids,
                "prompt_length": prompt_len,
            }

        transformed = dataset.map(sft_transform, remove_columns=dataset.column_names, num_proc=num_proc)
        # Filter out empty sequences
        return transformed.filter(lambda x: len(x["input_ids"]) > 0)

    def _adapt_for_preference(
        self, dataset: Any, fmt: DatasetFormat, mapping: Dict[str, str], num_proc: Optional[int]
    ) -> Any:
        """Adapts for DPO / ORPO: (prompt, chosen, rejected) tokenization."""
        tokenizer = self.tokenizer

        def preference_transform(example):
            prompt_str, chosen_str, rejected_str = self._extract_preference_triplet(example, fmt, mapping)

            # Tokenize chosen and rejected
            chosen_full = f"{prompt_str}\n{chosen_str}".strip()
            rejected_full = f"{prompt_str}\n{rejected_str}".strip()

            chosen_ids = tokenizer.encode(chosen_full, add_special_tokens=True)[: self.max_length]
            rejected_ids = tokenizer.encode(rejected_full, add_special_tokens=True)[: self.max_length]
            prompt_ids = tokenizer.encode(prompt_str, add_special_tokens=True)
            p_len = len(prompt_ids)

            return {
                "chosen_input_ids": chosen_ids,
                "chosen_prompt_length": min(p_len, len(chosen_ids)),
                "rejected_input_ids": rejected_ids,
                "rejected_prompt_length": min(p_len, len(rejected_ids)),
                "prompt": prompt_str,
                "chosen": chosen_str,
                "rejected": rejected_str,
            }

        transformed = dataset.map(preference_transform, remove_columns=dataset.column_names, num_proc=num_proc)
        return transformed.filter(
            lambda x: len(x["chosen_input_ids"]) > 0 and len(x["rejected_input_ids"]) > 0
        )

    def _adapt_for_grpo(
        self, dataset: Any, fmt: DatasetFormat, mapping: Dict[str, str], num_proc: Optional[int]
    ) -> Any:
        """Adapts for GRPO: extracts prompt / query for rollouts."""
        def grpo_transform(example):
            if fmt == DatasetFormat.GRPO_PROMPTS:
                p = example.get(mapping.get("prompt", "prompt"), "")
            elif fmt in (DatasetFormat.ALPACA, DatasetFormat.PROMPT_COMPLETION):
                p = example.get(mapping.get("instruction") or mapping.get("prompt"), "")
                inp = example.get(mapping.get("input", ""), "")
                if inp:
                    p = f"{p}\n\nInput: {inp}"
            elif fmt in (DatasetFormat.CHATML, DatasetFormat.SHAREGPT):
                msgs = self._to_messages(example, fmt, mapping)
                p = "\n".join([f"{m['role']}: {m['content']}" for m in msgs if m["role"] == "user"])
            else:
                p = str(next(iter(example.values()), ""))

            return {"prompt": DatasetCleaner.clean_text(p)}

        transformed = dataset.map(grpo_transform, remove_columns=dataset.column_names, num_proc=num_proc)
        return transformed.filter(lambda x: len(x["prompt"]) > 0)

    def _adapt_for_kto(
        self, dataset: Any, fmt: DatasetFormat, mapping: Dict[str, str], num_proc: Optional[int]
    ) -> Any:
        """Adapts for KTO: (prompt, completion, label)."""
        tokenizer = self.tokenizer

        def kto_transform(example):
            prompt = DatasetCleaner.clean_text(example.get(mapping.get("prompt", "prompt"), ""))
            comp = DatasetCleaner.clean_text(example.get(mapping.get("completion", "completion"), ""))
            lbl = bool(example.get(mapping.get("label", "label"), True))

            full_text = f"{prompt}\n{comp}".strip()
            ids = tokenizer.encode(full_text, add_special_tokens=True)[: self.max_length]
            p_len = len(tokenizer.encode(prompt, add_special_tokens=True))

            return {
                "input_ids": ids,
                "prompt_length": min(p_len, len(ids)),
                "label": lbl,
            }

        transformed = dataset.map(kto_transform, remove_columns=dataset.column_names, num_proc=num_proc)
        return transformed.filter(lambda x: len(x["input_ids"]) > 0)

    def _adapt_for_pretrain(
        self, dataset: Any, fmt: DatasetFormat, mapping: Dict[str, str], num_proc: Optional[int]
    ) -> Any:
        """Adapts raw text corpora for continuous ELBO pretraining / MLM."""
        tokenizer = self.tokenizer
        text_col = mapping.get("text", "text")

        def pretrain_transform(example):
            raw = example.get(text_col, "")
            ids = tokenizer.encode(raw, add_special_tokens=True)[: self.max_length]
            return {
                "input_ids": ids,
                "prompt_length": 0,  # Entire sequence can be masked in pretrain
            }

        transformed = dataset.map(pretrain_transform, remove_columns=dataset.column_names, num_proc=num_proc)
        return transformed.filter(lambda x: len(x["input_ids"]) > 0)

    # -----------------------------------------------------------------
    # Helper Utilities
    # -----------------------------------------------------------------

    def _to_messages(
        self, example: Dict[str, Any], fmt: DatasetFormat, mapping: Dict[str, str]
    ) -> List[Dict[str, str]]:
        """Normalize any sample into a list of {'role': ..., 'content': ...} dicts."""
        messages: List[Dict[str, str]] = []

        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})

        if fmt == DatasetFormat.CHATML:
            raw_msgs = example.get(mapping.get("messages", "messages"), [])
            for m in raw_msgs:
                if DatasetCleaner.is_valid_message(m):
                    messages.append({"role": m["role"], "content": DatasetCleaner.clean_text(m["content"])})

        elif fmt == DatasetFormat.SHAREGPT:
            raw_convs = example.get(mapping.get("conversations", "conversations"), [])
            role_map = {"human": "user", "gpt": "assistant", "system": "system", "chatgpt": "assistant"}
            for turn in raw_convs:
                frm = turn.get("from", "").lower()
                val = turn.get("value", "")
                role = role_map.get(frm, "user")
                if val:
                    messages.append({"role": role, "content": DatasetCleaner.clean_text(val)})

        elif fmt == DatasetFormat.ALPACA:
            inst = DatasetCleaner.clean_text(example.get(mapping.get("instruction", "instruction"), ""))
            inp = DatasetCleaner.clean_text(example.get(mapping.get("input", "input"), ""))
            out = DatasetCleaner.clean_text(example.get(mapping.get("output", "output"), ""))
            content = f"{inst}\n\nInput: {inp}" if inp else inst
            messages.append({"role": "user", "content": content})
            messages.append({"role": "assistant", "content": out})

        elif fmt == DatasetFormat.PROMPT_COMPLETION:
            p = DatasetCleaner.clean_text(example.get(mapping.get("prompt", "prompt"), ""))
            c = DatasetCleaner.clean_text(example.get(mapping.get("completion", "completion"), ""))
            messages.append({"role": "user", "content": p})
            messages.append({"role": "assistant", "content": c})

        elif fmt == DatasetFormat.RAW_TEXT:
            t = DatasetCleaner.clean_text(example.get(mapping.get("text", "text"), ""))
            messages.append({"role": "user", "content": t})

        return messages

    def _extract_preference_triplet(
        self, example: Dict[str, Any], fmt: DatasetFormat, mapping: Dict[str, str]
    ) -> Tuple[str, str, str]:
        """Extracts (prompt, chosen, rejected) text strings."""
        if fmt == DatasetFormat.PREFERENCE_CONVERSATIONAL:
            chosen_msgs = example.get(mapping.get("chosen", "chosen"), [])
            rejected_msgs = example.get(mapping.get("rejected", "rejected"), [])
            # Extract prompt from first turns
            prompt = "\n".join([f"{m['role']}: {m['content']}" for m in chosen_msgs[:-1]])
            chosen = chosen_msgs[-1]["content"] if chosen_msgs else ""
            rejected = rejected_msgs[-1]["content"] if rejected_msgs else ""
            return prompt, chosen, rejected

        prompt = DatasetCleaner.clean_text(example.get(mapping.get("prompt") or "prompt", ""))
        chosen = DatasetCleaner.clean_text(example.get(mapping.get("chosen") or "chosen", ""))
        rejected = DatasetCleaner.clean_text(example.get(mapping.get("rejected") or "rejected", ""))
        return prompt, chosen, rejected

    def _format_messages_plain(self, messages: List[Dict[str, str]]) -> str:
        """Format messages as plain text if tokenizer has no chat template."""
        return "\n".join([f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>" for m in messages])
