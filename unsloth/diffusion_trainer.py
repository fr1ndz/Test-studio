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
Discrete Diffusion Trainers for DiffusionGemma and Text-Diffusion Architectures.

Provides:
- DiffusionTrainingArguments (with hyperparameters for noise schedules, canvas sizes, DPO, GRPO, ORPO, KTO)
- DiffusionTrainer (Unified trainer handling SFT, DPO, ORPO, GRPO, KTO, Pretrain)
- DiffusionSFTTrainer
- DiffusionDPOTrainer
- DiffusionORPOTrainer
- DiffusionGRPOTrainer
- DiffusionKTOTrainer
"""

from __future__ import annotations
import copy
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import Trainer, TrainingArguments
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalPrediction

from .diffusion_engine import (
    BaseNoiseSchedule,
    CosineNoiseSchedule,
    DiffusionDataCollator,
    DiffusionDPOLoss,
    DiffusionGRPOEngine,
    DiffusionKTOLoss,
    DiffusionLoss,
    DiffusionORPOLoss,
    NoiseScheduleType,
    get_noise_schedule,
)
from .dataset_adapter import (
    DatasetFormat,
    TrainingMethod,
    UniversalDatasetAdapter,
)

logger = logging.getLogger(__name__)

__all__ = [
    "DiffusionTrainingArguments",
    "DiffusionTrainer",
    "DiffusionSFTTrainer",
    "DiffusionDPOTrainer",
    "DiffusionORPOTrainer",
    "DiffusionGRPOTrainer",
    "DiffusionKTOTrainer",
]


@dataclass
class DiffusionTrainingArguments(TrainingArguments):
    """
    Hyperparameters specifically designed for discrete diffusion language models.
    Configured with memory-efficient defaults for large (26B+) models.
    """
    optim: str = field(
        default="paged_adamw_8bit",
        metadata={"help": "The optimizer to use. Default 'paged_adamw_8bit' saves ~75% optimizer VRAM."},
    )
    gradient_checkpointing: bool = field(
        default=True,
        metadata={"help": "Enable gradient checkpointing to drastically reduce activation VRAM."},
    )
    per_device_train_batch_size: int = field(
        default=1,
        metadata={"help": "Batch size per GPU. Defaults to 1 for large (26B+) models."},
    )
    gradient_accumulation_steps: int = field(
        default=8,
        metadata={"help": "Number of updates steps to accumulate before backward/update."},
    )
    training_method: str = field(
        default="sft",
        metadata={"help": "Training methodology: 'sft', 'dpo', 'orpo', 'grpo', 'kto', 'pretrain'."},
    )
    noise_schedule: str = field(
        default="cosine",
        metadata={"help": "Noise schedule for discrete masking: 'cosine', 'linear', 'sqrt', 'log_linear'."},
    )
    min_mask_rate: float = field(
        default=0.05,
        metadata={"help": "Minimum mask probability at t=0."},
    )
    max_mask_rate: float = field(
        default=0.95,
        metadata={"help": "Maximum mask probability at t=1."},
    )
    canvas_block_size: int = field(
        default=256,
        metadata={"help": "Size of block diffusion canvas (e.g. 256 for DiffusionGemma)."},
    )
    mask_prompt: bool = field(
        default=False,
        metadata={"help": "Whether to mask prompt tokens (False = response-only masking for instruction tuning)."},
    )
    # DPO parameters
    beta_dpo: float = field(
        default=0.1,
        metadata={"help": "Temperature beta for Diffusion-DPO."},
    )
    label_smoothing_dpo: float = field(
        default=0.0,
        metadata={"help": "Label smoothing for Diffusion-DPO."},
    )
    # ORPO parameters
    lambda_orpo: float = field(
        default=0.1,
        metadata={"help": "Weight lambda for odds ratio penalty in Diffusion-ORPO."},
    )
    # KTO parameters
    beta_kto: float = field(
        default=0.1,
        metadata={"help": "Temperature beta for Diffusion-KTO."},
    )
    desirable_weight_kto: float = field(
        default=1.0,
        metadata={"help": "Weight for desirable samples in Diffusion-KTO."},
    )
    undesirable_weight_kto: float = field(
        default=1.0,
        metadata={"help": "Weight for undesirable samples in Diffusion-KTO."},
    )
    # GRPO parameters
    num_generations_grpo: int = field(
        default=4,
        metadata={"help": "Number of completions sampled per prompt in Diffusion-GRPO."},
    )
    clip_range_grpo: float = field(
        default=0.2,
        metadata={"help": "PPO/GRPO clipping range epsilon."},
    )
    # Active MoE parameters
    moe_active_training: bool = field(
        default=True,
        metadata={"help": "Enable active parameter MoE training with cyclic cluster rotation."},
    )
    moe_num_clusters: int = field(
        default=4,
        metadata={"help": "Number of expert clusters (e.g. 4 clusters of 32 experts for 128 experts)."},
    )
    moe_cluster_switch_steps: int = field(
        default=250,
        metadata={"help": "Number of training steps per cluster before rotating."},
    )


class DiffusionTrainer(Trainer):
    """
    Unified Trainer for discrete diffusion language models (such as DiffusionGemma).

    Handles forward passes with masked inputs, ELBO loss computation,
    preference optimization (DPO/ORPO), policy gradient rollouts (GRPO),
    and unpaired feedback (KTO).
    """

    def __init__(
        self,
        model: Any = None,
        args: Optional[DiffusionTrainingArguments] = None,
        data_collator: Optional[Any] = None,
        train_dataset: Optional[Any] = None,
        eval_dataset: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
        model_init: Optional[Callable[[], Any]] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        # Diffusion-specific extensions
        ref_model: Optional[Any] = None,
        reward_functions: Optional[List[Callable[..., List[float]]]] = None,
        auto_adapt_dataset: bool = True,
        max_length: int = 2048,
    ):
        if args is None:
            args = DiffusionTrainingArguments(output_dir="./diffusion_outputs")

        self.training_method = TrainingMethod(str(args.training_method).lower())
        self.ref_model = ref_model

        # Auto-adapt dataset if requested and raw dataset provided
        if auto_adapt_dataset and train_dataset is not None and tokenizer is not None:
            adapter = UniversalDatasetAdapter(
                tokenizer=tokenizer,
                target_method=self.training_method,
                max_length=max_length,
                canvas_block_size=args.canvas_block_size,
            )
            logger.info(f"DiffusionTrainer: Auto-adapting train_dataset for method='{self.training_method.value}'")
            train_dataset = adapter.adapt(train_dataset)

            if eval_dataset is not None:
                logger.info(f"DiffusionTrainer: Auto-adapting eval_dataset for method='{self.training_method.value}'")
                eval_dataset = adapter.adapt(eval_dataset)

        # Build default collator if none provided
        if data_collator is None and tokenizer is not None:
            noise_sched = get_noise_schedule(
                schedule=args.noise_schedule,
                min_mask_rate=args.min_mask_rate,
                max_mask_rate=args.max_mask_rate,
            )
            data_collator = DiffusionDataCollator(
                tokenizer=tokenizer,
                noise_schedule=noise_sched,
                max_length=max_length,
                canvas_block_size=args.canvas_block_size,
                mask_prompt=args.mask_prompt,
            )

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

        # Attach MoE cyclic callback if model is using active parameter MoE training
        moe_controller = getattr(model, "_moe_controller", None)
        if moe_controller is None and hasattr(model, "base_model"):
            moe_controller = getattr(model.base_model, "_moe_controller", None)
        if moe_controller is None and hasattr(model, "model"):
            moe_controller = getattr(model.model, "_moe_controller", None)

        if moe_controller is not None:
            from .moe_active import MoECyclicCallback
            self.add_callback(MoECyclicCallback(moe_controller, model))
            logger.info("DiffusionTrainer: Attached MoECyclicCallback for active parameter MoE training.")

        # Initialize loss modules
        self.diffusion_loss_fn = DiffusionLoss()
        self.dpo_loss_fn = DiffusionDPOLoss(
            beta=args.beta_dpo,
            label_smoothing=args.label_smoothing_dpo,
        )
        self.orpo_loss_fn = DiffusionORPOLoss(lambda_orpo=args.lambda_orpo)
        self.kto_loss_fn = DiffusionKTOLoss(
            beta=args.beta_kto,
            desirable_weight=args.desirable_weight_kto,
            undesirable_weight=args.undesirable_weight_kto,
        )
        self.grpo_engine = DiffusionGRPOEngine(
            num_generations=args.num_generations_grpo,
            clip_range=args.clip_range_grpo,
            kl_coeff=args.kl_coeff_grpo,
            reward_functions=reward_functions,
        )

    def _get_logits_from_model(self, model: Any, inputs: Dict[str, Any]) -> torch.Tensor:
        """Extract logits cleanly across varying model signatures and backbones."""
        # Filter inputs to only what model.forward accepts (including labels for selective lm_head)
        model_kwargs = {
            k: v for k, v in inputs.items()
            if k in ("input_ids", "attention_mask", "position_ids", "pixel_values", "pixel_attention_mask", "labels")
        }

        # Forward pass
        outputs = model(**model_kwargs)

        if hasattr(outputs, "logits"):
            return outputs.logits
        elif isinstance(outputs, tuple):
            return outputs[0]
        elif isinstance(outputs, torch.Tensor):
            return outputs
        else:
            raise ValueError(f"Unexpected output structure from diffusion model: {type(outputs)}")

    def compute_loss(
        self,
        model: Any,
        inputs: Dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Any]]:
        """
        Compute loss based on the configured training method.
        """
        if self.training_method in (TrainingMethod.SFT, TrainingMethod.PRETRAIN):
            return self._compute_sft_loss(model, inputs, return_outputs)
        elif self.training_method == TrainingMethod.DPO:
            return self._compute_dpo_loss(model, inputs, return_outputs)
        elif self.training_method == TrainingMethod.ORPO:
            return self._compute_orpo_loss(model, inputs, return_outputs)
        elif self.training_method == TrainingMethod.KTO:
            return self._compute_kto_loss(model, inputs, return_outputs)
        elif self.training_method == TrainingMethod.GRPO:
            return self._compute_grpo_loss(model, inputs, return_outputs)
        else:
            raise NotImplementedError(f"Unsupported training method: {self.training_method}")

    def _compute_sft_loss(
        self, model: Any, inputs: Dict[str, Any], return_outputs: bool
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Any]]:
        """ELBO-consistent Denoising Loss for SFT and Pretraining."""
        labels = inputs.get("labels")
        loss_weights = inputs.get("loss_weights")
        loss_mask = inputs.get("loss_mask")

        model_kwargs = {
            k: v for k, v in inputs.items()
            if k in ("input_ids", "attention_mask", "position_ids", "pixel_values", "pixel_attention_mask", "labels")
        }
        outputs = model(**model_kwargs)
        if hasattr(outputs, "loss") and outputs.loss is not None and loss_weights is None:
            loss = outputs.loss
            logits = outputs.logits
        else:
            logits = outputs.logits if hasattr(outputs, "logits") else (outputs[0] if isinstance(outputs, tuple) else outputs)
            loss = self.diffusion_loss_fn(
                logits=logits,
                labels=labels,
                loss_weights=loss_weights,
                loss_mask=loss_mask,
            )

        return (loss, {"logits": logits}) if return_outputs else loss

    def _compute_dpo_loss(
        self, model: Any, inputs: Dict[str, Any], return_outputs: bool
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Any]]:
        """Diffusion-DPO loss using implicit rewards from denoising ELBO."""
        # Split inputs for chosen and rejected
        chosen_inputs = {k.replace("chosen_", ""): v for k, v in inputs.items() if k.startswith("chosen_")}
        rejected_inputs = {k.replace("rejected_", ""): v for k, v in inputs.items() if k.startswith("rejected_")}

        # Policy forward passes
        chosen_logits = self._get_logits_from_model(model, chosen_inputs)
        rejected_logits = self._get_logits_from_model(model, rejected_inputs)

        policy_chosen_loss = self._compute_samplewise_loss(chosen_logits, chosen_inputs.get("labels"))
        policy_rejected_loss = self._compute_samplewise_loss(rejected_logits, rejected_inputs.get("labels"))

        # Reference forward passes (with no_grad)
        ref_model = self.ref_model if self.ref_model is not None else model
        with torch.no_grad():
            ref_chosen_logits = self._get_logits_from_model(ref_model, chosen_inputs)
            ref_rejected_logits = self._get_logits_from_model(ref_model, rejected_inputs)

            ref_chosen_loss = self._compute_samplewise_loss(ref_chosen_logits, chosen_inputs.get("labels"))
            ref_rejected_loss = self._compute_samplewise_loss(ref_rejected_logits, rejected_inputs.get("labels"))

        loss, chosen_rewards, rejected_rewards = self.dpo_loss_fn(
            policy_chosen_loss=policy_chosen_loss,
            policy_rejected_loss=policy_rejected_loss,
            reference_chosen_loss=ref_chosen_loss,
            reference_rejected_loss=ref_rejected_loss,
        )

        outputs = {
            "chosen_rewards": chosen_rewards,
            "rejected_rewards": rejected_rewards,
        }
        return (loss, outputs) if return_outputs else loss

    def _compute_orpo_loss(
        self, model: Any, inputs: Dict[str, Any], return_outputs: bool
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Any]]:
        """Diffusion-ORPO loss (monolithic preference optimization)."""
        chosen_inputs = {k.replace("chosen_", ""): v for k, v in inputs.items() if k.startswith("chosen_")}
        rejected_inputs = {k.replace("rejected_", ""): v for k, v in inputs.items() if k.startswith("rejected_")}

        chosen_logits = self._get_logits_from_model(model, chosen_inputs)
        rejected_logits = self._get_logits_from_model(model, rejected_inputs)

        policy_chosen_loss = self._compute_samplewise_loss(chosen_logits, chosen_inputs.get("labels"))
        policy_rejected_loss = self._compute_samplewise_loss(rejected_logits, rejected_inputs.get("labels"))

        loss, odds_loss = self.orpo_loss_fn(
            policy_chosen_loss=policy_chosen_loss,
            policy_rejected_loss=policy_rejected_loss,
        )

        return (loss, {"odds_loss": odds_loss}) if return_outputs else loss

    def _compute_kto_loss(
        self, model: Any, inputs: Dict[str, Any], return_outputs: bool
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Any]]:
        """Diffusion-KTO loss for unpaired preference data."""
        labels = inputs.get("labels")
        is_desirable = inputs.get("label", torch.ones(len(labels), dtype=torch.bool, device=labels.device))

        policy_logits = self._get_logits_from_model(model, inputs)
        policy_loss = self._compute_samplewise_loss(policy_logits, labels)

        ref_model = self.ref_model if self.ref_model is not None else model
        with torch.no_grad():
            ref_logits = self._get_logits_from_model(ref_model, inputs)
            ref_loss = self._compute_samplewise_loss(ref_logits, labels)

        loss = self.kto_loss_fn(
            policy_loss=policy_loss,
            reference_loss=ref_loss,
            is_desirable=is_desirable,
        )

        return (loss, {"policy_loss": policy_loss.mean()}) if return_outputs else loss

    def _compute_grpo_loss(
        self, model: Any, inputs: Dict[str, Any], return_outputs: bool
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Any]]:
        """Diffusion-GRPO step with group rollouts and policy gradient clipping."""
        prompts = inputs.get("prompt", [])
        if isinstance(prompts, str):
            prompts = [prompts]

        G = self.grpo_engine.num_generations
        all_completions: List[str] = []
        all_prompts_expanded: List[str] = []

        # Generate G completions per prompt using block diffusion generate
        tokenizer = self.tokenizer
        for p in prompts:
            prompt_inputs = tokenizer(p, return_tensors="pt").to(model.device)
            for _ in range(G):
                with torch.no_grad():
                    if hasattr(model, "generate"):
                        out_tokens = model.generate(**prompt_inputs, max_new_tokens=128)
                        text = tokenizer.decode(out_tokens[0], skip_special_tokens=True)
                    else:
                        text = "Generated text placeholder"
                all_completions.append(text)
                all_prompts_expanded.append(p)

        # Compute rewards and normalize advantages
        rewards = self.grpo_engine.compute_rewards(
            prompts=all_prompts_expanded,
            completions=all_completions,
        ).to(model.device)
        advantages = self.grpo_engine.compute_advantages(rewards, group_size=G)

        # Tokenize completions for policy loss
        gen_inputs = tokenizer(
            [f"{p}\n{c}" for p, c in zip(all_prompts_expanded, all_completions)],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.args.max_seq_length if hasattr(self.args, "max_seq_length") else 1024,
        ).to(model.device)

        logits = self._get_logits_from_model(model, gen_inputs)
        policy_loss_per_sample = self._compute_samplewise_loss(logits, gen_inputs.input_ids)
        policy_log_probs = -policy_loss_per_sample

        with torch.no_grad():
            old_log_probs = policy_log_probs.clone().detach()
            ref_model = self.ref_model if self.ref_model is not None else model
            ref_logits = self._get_logits_from_model(ref_model, gen_inputs)
            ref_loss_per_sample = self._compute_samplewise_loss(ref_logits, gen_inputs.input_ids)
            ref_log_probs = -ref_loss_per_sample

        loss = self.grpo_engine.compute_surrogate_loss(
            policy_log_probs=policy_log_probs,
            old_log_probs=old_log_probs,
            ref_log_probs=ref_log_probs,
            advantages=advantages,
        )

        return (loss, {"mean_reward": rewards.mean()}) if return_outputs else loss

    def _compute_samplewise_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute mean cross-entropy loss per sample in the batch (B,)."""
        B, L = labels.shape
        if logits.ndim == 2:
            valid_mask = (labels.view(-1) != -100)
            sample_losses = torch.zeros(B, device=logits.device, dtype=logits.dtype)
            if not valid_mask.any():
                return sample_losses
            valid_labels = labels.view(-1)[valid_mask]
            valid_ce = nn.functional.cross_entropy(logits, valid_labels, reduction="none")
            sample_indices = torch.arange(B, device=logits.device).unsqueeze(1).expand(B, L).reshape(-1)[valid_mask]
            sample_losses.index_add_(0, sample_indices, valid_ce)
            valid_counts = (labels != -100).sum(dim=1).clamp(min=1.0)
            return sample_losses / valid_counts

        V = logits.shape[-1]
        flat_logits = logits.view(-1, V)
        flat_labels = labels.view(-1)

        valid_mask = (flat_labels != -100)
        sample_losses = torch.zeros(B, device=logits.device, dtype=logits.dtype)
        if not valid_mask.any():
            return sample_losses

        valid_logits = flat_logits[valid_mask]
        valid_labels = flat_labels[valid_mask]

        valid_ce = nn.functional.cross_entropy(
            valid_logits,
            valid_labels,
            reduction="none",
        )

        sample_indices = torch.arange(B, device=logits.device).unsqueeze(1).expand(B, L).reshape(-1)[valid_mask]
        sample_losses.index_add_(0, sample_indices, valid_ce)

        valid_counts = (labels != -100).sum(dim=1).clamp(min=1.0)
        return sample_losses / valid_counts


# =====================================================================
# Specialized Subclass Aliases for Convenience & TRL Compatibility
# =====================================================================

class DiffusionSFTTrainer(DiffusionTrainer):
    """Specialized trainer for Supervised Fine-Tuning of diffusion models."""
    def __init__(self, *args, **kwargs):
        if "args" in kwargs and kwargs["args"] is not None:
            kwargs["args"].training_method = "sft"
        super().__init__(*args, **kwargs)


class DiffusionDPOTrainer(DiffusionTrainer):
    """Specialized trainer for Direct Preference Optimization of diffusion models."""
    def __init__(self, *args, **kwargs):
        if "args" in kwargs and kwargs["args"] is not None:
            kwargs["args"].training_method = "dpo"
        super().__init__(*args, **kwargs)


class DiffusionORPOTrainer(DiffusionTrainer):
    """Specialized trainer for Odds Ratio Preference Optimization of diffusion models."""
    def __init__(self, *args, **kwargs):
        if "args" in kwargs and kwargs["args"] is not None:
            kwargs["args"].training_method = "orpo"
        super().__init__(*args, **kwargs)


class DiffusionGRPOTrainer(DiffusionTrainer):
    """Specialized trainer for Group Relative Policy Optimization of diffusion models."""
    def __init__(self, *args, **kwargs):
        if "args" in kwargs and kwargs["args"] is not None:
            kwargs["args"].training_method = "grpo"
        super().__init__(*args, **kwargs)


class DiffusionKTOTrainer(DiffusionTrainer):
    """Specialized trainer for Kahneman-Tversky Optimization of diffusion models."""
    def __init__(self, *args, **kwargs):
        if "args" in kwargs and kwargs["args"] is not None:
            kwargs["args"].training_method = "kto"
        super().__init__(*args, **kwargs)
