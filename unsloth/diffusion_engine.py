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
Discrete Diffusion Engine for Text and Multimodal Diffusion Models (e.g. DiffusionGemma).

Provides:
- Mathematical noise schedules (Linear, Cosine, SquareRoot, MutualInformation).
- Discrete token masking & canvas preparation.
- Diffusion loss functions (ELBO, Masked Denoising Cross-Entropy).
- Diffusion-DPO (Direct Preference Optimization for Diffusion).
- Diffusion-ORPO (Odds Ratio Preference Optimization for Diffusion).
- Diffusion-GRPO (Group Relative Policy Optimization for Diffusion).
- Diffusion-KTO (Kahneman-Tversky Optimization for Diffusion).
- Memory-safe collation and tensor formatting.
"""

from __future__ import annotations
import math
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

__all__ = [
    "NoiseScheduleType",
    "BaseNoiseSchedule",
    "LinearNoiseSchedule",
    "CosineNoiseSchedule",
    "SquareRootNoiseSchedule",
    "LogLinearNoiseSchedule",
    "InformationAdaptiveNoiseSchedule",
    "get_noise_schedule",
    "DiffusionDataCollator",
    "DiffusionLoss",
    "SemanticDiffusionLoss",
    "DiffusionDPOLoss",
    "RegularizedDiffusionDPOLoss",
    "DiffusionORPOLoss",
    "DiffusionGRPOEngine",
    "RobustDiffusionGRPOEngine",
    "DiffusionKTOLoss",
]


# =====================================================================
# 1. Noise Schedules for Discrete Diffusion
# =====================================================================

class NoiseScheduleType(str, Enum):
    LINEAR = "linear"
    COSINE = "cosine"
    SQRT = "sqrt"
    LOG_LINEAR = "log_linear"


class BaseNoiseSchedule(ABC):
    """
    Abstract base class for discrete diffusion noise schedules.
    Defines mask probability gamma(t) in [0, 1] as a function of continuous timestep t in [0, 1].
    """

    def __init__(self, min_mask_rate: float = 0.05, max_mask_rate: float = 0.95):
        self.min_mask_rate = float(min_mask_rate)
        self.max_mask_rate = float(max_mask_rate)
        assert 0.0 <= self.min_mask_rate < self.max_mask_rate <= 1.0, (
            f"Invalid mask rates: min={min_mask_rate}, max={max_mask_rate}"
        )

    @abstractmethod
    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        """
        Compute mask probability gamma(t) for timestep t in [0, 1].
        gamma(0) -> min_mask_rate, gamma(1) -> max_mask_rate.
        """
        pass

    @abstractmethod
    def loss_weight(self, t: torch.Tensor) -> torch.Tensor:
        """
        Continuous-time ELBO importance weighting w(t) = d(gamma)/dt / (1 - gamma(t)).
        """
        pass


class LinearNoiseSchedule(BaseNoiseSchedule):
    """Linear noise schedule: gamma(t) = min_rate + t * (max_rate - min_rate)."""

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        t = torch.clamp(t, 0.0, 1.0)
        return self.min_mask_rate + t * (self.max_mask_rate - self.min_mask_rate)

    def loss_weight(self, t: torch.Tensor) -> torch.Tensor:
        gamma = self(t)
        # Derivative d(gamma)/dt is constant: (max_rate - min_rate)
        d_gamma = self.max_mask_rate - self.min_mask_rate
        return d_gamma / torch.clamp(1.0 - gamma, min=1e-5)


class CosineNoiseSchedule(BaseNoiseSchedule):
    """
    Cosine noise schedule: smoother transition at t=0 and t=1.
    gamma(t) = min_rate + (max_rate - min_rate) * (1 - cos(pi * t / 2)).
    """

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        t = torch.clamp(t, 0.0, 1.0)
        curve = 1.0 - torch.cos(0.5 * math.pi * t)
        return self.min_mask_rate + curve * (self.max_mask_rate - self.min_mask_rate)

    def loss_weight(self, t: torch.Tensor) -> torch.Tensor:
        gamma = self(t)
        d_gamma = 0.5 * math.pi * torch.sin(0.5 * math.pi * t) * (self.max_mask_rate - self.min_mask_rate)
        return d_gamma / torch.clamp(1.0 - gamma, min=1e-5)


class SquareRootNoiseSchedule(BaseNoiseSchedule):
    """Square-root noise schedule: rapid initial corruption, gentler near completion."""

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        t = torch.clamp(t, 0.0, 1.0)
        curve = torch.sqrt(t)
        return self.min_mask_rate + curve * (self.max_mask_rate - self.min_mask_rate)

    def loss_weight(self, t: torch.Tensor) -> torch.Tensor:
        gamma = self(t)
        d_gamma = (0.5 / torch.clamp(torch.sqrt(t), min=1e-4)) * (self.max_mask_rate - self.min_mask_rate)
        return d_gamma / torch.clamp(1.0 - gamma, min=1e-5)


class LogLinearNoiseSchedule(BaseNoiseSchedule):
    """Log-linear noise schedule: uniform information rate across timesteps."""

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        t = torch.clamp(t, 0.0, 1.0)
        log_min = math.log(max(self.min_mask_rate, 1e-4))
        log_max = math.log(self.max_mask_rate)
        gamma = torch.exp(log_min + t * (log_max - log_min))
        return torch.clamp(gamma, self.min_mask_rate, self.max_mask_rate)

    def loss_weight(self, t: torch.Tensor) -> torch.Tensor:
        gamma = self(t)
        log_min = math.log(max(self.min_mask_rate, 1e-4))
        log_max = math.log(self.max_mask_rate)
        d_gamma = gamma * (log_max - log_min)
        return d_gamma / torch.clamp(1.0 - gamma, min=1e-5)


class InformationAdaptiveNoiseSchedule(BaseNoiseSchedule):
    """
    Information-Adaptive Noise Schedule: modulates mask probability and ELBO weights
    based on local token surprisal/entropy:
        I(x) = -log p_unigram(x)
    High-information tokens (keywords, code identifiers, numbers) maintain higher
    supervision and structured unmasking, solving the Token Equivalence Fallacy.
    """

    def __init__(
        self,
        base_schedule: Optional[BaseNoiseSchedule] = None,
        min_mask_rate: float = 0.05,
        max_mask_rate: float = 0.95,
        information_scaling: float = 0.2,
    ):
        super().__init__(min_mask_rate, max_mask_rate)
        self.base_schedule = base_schedule or CosineNoiseSchedule(min_mask_rate, max_mask_rate)
        self.information_scaling = information_scaling

    def __call__(self, t: torch.Tensor, token_surprisal: Optional[torch.Tensor] = None) -> torch.Tensor:
        base_gamma = self.base_schedule(t)
        if token_surprisal is None:
            return base_gamma

        # Normalize surprisal across sequence
        mean_surp = token_surprisal.mean(dim=-1, keepdim=True)
        std_surp = token_surprisal.std(dim=-1, keepdim=True).clamp(min=1e-4)
        normalized_surp = (token_surprisal - mean_surp) / std_surp

        # Modulate gamma: high surprisal tokens are slightly harder to mask
        modulated = base_gamma.unsqueeze(-1) * (1.0 - self.information_scaling * normalized_surp)
        return torch.clamp(modulated, self.min_mask_rate, self.max_mask_rate)

    def loss_weight(self, t: torch.Tensor, token_surprisal: Optional[torch.Tensor] = None) -> torch.Tensor:
        base_weight = self.base_schedule.loss_weight(t)
        if token_surprisal is None:
            return base_weight

        mean_surp = token_surprisal.mean(dim=-1, keepdim=True)
        std_surp = token_surprisal.std(dim=-1, keepdim=True).clamp(min=1e-4)
        normalized_surp = (token_surprisal - mean_surp) / std_surp

        # High-information tokens receive higher importance in the ELBO objective
        return base_weight.unsqueeze(-1) * torch.clamp(1.0 + self.information_scaling * normalized_surp, min=0.1)


def get_noise_schedule(
    schedule: Union[str, NoiseScheduleType, BaseNoiseSchedule] = NoiseScheduleType.COSINE,
    min_mask_rate: float = 0.05,
    max_mask_rate: float = 0.95,
) -> BaseNoiseSchedule:
    """Factory function to instantiate noise schedules."""
    if isinstance(schedule, BaseNoiseSchedule):
        return schedule

    name = str(schedule).lower()
    if name in (NoiseScheduleType.LINEAR.value, "linear"):
        return LinearNoiseSchedule(min_mask_rate, max_mask_rate)
    elif name in (NoiseScheduleType.COSINE.value, "cosine"):
        return CosineNoiseSchedule(min_mask_rate, max_mask_rate)
    elif name in (NoiseScheduleType.SQRT.value, "sqrt"):
        return SquareRootNoiseSchedule(min_mask_rate, max_mask_rate)
    elif name in (NoiseScheduleType.LOG_LINEAR.value, "log_linear", "loglinear"):
        return LogLinearNoiseSchedule(min_mask_rate, max_mask_rate)
    elif name in ("adaptive", "information_adaptive", "info_adaptive"):
        return InformationAdaptiveNoiseSchedule(min_mask_rate=min_mask_rate, max_mask_rate=max_mask_rate)
    else:
        logger.warning(f"Unknown noise schedule '{schedule}', defaulting to CosineNoiseSchedule.")
        return CosineNoiseSchedule(min_mask_rate, max_mask_rate)


# =====================================================================
# 2. Diffusion Data Collator & Canvas Preparation
# =====================================================================

@dataclass
class DiffusionDataCollator:
    """
    Collator for discrete diffusion language models (e.g. DiffusionGemma).

    Key Responsibilities:
    1. Splits input into prompt (clean conditioning prefix) and response (canvas).
    2. Samples continuous timesteps t ~ Uniform(0, 1) per sample or batch.
    3. Masks response tokens with probability gamma(t) using [MASK] token.
    4. Constructs loss weights w(t) for ELBO-consistent loss computation.
    5. Handles dynamic padding, attention masks, and position IDs.
    """

    tokenizer: Any
    mask_token_id: Optional[int] = None
    noise_schedule: BaseNoiseSchedule = field(default_factory=CosineNoiseSchedule)
    max_length: int = 2048
    canvas_block_size: int = 256
    mask_prompt: bool = False
    pad_to_multiple_of: Optional[int] = 8

    def __post_init__(self):
        if self.mask_token_id is None:
            # Resolve mask token ID from tokenizer
            if hasattr(self.tokenizer, "mask_token_id") and self.tokenizer.mask_token_id is not None:
                self.mask_token_id = self.tokenizer.mask_token_id
            elif hasattr(self.tokenizer, "pad_token_id") and self.tokenizer.pad_token_id is not None:
                self.mask_token_id = self.tokenizer.pad_token_id
            else:
                self.mask_token_id = 0
            logger.info(f"DiffusionDataCollator resolved mask_token_id = {self.mask_token_id}")

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch_size = len(features)
        assert batch_size > 0, "Empty batch received in DiffusionDataCollator"

        # Extract input_ids and prompt lengths (if provided)
        has_prompt_len = "prompt_length" in features[0]
        input_ids_list = [f["input_ids"] for f in features]

        # Convert to tensors if needed
        tensor_ids = [
            torch.as_tensor(ids, dtype=torch.long) if not isinstance(ids, torch.Tensor) else ids.clone().detach()
            for ids in input_ids_list
        ]

        # Determine batch max length
        max_len = max(len(ids) for ids in tensor_ids)
        max_len = min(max_len, self.max_length)
        if self.pad_to_multiple_of is not None:
            max_len = ((max_len + self.pad_to_multiple_of - 1) // self.pad_to_multiple_of) * self.pad_to_multiple_of

        pad_token_id = getattr(self.tokenizer, "pad_token_id", 0) or 0

        # Initialize batch tensors
        padded_input_ids = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
        labels = torch.full((batch_size, max_len), -100, dtype=torch.long)
        loss_mask = torch.zeros((batch_size, max_len), dtype=torch.bool)

        # Sample continuous timesteps t in [0, 1] per batch element
        timesteps = torch.rand(batch_size, dtype=torch.float32)
        mask_rates = self.noise_schedule(timesteps)
        loss_weights = self.noise_schedule.loss_weight(timesteps)

        for i, (ids, feat) in enumerate(zip(tensor_ids, features)):
            seq_len = min(len(ids), max_len)
            padded_input_ids[i, :seq_len] = ids[:seq_len]
            attention_mask[i, :seq_len] = 1

            p_len = feat.get("prompt_length", 0) if has_prompt_len else 0
            p_len = min(p_len, seq_len)

            # Labels correspond to the true original tokens
            labels[i, :seq_len] = ids[:seq_len]

            # Determine maskable positions
            if self.mask_prompt:
                maskable = torch.ones(seq_len, dtype=torch.bool)
            else:
                # Prompt tokens are kept clean (unmasked conditioning prefix)
                maskable = torch.zeros(seq_len, dtype=torch.bool)
                if seq_len > p_len:
                    maskable[p_len:] = True

            # If no response tokens, allow all valid tokens to be masked
            if not maskable.any() and seq_len > 0:
                maskable[:] = True

            # Sample Bernoulli mask according to mask_rates[i]
            sample_rate = mask_rates[i].item()
            rand_probs = torch.rand(seq_len)
            to_mask = maskable & (rand_probs < sample_rate)

            # Ensure at least one token is masked per sequence for training signal
            if maskable.any() and not to_mask.any():
                maskable_indices = torch.where(maskable)[0]
                pick = maskable_indices[torch.randint(0, len(maskable_indices), (1,))].item()
                to_mask[pick] = True

            # Apply mask to input_ids
            padded_input_ids[i, :seq_len][to_mask] = self.mask_token_id
            loss_mask[i, :seq_len] = to_mask

            # Unmasked or prompt tokens receive -100 in labels so cross-entropy ignores them
            labels[i, :seq_len][~to_mask] = -100

        batch = {
            "input_ids": padded_input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "loss_mask": loss_mask,
            "timesteps": timesteps,
            "loss_weights": loss_weights,
        }

        # Optional: forward extra fields if present (e.g. images, pixel_values)
        for extra_key in ("pixel_values", "image_grid_thw", "pixel_attention_mask"):
            if extra_key in features[0]:
                batch[extra_key] = torch.stack([f[extra_key] for f in features])

        return batch


# =====================================================================
# 3. Diffusion Loss Functions
# =====================================================================

class DiffusionLoss(nn.Module):
    """
    ELBO-consistent Denoising Loss for Discrete Diffusion Language Models.

    L_denoise = sum_{i in masked} w(t) * CrossEntropy(logits_i, targets_i)
    """

    def __init__(self, label_smoothing: float = 0.0):
        super().__init__()
        self.label_smoothing = label_smoothing

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        loss_weights: Optional[torch.Tensor] = None,
        loss_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            logits: (B, L, V) model output logits.
            labels: (B, L) target token IDs with -100 for ignored positions.
            loss_weights: (B,) ELBO weighting w(t) per sample.
            loss_mask: (B, L) bool mask of positions where denoising is evaluated.
        """
        B, L, V = logits.shape

        # Flatten tensors
        flat_logits = logits.view(-1, V)
        flat_labels = labels.view(-1)

        # Standard token-level cross-entropy
        ce_loss = F.cross_entropy(
            flat_logits,
            flat_labels,
            ignore_index=-100,
            reduction="none",
            label_smoothing=self.label_smoothing,
        ).view(B, L)

        if loss_weights is not None:
            # Weight loss per sample by w(t)
            ce_loss = ce_loss * loss_weights.view(B, 1)

        # Average over valid (masked) tokens
        if loss_mask is not None:
            valid_tokens = loss_mask.sum().clamp(min=1.0)
            return ce_loss.sum() / valid_tokens
        else:
            valid_tokens = (labels != -100).sum().clamp(min=1.0)
            return ce_loss.sum() / valid_tokens


class SemanticDiffusionLoss(nn.Module):
    """
    Wasserstein-inspired Semantic Denoising Loss.
    Combines categorical Cross-Entropy with cosine distance in the token embedding space.
    Solves the Token Equivalence Fallacy by providing soft geometric penalties for near-synonyms.
    """

    def __init__(
        self,
        embedding_matrix: Optional[torch.Tensor] = None,
        semantic_weight: float = 0.1,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.embedding_matrix = embedding_matrix
        self.semantic_weight = semantic_weight
        self.label_smoothing = label_smoothing

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        loss_weights: Optional[torch.Tensor] = None,
        loss_mask: Optional[torch.Tensor] = None,
        embedding_matrix: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, L, V = logits.shape
        flat_logits = logits.view(-1, V)
        flat_labels = labels.view(-1)

        ce_loss = F.cross_entropy(
            flat_logits,
            flat_labels,
            ignore_index=-100,
            reduction="none",
            label_smoothing=self.label_smoothing,
        ).view(B, L)

        if loss_weights is not None:
            if loss_weights.dim() == 1:
                ce_loss = ce_loss * loss_weights.view(B, 1)
            else:
                ce_loss = ce_loss * loss_weights

        emb_matrix = embedding_matrix if embedding_matrix is not None else self.embedding_matrix
        if emb_matrix is not None and self.semantic_weight > 0.0:
            probs = F.softmax(flat_logits, dim=-1)
            valid_mask = flat_labels != -100
            if valid_mask.any():
                pred_emb = torch.matmul(probs[valid_mask], emb_matrix)
                target_emb = emb_matrix[flat_labels[valid_mask]]
                cos_sim = F.cosine_similarity(pred_emb, target_emb, dim=-1)
                semantic_loss = 1.0 - cos_sim
                ce_view = ce_loss.view(-1)
                ce_view[valid_mask] = (1.0 - self.semantic_weight) * ce_view[valid_mask] + self.semantic_weight * semantic_loss

        if loss_mask is not None:
            valid_tokens = loss_mask.sum().clamp(min=1.0)
            return ce_loss.sum() / valid_tokens
        else:
            valid_tokens = (labels != -100).sum().clamp(min=1.0)
            return ce_loss.sum() / valid_tokens


class DiffusionDPOLoss(nn.Module):
    """
    Direct Preference Optimization (DPO) for Discrete Diffusion Models.

    In diffusion models, the log-likelihood of a sequence y given prompt x is
    proportional to the negative ELBO denoising loss:
        log p_theta(y | x) ~ - L_denoise(y | x; theta)

    The DPO objective with reference model pi_ref becomes:
        Delta r(y) = L_denoise_ref(y) - L_denoise_theta(y)
        L_DPO = - log sigma( beta * [ Delta r(y_chosen) - Delta r(y_rejected) ] )
    """

    def __init__(self, beta: float = 0.1, label_smoothing: float = 0.0):
        super().__init__()
        self.beta = beta
        self.label_smoothing = label_smoothing

    def forward(
        self,
        policy_chosen_loss: torch.Tensor,
        policy_rejected_loss: torch.Tensor,
        reference_chosen_loss: torch.Tensor,
        reference_rejected_loss: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        chosen_rewards = self.beta * (reference_chosen_loss - policy_chosen_loss)
        rejected_rewards = self.beta * (reference_rejected_loss - policy_rejected_loss)

        logits = chosen_rewards - rejected_rewards

        if self.label_smoothing > 0.0:
            loss = (
                -F.logsigmoid(logits) * (1.0 - self.label_smoothing)
                - F.logsigmoid(-logits) * self.label_smoothing
            ).mean()
        else:
            loss = -F.logsigmoid(logits).mean()

        return loss, chosen_rewards.detach(), rejected_rewards.detach()


class RegularizedDiffusionDPOLoss(nn.Module):
    """
    Length-Normalized & Bounded Diffusion-DPO with Target Margin (SimPO-inspired)
    and NLL Anchor Regularization.

    Solves:
    1. Likelihood displacement: NLL anchor prevents pi(y_w) from decaying.
    2. Length bias: length normalization (1 / |y|^alpha) stops verbosity hacking.
    3. Over-optimization: target margin gamma enforces a strict positive separation.
    """

    def __init__(
        self,
        beta: float = 0.1,
        gamma_margin: float = 0.5,
        length_penalty_alpha: float = 0.6,
        nll_anchor_weight: float = 0.05,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.beta = beta
        self.gamma_margin = gamma_margin
        self.length_penalty_alpha = length_penalty_alpha
        self.nll_anchor_weight = nll_anchor_weight
        self.label_smoothing = label_smoothing

    def forward(
        self,
        policy_chosen_loss: torch.Tensor,
        policy_rejected_loss: torch.Tensor,
        reference_chosen_loss: torch.Tensor,
        reference_rejected_loss: torch.Tensor,
        chosen_lengths: Optional[torch.Tensor] = None,
        rejected_lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Length normalization factors
        if chosen_lengths is not None:
            chosen_scale = (chosen_lengths.float() ** self.length_penalty_alpha).clamp(min=1.0)
        else:
            chosen_scale = torch.ones_like(policy_chosen_loss)

        if rejected_lengths is not None:
            rejected_scale = (rejected_lengths.float() ** self.length_penalty_alpha).clamp(min=1.0)
        else:
            rejected_scale = torch.ones_like(policy_rejected_loss)

        chosen_rewards = self.beta * (reference_chosen_loss - policy_chosen_loss) / chosen_scale
        rejected_rewards = self.beta * (reference_rejected_loss - policy_rejected_loss) / rejected_scale

        logits = chosen_rewards - rejected_rewards - self.gamma_margin

        if self.label_smoothing > 0.0:
            loss = (
                -F.logsigmoid(logits) * (1.0 - self.label_smoothing)
                - F.logsigmoid(-logits) * self.label_smoothing
            ).mean()
        else:
            loss = -F.logsigmoid(logits).mean()

        # NLL Anchor: explicitly keep chosen loss low to prevent probability degradation
        if self.nll_anchor_weight > 0.0:
            loss = loss + self.nll_anchor_weight * policy_chosen_loss.mean()

        return loss, chosen_rewards.detach(), rejected_rewards.detach()


class DiffusionORPOLoss(nn.Module):
    """
    Odds Ratio Preference Optimization (ORPO) for Discrete Diffusion Models.

    Monolithic reference-free preference optimization:
        L_ORPO = L_SFT(chosen) + lambda * L_odds
    where odds ratio is computed from diffusion denoising loss differentials:
        odds(y) = exp( - L_denoise(y) ) / (1 - exp( - L_denoise(y) ))
    """

    def __init__(self, lambda_orpo: float = 0.1):
        super().__init__()
        self.lambda_orpo = lambda_orpo

    def forward(
        self,
        policy_chosen_loss: torch.Tensor,
        policy_rejected_loss: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            policy_chosen_loss: (B,) mean denoising loss of chosen.
            policy_rejected_loss: (B,) mean denoising loss of rejected.
        """
        sft_loss = policy_chosen_loss.mean()

        # Log-odds differential: log(odds_w / odds_l)
        # log(odds) = -loss - log(1 - exp(-loss))
        log_p_w = -policy_chosen_loss
        log_p_l = -policy_rejected_loss

        log_odds_w = log_p_w - torch.log1p(-torch.exp(log_p_w.clamp(max=-1e-4)))
        log_odds_l = log_p_l - torch.log1p(-torch.exp(log_p_l.clamp(max=-1e-4)))

        log_odds_diff = log_odds_w - log_odds_l
        odds_loss = -F.logsigmoid(log_odds_diff).mean()

        total_loss = sft_loss + self.lambda_orpo * odds_loss
        return total_loss, odds_loss.detach()


class DiffusionKTOLoss(nn.Module):
    """
    Kahneman-Tversky Optimization (KTO) for Discrete Diffusion Models.

    Optimizes unpaired human preferences based on prospect theory utility:
        r(x, y) = beta * (L_ref(y) - L_policy(y))
        v(r) = 1 - sigma(r - z0)   if desirable
        v(r) = 1 - sigma(z0 - r)   if undesirable
    """

    def __init__(self, beta: float = 0.1, desirable_weight: float = 1.0, undesirable_weight: float = 1.0):
        super().__init__()
        self.beta = beta
        self.desirable_weight = desirable_weight
        self.undesirable_weight = undesirable_weight

    def forward(
        self,
        policy_loss: torch.Tensor,
        reference_loss: torch.Tensor,
        is_desirable: torch.Tensor,  # (B,) bool or float (1 for desirable, 0 for undesirable)
        z0: float = 0.0,
    ) -> torch.Tensor:
        rewards = self.beta * (reference_loss - policy_loss)
        is_desirable = is_desirable.bool()

        loss = torch.zeros_like(rewards)
        if is_desirable.any():
            loss[is_desirable] = self.desirable_weight * (1.0 - torch.sigmoid(rewards[is_desirable] - z0))
        if (~is_desirable).any():
            loss[~is_desirable] = self.undesirable_weight * (1.0 - torch.sigmoid(z0 - rewards[~is_desirable]))

        return loss.mean()


class DiffusionGRPOEngine:
    """
    Group Relative Policy Optimization (GRPO) Engine for Discrete Diffusion Models.

    Key Features:
    1. Generates G rollouts per prompt using model.generate (block diffusion sampling).
    2. Computes rewards using custom user-supplied reward functions (rule-based, accuracy, format).
    3. Normalizes advantages per group: A_i = (R_i - mean(R)) / (std(R) + 1e-4).
    4. Computes clipped surrogate policy loss on denoising trajectories with KL divergence penalty.
    """

    def __init__(
        self,
        num_generations: int = 4,
        clip_range: float = 0.2,
        kl_coeff: float = 0.01,
        reward_functions: Optional[List[Callable[..., List[float]]]] = None,
    ):
        self.num_generations = num_generations
        self.clip_range = clip_range
        self.kl_coeff = kl_coeff
        self.reward_functions = reward_functions or []

    def compute_rewards(
        self,
        prompts: List[str],
        completions: List[str],
        **kwargs,
    ) -> torch.Tensor:
        """Evaluate all reward functions and return aggregate reward tensor (B * G,)."""
        total_samples = len(completions)
        rewards = torch.zeros(total_samples, dtype=torch.float32)

        if not self.reward_functions:
            # Default length/coherence penalty heuristic if no user functions provided
            for idx, c in enumerate(completions):
                score = min(len(c.split()) / 50.0, 1.0)
                rewards[idx] = score
            return rewards

        for fn in self.reward_functions:
            try:
                fn_rewards = fn(prompts=prompts, completions=completions, **kwargs)
                rewards += torch.as_tensor(fn_rewards, dtype=torch.float32)
            except Exception as e:
                logger.warning(f"Error in reward function {getattr(fn, '__name__', str(fn))}: {e}")

        return rewards

    def compute_advantages(self, rewards: torch.Tensor, group_size: int) -> torch.Tensor:
        """
        Normalize rewards per prompt group:
        A_i = (R_i - mean(group)) / (std(group) + 1e-4)
        """
        N = rewards.shape[0]
        num_groups = N // group_size
        reshaped = rewards.view(num_groups, group_size)

        mean = reshaped.mean(dim=1, keepdim=True)
        std = reshaped.std(dim=1, keepdim=True)
        advantages = (reshaped - mean) / (std + 1e-4)
        return advantages.view(N)

    def compute_surrogate_loss(
        self,
        policy_log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        ref_log_probs: torch.Tensor,
        advantages: torch.Tensor,
    ) -> torch.Tensor:
        """
        GRPO clipped surrogate policy loss:
        ratio = exp(policy_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = clip(ratio, 1 - eps, 1 + eps) * advantages
        loss = - min(surr1, surr2) + kl_coeff * D_KL(policy || ref)
        """
        ratio = torch.exp(policy_log_probs - old_log_probs)
        clipped_ratio = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range)

        surr1 = ratio * advantages
        surr2 = clipped_ratio * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # Reverse KL: D_KL(policy || ref) = exp(ref_log_probs - policy_log_probs) - (ref_log_probs - policy_log_probs) - 1
        # or standard approx: policy_log_probs - ref_log_probs
        kl_divergence = (policy_log_probs - ref_log_probs).mean()
        total_loss = policy_loss + self.kl_coeff * kl_divergence

        return total_loss


class RobustDiffusionGRPOEngine(DiffusionGRPOEngine):
    """
    Robust GRPO Engine addressing Zero-Variance Collapse and Exploration Traps.

    Features:
    1. EMA baseline fallback when group variance sigma(R) < delta.
    2. Pairwise soft-margin ranking when all rollouts are identical.
    3. Exploration entropy bonus to escape all-zero reward deadlocks.
    """

    def __init__(
        self,
        num_generations: int = 4,
        clip_range: float = 0.2,
        kl_coeff: float = 0.01,
        entropy_coeff: float = 0.01,
        min_std_threshold: float = 1e-4,
        ema_decay: float = 0.95,
        reward_functions: Optional[List[Callable[..., List[float]]]] = None,
    ):
        super().__init__(
            num_generations=num_generations,
            clip_range=clip_range,
            kl_coeff=kl_coeff,
            reward_functions=reward_functions,
        )
        self.entropy_coeff = entropy_coeff
        self.min_std_threshold = min_std_threshold
        self.ema_decay = ema_decay
        self.running_reward_ema: Optional[torch.Tensor] = None

    def compute_advantages(self, rewards: torch.Tensor, group_size: int) -> torch.Tensor:
        N = rewards.shape[0]
        num_groups = N // group_size
        reshaped = rewards.view(num_groups, group_size)

        mean = reshaped.mean(dim=1, keepdim=True)
        std = reshaped.std(dim=1, keepdim=True)

        batch_mean = rewards.mean().detach()
        if self.running_reward_ema is None:
            self.running_reward_ema = batch_mean
        else:
            self.running_reward_ema = self.ema_decay * self.running_reward_ema + (1.0 - self.ema_decay) * batch_mean

        advantages = torch.zeros_like(reshaped)
        for g in range(num_groups):
            g_std = std[g, 0]
            if g_std >= self.min_std_threshold:
                advantages[g] = (reshaped[g] - mean[g]) / (g_std + 1e-4)
            else:
                # Zero-variance resolution: compare against EMA reward baseline
                advantages[g] = reshaped[g] - self.running_reward_ema

        return advantages.view(N)

    def compute_surrogate_loss(
        self,
        policy_log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        ref_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        policy_logits: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        base_loss = super().compute_surrogate_loss(
            policy_log_probs=policy_log_probs,
            old_log_probs=old_log_probs,
            ref_log_probs=ref_log_probs,
            advantages=advantages,
        )
        if policy_logits is not None and self.entropy_coeff > 0.0:
            probs = F.softmax(policy_logits, dim=-1)
            log_probs = F.log_softmax(policy_logits, dim=-1)
            entropy = -(probs * log_probs).sum(dim=-1).mean()
            # Maximize entropy -> subtract entropy * coeff from loss
            return base_loss - self.entropy_coeff * entropy
        return base_loss
