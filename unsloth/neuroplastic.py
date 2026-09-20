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
Neuroplastic Synaptic Engine for Adaptive, Lifelong, and Continual Learning.

Implements the 4-tuple Synaptic State Machine:
    Synapse_ij(t) = (S_ij, P_ij, M_ij, E_ij)

Where:
- S: 2-bit permanent state (00=inactive, 01=exploratory, 10=active, 11=consolidated)
  Weight mapping: 00 -> 0.0, 01 -> alpha, 10 -> 1.0, 11 -> beta (0 < alpha < 1 < beta).
- P: Plasticity potential (accumulated pressure to change state, with decay lambda_P).
- M: Metaplasticity in {0, 1, 2, 3} (connection stability, scaling hysteresis thresholds Theta).
- E: Eligibility trace (short-term activity memory updated via STDP / Hebbian correlation).

Advanced Bio-Inspired Dynamics:
- Three/Four-Factor Plasticity: Delta_P = eta * R * N * U * E (Reward, Novelty, Uncertainty).
- Asymmetric Hysteresis: Theta^+ != Theta^- (prevents limit-cycle chattering).
- Metaplasticity Scaling: Theta = Theta_0 * (1 + alpha_M * M).
- Activity Homeostasis: theta_new = theta_old + eta_H * (r - r*).
- Synaptic Budgeting & Pruning: sum g(S_ij) <= B_j.
- Straight-Through Estimator (STE) for hybrid backpropagation + local plasticity.
"""

from __future__ import annotations
import math
import logging
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

__all__ = [
    "SynapticState",
    "NeuroplasticConfig",
    "NeuroplasticSynapseState",
    "NeuroplasticLinear",
    "make_model_neuroplastic",
    "export_neuroplastic_safetensors",
]


class SynapticState(IntEnum):
    """2-bit permanent synaptic state."""
    INACTIVE = 0       # 00: disconnected / zero weight
    EXPLORATORY = 1    # 01: weak, exploratory connection (alpha)
    ACTIVE = 2         # 10: strong, functional connection (1.0)
    CONSOLIDATED = 3   # 11: consolidated, long-term memory (beta)


@dataclass
class NeuroplasticConfig:
    """Hyperparameters governing neuroplastic dynamics."""
    # Weight values for 2-bit states: w = W(S)
    weight_alpha: float = 0.25   # exploratory weight (0 < alpha < 1)
    weight_active: float = 1.0   # active weight (1.0)
    weight_beta: float = 1.75    # consolidated weight (beta > 1)

    # STDP and Eligibility Trace
    gamma_e: float = 0.85        # eligibility trace decay (E_new = gamma_E * E_old + STDP)
    stdp_a_plus: float = 1.0     # LTP amplitude
    stdp_a_minus: float = 0.8    # LTD amplitude
    stdp_tau_plus: float = 20.0  # LTP time constant (ms)
    stdp_tau_minus: float = 20.0 # LTD time constant (ms)

    # Plasticity Potential P
    learning_rate: float = 0.05  # eta in Delta_P = eta * R * N * U * E
    lambda_p: float = 0.92       # potential decay factor (P_new = clip(lambda_P * P + Delta_P))
    p_max: float = 15.0          # saturation limit for P

    # Asymmetric Thresholds & Metaplasticity
    theta_plus_0: float = 5.0    # base potentiation threshold (00 -> 01 -> 10 -> 11)
    theta_minus_0: float = 7.0   # base depression threshold (11 -> 10 -> 01 -> 00)
    alpha_m: float = 0.5         # metaplasticity scaling factor: Theta = Theta_0 * (1 + alpha_M * M)
    m_decay_rate: float = 0.001  # passive metaplasticity decay for unused connections

    # Synaptic Budget & Structural Plasticity
    synaptic_budget: float = 256.0 # max sum of g(S) per postsynaptic neuron
    cost_per_synapse: float = 0.05 # structural connection cost

    # Homeostasis
    eta_h: float = 0.01          # homeostatic adaptation rate
    target_rate: float = 0.1     # target neuron firing rate r*


class SynapticWeightFunction(torch.autograd.Function):
    """
    Straight-Through Estimator (STE) for mapping discrete 2-bit state S to effective weights.
    Forward: w = W(S)
    Backward: passes gradients through to enable hybrid gradient + plastic training.
    """
    @staticmethod
    def forward(ctx, state: torch.Tensor, alpha: float, active: float, beta: float) -> torch.Tensor:
        ctx.save_for_backward(state)
        weights = torch.zeros_like(state, dtype=torch.float32)
        weights[state == SynapticState.EXPLORATORY] = alpha
        weights[state == SynapticState.ACTIVE] = active
        weights[state == SynapticState.CONSOLIDATED] = beta
        return weights

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None, None, None]:
        # Straight-through gradient: pass grad_output directly to state proxy
        return grad_output, None, None, None


class NeuroplasticSynapseState(nn.Module):
    """
    Vectorized representation of Synapse_ij(t) = (S_ij, P_ij, M_ij, E_ij).
    Operates over complete weight matrices (in_features, out_features).
    """

    def __init__(self, in_features: int, out_features: int, config: Optional[NeuroplasticConfig] = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.config = config or NeuroplasticConfig()

        # 1. Permanent State S in {0, 1, 2, 3} (2 bits per synapse)
        # Initialized with a mixture of active and exploratory synapses
        init_states = torch.randint(0, 3, (out_features, in_features), dtype=torch.uint8)
        self.register_buffer("S", init_states)

        # 2. Plasticity Potential P in [-P_max, P_max]
        self.register_buffer("P", torch.zeros((out_features, in_features), dtype=torch.float32))

        # 3. Metaplasticity M in {0, 1, 2, 3}
        self.register_buffer("M", torch.zeros((out_features, in_features), dtype=torch.uint8))

        # 4. Eligibility Trace E
        self.register_buffer("E", torch.zeros((out_features, in_features), dtype=torch.float32))

        # Homeostatic threshold per postsynaptic neuron: theta_j
        self.register_buffer("homeostatic_theta", torch.zeros(out_features, dtype=torch.float32))

        # Step counter for passive decay
        self.register_buffer("last_used_step", torch.zeros((out_features, in_features), dtype=torch.int64))
        self.register_buffer("step_count", torch.tensor(0, dtype=torch.int64))

    def get_effective_weights(self) -> torch.Tensor:
        """Compute effective weight matrix W(S_ij)."""
        cfg = self.config
        return SynapticWeightFunction.apply(
            self.S.float(),
            cfg.weight_alpha,
            cfg.weight_active,
            cfg.weight_beta,
        )

    def update_stdp_trace(
        self,
        pre_activity: torch.Tensor,
        post_activity: torch.Tensor,
        delta_t: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Update eligibility trace E via Hebbian / STDP correlation:
            E_new = gamma_E * E_old + STDP(Delta_t)
        """
        cfg = self.config

        # Compute pre-post correlation (outer product over batch)
        # pre_activity: (B, in_features) or (in_features,)
        # post_activity: (B, out_features) or (out_features,)
        if pre_activity.dim() == 1:
            pre_activity = pre_activity.unsqueeze(0)
        if post_activity.dim() == 1:
            post_activity = post_activity.unsqueeze(0)

        # Reshape for multi-dimensional sequence inputs: (B, T, D) -> (B*T, D)
        if pre_activity.dim() > 2:
            pre_activity = pre_activity.contiguous().view(-1, self.in_features)
        if post_activity.dim() > 2:
            post_activity = post_activity.contiguous().view(-1, self.out_features)

        # Batch-averaged Hebbian correlation
        correlation = torch.matmul(post_activity.t(), pre_activity) / max(pre_activity.shape[0], 1)

        if delta_t is not None:
            # Explicit STDP exponential kernel
            stdp_val = torch.where(
                delta_t > 0,
                cfg.stdp_a_plus * torch.exp(-delta_t / cfg.stdp_tau_plus),
                -cfg.stdp_a_minus * torch.exp(delta_t / cfg.stdp_tau_minus),
            )
            update = stdp_val * correlation
        else:
            # Rate-coded / continuous STDP surrogate
            update = correlation

        self.E.mul_(cfg.gamma_e).add_(update)
        return self.E

    def apply_plasticity(
        self,
        reward: Union[float, torch.Tensor] = 1.0,
        novelty: Union[float, torch.Tensor] = 1.0,
        uncertainty: Union[float, torch.Tensor] = 1.0,
    ) -> Dict[str, int]:
        """
        Apply three/four-factor modulation to plasticity potential P:
            Delta_P = eta * R * N * U * E
            P_new = clip(lambda_P * P + Delta_P, -P_max, P_max)
        Then evaluate asymmetric thresholds with metaplasticity scaling.
        """
        cfg = self.config
        self.step_count.add_(1)

        # 1. Modulated potential delta
        delta_p = cfg.learning_rate * reward * novelty * uncertainty * self.E
        new_p = torch.clamp(cfg.lambda_p * self.P + delta_p, -cfg.p_max, cfg.p_max)
        self.P.copy_(new_p)

        # 2. Dynamic Metaplastic Thresholds: Theta = Theta_0 * (1 + alpha_M * M)
        m_scale = 1.0 + cfg.alpha_m * self.M.float()
        theta_plus = cfg.theta_plus_0 * m_scale
        theta_minus = cfg.theta_minus_0 * m_scale

        # 3. Asymmetric State Transitions
        # Potentiation: P >= Theta^+ -> 00 -> 01 -> 10 -> 11
        potentiate_mask = self.P >= theta_plus
        # Depression: P <= -Theta^- -> 11 -> 10 -> 01 -> 00
        depress_mask = self.P <= -theta_minus

        transitions_up = 0
        transitions_down = 0

        if potentiate_mask.any():
            # Advance state: min(S + 1, 3)
            can_advance = potentiate_mask & (self.S < SynapticState.CONSOLIDATED)
            self.S[can_advance] += 1
            # Reset potential upon state transition
            self.P[can_advance] = 0.0
            # Increase metaplasticity M on reaching active or consolidated
            promoted_to_strong = can_advance & (self.S >= SynapticState.ACTIVE)
            self.M[promoted_to_strong] = torch.clamp(self.M[promoted_to_strong] + 1, 0, 3)
            self.last_used_step[can_advance] = self.step_count
            transitions_up = can_advance.sum().item()

        if depress_mask.any():
            # Regress state: max(S - 1, 0)
            can_regress = depress_mask & (self.S > SynapticState.INACTIVE)
            self.S[can_regress] -= 1
            # Reset potential upon state transition
            self.P[can_regress] = 0.0
            # Decrease metaplasticity M on depression
            demoted = can_regress & (self.M > 0)
            self.M[demoted] -= 1
            transitions_down = can_regress.sum().item()

        # 4. Synaptic Budgeting & Competition
        self._enforce_synaptic_budget()

        return {"promotions": transitions_up, "demotions": transitions_down}

    def update_homeostasis(self, current_rate: torch.Tensor):
        """
        Homeostatic threshold adaptation:
            theta_new = theta_old + eta_H * (r - r*)
        """
        cfg = self.config
        if current_rate.dim() > 1:
            current_rate = current_rate.mean(dim=list(range(current_rate.dim() - 1)))
        delta_h = cfg.eta_h * (current_rate - cfg.target_rate)
        self.homeostatic_theta.add_(delta_h)

    def _enforce_synaptic_budget(self):
        """
        Enforce max sum g(S_ij) <= B_j per postsynaptic neuron.
        g(00)=0, g(01)=0.2, g(10)=1.0, g(11)=1.5.
        Weakest / lowest-metaplasticity synapses are pruned first.
        """
        cfg = self.config
        g_cost = torch.zeros_like(self.S, dtype=torch.float32)
        g_cost[self.S == SynapticState.EXPLORATORY] = 0.2
        g_cost[self.S == SynapticState.ACTIVE] = 1.0
        g_cost[self.S == SynapticState.CONSOLIDATED] = 1.5

        # Sum cost per output neuron: (out_features,)
        neuron_costs = g_cost.sum(dim=1)
        over_budget = neuron_costs > cfg.synaptic_budget

        if over_budget.any():
            for neuron_idx in torch.where(over_budget)[0]:
                # Find synapses for this neuron sorted by (M, S, |P|)
                syn_s = self.S[neuron_idx]
                syn_m = self.M[neuron_idx]
                # Priority metric: lower is pruned first
                priority = syn_m.float() * 10.0 + syn_s.float() + self.P[neuron_idx].abs() * 0.1
                # Only prune non-inactive synapses
                active_synapses = torch.where(syn_s > SynapticState.INACTIVE)[0]
                if len(active_synapses) == 0:
                    continue
                sorted_indices = active_synapses[torch.argsort(priority[active_synapses])]
                # Demote/prune until within budget
                for syn_idx in sorted_indices:
                    self.S[neuron_idx, syn_idx] = SynapticState.INACTIVE
                    self.P[neuron_idx, syn_idx] = 0.0
                    self.M[neuron_idx, syn_idx] = 0
                    g_cost[neuron_idx, syn_idx] = 0.0
                    if g_cost[neuron_idx].sum() <= cfg.synaptic_budget:
                        break


class NeuroplasticLinear(nn.Module):
    """
    Drop-in neuroplastic Linear layer replacing or augmenting nn.Linear.

    Can operate in two modes:
    1. Standalone: pure neuroplastic synaptic weights W(S).
    2. Hybrid / Residual: base dense weight W_base + W(S_neuroplastic).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        hybrid_base_layer: Optional[nn.Linear] = None,
        config: Optional[NeuroplasticConfig] = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.config = config or NeuroplasticConfig()

        self.synapses = NeuroplasticSynapseState(in_features, out_features, self.config)

        if hybrid_base_layer is not None:
            self.base_layer = hybrid_base_layer
            # Freeze base weights to let neuroplasticity handle adaptation
            for p in self.base_layer.parameters():
                p.requires_grad = False
            self.is_hybrid = True
        else:
            self.base_layer = None
            self.is_hybrid = False
            if bias:
                self.bias = nn.Parameter(torch.zeros(out_features))
            else:
                self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        effective_w = self.synapses.get_effective_weights()

        if self.is_hybrid:
            base_out = self.base_layer(x)
            neuro_out = F.linear(x, effective_w, None)
            out = base_out + neuro_out
        else:
            out = F.linear(x, effective_w, self.bias)

        # Apply homeostatic thresholding
        out = out - self.synapses.homeostatic_theta

        # Cache activities for STDP update
        if self.training:
            with torch.no_grad():
                self.synapses.update_stdp_trace(pre_activity=x, post_activity=out)
                self.synapses.update_homeostasis(current_rate=F.relu(out).mean(dim=0))

        return out

    def step_plasticity(
        self,
        reward: Union[float, torch.Tensor] = 1.0,
        novelty: Union[float, torch.Tensor] = 1.0,
        uncertainty: Union[float, torch.Tensor] = 1.0,
    ) -> Dict[str, int]:
        """Trigger three/four-factor plasticity update across synapses."""
        return self.synapses.apply_plasticity(reward=reward, novelty=novelty, uncertainty=uncertainty)


def make_model_neuroplastic(
    model: nn.Module,
    target_modules: Optional[List[str]] = None,
    config: Optional[NeuroplasticConfig] = None,
    hybrid: bool = True,
) -> nn.Module:
    """
    Traverse model and convert target linear layers into NeuroplasticLinear layers.

    Args:
        model: Base PyTorch model (e.g. Gemma, DiffusionGemma, Llama).
        target_modules: Suffixes of layers to convert (e.g. ['q_proj', 'v_proj', 'gate_proj']).
        config: Neuroplasticity configuration.
        hybrid: If True, wraps existing weights with neuroplastic residual delta.
    """
    if target_modules is None:
        target_modules = ["gate_proj", "up_proj", "down_proj"]

    converted = 0
    for name, module in model.named_modules():
        for child_name, child_module in list(module.named_children()):
            if isinstance(child_module, nn.Linear) and any(child_name == tm or child_name.endswith(f"_{tm}") for tm in target_modules):
                if isinstance(child_module, NeuroplasticLinear):
                    continue

                neuro_layer = NeuroplasticLinear(
                    in_features=child_module.in_features,
                    out_features=child_module.out_features,
                    bias=child_module.bias is not None,
                    hybrid_base_layer=child_module if hybrid else None,
                    config=config,
                )
                # Move to same device and dtype
                device = next(child_module.parameters()).device
                dtype = next(child_module.parameters()).dtype
                neuro_layer.to(device=device, dtype=dtype)

                setattr(module, child_name, neuro_layer)
                converted += 1

    logger.info(f"make_model_neuroplastic: Converted {converted} layers to NeuroplasticLinear (hybrid={hybrid}).")
    return model


def export_neuroplastic_safetensors(model: nn.Module, save_directory: str) -> str:
    """
    Export all neuroplastic synaptic states (S, P, M, E, homeostatic_theta) in Safetensors format.
    """
    import os, json
    from safetensors.torch import save_file

    os.makedirs(save_directory, exist_ok=True)
    tensors_to_save: Dict[str, torch.Tensor] = {}
    metadata: Dict[str, Any] = {}

    for name, module in model.named_modules():
        if isinstance(module, NeuroplasticLinear):
            tensors_to_save[f"{name}.S"] = module.synapses.S
            tensors_to_save[f"{name}.P"] = module.synapses.P
            tensors_to_save[f"{name}.M"] = module.synapses.M
            tensors_to_save[f"{name}.E"] = module.synapses.E
            tensors_to_save[f"{name}.theta"] = module.synapses.homeostatic_theta
            metadata[name] = {
                "in_features": module.in_features,
                "out_features": module.out_features,
                "is_hybrid": module.is_hybrid,
            }

    out_file = os.path.join(save_directory, "neuroplastic_synapses.safetensors")
    save_file(tensors_to_save, out_file)

    meta_file = os.path.join(save_directory, "neuroplastic_meta.json")
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Exported neuroplastic states to safetensors: {out_file}")
    return out_file
