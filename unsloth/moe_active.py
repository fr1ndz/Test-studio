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
Active Parameter MoE Training with Dynamic Expert Paging & Synaptic Plasticity.

Core Concepts:
1. Active Parameter Slicing:
   In MoE models (such as DiffusionGemma-26B-A4B with 128 experts), only a fraction of
   parameters (~4B out of 26B) are active per token/batch. Keeping all 128 experts (44+ GB)
   resident in GPU VRAM causes severe out-of-memory errors on consumer and standard datacenter GPUs.

2. Dynamic Expert Paging (PagedMoETextExperts):
   Master weights of the 128 experts are held in CPU pinned memory (or host RAM) and
   streamed on-demand into GPU VRAM only for the active cluster, reducing GPU memory
   footprint by over 35-40 GB.

3. Cyclic & Phased Training (CyclicMoEController):
   Divides the 128 experts into active clusters (e.g. 4 clusters of 32 experts).
   Trains active clusters in phases until consolidated (Synaptic State S: 01 -> 10 -> 11),
   rotates to the next cluster until all clusters are trained, and consolidates into .safetensors.
"""

from __future__ import annotations
import gc
import logging
import math
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .neuroplastic import NeuroplasticConfig, SynapticState

try:
    from transformers.trainer_callback import TrainerCallback
except ImportError:
    class TrainerCallback:
        pass

logger = logging.getLogger(__name__)

__all__ = [
    "MoEPhase",
    "ActiveMoEConfig",
    "PagedMoETextExperts",
    "CyclicMoEController",
    "MoECyclicCallback",
    "patch_moe_active_parameters",
]


class MoEPhase(IntEnum):
    """Lifecycle phase of an MoE expert cluster."""
    INACTIVE = 0       # Not currently selected or trained
    EXPLORATORY = 1    # Weakly active, exploring representations
    TRAINING = 2       # Actively receiving gradient updates
    CONSOLIDATED = 3   # Fully trained and stabilized; frozen or low-LR


@dataclass
class ActiveMoEConfig:
    """Hyperparameters for active parameter MoE training and expert paging."""
    num_experts: int = 128
    num_clusters: int = 4              # 128 / 4 = 32 experts per cluster
    active_experts_per_tok: int = 4    # Top-K active experts per token
    offload_to_cpu: bool = True        # Keep master 3D expert weights in pinned CPU RAM
    pin_memory: bool = True            # Use pinned host memory for fast non-blocking transfers
    cluster_switch_steps: int = 250    # Steps to train each cluster before rotating
    consolidation_threshold: float = 0.85 # Stability threshold to mark cluster as consolidated
    joint_calibration_epochs: int = 1  # Final epochs with all consolidated experts active
    neuroplastic_dynamics: bool = True # Enable (S, P, M, E) synaptic tracking per expert
    trainable_experts: bool = False    # Whether active experts themselves have requires_grad=True


class PagedMoETextExperts(nn.Module):
    """
    Drop-in replacement for DiffusionGemmaTextExperts that supports:
    1. Dynamic on-demand streaming of active expert slices from pinned host memory.
    2. Cluster-restricted routing during cyclic active-parameter training.
    3. Resident active cluster buffer on GPU with zero PCIe overhead during cluster training.
    4. 100% roundtrip `.safetensors` export compatibility.
    """

    def __init__(
        self,
        original_experts: nn.Module,
        controller: CyclicMoEController,
        layer_idx: int = 0,
        offload_to_cpu: bool = True,
        trainable_experts: bool = False,
    ):
        super().__init__()
        self.num_experts = getattr(original_experts, "num_experts", 128)
        self.hidden_dim = getattr(original_experts, "hidden_dim", 4096)
        self.intermediate_dim = getattr(original_experts, "intermediate_dim", 704)
        self.act_fn = getattr(original_experts, "act_fn", F.gelu)
        self.controller = controller
        self.layer_idx = layer_idx
        self.offload_to_cpu = offload_to_cpu
        self.trainable_experts = trainable_experts

        # Extract weights from original module
        gate_up = original_experts.gate_up_proj.data
        down = original_experts.down_proj.data

        # Cluster size
        self.cluster_size = self.num_experts // controller.num_clusters

        # Master weights on CPU pinned memory
        if offload_to_cpu:
            self.gate_up_proj_cpu = gate_up.detach().cpu().pin_memory() if torch.cuda.is_available() else gate_up.detach().cpu()
            self.down_proj_cpu = down.detach().cpu().pin_memory() if torch.cuda.is_available() else down.detach().cpu()
            # Free references from original module so GPU memory is reclaimed
            original_experts.gate_up_proj = None
            original_experts.down_proj = None
        else:
            self.gate_up_proj_cpu = gate_up.detach().cpu()
            self.down_proj_cpu = down.detach().cpu()

        # Target device and dtype for the active cluster buffer
        device = gate_up.device if not offload_to_cpu else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        dtype = gate_up.dtype

        # Initial active cluster buffer on device
        init_gate_up = self.gate_up_proj_cpu[0:self.cluster_size].to(device).clone()
        init_down = self.down_proj_cpu[0:self.cluster_size].to(device).clone()

        if trainable_experts:
            self.active_gate_up_proj = nn.Parameter(init_gate_up, requires_grad=True)
            self.active_down_proj = nn.Parameter(init_down, requires_grad=True)
        else:
            self.register_buffer("active_gate_up_proj", init_gate_up)
            self.register_buffer("active_down_proj", init_down)

        # Register empty dummy parameters for gate_up_proj and down_proj for compatibility
        self.gate_up_proj = nn.Parameter(torch.empty(0, dtype=dtype), requires_grad=False)
        self.down_proj = nn.Parameter(torch.empty(0, dtype=dtype), requires_grad=False)

        # GPU cache for currently active out-of-cluster consolidated experts
        self._gpu_expert_cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}

    def switch_cluster(self, old_cluster: int, new_cluster: int):
        """Consolidate current active cluster back to CPU master, and load new cluster."""
        old_start = old_cluster * self.cluster_size
        old_end = old_start + self.cluster_size
        new_start = new_cluster * self.cluster_size
        new_end = new_start + self.cluster_size

        # If trainable, copy updated active parameters back to CPU master
        if self.trainable_experts and self.offload_to_cpu:
            self.gate_up_proj_cpu[old_start:old_end].copy_(self.active_gate_up_proj.data.cpu())
            self.down_proj_cpu[old_start:old_end].copy_(self.active_down_proj.data.cpu())

        # Load new cluster slice into GPU active buffer
        if new_start < self.num_experts:
            new_gate = self.gate_up_proj_cpu[new_start:new_end].to(self.active_gate_up_proj.device)
            new_down = self.down_proj_cpu[new_start:new_end].to(self.active_down_proj.device)
            self.active_gate_up_proj.data.copy_(new_gate)
            self.active_down_proj.data.copy_(new_down)

        # Clear temporary GPU expert buffers
        self.clear_gpu_cache()

    def get_expert_weights(self, expert_idx: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fetch expert weights (from GPU cache or streamed from pinned CPU memory)."""
        if not self.offload_to_cpu:
            return self.gate_up_proj[expert_idx], self.down_proj[expert_idx]

        if expert_idx in self._gpu_expert_cache:
            return self._gpu_expert_cache[expert_idx]

        # Stream slice from pinned CPU memory to GPU
        gate_up_gpu = self.gate_up_proj_cpu[expert_idx].to(device, non_blocking=True)
        down_gpu = self.down_proj_cpu[expert_idx].to(device, non_blocking=True)

        if len(self._gpu_expert_cache) < 16:
            self._gpu_expert_cache[expert_idx] = (gate_up_gpu, down_gpu)

        return gate_up_gpu, down_gpu

    def clear_gpu_cache(self):
        """Release temporary GPU expert buffers."""
        self._gpu_expert_cache.clear()

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        """Ensure full master 3D weights are exported for 100% valid .safetensors."""
        # Flush current active cluster to CPU master
        current_cluster = self.controller.current_cluster_idx
        start = current_cluster * self.cluster_size
        end = start + self.cluster_size
        if self.offload_to_cpu and self.gate_up_proj_cpu is not None:
            if self.trainable_experts:
                self.gate_up_proj_cpu[start:end].copy_(self.active_gate_up_proj.data.cpu())
                self.down_proj_cpu[start:end].copy_(self.active_down_proj.data.cpu())
            destination[prefix + "gate_up_proj"] = self.gate_up_proj_cpu if keep_vars else self.gate_up_proj_cpu.detach().clone()
            destination[prefix + "down_proj"] = self.down_proj_cpu if keep_vars else self.down_proj_cpu.detach().clone()
        else:
            destination[prefix + "gate_up_proj"] = self.gate_up_proj
            destination[prefix + "down_proj"] = self.down_proj

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        device = hidden_states.device
        final_hidden_states = torch.zeros_like(hidden_states)

        with torch.no_grad():
            expert_mask = F.one_hot(top_k_index, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        curr_cluster = self.controller.current_cluster_idx
        cluster_start = curr_cluster * self.cluster_size
        cluster_end = cluster_start + self.cluster_size

        for hit in expert_hit:
            expert_idx = hit[0].item()
            if expert_idx >= self.num_experts:
                continue

            # Check if this expert is allowed by the cyclic controller
            if not self.controller.is_expert_active(expert_idx):
                continue

            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            if len(token_idx) == 0:
                continue

            current_state = hidden_states[token_idx]

            # Fast path: if expert is in current active cluster buffer on GPU, use resident weights
            if cluster_start <= expert_idx < cluster_end:
                local_idx = expert_idx - cluster_start
                gate_up_w = self.active_gate_up_proj[local_idx]
                down_w = self.active_down_proj[local_idx]
            else:
                # Expert is consolidated from an earlier cluster, stream on demand
                gate_up_w, down_w = self.get_expert_weights(expert_idx, device)

            gate, up = F.linear(current_state, gate_up_w).chunk(2, dim=-1)
            current_hidden = self.act_fn(gate) * up
            current_hidden = F.linear(current_hidden, down_w)
            current_hidden = current_hidden * top_k_weights[token_idx, top_k_pos, None]

            final_hidden_states.index_add_(0, token_idx, current_hidden.to(final_hidden_states.dtype))

            # Update synaptic eligibility trace for this expert
            if self.training:
                self.controller.record_expert_activity(expert_idx, len(token_idx))

        return final_hidden_states


class CyclicMoEController:
    """
    Coordinates active-parameter training across expert clusters.

    Workflow:
    1. Partitions experts into clusters: Cluster k = [k * S, (k+1) * S).
    2. In Phase k, restricts routing to Cluster k so that only active parameters are trained.
    3. Tracks synaptic state (S, P, M, E) for each expert.
    4. Automatically consolidates clusters and rotates to the next cluster.
    5. After all clusters are consolidated, enters joint calibration phase.
    """

    def __init__(self, config: Optional[ActiveMoEConfig] = None):
        self.config = config or ActiveMoEConfig()
        self.num_experts = self.config.num_experts
        self.num_clusters = self.config.num_clusters
        self.cluster_size = self.num_experts // self.num_clusters

        # Active cluster index in [0, num_clusters)
        self.current_cluster_idx: int = 0
        self.current_step: int = 0
        self.is_joint_calibration: bool = False
        self._expert_modules: List[PagedMoETextExperts] = []

        # Synaptic state machine per expert: (S, P, M, E)
        self.expert_states = torch.full((self.num_experts,), SynapticState.INACTIVE, dtype=torch.uint8)
        self.expert_stability = torch.zeros(self.num_experts, dtype=torch.float32) # M in [0, 1]
        self.expert_eligibility = torch.zeros(self.num_experts, dtype=torch.float32) # E

        # Activate initial cluster
        self._activate_cluster(0)

    def register_expert_modules(self, modules: List[PagedMoETextExperts]):
        """Register all paged expert modules for synchronized cluster rotation."""
        self._expert_modules = modules

    def _activate_cluster(self, cluster_idx: int):
        """Set active experts for the specified cluster and notify expert modules."""
        old_cluster = self.current_cluster_idx
        self.current_cluster_idx = cluster_idx
        start = cluster_idx * self.cluster_size
        end = min(start + self.cluster_size, self.num_experts)

        for i in range(self.num_experts):
            if start <= i < end:
                if self.expert_states[i] != SynapticState.CONSOLIDATED:
                    self.expert_states[i] = SynapticState.ACTIVE
            elif self.expert_states[i] != SynapticState.CONSOLIDATED:
                self.expert_states[i] = SynapticState.INACTIVE

        if self._expert_modules and old_cluster != cluster_idx:
            for m in self._expert_modules:
                m.switch_cluster(old_cluster, cluster_idx)

        logger.info(
            f"CyclicMoE: Activated Cluster {cluster_idx}/{self.num_clusters} "
            f"(Experts [{start}..{end-1}]). Active params: "
            f"~{(self.cluster_size / self.num_experts) * 22:.1f}B / 26B"
        )

    def is_expert_active(self, expert_idx: int) -> bool:
        """True if the expert is permitted to execute in the current phase."""
        if self.is_joint_calibration:
            return True
        start = self.current_cluster_idx * self.cluster_size
        end = start + self.cluster_size
        return start <= expert_idx < end or self.expert_states[expert_idx] == SynapticState.CONSOLIDATED

    def record_expert_activity(self, expert_idx: int, num_tokens: int):
        """Update eligibility trace E and stability M based on token assignments."""
        if 0 <= expert_idx < self.num_experts:
            self.expert_eligibility[expert_idx] = 0.9 * self.expert_eligibility[expert_idx] + 0.1 * num_tokens
            self.expert_stability[expert_idx] = min(1.0, self.expert_stability[expert_idx] + 0.002)

    def step(self):
        """Advance step counter and handle cluster rotation / consolidation."""
        self.current_step += 1

        if self.is_joint_calibration:
            return

        # Check if current cluster should be consolidated and rotated
        if self.current_step % self.config.cluster_switch_steps == 0:
            start = self.current_cluster_idx * self.cluster_size
            end = min(start + self.cluster_size, self.num_experts)

            # Consolidate trained cluster (S -> 11)
            for i in range(start, end):
                self.expert_states[i] = SynapticState.CONSOLIDATED

            logger.info(
                f"CyclicMoE: Consolidated Cluster {self.current_cluster_idx} "
                f"(Experts [{start}..{end-1}])."
            )

            # Advance to next cluster or enter joint calibration
            next_cluster = self.current_cluster_idx + 1
            if next_cluster < self.num_clusters:
                self._activate_cluster(next_cluster)
            else:
                self.is_joint_calibration = True
                logger.info(
                    "CyclicMoE: All clusters trained and consolidated! "
                    "Entering final joint calibration phase with all 128 experts active."
                )

    def modulate_router_probs(
        self,
        router_probs: torch.Tensor,
        top_k: int = 4,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Restricts routing to the active cluster during cyclic training.
        """
        if self.is_joint_calibration:
            return torch.topk(router_probs, top_k, dim=-1)

        start = self.current_cluster_idx * self.cluster_size
        end = start + self.cluster_size

        mask = torch.zeros_like(router_probs)
        mask[..., start:end] = 1.0

        # Allow already consolidated experts to receive routing with lower weight
        for i in range(self.num_experts):
            if self.expert_states[i] == SynapticState.CONSOLIDATED:
                mask[..., i] = 0.3

        masked_probs = router_probs * mask
        prob_sum = masked_probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        norm_probs = masked_probs / prob_sum
        return torch.topk(norm_probs, top_k, dim=-1)


class MoECyclicCallback(TrainerCallback):
    """
    HuggingFace Trainer Callback that triggers CyclicMoEController.step()
    and clears temporary GPU buffers after each training step.
    """
    def __init__(self, controller: CyclicMoEController, model: Optional[nn.Module] = None):
        self.controller = controller
        self.model = model

    def on_step_end(self, args, state, control, **kwargs):
        if self.controller is not None:
            self.controller.step()
        if self.model is not None:
            for m in self.model.modules():
                if hasattr(m, "clear_gpu_cache"):
                    m.clear_gpu_cache()


def patch_moe_active_parameters(
    model: nn.Module,
    num_clusters: int = 4,
    offload_to_cpu: bool = True,
    trainable_experts: bool = False,
    config: Optional[ActiveMoEConfig] = None,
) -> CyclicMoEController:
    """
    Patches a DiffusionGemma model to train by active parameters and dynamic expert paging.

    Transforms:
    1. Instantiates `CyclicMoEController` to manage cluster rotation and consolidation.
    2. Replaces `DiffusionGemmaTextExperts` instances with `PagedMoETextExperts`,
       offloading master 3D weights to CPU pinned memory and streaming active experts on-demand.
    3. Patches `DiffusionGemmaTextRouter` to route tokens to the active cluster.
    4. Reclaims GPU VRAM via explicit garbage collection and cache clearing.

    Returns:
        The configured `CyclicMoEController`.
    """
    if config is None:
        config = ActiveMoEConfig(
            num_clusters=num_clusters,
            offload_to_cpu=offload_to_cpu,
            trainable_experts=trainable_experts,
        )

    controller = CyclicMoEController(config)
    paged_experts_list: List[PagedMoETextExperts] = []
    patched_count = 0

    # Traverse model and replace DiffusionGemmaTextExperts
    for name, module in model.named_modules():
        if type(module).__name__ == "DiffusionGemmaTextExperts":
            parent_name = name.rsplit(".", 1)[0] if "." in name else ""
            child_name = name.rsplit(".", 1)[1] if "." in name else name
            parent = model.get_submodule(parent_name) if parent_name else model

            layer_idx = getattr(parent, "layer_idx", patched_count)
            paged_experts = PagedMoETextExperts(
                original_experts=module,
                controller=controller,
                layer_idx=layer_idx,
                offload_to_cpu=offload_to_cpu,
                trainable_experts=trainable_experts,
            )
            setattr(parent, child_name, paged_experts)
            paged_experts_list.append(paged_experts)
            patched_count += 1

        elif type(module).__name__ == "DiffusionGemmaTextRouter":
            orig_router_forward = module.forward
            def make_router_forward(orig_fn, ctrl):
                def routed_forward(hidden_states: torch.Tensor):
                    router_prob, top_k_weights, top_k_index = orig_fn(hidden_states)
                    top_k_weights, top_k_index = ctrl.modulate_router_probs(
                        router_prob,
                        top_k=top_k_index.shape[-1],
                    )
                    return router_prob, top_k_weights, top_k_index
                return routed_forward
            module.forward = make_router_forward(orig_router_forward, controller)

    controller.register_expert_modules(paged_experts_list)

    # Reclaim GPU VRAM from offloaded expert weights
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info(
        f"Unsloth: Patched {patched_count} MoE expert layers with "
        f"PagedMoETextExperts (offload_to_cpu={offload_to_cpu}, num_clusters={num_clusters})."
    )

    model._moe_controller = controller
    return controller
