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
Advanced Model Transformations and Export Suite.

Provides:
1. Dense-to-MoE Transformation (MoEfication / Sparse Upcycling):
   - Converts dense FFN/MLP layers into Mixture-of-Experts (MoE) layers.
   - Configurable expert counts (e.g. 8, 16, 64, 128), top-k routing, and expert initialization.
   - Supports Sparse Upcycling (DeepSeek/Mixtral style) and Neuron Clustering (MoEfication).

2. 1-Bit & 1.58-Bit Quantization (BitNet b1.58 & Binary 1-Bit):
   - Ternary 1.58-bit {-1, 0, +1} (BitNet b1.58 absmean quantization).
   - Bipolar Binary 1-bit {-1, +1} (sign-magnitude quantization).
   - Unipolar Binary 1-bit {0, 1} (thresholded binary quantization).
   - Hardware-friendly bit-packing into uint8 tensors (8 weights/byte for 1-bit, 4 weights/byte for 1.58-bit).
   - 8x-16x model storage reduction with Safetensors export.
"""

from __future__ import annotations
import copy
import json
import logging
import math
import os
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

__all__ = [
    "OneBitMode",
    "MoEUpcyclingMethod",
    "DenseToMoEConverter",
    "OneBitQuantizer",
    "export_as_moe",
    "export_as_1bit",
    "export_transformed_model",
]


# =====================================================================
# 1. Dense-to-MoE Transformation (MoEfication & Sparse Upcycling)
# =====================================================================

class MoEUpcyclingMethod(str, Enum):
    SPARSE_UPCYCLING = "sparse_upcycling"  # Replicate & perturb weights (Mixtral/DeepSeek style)
    NEURON_CLUSTERING = "neuron_clustering"  # Split intermediate neurons into expert clusters


class MoERouter(nn.Module):
    """Top-K gating router for Mixture-of-Experts."""

    def __init__(self, hidden_dim: int, num_experts: int, top_k: int = 2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.weight = nn.Parameter(torch.empty(num_experts, hidden_dim))
        nn.init.orthogonal_(self.weight)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (..., hidden_dim)
        logits = F.linear(x, self.weight)  # (..., num_experts)
        scores = F.softmax(logits, dim=-1)
        top_scores, top_indices = torch.topk(scores, self.top_k, dim=-1)
        # Normalize top-k probabilities to sum to 1
        top_scores = top_scores / top_scores.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        return top_scores, top_indices


class MoEBlock(nn.Module):
    """Mixture-of-Experts block replacing a single dense MLP."""

    def __init__(
        self,
        experts: nn.ModuleList,
        router: MoERouter,
    ):
        super().__init__()
        self.experts = experts
        self.router = router
        self.num_experts = len(experts)
        self.top_k = router.top_k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x_flat = x.view(-1, orig_shape[-1])
        num_tokens = x_flat.shape[0]

        top_scores, top_indices = self.router(x_flat)  # (N, top_k), (N, top_k)

        out = torch.zeros_like(x_flat)
        for k_idx in range(self.top_k):
            expert_indices = top_indices[:, k_idx]
            weights = top_scores[:, k_idx].unsqueeze(-1)

            for exp_id in range(self.num_experts):
                mask = expert_indices == exp_id
                if mask.any():
                    inp = x_flat[mask]
                    exp_out = self.experts[exp_id](inp)
                    out[mask] += weights[mask] * exp_out

        return out.view(orig_shape)


class DenseToMoEConverter:
    """
    Transforms dense Transformer MLP layers into Mixture-of-Experts (MoE) layers.
    """

    @staticmethod
    def convert_model(
        model: nn.Module,
        num_experts: int = 8,
        num_experts_per_tok: int = 2,
        method: Union[str, MoEUpcyclingMethod] = MoEUpcyclingMethod.SPARSE_UPCYCLING,
        noise_std: float = 0.01,
        target_mlp_names: Optional[List[str]] = None,
    ) -> nn.Module:
        """
        Convert all dense MLPs in the model to MoE blocks.

        Args:
            model: PyTorch model to transform.
            num_experts: Number of experts per MoE layer (default: 8).
            num_experts_per_tok: Top-k experts activated per token (default: 2).
            method: 'sparse_upcycling' or 'neuron_clustering'.
            noise_std: Standard deviation of perturbation noise for sparse upcycling.
            target_mlp_names: Suffixes of MLP modules to replace (default: ['mlp', 'feed_forward']).
        """
        if isinstance(method, MoEUpcyclingMethod):
            pass
        elif hasattr(method, "value"):
            method = MoEUpcyclingMethod(method.value)
        else:
            val = str(method).lower()
            if "." in val:
                val = val.split(".")[-1]
            method = MoEUpcyclingMethod(val)
        if target_mlp_names is None:
            target_mlp_names = ["mlp", "feed_forward", "ffn"]

        converted_count = 0
        for name, module in model.named_modules():
            # Search child modules of layers
            for child_name, child_module in list(module.named_children()):
                if any(child_name == mlp_name or child_name.endswith(f"_{mlp_name}") for mlp_name in target_mlp_names):
                    # Check if already an MoE module
                    if isinstance(child_module, MoEBlock) or hasattr(child_module, "experts"):
                        continue

                    # Detect hidden dimension
                    hidden_dim = None
                    for param in child_module.parameters():
                        hidden_dim = param.shape[-1]
                        break

                    if hidden_dim is None:
                        continue

                    router = MoERouter(hidden_dim=hidden_dim, num_experts=num_experts, top_k=num_experts_per_tok)
                    experts = nn.ModuleList()

                    if method == MoEUpcyclingMethod.SPARSE_UPCYCLING:
                        # Replicate dense MLP with symmetry-breaking perturbation
                        for e in range(num_experts):
                            exp_mlp = copy.deepcopy(child_module)
                            if noise_std > 0.0 and e > 0:
                                with torch.no_grad():
                                    for p in exp_mlp.parameters():
                                        noise = torch.randn_like(p) * noise_std * p.std().clamp(min=1e-4)
                                        p.add_(noise)
                            experts.append(exp_mlp)

                    elif method == MoEUpcyclingMethod.NEURON_CLUSTERING:
                        # MoEfication: split intermediate neurons across experts
                        # For each expert, copy structure and slice intermediate weights
                        for e in range(num_experts):
                            exp_mlp = copy.deepcopy(child_module)
                            with torch.no_grad():
                                for p_name, p in exp_mlp.named_parameters():
                                    if "gate_proj" in p_name or "up_proj" in p_name:
                                        # Slice output dimension
                                        chunk_size = p.shape[0] // num_experts
                                        if chunk_size > 0:
                                            p.copy_(p[e * chunk_size : (e + 1) * chunk_size])
                                    elif "down_proj" in p_name:
                                        # Slice input dimension
                                        chunk_size = p.shape[1] // num_experts
                                        if chunk_size > 0:
                                            p.copy_(p[:, e * chunk_size : (e + 1) * chunk_size])
                            experts.append(exp_mlp)

                    moe_block = MoEBlock(experts=experts, router=router)
                    # Move to same device and dtype
                    device = next(child_module.parameters()).device
                    dtype = next(child_module.parameters()).dtype
                    moe_block.to(device=device, dtype=dtype)

                    setattr(module, child_name, moe_block)
                    converted_count += 1

        logger.info(f"DenseToMoEConverter: Converted {converted_count} dense MLP modules into MoE blocks ({num_experts} experts, top-{num_experts_per_tok}).")

        # Update model config if present
        if hasattr(model, "config"):
            setattr(model.config, "num_local_experts", num_experts)
            setattr(model.config, "num_experts_per_tok", num_experts_per_tok)
            setattr(model.config, "is_moe", True)

        return model


# =====================================================================
# 2. 1-Bit & 1.58-Bit Quantization (BitNet b1.58 & Binary 1-Bit)
# =====================================================================

class OneBitMode(str, Enum):
    TERNARY = "ternary"            # {-1, 0, +1} BitNet b1.58 absmean
    BINARY_BIPOLAR = "binary_pm1"  # {-1, +1} sign-magnitude
    BINARY_UNIPOLAR = "binary_01"  # {0, 1} thresholded binary


class OneBitQuantizer:
    """
    1-Bit and 1.58-Bit Quantizer with Bit-Packing.

    Modes:
    - TERNARY: W in {-1, 0, +1}, W_quant = round(clamp(W / mean(|W|), -1, 1))
    - BINARY_BIPOLAR: W in {-1, +1}, W_quant = sign(W) * mean(|W|)
    - BINARY_UNIPOLAR: W in {0, 1}, W_quant = (W > threshold) * scale
    """

    @staticmethod
    def quantize_tensor(
        tensor: torch.Tensor,
        mode: OneBitMode = OneBitMode.TERNARY,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize a floating-point weight tensor to 1-bit or 1.58-bit representation.

        Returns:
            Tuple of (discrete_weights, scale).
        """
        eps = 1e-6
        if mode == OneBitMode.TERNARY:
            # BitNet b1.58: absmean scaling
            scale = tensor.abs().mean().clamp(min=eps)
            scaled = tensor / scale
            discrete = torch.round(torch.clamp(scaled, -1.0, 1.0))
            return discrete, scale

        elif mode == OneBitMode.BINARY_BIPOLAR:
            scale = tensor.abs().mean().clamp(min=eps)
            discrete = torch.sign(tensor)
            # Replace 0s with +1
            discrete[discrete == 0] = 1.0
            return discrete, scale

        elif mode == OneBitMode.BINARY_UNIPOLAR:
            threshold = tensor.mean()
            scale = tensor.abs().max().clamp(min=eps)
            discrete = (tensor > threshold).float()
            return discrete, scale

        else:
            raise ValueError(f"Unknown OneBitMode: {mode}")

    @staticmethod
    def pack_binary(discrete_weights: torch.Tensor) -> torch.Tensor:
        """
        Pack 1-bit binary weights {-1, +1} or {0, 1} into uint8 bytes (8 weights per byte).
        """
        # Map {-1, +1} to {0, 1}
        bits = (discrete_weights > 0).to(torch.uint8)
        orig_shape = bits.shape
        flat = bits.contiguous().view(-1)
        pad_len = (8 - (flat.numel() % 8)) % 8
        if pad_len > 0:
            flat = F.pad(flat, (0, pad_len), value=0)

        reshaped = flat.view(-1, 8)
        packed = torch.zeros(reshaped.shape[0], dtype=torch.uint8, device=discrete_weights.device)
        for bit_idx in range(8):
            packed |= (reshaped[:, bit_idx] << bit_idx)

        return packed

    @staticmethod
    def unpack_binary(packed_tensor: torch.Tensor, orig_numel: int, bipolar: bool = True) -> torch.Tensor:
        """
        Unpack uint8 bytes back into discrete binary weights {-1, +1} or {0, 1}.
        """
        unpacked = torch.zeros(packed_tensor.shape[0] * 8, dtype=torch.float32, device=packed_tensor.device)
        for bit_idx in range(8):
            bit_vals = ((packed_tensor >> bit_idx) & 1).float()
            unpacked[bit_idx::8] = bit_vals

        unpacked = unpacked[:orig_numel]
        if bipolar:
            # Map {0, 1} -> {-1, +1}
            unpacked = unpacked * 2.0 - 1.0
        return unpacked

    @staticmethod
    def pack_ternary(discrete_weights: torch.Tensor) -> torch.Tensor:
        """
        Pack 1.58-bit ternary weights {-1, 0, +1} into uint8 bytes (4 weights per byte, 2 bits each).
        Mapping: 0 -> 0b00, +1 -> 0b01, -1 -> 0b10.
        """
        code = torch.zeros_like(discrete_weights, dtype=torch.uint8)
        code[discrete_weights == 1.0] = 0b01
        code[discrete_weights == -1.0] = 0b10

        flat = code.contiguous().view(-1)
        pad_len = (4 - (flat.numel() % 4)) % 4
        if pad_len > 0:
            flat = F.pad(flat, (0, pad_len), value=0)

        reshaped = flat.view(-1, 4)
        packed = (
            (reshaped[:, 0])
            | (reshaped[:, 1] << 2)
            | (reshaped[:, 2] << 4)
            | (reshaped[:, 3] << 6)
        )
        return packed

    @staticmethod
    def unpack_ternary(packed_tensor: torch.Tensor, orig_numel: int) -> torch.Tensor:
        """
        Unpack 2-bit ternary weights back into {-1, 0, +1}.
        """
        unpacked = torch.zeros(packed_tensor.shape[0] * 4, dtype=torch.float32, device=packed_tensor.device)
        shifts = [0, 2, 4, 6]
        for idx, shift in enumerate(shifts):
            code = (packed_tensor >> shift) & 0b11
            vals = torch.zeros_like(code, dtype=torch.float32)
            vals[code == 0b01] = 1.0
            vals[code == 0b10] = -1.0
            unpacked[idx::4] = vals

        return unpacked[:orig_numel]

    @staticmethod
    def quantize_model(
        model: nn.Module,
        mode: Union[str, OneBitMode] = OneBitMode.TERNARY,
        pack_bits: bool = True,
        exclude_modules: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Quantize all Linear modules in a model to 1-bit or 1.58-bit.

        Returns:
            Dictionary containing packed state_dict, scales, and metadata.
        """
        if isinstance(mode, OneBitMode):
            pass
        elif hasattr(mode, "value"):
            mode = OneBitMode(mode.value)
        else:
            val = str(mode).lower()
            if "." in val:
                val = val.split(".")[-1]
            mode = OneBitMode(val)
        if exclude_modules is None:
            exclude_modules = ["lm_head", "embed_tokens", "router", "vision_tower"]

        packed_state_dict: Dict[str, torch.Tensor] = {}
        scales: Dict[str, float] = {}
        shapes: Dict[str, List[int]] = {}

        for name, param in model.named_parameters():
            if not param.requires_grad and not any(k in name for k in ("weight", "bias")):
                continue

            # Check exclusion list
            if any(ex in name for ex in exclude_modules) or not name.endswith(".weight") or param.dim() < 2:
                packed_state_dict[name] = param.clone().detach()
                continue

            # Quantize weight matrix
            discrete, scale = OneBitQuantizer.quantize_tensor(param.data, mode=mode)
            scales[name] = scale.item()
            shapes[name] = list(param.shape)

            if pack_bits:
                if mode == OneBitMode.TERNARY:
                    packed = OneBitQuantizer.pack_ternary(discrete)
                else:
                    packed = OneBitQuantizer.pack_binary(discrete)
                packed_state_dict[f"{name}.packed"] = packed
            else:
                packed_state_dict[name] = discrete

        metadata = {
            "quant_mode": mode.value,
            "is_packed": pack_bits,
            "scales": scales,
            "shapes": shapes,
            "bits_per_weight": 1.58 if mode == OneBitMode.TERNARY else 1.0,
        }

        logger.info(
            f"OneBitQuantizer: Quantized {len(scales)} weight matrices to {mode.value} "
            f"({metadata['bits_per_weight']} bits/weight, packed={pack_bits})."
        )
        return {"state_dict": packed_state_dict, "metadata": metadata}


# =====================================================================
# 3. High-Level Export Functions
# =====================================================================

def export_as_moe(
    model: nn.Module,
    tokenizer: Any,
    save_directory: str,
    num_experts: int = 8,
    num_experts_per_tok: int = 2,
    method: str = "sparse_upcycling",
    safe_serialization: bool = True,
    **kwargs,
) -> str:
    """
    Transform dense model into an MoE architecture and save checkpoints.
    """
    os.makedirs(save_directory, exist_ok=True)
    logger.info(f"Exporting model transformed to MoE ({num_experts} experts, top-{num_experts_per_tok}) -> {save_directory}")

    # Convert model to MoE
    moe_model = DenseToMoEConverter.convert_model(
        model=model,
        num_experts=num_experts,
        num_experts_per_tok=num_experts_per_tok,
        method=method,
        **kwargs,
    )

    # Save tokenizer
    if tokenizer is not None and hasattr(tokenizer, "save_pretrained"):
        tokenizer.save_pretrained(save_directory)

    # Save model weights & config in safetensors format
    if hasattr(moe_model, "save_pretrained"):
        moe_model.save_pretrained(save_directory, safe_serialization=True)
    else:
        from safetensors.torch import save_file
        save_file(moe_model.state_dict(), os.path.join(save_directory, "model.safetensors"))

    # Save MoE metadata
    moe_meta = {
        "architecture_type": "MoE",
        "num_local_experts": num_experts,
        "num_experts_per_tok": num_experts_per_tok,
        "upcycling_method": method,
        "format": "safetensors",
    }
    with open(os.path.join(save_directory, "moe_config.json"), "w", encoding="utf-8") as f:
        json.dump(moe_meta, f, indent=2)

    logger.info(f"MoE Export complete (safetensors): {save_directory}")
    return save_directory


def export_as_1bit(
    model: nn.Module,
    tokenizer: Any,
    save_directory: str,
    mode: str = "ternary",
    pack_bits: bool = True,
    safe_serialization: bool = True,
    **kwargs,
) -> str:
    """
    Quantize model to 1-bit or 1.58-bit (BitNet b1.58) and export ultra-compact checkpoint in safetensors.
    """
    os.makedirs(save_directory, exist_ok=True)
    logger.info(f"Exporting model in {mode} 1-bit quantization (packed={pack_bits}) to safetensors -> {save_directory}")

    # Quantize
    quant_result = OneBitQuantizer.quantize_model(
        model=model,
        mode=mode,
        pack_bits=pack_bits,
        **kwargs,
    )

    # Save tokenizer
    if tokenizer is not None and hasattr(tokenizer, "save_pretrained"):
        tokenizer.save_pretrained(save_directory)

    # Save weights strictly via safetensors
    state_dict = quant_result["state_dict"]
    metadata = quant_result["metadata"]
    metadata["format"] = "safetensors"

    from safetensors.torch import save_file
    save_file(state_dict, os.path.join(save_directory, "model.safetensors"))

    # Save 1-bit metadata and scales
    with open(os.path.join(save_directory, "bit_quant_config.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # Save base config if present
    if hasattr(model, "config") and hasattr(model.config, "save_pretrained"):
        model.config.save_pretrained(save_directory)

    logger.info(f"1-Bit Export complete: {save_directory}")
    return save_directory


def export_transformed_model(
    model: nn.Module,
    tokenizer: Any,
    save_directory: str,
    to_moe: bool = False,
    to_1bit: bool = False,
    moe_kwargs: Optional[Dict[str, Any]] = None,
    bit_kwargs: Optional[Dict[str, Any]] = None,
    safe_serialization: bool = True,
) -> str:
    """
    Unified transformation and export pipeline:
    Supports Dense -> MoE upcycling, followed by optional 1-Bit quantization.
    """
    os.makedirs(save_directory, exist_ok=True)
    moe_kwargs = moe_kwargs or {}
    bit_kwargs = bit_kwargs or {}

    current_model = model

    # 1. Apply Dense -> MoE if requested
    if to_moe:
        logger.info("Pipeline step 1: Converting Dense to MoE...")
        current_model = DenseToMoEConverter.convert_model(
            model=current_model,
            **moe_kwargs,
        )

    # 2. Apply 1-Bit Quantization if requested
    if to_1bit:
        logger.info("Pipeline step 2: Quantizing to 1-Bit / 1.58-Bit...")
        return export_as_1bit(
            model=current_model,
            tokenizer=tokenizer,
            save_directory=save_directory,
            safe_serialization=safe_serialization,
            **bit_kwargs,
        )

    # 3. Standard save if only MoE
    if to_moe:
        return export_as_moe(
            model=current_model,
            tokenizer=tokenizer,
            save_directory=save_directory,
            safe_serialization=safe_serialization,
            **moe_kwargs,
        )

    # Fallback to standard save
    if hasattr(current_model, "save_pretrained"):
        current_model.save_pretrained(save_directory, safe_serialization=safe_serialization)
    if tokenizer is not None and hasattr(tokenizer, "save_pretrained"):
        tokenizer.save_pretrained(save_directory)

    return save_directory
