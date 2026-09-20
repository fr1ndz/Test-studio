// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export type ModelType = "vision" | "audio" | "embeddings" | "text" | "diffusion";
export type TrainingMethod =
  | "qlora"
  | "lora"
  | "full"
  | "cpt"
  | "diffusion_sft"
  | "diffusion_dpo"
  | "diffusion_orpo"
  | "diffusion_grpo"
  | "diffusion_kto"
  | "diffusion_pretrain"
  | "neuroplastic";

export function isTrainingMethod(value: unknown): value is TrainingMethod {
  return (
    value === "qlora" ||
    value === "lora" ||
    value === "full" ||
    value === "cpt" ||
    value === "diffusion_sft" ||
    value === "diffusion_dpo" ||
    value === "diffusion_orpo" ||
    value === "diffusion_grpo" ||
    value === "diffusion_kto" ||
    value === "diffusion_pretrain" ||
    value === "neuroplastic"
  );
}

export function isAdapterMethod(method: TrainingMethod): boolean {
  return method === "lora" || method === "qlora" || method === "cpt";
}

export function isDiffusionMethod(method: TrainingMethod): boolean {
  return (
    method === "diffusion_sft" ||
    method === "diffusion_dpo" ||
    method === "diffusion_orpo" ||
    method === "diffusion_grpo" ||
    method === "diffusion_kto" ||
    method === "diffusion_pretrain"
  );
}
export type DatasetSource = "huggingface" | "upload" | "s3";

/** S3 bucket configuration for loading datasets */
export interface S3Config {
  bucket: string;
  region: string;
  prefix?: string;
  accessKeyId?: string;
  secretAccessKey?: string;
  useIamRole?: boolean;
}
export type DatasetFormat =
  | "auto"
  | "alpaca"
  | "chatml"
  | "sharegpt"
  | "raw"
  | "dpo"
  | "orpo"
  | "grpo"
  | "kto"
  | "prompt_completion"
  | "multimodal";
export type GradientCheckpointing = "none" | "true" | "unsloth" | "mlx";
