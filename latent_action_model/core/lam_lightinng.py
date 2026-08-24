from typing import Dict, Tuple, Optional, Callable, Iterable, Any, List
from pathlib import Path
from collections import Counter
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim import Optimizer
from lightning import LightningModule
import lightning.pytorch as pl
OptimizerCallable = Callable[[Iterable], Optimizer]
import wandb
from .lam_model import LatentLAMModel
import logging
logging.basicConfig(format='%(message)s', level=logging.INFO)
import os
import shutil
from .utils.utils import eef_reconstruction_loss, charbonnier_loss
from ..data_loader.video_aug import (
    LAM_IMAGE_HW,
    LAM_PATCH_SIZE,
    gpu_two_view_video_aug,
)


class VJEPA_LAM(LightningModule):
    
    def __init__(
        self,
        dim: int = 1024,
        num_heads: int = 16,
        ffn_expansion_factor: int = 2,
        enc_layers: int = 4,
        codebook_size: int = 16,
        code_dim: int = 128,
        max_state_dim: int = 32,
        num_frames: int = 5,
        num_queries: int = 1,
        vq_kwargs: Optional[Dict[str, Any]] = None,
        dec_layers: int = 4,
        dropout: float = 0.1,
        lambda_aux: float = 0.2,
        loss_type: str = "l1",
        state_loss_type: str = "l1",
        project: str = 'UniVLA-latent_action_model',
        task_name: str = 'vjepa_lam',
        wandb_offline: bool = False,
        optimizer: OptimizerCallable = torch.optim.AdamW,
        weight_decay: float = 0.01,
        exclude_bias_norm_from_wd: bool = False,
        make_data_pair: bool = False,
        output_dir: str = "output_pairs",
        vision_model_id: str = "facebook/vjepa2-vitl-fpc64-256",
        warmup_steps: int = 0,
        lambda_diversity: float = 0.1,
        norm_latents: bool = False,
        norm_latents_type: str="l2",        
        disable_vq: bool = False,
        vq_type: str = "nsvq",
        enc_add_state: bool = False,
        enc_modal_mask: bool = False,
        latent_layer_to_use: Any = 23,
        multi_input: bool = False,
        num_embodiments: int = 32,
        image_hw: Tuple[int, int] = LAM_IMAGE_HW,
        patch_size: int = LAM_PATCH_SIZE,
        image_aug: bool = True,
        dual_view_aug: bool = False,
        decoder_last_ln: bool = True,
        **kwargs
    ):
        super().__init__()
        torch.cuda.empty_cache()
        torch.set_float32_matmul_precision('medium')

        self.save_hyperparameters()
        self.image_hw = (int(image_hw[0]), int(image_hw[1]))
        self.patch_size = int(patch_size)
        self.image_aug = image_aug
        self.dual_view_aug = bool(dual_view_aug)
        if self.patch_size != LAM_PATCH_SIZE:
            raise ValueError(
                f"Unsupported LAM patch_size={self.patch_size}. "
                f"Only {LAM_PATCH_SIZE} is supported in this branch."
            )
        
        self.lam = LatentLAMModel(
            dim=dim,
            num_heads=num_heads,
            ffn_expansion_factor=ffn_expansion_factor,
            enc_layers=enc_layers,
            codebook_size=codebook_size,
            code_dim=code_dim,
            num_frames=num_frames,
            num_queries=num_queries,
            dec_layers=dec_layers,
            dropout=dropout,
            vision_model_id=vision_model_id,
            vq_kwargs=vq_kwargs,
            norm_latents=norm_latents,
            norm_latents_type=norm_latents_type,
            disable_vq=disable_vq,
            vq_type=vq_type,
            enc_add_state=enc_add_state,
            enc_modal_mask=enc_modal_mask,
            latent_layer_to_use=latent_layer_to_use,
            multi_input=multi_input,
            max_state_dim=max_state_dim,
            num_embodiments=num_embodiments,
            image_hw=self.image_hw,
            patch_size=self.patch_size,
            decoder_last_ln=decoder_last_ln,
        )


        
        self.optimizer = optimizer
        self.weight_decay = weight_decay
        self.exclude_bias_norm_from_wd = bool(exclude_bias_norm_from_wd)
        self.codebook_size = codebook_size
        self.warmup_steps = int(warmup_steps)
        
        self.lambda_aux = lambda_aux
        self.lambda_diversity = lambda_diversity

        self.make_data_pair = make_data_pair
        self.output_dir = output_dir
        self._wandb_project = project
        self._wandb_task_name = task_name
        self._wandb_mode = "offline" if wandb_offline else "online"
        self._manual_wandb_enabled = os.environ.get("LAM_ENABLE_MANUAL_WANDB", "1") != "0"
        self._wandb_initialized = False

        self.loss_type = loss_type
        if state_loss_type not in {"l1", "l2"}:
            raise ValueError(f"Unsupported state_loss_type: {state_loss_type}")
        self.state_loss_type = state_loss_type
        # Run expensive unused-params scan only once at training start.
        self._unused_params_scanned = False
        self._optimizer_group_summary_printed = False
        try:
            self._spike_loss_threshold = float(os.environ.get("LAM_SPIKE_LOSS_THRESHOLD", "1.0"))
        except ValueError:
            self._spike_loss_threshold = 1.0
        try:
            self._spike_arm_loss_threshold = float(os.environ.get("LAM_SPIKE_ARM_LOSS_THRESHOLD", "0.2"))
        except ValueError:
            self._spike_arm_loss_threshold = 0.2
        try:
            self._spike_log_cooldown = max(1, int(os.environ.get("LAM_SPIKE_LOG_COOLDOWN", "200")))
        except ValueError:
            self._spike_log_cooldown = 200
        try:
            self._spike_log_max_logs = max(1, int(os.environ.get("LAM_SPIKE_MAX_LOGS", "50")))
        except ValueError:
            self._spike_log_max_logs = 50
        self._spike_log_count = 0
        self._last_spike_log_step = -10**12
        self._spike_armed = False
        self._spike_arm_step: Optional[int] = None
        self._last_step_tensors: Dict[str, Tensor] = {}

    def _is_global_zero(self) -> bool:
        trainer = getattr(self, "trainer", None)
        if trainer is not None:
            return bool(getattr(trainer, "is_global_zero", True))
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank() == 0
        return True

    def _maybe_init_wandb(self) -> None:
        if not self._manual_wandb_enabled or self._wandb_initialized or not self._is_global_zero():
            return
        if wandb.run is None:
            wandb.init(
                project=self._wandb_project,
                name=self._wandb_task_name,
                reinit=True,
                mode=self._wandb_mode,
            )
        self._wandb_initialized = True

    def _should_log_train_wandb(self) -> bool:
        if not self._manual_wandb_enabled or not self._wandb_initialized or not self._is_global_zero():
            return False
        if wandb.run is None:
            return False
        trainer = getattr(self, "trainer", None)
        every_n = max(1, int(getattr(trainer, "log_every_n_steps", 1) or 1)) if trainer is not None else 1
        return int(self.global_step) % every_n == 0

    def setup(self, stage: str) -> None:
        super().setup(stage)
        self._maybe_init_wandb()

    def _get_current_lr(self) -> Optional[float]:
        trainer = getattr(self, "trainer", None)
        if trainer is None:
            return None
        optimizers = getattr(trainer, "optimizers", None)
        if not optimizers:
            return None
        param_groups = getattr(optimizers[0], "param_groups", None)
        if not param_groups:
            return None
        lr = param_groups[0].get("lr", None)
        return float(lr) if lr is not None else None

    @staticmethod
    def _to_float(v: Any, default: float = 0.0) -> float:
        if isinstance(v, torch.Tensor):
            return float(v.detach().float().item())
        if v is None:
            return default
        try:
            return float(v)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _safe_quantile_1d(
        values: Tensor,
        quantiles: tuple[float, ...] = (0.5, 0.9, 0.99),
        max_elems: int = 200_000,
    ) -> Tensor:
        """Compute quantiles on at most `max_elems` values to avoid backend size limits."""
        if values.ndim != 1:
            values = values.reshape(-1)
        if values.numel() == 0:
            return torch.full((len(quantiles),), float("nan"), device=values.device, dtype=values.dtype)

        sampled = values
        if sampled.numel() > max_elems:
            step = max(1, sampled.numel() // max_elems)
            sampled = sampled[::step]
            if sampled.numel() > max_elems:
                sampled = sampled[:max_elems]

        q = torch.tensor(quantiles, device=sampled.device, dtype=sampled.dtype)
        try:
            return torch.quantile(sampled, q)
        except RuntimeError:
            # Fallback for rare backend-specific quantile limitations.
            sampled_cpu = sampled.cpu()
            q_cpu = q.cpu()
            result_cpu = torch.quantile(sampled_cpu, q_cpu)
            return result_cpu.to(values.device)

    @staticmethod
    def _summarize_spike_tensor(name: str, tensor: Optional[Tensor]) -> str:
        """Summarize tensor scale with norm/value quantiles for spike debugging."""
        if tensor is None:
            return f"{name}=none"
        if not isinstance(tensor, torch.Tensor):
            return f"{name}=invalid({type(tensor).__name__})"

        with torch.no_grad():
            t = tensor.detach().float()
            if t.numel() == 0:
                return f"{name}=empty"

            finite_mask = torch.isfinite(t)
            finite_ratio = float(finite_mask.float().mean().item())
            if finite_ratio < 1.0:
                t = t[finite_mask]
                if t.numel() == 0:
                    return f"{name}=nonfinite(finite_ratio={finite_ratio:.3f})"

            values_abs = t.abs().reshape(-1)
            values_q = VJEPA_LAM._safe_quantile_1d(values_abs)
            values_max = float(values_abs.max().item())

            if t.ndim > 0:
                norms = t.norm(dim=-1).reshape(-1)
            else:
                norms = values_abs
            norms_q = VJEPA_LAM._safe_quantile_1d(norms)
            norms_mean = float(norms.mean().item())
            norms_max = float(norms.max().item())

            return (
                f"{name}[norm_mean={norms_mean:.4g},norm_q50={float(norms_q[0].item()):.4g},"
                f"norm_q90={float(norms_q[1].item()):.4g},norm_q99={float(norms_q[2].item()):.4g},"
                f"norm_max={norms_max:.4g},abs_q99={float(values_q[2].item()):.4g},"
                f"abs_max={values_max:.4g},finite={finite_ratio:.3f}]"
            )

    def _maybe_log_spike_batch(
        self,
        batch: Dict,
        loss_value: float,
        aux_losses: Dict[str, Any],
        current_lr: Optional[float],
        recon: Optional[Tensor] = None,
        target: Optional[Tensor] = None,
        dec_in: Optional[Tensor] = None,
    ) -> None:
        # Arm spike diagnostics only after loss first reaches a stable low regime.
        if not self._spike_armed:
            if loss_value <= self._spike_arm_loss_threshold:
                self._spike_armed = True
                self._spike_arm_step = int(self.global_step)
                if self._is_global_zero():
                    arm_msg = (
                        f"[SPIKE_ARMED] step={int(self.global_step)} "
                        f"loss={loss_value:.4f} arm_threshold={self._spike_arm_loss_threshold:.4f}"
                    )
                    if getattr(self, "_trainer", None) is not None:
                        self.print(arm_msg)
                    else:
                        print(arm_msg)
            return
        if loss_value < self._spike_loss_threshold:
            return
        if not self._is_global_zero():
            return
        if self.global_step - self._last_spike_log_step < self._spike_log_cooldown:
            return
        if self._spike_log_count >= self._spike_log_max_logs:
            return

        dataset_names = [str(x) for x in batch.get("dataset_names", [])]
        trajectory_ids = list(batch.get("trajectory_ids", []))
        base_indices_obj = batch.get("base_indices", [])
        if isinstance(base_indices_obj, torch.Tensor):
            base_indices = base_indices_obj.detach().cpu().tolist()
        else:
            base_indices = list(base_indices_obj)

        counts = Counter(dataset_names)
        top_sources = ", ".join([f"{k}:{v}" for k, v in counts.most_common(4)]) if counts else "unknown"

        examples: list[str] = []
        n = min(len(dataset_names), 8)
        for i in range(n):
            traj = trajectory_ids[i] if i < len(trajectory_ids) else -1
            base = base_indices[i] if i < len(base_indices) else -1
            examples.append(f"{dataset_names[i]}#{traj}@{base}")
        examples_str = "; ".join(examples) if examples else "none"

        recon_loss_value = self._to_float(aux_losses.get("recon_loss"), default=float("nan"))
        state = self._to_float(
            aux_losses.get("state_loss", aux_losses.get("state_loss_skipped")),
            default=float("nan"),
        )
        lr_str = f"{current_lr:.6g}" if current_lr is not None else "n/a"
        msg = (
            f"[SPIKE] step={int(self.global_step)} loss={loss_value:.4f} "
            f"recon={recon_loss_value:.4f} state={state:.4f} lr={lr_str} "
            f"sources={top_sources} examples={examples_str}"
        )
        if getattr(self, "_trainer", None) is not None:
            self.print(msg)
        else:
            print(msg)

        err_summary = "recon_minus_tgt=unavailable"
        if isinstance(recon, torch.Tensor) and isinstance(target, torch.Tensor):
            if tuple(recon.shape) == tuple(target.shape):
                err_summary = self._summarize_spike_tensor("recon_minus_tgt", recon - target)
            else:
                err_summary = (
                    "recon_minus_tgt=shape_mismatch("
                    f"recon={tuple(recon.shape)},tgt={tuple(target.shape)})"
                )
        stats_msg = (
            f"[SPIKE_STATS] step={int(self.global_step)} "
            f"{self._summarize_spike_tensor('recon', recon)} | "
            f"{self._summarize_spike_tensor('tgt', target)} | "
            f"{self._summarize_spike_tensor('dec_in', dec_in)} | "
            f"{err_summary}"
        )
        if getattr(self, "_trainer", None) is not None:
            self.print(stats_msg)
        else:
            print(stats_msg)
        self._spike_log_count += 1
        self._last_spike_log_step = int(self.global_step)

    def _build_optimizer_param_groups(self) -> Tuple[List[Dict[str, Any]], List[str], List[str]]:
        """Build AdamW decay/no-decay param groups for LAM training."""
        norm_module_types = (
            nn.LayerNorm,
            nn.BatchNorm1d,
            nn.BatchNorm2d,
            nn.BatchNorm3d,
            nn.SyncBatchNorm,
            nn.GroupNorm,
            nn.InstanceNorm1d,
            nn.InstanceNorm2d,
            nn.InstanceNorm3d,
        )

        trainable_param_names: Dict[int, str] = {}
        for full_name, param in self.named_parameters():
            if param.requires_grad and id(param) not in trainable_param_names:
                trainable_param_names[id(param)] = full_name

        decay_params: List[torch.nn.Parameter] = []
        no_decay_params: List[torch.nn.Parameter] = []
        decay_names: List[str] = []
        no_decay_names: List[str] = []
        assigned_param_ids = set()

        for module_name, module in self.named_modules():
            is_norm_module = isinstance(module, norm_module_types)
            for param_name, param in module.named_parameters(recurse=False):
                if not param.requires_grad:
                    continue
                param_id = id(param)
                if param_id in assigned_param_ids:
                    continue

                full_name = f"{module_name}.{param_name}" if module_name else param_name
                is_bias_like = (param_name == "b") or param_name.endswith("bias")
                if is_norm_module or is_bias_like:
                    no_decay_params.append(param)
                    no_decay_names.append(full_name)
                else:
                    decay_params.append(param)
                    decay_names.append(full_name)
                assigned_param_ids.add(param_id)

        unassigned_ids = set(trainable_param_names.keys()) - assigned_param_ids
        extra_ids = assigned_param_ids - set(trainable_param_names.keys())
        if unassigned_ids or extra_ids:
            missing_names = [trainable_param_names[param_id] for param_id in sorted(unassigned_ids)]
            extra_names = []
            if extra_ids:
                reverse_assigned = {id(p): n for n, p in zip(decay_names + no_decay_names, decay_params + no_decay_params)}
                extra_names = [reverse_assigned.get(param_id, str(param_id)) for param_id in sorted(extra_ids)]
            raise RuntimeError(
                "Optimizer param grouping mismatch: "
                f"missing={missing_names[:10]}, extra={extra_names[:10]}"
            )

        param_groups: List[Dict[str, Any]] = [{"params": decay_params}]
        if no_decay_params:
            param_groups.append({"params": no_decay_params, "weight_decay": 0.0})
        return param_groups, decay_names, no_decay_names

    def _log_optimizer_group_summary(self, decay_names: List[str], no_decay_names: List[str]) -> None:
        """Print one-time param-group summary on rank0 for sanity check."""
        if self._optimizer_group_summary_printed:
            return
        is_rank0 = True
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            is_rank0 = torch.distributed.get_rank() == 0
        if not is_rank0:
            return

        decay_preview = ", ".join(decay_names[:5]) if decay_names else "None"
        no_decay_preview = ", ".join(no_decay_names[:5]) if no_decay_names else "None"
        msg = (
            "[optimizer] exclude_bias_norm_from_wd=True | "
            f"decay={len(decay_names)} no_decay={len(no_decay_names)} | "
            f"decay_examples=[{decay_preview}] | "
            f"no_decay_examples=[{no_decay_preview}]"
        )
        if getattr(self, "_trainer", None) is not None:
            self.print(msg)
        else:
            print(msg)
        self._optimizer_group_summary_printed = True

    def shared_step(self, batch: Dict) -> Tuple[Tensor, Dict]:
        return self._compute_step(batch=batch, vq_training=True)

    @staticmethod
    def _as_nhwc_uint8_clip(clip: torch.Tensor) -> torch.Tensor:
        """Convert one clip to uint8 [T,H,W,C] for GPU augmentation."""
        if clip.ndim != 4:
            raise ValueError(f"Expected 4D clip tensor, got shape {tuple(clip.shape)}")
        if clip.shape[-1] == 3:
            out = clip
        elif clip.shape[1] == 3:
            out = clip.permute(0, 2, 3, 1).contiguous()
        else:
            raise ValueError(f"Unable to infer channel dimension for clip shape {tuple(clip.shape)}")

        if out.dtype == torch.uint8:
            return out.contiguous()

        if out.is_floating_point():
            max_val = float(out.max().item()) if out.numel() > 0 else 0.0
            scale = 255.0 if max_val <= 1.0 + 1e-6 else 1.0
            out = out.mul(scale).clamp(0, 255).to(torch.uint8)
        else:
            out = out.clamp(0, 255).to(torch.uint8)
        return out.contiguous()

    @staticmethod
    def _as_nhwc_uint8_batch(videos: torch.Tensor) -> torch.Tensor:
        """Convert batched videos to uint8 [B,T,H,W,C] once before GPU augmentation."""
        if videos.ndim != 5:
            raise ValueError(f"Expected 5D videos tensor, got shape {tuple(videos.shape)}")

        if videos.shape[-1] == 3:
            out = videos
        elif videos.shape[2] == 3:
            out = videos.permute(0, 1, 3, 4, 2)
        else:
            raise ValueError(f"Unable to infer channel dimension for batched videos: {tuple(videos.shape)}")

        if out.dtype == torch.uint8:
            return out.contiguous()

        if out.is_floating_point():
            max_val = float(out.max().item()) if out.numel() > 0 else 0.0
            scale = 255.0 if max_val <= 1.0 + 1e-6 else 1.0
            out = out.mul(scale).clamp(0, 255).to(torch.uint8)
        else:
            out = out.clamp(0, 255).to(torch.uint8)
        return out.contiguous()

    def _augment_video_list_on_gpu(self, videos: List[torch.Tensor], training: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply two-view GPU augmentation for heterogeneous-shape batches."""
        groups: Dict[Tuple[int, int], List[Tuple[int, torch.Tensor]]] = {}
        for idx, clip in enumerate(videos):
            if not isinstance(clip, torch.Tensor):
                clip = torch.as_tensor(clip, device=self.device)
            clip_nhwc = self._as_nhwc_uint8_clip(clip)
            key = (int(clip_nhwc.shape[-3]), int(clip_nhwc.shape[-2]))
            groups.setdefault(key, []).append((idx, clip_nhwc))

        view1_by_index: Dict[int, torch.Tensor] = {}
        view2_by_index: Dict[int, torch.Tensor] = {}
        for items in groups.values():
            group_batch = torch.stack([clip for _, clip in items], dim=0).contiguous()
            video1, video2 = gpu_two_view_video_aug(
                group_batch,
                output_size=self.image_hw,
                training=training,
                dual_view_aug=self.dual_view_aug,
            )
            for pos, (orig_idx, _) in enumerate(items):
                view1_by_index[orig_idx] = video1[pos]
                view2_by_index[orig_idx] = video2[pos]

        ordered_view1 = [view1_by_index[i] for i in range(len(videos))]
        ordered_view2 = [view2_by_index[i] for i in range(len(videos))]
        return torch.stack(ordered_view1, dim=0), torch.stack(ordered_view2, dim=0)

    def transfer_batch_to_device(self, batch: Any, device: torch.device, dataloader_idx: int) -> Any:
        """Move batch to device and build video1/video2 on GPU when raw uint8 clips are provided."""
        batch = super().transfer_batch_to_device(batch, device, dataloader_idx)
        if not isinstance(batch, dict) or "videos" not in batch:
            return batch

        training_aug = bool(self.training) and self.image_aug
        videos = batch["videos"]

        if isinstance(videos, torch.Tensor):
            if videos.ndim == 5 and (videos.shape[-1] == 3 or videos.shape[2] == 3):
                videos_nhwc = self._as_nhwc_uint8_batch(videos)
                video1, video2 = gpu_two_view_video_aug(
                    videos_nhwc,
                    output_size=self.image_hw,
                    training=training_aug,
                    dual_view_aug=self.dual_view_aug,
                )
                batch["videos"] = video1
                batch["dec_videos"] = video2
            elif "dec_videos" not in batch:
                # Compatibility path for older collate outputs.
                batch["dec_videos"] = videos
            return batch

        if isinstance(videos, list):
            if len(videos) == 0:
                raise ValueError("Received empty video list in batch.")
            video1, video2 = self._augment_video_list_on_gpu(videos, training=training_aug)
            batch["videos"] = video1
            batch["dec_videos"] = video2
            return batch

        raise TypeError(f"Unsupported 'videos' type in batch: {type(videos)!r}")
    
    def shared_inference_step(self, batch: Dict) -> Tuple[Tensor, Dict]:
        return self._compute_step(batch=batch, vq_training=False)

    def _compute_step(self, batch: Dict, vq_training: bool) -> Tuple[Tensor, Dict]:
        videos = batch["videos"]
        states = batch.get("states", batch.get("proprio", None))
        if states is None:
            raise KeyError("LAM training/inference requires `states` (or legacy `proprio`) in batch.")
        state_mask = batch.get("state_mask", None)
        dec_videos = batch["dec_videos"]
        if vq_training:
            if "embodiment_ids" not in batch:
                raise KeyError("LAM training requires `embodiment_ids` in batch.")
            embodiment_ids = batch["embodiment_ids"]
            if not isinstance(embodiment_ids, torch.Tensor):
                raise TypeError(
                    f"LAM training expects `embodiment_ids` as torch.Tensor, got {type(embodiment_ids).__name__}."
                )
        else:
            embodiment_ids = batch.get("embodiment_ids", None)
            if embodiment_ids is not None and not isinstance(embodiment_ids, torch.Tensor):
                raise TypeError(
                    f"LAM inference expects `embodiment_ids` as torch.Tensor when provided, got {type(embodiment_ids).__name__}."
                )
        # print("videos shape:", videos.shape)
        if vq_training:
            recon, dec_in, tgt, perplexity, indices, delta_s_pred, features, _, entropy_loss, vq_loss = self.lam(
                videos,
                states,
                dec_videos,
                state_mask=state_mask,
                embodiment_ids=embodiment_ids,
            )
        else:
            recon, dec_in, tgt, perplexity, indices, delta_s_pred, features, _, entropy_loss, vq_loss = self.lam.inference(
                videos,
                states,
                dec_videos,
                state_mask=state_mask,
                embodiment_ids=embodiment_ids,
            )

        if recon is None:
            raise RuntimeError("Decoder output is None; check latent_mode / decoder setup.")
        if recon.shape != tgt.shape:
            raise RuntimeError(f"Decoder output shape {recon.shape} mismatch target {tgt.shape}.")

        target = tgt
        if vq_training:
            # Keep detached references for spike-only diagnostics in training_step.
            self._last_step_tensors = {
                "recon": recon.detach(),
                "target": target.detach(),
                "dec_in": dec_in.detach(),
            }
        else:
            self._last_step_tensors = {}
        # recon_loss = F.mse_loss(recon, target)
        with torch.no_grad():
            cos_sim_metric = F.cosine_similarity(recon, target, dim=-1).mean()
            l1_loss_metric = F.l1_loss(recon, target)
        if self.loss_type == "l1":
            recon_loss = F.l1_loss(recon, target)
            loss = recon_loss
        elif self.loss_type == "smooth_l1":
            recon_loss = F.smooth_l1_loss(recon, target, beta=0.1)
            loss = recon_loss
        elif self.loss_type == "cos":
            cos_sim = F.cosine_similarity(recon, target, dim=-1).mean()
            recon_loss = F.smooth_l1_loss(recon, target, beta=0.1)
            loss = recon_loss + (1 - cos_sim)
        elif self.loss_type == "charbonnier":
            recon_loss = charbonnier_loss(recon, target, eps=1e-3)
            loss = recon_loss
        elif self.loss_type == "delta":
            recon_loss = F.smooth_l1_loss(recon, target-dec_in, beta=0.1)
            loss = recon_loss
        elif self.loss_type == "l2":
            recon_loss = F.mse_loss(recon, target)
            loss = recon_loss
        else:
            recon_loss = F.mse_loss(recon, target)
            loss = recon_loss
        entropy_loss = self.lambda_diversity * entropy_loss
        total_loss = loss + entropy_loss + vq_loss
        # total_loss = loss
        aux_loss = torch.tensor(0.0, device=self.device)
        aux_loss_logs: Dict[str, Tensor] = {}

        if delta_s_pred is not None:
            if vq_training and "delta_proprio" not in batch:
                raise KeyError("LAM training requires `delta_proprio` in batch for state auxiliary loss.")
            if vq_training and state_mask is None:
                raise KeyError("LAM training requires `state_mask` in batch for state auxiliary loss.")

            state_deltas = batch.get("delta_proprio", None)
            if state_deltas is not None and not isinstance(state_deltas, torch.Tensor):
                state_deltas = torch.as_tensor(state_deltas, device=delta_s_pred.device, dtype=delta_s_pred.dtype)
            if state_mask is not None and not isinstance(state_mask, torch.Tensor):
                state_mask = torch.as_tensor(state_mask, device=delta_s_pred.device)

            if embodiment_ids is None:
                robot_mask = torch.zeros(delta_s_pred.shape[0], dtype=torch.bool, device=delta_s_pred.device)
            else:
                robot_mask = embodiment_ids.view(-1).to(device=delta_s_pred.device, dtype=torch.long) != 0
            if robot_mask.any():
                if state_deltas is None:
                    raise KeyError("State auxiliary loss requires `delta_proprio`.")
                if state_mask is None:
                    raise KeyError("State auxiliary loss requires `state_mask`.")

                delta_s_pred_robot = delta_s_pred[robot_mask]
                delta_robot = state_deltas[robot_mask]
                state_mask_robot = state_mask[robot_mask]

                if state_mask_robot.ndim == 3 and state_mask_robot.shape[1] == 2:
                    state_mask_robot = state_mask_robot.any(dim=1)

                state_loss = eef_reconstruction_loss(
                    delta_s_pred_robot,
                    state_delta=delta_robot,
                    state_mask=state_mask_robot,
                    state_loss_type=self.state_loss_type,
                )
                aux_loss = self.lambda_aux * state_loss
                aux_loss_logs["state_loss"] = aux_loss.item()
                total_loss = total_loss + aux_loss
            else:
                # Keep state decoder in autograd graph for all-human batches to avoid
                # unused-parameter warnings under DDP while preserving zero contribution.
                dummy_state_loss = delta_s_pred.sum() * 0.0
                total_loss = total_loss + dummy_state_loss
                aux_loss_logs["state_loss_skipped"] = 0.0

            logs: Dict[str, Tensor] = {
                "recon_loss": recon_loss,
                "vq_loss": vq_loss,
                "perplexity": perplexity,
                "cos_sim_metric": cos_sim_metric,
                "l1_loss_metric": l1_loss_metric,
                # "dec_in": dec_in.mean(),
                # "dec_in_std": dec_in.std(),
                # "tgt": tgt.mean(),
                # "tgt_std": tgt.std(),
                # "recon": recon.mean(),
                # "recon_std": recon.std(),
                **aux_loss_logs,
            }
            if getattr(self.lam, "vq", None) is not None:
                vq_module = self.lam.vq
                if hasattr(vq_module, "last_sample_entropy"):
                    logs["sample_entropy"] = vq_module.last_sample_entropy
                if hasattr(vq_module, "last_codebook_entropy"):
                    logs["codebook_entropy"] = vq_module.last_codebook_entropy
                if hasattr(vq_module, "nodes_norm"):
                    logs["nodes_norm"] = vq_module.nodes_norm
                if hasattr(vq_module, "last_commitment_loss"):
                    logs["commitment_loss"] = vq_module.last_commitment_loss
                if hasattr(vq_module, "last_orthogonal_loss") and vq_module.last_orthogonal_loss is not None:
                    logs["orthogonal_loss"] = vq_module.last_orthogonal_loss
                if hasattr(vq_module, "last_avg_unique_codes"):
                    logs["avg_unique_codes"] = vq_module.last_avg_unique_codes
                if self.lambda_diversity >0:
                    logs["entropy_loss"] = entropy_loss
                if hasattr(vq_module, "last_slot_inter_redundancy") and vq_module.last_slot_inter_redundancy is not None:
                    logs["slot_inter_redundancy"] = vq_module.last_slot_inter_redundancy
                if hasattr(vq_module, "last_slot_inner_redundancy") and vq_module.last_slot_inner_redundancy is not None:
                    logs["slot_inner_redundancy"] = vq_module.last_slot_inner_redundancy
                if hasattr(vq_module, "last_min_inter_code_dist"):
                    logs["min_inter_code_dist"] = vq_module.last_min_inter_code_dist
                if hasattr(vq_module, "last_avg_inter_code_dist"):
                    logs["avg_inter_code_dist"] = vq_module.last_avg_inter_code_dist
        return total_loss, logs

    def training_step(self, batch: Dict, batch_idx: int) -> Tensor:
        loss, aux_losses = self.shared_step(batch)
        current_lr = self._get_current_lr()
        scalar_loss = float(loss.detach().float().item())
        if current_lr is not None:
            aux_losses = {**aux_losses, "lr": current_lr}
        self._maybe_log_spike_batch(
            batch=batch,
            loss_value=scalar_loss,
            aux_losses=aux_losses,
            current_lr=current_lr,
            recon=self._last_step_tensors.get("recon"),
            target=self._last_step_tensors.get("target"),
            dec_in=self._last_step_tensors.get("dec_in"),
        )
        self._last_step_tensors = {}
        
        self.log_dict(
            {**{"train_loss": loss}, **{f"train/{k}": v for k, v in aux_losses.items()}},
            prog_bar=True,
            logger=False,
            on_step=True,
            on_epoch=False,
            sync_dist=False
        )
        if self._should_log_train_wandb():
            wandb_logs = {"train_loss": scalar_loss}
            for k, v in aux_losses.items():
                if isinstance(v, torch.Tensor):
                    wandb_logs[f"train/{k}"] = v.item()
                else:
                    wandb_logs[f"train/{k}"] = v
            wandb.log(wandb_logs, step=self.global_step)
        
        return loss

    def on_train_epoch_start(self) -> None:
        """Synchronize mixture epoch so per-sample RNG changes with training epoch."""
        trainer = getattr(self, "trainer", None)
        if trainer is None:
            return

        datamodule = getattr(trainer, "datamodule", None)
        if datamodule is None:
            return

        current_epoch = int(getattr(trainer, "current_epoch", 0))
        if hasattr(datamodule, "set_mixture_epoch"):
            datamodule.set_mixture_epoch(current_epoch)
            return

        # Backward-compatible fallback for custom datamodules.
        train_dataset = getattr(datamodule, "train_dataset", None)
        mixture = getattr(train_dataset, "mixture", None) if train_dataset is not None else None
        if mixture is not None and hasattr(mixture, "set_epoch"):
            mixture.set_epoch(current_epoch)
    
    @torch.no_grad()
    def validation_step(self, batch: Dict, batch_idx: int) -> Tensor:
        loss, aux_losses = self.shared_inference_step(batch)
        self.log_dict(
            {**{"val_loss": loss}, **{f"val/{k}": v for k, v in aux_losses.items()}},
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        return loss
    
    def on_after_backward(self) -> None:
        if self._unused_params_scanned:
            return
        if not getattr(self.trainer, "is_global_zero", True):
            return
        unused = []
        for name, p in self.named_parameters():
            if p.requires_grad and p.grad is None:
                unused.append(name)
        if unused:
            self.print(f"UNUSED params ({len(unused)}): " + ", ".join(unused))
        self._unused_params_scanned = True
    
    # def on_train_epoch_end(self):
    #     is_distributed = torch.distributed.is_available() and torch.distributed.is_initialized()

    #     if is_distributed:
    #         self.trainer.strategy.barrier()

    #     if getattr(self.trainer, "is_global_zero", True):
    #         with torch.no_grad():
    #             if hasattr(self.lam.vq, 'replace_unused_codebooks'):
    #                 self.lam.vq.replace_unused_codebooks()
    #             if hasattr(self.lam.vq, 'reset_node_count'):
    #                 self.lam.vq.reset_node_count()

    #     if is_distributed:
    #         state_list = [self.lam.vq.state_dict()] if getattr(self.trainer, "is_global_zero", True) else [None]
    #         torch.distributed.broadcast_object_list(state_list, src=0)
    #         with torch.no_grad():
    #             self.lam.vq.load_state_dict(state_list[0])
    #         self.trainer.strategy.barrier()

    def on_test_epoch_end(self):
        if self.make_data_pair:
            import os
            os.makedirs(self.output_dir, exist_ok=True)
            
            if hasattr(self.lam.vq, 'node_count'):
                usage = self.lam.vq.node_count
                top_indices = torch.topk(usage, min(16, self.codebook_size), largest=True, sorted=True).indices
                
                top_latents = self.lam.vq.codebooks[top_indices]
                torch.save(top_latents, f"{self.output_dir}/top_16.pt")
                
                with open(f"{self.output_dir}/top_16.txt", "w") as f:
                    f.write(" ".join([str(i.item()) for i in top_indices]))
        
        if hasattr(self.lam.vq, 'node_count'):
            self.plot_usage_distribution(self.lam.vq.node_count, "unsorted_usage")
            sorted_usage, _ = torch.sort(self.lam.vq.node_count)
            self.plot_usage_distribution(sorted_usage, "sorted_usage")

    def plot_usage_distribution(self, usage, filename):
        import matplotlib.pyplot as plt
        from matplotlib.ticker import NullLocator
        
        data = usage.cpu().numpy()
        
        n = 1
        for n in range(1, 10):
            if (2 ** n) ** 2 <= len(data) < (2 ** (n + 1)) ** 2:
                break
        
        data = data.reshape(2 ** n, -1)
        
        fig, ax = plt.subplots()
        cax = ax.matshow(data, interpolation="nearest")
        fig.colorbar(cax)
        plt.axis("off")
        plt.gca().set_axis_off()
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.gca().xaxis.set_major_locator(NullLocator())
        plt.gca().yaxis.set_major_locator(NullLocator())
        plt.savefig(f"{filename}.png", bbox_inches="tight", pad_inches=0.0)
        plt.close()

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Drop the frozen visual encoder from the checkpoint.

        `lam.vision_encoder.*` is never trained and is re-loaded from
        `vision_model_id` on every `LatentLAMModel.__init__`, so persisting it
        only wastes disk. With VGGT-1B that is 909 M params -> ~3.6 GB per
        checkpoint file.
        """
        state_dict = checkpoint.get("state_dict")
        if not state_dict:
            return
        for key in [k for k in state_dict if k.startswith("lam.vision_encoder.")]:
            del state_dict[key]

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Re-inject the frozen encoder weights stripped by `on_save_checkpoint`.

        They are already loaded in `__init__`, so copying them back from the live
        module lets Lightning keep loading strictly instead of silencing genuinely
        missing trained weights.
        """
        state_dict = checkpoint.get("state_dict")
        if state_dict is None:
            return
        for key, value in self.state_dict().items():
            if key.startswith("lam.vision_encoder.") and key not in state_dict:
                state_dict[key] = value

    def configure_optimizers(self) -> Any:
        if self.exclude_bias_norm_from_wd:
            param_groups, decay_names, no_decay_names = self._build_optimizer_param_groups()
            optim = self.optimizer(param_groups)
            self._log_optimizer_group_summary(decay_names, no_decay_names)
        else:
            optim = self.optimizer(self.parameters())
        # optim = self.optimizer(filter(lambda p: p.requires_grad, self.parameters()))
        if self.warmup_steps > 0:
            def lr_lambda(current_step: int) -> float:
                if current_step < self.warmup_steps:
                    return float(current_step + 1) / float(self.warmup_steps)
                return 1.0

            scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_lambda)
            return {
                "optimizer": optim,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1,
                },
            }

        return optim


    @torch.no_grad()
    def test_step(self, batch: Dict, batch_idx: int) -> Tensor:
        loss, aux_losses = self.shared_inference_step(batch)
        
        self.log_dict(
            {**{"test_loss": loss}, **{f"test/{k}": v for k, v in aux_losses.items()}},
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True
        )
        
        return loss
        

class CodebookMaintenanceCallback(pl.Callback):
    def __init__(self, interval_steps: int = 1000):
        super().__init__()
        self.interval_steps = interval_steps

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
    # def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # if trainer.global_step% self.interval_steps != 0 or trainer.global_step < 100:
        #     return
        if (trainer.global_step % self.interval_steps == 0 and trainer.global_step <= 10000 and trainer.global_step > 100):

            is_distributed = torch.distributed.is_available() and torch.distributed.is_initialized()

            if is_distributed:
                torch.distributed.barrier()

            with torch.no_grad():
                if hasattr(pl_module.lam.vq, 'replace_unused_codebooks'):
                    _num_replaced, _replaced_indices = pl_module.lam.vq.replace_unused_codebooks()
                if hasattr(pl_module.lam.vq, 'reset_node_count'):
                    pl_module.lam.vq.reset_node_count()

            if is_distributed:
                torch.distributed.barrier()


class SaveConfigToCheckpointCallback(pl.Callback):
    def __init__(self, config_path: str = "", filename: str = ""):
        super().__init__()
        self.config_path = config_path or os.environ.get("LAM_CONFIG_PATH", "")
        self.filename = filename or ""

    def _resolve_source_config_path(self) -> Optional[str]:
        if self.config_path and os.path.isfile(self.config_path):
            return self.config_path
        env_config_path = os.environ.get("LAM_CONFIG_PATH", "")
        if env_config_path and os.path.isfile(env_config_path):
            return env_config_path

        base_dir = Path(__file__).resolve().parents[1]
        config_dir = base_dir / "config"
        for candidate in ("dino_base_ae.yaml", "lam-vjepa.yaml"):
            path = config_dir / candidate
            if path.is_file():
                return str(path)
        for path in sorted(config_dir.glob("*.yaml")):
            if path.is_file():
                return str(path)
        return None

    def _resolve_filename(self, src_config_path: str) -> str:
        if self.filename:
            return self.filename
        return os.path.basename(src_config_path)

    def _resolve_log_dir(self, trainer) -> Optional[str]:
        logger_obj = trainer.logger
        if logger_obj is None:
            return None
        if hasattr(logger_obj, 'log_dir') and logger_obj.log_dir is not None:
            return logger_obj.log_dir
        save_dir = getattr(logger_obj, 'save_dir', None)
        name = getattr(logger_obj, 'name', None)
        version = getattr(logger_obj, 'version', None)
        parts = [p for p in [save_dir, name, f"version_{version}" if version is not None else None] if p]
        if parts:
            return os.path.join(*parts)
        return None

    @staticmethod
    def _is_global_zero(trainer) -> bool:
        return bool(getattr(trainer, "is_global_zero", True))

    def on_fit_start(self, trainer, pl_module):
        if not self._is_global_zero(trainer):
            return
        src_config_path = self._resolve_source_config_path()
        if not src_config_path:
            pl_module.print("No available config file found; skipping config save")
            return

        log_dir = self._resolve_log_dir(trainer)
        if not log_dir:
            pl_module.print("Could not resolve logger save directory; skipping config save")
            return

        filename = self._resolve_filename(src_config_path)
        try:
            os.makedirs(log_dir, exist_ok=True)
            dst_path = os.path.join(log_dir, filename)
            shutil.copyfile(src_config_path, dst_path)
            pl_module.print(f"Saved config to {dst_path}")
        except Exception as e:
            pl_module.print(f"Failed to save config: {e}")

    def on_train_epoch_end(self, trainer, pl_module):
        if not self._is_global_zero(trainer):
            return
        log_dir = self._resolve_log_dir(trainer)
        if not log_dir:
            return
        self._copy_external_log(log_dir, pl_module)

    def _copy_external_log(self, log_dir: str, pl_module) -> None:
        src_log = os.environ.get("LAM_TRAIN_LOG_FILE", "")
        if not src_log:
            return
        try:
            if os.path.isfile(src_log):
                dst_log = os.path.join(log_dir, "train.log")
                shutil.copyfile(src_log, dst_log)
                pl_module.print(f"Copied training log to {dst_log}")
        except Exception:
            pass

    def on_exception(self, trainer, pl_module, exception):
        if not self._is_global_zero(trainer):
            return
        log_dir = self._resolve_log_dir(trainer)
        if not log_dir:
            return
        self._copy_external_log(log_dir, pl_module)
