# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
import argparse
import os
import sys
from pathlib import Path

import wandb

os.environ.setdefault("HF_DATASETS_CACHE", "/tmp/hf_datasets_cache")

import json

import torch
import torch.distributed as dist
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_optimizer_state_dict,
    set_optimizer_state_dict,
)
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import gc
import pdb
from datetime import datetime

from configs import VA_CONFIGS
from dataset import MultiLatentLeRobotDataset
from distributed.fsdp import apply_ac, shard_model
from distributed.util import _configure_model, dist_max, dist_mean, init_distributed
from einops import rearrange
from modules.alignment import (
    unified_dest_and_motion,
)
from modules.utils import (
    load_transformer,
)
from utils import (
    FlowMatchScheduler,
    collate_get_mask,
    data_seq_to_patch,
    get_mesh_id,
    init_logger,
    logger,
    modelswitch,
    sample_timestep_id,
    warmup_constant_lambda,
)

FIRST = True

class Trainer:
    def __init__(self, config):
        if config.enable_wandb and config.rank == 0:
            keyword = getattr(config, 'keyword', '')
            wandb.login(host=os.environ['WANDB_BASE_URL'], key=os.environ['WANDB_API_KEY'])
            self.wandb = wandb
            self.wandb.init(
                entity=os.environ["WANDB_TEAM_NAME"],
                project=os.getenv("WANDB_PROJECT", "va_robotwin"),
                # dir=log_dir,
                config=config,
                mode="online",
                name = f'{keyword}_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
                # name=os.path.basename(os.path.normpath(job_config.job.dump_folder))
            )
            logger.info("WandB logging enabled")
        self.step = 0
        self.config = config
        self.device = torch.device(f"cuda:{config.local_rank}")
        self.dtype = config.param_dtype
        self.patch_size = config.patch_size
        # print(config.max_tokens)
        self.enable_trace = config.enable_trace
        self.trace_coef = getattr(config, 'trace_coef', 0.05)
        self.K_frames = getattr(config, 'K_frames', 3)
        self.align_layer = getattr(config, 'align_layer', 16)
        
        self.loss_weights = getattr(
            config, 'loss_weights', {
                'dest_loss':0.01,
                'motion_loss':0.01,
                'trace_loss':0.01,                                     
            }
        )
        
        # Load models
        logger.info("Loading models...")

        # Load and shard transformer with FSDP
        logger.info("Loading transformer...")
        
        if hasattr(config, 'resume_from') and config.resume_from:
            transformer_path = os.path.join(config.resume_from, 'transformer')
            if config.rank == 0:
                logger.info(f"Resuming from checkpoint: {transformer_path}")
        else:
            transformer_path = os.path.join(config.wan22_pretrained_model_name_or_path, 'transformer')
        
        print('*'*20,transformer_path)
        modelswitch(transformer_path, is_train = True)
        self.transformer = load_transformer(
            transformer_path,
            torch_dtype=torch.float32,
            torch_device='cpu',
        )
        self.transformer._init_trace_parameters(
            data_type = torch.float32,
            align_layer = self.align_layer
        )
        logger.info("Setting up activation checkpointing ...")
        apply_ac(self.transformer)

        logger.info("Setting up FSDP...")
        shard_fn = shard_model
        self.transformer = _configure_model(
            model=self.transformer,
            shard_fn=shard_fn,
            param_dtype=self.dtype,
            device=self.device,
            eval_mode=False,
        )

        self.transformer.train()
        self.transformer.requires_grad_(True)
        self.trainable_params = tuple(
            p for p in self.transformer.parameters() if p.requires_grad
        )

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.trainable_params,
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
            eps=1e-8,
            weight_decay=config.weight_decay,
            fused=True,
            foreach=False,
        )

        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, 
            lr_lambda=lambda step: warmup_constant_lambda(step, warmup_steps=config.warmup_steps))

        # Setup dataloaders
        logger.info("Setting up datasets...")
        train_dataset = MultiLatentLeRobotDataset(
            config=config,
            num_init_worker=1
        )
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=config.world_size,
            rank=config.rank,
            shuffle=True,
            seed=42
        ) if config.world_size > 1 else None

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=(train_sampler is None), 
            num_workers=config.load_worker,
            sampler=train_sampler,
            collate_fn=collate_get_mask,
        )

        self.train_scheduler_latent = FlowMatchScheduler(shift=self.config.snr_shift, sigma_min=0.0, extra_one_step=True)
        self.train_scheduler_latent.set_timesteps(1000, training=True)
        self.train_scheduler_action = FlowMatchScheduler(shift=self.config.action_snr_shift, sigma_min=0.0, extra_one_step=True)
        self.train_scheduler_action.set_timesteps(1000, training=True)

        self.save_dir = Path(config.save_root) / "checkpoints"
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.gradient_accumulation_steps = getattr(config, 'gradient_accumulation_steps', 1)
        # if hasattr(config, 'resume_from') and config.resume_from:
        #     self._load_training_state(config.resume_from)
    
    @torch.no_grad()
    def _add_noise(self, latent, train_scheduler, action_mask=False, action_mode=False, noisy_cond_prob=0.):
        B, C, F, H, W = latent.shape

        # sample timesteps for each frame, it's for frames only!
        # timestep_ids.shape = [F]
        timestep_ids = sample_timestep_id(batch_size=F, num_train_timesteps=train_scheduler.num_train_timesteps)
        # noise generation
        noise = torch.zeros_like(latent).normal_()
        # actual timesteps from timestep_ids: [F]
        timesteps = train_scheduler.timesteps[timestep_ids].to(device=self.device)
        # each frame adds noise with different level of 
        # timesteps: frame1:[10], frame2:[20], frame3[15]
        noisy_latents =train_scheduler.add_noise(latent, noise, timesteps, t_dim=2)
        # target = noise - latent
        targets =train_scheduler.training_target(latent, noise, timesteps)

        patch_f, patch_h, patch_w = self.patch_size
        if action_mode:
            patch_f = patch_h = patch_w = 1
        
        # generate positional embeds
        latent_grid_id = get_mesh_id(
            latent.shape[-3] // patch_f,  # F
            latent.shape[-2] // patch_h,  # H
            latent.shape[-1] // patch_w,  # W
            t=1 if action_mode else 0,  # 1 for action mode (0 for latent), not used
            f_w=1,
            f_shift=0,
            action=action_mode
        ).to(self.device)  # shape: [4, seq_len] t / f / h / w , be implemented flex attn???
        latent_grid_id = latent_grid_id[None].repeat(B, 1, 1)

        # add noise for condition as well
        if torch.rand(1).item() < noisy_cond_prob:
            cond_timestep_ids = sample_timestep_id(
                    batch_size=F,
                    min_timestep_bd=0.5, 
                    max_timestep_bd=1.0, 
                    num_train_timesteps=train_scheduler.num_train_timesteps,
                )
            noise = torch.zeros_like(latent).normal_()
            cond_timesteps = train_scheduler.timesteps[cond_timestep_ids].to(device=self.device)
            latent = train_scheduler.add_noise(latent, noise, cond_timesteps, t_dim=2)
        else:
            cond_timesteps = torch.zeros_like(timesteps)

        #  mask
        if action_mask is not None:
            noisy_latents *= action_mask.float()
            targets *= action_mask.float()
            latent *= action_mask.float()

        return dict(
            timesteps=timesteps[None].repeat(B, 1),
            noisy_latents=noisy_latents,
            targets=targets,
            latent=latent,
            cond_timesteps=cond_timesteps[None].repeat(B, 1),
            grid_id=latent_grid_id,
        )

    @torch.no_grad()
    def _prepare_input_dict(self, batch_dict, config):
        """Prepare input dict following infer code pattern from wan_va_server.py."""
        # Generate grid_id following infer code (no batch dimension yet)
        # For action mode: get_mesh_id(shape[-3], shape[-2], shape[-1], t=1, f_w=1, f_shift, action=True)
        latent_dict = self._add_noise(
            latent=batch_dict['latents'], 
            train_scheduler=self.train_scheduler_latent, 
            action_mask=None, 
            action_mode=False,
            noisy_cond_prob=0.5
        )
        
        action_dict = self._add_noise(
            latent=batch_dict['actions'], 
            train_scheduler=self.train_scheduler_action, 
            action_mask=batch_dict['actions_mask'], 
            action_mode=True,
            noisy_cond_prob=0.0
        )

        # batch_dict['text_embed_real'] = batch_dict['text_emb']
        B, T, D = batch_dict['text_emb'].shape
        if T < config.max_tokens:
            batch_dict['text_emb'] = F.pad(
                batch_dict['text_emb'],
                (0, 0, 0, config.max_tokens - T),  # (D_left, D_right, T_left, T_right)
            )
        if batch_dict['text_emb'].dtype != torch.bfloat16:
            batch_dict['text_emb'] = batch_dict['text_emb'].to(torch.bfloat16)
        if D != 4096:
            return False
        latent_dict['text_emb'] = batch_dict['text_emb']
        action_dict['text_emb'] = batch_dict['text_emb']
        action_dict['actions_mask'] = batch_dict['actions_mask']

        global FIRST
        if FIRST:
            for key in latent_dict:
                if isinstance(latent_dict[key],torch.Tensor) or isinstance(latent_dict[key],np.ndarray):
                    print(key, latent_dict[key].shape)
                else:
                    print(key, latent_dict[key])
            FIRST = False

        input_dict = {
            'latent_dict': latent_dict,
            'action_dict': action_dict,
            'chunk_size': torch.randint(2, 5, (1,)).item(),
            'window_size': torch.randint(8, 65, (1,)).item(),
        }

        if 'trace' in batch_dict:
            input_dict['trace'] = batch_dict['trace']
        
        return input_dict

    def convert_input_format(self, input_dict):
        """Convert input dict to match transformer input format if needed."""
        def move_to_device(value):
            if torch.is_tensor(value):
                return value.to(self.device)
            if isinstance(value, dict):
                return {k: move_to_device(v) for k, v in value.items()}
            if isinstance(value, list):
                return [move_to_device(v) for v in value]
            return value

        return {key: move_to_device(value) for key, value in input_dict.items()}

    def compute_loss(self,
        input_dict,
        pred
    ):  
        alignment_loss = torch.tensor(0.0)
        if len(pred) == 3:
            latent_pred, action_pred, alignment_loss = pred
        else:
            latent_pred, action_pred = pred
        # print(alignment_loss)
        action_pred = rearrange(action_pred, 'b (f n) c -> b c f n 1', f=input_dict['action_dict']['targets'].shape[-3])
        latent_pred = data_seq_to_patch(
                        self.patch_size, latent_pred,
                        input_dict['latent_dict']['targets'].shape[-3], input_dict['latent_dict']['targets'].shape[-2],
                        input_dict['latent_dict']['targets'].shape[-1], batch_size=latent_pred.shape[0])
        Bn, Fn = input_dict['latent_dict']['timesteps'].shape
        latent_loss_weight = self.train_scheduler_latent.training_weight(input_dict['latent_dict']['timesteps'].flatten()).reshape(Bn, Fn)
        action_loss_weight = self.train_scheduler_action.training_weight(input_dict['action_dict']['timesteps'].flatten()).reshape(Bn, Fn)

        # Frame-wise video loss calculation
        latent_loss = F.mse_loss(latent_pred.float(), input_dict['latent_dict']['targets'].float().detach(), reduction='none')
        latent_loss = latent_loss * latent_loss_weight[:, None, :, None, None]
        # Permute to (B, F, H, W, C) and flatten to (B*F, H*W*C)
        latent_loss = latent_loss.permute(0, 2, 3, 4, 1)  # (B, C, F, H, W) -> (B, F, H, W, C)
        latent_loss = latent_loss.flatten(0, 1).flatten(1)  # (B, F, H, W, C) -> (B*F, H*W*C)
        # Sum per frame and compute mask per frame
        latent_loss_per_frame = latent_loss.sum(dim=1)  # (B*F,)
        latent_mask_per_frame = torch.ones_like(latent_loss).sum(dim=1)  # (B*F,)
        latent_loss = (latent_loss_per_frame / (latent_mask_per_frame + 1e-6)).mean()

        # Frame-wise action loss calculation
        action_loss = F.mse_loss(action_pred.float(), input_dict['action_dict']['targets'].float().detach(), reduction='none')
        action_loss = action_loss * action_loss_weight[:, None, :, None, None]
        action_loss = action_loss * input_dict['action_dict']['actions_mask'].float()
        # Permute to (B, F, H, W, C) and flatten to (B*F, H*W*C)
        action_loss = action_loss.permute(0, 2, 3, 4, 1)  # (B, C, F, H, W) -> (B, F, H, W, C)
        action_mask = input_dict['action_dict']['actions_mask'].float().permute(0, 2, 3, 4, 1)  # (B, C, F, H, W) -> (B, F, H, W, C)
        action_loss = action_loss.flatten(0, 1).flatten(1)  # (B, F, H, W, C) -> (B*F, H*W*C)
        action_mask = action_mask.flatten(0, 1).flatten(1)  # (B, F, H, W, C) -> (B*F, H*W*C)
        # Sum per frame and normalize by mask per frame
        action_loss_per_frame = action_loss.sum(dim=1)  # (B*F,)
        action_mask_per_frame = action_mask.sum(dim=1)  # (B*F,)
        action_loss = (action_loss_per_frame / (action_mask_per_frame + 1e-6)).mean()

        if not isinstance(alignment_loss, dict):
            alignment_loss = {
                'trace_loss': alignment_loss
            }
        alignment_loss = self.formulize_traceloss(alignment_loss)
        
        return latent_loss / self.gradient_accumulation_steps, action_loss / self.gradient_accumulation_steps, alignment_loss

    
    def formulize_traceloss(self, alignment_loss):
        # alignment_loss['dest_loss'] *= loss_weights['dest_weight']
        # alignment_loss['motion_loss'] *= loss_weights['motion_weight']
        # alignment_loss['total'] = 0
        for key in alignment_loss:
            alignment_loss[key] *= self.loss_weights[key]
            alignment_loss[key] /= self.gradient_accumulation_steps
            # alignment_loss['total'] += alignment_loss[key]
        
        alignment_loss['total'] = sum([alignment_loss[key] for key in alignment_loss])
        
        return alignment_loss
    
    def _build_alignment_show(self,accumulated_align_losses):
        alignment_loss_show = {}
        max_alignment_loss_show = {}
        for key in accumulated_align_losses:
            alignment_loss_show[key] = dist_mean(torch.stack(accumulated_align_losses[key]).sum()).detach().cpu().item() if self.enable_trace else 0
            max_alignment_loss_show[key] = dist_max(torch.stack(accumulated_align_losses[key]).sum()).detach().cpu().item() if self.enable_trace else 0
        
        return alignment_loss_show, max_alignment_loss_show
    
    def _finalize_optimizer_step(
        self,
        accumulated_latent_losses,
        accumulated_action_losses,
        accumulated_align_losses,
        progress_bar,
    ):
        num_accumulated_batches = len(accumulated_latent_losses)
        total_norm = torch.nn.utils.clip_grad_norm_(self.transformer.parameters(), 2.0)
        self.optimizer.step()
        self.lr_scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)

        lr = self.lr_scheduler.get_last_lr()[0]

        latent_loss_show = dist_mean(torch.stack(accumulated_latent_losses).sum()).detach().cpu().item()
        action_loss_show = dist_mean(torch.stack(accumulated_action_losses).sum()).detach().cpu().item()

        max_latent_loss_show = dist_max(torch.stack(accumulated_latent_losses).sum()).detach().cpu().item()
        max_action_loss_show = dist_max(torch.stack(accumulated_action_losses).sum()).detach().cpu().item()

        # alignment_loss_show = dist_mean(torch.stack(accumulated_align_losses).sum()).detach().cpu().item() if self.enable_trace else 0
        # max_alignment_loss_show = dist_max(torch.stack(accumulated_align_losses).sum()).detach().cpu().item() if self.enable_trace else 0

        alignment_loss_show, max_alignment_loss_show = self._build_alignment_show(accumulated_align_losses)

        torch.cuda.synchronize()
        if self.step % self.config.gc_interval == 0:
            torch.cuda.empty_cache()
            gc.collect()

        if self.config.rank == 0:
            progress_bar.n += num_accumulated_batches
            postfix = {
                'latent_loss': f'{latent_loss_show:.5f}',
                'action_loss': f'{action_loss_show:.5f}',
                # 'alignment_loss': f'{alignment_loss_show:.5f}',
                'step': self.step,
                'grad_norm': f'{total_norm.item():.3f}',
                'lr': f'{lr:.2e}'
            }
            for key in alignment_loss_show:
                postfix[key] = f'{alignment_loss_show[key]:.5f}'
            progress_bar.set_postfix(postfix)
            if self.config.enable_wandb:
                wandb_metrics = {
                    'loss_metrics/global_avg_video_loss': latent_loss_show,
                    'loss_metrics/global_avg_action_loss': action_loss_show,
                    # 'loss_metrics/global_avg_alignment_loss': alignment_loss_show,
                    'loss_metrics/global_max_video_loss': max_latent_loss_show,
                    'loss_metrics/global_max_action_loss': max_action_loss_show,
                    # 'loss_metrics/global_max_alignment_loss': max_alignment_loss_show,
                    'grad_norm': total_norm.item(),
                    'lr': lr,
                }
                for key in alignment_loss_show:
                    wandb_metrics[f'loss_metrics/global_max_{key}'] = max_alignment_loss_show[key]
                    wandb_metrics[f'loss_metrics/global_avg_{key}'] = alignment_loss_show[key]
                    
                self.wandb.log(wandb_metrics, step=self.step)

        self.step += 1
        if self.step % self.config.save_interval == 0:
            if self.config.rank == 0:
                logger.info(f"Starting save model at step {self.step}")
            self.save_checkpoint()

    def build_alignment_loss_accumulated(self, alignment_loss, accumulated_align_losses):
        for key in alignment_loss:
            # if key != 'total':
            # print(alignment_loss[key],accumulated_align_losses)
            if key not in accumulated_align_losses:
                accumulated_align_losses[key] = []
            accumulated_align_losses[key].append(alignment_loss[key].detach() if self.enable_trace else 0)

        return accumulated_align_losses

    def clear_alignment_loss_accumulated(self, alignment_loss, accumulated_align_losses):
        for key in alignment_loss:
            # if key != 'total':
            accumulated_align_losses[key] = []
            
        return accumulated_align_losses
    
    def _run_train_micro_step(
        self,
        input_dict,
        valid_batch_count,
        accumulated_latent_losses,
        accumulated_action_losses,
        accumulated_align_losses,
        progress_bar,
        is_last_valid_batch=False,
    ):
        should_sync = (
            (valid_batch_count + 1) % self.gradient_accumulation_steps == 0
            or is_last_valid_batch
        )
        self.transformer.set_requires_gradient_sync(should_sync)

        output = self.transformer(input_dict, alignment_module=unified_dest_and_motion, train_mode=True)
        latent_loss, action_loss, alignment_loss = self.compute_loss(input_dict, output)
        loss = latent_loss + action_loss + alignment_loss['total']

        loss.backward()

        accumulated_latent_losses.append(latent_loss.detach())
        accumulated_action_losses.append(action_loss.detach())
        
        
        # accumulated_align_losses.append(alignment_loss.detach() if self.enable_trace else 0)
        accumulated_align_losses = self.build_alignment_loss_accumulated(alignment_loss, accumulated_align_losses)
        
        if should_sync:
            self._finalize_optimizer_step(
                accumulated_latent_losses,
                accumulated_action_losses,
                accumulated_align_losses,
                # alignment_loss,
                progress_bar,
            )
            accumulated_latent_losses = []
            accumulated_action_losses = []
            # accumulated_align_losses = []
            self.clear_alignment_loss_accumulated(alignment_loss, accumulated_align_losses)

        return (
            valid_batch_count + 1,
            accumulated_latent_losses,
            accumulated_action_losses,
            accumulated_align_losses,
            # alignment_loss,
        )

    def train_epoch(self):
        self.transformer.train()

        # Use manual progress bar control to only update on optimizer steps
        progress_bar = tqdm(
            total=len(self.train_loader),
            desc="Training",
            disable=(self.config.rank != 0),
            leave=True, 
            dynamic_ncols=True
        )

        self.optimizer.zero_grad(set_to_none=True)
        accumulated_latent_losses = []
        accumulated_action_losses = []
        accumulated_align_losses = {}
        valid_batch_count = 0
        pending_input_dict = None
        for batch in self.train_loader:
            if self.step >= self.config.num_steps:
                break
            batch = self.convert_input_format(batch)

            input_dict = self._prepare_input_dict(batch, self.config)
            if isinstance(input_dict,bool) and not input_dict:
                continue

            if pending_input_dict is not None:
                (
                    valid_batch_count,
                    accumulated_latent_losses,
                    accumulated_action_losses,
                    accumulated_align_losses,
                ) = self._run_train_micro_step(
                    pending_input_dict,
                    valid_batch_count,
                    accumulated_latent_losses,
                    accumulated_action_losses,
                    accumulated_align_losses,
                    progress_bar,
                    is_last_valid_batch=False,
                )

            pending_input_dict = input_dict

        if pending_input_dict is not None and self.step < self.config.num_steps:
            (
                valid_batch_count,
                accumulated_latent_losses,
                accumulated_action_losses,
                accumulated_align_losses,
            ) = self._run_train_micro_step(
                pending_input_dict,
                valid_batch_count,
                accumulated_latent_losses,
                accumulated_action_losses,
                accumulated_align_losses,
                progress_bar,
                is_last_valid_batch=True,
            )

        progress_bar.close()

    def save_checkpoint(self,):
        """Save model checkpoint in the same format as pretrained model."""
        try:
            state_dict = get_model_state_dict(
                self.transformer,
                options=StateDictOptions(full_state_dict=True, cpu_offload=True),
            )
            state_dict_bf16 = {k: v.to(torch.bfloat16) for k, v in state_dict.items()}
            # optim_state = get_optimizer_state_dict(
            #         self.transformer, self.optimizer,
            #         options=StateDictOptions(full_state_dict=True, cpu_offload=True),
            #     )

            # Only rank 0 saves the checkpoint
            if self.config.rank == 0:
                checkpoint_dir = self.save_dir / f"checkpoint_step_{self.step}"
                checkpoint_dir.mkdir(parents=True, exist_ok=True)

                # Save transformer in the same format as pretrained model
                transformer_dir = checkpoint_dir / "transformer"
                transformer_dir.mkdir(parents=True, exist_ok=True)

                logger.info(f"Saving transformer to {transformer_dir}")

                # Manually save in diffusers format (outside FSDP context to avoid deadlock)
                # Save model weights
                model_file = transformer_dir / "diffusion_pytorch_model.safetensors"
                save_file(state_dict_bf16, model_file)

                # Save config (copy from original transformer config and update _name_or_path)
                config_file = transformer_dir / "config.json"
                config_dict = dict(self.transformer.config)
                config_dict.pop('_name_or_path', None)
                with open(config_file, 'w') as f:
                    json.dump(config_dict, f, indent=2)

                # # Save optimizer state and training metadata in PyTorch format
                # training_state_path = checkpoint_dir / "training_state.pt"
                # logger.info(f"Saving training state to {training_state_path}")
                # torch.save({
                #     'step': self.step,
                #     'optimizer_state_dict': optim_state,
                #     'config': vars(self.config),
                # }, training_state_path)

                logger.info(f"Checkpoint saved successfully at step {self.step}")

            # Synchronize all processes after saving
            if dist.is_initialized():
                dist.barrier()

        except Exception as e:
            if self.config.rank == 0:
                logger.error(f"Failed to save checkpoint: {e}")
                import traceback
                logger.error(traceback.format_exc())
            # Ensure all processes stay synchronized even on error
            if dist.is_initialized():
                dist.barrier()

    def _load_training_state(self, checkpoint_path):
        """Load training state (optimizer + step) after FSDP and optimizer creation."""
        checkpoint_dir = Path(checkpoint_path)
        training_state_path = checkpoint_dir / "training_state.pt"

        if not training_state_path.exists():
            if self.config.rank == 0:
                logger.warning(f"Training state not found: {training_state_path}, starting from step 0")
            return

        if self.config.rank == 0:
            logger.info(f"Loading training state from {training_state_path}")

        # All ranks load the training state directly
        training_state = torch.load(training_state_path, map_location='cpu', weights_only=False)

        # All ranks load optimizer state (required for FSDP)
        set_optimizer_state_dict(
            self.transformer, self.optimizer,
            optim_state_dict=training_state['optimizer_state_dict'],
            options=StateDictOptions(full_state_dict=True, strict=False)
        )
        self.step = training_state.get('step', 0)

        if self.config.rank == 0:
            logger.info(f"Training state loaded, resuming from step {self.step}")

        # Synchronize all ranks
        if dist.is_initialized():
            dist.barrier()

    def train(self):
        """Main training loop."""
        logger.info(f"Starting training for {self.config.num_steps} steps...")

        while self.step < self.config.num_steps:
            self.train_epoch()
            if dist.is_initialized():
                dist.barrier()

        logger.info("Training completed!")


def run(args):
    """Main entry point."""
    config = VA_CONFIGS[args.config_name] # datasets

    rank = int(os.getenv("RANK", 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    print(f'-------world_size:{world_size}---------')
    if world_size > 1:
        print('world_size, local_rank, rank',world_size, local_rank, rank)
        init_distributed(world_size, local_rank, rank)
    else:
        # 单进程：确保当前卡设置好
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
    # init_distributed(world_size, local_rank, rank)

    config.rank = rank
    config.local_rank = local_rank
    config.world_size = world_size

    if args.save_root is not None:
        config.save_root = args.save_root
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.gradient_accumulation_steps is not None:
        config.gradient_accumulation_steps = args.gradient_accumulation_steps
    if args.num_steps is not None:
        config.num_steps = args.num_steps
    if args.load_worker is not None:
        config.load_worker = args.load_worker
    if args.disable_wandb:
        config.enable_wandb = False

    if rank == 0:
        logger.info(f"Using config: {args.config_name}")
        logger.info(f"World size: {world_size}, Local rank: {local_rank}")
    # pdb.set_trace()
    trainer = Trainer(config)
    trainer.train()


def main():
    """Parse arguments and run training."""
    parser = argparse.ArgumentParser(description="Train WAN model for robotics")
    parser.add_argument(
        "--config-name",
        type=str,
        default='robotwin_train',
        help="Config name",
    )
    parser.add_argument(
        "--save-root",
        type=str,
        default=None,
        help="Root directory for saving checkpoints",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override training batch size",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=None,
        help="Override gradient accumulation steps",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=None,
        help="Override total optimizer steps",
    )
    parser.add_argument(
        "--load-worker",
        type=int,
        default=None,
        help="Override DataLoader worker count",
    )
    parser.add_argument(
        "--disable-wandb",
        action="store_true",
        help="Disable wandb logging regardless of config",
    )

    args = parser.parse_args()
    run(args)


if __name__ == "__main__":

    init_logger()
    main()
