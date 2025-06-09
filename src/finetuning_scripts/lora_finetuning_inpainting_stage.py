import json
import math
import torch
import wandb
from pathlib import Path
from pydantic import BaseModel
from accelerate import Accelerator
from accelerate.logging import get_logger
from peft import LoraConfig, get_peft_model
from fastcore.script import call_parse

from diffusers.utils.import_utils import is_xformers_available
from diffusers.optimization import get_scheduler
import bitsandbytes as bnb
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F
from tqdm import tqdm
# Local imports
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pcdms.InpaintingStage import *
from CelebrityDataset import CelebrityDataset, CelebrityCollateFn
from utils import set_seed, show_images

UNET_TARGET_MODULES = ["to_k", "to_q", "to_v", "to_out.0"]
logger = get_logger(__name__)

class LoraFinetuningConfig(BaseModel):
    # Configuration for Dataset
    celebrity_name: str = 'mohamed_salah'
    root_dir: str = "datasets/celebrities"
    image_resize: tuple = (512, 512)
    num_images: int = 35
    seed: int = 42
    max_samples: int = 6000
    
    num_dataloader_workers: int = 4
    
    # Configuration for LoRA Finetuning
    lora_rank: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_bias: str = "none"
    
    # Configuration for Training
    output_dir: str = "checkpoint/lora"
    train_batch_size: int = 4
    train_num_epochs: int = 10
    train_save_steps: int = 500
    gradient_accumulation_steps: int = 2
    gradient_checkpointing: bool = True
    
    learning_rate: float = 5e-6
    scale_lr: bool = True
    lr_scheduler: str = "constant"
    lr_warmup_steps: int = 500
    lr_num_cycles: int = 1
    lr_power: float = 1.0
    
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_weight_decay: float = 1e-2
    adam_epsilon: float = 1e-08
    max_grad_norm: float = 1.0
    
    resume_from_checkpoint: str = None
    allow_tf32: bool = False
    mixed_precision: str = "fp16"
    
    wandb_key: str = None
    wandb_project_name: str = "lora-finetuning-inpainting"
    
    device: str = 'cuda'

def lora_finetuning(config: LoraFinetuningConfig):
    output_dir = Path(config.output_dir)
    logging_dir = output_dir / "wandb"
    
    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        mixed_precision=config.mixed_precision,
        project_dir=logging_dir,
        log_with="wandb" if config.wandb_key else None,
    )
    
    if accelerator.is_local_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        logging_dir.mkdir(parents=True, exist_ok=True)
        
    if config.wandb_key:
        wandb.login(key=config.wandb_key)
        wandb.init(project=config.wandb_project_name)
    
    logger.info(accelerator.state)
    
    set_seed(config.seed)
    
    stage_config = InpaintingConfig(precision=config.mixed_precision)
    stage = InpaintingStage(stage_config)
    
    lora_config = LoraConfig(
        r=config.lora_rank,
        bias=config.lora_bias,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=UNET_TARGET_MODULES,
    )
    
    unet = get_peft_model(stage.sd_model.unet, lora_config)
    unet.print_trainable_parameters()
    
    if is_xformers_available():
        unet.enable_xformers_memory_efficient_attention()
    if config.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
    if config.scale_lr:
        config.learning_rate = (
            config.learning_rate * config.gradient_accumulation_steps * config.train_batch_size * accelerator.num_processes
        )

    optimizer = bnb.optim.AdamW8bit(
        params=unet.parameters(),
        lr=config.learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
        weight_decay=config.adam_weight_decay,
        eps=config.adam_epsilon,
    )
    
    train_dataset = CelebrityDataset(
        config.root_dir, 
        config.celebrity_name, 
        config.image_resize, 
        config.num_images, 
        config.seed, 
        config.max_samples
    )
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=config.train_batch_size, 
        shuffle=True, 
        pin_memory=True, 
        num_workers=config.num_dataloader_workers, 
        collate_fn=CelebrityCollateFn
    )
    
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / config.gradient_accumulation_steps)
    max_train_steps = config.train_num_epochs * num_update_steps_per_epoch
    
    lr_scheduler = get_scheduler(
        config.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps * config.gradient_accumulation_steps,
        num_training_steps=max_train_steps * config.gradient_accumulation_steps,
        num_cycles=config.lr_num_cycles,
        power=config.lr_power,
    )
    
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )
    
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        
    total_batch_size = config.train_batch_size * accelerator.num_processes * config.gradient_accumulation_steps
    
    global_step = 0
    first_epoch = 0
    
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {config.train_num_epochs}")
    logger.info(f"  Instantaneous batch size per device = {config.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {config.gradient_accumulation_steps}")

    if config.resume_from_checkpoint:
        checkpoint_path = Path(config.resume_from_checkpoint)
        logger.info(f"Resuming from checkpoint: {checkpoint_path}")
        unet.module.load_pretrained(checkpoint_path)
        accelerator.load_state(checkpoint_path)
        # load global step and epoch from trainer state
        with open(Path(config.resume_from_checkpoint) / "trainer_state.json") as f:
            state = json.load(f)
        global_step = state["global_step"]
        first_epoch = state["epoch"]
        
    progress_bar = tqdm(range(global_step, max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")
    
    for epoch in range(first_epoch, config.train_num_epochs):
        unet.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                for key in ["mask", "source_image", "target_image", "source_target_pose", "source_target_image", "vae_source_mask_image"]:
                    batch[key] = batch[key].to(accelerator.device, dtype=weight_dtype)
                # Convert images to latent space
                latents = stage.vae.encode(batch["source_target_image"]).latent_dist.sample() * stage.vae.config.scaling_factor
                masked_latents = stage.vae.encode(batch["vae_source_mask_image"]).latent_dist.sample() * stage.vae.config.scaling_factor
                cond_image_feature_p = stage.image_encoder_p(batch["source_image"]).last_hidden_state
                cond_image_feature_g = stage.image_encoder_g(batch["target_image"]).image_embeds.unsqueeze(1)
                
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(
                    0, stage.noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device
                ).long()
                noisy_latents = stage.noise_scheduler.add_noise(latents, noise, timesteps)
                unet_input = torch.cat([noisy_latents, batch["mask"], masked_latents], dim=1)
                
                model_pred = stage.sd_model(
                    unet_input,
                    timesteps,
                    cond_image_feature_p,
                    cond_image_feature_g,
                    batch["source_target_pose"],
                )
                
                if stage.noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif stage.noise_scheduler.config.prediction_type == "v_prediction":
                    target = stage.noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {stage.noise_scheduler.config.prediction_type}")
                
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                accelerator.backward(loss)
                
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), config.max_grad_norm)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                    global_step += 1
                    logs = {"loss": loss.item(), "lr": lr_scheduler.get_last_lr()[0]}
                    progress_bar.update(1)
                    progress_bar.set_postfix(**logs)
                    accelerator.log(logs, step=global_step)
                
                if accelerator.is_main_process and global_step % config.train_save_steps == 0:
                    # log images to wandb
                    outputs = latents / stage.vae.config.scaling_factor
                    outputs  = stage.vae.decode(outputs).sample
                    accelerator.log({
                        "epoch": epoch,
                        "step": global_step,
                        "Targets": wandb.Image(show_images(batch["source_target_image"])),
                        "Samples": wandb.Image(show_images(outputs)),
                    }, step=global_step)
                    
                    accelerator.wait_for_everyone()
                    checkpoint_path = output_dir / f"checkpoint-{global_step}"
                    checkpoint_path.mkdir(parents=True, exist_ok=True)
                    # Save LoRA model
                    unet.module.save_pretrained(checkpoint_path)
                    # Save optimizer, scheduler, RNG state, etc.
                    accelerator.save_state(checkpoint_path)
                    # Save step/epoch
                    with open(checkpoint_path / "trainer_state.json", "w") as f:
                        json.dump({"global_step": global_step, "epoch": epoch}, f)
                        
                    logger.info(f"Saved full checkpoint to {checkpoint_path}")

    if accelerator.is_main_process:
        accelerator.wait_for_everyone()
        checkpoint_path = output_dir / f"checkpoint-latest"
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        unet.module.save_pretrained(checkpoint_path)
        accelerator.save_state(checkpoint_path)
        with open(checkpoint_path / "trainer_state.json", "w") as f:
            json.dump({"global_step": global_step, "epoch": epoch}, f)
        logger.info(f"Saved full checkpoint to {checkpoint_path}")
        
@call_parse
def main(config_path: str = "config/lora_finetune_example.json" # Path to config JSON file
        ):
    """Run LoRA fine-tuning using a JSON config."""
    config_path = BASE_DIR / config_path
    with open(config_path, "r") as f:
        cfg_dict = json.load(f)
    config = LoraFinetuningConfig(**cfg_dict)
    lora_finetuning(config)