import gc
import json
import math
import torch
import wandb
import logging
import argparse
import diffusers
import transformers
from PIL import Image
from pathlib import Path
from tqdm.auto import tqdm
from pydantic import BaseModel
import torch.nn.functional as F
from accelerate import Accelerator
from fastcore.script import call_parse
from accelerate.logging import get_logger
from peft import LoraConfig, get_peft_model
from transformers import CLIPImageProcessor
from torch.utils.data import Dataset, DataLoader
from diffusers.optimization import get_scheduler
from diffusers.utils.import_utils import is_xformers_available
from loguru import logger
# Local imports
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import set_seed, show_images
from constants import BASE_DIR

UNET_TARGET_MODULES = ["to_k", "to_q", "to_v", "to_out.0"]
# logger = get_logger(__name__)

class LoraFinetuningConfig(BaseModel):
    # Configuration for Dataset
    celebrity_name: str = 'mohamed_salah'
    root_dir: str = "datasets/celebrities"
    image_resize: tuple = (512, 512)
    num_images: int = 35
    seed: int = 42
    max_samples: int = 6000
    
    num_dataloader_workers: int = 2
    
    # Configuration for LoRA Finetuning
    lora_rank: int = 4
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    lora_bias: str = "none"
    
    # Configuration for Training
    output_dir: str = "checkpoint/lora"
    train_batch_size: int = 4
    train_num_epochs: int = 10
    train_save_steps: int = 500
    gradient_accumulation_steps: int = 2
    gradient_checkpointing: bool = True
    max_train_steps: int = 2000
    
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
    
    noise_offset: float = 0.1
    set_grads_to_none: bool = True
    
    validate_every_n_steps: int = 500
    preload_embeddings: bool = True
    
    device: str = 'cuda'

def checkpoint_model(model, output_dir, global_step, epoch, accelerator):
    """
    Save the model checkpoint.
    
    Args:
        model: The model to save.
        output_dir: Directory to save the checkpoint.
        global_step: Current training step.
        epoch: Current epoch.
        accelerator: Accelerator instance for distributed training.
    """
    checkpoint_path = output_dir / f"checkpoint-{global_step}"
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    
    # Save LoRA model
    model.save_pretrained(checkpoint_path)
    
    # Save optimizer, scheduler, RNG state, etc.
    accelerator.save_state(checkpoint_path)
    
    # Save step/epoch
    with open(checkpoint_path / "trainer_state.json", "w") as f:
        json.dump({"global_step": global_step, "epoch": epoch}, f)
    
    logger.info(f"Saved full checkpoint to {checkpoint_path}")
   
def get_image_embeddings_g(image_encoder_g, dataset, device, weight_dtype, image_size, batch_size=32) -> dict[str, torch.Tensor]:
    """
    Get the image embeddings for the target images in batches.
    """
    image_encoder_g = image_encoder_g.to(device)
    image_encoder_g.eval()
    clip_image_processor = CLIPImageProcessor()

    # Collect image paths
    image_paths = list(dataset.cleaned_images_directory.glob("*.[jp][pn]*g"))
    outputs = {}

    # Preprocessing images
    def load_and_preprocess(image_path):
        img = Image.open(image_path).convert("RGB").resize(image_size, Image.BICUBIC)
        return clip_image_processor(images=img, return_tensors="pt").pixel_values.squeeze(0)

    # Load all processed tensors into memory (can be optimized with a Dataset if needed)
    processed_images = [load_and_preprocess(p) for p in tqdm(image_paths, desc="Preprocessing")]

    # Batch through the image embeddings
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Embedding"):
        batch_images = processed_images[i:i+batch_size]
        batch_tensor = torch.stack(batch_images).to(device, dtype=weight_dtype, memory_format=torch.contiguous_format)

        with torch.no_grad():
            batch_embeds = image_encoder_g(batch_tensor).image_embeds.unsqueeze(1).to("cpu")

        for path, embed in zip(image_paths[i:i+batch_size], batch_embeds):
            outputs[path] = embed

    image_encoder_g = image_encoder_g.to("cpu")
    return outputs

def get_image_embeddings_p(image_encoder_p, dataset, device, weight_dtype, image_size, batch_size=32) -> dict[str, torch.Tensor]:
    """
    Get the image embeddings for the source images.
    """
    image_encoder_p = image_encoder_p.to(device)
    image_encoder_p.eval()
    
    clip_image_processor = CLIPImageProcessor()
    
    image_paths = list(dataset.cleaned_images_directory.glob("*.[jp][pn]*g"))
    outputs = {}
    
    # Preprocessing images
    def load_and_preprocess(image_path):
        img = Image.open(image_path).convert("RGB").resize(image_size, Image.BICUBIC)
        return clip_image_processor(images=img, return_tensors="pt").pixel_values.squeeze(0)
    # Load all processed tensors into memory (can be optimized with a Dataset if needed)
    processed_images = [load_and_preprocess(p) for p in tqdm(image_paths, desc="Preprocessing")]

    for i in tqdm(range(0, len(image_paths), batch_size), desc="Embedding"):
        batch_images = processed_images[i:i+batch_size]
        batch_tensor = torch.stack(batch_images).to(device, dtype=weight_dtype, memory_format=torch.contiguous_format)
        
        with torch.no_grad():
            batch_embeds = image_encoder_p(batch_tensor).last_hidden_state.to("cpu")
            
        for path, embed in zip(image_paths[i:i+batch_size], batch_embeds):
            outputs[path] = embed
            
    image_encoder_p = image_encoder_p.to("cpu")
    return outputs
   
   
def lora_finetuning(config: LoraFinetuningConfig):
    import bitsandbytes as bnb
    from pcdms.InpaintingStage import InpaintingStage, InpaintingConfig, InpaintingSampleInput
    from CelebrityDataset import CelebrityDataset, CelebrityCollateFn
    
    
    output_dir = Path(config.output_dir)
    logging_dir = output_dir / "wandb"
            
    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        mixed_precision=config.mixed_precision,
        project_dir=logging_dir,
        log_with="wandb" if config.wandb_key else None,
    )
     # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO, )
    # logger.info(accelerator.state, main_process_only=False)
    logger.info(accelerator.state)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()
    
    if accelerator.is_local_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        logging_dir.mkdir(parents=True, exist_ok=True)
    # Set the seed for reproducibility
    set_seed(config.seed)
    # log accelerator state
    logger.info(accelerator.state)
    
    stage_config = InpaintingConfig(precision=config.mixed_precision)
    if config.preload_embeddings:
        stage_config.device = "cpu"
    stage = InpaintingStage(stage_config)
    
    lora_config = LoraConfig(
        r=config.lora_rank,
        bias=config.lora_bias,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=UNET_TARGET_MODULES,
    )
    
    sd_model = stage.sd_model
    if is_xformers_available():
        sd_model.unet.enable_xformers_memory_efficient_attention()
    if config.gradient_checkpointing:
        sd_model.enable_gradient_checkpointing()
    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
    if config.scale_lr:
        config.learning_rate = (
                config.learning_rate 
                * config.gradient_accumulation_steps 
                * config.train_batch_size 
                * accelerator.num_processes
        )
        
    sd_model = get_peft_model(stage.sd_model, lora_config)
    sd_model.print_trainable_parameters()
    

    optimizer = bnb.optim.AdamW8bit(
        params=sd_model.parameters(),
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
        config.max_samples,
        split_ratio=0.999
    )
    val_dataset = CelebrityDataset(
        config.root_dir, 
        config.celebrity_name, 
        config.image_resize, 
        config.num_images, 
        config.seed, 
        config.max_samples,
        split="val",
        split_ratio=0.999
    )
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=accelerator.num_processes, rank=accelerator.process_index, shuffle=True)

    train_dataloader = DataLoader(
        train_dataset, 
        sampler=train_sampler,
        batch_size=config.train_batch_size, 
        pin_memory=True, 
        num_workers=config.num_dataloader_workers, 
        collate_fn= lambda batch : CelebrityCollateFn(batch, image_size=config.image_resize)
    )
    output_embeddings_p = {}
    output_embeddings_g = {}
    if accelerator.is_main_process and config.preload_embeddings:
        logger.info("Preloading embeddings")
        output_embeddings_p = get_image_embeddings_p(stage.image_encoder_p, train_dataset, accelerator.device, stage.weight_dtype, config.image_resize)
        output_embeddings_g = get_image_embeddings_g(stage.image_encoder_g, train_dataset, accelerator.device, stage.weight_dtype, config.image_resize)
        logger.info(f"Preloaded embeddings for {len(output_embeddings_p)} images")
        # since we are moved all to cpu, we need to move the model to the device
        logger.info("move sd_model and vae to device")
        stage.sd_model = stage.sd_model.to(config.device)
        stage.vae = stage.vae.to(config.device)
    
    accelerator.wait_for_everyone()

        
    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / config.gradient_accumulation_steps)
    if config.max_train_steps is None:
        config.max_train_steps = config.train_num_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True
        
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / config.gradient_accumulation_steps)
    max_train_steps = config.train_num_epochs * num_update_steps_per_epoch
    
    lr_scheduler = get_scheduler(
        config.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=max_train_steps * accelerator.num_processes,
        num_cycles=config.lr_num_cycles,
        power=config.lr_power,
    )
    
    # Prepare everything with our `accelerator`
    sd_model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        sd_model, optimizer, train_dataloader, lr_scheduler
    )
    
    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        
    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / config.gradient_accumulation_steps)
    if overrode_max_train_steps:
        config.max_train_steps = config.train_num_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    config.train_num_epochs = math.ceil(config.max_train_steps / num_update_steps_per_epoch)

    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name=config.wandb_project_name, config=config.model_dump(),
        )
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
        sd_model.load_pretrained(checkpoint_path)
        accelerator.load_state(checkpoint_path)
        # load global step and epoch from trainer state
        with open(Path(config.resume_from_checkpoint) / "trainer_state.json") as f:
            state = json.load(f)
        global_step = state["global_step"]
        first_epoch = state["epoch"]
        
    progress_bar = tqdm(range(global_step, max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")
    
    sd_model.train()
    for epoch in range(first_epoch, config.train_num_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(sd_model):
                for key in ["mask", "source_target_pose", "source_target_image", "vae_source_mask_image"]:
                    batch[key] = batch[key].to(accelerator.device, dtype=weight_dtype)
                with torch.no_grad():
                    # Convert images to latent space
                    latents = stage.vae.encode(batch["source_target_image"]).latent_dist.sample() * stage.vae.config.scaling_factor
                    masked_latents = stage.vae.encode(batch["vae_source_mask_image"]).latent_dist.sample() * stage.vae.config.scaling_factor
                    if config.preload_embeddings:
                        cond_image_feature_p = torch.stack([output_embeddings_p[s_img_path] for s_img_path in batch["s_img_path"]], dim=0).to(accelerator.device, dtype=weight_dtype)
                        cond_image_feature_g = torch.stack([output_embeddings_g[t_img_path] for t_img_path in batch["t_img_path"]], dim=0).to(accelerator.device, dtype=weight_dtype)
                    else:
                        batch["source_image"] = batch["source_image"].to(accelerator.device, dtype=weight_dtype)
                        batch["target_image"] = batch["target_image"].to(accelerator.device, dtype=weight_dtype)
                        cond_image_feature_p = stage.image_encoder_p(batch["source_image"]).last_hidden_state
                        cond_image_feature_g = stage.image_encoder_g(batch["target_image"]).image_embeds.unsqueeze(1)
                
                noise = torch.randn_like(latents)
                if config.noise_offset:
                    # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                    noise += config.noise_offset * torch.randn(
                        (latents.shape[0], latents.shape[1], 1, 1), device=latents.device
                    )
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
                    accelerator.clip_grad_norm_(sd_model.parameters(), config.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=config.set_grads_to_none)
                
            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
            
                if global_step % config.train_save_steps == 0:
                    checkpoint_model(sd_model, output_dir, global_step, epoch, accelerator)

            logs = {"loss": loss.item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            if global_step >= config.max_train_steps:
                break
            
            if accelerator.is_main_process and global_step % config.validate_every_n_steps == 0:
                logger.info("freeing memory")
                del output_embeddings_p, output_embeddings_g, batch, latents, masked_latents, noise, timesteps, noisy_latents, unet_input, model_pred, target
                gc.collect()
                torch.cuda.empty_cache()

                logger.info(f"Running validation at step {global_step}")
                sd_model.eval()
                stage.sd_model = sd_model
                stage.image_encoder_p = stage.image_encoder_p.to(accelerator.device)
                stage.image_encoder_g = stage.image_encoder_g.to(accelerator.device)
                stage.device = accelerator.device
                output_images = []
                with torch.no_grad():
                    for idx in tqdm(range(len(val_dataset)), desc="Validation"):
                        output = stage([InpaintingSampleInput.model_validate(val_dataset[idx])])
                        output_images.append(output)
                logger.info("logging images to wandb")
                accelerator.log({
                    "outputs": wandb.Image(show_images(torch.stack(output_images, dim=1).squeeze(0)))
                }, step=global_step)
                stage.image_encoder_p = stage.image_encoder_p.to("cpu")
                stage.image_encoder_g = stage.image_encoder_g.to("cpu")
                sd_model.train()

    if accelerator.is_main_process:
        checkpoint_model(sd_model, output_dir, global_step, epoch, accelerator)

    accelerator.wait_for_everyone()
    accelerator.end_training()
    
def main():
    parser = argparse.ArgumentParser(description="Run LoRA fine-tuning using a JSON config.")
    parser.add_argument(
        "--config_path",
        type=str,
        default="config/lora_finetune_example.json",
        help="Path to config JSON file"
    )
    args = parser.parse_args()

    config_path = BASE_DIR / args.config_path
    with open(config_path, "r") as f:
        cfg_dict = json.load(f)

    config = LoraFinetuningConfig(**cfg_dict)

    if config.wandb_key:
        wandb.login(key=config.wandb_key)
        wandb.init(project=config.wandb_project_name)

    lora_finetuning(config)

if __name__ == "__main__":
    main()
