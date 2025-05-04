import os
import sys
import json
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from loguru import logger
import time

from tqdm.auto import tqdm
import numpy as np
from PIL import Image  
import matplotlib.pyplot as plt

import torch
import accelerate
from torch import nn
from torch.nn import functional as F

from  torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPTextModel, CLIPTokenizer, CLIPImageProcessor, Dinov2Model

import diffusers
from diffusers import (
    PNDMScheduler, # scheduler
    UNet2DConditionModel, # unet
    AutoencoderKL, # vae
)
# ----------- local imports ----------- 
from FashionDataset import FashionDataset
from net.ZeroConvNet import ZeroConvNet
from constants import MODELS_SAVE_PATH
from utils import show_images, build_wandb_run
import wandb

DEBUGGING = True
model_save_directory = MODELS_SAVE_PATH / "finetune_f16_cond_as_channel_fashion_16"
model_save_directory.mkdir(exist_ok=True, parents=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("the device used", device)

generator = torch.Generator(device=device).manual_seed(42)
generator_cpu = torch.Generator(device='cpu').manual_seed(42)

# DEBUGGING
# torch.autograd.set_detect_anomaly(True)
    
dtype=torch.bfloat16

batch_size = 4
accumulation_steps = 4
num_epochs = 3
learning_rate = 1e-5
log_every =  64
save_every = 128
# small_dataset_size = 1024
small_dataset_size = 1024

def _get_prompt_embeddings():
    """ get inital prompt embeddings to help model """
    # text encoder components
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=dtype)
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=dtype)#.to(device)

    prompt = "An ultra high-resolution, photorealistic image in 8K, A highly detailed photo, showcasing vivid colors and rich contrasts, portrait photo, perfect anatomy, 4k, high quality"
    negative_prompt = "nudity (deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime), text, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck"
        
    # Tokenize positive and negative prompts
    positive_token_ids = tokenizer(prompt, padding="max_length", truncation=True, return_tensors="pt")['input_ids']#.to(device)
    negative_token_ids = tokenizer(negative_prompt, padding="max_length", truncation=True, return_tensors="pt")['input_ids']#.to(device)

    # Generate embeddings
    with torch.no_grad():
        positive_embeddings = text_encoder(positive_token_ids).last_hidden_state.type(dtype).to(device)
        negative_embeddings = text_encoder(negative_token_ids).last_hidden_state.type(dtype).to(device)

        combined_embeddings = torch.cat([negative_embeddings, positive_embeddings], dim=0)#.type(dtype)
    
    return positive_embeddings, negative_embeddings, combined_embeddings

def get_mask(image_size):
        """ generate grey mask with white for src_img and black for tar_img """
        black_ = torch.full((image_size), 0, dtype=dtype).to(device)
        white_ = torch.full(image_size, 1, dtype=dtype).to(device)
        
        return torch.cat([white_, black_], dim=1)

        

# ====================== load models ======================
model_id = "SG161222/Realistic_Vision_V6.0_B1_noVAE"

scheduler = PNDMScheduler.from_pretrained(model_id, subfolder="scheduler")
vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse-original", torch_dtype=dtype).to(device)

net = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet",
                                        in_channels=13,
                                        low_cpu_mem_usage=False,
                                        ignore_mismatched_sizes=True,
                                        torch_dtype=dtype).to(device)
# net = ZeroConvNet(unet, in_channels=13, out_channels=4).type(dtyp/;9878e).to(device)



# ====================== prepare assets ======================
run = build_wandb_run(config = {
    "model_id": model_id,
    "condition": "as second channel",
    "dataset": "FASHION-1000-sample",
    "learning_rate": learning_rate,
    "dtype": dtype,
    "batch_size": batch_size,
    "accumulation_steps" : accumulation_steps,
    "num_epochs": num_epochs,
    "small_dataset_size": small_dataset_size
})


positive_embeddings, negative_embeddings, combined_embeddings = _get_prompt_embeddings()
embeddings = positive_embeddings.repeat(batch_size, 1, 1)

train_dataset = FashionDataset(small_dataset_size=small_dataset_size)
train_dataloader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, generator=generator_cpu
)

inputs = next(iter(train_dataloader))
# Define the optimizer
optimizer = torch.optim.AdamW(net.parameters(), lr=learning_rate)
lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
loss_fn = nn.MSELoss()

optimizer.zero_grad()

# ============================================================
# ============================================================
# ============================================================
# ============================================================
mask = get_mask((256 // 8, 352 // 16)).unsqueeze(0).unsqueeze(0) #[white, black]
mask = mask.expand(batch_size, -1, -1, -1)

for epoch in range(num_epochs):
    total_loss = 0
    logger.info(f"EPOCH: {epoch}")
    for step, inputs in enumerate(tqdm(train_dataloader)):
        # 1. prepare inputs
        src_mask = inputs['src_mask'].type(dtype).to(device)
        st_pose = inputs['st_pose'].type(dtype).to(device)
        src_tar = inputs['src_tar'].type(dtype).to(device)
        # 2. prepare latents
        with torch.no_grad():
            # masked_latents
            src_mask_latents = vae.encode(src_mask).latent_dist.sample() * 0.18215
            # latents 
            src_tar_latents = vae.encode(src_tar).latent_dist.sample() * 0.18215 
            
            st_pose_latents = vae.encode(st_pose).latent_dist.sample() * 0.18215 

        # 3. add noise to src_tar_latents & concate all in channel dimension
        noise = torch.randn_like(src_tar_latents, device=device, dtype=dtype)
        timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (batch_size,), device=device, ).long()

        noisy_latents = scheduler.add_noise(src_tar_latents, noise, timesteps)
        noisy_latents = torch.cat([noisy_latents, src_mask_latents, mask, st_pose_latents], dim=1).type(dtype)

        # return 1
        # Forward pass
        noise_pred = net(noisy_latents, timesteps, encoder_hidden_states=embeddings).sample
        loss = loss_fn(noise_pred, noise) /accumulation_steps
        # Backward pass
        loss.backward()
        total_loss += loss.item()
        run.log({'loss':loss.item()})
        
        if (step + 1) % accumulation_steps == 0 or (step + 1) == len(train_dataloader):
            optimizer.step()
            optimizer.zero_grad()
            
        if (step+1) % log_every == 0:
            scheduler.set_timesteps(50)
            # Prep latents, 
            latents = torch.randn((batch_size, 4, 256 // 8, 352 // 8), generator=generator, device=device, dtype=dtype)
            for i, t in tqdm(enumerate(scheduler.timesteps), total=len(scheduler.timesteps)):
                latent_model_input = torch.cat([latents, src_mask_latents, mask, st_pose_latents], dim=1)
                with torch.no_grad():
                    noise_pred = net(latent_model_input, t, encoder_hidden_states=embeddings).sample
                latents = scheduler.step(noise_pred, t, latents).prev_sample
            # decod sample
            latents = (1 / 0.18215 * latents).clip(-1, 1)
            with torch.no_grad():
                outputs = vae.decode(latents).sample
            # log
            run.log({'epoch': epoch, 
                     'step': step, 
                     'Targets': wandb.Image(show_images(src_tar.type(torch.float32))),
                     'Samples': wandb.Image(show_images(outputs.type(torch.float32)))
            })
            
        if (step+1) % save_every == 0:
            torch.save({
                'optimizer': optimizer.state_dict(),
                'model': net.state_dict()
            }, model_save_directory / f"step_{step}.pth.tar")
            
    logger.info(f"EPOCH: {epoch}\t loss: {total_loss / len(train_dataloader):.4f}")
    lr_scheduler.step()


print("============ FINISH ============ ")
for name, param in net.unet.named_parameters():
    if torch.isnan(param).any():
        print("FUCKKKKK, ", name)
        break
run.finish()