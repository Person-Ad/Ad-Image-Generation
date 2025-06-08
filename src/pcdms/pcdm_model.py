import os
import sys
import json
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../external/PCDMs/src')))

import torch
import torch.nn as nn
from diffusers import DDIMScheduler
from constants import MODELS_SAVE_PATH, BASE_DIR
from pipelines.PCDMs_pipeline import PCDMsPipeline
from transformers import CLIPVisionModelWithProjection
from diffusers.models.controlnets.controlnet import ControlNetConditioningEmbedding
from models.stage2_inpaint_unet_2d_condition import Stage2_InapintUNet2DConditionModel

from pcdms.pcdm_utils import *

def load_pcdm_checkpoint(ckpt_path = MODELS_SAVE_PATH / "pcdm/pcdms_ckpt.pt"):
    model_ckpt = torch.load(ckpt_path)
    
    unet_dict = {}
    pose_proj_dict = {}
    image_proj_model_dict = {}

    for key, value in model_ckpt['module'].items():
        # sub models
        model_name = key.split('.')[0]
        model_key = key[len(model_name)+1:]
        # put weights in correct dict
        if model_name == 'pose_proj':
            pose_proj_dict[model_key] = value
        elif model_name == 'unet':
            unet_dict[model_key] = value
        elif model_name == 'image_proj_model':
            image_proj_model_dict[model_key] = value
        else:
            raise FileNotFoundError("no model called that")
        
    return unet_dict,  pose_proj_dict,  image_proj_model_dict
# ImageProjModel will project `embeddings` output from `image_encoder` to input to SD
class ImageProjModel(torch.nn.Module):
    """SD model with image prompt"""
    def __init__(self, in_dim, hidden_dim, out_dim, dropout = 0.):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, out_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):  
        return self.net(x)

def load_pipeline_components(device, dtype):
    # load unet from "stable diffusion v2.1" and fed it to stag2model
    unet = Stage2_InapintUNet2DConditionModel.from_pretrained("stabilityai/stable-diffusion-2-1-base", 
                                                              subfolder="unet",
                                                              in_channels=9, 
                                                              low_cpu_mem_usage=False, ignore_mismatched_sizes=True,
                                                              torch_dtype=dtype,
    )#.to(device)

    noise_scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
        steps_offset=1,
    )
    
    image_proj_model = ImageProjModel(in_dim=1536, hidden_dim=768, out_dim=1024).to(device).to(dtype)
    
    # pose encoder
    pose_proj_model = ControlNetConditioningEmbedding(
            conditioning_embedding_channels=320,
            block_out_channels=(16, 32, 96, 256),
            conditioning_channels=3
    ).to(device).to(dtype=dtype)
    
    # load checkpoint
    unet_dict,  pose_proj_dict,  image_proj_model_dict = load_pcdm_checkpoint()
    unet.load_state_dict(unet_dict)
    pose_proj_model.load_state_dict(pose_proj_dict)
    image_proj_model.load_state_dict(image_proj_model_dict)
    
    pipe = PCDMsPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-base", 
                                        unet=unet,  
                                        scheduler=noise_scheduler,
                                        feature_extractor=None,
                                        safety_checker=None,
                                        torch_dtype=dtype, 
                                )#.to(device)
    pipe.enable_model_cpu_offload()
    
    return pipe, unet, pose_proj_model, image_proj_model

if __name__ == "__main__":
    dtype = torch.bfloat16
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("start load components ... ")
    pipe, unet, pose_proj_model, image_proj_model = load_pipeline_components(device, dtype)
    
    num_samples = 1
    image_size = (512, 512) # W, H
    s_img_path = BASE_DIR / 'external/PCDMs/imgs/img1.png'
    target_pose_img = BASE_DIR / 'external/PCDMs/imgs/pose1.png'
    
    print("start inference ... ")
    output = inference_one_image(pipe, pose_proj_model, image_proj_model, s_img_path, target_pose_img, image_size=image_size, dtype=dtype).images[-1]
    
    output.show()
    