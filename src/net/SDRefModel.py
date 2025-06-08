import torch
from diffusers import UNet2DConditionModel

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from net import ImageProjModel_p

class SDRefModel(torch.nn.Module):
    """SD model for Refinement"""
    def __init__(self, unet: UNet2DConditionModel) -> None:
        super().__init__()
        self.image_proj_model_p = ImageProjModel_p(in_dim=1536, hidden_dim=768, out_dim=1024)

        self.unet = unet


    def forward(self, noisy_latents, timesteps,  s_img_embed):  # img_f 8,1024
        """
        encoder_hidden_states : 8,77, 1024
        control_image_feature:  8, 257, 1024
        """
        extra_image_embeddings_p = self.image_proj_model_p(s_img_embed)  # s_img: bs,257,1536

        pred_noise = self.unet(noisy_latents, timesteps, encoder_hidden_states=extra_image_embeddings_p).sample
        return pred_noise
