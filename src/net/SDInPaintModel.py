import torch
from diffusers.models.controlnets.controlnet import ControlNetConditioningEmbedding

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from net.ImageProjModel_p import ImageProjModel_p
from PCDMs.src.models.stage2_inpaint_unet_2d_condition import Stage2_InapintUNet2DConditionModel

    
class SDInPaintModel(torch.nn.Module):
    """Inpainting Conditional SD Model """
    def __init__(self, unet: Stage2_InapintUNet2DConditionModel) -> None:
        super().__init__()
        # MLP Layer between Dino Output to cat with OpenClip/Prior Embeddings
        self.image_proj_model_p = ImageProjModel_p(in_dim=1536, hidden_dim=768, out_dim=1024)

        self.unet = unet
        
        self.pose_proj = ControlNetConditioningEmbedding(
            conditioning_embedding_channels=320,
            block_out_channels=(16, 32, 96, 256),
            conditioning_channels=3)


    def forward(self, noisy_latents, timesteps, simg_f_p, timg_f_g, pose_f):
        """
        Args:
            - simg_f_p: embeddings output from src image using DinoV2
            - timg_f_g: embeddings output from tar,src image using Prior/OpenClip
            - pose_f:   src_pose concated in width dimension with tar_pose prediction openpose
        """
        extra_image_embeddings_p = self.image_proj_model_p(simg_f_p)
        extra_image_embeddings_g = timg_f_g

        encoder_image_hidden_states = torch.cat([extra_image_embeddings_p ,extra_image_embeddings_g], dim=1)
        pose_cond = self.pose_proj(pose_f)

        pred_noise = self.unet(noisy_latents, 
                               timesteps, 
                               class_labels=timg_f_g, 
                               encoder_hidden_states=encoder_image_hidden_states, 
                               my_pose_cond=pose_cond).sample
        return pred_noise
