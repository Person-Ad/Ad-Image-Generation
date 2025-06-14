from tqdm.auto import tqdm
from pathlib import Path
import torch
from PIL import Image
from loguru import logger
from pydantic import BaseModel

from torchvision import transforms
from controlnet_aux import OpenposeDetector
from diffusers import AutoencoderKL, DDPMScheduler
from transformers import (
    CLIPVisionModelWithProjection,
    CLIPImageProcessor, 
    Dinov2Model
)

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from constants import BASE_DIR
from net.SDInPaintModel import SDInPaintModel, Stage2_InapintUNet2DConditionModel
from scripts.download_chekpoints import download_checkpoint

class InpaintingConfig(BaseModel):
    """
    Configuration for the Inpainting Stage.
    """
    checkpoint_path: str = "models/pcdm/s2_512.pt"
    pretrained_model_path: str = "stabilityai/stable-diffusion-2-1-base"
    vae_model_path: str = "https://huggingface.co/stabilityai/sd-vae-ft-mse-original/vae-ft-mse-840000-ema-pruned.ckpt"
    
    # TODO: add sequential offloading option
    # sequential_offloading = False  # Whether offloading models one by one or not
    
    user_prior_stage: bool = False # Whether to use user prior stage or Clip
    device: str = "cuda"  # Device to run the model on, e.g., 'cuda' or 'cpu'
    precision: str = "fp16"  # Precision for model weights, e.g., 'fp16' or 'fp32'
    
    preloaded_feature_dino_path: str | Path | None = None 
    preloaded_feature_clip_path: str | Path | None = None 
    

class InpaintingSampleInput(BaseModel):
    """
    Input data for the Inpainting Stage.
    """
    s_img_path: str | Path  # Path to the source image
    t_img_path: str  | Path # Path to the target image
    s_pose_path: str | Path | None = None  # Optional path to the source pose image
    t_pose_path: str | Path | None = None  # Optional path to the target pose image
    image_size: tuple = (512, 512)  # Size of the images, default is (512, 512)
    
class InpaintingInferenceConfig(BaseModel):
    num_inference_steps: int = 50  # Number of inference steps for the model
    guidance_scale: float = 2  # Guidance scale for the model


class InpaintingProcessor:
    """
    Processor for handling image inputs and transformations for the Inpainting Stage.
    """
    def __init__(self, device='cuda', preloaded_feature_dino=False, preloaded_feature_clip=False) -> None:
        self.clip_image_processor = CLIPImageProcessor()
        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        self.device = torch.device(device)
        

    def inference_pose(self, img):
        openpose = OpenposeDetector.from_pretrained("lllyasviel/ControlNet").to(self.device)
        return openpose(img, detect_resolution=img.size[0],  include_hand=True).resize(img.size, Image.BICUBIC)
        
    def process_input(self, s_img_path, t_img_path, s_pose_path = None, t_pose_path = None, image_size=(512, 512)):
        s_img = Image.open(s_img_path).convert("RGB").resize(image_size, Image.BICUBIC)
        t_img = Image.open(t_img_path).convert("RGB").resize(image_size, Image.BICUBIC)

        clip_s_img = self.clip_image_processor(images=s_img, return_tensors="pt").pixel_values.squeeze(dim=0)
        clip_t_img = self.clip_image_processor(images=t_img, return_tensors="pt").pixel_values.squeeze(dim=0)

        # s_img, black_mask
        black_img = Image.new("RGB", image_size, (0, 0, 0))
        s_img_mask = Image.new("RGB", (image_size[0] * 2, image_size[1]))
        s_img_mask.paste(s_img, (0, 0))
        s_img_mask.paste(black_img, (image_size[0], 0))

        st_img = (Image.new("RGB", (image_size[0] * 2, image_size[1])))
        st_img.paste(s_img, (0, 0))
        st_img.paste(black_img, (image_size[0], 0))

        s_pose = self.inference_pose(s_img) if not s_pose_path else Image.open(s_pose_path).convert("RGB").resize(image_size, Image.BICUBIC)
        t_pose = self.inference_pose(t_img) if not t_pose_path else Image.open(t_pose_path).convert("RGB").resize(image_size, Image.BICUBIC)

        st_pose = Image.new("RGB", (image_size[0] * 2, image_size[1]))
        st_pose.paste(s_pose, (0, 0))
        st_pose.paste(t_pose, (image_size[0], 0))

        # final transformations
        trans_s_img_mask = self.img_transform(s_img_mask)
        trans_st_img = self.img_transform(st_img)
        trans_st_pose = self.img_transform(st_pose)
        
        return {
            "source_image": clip_s_img, 
            "target_image": clip_t_img, 
            "vae_source_mask_image": trans_s_img_mask,
            "source_target_pose": trans_st_pose,
            "source_target_image": trans_st_img,
        }

    def process_batch(self, inputs: list[InpaintingSampleInput]):
        batch = [self.process_input(**sample.model_dump()) for sample in inputs]
        batch_size = len(batch)
        # mask
        mask1 = torch.ones((batch_size, 1, int(inputs[0].image_size[0] / 8), int(inputs[0].image_size[0] / 8)))
        mask0 = torch.zeros((batch_size, 1, int(inputs[0].image_size[0] / 8), int(inputs[0].image_size[0] / 8)))
        mask = torch.cat([mask1, mask0], dim=3)
        
        # dimensions handling
        source_images = torch.stack([sample['source_image'] for sample in batch]).to(memory_format=torch.contiguous_format).float()
        target_image = torch.stack([sample['target_image'] for sample in batch]).to(memory_format=torch.contiguous_format).float()
        vae_source_mask_images = torch.stack([sample['vae_source_mask_image'] for sample in batch]).to(memory_format=torch.contiguous_format).float()
        source_target_poses = torch.stack([sample['source_target_pose'] for sample in batch]).to(memory_format=torch.contiguous_format).float()
        source_target_images = torch.stack([sample['source_target_image'] for sample in batch]).to(memory_format=torch.contiguous_format).float()

        return {
            "source_image": source_images, 
            "target_image": target_image, 
            "vae_source_mask_image": vae_source_mask_images,
            "source_target_pose": source_target_poses,
            "source_target_image": source_target_images,
            "mask": mask
        }
        
class InpaintingStage():
    
    def __init__(self, config: InpaintingConfig, accelerator = None) -> None:
        self.config = config
        if not accelerator:
            self.device = torch.device("cuda" if torch.cuda.is_available() and config.device == "cuda" else "cpu")
        else:
            self.device = accelerator.device
        
        self.weight_dtype = torch.float16 if config.precision == "fp16" else torch.bfloat16 if config.precision == "bf16" else torch.float32  

        self.processor = InpaintingProcessor(self.device)

        self.checkpoint_path: Path = BASE_DIR / config.checkpoint_path
        if not self.checkpoint_path.exists():
            download_checkpoint(self.checkpoint_path.name, self.checkpoint_path.parent)
        
        self.load_pipeline()

        
    def load_pipeline(self):
        logger.info("start loading pipeline ...")
        progress_bar = tqdm(total=5, desc="Loading Inpainting Stage Pipeline")

        progress_bar.set_description("Loading UNet Model")
        self.unet = Stage2_InapintUNet2DConditionModel.from_pretrained(self.config.pretrained_model_path, subfolder="unet",
                                                   in_channels=9, class_embed_type="projection" ,projection_class_embeddings_input_dim=1024,
                                                  low_cpu_mem_usage=False, ignore_mismatched_sizes=True)
        self.unet.requires_grad_(False)
        
        progress_bar.update(1)
        progress_bar.set_description("Loading VAE Model")
        self.vae = AutoencoderKL.from_single_file(self.config.vae_model_path, torch_dtype=self.weight_dtype).to(self.device)
        self.vae.requires_grad_(False)
        
        self.noise_scheduler = DDPMScheduler.from_pretrained(self.config.pretrained_model_path, subfolder="scheduler")

        progress_bar.update(1)
        progress_bar.set_description("Loading DinoV2 Model")
        if not self.config.preloaded_feature_dino_path:
            self.image_encoder_p = Dinov2Model.from_pretrained("facebook/dinov2-giant").to(self.device, dtype=self.weight_dtype)
            self.image_encoder_p.requires_grad_(False)
        else:
            self.image_encoder_p_dict = torch.load(self.config.preloaded_feature_dino_path)
            self.image_encoder_p_dict = {k.name: v for k, v in self.image_encoder_p_dict.items()}
            
            logger.info(f"loaded dino feats of {len(self.image_encoder_p_dict)} features")
            
        progress_bar.update(1)
        progress_bar.set_description("Loading Prior/Clip Model")
        if not self.config.preloaded_feature_clip_path:
            if not self.config.user_prior_stage:
                self.image_encoder_g = CLIPVisionModelWithProjection.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K").to(self.device, dtype=self.weight_dtype)
            else:
                raise NotImplementedError("User prior stage is not implemented yet.")
            self.image_encoder_g.requires_grad_(False)
        else:
            self.image_encoder_g_dict = torch.load(self.config.preloaded_feature_clip_path)
            self.image_encoder_g_dict = {k.name: v for k, v in self.image_encoder_g_dict.items()}
            logger.info(f"loaded clip feats of {len(self.image_encoder_g_dict)} features")
            
        progress_bar.update(1)
        progress_bar.set_description("Loading Full Checkpoint")
        self.sd_model = SDInPaintModel(self.unet)
        self.sd_model.load_state_dict(torch.load(self.checkpoint_path)['module'])
        self.sd_model.requires_grad_(False)
        self.sd_model.to(self.device, dtype=self.weight_dtype)

        progress_bar.update(1)
        progress_bar.set_description("Pipeline Loaded")
        logger.info("pipeline loaded successfully")
        
    def to(self, device):
        self.vae = self.vae.to(device)
        self.image_encoder_p = self.image_encoder_p.to(device)
        self.image_encoder_g = self.image_encoder_g.to(device)
        self.sd_model = self.sd_model.to(device)
        
    def __call__(self, input_paths: list[InpaintingSampleInput], config: InpaintingInferenceConfig = InpaintingInferenceConfig()):
        """
        Args:
            s_img_path (str): Path to the source image.
            t_img_path (str): Path to the target image.
            s_pose_path (str, optional): Path to the source pose image. Defaults to None.
            t_pose_path (str, optional): Path to the target pose image. Defaults to None.
            image_size (tuple, optional): Size of the images. Defaults to (512, 512).
        """
        inputs = self.processor.process_batch(input_paths)

        for key in ["mask", "source_target_pose", "source_target_image", "vae_source_mask_image"]:
            inputs[key] = inputs[key].to(self.device, dtype=self.weight_dtype)
    
        with torch.no_grad():
            # Convert images to latent space
            latents = self.vae.encode(inputs["source_target_image"]).latent_dist.sample() * self.vae.config.scaling_factor
            # Get the masked image latents
            masked_latents = self.vae.encode(inputs["vae_source_mask_image"]).latent_dist.sample() * self.vae.config.scaling_factor
            # Get the image embedding for conditioning
            if self.config.preloaded_feature_dino_path:
                cond_image_feature_p = (torch.stack([ self.image_encoder_p_dict[sample.s_img_path.name] for sample in input_paths], dim=0)
                                        .to(self.device, dtype=self.weight_dtype))
            else:
                cond_image_feature_p = self.image_encoder_p(inputs["source_image"].to(self.device, dtype=self.weight_dtype)).last_hidden_state
            
            if self.config.preloaded_feature_clip_path:
                cond_image_feature_g = (torch.stack([ self.image_encoder_g_dict[sample.t_img_path.name] for sample in input_paths], dim=0)
                                        .to(self.device, dtype=self.weight_dtype))
            else:
                cond_image_feature_g = self.image_encoder_g(inputs["target_image"].to(self.device, dtype=self.weight_dtype)).image_embeds.unsqueeze(1)


        ####################################################################
        # Denoising loop
        ####################################################################
        self.noise_scheduler.set_timesteps(config.num_inference_steps)
        noise = torch.randn_like(latents)
        latents = self.noise_scheduler.add_noise(latents, noise, self.noise_scheduler.timesteps[0]).to(self.device, dtype=self.weight_dtype) 
        
        for t in tqdm(self.noise_scheduler.timesteps):
            unet_input = torch.cat([latents, inputs["mask"], masked_latents], dim=1)
            t_tensor = torch.tensor([t], device=self.device, dtype=torch.float32)

            with torch.no_grad():
                # CONDITIONAL prediction
                noise_pred_cond = self.sd_model(
                    unet_input,
                    t_tensor,
                    cond_image_feature_p,  # conditioned
                    cond_image_feature_g,
                    inputs["source_target_pose"],
                )

                # UNCONDITIONAL prediction
                noise_pred_uncond = self.sd_model(
                    unet_input,
                    t_tensor,
                    torch.zeros_like(cond_image_feature_p),  # unconditioned
                    torch.zeros_like(cond_image_feature_g),
                    torch.zeros_like(inputs["source_target_pose"]),
                )

                # CFG interpolation
                noise_pred = noise_pred_uncond + config.guidance_scale * (noise_pred_cond - noise_pred_uncond)
                # Denoising step
                latents = self.noise_scheduler.step(noise_pred, t, latents).prev_sample

        outputs = latents / self.vae.config.scaling_factor
        # Ensure the output tensor matches the VAE's weight dtype
        outputs = outputs.to(dtype=self.weight_dtype)
        with torch.no_grad():
            outputs  = self.vae.decode(outputs).sample
        return outputs