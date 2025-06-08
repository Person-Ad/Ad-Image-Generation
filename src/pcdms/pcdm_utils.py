import torch
from PIL import Image
from torchvision import transforms
from controlnet_aux import OpenposeDetector
from transformers import CLIPImageProcessor, Dinov2Model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = torch.Generator(device=device).manual_seed(42)

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]), # transform pixels in range [-1, 1]
])



def inference_pose(img, openpose):
    pose = openpose(img, detect_resolution=img.size[1],  include_hand=True).resize(img.size, Image.BICUBIC)
    return pose

def get_mask(latent_size, dtype = torch.float32):
    mask1 = torch.ones((1, 1, int(latent_size[0]), int(latent_size[1] / 2))).to(device, dtype) # HxW
    mask0 = torch.zeros((1, 1, int(latent_size[0]), int(latent_size[1] / 2))).to(device, dtype) # HxW
    mask = torch.cat([mask1, mask0], dim=3)
    return mask

def get_inpainting_inputs(pipe, s_img, dtype = torch.float32):
    """ concatenate source with mask - do basic processing - return latents output from VAE """
    # 1. concatenate source with mask
    black_image = Image.new("RGB", s_img.size, (0, 0, 0)).resize(s_img.size, Image.BICUBIC)
    
    s_img_t_mask = Image.new("RGB", (s_img.width * 2, s_img.height))
    s_img_t_mask.paste(s_img, (0, 0))
    s_img_t_mask.paste(black_image, (s_img.width, 0))
    # 2. do basic processing
    vae_image = torch.unsqueeze(img_transform(s_img_t_mask), 0)
    # 3. get latents from VAE
    with torch.inference_mode():
        latents = pipe.vae.encode(vae_image.to(device, dtype=dtype)).latent_dist.sample()
        latents = latents * 0.18215 # since VAE paper do that 
        
    return latents

def get_inpainting_cond(pose_proj_model, openpose, s_img, t_pose, dtype = torch.float32):
    """ concatenate source pose with target pose -- project conditions"""
    # 1. concatenate source pose with target pose
    s_pose = inference_pose(s_img, openpose).resize(s_img.size, Image.BICUBIC)
    st_pose = Image.new("RGB", (s_pose.width * 2, s_pose.height))
    st_pose.paste(s_pose, (0, 0))
    st_pose.paste(t_pose, (s_pose.width, 0))

    # 2. project conditions
    cond_st_pose = torch.unsqueeze(img_transform(st_pose), 0)
    with torch.inference_mode():
        cond_pose = pose_proj_model(cond_st_pose.to(dtype=dtype, device=device))
    
    return cond_pose

def get_image_embeddings(image_encoder, image_proj_model, s_img, num_samples = 1, dtype = torch.float32):
    clip_image_processor = CLIPImageProcessor()
    # do basic processing 
    clip_s_img = clip_image_processor(images=s_img, return_tensors="pt").pixel_values 
    # projected encoded embeddings for both (conditional & uncondational)
    with torch.inference_mode():
        images_embeds = image_encoder(clip_s_img.to(device, dtype=dtype)).last_hidden_state
        image_prompt_embeds = image_proj_model(images_embeds)
        uncond_image_prompt_embeds = image_proj_model(torch.zeros_like(images_embeds))
    
    # repeat inputs to count for unconditional embeddings
    bs_embed, seq_len, _ = image_prompt_embeds.shape
    image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1).view(bs_embed * num_samples, seq_len, -1)
    uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(1, num_samples, 1).view(bs_embed * num_samples, seq_len, -1)
    
    return image_prompt_embeds, uncond_image_prompt_embeds




def inference_one_image(pipe,
                        pose_proj_model,
                        image_proj_model,
                        image_encoder,
                        openpose,
                        s_img_path = './imgs/img1.png', target_pose_path = './imgs/pose1.png', 
                        image_size = (512, 512),
                        num_inference_steps = 50,
                        guidance_scale = 2.0,
                        generator = generator, 
                        dtype = torch.float32):
    # ======================== Preprocessing ==========================================
    # 1. read image
    s_img = Image.open(s_img_path).convert("RGB").resize(image_size, Image.BICUBIC)
    t_pose = Image.open(target_pose_path).convert("RGB").resize((image_size), Image.BICUBIC)
    # 2. get inpainting input
    simg_mask_latents = get_inpainting_inputs(pipe, s_img, dtype) # batch_size x 4 x latent_h x latent_w
    mask = get_mask(simg_mask_latents.shape[2:], dtype)  # 1 x 1 x latent_h x latent_w
    # 3. get conditional pose
    cond_pose = get_inpainting_cond(pose_proj_model, openpose, s_img, t_pose, dtype=dtype)
    # 4. get image embeddings
    image_prompt_embeds, uncond_image_prompt_embeds = get_image_embeddings(image_encoder, image_proj_model, s_img, dtype=dtype)

    print("simg_mask_latents", simg_mask_latents.shape)
    print("mask", mask.shape)
    print("cond_pose", cond_pose.shape)
    print("prompt_embeds", image_prompt_embeds.shape)
    print("negative_prompt_embeds", uncond_image_prompt_embeds.shape)
    # ======================== Pipeline  ==========================================
    return pipe(
            simg_mask_latents= simg_mask_latents,
            mask = mask,
            cond_pose = cond_pose,
            prompt_embeds=image_prompt_embeds,
            negative_prompt_embeds=uncond_image_prompt_embeds, # ??
            height=image_size[1],
            width=image_size[0]*2, # for inpainting mask
            num_images_per_prompt=1,
            guidance_scale=guidance_scale,
            generator=generator,
            num_inference_steps=num_inference_steps,
        )