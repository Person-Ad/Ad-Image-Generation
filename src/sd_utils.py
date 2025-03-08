import torch
from transformers import CLIPTextModel, CLIPTokenizer


def encode_vae(img_clean, vae):
    if img_clean.ndim == 3:
        img_clean = img_clean.unsqueeze(0) # VAE excpect (B, C, H, W)
    img_clean = img_clean.type(vae.dtype).to(vae.device)
    
    with torch.no_grad():
        img_latents = vae.encode(img_clean).latent_dist.sample() * vae.config.scaling_factor
    return img_latents

def decode_vae(img_latents, vae):
    outputs = img_latents / vae.config.scaling_factor
    with torch.no_grad():
        outputs  = vae.decode(outputs).sample
    return outputs


def  get_prompt_embeddings(device, dtype):
    """ get inital prompt embeddings to help model """
    # text encoder components
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=dtype)
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=dtype)#.to(device)

    prompt = "An ultra high-resolution, photorealistic image in 8K, A highly detailed photo, showcasing vivid colors and rich contrasts, portrait photo, perfect anatomy, 4k, high quality, full cloth"
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

def get_mask(image_size, dtype, device):
        """ generate grey mask with white for src_img and black for tar_img """
        black_ = torch.full((image_size), 0, dtype=dtype).to(device)
        white_ = torch.full(image_size, 1, dtype=dtype).to(device)
        
        return torch.cat([white_, black_], dim=1)
