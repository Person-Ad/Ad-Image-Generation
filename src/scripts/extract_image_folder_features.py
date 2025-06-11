import torch
from PIL import Image
from pathlib import Path
from tqdm.auto import tqdm
from loguru import logger
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor, Dinov2Model
# Local imports
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from constants import BASE_DIR

def extract_image_folder_clip_features(input_dir: Path, device, weight_dtype, output_path: Path = None, image_size = (512, 512), batch_size=32) -> dict[str, torch.Tensor]:
    """
    Get the image embeddings for the target images in batches.
    Args:
        - input_dir (Path): Input Directory contain images of (jpg, png, jpeg)
        - device (torch.device): cuda or cpu
        - weight_dtype (torch.dtype)
        - output_path (Path | None): Output Path of  will pt file contain output, if None will return without saving (example: clip_output.pt)
        - image_size (Tuple | None): if none no resize happen
        - batch_size (int)
    Returns:
        - output: dict[str, torch.Tensor] of path of image file in key and value is features
    """
    logger.info("start extracting clip features...")
    
    image_encoder_g = CLIPVisionModelWithProjection.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K").to(device, dtype=weight_dtype)
    image_encoder_g.requires_grad_(False)
    image_encoder_g.eval()
    
    clip_image_processor = CLIPImageProcessor()

    # Collect image paths
    image_paths = list(input_dir.glob("*.[jp][pn]*g"))
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
    
    if output_path:
        torch.save(outputs, output_path)    
    
    return outputs


def extract_image_folder_dino_features(input_dir: Path, device, weight_dtype, output_path: Path = None, image_size = (512, 512), batch_size=32) -> dict[str, torch.Tensor]:
    """
    Get the image embeddings for the target images in batches.
    Args:
        - input_dir (Path): Input Directory contain images of (jpg, png, jpeg)
        - device (torch.device): cuda or cpu
        - weight_dtype (torch.dtype)
        - output_path (Path | None): Output Path of  will pt file contain output, if None will return without saving (example: clip_output.pt)
        - image_size (Tuple | None): if none no resize happen
        - batch_size (int)
    Returns:
        - output: dict[str, torch.Tensor] of path of image file in key and value is features
    """
    logger.info("start extracting dino features...")
    image_encoder_p = Dinov2Model.from_pretrained("facebook/dinov2-giant").to(device, dtype=weight_dtype)
    image_encoder_p.requires_grad_(False)
    image_encoder_p.eval()
    
    clip_image_processor = CLIPImageProcessor()

    # Collect image paths
    image_paths = list(input_dir.glob("*.[jp][pn]*g"))
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
            batch_embeds = image_encoder_p(batch_tensor).last_hidden_state.to("cpu")

        for path, embed in zip(image_paths[i:i+batch_size], batch_embeds):
            outputs[path] = embed
    
    if output_path:
        torch.save(outputs, output_path)    
    
    return outputs

if __name__ == "__main__":
    celebrity_dir = BASE_DIR / "datasets/celebrities/mo_salah"
    input_dir = celebrity_dir / "cleaned"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_dtype = torch.float16
    
    extract_image_folder_clip_features(input_dir, device, weight_dtype, celebrity_dir / "clip.pt", image_size = (512, 512), batch_size=32)
    extract_image_folder_dino_features(input_dir, device, weight_dtype, celebrity_dir / "dino.pt", image_size = (512, 512), batch_size=32)