import torch
from PIL import Image, ImageOps
from pathlib import Path
from tqdm.auto import tqdm

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from constants import BASE_DIR

def resize_and_pad(image, image_resize=(512, 512), pad_color=255):
    """Resize the image keeping aspect ratio and pad the shorter side."""
    original_size = image.size
    ratio = min(image_resize[0] / original_size[0], image_resize[1] / original_size[1])
    new_size = (int(original_size[0] * ratio), int(original_size[1] * ratio))
    image = image.resize(new_size, Image.BICUBIC)

    delta_w = image_resize[0] - new_size[0]
    delta_h = image_resize[1] - new_size[1]
    padding = (delta_w // 2, delta_h // 2, delta_w - delta_w // 2, delta_h - delta_h // 2)
    return ImageOps.expand(image, padding, fill=pad_color)

def resize_and_crop(image, image_resize=(512, 512)):
    """Resize while maintaining aspect ratio and center-crop to fit the target size."""
    image = ImageOps.fit(image, image_resize, Image.BICUBIC, centering=(0.5, 0.5))
    return image

def resize_folder_images(input_dir: Path, output_dir: Path, image_resize = (512, 512), pad = None):
    """
    Args:
        - input_dir (Path): Input Directory contain images of (jpg, png, jpeg)
        - output_dir (Path): Output Directory will contain openpose images
        - image_resize (Tuple | None): if none no resize happen
        - pad: pad shorter side if possible (None, 0 for black, 255 for white)
    """
    output_dir.mkdir(exist_ok=True, parents=True)
    for image_path in tqdm(input_dir.glob("*.[jp][pn]*g")): 
        # 1. read image and preprocess it
        image = Image.open(image_path).convert("RGB")
        if image_resize:
            if pad is not None:
                image = resize_and_pad(image, image_resize, pad_color=pad)
            else:
                image = resize_and_crop(image, image_resize)
        # 3. save image
        image.save(output_dir / image_path.name)
        
if __name__ == "__main__":
    resize_folder_images(BASE_DIR / "datasets/celebrities/meh/filtered_faces_cleaned", BASE_DIR / "datasets/celebrities/meh/filtered_faces_cleaned_resized", pad=0)