import torch
from PIL import Image
from pathlib import Path
from tqdm.auto import tqdm

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from constants import BASE_DIR

def resize_folder_images(input_dir: Path, output_dir: Path, image_resize = (512, 512)):
    """
    Args:
        - input_dir (Path): Input Directory contain images of (jpg, png, jpeg)
        - output_dir (Path): Output Directory will contain openpose images
        - image_resize (Tuple | None): if none no resize happen
    """
    output_dir.mkdir(exist_ok=True, parents=True)
    for image_path in tqdm(input_dir.glob("*.[jp][pn]*g")): 
        # 1. read image and preprocess it
        image = Image.open(image_path).convert("RGB")
        if image_resize:
            image = image.resize(image_resize, Image.BICUBIC)
        # 3. save image
        image.save(output_dir / image_path.name)
        
if __name__ == "__main__":
    resize_folder_images(BASE_DIR / "datasets/celebrities/mohamed_salah/cropped", BASE_DIR / "datasets/celebrities/mohamed_salah/resized")