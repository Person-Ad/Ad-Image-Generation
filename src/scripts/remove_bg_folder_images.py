import torch
from PIL import Image
from pathlib import Path
from tqdm.auto import tqdm
from rembg import remove

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from constants import BASE_DIR

def remove_bg_folder_images(input_dir: Path, output_dir: Path, image_resize = (512, 512)):
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
        output_image = remove(image)
        # 3. save image
        output_image.save(output_dir / (image_path.stem + ".png"))
        
if __name__ == "__main__":
    remove_bg_folder_images(BASE_DIR / "datasets/celebrities/mo_salah/cropped", BASE_DIR / "datasets/celebrities/mo_salah/cleaned")