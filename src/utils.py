from pathlib import Path
import gdown
import zipfile

import torchvision
from PIL import Image
import numpy as np

from loguru import logger

def download_gfile(file_id, destination: Path):
    if destination.exists():
        logger.warning(f"File already exists: {destination}")
    else:
        gdown.download(id=file_id, output=str(destination), quiet=False)

def unzip_file(zip_path, extract_to):
    if extract_to.exists():
        logger.warning(f"Already extracted: {extract_to}")
    else:
        logger.info(f"Extracting {zip_path.name}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)


def show_images(x):
    """Given a batch of images x, make a grid and convert to PIL"""
    x = x * 0.5 + 0.5  # Map from (-1, 1) back to (0, 1)
    grid = torchvision.utils.make_grid(x) # C, H, W
    grid_im = grid.detach().cpu().permute(1, 2, 0).clip(0, 1) * 255 # H, W, C 
    grid_im = Image.fromarray(np.array(grid_im).astype(np.uint8))
    return grid_im


def make_grid(images, size=256):
    """Given a list of PIL images, stack them together into a line for easy viewing"""
    output_im = Image.new("RGB", (size * len(images), size))
    for i, im in enumerate(images):
        output_im.paste(im.resize((size, size)), (i * size, 0))
    return output_im
