import torch
from PIL import Image
from pathlib import Path
from tqdm.auto import tqdm
from rembg import remove, new_session
import sys
import os
import numpy as np
from typing import Optional, Tuple, Union

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from constants import BASE_DIR

class AdvancedBackgroundRemover:
    """Advanced background removal with multiple model options and fine-tuning capabilities"""
    
    def __init__(self, model_name: str = 'u2net_human_seg'):
        """
        Initialize with specific model
        
        Available models:
        - 'u2net': General purpose
        - 'u2net_human_seg': Optimized for human segmentation
        - 'u2netp': Lighter version of u2net
        - 'silueta': Good for objects
        - 'isnet-general-use': Latest general model
        """
        self.model_name = model_name
        self.session = new_session(model_name)
    
    def remove_background(self, 
                         image: Image.Image,
                         alpha_matting: bool = False,
                         alpha_matting_foreground_threshold: int = 270,
                         alpha_matting_background_threshold: int = 10,
                         alpha_matting_erode_size: int = 10) -> Image.Image:
        """
        Remove background with advanced options
        
        Args:
            image: Input PIL Image
            alpha_matting: Enable alpha matting for better edges
            alpha_matting_foreground_threshold: Higher = more aggressive foreground
            alpha_matting_background_threshold: Lower = more aggressive background removal
            alpha_matting_erode_size: Edge refinement size
        """
        if alpha_matting:
            return remove(
                image, 
                session=self.session,
                alpha_matting=True,
                alpha_matting_foreground_threshold=alpha_matting_foreground_threshold,
                alpha_matting_background_threshold=alpha_matting_background_threshold,
                alpha_matting_erode_size=alpha_matting_erode_size
            )
        else:
            return remove(image, session=self.session)
    
    def get_mask_only(self, image: Image.Image) -> Image.Image:
        """Get only the mask without applying it"""
        # Remove background and extract alpha channel as mask
        result = self.remove_background(image)
        if result.mode == 'RGBA':
            # Extract alpha channel as mask
            mask = result.split()[-1]  # Get alpha channel
            return mask
        else:
            # Convert to grayscale if no alpha
            return result.convert('L')
    
    def apply_custom_background(self, 
                              image: Image.Image, 
                              background: Union[Image.Image, tuple, str]) -> Image.Image:
        """
        Apply custom background
        
        Args:
            image: Input image
            background: Can be PIL Image, RGB tuple (255,255,255), or color name 'white'
        """
        # Remove background first
        foreground = self.remove_background(image)
        
        # Create background
        if isinstance(background, Image.Image):
            bg = background.resize(foreground.size)
        elif isinstance(background, tuple):
            bg = Image.new('RGB', foreground.size, background)
        elif isinstance(background, str):
            bg = Image.new('RGB', foreground.size, background)
        else:
            raise ValueError("Background must be PIL Image, RGB tuple, or color string")
        
        # Composite
        if foreground.mode == 'RGBA':
            result = Image.alpha_composite(bg.convert('RGBA'), foreground)
            return result.convert('RGB')
        else:
            return foreground
    
    def refine_edges(self, image: Image.Image, blur_radius: float = 1.0) -> Image.Image:
        """Refine edges of the segmented image"""
        from PIL import ImageFilter
        
        result = self.remove_background(image)
        if result.mode == 'RGBA':
            # Blur only the alpha channel for softer edges
            r, g, b, a = result.split()
            a_blurred = a.filter(ImageFilter.GaussianBlur(radius=blur_radius))
            return Image.merge('RGBA', (r, g, b, a_blurred))
        return result

def remove_bg_folder_images_advanced(
    input_dir: Path, 
    output_dir: Path, 
    model_name: str = 'u2net_human_seg',
    image_resize: Optional[Tuple[int, int]] = None,
    alpha_matting: bool = False,
    custom_background: Optional[Union[str, tuple]] = None,
    save_mask: bool = False,
    refine_edges: bool = False
):
    """
    Advanced background removal for folder of images
    
    Args:
        input_dir: Input directory containing images
        output_dir: Output directory for processed images
        model_name: rembg model to use
        image_resize: Resize tuple (width, height) or None
        alpha_matting: Enable alpha matting for better edges
        custom_background: Apply custom background (color tuple or string)
        save_mask: Save mask files separately
        refine_edges: Apply edge refinement
    """
    output_dir.mkdir(exist_ok=True, parents=True)
    
    if save_mask:
        mask_dir = output_dir / "masks"
        mask_dir.mkdir(exist_ok=True, parents=True)
    
    # Initialize remover
    remover = AdvancedBackgroundRemover(model_name)
    
    print(f"Using model: {model_name}")
    print(f"Alpha matting: {alpha_matting}")
    print(f"Custom background: {custom_background}")
    
    for image_path in tqdm(input_dir.glob("*.[jp][pn]*g"), desc="Processing images"):
        try:
            # Read and preprocess image
            image = Image.open(image_path).convert("RGB")
            
            if image_resize:
                image = image.resize(image_resize, Image.Resampling.LANCZOS)
            
            # Remove background
            if refine_edges:
                output_image = remover.refine_edges(image)
            else:
                output_image = remover.remove_background(
                    image, 
                    alpha_matting=alpha_matting
                )
            
            # Apply custom background if specified
            if custom_background:
                output_image = remover.apply_custom_background(image, custom_background)
            
            # Save main image
            output_path = output_dir / (image_path.stem + ".png")
            output_image.save(output_path)
            
            # Save mask if requested
            if save_mask:
                mask = remover.get_mask_only(image)
                mask_path = mask_dir / (image_path.stem + "_mask.png")
                mask.save(mask_path)
                
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            continue

def compare_models(image_path: Path, output_dir: Path):
    """Compare different rembg models on the same image"""
    models = [
        'u2net',
        'u2net_human_seg', 
        'u2netp',
        'silueta',
        'isnet-general-use'
    ]
    
    output_dir.mkdir(exist_ok=True, parents=True)
    image = Image.open(image_path).convert("RGB")
    
    for model in models:
        try:
            print(f"Testing {model}...")
            remover = AdvancedBackgroundRemover(model)
            result = remover.remove_background(image)
            result.save(output_dir / f"{image_path.stem}_{model}.png")
        except Exception as e:
            print(f"Error with {model}: {e}")

def batch_process_with_different_settings(input_dir: Path, base_output_dir: Path):
    """Process images with different settings for comparison"""
    
    setting = {
            'model_name': 'u2net_human_seg',
            'alpha_matting': True,
            'refine_edges': True
        }
        
    remove_bg_folder_images_advanced(
        input_dir=input_dir,
        output_dir=base_output_dir,
        **setting
    )

if __name__ == "__main__":
    input_path = BASE_DIR / "datasets/celebrities/mo_salah/cropped"
    output_path = BASE_DIR / "datasets/celebrities/mo_salah/cleaned"
    
    # Basic usage with u2net_human_seg
    remove_bg_folder_images_advanced(
        input_dir=input_path,
        output_dir=output_path,
        model_name='u2net_human_seg',
        alpha_matting=True,  # Better edge quality
        save_mask=False,      # Save masks separately
        refine_edges=True    # Softer edges
    )
    
    # Compare different models (uncomment to test)
    # sample_image = next(input_path.glob("*.[jp][pn]*g"))
    # compare_models(sample_image, BASE_DIR / "model_comparison")
    
    # Batch process with different settings (uncomment to test)
    # batch_process_with_different_settings(input_path, BASE_DIR / "batch_comparison")