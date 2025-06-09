import torch
from PIL import Image
from pathlib import Path
from tqdm.auto import tqdm
from controlnet_aux import OpenposeDetector

def extract_poses(input_dir: Path, output_dir: Path, image_resize = (512, 512)):
    """
    Args:
        - input_dir (Path): Input Directory contain images of (jpg, png, jpeg)
        - output_dir (Path): Output Directory will contain openpose images
        - image_resize (Tuple | None): if none no resize happen
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    openpose = OpenposeDetector.from_pretrained("lllyasviel/ControlNet").to(device)
    
    output_dir.mkdir(exist_ok=True, parents=True)
    for image_path in tqdm(input_dir.glob("*.[jp][pn]*g")): 
        # 1. read image and preprocess it
        image = Image.open(image_path).convert("RGB")
        if image_resize:
            image = image.resize(image_resize, Image.BICUBIC)
        # 2. detect pose
        pose = openpose(image, detect_resolution=image.size[1], include_body=True, include_face=True)
        # 3. save image
        pose.save(output_dir / image_path.name)