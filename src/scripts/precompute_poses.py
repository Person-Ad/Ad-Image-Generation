import os
import sys
import json
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from concurrent.futures import ThreadPoolExecutor, as_completed

from PIL import Image
from tqdm.auto import tqdm

import torch
from controlnet_aux import OpenposeDetector
from constants import TRAIN_PATH, TRAIN_JSON_PATH, TRAIN_POSES_PATH


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
openpose = OpenposeDetector.from_pretrained("lllyasviel/ControlNet").to(device)

def process_pose_pair(src_img, tar_img):
    src_pose = openpose(src_img, detect_resolution=src_img.size[1]).resize(src_img.size, Image.BICUBIC)
    tar_pose = openpose(tar_img, detect_resolution=src_img.size[1]).resize(src_img.size, Image.BICUBIC)
    return (src_pose, tar_pose)

def generate_poses(image_pairs_list):
    poses = []
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_pose_pair, src, tar) for src, tar in image_pairs_list]
        for future in tqdm(as_completed(futures), total=len(futures)):
            poses.append(future.result())
    return poses

if __name__ == "__main__":

    assert TRAIN_PATH.exists(), "No Train Dataset"
    assert TRAIN_JSON_PATH.exists(), "No Train Info"
    TRAIN_POSES_PATH.mkdir(exist_ok=True, parents=True)
    
    with open(TRAIN_JSON_PATH, 'r') as file:
        data = json.load(file)[:5_000] # TODO: till now -- remove it later
        
    for item in tqdm(data):
        src_path, tar_path = item['source_image'], item['target_image']
        src_path, tar_path = src_path.split(".")[0] + ".png", tar_path.split(".")[0] + ".png"
        
        if not (TRAIN_POSES_PATH / src_path).exists():
            src_img = Image.open(TRAIN_PATH / src_path).convert("RGB")  # Convert to 3 channels
            src_pose = openpose(src_img, detect_resolution=src_img.size[1], include_body=True, include_face=True).resize(src_img.size, Image.BICUBIC)
            src_pose.save(TRAIN_POSES_PATH / src_path)
            
        if not (TRAIN_POSES_PATH / tar_path).exists():
            tar_img = Image.open(TRAIN_PATH / tar_path).convert("RGB")  # Convert to 3 channels
            tar_pose = openpose(tar_img, detect_resolution=tar_img.size[1], include_body=True, include_face=True).resize(tar_img.size, Image.BICUBIC)
            tar_pose.save(TRAIN_POSES_PATH / tar_path)
    print("Done!")