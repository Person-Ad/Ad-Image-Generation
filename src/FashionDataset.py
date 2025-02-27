"""
This Module Will assume you downloaded data using `scripts/download_data.py` and have configured any paths in `constants.py`
Also assume you run precompute_poses.py to generate the poses for the dataset || downloaded it using `scripts/download_data.py`
"""

import os
import sys
import json
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from PIL import Image
from fastcore.script import call_parse, Param
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from constants import TRAIN_PATH, TRAIN_JSON_PATH, TRAIN_POSES_PATH

class FashionDataset(Dataset):
    def __init__(self, 
                 root_dir: Path = TRAIN_PATH, 
                 poses_dir: Path = TRAIN_POSES_PATH,
                 info_json_path: Path = TRAIN_JSON_PATH,
                 image_size = (256, 352),
                 small_dataset_size = 1024):
        """ Fashion Dataset 
        Args:
            root_dir (Path): Path to the dataset images
            poses_dir (Path): Path to the dataset poses
            info_json_path (Path): Path to the dataset info json
            image_size (tuple): Image size to resize to
        """
        super(FashionDataset, self).__init__()
        self.small_dataset_size = small_dataset_size
        self.image_size = image_size
        
        assert root_dir.exists(), "No Dataset Exists"
        assert poses_dir.exists(), "No Poses Exists"
        assert info_json_path.exists(), "No Dataset Info Exists"
        
        self.root_dir = root_dir
        self.poses_dir = poses_dir
        self.transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        
        with open(info_json_path, 'r') as file:
            self.data = json.load(file)[:self.small_dataset_size] # TODO: till now -- remove it later

    def __len__(self):
        return len(self.data) # train: 99517

    def __getitem__(self, idx):
        src_path, tar_path = self.data[idx]['source_image'], self.data[idx]['target_image']
        src_path, tar_path = src_path.split(".")[0] + ".png", tar_path.split(".")[0] + ".png"
        # read src/tar images
        src_img = Image.open(self.root_dir / src_path).convert("RGB")  # Convert to 3 channels
        tar_img = Image.open(self.root_dir / tar_path).convert("RGB")  # Convert to 3 channels
        # read src/tar poses
        src_pose = Image.open(self.poses_dir / src_path).convert("RGB")  # Convert to 3 channels
        tar_pose = Image.open(self.poses_dir / tar_path).convert("RGB")  # Convert to 3 channels
        # concatentate inputs
        src_mask = self.concatenate_src_mask(src_img)
        st_pose = self.concatenate_src_tar(src_pose, tar_pose)
        src_tar = self.concatenate_src_tar(src_img, tar_img)
        
        if self.transform:
            src_mask  = self.transform(src_mask)
            st_pose   = self.transform(st_pose)
            src_tar   = self.transform(src_tar)

        return {"src_mask": src_mask, "st_pose": st_pose, "src_tar": src_tar} 
    
    def concatenate_src_mask(self, src_img):
        """ concatenate source with mask """
        black_image = Image.new("RGB", src_img.size, (0, 0, 0)).resize(src_img.size, Image.BICUBIC)
        
        src_mask = Image.new("RGB", (src_img.width * 2, src_img.height))
        src_mask.paste(src_img, (0, 0))
        src_mask.paste(black_image, (src_img.width, 0))

        return src_mask

    def concatenate_src_tar(self, src_img, tar_img):
        """ concatenate source with target """
        src_tar = Image.new("RGB", (src_img.width * 2, src_img.height))
        src_tar.paste(src_img, (0, 0))
        src_tar.paste(tar_img, (src_img.width, 0))

        return src_tar
    
    
        

if __name__ == "__main__":
    dataset = FashionDataset()
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    for idx, data in enumerate(dataloader):
        print(data)
        break