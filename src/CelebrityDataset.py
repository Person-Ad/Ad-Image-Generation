import torch
import random
from PIL import Image
from pathlib import Path
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F
from loguru import logger
from itertools import permutations

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from constants import BASE_DIR
from scripts.celebrity_scrapping  import scrap_celebrity, ScraperConfig
from scripts.extract_poses import extract_poses
from scripts.resize_folder_images import resize_folder_images
from scripts.adv_remove_bg_folder_images import remove_bg_folder_images_advanced
from pcdms.InpaintingStage import InpaintingProcessor


def CelebrityCollateFn(batch, image_size=(512, 512)):
    batch_size = len(batch)
    # dimensions handling
    source_images = torch.stack([sample['source_image'] for sample in batch]).to(memory_format=torch.contiguous_format).float()
    target_image = torch.stack([sample['target_image'] for sample in batch]).to(memory_format=torch.contiguous_format).float()
    vae_source_mask_images = torch.stack([sample['vae_source_mask_image'] for sample in batch]).to(memory_format=torch.contiguous_format).float()
    source_target_poses = torch.stack([sample['source_target_pose'] for sample in batch]).to(memory_format=torch.contiguous_format).float()
    source_target_images = torch.stack([sample['source_target_image'] for sample in batch]).to(memory_format=torch.contiguous_format).float()

    mask1 = torch.ones((batch_size, 1, int(image_size[0] / 8), int(image_size[0] / 8)))
    mask0 = torch.zeros((batch_size, 1, int(image_size[0] / 8), int(image_size[0] / 8)))
    mask = torch.cat([mask1, mask0], dim=3)
    
    return {
        "s_img_path": batch['s_img_path'],
        "t_img_path": batch['t_img_path'],
        "source_image": source_images, 
        "target_image": target_image, 
        "vae_source_mask_image": vae_source_mask_images,
        "source_target_pose": source_target_poses,
        "source_target_image": source_target_images,
        "mask": mask
    }

class CelebrityDataset(Dataset):
    def __init__(self, 
                 root_dir: str = "datasets/celebrities",
                 celebrity_name: str = 'mohamed_salah',
                 image_resize=(512, 512),
                 num_images=35,
                 seed = 42,
                 max_samples = 6000,
                 split: str = "train",
                 split_ratio: float = 0.8):
        """ Celebrity Dataset 
        Args:
            root_dir (str): Path to the dataset celebrities
            celebrity_name (str): Name of Celebrity to Scrap
            image_resize (tuple): Image size to resize to
            num_images (int): Number of images to download if not exist 
            seed (int): Seed used to generate permutations of pairs of images
            max_samples (int| None): Max Number of training samples
            split (str | None): Dataset split (train, val, test). If None, all samples are returned.
            split_ratio (float): Ratio of training samples to total samples if train split is selected.
            If split is "val", the remaining samples are used for validation.
        """
        super(CelebrityDataset, self).__init__()
        random.seed(seed)

        self.processor = InpaintingProcessor()
        self.image_resize = image_resize

        self.directory = BASE_DIR / root_dir / celebrity_name
        self.images_directory = self.directory / "cropped"
        self.poses_directory = self.directory / "poses"
        self.cleaned_images_directory = self.directory / "cleaned"
        
        if not self.images_directory.exists():
            logger.info(f"Scrapping celebrity {celebrity_name} as Didn't Find {self.images_directory}")
            scrap_config = ScraperConfig(celebrity_name=celebrity_name,
                                         num_images=num_images,
                                         output_dir=str(self.images_directory),
                                         center_crop=True)
            scrap_celebrity(scrap_config)
            
        if not self.cleaned_images_directory.exists():
            logger.info(f"Cleaning images of celebrity {celebrity_name} as Didn't Find {self.cleaned_images_directory}")
            resize_folder_images(self.images_directory, self.cleaned_images_directory, image_resize)
            remove_bg_folder_images_advanced(
                input_dir=self.cleaned_images_directory,
                output_dir=self.cleaned_images_directory,
                model_name='u2net_human_seg',
                alpha_matting=True,  # Better edge quality
                save_mask=False,      # Save masks separately
                refine_edges=True    # Softer edges
            )
    
                
        if not self.poses_directory.exists():
            logger.info(f"Extracting poses of celebrity {celebrity_name} as Didn't Find {self.poses_directory}")
            extract_poses(self.cleaned_images_directory, self.poses_directory, image_resize)
        
        logger.info(f"Generating pairs of images")
        image_names = list(self.cleaned_images_directory.glob("*.[jp][pn]*g"))
        image_names = [s.name for s in image_names]
        
        self.src_tar_pairs = list(permutations(image_names, 2))
        random.shuffle(self.src_tar_pairs)
        if max_samples is not None:
            self.src_tar_pairs = self.src_tar_pairs[:max_samples]
        
        self.split = split
        if split == "train":
            split_index = int(len(self.src_tar_pairs) * split_ratio)
            self.src_tar_pairs = self.src_tar_pairs[:split_index]
        elif split == "val":
            split_index = int(len(self.src_tar_pairs) * split_ratio)
            self.src_tar_pairs = self.src_tar_pairs[split_index:]

    def __len__(self):
        return len(self.src_tar_pairs)
    
    def __getitem__(self, idx):
        src_tar_pair = self.src_tar_pairs[idx]
        s_img_path = self.cleaned_images_directory / src_tar_pair[0]
        t_img_path = self.cleaned_images_directory / src_tar_pair[1]
        s_pose_path = self.poses_directory / src_tar_pair[0]
        t_pose_path = self.poses_directory / src_tar_pair[1]
        
        if self.split == "val":
            return {
                "s_img_path": s_img_path,
                "t_img_path": t_img_path,
                "s_pose_path": s_pose_path,
                "t_pose_path": t_pose_path,
                "image_size": self.image_resize
            }
            
        return {
            "s_img_path": s_img_path,
            "t_img_path": t_img_path,
            **self.processor.process_input(s_img_path, 
                                            t_img_path, 
                                            s_pose_path, 
                                            t_pose_path,
                                            self.image_resize)}
        
if __name__ == "__main__":
    dataset = CelebrityDataset(celebrity_name="mo_salah", image_resize=(1024, 1024))
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, pin_memory=True, collate_fn=lambda x: CelebrityCollateFn(x, (1024, 1024)))
    
    for batch in dataloader:
        print(batch['source_image'].shape, batch['source_target_pose'].shape, batch['source_target_image'].shape)
        break
    