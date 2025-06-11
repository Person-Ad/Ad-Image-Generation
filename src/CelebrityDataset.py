import torch
import random
from pathlib import Path
from loguru import logger
from itertools import permutations
from typing import Dict, Optional, Tuple, List
from torch.utils.data import Dataset, DataLoader

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from constants import BASE_DIR
from scripts.celebrity_scrapping  import scrap_celebrity, ScraperConfig
from scripts.extract_poses import extract_poses
from scripts.resize_folder_images import resize_folder_images
from scripts.adv_remove_bg_folder_images import remove_bg_folder_images_advanced
from pcdms.InpaintingStage import InpaintingProcessor
from utils import show_images

def compute_cosine_similarity(emb1: torch.Tensor, emb2: torch.Tensor) -> float:
    """Compute cosine similarity between two embeddings."""
    if emb1.shape != emb2.shape:
        logger.warning("Embedding dimensions do not match")
        return 0.0
    return torch.cosine_similarity(emb1, emb2, dim=0).item()


def celebrity_collate_fn(batch: List[Dict], image_size: Tuple[int, int] = (512, 512)) -> Dict:
    """Collate function for batching dataset samples."""
    batch_size = len(batch)
    keys = [
        "source_image",
        "target_image",
        "vae_source_mask_image",
        "source_target_pose",
        "source_target_image",
    ]
    tensors = {
        key: torch.stack([sample[key] for sample in batch])
        .to(memory_format=torch.contiguous_format)
        .float()
        for key in keys
    }
    
    mask1 = torch.ones(batch_size, 1, image_size[0] // 8, image_size[0] // 8)
    mask0 = torch.zeros(batch_size, 1, image_size[0] // 8, image_size[0] // 8)
    mask = torch.cat([mask1, mask0], dim=3)
    
    return {
        "s_img_path": [sample["s_img_path"] for sample in batch],
        "t_img_path": [sample["t_img_path"] for sample in batch],
        "mask": mask,
        **tensors,
    }


class CelebrityDataset(Dataset):
    def __init__(
        self,
        root_dir: str = "datasets/celebrities",
        celebrity_name: str = "mohamed_salah",
        image_size: Tuple[int, int] = (512, 512),
        num_images: int = 35,
        seed: int = 42,
        embedding_dict: Optional[Dict[Path, torch.Tensor]] = None,
        similarity_threshold: float = 0.7,
        max_samples: Optional[int] = 6000,
        split: str = "train",
        split_ratio: float = 0.8,
    ):
        """Dataset for celebrity image pairs with optional embedding-based alignment.

        Args:
            root_dir: Base directory for the dataset.
            celebrity_name: Name of the celebrity to process.
            image_size: Target size for resized images.
            num_images: Number of images to scrape if not already present.
            seed: Random seed for reproducibility.
            embedding_dict: Dictionary mapping image paths to embeddings for aligned pairs.
            similarity_threshold: Minimum cosine similarity for embedding-based pairs.
            max_samples: Maximum number of source-target pairs.
            split: Dataset split ("train", "val", or "test").
            split_ratio: Ratio of training samples to total samples.
        """
        super().__init__()
        random.seed(seed)

        self.image_size = image_size
        self.celebrity_name = celebrity_name
        self.directory = BASE_DIR / root_dir / celebrity_name
        self.embedding_dict = embedding_dict
        self.similarity_threshold = similarity_threshold
        self.split = split
        self.processor = InpaintingProcessor()

        # Initialize directories
        self.images_dir = self.directory / "cropped"
        self.cleaned_dir = self.directory / "cleaned"
        self.poses_dir = self.directory / "poses"

        # Prepare dataset
        self._prepare_dataset(num_images)
        self._generate_pairs(max_samples)
        self._apply_split(split, split_ratio)

    def _prepare_dataset(self, num_images: int) -> None:
        """Prepare dataset by scraping, cleaning, and extracting poses."""
        # Scrape images if cropped directory doesn't exist
        if not self.images_dir.exists() or not list(self.images_dir.glob("*.[jp][pn]*g")):
            logger.info(f"Scraping {self.celebrity_name} images to {self.images_dir}")
            scraper_config = ScraperConfig(
                celebrity_name=self.celebrity_name,
                num_images=num_images,
                output_dir=str(self.images_dir),
                center_crop=True,
            )
            scrap_celebrity(scraper_config)

        # Resize and remove background if cleaned directory doesn't exist
        if not self.cleaned_dir.exists() or not list(self.cleaned_dir.glob("*.[jp][pn]*g")):
            logger.info(f"Cleaning images to {self.cleaned_dir}")
            resize_folder_images(self.images_dir, self.cleaned_dir, self.image_size)
            remove_bg_folder_images_advanced(
                input_dir=self.cleaned_dir,
                output_dir=self.cleaned_dir,
                model_name="u2net_human_seg",
                alpha_matting=True,
                save_mask=False,
                refine_edges=True,
            )

        # Extract poses if poses directory doesn't exist
        if not self.poses_dir.exists() or not list(self.poses_dir.glob("*.[jp][pn]*g")):
            logger.info(f"Extracting poses to {self.poses_dir}")
            extract_poses(self.cleaned_dir, self.poses_dir, self.image_size)

    def _generate_pairs(self, max_samples: Optional[int]) -> None:
        """Generate source-target image pairs, optionally using embeddings."""
        image_paths = list(self.cleaned_dir.glob("*.[jp][pn]*g"))
        if len(image_paths) < 2:
            raise ValueError(f"Not enough images in {self.cleaned_dir} to form pairs")

        if self.embedding_dict:
            logger.info("Generating aligned pairs using embeddings")
            self.src_tar_pairs = []
            for src_path, tar_path in permutations(image_paths, 2):
                if (
                    src_path.name in self.embedding_dict
                    and tar_path.name in self.embedding_dict
                ):
                    similarity = compute_cosine_similarity(
                        self.embedding_dict[src_path.name].squeeze(0),
                        self.embedding_dict[tar_path.name].squeeze(0),
                    )
                    if similarity >= self.similarity_threshold:
                        self.src_tar_pairs.append((src_path.name, tar_path.name))
        else:
            logger.info("Generating random pairs")
            self.src_tar_pairs = list(permutations([p.name for p in image_paths], 2))

        random.shuffle(self.src_tar_pairs)
        if max_samples:
            self.src_tar_pairs = self.src_tar_pairs[:max_samples]

    def _apply_split(self, split: str, split_ratio: float) -> None:
        """Apply dataset split."""
        if split not in ["train", "val", "test"]:
            raise ValueError(f"Invalid split: {split}")

        if split == "train":
            split_index = int(len(self.src_tar_pairs) * split_ratio)
            self.src_tar_pairs = self.src_tar_pairs[:split_index]
        elif split == "val":
            split_index = int(len(self.src_tar_pairs) * split_ratio)
            self.src_tar_pairs = self.src_tar_pairs[split_index:]
        # Test split uses all pairs (no change needed)

        logger.info(f"Split '{split}' contains {len(self.src_tar_pairs)} pairs")

    def __len__(self) -> int:
        return len(self.src_tar_pairs)

    def __getitem__(self, idx: int) -> Dict:
        """Get a single dataset item."""
        src_name, tar_name = self.src_tar_pairs[idx]
        s_img_path = self.cleaned_dir / src_name
        t_img_path = self.cleaned_dir / tar_name
        s_pose_path = self.poses_dir / src_name
        t_pose_path = self.poses_dir / tar_name

        if self.split == "val":
            return {
                "s_img_path": s_img_path,
                "t_img_path": t_img_path,
                "s_pose_path": s_pose_path,
                "t_pose_path": t_pose_path,
                "image_size": self.image_size,
            }

        return {
            "s_img_path": s_img_path,
            "t_img_path": t_img_path,
            **self.processor.process_input(
                s_img_path,
                t_img_path,
                s_pose_path,
                t_pose_path,
                self.image_size,
            ),
        }


if __name__ == "__main__":
    # Example embedding dictionary (for testing)
    embedding_dict = torch.load(BASE_DIR / "datasets/celebrities/mo_salah/clip.pt")
    dataset = CelebrityDataset(
        celebrity_name="mo_salah",
        image_size=(1024, 1024),
        embedding_dict=embedding_dict,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        pin_memory=True,
        collate_fn=lambda x: celebrity_collate_fn(x, (1024, 1024)),
    )
    logger.info(f"Length of Dataset is {len(dataset)}")
    
    
    for idx, batch in enumerate(dataloader):
        # show_images(batch['source_target_image']).save(f'all_pairs_{idx}.jpg')
        logger.info(
            f"Batch shapes: source_image={batch['source_image'].shape}, "
            f"source_target_pose={batch['source_target_pose'].shape}, "
            f"source_target_image={batch['source_target_image'].shape}"
        )
        break