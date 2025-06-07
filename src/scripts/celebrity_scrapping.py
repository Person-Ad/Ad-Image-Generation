from fastcore.script import *
from pydantic import BaseModel, Field, field_validator
from icrawler.builtin import GoogleImageCrawler
import cv2
from pathlib import Path
from loguru import logger
from ultralytics import YOLO

class ScraperConfig(BaseModel):
    celebrity_name: str = Field(..., description="Celebrity name to search")
    num_images: int = Field(..., gt=0, le=1000, description="Number of images to download (max 1000)")
    output_dir: str = Field("datasets/celebrities", description="Directory to save the cropped images")
    center_crop: bool = Field(False, description="Detect person in photo and center crop on that person")
    downloader_threads: int = Field(4, description="Number of threads used to download images")
    
    @field_validator('celebrity_name', mode='after')
    @classmethod
    def no_empty_name(cls, v):
        if not v.strip():
            raise ValueError("Celebrity name cannot be empty")
        return v.strip()
    

def download_images(celebrity_name: str, num_images: int, output_folder: str, downloader_threads: int):
    save_dir = Path(output_folder) / celebrity_name.replace(" ", "_")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    crawler = GoogleImageCrawler(storage={'root_dir': save_dir}, downloader_threads=downloader_threads)
    crawler.crawl(keyword=celebrity_name, max_num=num_images)
    
    return save_dir

    

def center_crop_person(img_path: Path, out_dir: Path, person_detector: YOLO):
    image = cv2.imread(str(img_path))
    if image is None:
        return
    
    results = person_detector(str(img_path))
    if not results or len(results) == 0:
        return
    
    h, w, _ = image.shape
    detections = results[0].boxes  # first (and only) image
    if detections is None or detections.cls is None:
        return

    # Filter only person class (YOLO COCO class ID 0 is person)
    person_boxes = [
        (box.xyxy[0], float(box.conf)) for box, cls in zip(detections, detections.cls)
        if int(cls) == 0
    ]
    
    if not person_boxes:
        return

    # Get box with highest confidence
    best_box, _ = max(person_boxes, key=lambda b: b[1])
    x1, y1, x2, y2 = map(int, best_box)

    # box_w, box_h = x2 - x1, y2 - y1
    padding_x, padding_y = 20, 20

    x_min = max(x1 - padding_x, 0)
    x_max = min(x2 + padding_x, w)
    y_min = max(y1 - padding_y, 0)
    y_max = min(y2 + padding_y, h)

    cropped = image[y_min:y_max, x_min:x_max]
    output_path = out_dir / img_path.name
    cv2.imwrite(str(output_path), cropped)



@call_parse
def main(
    celebrity_name: str,    # Celebrity name to search
    num_images: int,        # Number of images to download (max 1000)
    output_dir: str = "datasets/celebrities", # Directory to save the cropped images | default "datasets/celebrities"
    center_crop: bool = False,                # Detect person in photo and center crop on that person | default false
    downloader_threads: int = 4,              # Number of threads used to download images | default 4
):
    # 1. validate configurations
    cfg = ScraperConfig(celebrity_name=celebrity_name, num_images=num_images, output_dir=output_dir, downloader_threads=downloader_threads, center_crop=center_crop)
    # 2. download images
    logger.info(f"Downloading {cfg.num_images} images of {cfg.celebrity_name}...")
    download_path = download_images(cfg.celebrity_name, cfg.num_images, cfg.output_dir, cfg.downloader_threads)
    # 3. detect and crop
    if cfg.center_crop:
        logger.info(f"Cropping images to center person...")
        model = YOLO("yolo11n.pt")
        for img_file in Path(download_path).glob("*.[jp][pn]*g"):
            center_crop_person(img_file, download_path, model)
    
    logger.success(f"Images saved in: {download_path}")