from PIL import Image
from pathlib import Path
from tqdm.auto import tqdm
from retinaface import RetinaFace

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from constants import BASE_DIR

def extract_faces_from_folder(input_dir: Path, output_dir: Path, align=True, expand_face_area=30):
    """
    Args:
        - input_dir (Path): Input Directory contain images of (jpg, png, jpeg)
        - output_dir (Path): Output Directory will contain openpose images
        - image_resize (Tuple | None): if none no resize happen
        - align (boolean): Align face or note (default True)
        - expand_face_area (int): (default 30)
    """
    output_dir.mkdir(exist_ok=True, parents=True)
    paths = list(input_dir.glob("*.[jp][pn]*g"))
    for image_path in tqdm(paths): 
        faces = RetinaFace.extract_faces(image_path, align=align, expand_face_area=expand_face_area)
        for i, face in enumerate(faces):
            if face.shape[0] >= 256 or face.shape[1] >= 256:
                Image.fromarray(face).save(output_dir /  f"{image_path.stem}_{i}.png")

if __name__ == "__main__":
    extract_faces_from_folder(
        BASE_DIR / "datasets/celebrities/meh/raw",
        BASE_DIR / "datasets/celebrities/meh/filtered_faces",
    )