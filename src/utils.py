from pathlib import Path
import gdown
import zipfile

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
