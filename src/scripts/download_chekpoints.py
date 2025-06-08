import os
import sys
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fastcore.script import call_parse, Param
from utils import download_gfile, unzip_file

from loguru import logger

file_ids = {
    "s2_512.pt": "1JhWeScr9bQtoQmB503VDyomaDmxBHail",                
    "s3-512.pt": "11JZXfYVlgLFqmE8jCbLWjQWO7LrwYq-I"
}

def download_checkpoint(file_name: str, output_dir: Path = Path('models/pcdm')):
    """Download a checkpoint file from Google Drive."""
    file_id = file_ids.get(file_name)
    
    if not file_id:
        raise ValueError(f"File ID for {file_name} not found in file_ids dictionary.")
    
    destination = output_dir / file_name
    download_gfile(file_id, destination)
    
    if destination.suffix == ".zip":
        unzip_file(destination, output_dir / file_name.replace('.zip', ''))
    logger.success(f"Downloaded {file_name} to {destination}")
    
@call_parse
def main(
    file_name: str = 's2_512.pt', # Name of the checkpoint file to download
    output_dir: str = 'models/pcdm' # Directory to save the downloaded checkpoint
):
    """Download specified checkpoint file from Google Drive."""
    # Destination folder
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Downloading checkpoint file: {file_name}")
    
    try:
        download_checkpoint(file_name, output_dir)
    except ValueError as e:
        logger.error(e)