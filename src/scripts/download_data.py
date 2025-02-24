import os
import sys
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fastcore.script import call_parse, Param
from utils import download_gfile, unzip_file

from loguru import logger

file_ids = {
    "all_data_png": "1-2KyPSiJpjCZ83xa4x66Y0Bs-oEqpHzb",                
    "normalized_pose_txt": "1MLr39aglmIzkF5POBBU9b6GiOLduOQoi",         
    "train_lst_256_png": "1RXFza68D7A0McfTd7iogDuFuD7XKMVEN",
    "test_lst_256_png": "17P_G-hiqnWY-gmSHcDhxExk2Iki1iUwk",
    "train_lst_512_png": "144IJdAeczlKJvwMfMN9R-q0x_-NTFELz",
    "test_lst_512_png": "1okF3Xo7-DLoV4czyBEFcEDJgOR65W61t",
    "test_data.json": "1AYheq5Xi1DwAJHnTNeLmTYeXNZDuF7lN",
    "train_data.json": "1YSJFvHissnsOdBPPRCZH8ELHAVFdRC44",
}

# Destination folder
DATA_DIR = Path("datasets/fashion")
DATA_DIR.mkdir(parents=True, exist_ok=True)

parser_help_msg = """
list of files to download (or 'all')\n
Available files: "all_data_png, normalized_pose_txt, train_lst_256_png, test_lst_256_png, train_lst_512_png, test_lst_512_png, test_data.json, train_data.json"
"""

@call_parse
def main(
    files: Param(parser_help_msg, str) = 'all'
):
    """Download specified files from Google Drive."""
    # Download all files if not specified
    files_to_download = file_ids.keys() if files == 'all' else map(str.strip, files.split(','))
    files_to_download = list(files_to_download)
        
    logger.info(f"Downloading files:  {files_to_download}")
    
    for file_key in files_to_download:
        # Download file if it exists in the file_ids dictionary
        if file_key in file_ids:
            file_name = f"{file_key}.zip" if not file_key.endswith("json") else file_key
            destination = DATA_DIR / file_name

            download_gfile(file_ids[file_key], destination)
            
            if destination.suffix == ".zip":
                unzip_file(destination, DATA_DIR)
        else:
            logger.warning(f"Unknown file key: {file_key}")
