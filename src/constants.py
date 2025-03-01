from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent 
DATA_PATH = BASE_DIR / "datasets/fashion"
MODELS_SAVE_PATH = BASE_DIR / "models"

TRAIN_PATH = DATA_PATH / "train_lst_256_png/train_lst_256_png"    
TRAIN_POSES_PATH = DATA_PATH / "train_lst_256_png/poses"    
TRAIN_JSON_PATH = DATA_PATH / "train_data.json"

TEST_PATH = DATA_PATH / "test_lst_256_png/test_lst_256_png"    
TEST_JSON_PATH = DATA_PATH / "train_data.json"

