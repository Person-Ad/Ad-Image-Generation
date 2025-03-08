from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent 
DATA_PATH = BASE_DIR / "datasets/fashion"
MODELS_SAVE_PATH = BASE_DIR / "models"

# TRAIN_PATH = DATA_PATH / "train_lst_256_png/train_lst_256_png"    
TRAIN_PATH = DATA_PATH / "3430_train_sample"    
TRAIN_POSES_PATH = DATA_PATH / "3430_train_sample/poses"    
TRAIN_JSON_PATH = DATA_PATH / "train_data_3430.json"

TEST_PATH = DATA_PATH / "test_lst_256_png/test_lst_256_png"    
TEST_JSON_PATH = DATA_PATH / "test_data.json"
