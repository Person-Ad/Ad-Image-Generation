from pathlib import Path
DATA_PATH = Path("./datasets/fashion", absolute=True)
MODELS_SAVE_PATH = Path("./models", absolute=True)

TRAIN_PATH = DATA_PATH / "train_lst_256_png/train_lst_256_png"    
TRAIN_POSES_PATH = DATA_PATH / "train_lst_256_png/poses"    
TRAIN_JSON_PATH = DATA_PATH / "train_data.json"

TEST_PATH = DATA_PATH / "test_lst_256_png/test_lst_256_png"    
TEST_JSON_PATH = DATA_PATH / "train_data.json"

