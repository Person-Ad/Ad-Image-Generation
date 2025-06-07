# Ad Image Generation Project

This project focuses on fine-tuning Stable Diffusion models for inpainting and pose transfer using connected source and target poses. It also includes a data downloader to fetch required datasets from Google Drive.

## 📦 Features
- Data downloader script for Google Drive files.
- Image generation using diffusion models.
- Pose transfer and inpainting.

## 🔧 Requirements
This project uses Python 3.10+ and manages dependencies with [Conda](https://docs.conda.io/).

### Dependencies
- `numpy`
- `Pillow`
- `matplotlib`
- `torch`
- `torchvision`
- `transformers`
- `diffusers`
- `gdown`
- `fastcore`

## 🚀 Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Person-Ad/Ad-Image-Generation.git
   cd Ad-Image-Generation
   ```

2. **Create a Conda environment:**
   ```bash
   conda create --name ad_image_gen python=3.12
   conda activate ad_image_gen
   ```

3. **Install dependencies:**
   ```bash
   conda install numpy pillow matplotlib pytorch torchvision -c pytorch
   pip install -U tqdm accelerate transformers diffusers gdown fastcore loguru controlnet_aux open_clip_torch xformers
   # pip install basicsr facexlib gfpgan -- face restorer (deprected) -- 
   ```

## 📥 Downloading Data

Run the following command to download all datasets:
```bash
python ./scripts/download_data.py --files all
```

Download specific files:
```bash
python ./scripts/download_data.py --files "train_lst_256_png, test_lst_256_png, train_data.json, test_data.json"
```

## 🏃‍♂️ Usage Example
[] Todo

## 🔍 Project Structure
```
ad-image-generation/
│
├── datasets/
│   └── fashion/  # Downloaded datasets
│
├── scripts/
│   └── download_data.py  # Script for downloading datasets
│
├── src/
│   ├── models/  # Model definitions
│   ├── utils.py  # Utility functions
│
├── README.md
└── environment.yml  # Conda environment configuration
```

## ✅ License
This project is licensed under the Apache License.

