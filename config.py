# config.py
import os
import torch

# -----------------------------------------------------------------------------
# CORE SETTINGS (USER-CONFIGURABLE)
# -----------------------------------------------------------------------------
# 1. Set the primary mode for the entire pipeline.
#    Options: "video" or "image"
MODE = "image"

# 2. Define the paths to your datasets.
real_videos_paths = [
    "data/datasets/FaceForensics++/original_sequences/youtube/c23/videos",
]
fake_videos_paths = [
    "data/datasets/FaceForensics++/manipulated_sequences/Deepfakes/c23/videos",
]
real_images_paths = [
    # "/mnt/data/Image_DS/test_real",
    "data/real",
]
fake_images_paths = [
    # "/mnt/data/Image_DS/test_fake",
    "data/fake",
]

# -----------------------------------------------------------------------------
# BASE PATHS & DYNAMIC CONFIGURATION (GENERALLY DO NOT EDIT)
# -----------------------------------------------------------------------------
BASE_OUTPUT_PATH = "processed_data"
MODELS_BASE_PATH = "models"
RESULTS_BASE_PATH = "results"
ARCHITECTURE_CONFIG_PATH = "configs/architecture.yaml"

if MODE == "video":
    real_data_paths = real_videos_paths
    fake_data_paths = fake_videos_paths
    supported_formats = ('.mp4', '.avi', '.mov', '.mkv')
elif MODE == "image":
    real_data_paths = real_images_paths
    fake_data_paths = fake_images_paths
    supported_formats = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')
else:
    raise ValueError(f"Invalid MODE '{MODE}' specified. Choose 'video' or 'image'.")

output_path = os.path.join(BASE_OUTPUT_PATH, MODE)
models_path = os.path.join(MODELS_BASE_PATH, MODE)
results_path = os.path.join(RESULTS_BASE_PATH, MODE)
test_model_path = os.path.join(models_path, "best_model.pth")
architecture_config_path = ARCHITECTURE_CONFIG_PATH

# -----------------------------------------------------------------------------
# TRAINING & PREPROCESSING PARAMETERS
# -----------------------------------------------------------------------------
# --- General ---
device = "cuda" if torch.cuda.is_available() else "cpu"
run_preprocessing = True

# --- Preprocessing ---
PREPROCESSING_RESIZE_FACTOR = 0.5
preprocessing_workers = os.cpu_count() or 4

# --- Training ---
num_epochs = 50
batch_size = 32
learning_rate = 0.0001  # A more standard learning rate for Adam optimizer
validation_split = 0.15
early_stopping_patience = 7

# --- MODIFIED: Model Resumption Logic ---
# To resume training from a specific checkpoint, set this path.
# For example: "models/image/best_model.pth"
# If set to None, the script will automatically check for a 'best_model.pth'
# in the current mode's model directory and offer to resume from there.
resume_model_path = None
train_workers = (os.cpu_count() - 4) or 4

# --- Evaluation / Testing ---
test_batch_size = 32

# -----------------------------------------------------------------------------
# CONFIGURATION SUMMARY
# -----------------------------------------------------------------------------
print(f"--- CONFIGURATION LOADED [MODE: {MODE.upper()}] ---")
print(f"Device: {device}")
print(f"Real Data Paths: {real_data_paths}")
print(f"Fake Data Paths: {fake_data_paths}")
print(f"Data Output Path (Crops/Boxes): {output_path}")
print(f"Models Path: {models_path}")
print(f"Results Path: {results_path}")
print("-" * 50)