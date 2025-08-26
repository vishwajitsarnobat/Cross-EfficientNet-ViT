# config.py
import os

# --- PATHS ---

# Directory where preprocessed data (bounding boxes, crops) will be saved.
output_path = "/mnt/data/Cross-EfficientNet-data/processed_data"

# Directories containing your custom training and validation videos.
# The training script will automatically split this data for validation.
real_videos_paths = ["/mnt/data/Cross-EfficientNet-data/train_real"]
fake_videos_paths = ["/mnt/data/Cross-EfficientNet-data/train_fake"]

# Directories containing your custom test videos.
test_real_videos_paths = ["/mnt/data/Cross-EfficientNet-data/test_real"]
test_fake_videos_paths = ["/mnt/data/Cross-EfficientNet-data/test_fake"]

# Path to the model architecture configuration file.
architecture_config_path = "configs/architecture.yaml"

# Directory where trained model checkpoints will be saved.
models_path = "models"

# Path to a pre-trained model for fine-tuning or resuming training.
# Set to an empty string "" if training from scratch.
# Example: "models/efficientnet_checkpoint30_All"
resume_model_path = "models/cross_efficient_vit.pth"

# --- TRAINING & VALIDATION SETTINGS ---

# Percentage of the data to be used for validation (e.g., 0.15 for 15%).
validation_split = 0.15

# Number of CPU workers for loading data.
workers = 8


# --- PREDICTION SETTINGS ---

# Path to the trained model to be used for single video prediction.
prediction_model_path = "models/cross_efficient_vit.pth"