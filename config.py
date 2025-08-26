# config.py
import os
import torch

# --- GENERAL SETTINGS ---
# Set to True to run face detection and cropping. Set to False if data is already processed.
# This will intelligently skip any work that has already been completed within the correct scope (train/test).
run_preprocessing = True
# Determine the device to be used for training and testing.
device = "cuda" if torch.cuda.is_available() else "cpu"


# --- PATHS ---

# Directory where preprocessed data (bounding boxes, crops) will be saved.
output_path = "processed_data"

# Directories containing your custom training and validation videos.
real_videos_paths = ["/home/vishwajit/Workspace/NTRO/Celeb-DF-V2/train-real"]
fake_videos_paths = ["/home/vishwajit/Workspace/NTRO/Celeb-DF-V2/train-fake"]

# Directories containing your custom test videos.
test_real_videos_paths = ["/home/vishwajit/Workspace/NTRO/Celeb-DF-V2/test-real"]
test_fake_videos_paths = ["/home/vishwajit/Workspace/NTRO/Celeb-DF-V2/test-fake"]

# Path to the model architecture configuration file.
architecture_config_path = "configs/architecture.yaml"

# Directory where trained models and results (plots, etc.) will be saved.
models_path = "models"
results_path = "results"


# --- PREPROCESSING SETTINGS ---
# Number of CPU processes for preprocessing.
preprocessing_workers = os.cpu_count() - 2 if os.cpu_count() > 2 else 1


# --- TRAINING SETTINGS ---
# Path to a pre-trained model for fine-tuning. Set to None if training from scratch.
resume_model_path = "models/cross_efficient_vit.pth"
# Percentage of the data to be used for validation.
validation_split = 0.15
# Number of CPU workers for loading data.
train_workers = 8
# Learning rate for the optimizer.
learning_rate = 1e-4
# Number of training epochs.
num_epochs = 2
# Epochs to wait for validation loss improvement before stopping early.
early_stopping_patience = 5
# Batch size for training and validation.
batch_size = 16


# --- TESTING SETTINGS ---
# Path to the trained model to be used for testing.
test_model_path = "models/best_model.pth"
# Batch size for prediction on test set.
test_batch_size = 16