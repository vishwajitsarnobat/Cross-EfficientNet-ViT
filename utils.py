# utils.py
import os
import cv2
import torch
import numpy as np
from statistics import mean
import json
from glob import glob
from pathlib import Path
import random
import config # Import the new config file

def get_media_paths_and_labels(real_dirs, fake_dirs):
    """
    Gets all media file paths from specified directories and their subdirectories,
    and assigns labels. Uses the supported_formats from the config file.
    Label 0 for REAL, 1 for FAKE.
    """
    media_paths = []
    
    # Add real media (label 0)
    for dir_path in real_dirs:
        for root, _, files in os.walk(dir_path):
            for file in files:
                if file.lower().endswith(config.supported_formats):
                    media_paths.append((os.path.join(root, file), 0.0))
    
    # Add fake media (label 1)
    for dir_path in fake_dirs:
        for root, _, files in os.walk(dir_path):
            for file in files:
                if file.lower().endswith(config.supported_formats):
                    media_paths.append((os.path.join(root, file), 1.0))
                    
    return media_paths

def get_all_media_paths(real_dirs, fake_dirs):
    """
    Scans specified directories and all their subdirectories to find all
    media file paths based on the formats defined in the config.
    """
    media_paths = []
    
    for dir_path in real_dirs + fake_dirs:
        for root, _, files in os.walk(dir_path):
            for file in files:
                if file.lower().endswith(config.supported_formats):
                    media_paths.append(os.path.join(root, file))
    return media_paths

def custom_round(values):
    """Rounds predictions to 0 or 1 based on a 0.5 threshold."""
    return (np.asarray(values) > 0.5).astype(int)

def aggregate_predictions(preds):
    """
    Aggregates frame/crop predictions for a single media file.
    If any single prediction has a high fake probability, classify as fake.
    Otherwise, return the mean probability. This works for both single-crop
    images and multi-frame videos.
    """
    if not preds:
        return 0.0
        
    for pred_value in preds:
        if pred_value > 0.65: # Suspicion threshold
            return pred_value
            
    return mean(preds)

def shuffle_dataset(dataset):
    """Shuffles a list in-place with a fixed seed for reproducibility."""
    random.seed(42)
    random.shuffle(dataset)
    return dataset

def get_n_params(model):
    """Calculates the number of parameters in a PyTorch model."""
    return sum(p.numel() for p in model.parameters())

def check_correct(preds, labels):
    """
    Checks the number of correct predictions in a batch and returns the predictions tensor.
    """
    preds_rounded = (torch.sigmoid(preds).cpu() > 0.5).int()
    correct = (preds_rounded == labels.cpu()).sum().item()
    return correct, preds_rounded, None

def resize(image, image_size):
    """A simple cv2 resize wrapper with error handling."""
    try:
        return cv2.resize(image, dsize=(image_size, image_size))
    except Exception as e:
        print(f"Could not resize image: {e}")
        return []

# --- Note: The following functions are for parsing datasets with metadata.json files. ---
# --- They are not used in our current workflow but are included for completeness. ---

def get_original_video_paths(root_dir, basename=False):
    originals = set()
    originals_v = set()
    for json_path in glob(os.path.join(root_dir, "*/metadata.json")):
        dir = Path(json_path).parent
        with open(json_path, "r") as f:
            metadata = json.load(f)
        for k, v in metadata.items():
            original = v.get("original", None)
            if v["label"] == "REAL":
                original = k
                originals_v.add(original)
                originals.add(os.path.join(dir, original))
    originals = list(originals)
    originals_v = list(originals_v)

    return originals_v if basename else originals

def get_original_with_fakes(root_dir):
    pairs = []
    for json_path in glob(os.path.join(root_dir, "*/metadata.json")):
        with open(json_path, "r") as f:
            metadata = json.load(f)
        for k, v in metadata.items():
            original = v.get("original", None)
            if v["label"] == "FAKE":
                pairs.append((original[:-4], k[:-4] ))

    return pairs

def get_originals_and_fakes(root_dir):
    originals = []
    fakes = []
    for json_path in glob(os.path.join(root_dir, "*/metadata.json")):
        with open(json_path, "r") as f:
            metadata = json.load(f)
        for k, v in metadata.items():
            if v["label"] == "FAKE":
                fakes.append(k[:-4])
            else:
                originals.append(k[:-4])

    return originals, fakes