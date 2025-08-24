# utils.py
import os
import cv2
import torch
import numpy as np
from statistics import mean

def get_video_paths_and_labels(real_dirs, fake_dirs):
    """
    Gets all video paths from specified directories and assigns labels.
    Label 0 for REAL, 1 for FAKE.
    """
    video_paths = []
    
    # Add real videos
    for dir_path in real_dirs:
        for root, _, files in os.walk(dir_path):
            for file in files:
                if file.lower().endswith(('.mp4', '.avi', '.mov')):
                    video_paths.append((os.path.join(root, file), 0.0)) # 0 for REAL
    
    # Add fake videos
    for dir_path in fake_dirs:
        for root, _, files in os.walk(dir_path):
            for file in files:
                if file.lower().endswith(('.mp4', '.avi', '.mov')):
                    video_paths.append((os.path.join(root, file), 1.0)) # 1 for FAKE
                    
    return video_paths

def custom_round(values):
    """Rounds predictions to 0 or 1 based on a 0.5 threshold."""
    return (np.asarray(values) > 0.5).astype(int)

def custom_video_round(preds):
    """
    Aggregates frame predictions for a single video.
    If any frame has a high fake probability, classify the video as fake.
    Otherwise, return the mean probability.
    """
    for pred_value in preds:
        if pred_value > 0.65: # A slightly higher threshold to be more certain
            return pred_value
    return mean(preds) if preds else 0.0

def shuffle_dataset(dataset):
    import random
    random.seed(42)
    random.shuffle(dataset)
    return dataset

def get_n_params(model):
    """Calculates the number of parameters in a model."""
    return sum(p.numel() for p in model.parameters())

def check_correct(preds, labels):
    """Checks the number of correct predictions in a batch."""
    preds_rounded = (torch.sigmoid(preds).cpu() > 0.5).int()
    correct = (preds_rounded == labels.cpu()).sum().item()
    positive_class = preds_rounded.sum().item()
    negative_class = len(preds_rounded) - positive_class
    return correct, positive_class, negative_class