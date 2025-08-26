# train.py
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
import os
import collections
from tqdm import tqdm
from functools import partial
import cv2
import math
import yaml
from multiprocessing import Pool
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

import config
from preprocessing.detect_faces import run_face_detection
from preprocessing.extract_crops import run_crop_extraction

# --- CORRECTED: Importing YOUR dataset class from the correct file ---
from deepfakes_dataset import DeepFakesDataset

from cross_efficient_vit import CrossEfficientViT
from torch.optim import lr_scheduler
from utils import get_video_paths_and_labels, get_all_video_paths, check_correct, get_n_params

def get_frame_paths(video_path_label_tuple):
    video_path, label = video_path_label_tuple
    video_id = os.path.splitext(os.path.basename(video_path))[0]
    crops_dir = os.path.join(config.output_path, "crops", video_id)
    if not os.path.exists(crops_dir):
        return []
    return [(os.path.join(crops_dir, f), label) for f in os.listdir(crops_dir)]

def plot_metrics(history, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle('Training Metrics')
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Validation Loss')
    ax1.set_title('Loss vs. Epochs'); ax1.set_xlabel('Epoch'); ax1.legend()
    ax2.plot(history['train_acc'], label='Train Accuracy')
    ax2.plot(history['val_acc'], label='Validation Accuracy')
    ax2.set_title('Accuracy vs. Epochs'); ax2.set_xlabel('Epoch'); ax2.legend()
    plt.savefig(os.path.join(save_path, "training_plots.png"))
    plt.close()
    print(f"Training plots saved to {os.path.join(save_path, 'training_plots.png')}")

def plot_confusion_matrix(labels, preds, save_path, epoch):
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
    plt.title(f'Best Validation Confusion Matrix (Epoch {epoch})')
    plt.xlabel('Predicted'); plt.ylabel('Actual')
    save_file = os.path.join(save_path, "best_validation_confusion_matrix.png")
    plt.savefig(save_file)
    plt.close()
    print(f"Saved best confusion matrix plot to {save_file}")

def main():
    print("--- Scope: Training ---")
    train_val_video_paths = get_all_video_paths(config.real_videos_paths, config.fake_videos_paths)

    if config.run_preprocessing:
        run_face_detection(train_val_video_paths)
        run_crop_extraction(train_val_video_paths)
    else:
        print("Skipping preprocessing as per config.")

    os.makedirs(config.models_path, exist_ok=True)
    os.makedirs(config.results_path, exist_ok=True)
    
    with open(config.architecture_config_path, 'r') as ymlfile:
        arch_config = yaml.safe_load(ymlfile)
    image_size = arch_config['model']['image-size']

    print("Gathering frame paths for training and validation sets...")
    train_val_videos_with_labels = get_video_paths_and_labels(config.real_videos_paths, config.fake_videos_paths)
    train_videos, val_videos = train_test_split(train_val_videos_with_labels, test_size=config.validation_split, random_state=42, stratify=[lbl for _, lbl in train_val_videos_with_labels])

    with Pool(processes=config.train_workers) as p:
        train_frames = list(tqdm(p.imap(get_frame_paths, train_videos), total=len(train_videos), desc="Loading train frame paths"))
        val_frames = list(tqdm(p.imap(get_frame_paths, val_videos), total=len(val_videos), desc="Loading val frame paths"))

    train_frame_paths = [item for sublist in train_frames for item in sublist]
    val_frame_paths = [item for sublist in val_frames for item in sublist]
    
    print(f"Found {len(train_frame_paths)} training frames and {len(val_frame_paths)} validation frames.")

    train_counters = collections.Counter(label for _, label in train_frame_paths)
    class_weights = train_counters[0] / train_counters[1] if train_counters[1] > 0 else 1.0

    # --- CORRECTED: Using the imported DeepFakesDataset class ---
    train_dataset = DeepFakesDataset(train_frame_paths, image_size, mode='train')
    val_dataset = DeepFakesDataset(val_frame_paths, image_size, mode='validation')

    dl = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.train_workers)
    val_dl = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.train_workers)

    model = CrossEfficientViT(config=arch_config).to(config.device)
    optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate, weight_decay=arch_config['training']['weight-decay'])
    scheduler = lr_scheduler.StepLR(optimizer, step_size=arch_config['training']['step-size'], gamma=arch_config['training']['gamma'])
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([class_weights]).to(config.device))

    if config.resume_model_path and os.path.exists(config.resume_model_path):
        print(f"Resuming training by loading model from: {config.resume_model_path}")
        model.load_state_dict(torch.load(config.resume_model_path, map_location=config.device))
    
    print(f"Model Parameters: {get_n_params(model)}")

    best_val_loss = math.inf
    epochs_no_improve = 0
    history = collections.defaultdict(list)

    for epoch in range(config.num_epochs):
        current_epoch = epoch + 1
        print(f"\n--- EPOCH {current_epoch}/{config.num_epochs} ---")
        model.train()
        total_loss, train_correct, total_train_samples = 0, 0, 0
        for images, labels in tqdm(dl, desc='TRAIN'):
            images, labels = images.to(config.device), labels.to(config.device).unsqueeze(1)
            optimizer.zero_grad()
            y_pred = model(images)
            loss = loss_fn(y_pred, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * images.size(0)
            corrects, _, _ = check_correct(y_pred, labels)
            train_correct += corrects
            total_train_samples += len(labels)
        
        avg_train_loss = total_loss / total_train_samples
        avg_train_acc = train_correct / total_train_samples
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(avg_train_acc)
        scheduler.step()

        model.eval()
        total_val_loss, val_correct, total_val_samples = 0, 0, 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for images, labels in tqdm(val_dl, desc='VALIDATION'):
                images, labels = images.to(config.device), labels.to(config.device).unsqueeze(1)
                y_pred = model(images)
                loss = loss_fn(y_pred, labels)
                total_val_loss += loss.item() * images.size(0)
                
                corrects, preds_tensor, _ = check_correct(y_pred, labels)
                val_correct += corrects
                all_preds.extend(preds_tensor.cpu().numpy())
                all_labels.extend(labels.cpu().numpy().flatten())
                total_val_samples += len(labels)

        avg_val_loss = total_val_loss / total_val_samples
        avg_val_acc = val_correct / total_val_samples
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(avg_val_acc)
        print(f"Epoch {current_epoch} Summary | Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f} | Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_acc:.4f}")
        
        print("\n--- Validation Report ---")
        print(classification_report(all_labels, all_preds, target_names=['Real', 'Fake'], zero_division=0))
        cm = confusion_matrix(all_labels, all_preds)
        print("Confusion Matrix (Real: 0, Fake: 1):")
        print(cm)
        print("-------------------------\n")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), os.path.join(config.models_path, "best_model.pth"))
            print(f"Validation loss improved to {avg_val_loss:.4f}. Model saved to 'best_model.pth'.")
            plot_confusion_matrix(all_labels, all_preds, config.results_path, epoch=current_epoch)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= config.early_stopping_patience:
                print(f"Early stopping! Validation loss has not improved for {epochs_no_improve} epochs.")
                break
    
    print("\n--- Training Finished ---")
    plot_metrics(history, config.results_path)
    torch.save(model.state_dict(), os.path.join(config.models_path, "final_model.pth"))
    print("Final model saved to 'final_model.pth'.")

if __name__ == "__main__":
    main()