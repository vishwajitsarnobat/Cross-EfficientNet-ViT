# train.py
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import os
import collections
from tqdm import tqdm
from multiprocessing import Pool
import yaml
import json # Import the json library
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from termcolor import cprint

# Import from our generalized project files
import config
from preprocessing.detect_faces import run_face_detection
from preprocessing.extract_crops import run_crop_extraction
from deepfakes_dataset import DeepFakesDataset
from cross_efficient_vit import CrossEfficientViT
from torch.optim import lr_scheduler
from utils import get_media_paths_and_labels, get_all_media_paths, check_correct, get_n_params

MIN_TRAIN_CROPS = 100
MIN_VAL_CROPS = 20

def get_crop_paths(media_path_label_tuple):
    """Worker function to find all crop image paths for a given media file."""
    media_path, label = media_path_label_tuple
    media_id = os.path.splitext(os.path.basename(media_path))[0]
    crops_dir = os.path.join(config.output_path, "crops", media_id)
    if not os.path.exists(crops_dir):
        return []
    return [(os.path.join(crops_dir, f), label) for f in os.listdir(crops_dir)]

# --- MODIFIED: Metrics plotting function now takes the current epoch ---
def plot_and_save_metrics(history, save_path):
    """Saves plots for training/validation loss and accuracy and updates history JSON."""
    # 1. Save Plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle('Training Metrics')
    
    epochs = range(1, len(history['train_loss']) + 1)

    ax1.plot(epochs, history['train_loss'], 'bo-', label='Train Loss')
    ax1.plot(epochs, history['val_loss'], 'ro-', label='Validation Loss')
    ax1.set_title('Loss vs. Epochs'); ax1.set_xlabel('Epoch'); ax1.legend()

    ax2.plot(epochs, history['train_acc'], 'bo-', label='Train Accuracy')
    ax2.plot(epochs, history['val_acc'], 'ro-', label='Validation Accuracy')
    ax2.set_title('Accuracy vs. Epochs'); ax2.set_xlabel('Epoch'); ax2.legend()
    
    plt.savefig(os.path.join(save_path, "training_plots.png"))
    plt.close()

    # 2. Save History to JSON
    # Convert defaultdict to a regular dict for clean JSON output
    history_dict = dict(history)
    with open(os.path.join(save_path, "training_history.json"), 'w') as f:
        json.dump(history_dict, f, indent=4)

    print(f"Updated training plots and history JSON in '{save_path}'")

def plot_confusion_matrix(labels, preds, save_path, epoch):
    """Saves a confusion matrix plot for the best validation epoch."""
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
    cprint(f"--- Scope: Training --- [MODE: {config.MODE.upper()}]", "yellow")
    train_val_media_paths = get_all_media_paths(config.real_data_paths, config.fake_data_paths)

    if not train_val_media_paths:
        cprint("Error: No media files found in specified data directories.", "red")
        return

    if config.run_preprocessing:
        run_face_detection(train_val_media_paths)
        run_crop_extraction(train_val_media_paths)
    else:
        print("Skipping preprocessing as per config.")

    os.makedirs(config.models_path, exist_ok=True)
    os.makedirs(config.results_path, exist_ok=True)
    
    with open(config.architecture_config_path, 'r') as ymlfile:
        arch_config = yaml.safe_load(ymlfile)
    image_size = arch_config['model']['image-size']

    print("Gathering crop paths for training and validation sets...")
    train_val_media_with_labels = get_media_paths_and_labels(config.real_data_paths, config.fake_data_paths)
    
    if len(train_val_media_with_labels) < 2:
        cprint("Error: Not enough media files to create a train/validation split.", "red")
        return

    train_media, val_media = train_test_split(
        train_val_media_with_labels, 
        test_size=config.validation_split, 
        random_state=42, 
        stratify=[lbl for _, lbl in train_val_media_with_labels]
    )

    with Pool(processes=config.train_workers) as p:
        train_crops = list(tqdm(p.imap(get_crop_paths, train_media), total=len(train_media), desc="Loading train crop paths"))
        val_crops = list(tqdm(p.imap(get_crop_paths, val_media), total=len(val_media), desc="Loading val crop paths"))

    train_crop_paths = [item for sublist in train_crops for item in sublist]
    val_crop_paths = [item for sublist in val_crops for item in sublist]
    print(f"Found {len(train_crop_paths)} training crops and {len(val_crop_paths)} validation crops.")

    if len(train_crop_paths) < MIN_TRAIN_CROPS or len(val_crop_paths) < MIN_VAL_CROPS:
        cprint("Training cannot proceed due to insufficient data after preprocessing.", "red", attrs=['bold'])
        if len(train_crop_paths) < MIN_TRAIN_CROPS:
            cprint(f" - Found {len(train_crop_paths)} training crops, but require at least {MIN_TRAIN_CROPS}.", "red")
        if len(val_crop_paths) < MIN_VAL_CROPS:
            cprint(f" - Found {len(val_crop_paths)} validation crops, but require at least {MIN_VAL_CROPS}.", "red")
        cprint("Action: Run `uv run audit_preprocessing.py` to diagnose your dataset.", "yellow")
        return

    train_counters = collections.Counter(label for _, label in train_crop_paths)
    class_weights = train_counters[0] / train_counters[1] if train_counters[1] > 0 else 1.0

    train_dataset = DeepFakesDataset(train_crop_paths, image_size, mode='train')
    val_dataset = DeepFakesDataset(val_crop_paths, image_size, mode='validation')

    dl = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.train_workers)
    val_dl = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.train_workers)

    model = CrossEfficientViT(config=arch_config).to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=arch_config['training']['weight-decay'])
    scheduler = lr_scheduler.StepLR(optimizer, step_size=arch_config['training']['step-size'], gamma=arch_config['training']['gamma'])
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([class_weights]).to(config.device))

    model_to_load = config.resume_model_path
    if model_to_load is None:
        potential_resume_path = os.path.join(config.models_path, "best_model.pth")
        if os.path.exists(potential_resume_path):
            cprint(f"Found existing 'best_model.pth'. Resuming training.", "yellow")
            model_to_load = potential_resume_path
    
    if model_to_load and os.path.exists(model_to_load):
        cprint(f"Loading weights from: {model_to_load}", "green")
        model.load_state_dict(torch.load(model_to_load, map_location=config.device))
    else:
        cprint("Starting training from scratch.", "green")
    
    print(f"Model Parameters: {get_n_params(model)}")

    best_val_loss = float('inf')
    epochs_no_improve = 0
    history = collections.defaultdict(list)

    for epoch in range(config.num_epochs):
        current_epoch = epoch + 1
        print(f"\n--- EPOCH {current_epoch}/{config.num_epochs} ---")
        
        # --- Training Phase ---
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
        
        avg_train_loss = total_loss / total_train_samples if total_train_samples > 0 else 0
        avg_train_acc = train_correct / total_train_samples if total_train_samples > 0 else 0
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(avg_train_acc)
        scheduler.step()

        # --- Validation Phase ---
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

        avg_val_loss = total_val_loss / total_val_samples if total_val_samples > 0 else 0
        avg_val_acc = val_correct / total_val_samples if total_val_samples > 0 else 0
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(avg_val_acc)
        print(f"Epoch {current_epoch} Summary | Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f} | Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_acc:.4f}")
        
        plot_and_save_metrics(history, config.results_path)
        
        print("\n--- Validation Report ---")
        print(classification_report(all_labels, all_preds, target_names=['Real', 'Fake'], zero_division=0))
        cm = confusion_matrix(all_labels, all_preds)
        print("Confusion Matrix (Real: 0, Fake: 1):\n", cm)
        print("-------------------------\n")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), os.path.join(config.models_path, "best_model.pth"))
            cprint(f"Validation loss improved to {avg_val_loss:.4f}. Model saved to 'best_model.pth'.", "green")
            plot_confusion_matrix(all_labels, all_preds, config.results_path, epoch=current_epoch)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= config.early_stopping_patience:
                print(f"Early stopping! Validation loss has not improved for {epochs_no_improve} epochs.")
                break
    
    print("\n--- Training Finished ---")
    torch.save(model.state_dict(), os.path.join(config.models_path, "final_model.pth"))
    print("Final model saved to 'final_model.pth'.")

if __name__ == "__main__":
    main()