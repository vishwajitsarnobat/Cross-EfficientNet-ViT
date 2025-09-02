# evaluate.py
import os
import argparse
import json
import yaml
import torch
import numpy as np
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
from termcolor import cprint
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Import from our generalized project files
import config
from preprocessing.detect_faces import run_face_detection
from preprocessing.extract_crops import run_crop_extraction
from deepfakes_dataset import DeepFakesDataset
from cross_efficient_vit import CrossEfficientViT
from utils import aggregate_predictions, get_media_paths_and_labels

def create_visual_report_video(video_path, frame_preds, final_verdict, confidence, fps=30.0, use_time_axis=False):
    """
    Generates a PNG report for a VIDEO, containing the verdict, confidence,
    and a gradient-filled probability graph over time/frames.
    """
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    output_filename = os.path.join(config.results_path, f"{base_name}_prediction_report.png")
    
    frame_preds = np.array(frame_preds)
    num_frames = len(frame_preds)
    fake_threshold = 0.5

    # --- Setup X-Axis ---
    if use_time_axis and fps > 0:
        x_axis = np.arange(num_frames) / fps
        x_label = "Time (seconds)"
    else:
        x_axis = np.arange(num_frames)
        x_label = "Frame Number"

    # --- Plotting ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax_verdict, ax_plot) = plt.subplots(
        2, 1, figsize=(16, 9), dpi=120, 
        gridspec_kw={'height_ratios': [1, 5]}
    )
    fig.patch.set_facecolor('#f8f9fa')

    # --- Top Panel: Verdict and Confidence ---
    ax_verdict.set_facecolor('#f8f9fa')
    ax_verdict.axis('off')
    verdict_color = '#d9534f' if final_verdict == "FAKE" else '#5cb85c'
    ax_verdict.text(0.5, 0.65, f"VERDICT: {final_verdict}", ha='center', va='center', fontsize=32, fontweight='bold', color=verdict_color)
    ax_verdict.text(0.5, 0.25, f"Confidence: {confidence:.2%}", ha='center', va='center', fontsize=20, color='#343a40')

    # --- Bottom Panel: The Graph ---
    ax_plot.set_facecolor('#ffffff')
    ax_plot.plot(x_axis, frame_preds, color='#007bff', linewidth=2.5, label='Fake Probability')
    ax_plot.axhline(y=fake_threshold, color='#343a40', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Fake Threshold ({fake_threshold})')
    ax_plot.fill_between(x_axis, fake_threshold, frame_preds, where=frame_preds >= fake_threshold, facecolor='#d9534f', interpolate=True, alpha=0.4)
    ax_plot.fill_between(x_axis, fake_threshold, frame_preds, where=frame_preds < fake_threshold, facecolor='#5cb85c', interpolate=True, alpha=0.4)
    ax_plot.set_title('Frame-by-Frame Fake Probability Analysis', fontsize=18, fontweight='bold', pad=20)
    ax_plot.set_xlabel(x_label, fontsize=14)
    ax_plot.set_ylabel('Probability Score', fontsize=14)
    ax_plot.set_ylim(0, 1)
    if num_frames > 1:
        ax_plot.set_xlim(0, x_axis[-1])
    ax_plot.tick_params(axis='both', which='major', labelsize=12)
    ax_plot.spines['top'].set_visible(False)
    ax_plot.spines['right'].set_visible(False)
    ax_plot.legend(fontsize=12)
    
    plt.tight_layout(pad=3.0)
    plt.savefig(output_filename)
    plt.close()
    return output_filename

def save_json_report(media_path, verdict, confidence, crop_preds, total_frames, fps=0):
    """Saves a detailed JSON report for either a video or an image."""
    base_name = os.path.splitext(os.path.basename(media_path))[0]
    output_filename = os.path.join(config.results_path, f"{base_name}_prediction_report.json")
    duration = total_frames / fps if fps > 0 else 0

    report = {
        'media_information': {
            'filename': os.path.basename(media_path),
            'path': media_path,
            'media_type': 'video' if fps > 0 else 'image',
            'total_frames_or_images': total_frames,
            'faces_detected': len(crop_preds)
        },
        'prediction_summary': {
            'verdict': verdict,
            'confidence': round(confidence, 4)
        },
        'per_crop_results': {
            'fake_probabilities': [round(float(p), 4) for p in crop_preds]
        }
    }
    if duration > 0:
        report['media_information']['duration_seconds'] = round(duration, 2)

    with open(output_filename, 'w') as f:
        json.dump(report, f, indent=4)
    return output_filename

def predict_single_file(file_path, model, transform):
    """Runs the full prediction pipeline on a single media file."""
    is_video = file_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))
    input_type = "Video" if is_video else "Image"
    cprint(f"-> Analyzing {input_type}: {os.path.basename(file_path)}", "yellow")

    run_face_detection([file_path])
    run_crop_extraction([file_path])

    media_id = os.path.splitext(os.path.basename(file_path))[0]
    crops_dir = os.path.join(config.output_path, "crops", media_id)
    if not os.path.exists(crops_dir) or not os.listdir(crops_dir):
        cprint(f"   - Warning: No faces were detected. Skipping.", 'yellow')
        return None, None

    crops = []
    for crop_file in sorted(os.listdir(crops_dir)):
        image = cv2.imread(os.path.join(crops_dir, crop_file))
        if image is None: continue
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        crops.append(transform(image=image_rgb)['image'])
    
    if not crops:
        cprint(f"   - Error: Failed to load valid crops from '{crops_dir}'.", 'red')
        return None, None
        
    crops_tensor = torch.stack(crops).to(config.device)

    crop_preds = []
    with torch.no_grad():
        for i in range(0, len(crops_tensor), config.test_batch_size):
            batch = crops_tensor[i : i + config.test_batch_size]
            pred = torch.sigmoid(model(batch))
            crop_preds.extend(pred.cpu().numpy().flatten().tolist())
    
    final_score = aggregate_predictions(crop_preds)
    final_verdict = "FAKE" if final_score > 0.5 else "REAL"
    return final_verdict, (crop_preds, final_score, is_video)

def evaluate_directory(dir_path, model, transform):
    """Evaluates all media in a directory and produces a final report."""
    cprint(f"\n--- Running Evaluation on Test Directory: {dir_path} ---", "cyan", attrs=['bold'])
    
    # Use config paths directly for test data
    media_with_labels = get_media_paths_and_labels(config.real_data_paths, config.fake_data_paths)
    
    if not media_with_labels:
        cprint("Error: No media files found in the test directories specified in config.py.", "red")
        return

    all_true_labels = []
    all_pred_labels = []

    for file_path, label in tqdm(media_with_labels, desc="Evaluating Test Set"):
        true_label_str = "REAL" if label == 0.0 else "FAKE"
        predicted_verdict, _ = predict_single_file(file_path, model, transform)
        
        if predicted_verdict is None: continue

        all_true_labels.append(true_label_str)
        all_pred_labels.append(predicted_verdict)

    print("\n\n--- TEST SET EVALUATION COMPLETE ---")
    report = classification_report(all_true_labels, all_pred_labels, target_names=['FAKE', 'REAL'], zero_division=0)
    print(report)

    cm = confusion_matrix(all_true_labels, all_pred_labels, labels=['REAL', 'FAKE'])
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['REAL', 'FAKE'], yticklabels=['REAL', 'FAKE'])
    plt.title('Test Set Confusion Matrix')
    plt.xlabel('Predicted'); plt.ylabel('Actual')
    save_file = os.path.join(config.results_path, "test_set_confusion_matrix.png")
    plt.savefig(save_file)
    plt.close()
    cprint(f"✓ Test set confusion matrix saved to: {save_file}", 'cyan')

def main():
    parser = argparse.ArgumentParser(description="Evaluate deepfakes on a single file or a test directory.")
    parser.add_argument("input_path", type=str, help="Path to the media file OR the root test directory.")
    parser.add_argument("--model_path", type=str, default=config.test_model_path, help="Path to the trained model checkpoint.")
    parser.add_argument("--time_axis", action="store_true", help="For single video predictions, use time for the x-axis.")
    opt = parser.parse_args()

    if not os.path.exists(opt.input_path):
        cprint(f"Error: Input path not found at '{opt.input_path}'", 'red')
        return

    os.makedirs(config.results_path, exist_ok=True)
    with open(config.architecture_config_path, 'r') as ymlfile:
        arch_config = yaml.safe_load(ymlfile)
    image_size = arch_config['model']['image-size']
    transform = DeepFakesDataset(frame_label_list=[], image_size=image_size, mode='validation').create_val_transform(image_size)

    model = CrossEfficientViT(config=arch_config)
    if not os.path.exists(opt.model_path):
        cprint(f"Error: Model file not found at '{opt.model_path}'", 'red')
        return
    model.load_state_dict(torch.load(opt.model_path, map_location=config.device))
    model.eval().to(config.device)
    cprint(f"Model loaded from {opt.model_path}", "green")

    if os.path.isfile(opt.input_path):
        verdict, results = predict_single_file(opt.input_path, model, transform)
        if verdict:
            crop_preds, final_score, is_video = results
            confidence = final_score if verdict == "FAKE" else 1 - final_score
            verdict_color = 'red' if verdict == "FAKE" else 'green'
            cprint(f"\n--- Prediction Summary ---", attrs=['bold'])
            cprint(f"Verdict:   {verdict}", verdict_color, attrs=['bold'])
            cprint(f"Confidence: {confidence:.2%}", verdict_color)
            cprint("------------------------\n")
            
            total_frames = 1
            fps = 0
            if is_video:
                cap = cv2.VideoCapture(opt.input_path)
                fps = cap.get(cv2.CAP_PROP_FPS)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
            
            json_path = save_json_report(opt.input_path, verdict, confidence, crop_preds, total_frames, fps)
            cprint(f"✓ JSON data saved to:   {json_path}", 'cyan')

            if is_video:
                report_path = create_visual_report_video(opt.input_path, crop_preds, verdict, confidence, fps, opt.time_axis)
                cprint(f"✓ Visual report saved to: {report_path}", 'cyan')

    elif os.path.isdir(opt.input_path):
        evaluate_directory(opt.input_path, model, transform)
    else:
        cprint(f"Error: Input path '{opt.input_path}' is not a valid file or directory.", 'red')

if __name__ == "__main__":
    main()