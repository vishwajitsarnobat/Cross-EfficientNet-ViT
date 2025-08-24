# predict.py
import torch
import cv2
import numpy as np
import os
import argparse
import yaml
from PIL import Image
from tqdm import tqdm

# Import necessary components from your project
from face_detector import FacenetDetector
from cross_efficient_vit import CrossEfficientViT
from test import create_val_transform # Re-use the transform from test.py
from utils import custom_video_round
from config import architecture_config_path, prediction_model_path

def predict_single_video(video_path, model, detector, image_size, batch_size=32):
    """
    Performs face detection and prediction on a single video.
    """
    # --- 1. Face Detection ---
    capture = cv2.VideoCapture(video_path)
    frames_num = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    
    all_frames = []
    for i in range(frames_num):
        success, frame = capture.read()
        if not success:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        # Downsize for faster detection
        frame = frame.resize(size=[s // 2 for s in frame.size])
        all_frames.append(frame)
    
    if not all_frames:
        print("Could not read any frames from the video.")
        return None, 0

    # Detect faces in batches
    face_cropper = create_val_transform(image_size)
    cropped_faces = []

    for i in tqdm(range(0, len(all_frames), batch_size), desc="Detecting faces"):
        batch_frames = all_frames[i : i + batch_size]
        # detector.detect returns bounding boxes and probabilities
        batch_boxes, _ = detector.detector.detect(batch_frames, landmarks=False)

        for j, boxes in enumerate(batch_boxes):
            if boxes is None:
                continue
            
            original_frame_pil = batch_frames[j]
            original_frame_cv = cv2.cvtColor(np.array(original_frame_pil), cv2.COLOR_RGB2BGR)

            for box in boxes:
                # Rescale box coordinates back to original frame size
                box = [int(coord * 2) for coord in box]
                xmin, ymin, xmax, ymax = box
                
                # Crop face from the original resolution frame
                crop = original_frame_cv[ymin:ymax, xmin:xmax]
                if crop.size > 0:
                    transformed_crop = face_cropper(image=crop)['image']
                    cropped_faces.append(transformed_crop)
    
    num_faces = len(cropped_faces)
    if num_faces == 0:
        print("No faces were detected in the video.")
        return None, 0

    # --- 2. Prediction ---
    model.eval()
    predictions = []
    with torch.no_grad():
        for i in tqdm(range(0, len(cropped_faces), batch_size), desc="Predicting"):
            batch = cropped_faces[i : i + batch_size]
            batch = torch.from_numpy(np.array(batch)).cuda().float()
            
            # Ensure batch is in CHW format
            if batch.shape[-1] == 3:
                batch = batch.permute(0, 3, 1, 2)
            
            pred = model(batch)
            predictions.extend(torch.sigmoid(pred).cpu().numpy().flatten())

    return predictions, num_faces


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict if a video is a deepfake.")
    parser.add_argument('--video_path', required=True, type=str, help='Path to the video file to predict.')
    opt = parser.parse_args()

    # --- Load Model and Detector ---
    print("Loading model and face detector...")
    # Load architecture config
    with open(architecture_config_path, 'r') as ymlfile:
        config = yaml.safe_load(ymlfile)
    
    # Load the trained model
    model = CrossEfficientViT(config=config)
    model.load_state_dict(torch.load(prediction_model_path))
    model = model.cuda()
    
    # Load the face detector
    face_detector = FacenetDetector(device="cuda:0")

    # --- Perform Prediction ---
    predictions, face_count = predict_single_video(
        video_path=opt.video_path,
        model=model,
        detector=face_detector,
        image_size=config['model']['image-size']
    )

    # --- Display Results ---
    print("\n" + "="*30)
    print("      PREDICTION RESULTS")
    print("="*30)
    print(f"Video Path: {opt.video_path}")
    print(f"Total Faces Detected: {face_count}")

    if predictions:
        # Aggregate frame predictions to get a single video score
        final_score = custom_video_round(predictions)
        verdict = "FAKE" if final_score > 0.5 else "REAL"
        
        print(f"\nFinal Verdict: {verdict}")
        print(f"Confidence Score (0=REAL, 1=FAKE): {final_score:.4f}")
    else:
        print("\nCould not make a prediction.")
    print("="*30)