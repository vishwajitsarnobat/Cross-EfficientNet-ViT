# preprocessing/face_detector.py
import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import List

import cv2
cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)

import torch # Import torch to catch the specific error
from PIL import Image
from facenet_pytorch.models.mtcnn import MTCNN
from torch.utils.data import Dataset
import config

class VideoFaceDetector(ABC):
    def __init__(self, **kwargs) -> None:
        super().__init__()
    @property
    @abstractmethod
    def _batch_size(self) -> int:
        pass
    @abstractmethod
    def _detect_faces(self, frames) -> List:
        pass

class FacenetDetector(VideoFaceDetector):
    def __init__(self, device="cpu") -> None:
        super().__init__()
        self.detector = MTCNN(margin=0, thresholds=[0.85, 0.95, 0.95], device=device)

    # --- MODIFIED METHOD ---
    def _detect_faces(self, frames) -> List:
        """
        MODIFIED: Wrapped the detector call in a try-except block to gracefully
        handle a known edge case in facenet-pytorch where it crashes on empty batches.
        """
        try:
            batch_boxes, *_ = self.detector.detect(frames, landmarks=False)
            # Ensure the output is a list, even if no faces are found in any image
            if batch_boxes is None:
                return [None] * len(frames)
            return [b.tolist() if b is not None else None for b in batch_boxes]
        except RuntimeError as e:
            # If the specific torch.cat error occurs, it means no faces were found in the batch.
            # We return a list of Nones to match the batch size, which our pipeline handles correctly.
            if "torch.cat(): expected a non-empty list of Tensors" in str(e):
                return [None] * len(frames)
            else:
                # If it's a different runtime error, we should raise it to be aware of it.
                raise e
    # --- END MODIFICATION ---

    @property
    def _batch_size(self):
        return 32

class MediaDataset(Dataset):
    def __init__(self, media_paths: list) -> None:
        super().__init__()
        self.media_paths = media_paths
        self.resize_factor = config.PREPROCESSING_RESIZE_FACTOR

    def __getitem__(self, index: int):
        media_path = self.media_paths[index]
        frames = OrderedDict()
        
        if media_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            try:
                capture = cv2.VideoCapture(media_path)
                frames_num = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
                for i in range(frames_num):
                    capture.grab()
                    success, frame = capture.retrieve()
                    if not success: continue
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = Image.fromarray(frame)
                    if self.resize_factor != 1.0:
                        new_size = [int(s * self.resize_factor) for s in frame.size]
                        frame = frame.resize(size=new_size)
                    frames[i] = frame
            finally:
                if 'capture' in locals() and capture.isOpened():
                    capture.release()
        
        elif media_path.lower().endswith(config.supported_formats):
            try:
                frame = cv2.imread(media_path)
                if frame is not None and frame.shape[0] > 0 and frame.shape[1] > 0:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = Image.fromarray(frame)
                    if self.resize_factor != 1.0:
                        new_size = [int(s * self.resize_factor) for s in frame.size]
                        frame = frame.resize(size=new_size)
                    frames[0] = frame
            except Exception:
                pass
        
        return media_path, list(frames.keys()), list(frames.values())

    def __len__(self) -> int:
        return len(self.media_paths)