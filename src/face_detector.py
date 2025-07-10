"""
Face Detection Module using YOLOv8
Handles real-time face detection in video frames
"""

import cv2
import numpy as np
import logging
from typing import List, Tuple, Optional
import os
import sys

# Add parent directory to path to import mock models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    logging.warning("YOLOv8 not available, using mock face detector")

from mock_ml_models import MockYOLOFaceDetector


class FaceDetector:
    """YOLOv8-based face detector"""
    
    def __init__(self, config: dict):
        """
        Initialize the face detector
        
        Args:
            config: Configuration dictionary containing detection parameters
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Detection parameters
        self.confidence_threshold = config['detection']['confidence_threshold']
        self.model_name = config['detection']['model_name']
        self.detection_size = tuple(config['detection']['face_detection_size'])
        
        # Initialize model (YOLO or mock)
        self.model = None
        self._load_model()
        
        self.logger.info(f"Face detector initialized with confidence threshold: {self.confidence_threshold}")
    
    def _load_model(self):
        """Load the face detection model"""
        try:
            if YOLO_AVAILABLE:
                # Try to load pre-trained face detection model
                if os.path.exists(self.model_name):
                    self.model = YOLO(self.model_name)
                else:
                    # Use YOLOv8n and adapt for face detection
                    self.model = YOLO('yolov8n.pt')
                self.logger.info(f"YOLO model loaded: {self.model_name}")
            else:
                # Use mock face detector
                self.model = MockYOLOFaceDetector(self.confidence_threshold)
                self.logger.info("Mock face detector loaded for demo")
            
        except Exception as e:
            self.logger.warning(f"Failed to load YOLO model: {str(e)}, using mock detector")
            self.model = MockYOLOFaceDetector(self.confidence_threshold)
    
    def detect_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """
        Detect faces in a frame
        
        Args:
            frame: Input frame as numpy array
            
        Returns:
            List of face detections as (x1, y1, x2, y2, confidence)
        """
        if self.model is None:
            self.logger.error("Model not loaded")
            return []
        
        try:
            # Check if using mock model
            if isinstance(self.model, MockYOLOFaceDetector):
                return self.model.detect_faces(frame)
            
            # Original YOLO detection code
            # Resize frame for detection if needed
            original_height, original_width = frame.shape[:2]
            
            # Run inference
            results = self.model(frame, conf=self.confidence_threshold, verbose=False)
            
            faces = []
            
            for result in results:
                if result.boxes is not None:
                    boxes = result.boxes
                    
                    for box in boxes:
                        # Get bounding box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        class_id = box.cls[0].cpu().numpy()
                        
                        # Filter for person class (class 0 in COCO) or if using face-specific model
                        # For general YOLO, we'll detect persons and then crop head region
                        if class_id == 0:  # person class
                            # Estimate face region (top 20% of person bounding box)
                            face_height = (y2 - y1) * 0.3
                            face_y2 = y1 + face_height
                            
                            # Ensure face region is valid
                            if face_height > self.config['system']['min_face_size']:
                                faces.append((
                                    int(x1), int(y1), 
                                    int(x2), int(face_y2), 
                                    float(confidence)
                                ))
                        
                        # If using a face-specific model, all detections are faces
                        elif 'face' in self.model_name.lower():
                            faces.append((
                                int(x1), int(y1), 
                                int(x2), int(y2), 
                                float(confidence)
                            ))
            
            self.logger.debug(f"Detected {len(faces)} faces")
            return faces
            
        except Exception as e:
            self.logger.error(f"Error in face detection: {str(e)}")
            return []
    
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess frame for detection
        
        Args:
            frame: Input frame
            
        Returns:
            Preprocessed frame
        """
        # Resize if frame is too large
        height, width = frame.shape[:2]
        max_width = self.config['video']['resize_width']
        max_height = self.config['video']['resize_height']
        
        if width > max_width or height > max_height:
            # Calculate scaling factor
            scale = min(max_width / width, max_height / height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            
            frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
            self.logger.debug(f"Resized frame from {width}x{height} to {new_width}x{new_height}")
        
        return frame
    
    def crop_face(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """
        Crop face from frame with padding
        
        Args:
            frame: Input frame
            bbox: Bounding box (x1, y1, x2, y2)
            
        Returns:
            Cropped face image or None if invalid
        """
        x1, y1, x2, y2 = bbox
        padding = self.config['system']['face_crop_padding']
        
        # Add padding
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(frame.shape[1], x2 + padding)
        y2 = min(frame.shape[0], y2 + padding)
        
        # Validate crop dimensions
        if x2 <= x1 or y2 <= y1:
            self.logger.warning(f"Invalid face crop dimensions: {x1}, {y1}, {x2}, {y2}")
            return None
        
        # Check minimum face size
        face_width = x2 - x1
        face_height = y2 - y1
        min_size = self.config['system']['min_face_size']
        
        if face_width < min_size or face_height < min_size:
            self.logger.debug(f"Face too small: {face_width}x{face_height}")
            return None
        
        # Crop face
        face_crop = frame[y1:y2, x1:x2]
        
        if face_crop.size == 0:
            self.logger.warning("Empty face crop")
            return None
        
        return face_crop
    
    def get_face_center(self, bbox: Tuple[int, int, int, int]) -> Tuple[int, int]:
        """
        Get center point of face bounding box
        
        Args:
            bbox: Bounding box (x1, y1, x2, y2)
            
        Returns:
            Center coordinates (cx, cy)
        """
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        return cx, cy
    
    def visualize_detections(self, frame: np.ndarray, 
                           detections: List[Tuple[int, int, int, int, float]]) -> np.ndarray:
        """
        Draw face detection boxes on frame
        
        Args:
            frame: Input frame
            detections: List of face detections
            
        Returns:
            Frame with drawn detections
        """
        vis_frame = frame.copy()
        
        for detection in detections:
            x1, y1, x2, y2, confidence = detection
            
            # Draw bounding box
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw confidence score
            label = f'Face: {confidence:.2f}'
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(vis_frame, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), (0, 255, 0), -1)
            cv2.putText(vis_frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        return vis_frame
