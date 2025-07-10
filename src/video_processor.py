"""
Video Processing Module
Main video processing pipeline that coordinates all components
"""

import cv2
import numpy as np
import time
import os
from datetime import datetime
from pathlib import Path
import threading
import queue
import logging
from typing import Optional, List, Tuple, Dict

from face_detector import FaceDetector
from face_recognizer import FaceRecognizer
from face_tracker import FaceTracker
from database_manager import DatabaseManager
from logger_manager import LoggerManager
import utils


class VideoProcessor:
    """Main video processing pipeline"""
    
    def __init__(self, video_source: str, config: dict, 
                 db_manager: DatabaseManager, logger_manager: LoggerManager):
        """
        Initialize video processor
        
        Args:
            video_source: Path to video file or RTSP stream
            config: Configuration dictionary
            db_manager: Database manager instance
            logger_manager: Logger manager instance
        """
        self.video_source = video_source
        self.config = config
        self.db_manager = db_manager
        self.logger_manager = logger_manager
        self.logger = logger_manager.get_logger('video_processor')
        
        # Initialize components
        self.face_detector = FaceDetector(config)
        self.face_recognizer = FaceRecognizer(config)
        self.face_tracker = FaceTracker(config)
        
        # Video capture
        self.cap = None
        self.frame_width = 0
        self.frame_height = 0
        self.fps = 0
        
        # Processing parameters
        self.skip_frames = config['detection']['skip_frames']
        self.frame_count = 0
        self.detection_frame = 0
        
        # Performance tracking
        self.processing_times = {
            'detection': [],
            'recognition': [],
            'tracking': [],
            'total': []
        }
        
        # Control flags
        self.is_running = False
        self.show_visualization = False
        
        # Threading
        self.process_queue = queue.Queue(maxsize=10)
        self.result_queue = queue.Queue()
        
        self.logger.info(f"Video processor initialized for source: {video_source}")
        self.logger_manager.log_configuration(config)
    
    def _initialize_video_capture(self) -> bool:
        """
        Initialize video capture
        
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            self.cap = cv2.VideoCapture(self.video_source)
            
            if not self.cap.isOpened():
                self.logger.error(f"Failed to open video source: {self.video_source}")
                return False
            
            # Get video properties
            self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            # Update tracker with frame dimensions
            self.face_tracker.update_frame_dimensions(self.frame_width, self.frame_height)
            
            self.logger.info(f"Video initialized: {self.frame_width}x{self.frame_height} @ {self.fps:.1f} FPS")
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing video capture: {str(e)}")
            return False
    
    def _generate_face_id(self) -> str:
        """
        Generate unique face ID
        
        Returns:
            Unique face identifier
        """
        # Simple sequential ID generation
        existing_faces = self.db_manager.get_all_faces()
        face_count = len(existing_faces) + 1
        return f"face_{face_count:06d}"
    
    def _save_face_image(self, face_crop: np.ndarray, face_id: str, 
                        event_type: str, timestamp: datetime) -> str:
        """
        Save face image to disk
        
        Args:
            face_crop: Cropped face image
            face_id: Face identifier
            event_type: 'entry' or 'exit'
            timestamp: Event timestamp
            
        Returns:
            Path to saved image
        """
        try:
            # Create directory structure
            date_str = timestamp.strftime('%Y-%m-%d')
            base_dir = Path(self.config['logging']['image_base_path'])
            
            if event_type == 'entry':
                save_dir = base_dir / 'entries' / date_str
            else:
                save_dir = base_dir / 'exits' / date_str
            
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate filename
            time_str = timestamp.strftime('%H%M%S')
            filename = f"{face_id}_{event_type}_{time_str}.jpg"
            file_path = save_dir / filename
            
            # Save image
            success = cv2.imwrite(str(file_path), face_crop)
            
            if success:
                self.logger_manager.log_file_operation('save', str(file_path), True)
                return str(file_path)
            else:
                self.logger_manager.log_file_operation('save', str(file_path), False, "OpenCV write failed")
                return ""
                
        except Exception as e:
            self.logger.error(f"Error saving face image: {str(e)}")
            return ""
    
    def _save_registered_face(self, face_crop: np.ndarray, face_id: str) -> str:
        """
        Save registered face image
        
        Args:
            face_crop: Cropped face image
            face_id: Face identifier
            
        Returns:
            Path to saved image
        """
        try:
            # Create faces directory
            faces_dir = Path('data/faces')
            faces_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate filename
            filename = f"{face_id}.jpg"
            file_path = faces_dir / filename
            
            # Save image
            success = cv2.imwrite(str(file_path), face_crop)
            
            if success:
                self.logger_manager.log_file_operation('save', str(file_path), True)
                return str(file_path)
            else:
                self.logger_manager.log_file_operation('save', str(file_path), False, "OpenCV write failed")
                return ""
                
        except Exception as e:
            self.logger.error(f"Error saving registered face: {str(e)}")
            return ""
    
    def _process_frame(self, frame: np.ndarray) -> Tuple[List, List]:
        """
        Process a single frame
        
        Args:
            frame: Input frame
            
        Returns:
            Tuple of (detections, recognized_faces)
        """
        # Preprocess frame
        processed_frame = self.face_detector.preprocess_frame(frame)
        
        # Detect faces
        start_time = time.time()
        detections = self.face_detector.detect_faces(processed_frame)
        detection_time = time.time() - start_time
        self.processing_times['detection'].append(detection_time)
        
        self.logger_manager.log_face_detection(
            self.frame_count, len(detections), detection_time
        )
        
        # Recognize faces
        start_time = time.time()
        recognized_faces = []
        
        for detection in detections:
            x1, y1, x2, y2, conf = detection
            
            # Crop face
            face_crop = self.face_detector.crop_face(processed_frame, (x1, y1, x2, y2))
            
            if face_crop is not None:
                # Extract embedding
                embedding = self.face_recognizer.extract_embedding(face_crop)
                
                if embedding is not None:
                    # Try to recognize face
                    face_id, confidence = self.face_recognizer.recognize_face(embedding)
                    
                    if face_id is None and self.face_recognizer.should_register_face(embedding):
                        # Register new face
                        face_id = self._generate_face_id()
                        
                        if self.face_recognizer.register_face(face_id, embedding):
                            # Save to database
                            self.db_manager.register_face(face_id, embedding)
                            
                            # Save face image
                            self._save_registered_face(face_crop, face_id)
                            
                            self.logger_manager.log_face_recognition(
                                face_id, f"new_detection_{self.frame_count}", 
                                confidence, is_new_registration=True
                            )
                        else:
                            face_id = None
                    
                    if face_id:
                        self.logger_manager.log_face_recognition(
                            face_id, f"detection_{self.frame_count}", confidence
                        )
                    
                    recognized_faces.append(face_id)
                else:
                    recognized_faces.append(None)
            else:
                recognized_faces.append(None)
        
        recognition_time = time.time() - start_time
        self.processing_times['recognition'].append(recognition_time)
        
        return detections, recognized_faces
    
    def _handle_entry_exit_events(self, events: List, frame: np.ndarray):
        """
        Handle entry and exit events
        
        Args:
            events: List of (TrackedFace, event_type) tuples
            frame: Current frame
        """
        for tracked_face, event_type in events:
            try:
                # Crop face from current frame
                face_crop = self.face_detector.crop_face(frame, tracked_face.bbox)
                
                if face_crop is not None:
                    # Save face image
                    timestamp = datetime.now()
                    image_path = self._save_face_image(
                        face_crop, tracked_face.face_id or tracked_face.track_id, 
                        event_type, timestamp
                    )
                    
                    # Log to database
                    self.db_manager.log_event(
                        face_id=tracked_face.face_id,
                        track_id=tracked_face.track_id,
                        event_type=event_type,
                        image_path=image_path,
                        confidence=tracked_face.confidence,
                        bbox=tracked_face.bbox,
                        frame_number=self.frame_count,
                        metadata={'timestamp': timestamp.isoformat()}
                    )
                    
                    # Log event
                    self.logger_manager.log_entry_exit(
                        tracked_face.face_id, tracked_face.track_id, 
                        event_type, image_path, tracked_face.bbox, self.frame_count
                    )
                
            except Exception as e:
                self.logger.error(f"Error handling {event_type} event: {str(e)}")
    
    def process_video(self, show_window: bool = False) -> bool:
        """
        Main video processing loop
        
        Args:
            show_window: Whether to show visualization window
            
        Returns:
            True if processing completed successfully, False otherwise
        """
        if not self._initialize_video_capture():
            return False
        
        self.is_running = True
        self.show_visualization = show_window
        
        self.logger.info("Starting video processing")
        
        try:
            frame_start_time = time.time()
            
            while self.is_running:
                ret, frame = self.cap.read()
                
                if not ret:
                    self.logger.info("End of video stream reached")
                    break
                
                self.frame_count += 1
                
                # Skip frames for performance
                should_detect = (self.frame_count % (self.skip_frames + 1)) == 0
                
                if should_detect:
                    self.detection_frame += 1
                    
                    # Process frame
                    total_start_time = time.time()
                    detections, recognized_faces = self._process_frame(frame)
                    
                    # Update tracker
                    tracking_start_time = time.time()
                    active_tracks = self.face_tracker.update(detections, recognized_faces)
                    tracking_time = time.time() - tracking_start_time
                    self.processing_times['tracking'].append(tracking_time)
                    
                    # Handle entry/exit events
                    events = self.face_tracker.get_entry_exit_events()
                    if events:
                        self._handle_entry_exit_events(events, frame)
                    
                    total_time = time.time() - total_start_time
                    self.processing_times['total'].append(total_time)
                    
                    # Log performance metrics
                    if self.detection_frame % 50 == 0:
                        avg_detection = np.mean(self.processing_times['detection'][-10:])
                        avg_recognition = np.mean(self.processing_times['recognition'][-10:])
                        avg_tracking = np.mean(self.processing_times['tracking'][-10:])
                        avg_total = np.mean(self.processing_times['total'][-10:])
                        
                        metrics = {
                            'detection_time': f"{avg_detection:.3f}s",
                            'recognition_time': f"{avg_recognition:.3f}s",
                            'tracking_time': f"{avg_tracking:.3f}s",
                            'total_time': f"{avg_total:.3f}s",
                            'fps': f"{1.0/avg_total:.1f}"
                        }
                        
                        self.logger_manager.log_performance_metrics(metrics)
                
                # Calculate current FPS
                current_time = time.time()
                elapsed_time = current_time - frame_start_time
                current_fps = 1.0 / elapsed_time if elapsed_time > 0 else 0
                frame_start_time = current_time
                
                # Log video processing stats
                self.logger_manager.log_video_processing(
                    self.frame_count, current_fps, len(self.face_tracker.tracked_faces)
                )
                
                # Visualization
                if self.show_visualization:
                    vis_frame = self.face_tracker.visualize_tracking(frame)
                    
                    # Add detection boxes if this was a detection frame
                    if should_detect and 'detections' in locals():
                        vis_frame = self.face_detector.visualize_detections(vis_frame, detections)
                    
                    cv2.imshow('Face Tracker', vis_frame)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        self.logger.info("Processing stopped by user")
                        break
                    elif key == ord('s'):
                        # Save current frame
                        save_path = f"debug_frame_{self.frame_count}.jpg"
                        cv2.imwrite(save_path, vis_frame)
                        self.logger.info(f"Frame saved to {save_path}")
                
                # Limit FPS if needed
                fps_limit = self.config['video'].get('fps_limit', 30)
                if fps_limit > 0:
                    time.sleep(max(0, 1.0/fps_limit - elapsed_time))
            
            # Final statistics
            stats = self.face_tracker.get_statistics()
            self.logger_manager.log_session_summary(stats)
            
            return True
            
        except KeyboardInterrupt:
            self.logger.info("Processing interrupted by user")
            return True
        except Exception as e:
            self.logger_manager.log_system_error('video_processor', str(e), e)
            return False
        finally:
            self._cleanup()
    
    def _cleanup(self):
        """Cleanup resources"""
        if self.cap:
            self.cap.release()
        
        if self.show_visualization:
            cv2.destroyAllWindows()
        
        self.is_running = False
        self.logger.info("Video processor cleanup completed")
    
    def stop_processing(self):
        """Stop video processing"""
        self.is_running = False
        self.logger.info("Stop signal sent to video processor")
    
    def get_statistics(self) -> Dict:
        """
        Get processing statistics
        
        Returns:
            Dictionary with processing statistics
        """
        tracker_stats = self.face_tracker.get_statistics()
        db_stats = self.db_manager.get_visitor_statistics()
        
        # Calculate average processing times
        avg_times = {}
        for key, times in self.processing_times.items():
            if times:
                avg_times[f'avg_{key}_time'] = np.mean(times)
                avg_times[f'max_{key}_time'] = np.max(times)
            else:
                avg_times[f'avg_{key}_time'] = 0
                avg_times[f'max_{key}_time'] = 0
        
        recognition_stats = self.face_recognizer.get_embedding_statistics()
        
        return {
            'frame_count': self.frame_count,
            'detection_frames': self.detection_frame,
            'skip_frames': self.skip_frames,
            **tracker_stats,
            **db_stats,
            **avg_times,
            **recognition_stats
        }
