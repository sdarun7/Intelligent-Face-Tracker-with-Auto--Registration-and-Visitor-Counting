"""
Logger Manager Module
Handles comprehensive logging for the face tracking system
"""

import logging
import os
from datetime import datetime
from typing import Optional
from logging.handlers import RotatingFileHandler
import json


class LoggerManager:
    """Centralized logging manager for the face tracking system"""
    
    def __init__(self, log_level: int = logging.INFO, config: dict = None):
        """
        Initialize the logger manager
        
        Args:
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            config: Configuration dictionary
        """
        self.log_level = log_level
        
        # Default configuration
        default_config = {
            'logging': {
                'log_file': 'logs/events.log',
                'max_log_size_mb': 100,
                'backup_count': 5
            }
        }
        
        self.config = config if config else default_config
        
        # Ensure logs directory exists
        log_dir = os.path.dirname(self.config['logging']['log_file'])
        os.makedirs(log_dir, exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        
        # Main logger for this class
        self.logger = self.get_logger('logger_manager')
        self.logger.info("Logger manager initialized")
    
    def _setup_logging(self):
        """Setup logging configuration"""
        # Root logger configuration
        root_logger = logging.getLogger()
        root_logger.setLevel(self.log_level)
        
        # Clear any existing handlers
        root_logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(self.log_level)
        
        # File handler with rotation
        max_bytes = self.config['logging']['max_log_size_mb'] * 1024 * 1024
        file_handler = RotatingFileHandler(
            self.config['logging']['log_file'],
            maxBytes=max_bytes,
            backupCount=self.config['logging']['backup_count']
        )
        file_handler.setLevel(self.log_level)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        
        # Add handlers to root logger
        root_logger.addHandler(console_handler)
        root_logger.addHandler(file_handler)
        
        # Suppress verbose logging from some libraries
        logging.getLogger('ultralytics').setLevel(logging.WARNING)
        logging.getLogger('insightface').setLevel(logging.WARNING)
        logging.getLogger('onnxruntime').setLevel(logging.WARNING)
    
    def get_logger(self, name: str) -> logging.Logger:
        """
        Get a logger with the specified name
        
        Args:
            name: Logger name
            
        Returns:
            Configured logger instance
        """
        return logging.getLogger(name)
    
    def log_face_detection(self, frame_number: int, detections_count: int, 
                          processing_time: float):
        """
        Log face detection event
        
        Args:
            frame_number: Frame number
            detections_count: Number of faces detected
            processing_time: Time taken for detection
        """
        logger = self.get_logger('face_detection')
        logger.debug(f"Frame {frame_number}: {detections_count} faces detected in {processing_time:.3f}s")
    
    def log_face_recognition(self, face_id: Optional[str], track_id: str, 
                           confidence: float, is_new_registration: bool = False):
        """
        Log face recognition event
        
        Args:
            face_id: Recognized face ID (None if not recognized)
            track_id: Track ID
            confidence: Recognition confidence
            is_new_registration: Whether this is a new face registration
        """
        logger = self.get_logger('face_recognition')
        
        if is_new_registration:
            logger.info(f"NEW FACE REGISTERED: {face_id} (Track: {track_id})")
        elif face_id:
            logger.info(f"FACE RECOGNIZED: {face_id} (Track: {track_id}, Confidence: {confidence:.3f})")
        else:
            logger.debug(f"Face not recognized (Track: {track_id}, Best confidence: {confidence:.3f})")
    
    def log_tracking_event(self, track_id: str, event_type: str, face_id: Optional[str] = None):
        """
        Log face tracking event
        
        Args:
            track_id: Track ID
            event_type: Type of tracking event
            face_id: Associated face ID if available
        """
        logger = self.get_logger('face_tracking')
        
        if face_id:
            logger.info(f"TRACKING {event_type.upper()}: {track_id} (Face: {face_id})")
        else:
            logger.debug(f"Tracking {event_type}: {track_id}")
    
    def log_entry_exit(self, face_id: Optional[str], track_id: str, event_type: str, 
                      image_path: str, bbox: tuple, frame_number: int):
        """
        Log entry/exit event
        
        Args:
            face_id: Face ID (None if unrecognized)
            track_id: Track ID
            event_type: 'entry' or 'exit'
            image_path: Path to saved image
            bbox: Bounding box coordinates
            frame_number: Frame number
        """
        logger = self.get_logger('entry_exit')
        
        x1, y1, x2, y2 = bbox
        
        log_message = (
            f"{event_type.upper()} EVENT - "
            f"Face: {face_id or 'Unknown'}, "
            f"Track: {track_id}, "
            f"Frame: {frame_number}, "
            f"BBox: ({x1},{y1},{x2},{y2}), "
            f"Image: {image_path}"
        )
        
        logger.info(log_message)
    
    def log_database_operation(self, operation: str, success: bool, details: str = ""):
        """
        Log database operation
        
        Args:
            operation: Type of database operation
            success: Whether operation was successful
            details: Additional details
        """
        logger = self.get_logger('database')
        
        if success:
            logger.debug(f"DB {operation} successful: {details}")
        else:
            logger.error(f"DB {operation} failed: {details}")
    
    def log_video_processing(self, frame_number: int, fps: float, total_faces: int):
        """
        Log video processing statistics
        
        Args:
            frame_number: Current frame number
            fps: Current FPS
            total_faces: Total faces being tracked
        """
        logger = self.get_logger('video_processing')
        
        # Log every 100 frames to avoid spam
        if frame_number % 100 == 0:
            logger.info(f"Frame {frame_number}: FPS={fps:.1f}, Active faces={total_faces}")
    
    def log_system_error(self, component: str, error_message: str, exception: Exception = None):
        """
        Log system error
        
        Args:
            component: Component where error occurred
            error_message: Error description
            exception: Exception object if available
        """
        logger = self.get_logger('system_error')
        
        if exception:
            logger.error(f"{component} ERROR: {error_message}", exc_info=True)
        else:
            logger.error(f"{component} ERROR: {error_message}")
    
    def log_configuration(self, config: dict):
        """
        Log system configuration
        
        Args:
            config: Configuration dictionary
        """
        logger = self.get_logger('configuration')
        
        # Log important configuration parameters
        detection_config = config.get('detection', {})
        recognition_config = config.get('recognition', {})
        tracking_config = config.get('tracking', {})
        
        config_summary = {
            'detection': {
                'confidence_threshold': detection_config.get('confidence_threshold'),
                'skip_frames': detection_config.get('skip_frames'),
                'max_disappear_frames': detection_config.get('max_disappear_frames')
            },
            'recognition': {
                'similarity_threshold': recognition_config.get('similarity_threshold'),
                'registration_threshold': recognition_config.get('registration_threshold')
            },
            'tracking': {
                'max_distance': tracking_config.get('max_distance'),
                'max_age': tracking_config.get('max_age'),
                'min_hits': tracking_config.get('min_hits')
            }
        }
        
        logger.info(f"System configuration: {json.dumps(config_summary, indent=2)}")
    
    def log_performance_metrics(self, metrics: dict):
        """
        Log performance metrics
        
        Args:
            metrics: Dictionary containing performance metrics
        """
        logger = self.get_logger('performance')
        
        metrics_str = ", ".join([f"{k}={v}" for k, v in metrics.items()])
        logger.info(f"Performance metrics: {metrics_str}")
    
    def log_session_summary(self, stats: dict):
        """
        Log session summary
        
        Args:
            stats: Session statistics dictionary
        """
        logger = self.get_logger('session')
        
        summary = (
            f"SESSION SUMMARY - "
            f"Unique visitors: {stats.get('unique_visitors', 0)}, "
            f"Total entries: {stats.get('total_entries', 0)}, "
            f"Total exits: {stats.get('total_exits', 0)}, "
            f"Currently inside: {stats.get('currently_inside', 0)}, "
            f"Frames processed: {stats.get('frame_number', 0)}"
        )
        
        logger.info(summary)
    
    def create_daily_summary(self):
        """Create a daily summary log entry"""
        logger = self.get_logger('daily_summary')
        
        today = datetime.now().strftime('%Y-%m-%d')
        logger.info(f"Daily summary for {today} - Check database for detailed statistics")
    
    def log_file_operation(self, operation: str, file_path: str, success: bool, error: str = ""):
        """
        Log file operation
        
        Args:
            operation: Type of file operation (save, load, delete, etc.)
            file_path: Path of the file
            success: Whether operation was successful
            error: Error message if operation failed
        """
        logger = self.get_logger('file_operations')
        
        if success:
            logger.debug(f"File {operation} successful: {file_path}")
        else:
            logger.error(f"File {operation} failed: {file_path} - {error}")
    
    def set_log_level(self, level: int):
        """
        Change logging level at runtime
        
        Args:
            level: New logging level
        """
        self.log_level = level
        
        # Update all handlers
        root_logger = logging.getLogger()
        root_logger.setLevel(level)
        
        for handler in root_logger.handlers:
            handler.setLevel(level)
        
        logger = self.get_logger('logger_manager')
        logger.info(f"Log level changed to {logging.getLevelName(level)}")
