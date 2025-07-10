"""
Utility functions for the face tracking system
"""

import json
import os
import logging
import cv2
import numpy as np
from typing import Dict, Optional, Tuple, List
from pathlib import Path
import hashlib
from datetime import datetime, timedelta


def load_config(config_path: str) -> Optional[Dict]:
    """
    Load configuration from JSON file
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary or None if failed
    """
    try:
        if not os.path.exists(config_path):
            print(f"Configuration file not found: {config_path}")
            return None
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Validate required sections
        required_sections = ['detection', 'recognition', 'tracking', 'database', 'logging']
        for section in required_sections:
            if section not in config:
                print(f"Missing required configuration section: {section}")
                return None
        
        print(f"Configuration loaded successfully from {config_path}")
        return config
        
    except json.JSONDecodeError as e:
        print(f"Invalid JSON in configuration file: {str(e)}")
        return None
    except Exception as e:
        print(f"Error loading configuration: {str(e)}")
        return None


def save_config(config: Dict, config_path: str) -> bool:
    """
    Save configuration to JSON file
    
    Args:
        config: Configuration dictionary
        config_path: Path to save configuration file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        
        print(f"Configuration saved to {config_path}")
        return True
        
    except Exception as e:
        print(f"Error saving configuration: {str(e)}")
        return False


def validate_video_source(video_source: str) -> bool:
    """
    Validate video source (file or stream)
    
    Args:
        video_source: Path to video file or stream URL
        
    Returns:
        True if valid, False otherwise
    """
    try:
        # Check if it's a file
        if os.path.isfile(video_source):
            # Check file extension
            valid_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm']
            file_ext = os.path.splitext(video_source)[1].lower()
            
            if file_ext not in valid_extensions:
                print(f"Unsupported video format: {file_ext}")
                return False
            
            return True
        
        # Check if it's a stream URL
        elif video_source.startswith(('rtsp://', 'http://', 'https://')):
            return True
        
        # Check if it's a camera index
        elif video_source.isdigit():
            return True
        
        else:
            print(f"Invalid video source: {video_source}")
            return False
            
    except Exception as e:
        print(f"Error validating video source: {str(e)}")
        return False


def create_directory_structure(base_path: str = ".") -> bool:
    """
    Create necessary directory structure
    
    Args:
        base_path: Base path for directory creation
        
    Returns:
        True if successful, False otherwise
    """
    try:
        directories = [
            'logs',
            'logs/entries',
            'logs/exits',
            'data',
            'data/faces',
            'data/exports'
        ]
        
        base_path = Path(base_path)
        
        for directory in directories:
            dir_path = base_path / directory
            dir_path.mkdir(parents=True, exist_ok=True)
        
        print("Directory structure created successfully")
        return True
        
    except Exception as e:
        print(f"Error creating directory structure: {str(e)}")
        return False


def resize_image(image: np.ndarray, max_width: int = 1280, max_height: int = 720) -> np.ndarray:
    """
    Resize image while maintaining aspect ratio
    
    Args:
        image: Input image
        max_width: Maximum width
        max_height: Maximum height
        
    Returns:
        Resized image
    """
    height, width = image.shape[:2]
    
    if width <= max_width and height <= max_height:
        return image
    
    # Calculate scaling factor
    scale = min(max_width / width, max_height / height)
    
    new_width = int(width * scale)
    new_height = int(height * scale)
    
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return resized


def calculate_image_hash(image: np.ndarray) -> str:
    """
    Calculate hash of image for duplicate detection
    
    Args:
        image: Input image
        
    Returns:
        Image hash string
    """
    # Convert to grayscale and resize to standard size
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    resized = cv2.resize(gray, (8, 8), interpolation=cv2.INTER_AREA)
    
    # Calculate hash
    hash_bytes = hashlib.md5(resized.tobytes()).hexdigest()
    return hash_bytes


def format_timestamp(timestamp: datetime = None) -> str:
    """
    Format timestamp for consistent logging
    
    Args:
        timestamp: Timestamp to format (current time if None)
        
    Returns:
        Formatted timestamp string
    """
    if timestamp is None:
        timestamp = datetime.now()
    
    return timestamp.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]


def parse_timestamp(timestamp_str: str) -> Optional[datetime]:
    """
    Parse timestamp string
    
    Args:
        timestamp_str: Timestamp string to parse
        
    Returns:
        Datetime object or None if parsing failed
    """
    try:
        # Try different formats
        formats = [
            '%Y-%m-%d %H:%M:%S.%f',
            '%Y-%m-%d %H:%M:%S',
            '%Y-%m-%dT%H:%M:%S.%f',
            '%Y-%m-%dT%H:%M:%S'
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(timestamp_str, fmt)
            except ValueError:
                continue
        
        return None
        
    except Exception:
        return None


def get_date_range(days_back: int = 7) -> Tuple[str, str]:
    """
    Get date range for queries
    
    Args:
        days_back: Number of days to go back
        
    Returns:
        Tuple of (start_date, end_date) strings
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    
    return start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')


def clean_filename(filename: str) -> str:
    """
    Clean filename for safe file system operations
    
    Args:
        filename: Original filename
        
    Returns:
        Cleaned filename
    """
    # Remove or replace invalid characters
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    
    # Remove leading/trailing spaces and dots
    filename = filename.strip(' .')
    
    # Limit length
    if len(filename) > 255:
        name, ext = os.path.splitext(filename)
        filename = name[:255-len(ext)] + ext
    
    return filename


def ensure_image_format(image: np.ndarray, target_format: str = 'BGR') -> np.ndarray:
    """
    Ensure image is in specified format
    
    Args:
        image: Input image
        target_format: Target format ('BGR', 'RGB', 'GRAY')
        
    Returns:
        Image in target format
    """
    if len(image.shape) == 2:
        # Grayscale image
        if target_format == 'BGR':
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif target_format == 'RGB':
            return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            return image
    
    elif len(image.shape) == 3:
        # Color image
        if target_format == 'GRAY':
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif target_format == 'RGB':
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            return image
    
    return image


def calculate_fps(frame_times: List[float], window_size: int = 30) -> float:
    """
    Calculate FPS from frame processing times
    
    Args:
        frame_times: List of frame processing times
        window_size: Window size for averaging
        
    Returns:
        Calculated FPS
    """
    if len(frame_times) < 2:
        return 0.0
    
    # Use recent frames for calculation
    recent_times = frame_times[-window_size:]
    
    if len(recent_times) < 2:
        return 0.0
    
    # Calculate time differences
    time_diffs = [recent_times[i] - recent_times[i-1] for i in range(1, len(recent_times))]
    
    if not time_diffs:
        return 0.0
    
    avg_time_diff = sum(time_diffs) / len(time_diffs)
    
    if avg_time_diff <= 0:
        return 0.0
    
    return 1.0 / avg_time_diff


def get_system_info() -> Dict:
    """
    Get system information
    
    Returns:
        Dictionary with system information
    """
    import platform
    import psutil
    
    try:
        info = {
            'platform': platform.platform(),
            'python_version': platform.python_version(),
            'cpu_count': psutil.cpu_count(),
            'memory_total': psutil.virtual_memory().total,
            'memory_available': psutil.virtual_memory().available,
            'opencv_version': cv2.__version__
        }
        
        # Try to get GPU information
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            if gpus:
                info['gpu_name'] = gpus[0].name
                info['gpu_memory'] = gpus[0].memoryTotal
        except ImportError:
            info['gpu_name'] = 'Not available'
            info['gpu_memory'] = 0
        
        return info
        
    except Exception as e:
        return {'error': str(e)}


def log_system_info(logger: logging.Logger):
    """
    Log system information
    
    Args:
        logger: Logger instance
    """
    info = get_system_info()
    
    logger.info("System Information:")
    for key, value in info.items():
        logger.info(f"  {key}: {value}")


def validate_bbox(bbox: Tuple[int, int, int, int], 
                  frame_width: int, frame_height: int) -> Tuple[int, int, int, int]:
    """
    Validate and clip bounding box to frame dimensions
    
    Args:
        bbox: Bounding box (x1, y1, x2, y2)
        frame_width: Frame width
        frame_height: Frame height
        
    Returns:
        Validated bounding box
    """
    x1, y1, x2, y2 = bbox
    
    # Ensure coordinates are within frame bounds
    x1 = max(0, min(x1, frame_width - 1))
    y1 = max(0, min(y1, frame_height - 1))
    x2 = max(x1 + 1, min(x2, frame_width))
    y2 = max(y1 + 1, min(y2, frame_height))
    
    return x1, y1, x2, y2


def create_color_map(num_colors: int) -> List[Tuple[int, int, int]]:
    """
    Create a list of distinct colors for visualization
    
    Args:
        num_colors: Number of colors to generate
        
    Returns:
        List of RGB color tuples
    """
    colors = []
    
    for i in range(num_colors):
        hue = (i * 137.508) % 360  # Golden angle approximation
        saturation = 0.7 + (i % 3) * 0.1  # Vary saturation
        value = 0.8 + (i % 2) * 0.2  # Vary brightness
        
        # Convert HSV to RGB
        import colorsys
        rgb = colorsys.hsv_to_rgb(hue/360, saturation, value)
        
        # Convert to 0-255 range
        rgb_255 = tuple(int(c * 255) for c in rgb)
        colors.append(rgb_255)
    
    return colors


def export_logs_to_csv(log_file_path: str, output_path: str, 
                      date_filter: str = None) -> bool:
    """
    Export log file to CSV format
    
    Args:
        log_file_path: Path to log file
        output_path: Path for CSV output
        date_filter: Date filter (YYYY-MM-DD format)
        
    Returns:
        True if successful, False otherwise
    """
    try:
        import csv
        import re
        
        # Log line pattern
        log_pattern = r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+ - (\w+) - (\w+) - (.+)'
        
        with open(log_file_path, 'r') as log_file, open(output_path, 'w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(['Timestamp', 'Module', 'Level', 'Message'])
            
            for line in log_file:
                match = re.match(log_pattern, line.strip())
                if match:
                    timestamp, module, level, message = match.groups()
                    
                    # Apply date filter if specified
                    if date_filter and not timestamp.startswith(date_filter):
                        continue
                    
                    writer.writerow([timestamp, module, level, message])
        
        print(f"Logs exported to {output_path}")
        return True
        
    except Exception as e:
        print(f"Error exporting logs to CSV: {str(e)}")
        return False
