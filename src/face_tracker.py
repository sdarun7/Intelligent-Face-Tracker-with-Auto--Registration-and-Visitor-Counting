"""
Face Tracking Module
Handles face tracking across video frames and entry/exit detection
"""

import cv2
import numpy as np
import logging
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field
from datetime import datetime
import uuid


@dataclass
class TrackedFace:
    """Data class for tracked face information"""
    track_id: str
    face_id: Optional[str] = None
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)
    center: Tuple[int, int] = (0, 0)
    confidence: float = 0.0
    age: int = 0
    hits: int = 0
    disappeared_frames: int = 0
    last_seen_frame: int = 0
    entry_logged: bool = False
    exit_logged: bool = False
    inside_zone: bool = False
    embedding: Optional[np.ndarray] = None
    trajectory: List[Tuple[int, int]] = field(default_factory=list)
    
    def update_position(self, bbox: Tuple[int, int, int, int], frame_number: int):
        """Update face position and trajectory"""
        self.bbox = bbox
        self.center = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
        self.trajectory.append(self.center)
        self.last_seen_frame = frame_number
        self.age += 1
        self.hits += 1
        self.disappeared_frames = 0
        
        # Keep trajectory length manageable
        if len(self.trajectory) > 30:
            self.trajectory = self.trajectory[-30:]


class FaceTracker:
    """OpenCV-based face tracker with entry/exit detection"""
    
    def __init__(self, config: dict):
        """
        Initialize the face tracker
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Tracking parameters
        self.max_distance = config['tracking']['max_distance']
        self.max_age = config['tracking']['max_age']
        self.min_hits = config['tracking']['min_hits']
        self.iou_threshold = config['tracking']['iou_threshold']
        self.max_disappear_frames = config['detection']['max_disappear_frames']
        
        # Entry/exit detection
        self.entry_exit_zone_ratio = config['system']['entry_exit_zone_ratio']
        self.frame_width = 0
        self.frame_height = 0
        self.entry_zones = {}  # Define entry/exit zones
        
        # Tracked faces
        self.tracked_faces: Dict[str, TrackedFace] = {}
        self.next_track_id = 1
        self.frame_number = 0
        
        # Statistics
        self.total_entries = 0
        self.total_exits = 0
        self.unique_visitors = set()
        
        self.logger.info("Face tracker initialized")
    
    def update_frame_dimensions(self, width: int, height: int):
        """
        Update frame dimensions and recalculate zones
        
        Args:
            width: Frame width
            height: Frame height
        """
        if self.frame_width != width or self.frame_height != height:
            self.frame_width = width
            self.frame_height = height
            self._calculate_entry_exit_zones()
            self.logger.debug(f"Updated frame dimensions: {width}x{height}")
    
    def _calculate_entry_exit_zones(self):
        """Calculate entry and exit zones based on frame dimensions"""
        zone_width = int(self.frame_width * self.entry_exit_zone_ratio)
        zone_height = self.frame_height
        
        # Define zones (can be customized based on camera setup)
        self.entry_zones = {
            'left': (0, 0, zone_width, zone_height),
            'right': (self.frame_width - zone_width, 0, self.frame_width, zone_height),
            'top': (0, 0, self.frame_width, int(zone_height * self.entry_exit_zone_ratio)),
            'bottom': (0, self.frame_height - int(zone_height * self.entry_exit_zone_ratio), 
                      self.frame_width, self.frame_height)
        }
        
        self.logger.debug(f"Entry/exit zones calculated: {self.entry_zones}")
    
    def _calculate_distance(self, center1: Tuple[int, int], center2: Tuple[int, int]) -> float:
        """Calculate Euclidean distance between two centers"""
        return np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
    
    def _calculate_iou(self, bbox1: Tuple[int, int, int, int], 
                      bbox2: Tuple[int, int, int, int]) -> float:
        """Calculate Intersection over Union of two bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _is_in_entry_zone(self, center: Tuple[int, int]) -> bool:
        """Check if point is in any entry zone"""
        cx, cy = center
        
        for zone_name, (x1, y1, x2, y2) in self.entry_zones.items():
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                return True
        
        return False
    
    def _detect_entry_exit(self, tracked_face: TrackedFace) -> Optional[str]:
        """
        Detect if face is entering or exiting based on trajectory
        
        Args:
            tracked_face: TrackedFace object
            
        Returns:
            'entry', 'exit', or None
        """
        if len(tracked_face.trajectory) < 3:
            return None
        
        current_in_zone = self._is_in_entry_zone(tracked_face.center)
        
        # Simple entry/exit detection based on zone transition
        if not tracked_face.inside_zone and current_in_zone:
            # Just entered the zone
            tracked_face.inside_zone = True
            if not tracked_face.entry_logged:
                return 'entry'
        elif tracked_face.inside_zone and not current_in_zone:
            # Just left the zone
            tracked_face.inside_zone = False
            if not tracked_face.exit_logged and tracked_face.entry_logged:
                return 'exit'
        
        return None
    
    def update(self, detections: List[Tuple[int, int, int, int, float]], 
              face_ids: List[Optional[str]] = None) -> List[TrackedFace]:
        """
        Update tracker with new detections
        
        Args:
            detections: List of face detections (x1, y1, x2, y2, confidence)
            face_ids: Optional list of recognized face IDs
            
        Returns:
            List of currently tracked faces
        """
        self.frame_number += 1
        
        if face_ids is None:
            face_ids = [None] * len(detections)
        
        # Convert detections to centers
        detection_centers = []
        detection_bboxes = []
        
        for detection in detections:
            x1, y1, x2, y2, conf = detection
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            detection_centers.append(center)
            detection_bboxes.append((x1, y1, x2, y2))
        
        # Match detections to existing tracks
        matched_tracks = set()
        matched_detections = set()
        
        # Calculate distance matrix
        for track_id, tracked_face in self.tracked_faces.items():
            if tracked_face.disappeared_frames > 0:
                continue
                
            best_match_idx = None
            best_distance = float('inf')
            
            for i, detection_center in enumerate(detection_centers):
                if i in matched_detections:
                    continue
                
                # Calculate distance
                distance = self._calculate_distance(tracked_face.center, detection_center)
                
                # Also consider IoU
                iou = self._calculate_iou(tracked_face.bbox, detection_bboxes[i])
                
                # Combined score (lower is better)
                score = distance - (iou * 100)  # Give weight to IoU
                
                if score < best_distance and distance < self.max_distance:
                    best_distance = score
                    best_match_idx = i
            
            # Update matched track
            if best_match_idx is not None:
                detection = detections[best_match_idx]
                x1, y1, x2, y2, conf = detection
                
                tracked_face.update_position((x1, y1, x2, y2), self.frame_number)
                tracked_face.confidence = conf
                
                # Update face ID if recognized
                if face_ids[best_match_idx] is not None:
                    if tracked_face.face_id is None:
                        tracked_face.face_id = face_ids[best_match_idx]
                        self.unique_visitors.add(face_ids[best_match_idx])
                
                matched_tracks.add(track_id)
                matched_detections.add(best_match_idx)
        
        # Create new tracks for unmatched detections
        for i, detection in enumerate(detections):
            if i in matched_detections:
                continue
            
            # Create new track
            track_id = f"track_{self.next_track_id:04d}"
            self.next_track_id += 1
            
            x1, y1, x2, y2, conf = detection
            new_track = TrackedFace(
                track_id=track_id,
                face_id=face_ids[i],
                bbox=(x1, y1, x2, y2),
                center=((x1 + x2) // 2, (y1 + y2) // 2),
                confidence=conf,
                last_seen_frame=self.frame_number
            )
            
            # Add to trajectory
            new_track.trajectory.append(new_track.center)
            
            # Add face ID to unique visitors if recognized
            if face_ids[i] is not None:
                self.unique_visitors.add(face_ids[i])
            
            self.tracked_faces[track_id] = new_track
            self.logger.debug(f"Created new track: {track_id}")
        
        # Update disappeared frames for unmatched tracks
        tracks_to_remove = []
        for track_id, tracked_face in self.tracked_faces.items():
            if track_id not in matched_tracks:
                tracked_face.disappeared_frames += 1
                
                # Remove tracks that have disappeared for too long
                if tracked_face.disappeared_frames > self.max_disappear_frames:
                    tracks_to_remove.append(track_id)
        
        # Remove old tracks
        for track_id in tracks_to_remove:
            del self.tracked_faces[track_id]
            self.logger.debug(f"Removed track: {track_id}")
        
        # Return list of active tracks
        active_tracks = [
            track for track in self.tracked_faces.values() 
            if track.hits >= self.min_hits and track.disappeared_frames == 0
        ]
        
        return active_tracks
    
    def get_entry_exit_events(self) -> List[Tuple[TrackedFace, str]]:
        """
        Get entry/exit events for current frame
        
        Returns:
            List of (TrackedFace, event_type) tuples
        """
        events = []
        
        for tracked_face in self.tracked_faces.values():
            if tracked_face.hits < self.min_hits:
                continue
            
            event_type = self._detect_entry_exit(tracked_face)
            
            if event_type == 'entry' and not tracked_face.entry_logged:
                tracked_face.entry_logged = True
                self.total_entries += 1
                events.append((tracked_face, 'entry'))
                self.logger.info(f"Entry detected: {tracked_face.track_id} (Face: {tracked_face.face_id})")
            
            elif event_type == 'exit' and not tracked_face.exit_logged:
                tracked_face.exit_logged = True
                self.total_exits += 1
                events.append((tracked_face, 'exit'))
                self.logger.info(f"Exit detected: {tracked_face.track_id} (Face: {tracked_face.face_id})")
        
        return events
    
    def visualize_tracking(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw tracking information on frame
        
        Args:
            frame: Input frame
            
        Returns:
            Frame with tracking visualization
        """
        vis_frame = frame.copy()
        
        # Draw entry/exit zones
        for zone_name, (x1, y1, x2, y2) in self.entry_zones.items():
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
            cv2.putText(vis_frame, f"{zone_name} zone", (x1 + 5, y1 + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # Draw tracked faces
        for tracked_face in self.tracked_faces.values():
            if tracked_face.disappeared_frames > 0:
                continue
            
            x1, y1, x2, y2 = tracked_face.bbox
            
            # Choose color based on status
            if tracked_face.face_id:
                color = (0, 255, 0)  # Green for recognized
                label = f"ID: {tracked_face.face_id}"
            else:
                color = (0, 0, 255)  # Red for unrecognized
                label = f"Track: {tracked_face.track_id}"
            
            # Draw bounding box
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw trajectory
            if len(tracked_face.trajectory) > 1:
                trajectory_points = np.array(tracked_face.trajectory, dtype=np.int32)
                cv2.polylines(vis_frame, [trajectory_points], False, color, 2)
            
            # Draw label
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(vis_frame, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(vis_frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Draw status indicators
            status_y = y2 + 20
            if tracked_face.entry_logged:
                cv2.putText(vis_frame, "ENTERED", (x1, status_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            if tracked_face.exit_logged:
                cv2.putText(vis_frame, "EXITED", (x1, status_y + 15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        
        # Draw statistics
        stats_text = [
            f"Unique Visitors: {len(self.unique_visitors)}",
            f"Total Entries: {self.total_entries}",
            f"Total Exits: {self.total_exits}",
            f"Currently Inside: {self.total_entries - self.total_exits}",
            f"Active Tracks: {len([t for t in self.tracked_faces.values() if t.disappeared_frames == 0])}"
        ]
        
        for i, text in enumerate(stats_text):
            cv2.putText(vis_frame, text, (10, 30 + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(vis_frame, text, (10, 30 + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        
        return vis_frame
    
    def get_statistics(self) -> dict:
        """
        Get tracking statistics
        
        Returns:
            Dictionary with tracking statistics
        """
        active_tracks = len([t for t in self.tracked_faces.values() if t.disappeared_frames == 0])
        
        return {
            'unique_visitors': len(self.unique_visitors),
            'total_entries': self.total_entries,
            'total_exits': self.total_exits,
            'currently_inside': self.total_entries - self.total_exits,
            'active_tracks': active_tracks,
            'total_tracks_created': self.next_track_id - 1,
            'frame_number': self.frame_number
        }
