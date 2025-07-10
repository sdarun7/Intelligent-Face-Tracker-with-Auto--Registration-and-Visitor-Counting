import sqlite3
import logging
import json
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import os
import threading


class DatabaseManager:
    
    def __init__(self, db_path: str):
        
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        self._lock = threading.Lock()

        db_dir = os.path.dirname(db_path)
        if db_dir: 
            os.makedirs(db_dir, exist_ok=True)
        
        self.logger.info(f"Database manager initialized with path: {db_path}")
    
    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row  # Enable column access by name
        conn.execute("PRAGMA foreign_keys = ON")  # Enable foreign keys
        return conn
    
    def initialize_database(self):
        with self._lock:
            try:
                conn = self._get_connection()
                cursor = conn.cursor()

                cursor.execute()

                cursor.execute()

                cursor.execute()
                cursor.execute()
                cursor.execute()
                cursor.execute()
                
                conn.commit()
                conn.close()
                
                self.logger.info("Database initialized successfully")
                
            except Exception as e:
                self.logger.error(f"Error initializing database: {str(e)}")
                raise
    
    def register_face(self, face_id: str, embedding: np.ndarray, metadata: dict = None) -> bool:
        
        with self._lock:
            try:
                conn = self._get_connection()
                cursor = conn.cursor()
                
                cursor.execute("SELECT id FROM faces WHERE id = ?", (face_id,))
                if cursor.fetchone():
                    self.logger.warning(f"Face {face_id} already exists in database")
                    conn.close()
                    return False

                embedding_blob = embedding.tobytes()
                metadata_json = json.dumps(metadata) if metadata else None
                
                cursor.execute((face_id, embedding_blob, metadata_json))
                
                conn.commit()
                conn.close()
                
                self.logger.info(f"Registered face {face_id} in database")
                return True
                
            except Exception as e:
                self.logger.error(f"Error registering face {face_id}: {str(e)}")
                return False
    
    def get_face_embedding(self, face_id: str) -> Optional[np.ndarray]:
        
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute("SELECT embedding FROM faces WHERE id = ?", (face_id,))
            result = cursor.fetchone()
            conn.close()
            
            if result:
                embedding_blob = result[0]
                embedding = np.frombuffer(embedding_blob, dtype=np.float32)
                return embedding
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting embedding for face {face_id}: {str(e)}")
            return None
    
    def log_event(self, face_id: Optional[str], track_id: str, event_type: str,
                  image_path: str, confidence: float, bbox: Tuple[int, int, int, int],
                  frame_number: int, metadata: dict = None) -> bool:
       
        with self._lock:
            try:
                conn = self._get_connection()
                cursor = conn.cursor()
                
                x1, y1, x2, y2 = bbox
                metadata_json = json.dumps(metadata) if metadata else None

                cursor.execute((face_id, track_id, event_type, image_path, confidence,
                      x1, y1, x2, y2, frame_number, metadata_json))

                if face_id:
                    if event_type == 'entry':
                        cursor.execute((face_id,))
                    elif event_type == 'exit':
                        cursor.execute((face_id,))
                
                conn.commit()
                conn.close()
                
                self.logger.debug(f"Logged {event_type} event for face {face_id}, track {track_id}")
                return True
                
            except Exception as e:
                self.logger.error(f"Error logging event: {str(e)}")
                return False
    
    def get_face_events(self, face_id: str, limit: int = None) -> List[Dict]:
        
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            if limit:
                query += f" LIMIT {limit}"
            
            cursor.execute(query, (face_id,))
            results = cursor.fetchall()
            conn.close()
            
            events = []
            for row in results:
                event = dict(row)
                if event['metadata']:
                    event['metadata'] = json.loads(event['metadata'])
                events.append(event)
            
            return events
            
        except Exception as e:
            self.logger.error(f"Error getting events for face {face_id}: {str(e)}")
            return []
    
    def get_all_faces(self) -> List[Dict]:
        
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute()
            
            results = cursor.fetchall()
            conn.close()
            
            faces = []
            for row in results:
                face = dict(row)
                if face['metadata']:
                    face['metadata'] = json.loads(face['metadata'])
                faces.append(face)
            
            return faces
            
        except Exception as e:
            self.logger.error(f"Error getting all faces: {str(e)}")
            return []
    
    def get_visitor_statistics(self) -> Dict:
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
                    
            cursor.execute()
            unique_visitors = cursor.fetchone()[0]
                        
            cursor.execute()
            total_entries = cursor.fetchone()[0]
                        
            cursor.execute()
            total_exits = cursor.fetchone()[0]
                        
            cursor.execute()
            today_entries = cursor.fetchone()[0]
            
            cursor.execute()
            today_exits = cursor.fetchone()[0]
            
            cursor.execute()
            
            recent_activity = {row[0]: row[1] for row in cursor.fetchall()}
            
            conn.close()
            
            return {
                'unique_visitors': unique_visitors,
                'total_entries': total_entries,
                'total_exits': total_exits,
                'currently_inside': total_entries - total_exits,
                'today_entries': today_entries,
                'today_exits': today_exits,
                'today_currently_inside': today_entries - today_exits,
                'recent_entries': recent_activity.get('entry', 0),
                'recent_exits': recent_activity.get('exit', 0)
            }
            
        except Exception as e:
            self.logger.error(f"Error getting visitor statistics: {str(e)}")
            return {
                'unique_visitors': 0,
                'total_entries': 0,
                'total_exits': 0,
                'currently_inside': 0,
                'today_entries': 0,
                'today_exits': 0,
                'today_currently_inside': 0,
                'recent_entries': 0,
                'recent_exits': 0
            }
    
    def get_events_by_date(self, date: str = None, event_type: str = None) -> List[Dict]:
        
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            query = "SELECT * FROM events WHERE 1=1"
            params = []
            
            if date:
                query += " AND DATE(timestamp) = ?"
                params.append(date)
            else:
                query += " AND DATE(timestamp) = DATE('now')"
            
            if event_type:
                query += " AND event_type = ?"
                params.append(event_type)
            
            query += " ORDER BY timestamp DESC"
            
            cursor.execute(query, params)
            results = cursor.fetchall()
            conn.close()
            
            events = []
            for row in results:
                event = dict(row)
                if event['metadata']:
                    event['metadata'] = json.loads(event['metadata'])
                events.append(event)
            
            return events
            
        except Exception as e:
            self.logger.error(f"Error getting events by date: {str(e)}")
            return []
    
    def cleanup_old_data(self, days_to_keep: int = 30) -> bool:
        
        with self._lock:
            try:
                conn = self._get_connection()
                cursor = conn.cursor()

                cursor.execute("""
                    DELETE FROM events 
                    WHERE timestamp < datetime('now', '-{} days')
                """.format(days_to_keep))
                
                deleted_events = cursor.rowcount

                cursor.execute()
                
                deleted_faces = cursor.rowcount
                
                conn.commit()
                conn.close()
                
                self.logger.info(f"Cleanup completed: deleted {deleted_events} events and {deleted_faces} faces")
                return True
                
            except Exception as e:
                self.logger.error(f"Error during cleanup: {str(e)}")
                return False
    
    def export_data(self, output_path: str, date_from: str = None, date_to: str = None) -> bool:
        
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute("SELECT id, first_seen, total_entries, total_exits, last_seen, created_at FROM faces")
            faces = [dict(row) for row in cursor.fetchall()]
            
            query = "SELECT * FROM events WHERE 1=1"
            params = []
            
            if date_from:
                query += " AND DATE(timestamp) >= ?"
                params.append(date_from)
            
            if date_to:
                query += " AND DATE(timestamp) <= ?"
                params.append(date_to)
            
            query += " ORDER BY timestamp"
            
            cursor.execute(query, params)
            events = [dict(row) for row in cursor.fetchall()]
            
            conn.close()

            export_data = {
                'export_timestamp': datetime.now().isoformat(),
                'date_range': {
                    'from': date_from,
                    'to': date_to
                },
                'statistics': self.get_visitor_statistics(),
                'faces': faces,
                'events': events
            }
            
            # Write to file
            with open(output_path, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            self.logger.info(f"Data exported to {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error exporting data: {str(e)}")
            return False
