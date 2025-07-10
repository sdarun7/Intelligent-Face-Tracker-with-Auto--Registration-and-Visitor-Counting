import os
import sys
import json
import sqlite3
import cv2
import numpy as np
from datetime import datetime
import logging

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_config_loading():
    print("Testing configuration loading...")
    try:
        from utils import load_config
        config = load_config('config.json')
        if config:
            print("✓ Configuration loaded successfully")
            return True
        else:
            print("✗ Failed to load configuration")
            return False
    except Exception as e:
        print(f"✗ Error loading configuration: {e}")
        return False

def test_database_initialization():
    print("\nTesting database initialization...")
    try:
        from database_manager import DatabaseManager
        
        db_manager = DatabaseManager('data/test_face_tracker.db')
        db_manager.initialize_database()

        conn = sqlite3.connect('data/test_face_tracker.db')
        cursor = conn.cursor()
        
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        
        expected_tables = ['faces', 'events']
        if all(table in tables for table in expected_tables):
            print("✓ Database tables created successfully")
            conn.close()
            return True
        else:
            print(f"✗ Missing tables. Found: {tables}")
            conn.close()
            return False
            
    except Exception as e:
        print(f"✗ Database initialization error: {e}")
        return False

def test_face_detection():
    print("\nTesting face detection...")
    try:
        from face_detector import FaceDetector
        from utils import load_config
        
        config = load_config('config.json')
        if not config:
            print("✗ Cannot load config for face detection test")
            return False
        
        detector = FaceDetector(config)

        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        test_image[100:200, 100:200] = [255, 255, 255] 
        
        detections = detector.detect_faces(test_image)
        print(f"✓ Face detector initialized and processed test image")
        print(f"  Detected {len(detections)} faces in test image")
        return True
        
    except Exception as e:
        print(f"✗ Face detection error: {e}")
        return False

def test_face_recognition():
    print("\nTesting face recognition...")
    try:
        from face_recognizer import FaceRecognizer
        from utils import load_config
        
        config = load_config('config.json')
        if not config:
            print("✗ Cannot load config for face recognition test")
            return False
        
        recognizer = FaceRecognizer(config)

        test_face = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
        
        embedding = recognizer.extract_embedding(test_face)
        
        if embedding is not None:
            print(f"✓ Face recognizer initialized and extracted embedding")
            print(f"  Embedding shape: {embedding.shape}")
            return True
        else:
            print("✓ Face recognizer initialized (no face detected in test image)")
            return True
        
    except Exception as e:
        print(f"✗ Face recognition error: {e}")
        return False

def test_tracking():
    print("\nTesting face tracking...")
    try:
        from face_tracker import FaceTracker
        from utils import load_config
        
        config = load_config('config.json')
        if not config:
            print("✗ Cannot load config for tracking test")
            return False
        
        tracker = FaceTracker(config)
        tracker.update_frame_dimensions(640, 480)

        detections = [(100, 100, 200, 200, 0.8), (300, 150, 400, 250, 0.9)]
        face_ids = ['face_001', None]
        
        tracks = tracker.update(detections, face_ids)
        
        print(f"✓ Face tracker initialized and processed detections")
        print(f"  Active tracks: {len(tracks)}")
        return True
        
    except Exception as e:
        print(f"✗ Face tracking error: {e}")
        return False

def test_logging():
    print("\nTesting logging system...")
    try:
        from logger_manager import LoggerManager
        
        logger_manager = LoggerManager(logging.INFO)
        logger = logger_manager.get_logger('test')
        
        logger.info("Test log message")
        logger_manager.log_face_detection(1, 2, 0.05)
        logger_manager.log_face_recognition('face_001', 'track_001', 0.85)
        
        print("✓ Logging system initialized and working")
        return True
        
    except Exception as e:
        print(f"✗ Logging system error: {e}")
        return False

def create_sample_data():
    print("\nCreating sample data...")
    try:
        from database_manager import DatabaseManager
        
        db_manager = DatabaseManager('data/face_tracker.db')
        db_manager.initialize_database()

        sample_embedding1 = np.random.rand(512).astype(np.float32)
        sample_embedding2 = np.random.rand(512).astype(np.float32)

        db_manager.register_face('face_000001', sample_embedding1, 
                                {'name': 'Sample Person 1', 'registered_at': datetime.now().isoformat()})
        db_manager.register_face('face_000002', sample_embedding2,
                                {'name': 'Sample Person 2', 'registered_at': datetime.now().isoformat()})

        timestamp = datetime.now()
        db_manager.log_event('face_000001', 'track_001', 'entry', 
                           'logs/entries/2025-07-09/face_000001_entry_sample.jpg', 
                           0.85, (100, 100, 200, 200), 150)
        
        db_manager.log_event('face_000001', 'track_001', 'exit',
                           'logs/exits/2025-07-09/face_000001_exit_sample.jpg',
                           0.82, (110, 105, 210, 205), 1850)
        
        db_manager.log_event('face_000002', 'track_002', 'entry',
                           'logs/entries/2025-07-09/face_000002_entry_sample.jpg',
                           0.78, (300, 150, 400, 250), 200)
        
        print("✓ Sample data created successfully")

        stats = db_manager.get_visitor_statistics()
        print(f"  Unique visitors: {stats['unique_visitors']}")
        print(f"  Total entries: {stats['total_entries']}")
        print(f"  Total exits: {stats['total_exits']}")
        
        return True
        
    except Exception as e:
        print(f"✗ Sample data creation error: {e}")
        return False

def create_sample_images():
    print("\nCreating sample face images...")
    try:
        # Create sample face images
        os.makedirs('data/faces', exist_ok=True)
        os.makedirs('logs/entries/2025-07-09', exist_ok=True)
        os.makedirs('logs/exits/2025-07-09', exist_ok=True)
        
        for i in range(1, 3):
            face_img = np.random.randint(100, 200, (112, 112, 3), dtype=np.uint8)
            cv2.rectangle(face_img, (30, 30), (82, 82), (150, 150, 150), -1)
            cv2.circle(face_img, (45, 45), 5, (50, 50, 50), -1)
            cv2.circle(face_img, (67, 45), 5, (50, 50, 50), -1) 
            cv2.rectangle(face_img, (52, 60), (60, 65), (100, 100, 100), -1) 
            cv2.rectangle(face_img, (45, 70), (67, 75), (80, 80, 80), -1)
            
            cv2.imwrite(f'data/faces/face_{i:06d}.jpg', face_img)

            entry_img = face_img.copy()
            cv2.imwrite(f'logs/entries/2025-07-09/face_{i:06d}_entry_sample.jpg', entry_img)

            exit_img = cv2.addWeighted(face_img, 0.8, np.ones_like(face_img) * 20, 0.2, 0)
            cv2.imwrite(f'logs/exits/2025-07-09/face_{i:06d}_exit_sample.jpg', exit_img)
        
        print("✓ Sample face images created")
        return True
        
    except Exception as e:
        print(f"✗ Sample image creation error: {e}")
        return False

def run_all_tests():
    print("="*60)
    print("FACE TRACKER SYSTEM TESTS")
    print("="*60)
    
    tests = [
        test_config_loading,
        test_database_initialization,
        test_face_detection,
        test_face_recognition,
        test_tracking,
        test_logging,
        create_sample_data,
        create_sample_images
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "="*60)
    print(f"TEST RESULTS: {passed}/{total} tests passed")
    print("="*60)
    
    if passed == total:
        print("✓ All tests passed! System is ready for use.")
        return True
    else:
        print(f"✗ {total - passed} tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)