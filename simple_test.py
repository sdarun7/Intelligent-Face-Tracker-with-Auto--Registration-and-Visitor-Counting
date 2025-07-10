import sys
import os
sys.path.append('src')

try:
    import cv2
    print("✓ OpenCV imported successfully")
except ImportError as e:
    print(f"✗ OpenCV import failed: {e}")

try:
    import numpy as np
    print("✓ NumPy imported successfully")
except ImportError as e:
    print(f"✗ NumPy import failed: {e}")

try:
    from sklearn.metrics.pairwise import cosine_similarity
    print("✓ scikit-learn imported successfully")
except ImportError as e:
    print(f"✗ scikit-learn import failed: {e}")

try:
    from mock_ml_models import MockYOLOFaceDetector, MockInsightFaceRecognizer
    print("✓ Mock models imported successfully")
except ImportError as e:
    print(f"✗ Mock models import failed: {e}")

try:
    from utils import load_config
    config = load_config('config.json')
    if config:
        print("✓ Configuration loaded successfully")
    else:
        print("✗ Configuration loading failed")
except Exception as e:
    print(f"✗ Configuration loading error: {e}")

try:
    from database_manager import DatabaseManager
    db_manager = DatabaseManager('test.db')
    db_manager.initialize_database()
    print("✓ Database manager initialized successfully")
except Exception as e:
    print(f"✗ Database manager failed: {e}")

try:
    from logger_manager import LoggerManager
    logger_manager = LoggerManager()
    print("✓ Logger manager initialized successfully")
except Exception as e:
    print(f"✗ Logger manager failed: {e}")

print("\nTesting mock models...")

try:
    detector = MockYOLOFaceDetector()
    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    detections = detector.detect_faces(test_frame)
    print(f"✓ Mock face detector works: {len(detections)} detections")
except Exception as e:
    print(f"✗ Mock face detector failed: {e}")

try:
    recognizer = MockInsightFaceRecognizer()
    test_face = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    embedding = recognizer.extract_embedding(test_face)
    print(f"✓ Mock face recognizer works: embedding shape {embedding.shape if embedding is not None else 'None'}")
except Exception as e:
    print(f"✗ Mock face recognizer failed: {e}")

print("\nAll tests completed!")