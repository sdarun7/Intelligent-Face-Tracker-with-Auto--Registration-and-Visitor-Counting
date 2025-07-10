"""
Face Recognition Module using InsightFace
Handles face embedding generation and recognition
"""

import numpy as np
import cv2
import logging
from typing import Optional, Tuple
import pickle
import os
import sys
from sklearn.metrics.pairwise import cosine_similarity

# Add parent directory to path to import mock models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import insightface
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False
    logging.warning("InsightFace not available, using mock recognizer")

from mock_ml_models import MockInsightFaceRecognizer


class FaceRecognizer:
    """InsightFace-based face recognizer"""
    
    def __init__(self, config: dict):
        """
        Initialize the face recognizer
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Recognition parameters
        self.similarity_threshold = config['recognition']['similarity_threshold']
        self.registration_threshold = config['recognition']['registration_threshold']
        self.embedding_dimension = config['recognition']['embedding_dimension']
        self.model_name = config['recognition']['model_name']
        
        # Initialize model (InsightFace or mock)
        self.app = None
        self._load_model()
        
        # Storage for known face embeddings
        self.known_embeddings = {}  # {face_id: embedding}
        self.embedding_cache_path = "data/embeddings_cache.pkl"
        self._load_embeddings_cache()
        
        self.logger.info("Face recognizer initialized successfully")
    
    def _load_model(self):
        """Load the face recognition model"""
        try:
            if INSIGHTFACE_AVAILABLE:
                # Initialize FaceAnalysis
                self.app = FaceAnalysis(name=self.model_name, providers=['CPUExecutionProvider'])
                self.app.prepare(ctx_id=0, det_size=(640, 640))
                self.logger.info(f"InsightFace model loaded: {self.model_name}")
            else:
                # Use mock face recognizer
                self.app = MockInsightFaceRecognizer(self.similarity_threshold)
                self.logger.info("Mock face recognizer loaded for demo")
            
        except Exception as e:
            self.logger.warning(f"Failed to load InsightFace model: {str(e)}, using mock recognizer")
            self.app = MockInsightFaceRecognizer(self.similarity_threshold)
    
    def _load_embeddings_cache(self):
        """Load cached embeddings from disk"""
        if os.path.exists(self.embedding_cache_path):
            try:
                with open(self.embedding_cache_path, 'rb') as f:
                    self.known_embeddings = pickle.load(f)
                self.logger.info(f"Loaded {len(self.known_embeddings)} cached embeddings")
            except Exception as e:
                self.logger.warning(f"Failed to load embeddings cache: {str(e)}")
                self.known_embeddings = {}
    
    def _save_embeddings_cache(self):
        """Save embeddings cache to disk"""
        try:
            os.makedirs(os.path.dirname(self.embedding_cache_path), exist_ok=True)
            with open(self.embedding_cache_path, 'wb') as f:
                pickle.dump(self.known_embeddings, f)
            self.logger.debug("Embeddings cache saved")
        except Exception as e:
            self.logger.error(f"Failed to save embeddings cache: {str(e)}")
    
    def extract_embedding(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract face embedding from face image
        
        Args:
            face_image: Cropped face image
            
        Returns:
            Face embedding vector or None if extraction fails
        """
        if self.app is None:
            self.logger.error("Face recognition model not loaded")
            return None
        
        # Check if using mock model
        if isinstance(self.app, MockInsightFaceRecognizer):
            return self.app.extract_embedding(face_image)
        
        try:
            # Ensure image is in correct format
            if len(face_image.shape) == 3 and face_image.shape[2] == 3:
                # Convert BGR to RGB if needed
                face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            else:
                face_rgb = face_image
            
            # Get face analysis
            faces = self.app.get(face_rgb)
            
            if len(faces) == 0:
                self.logger.debug("No face detected in crop")
                return None
            
            # Use the first (and likely only) detected face
            face = faces[0]
            embedding = face.embedding
            
            # Normalize embedding
            embedding = embedding / np.linalg.norm(embedding)
            
            self.logger.debug(f"Extracted embedding with shape: {embedding.shape}")
            return embedding
            
        except Exception as e:
            self.logger.error(f"Error extracting embedding: {str(e)}")
            return None
    
    def recognize_face(self, face_embedding: np.ndarray) -> Tuple[Optional[str], float]:
        """
        Recognize face from embedding
        
        Args:
            face_embedding: Face embedding vector
            
        Returns:
            Tuple of (face_id, confidence) or (None, 0.0) if not recognized
        """
        # Check if using mock model
        if isinstance(self.app, MockInsightFaceRecognizer):
            return self.app.recognize_face(face_embedding)
        
        if len(self.known_embeddings) == 0:
            return None, 0.0
        
        try:
            best_match_id = None
            best_similarity = 0.0
            
            # Compare with all known embeddings
            for face_id, known_embedding in self.known_embeddings.items():
                # Calculate cosine similarity
                similarity = cosine_similarity(
                    face_embedding.reshape(1, -1), 
                    known_embedding.reshape(1, -1)
                )[0][0]
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match_id = face_id
            
            # Check if similarity is above threshold
            if best_similarity >= self.similarity_threshold:
                self.logger.debug(f"Recognized face {best_match_id} with similarity {best_similarity:.3f}")
                return best_match_id, best_similarity
            else:
                self.logger.debug(f"No match found. Best similarity: {best_similarity:.3f}")
                return None, best_similarity
                
        except Exception as e:
            self.logger.error(f"Error in face recognition: {str(e)}")
            return None, 0.0
    
    def register_face(self, face_id: str, face_embedding: np.ndarray) -> bool:
        """
        Register a new face embedding
        
        Args:
            face_id: Unique face identifier
            face_embedding: Face embedding vector
            
        Returns:
            True if registration successful, False otherwise
        """
        try:
            # Check if using mock model
            if isinstance(self.app, MockInsightFaceRecognizer):
                return self.app.register_face(face_id, face_embedding)
            
            # Check if face_id already exists
            if face_id in self.known_embeddings:
                self.logger.warning(f"Face ID {face_id} already exists")
                return False
            
            # Validate embedding
            if face_embedding is None or len(face_embedding) == 0:
                self.logger.error("Invalid embedding for registration")
                return False
            
            # Store embedding
            self.known_embeddings[face_id] = face_embedding
            
            # Save to cache
            self._save_embeddings_cache()
            
            self.logger.info(f"Registered new face: {face_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error registering face {face_id}: {str(e)}")
            return False
    
    def update_face_embedding(self, face_id: str, new_embedding: np.ndarray) -> bool:
        """
        Update existing face embedding (for improved recognition)
        
        Args:
            face_id: Face identifier
            new_embedding: New embedding to add/average
            
        Returns:
            True if update successful, False otherwise
        """
        try:
            if face_id not in self.known_embeddings:
                self.logger.warning(f"Face ID {face_id} not found for update")
                return False
            
            # Simple averaging of embeddings for now
            current_embedding = self.known_embeddings[face_id]
            updated_embedding = (current_embedding + new_embedding) / 2
            
            # Normalize
            updated_embedding = updated_embedding / np.linalg.norm(updated_embedding)
            
            self.known_embeddings[face_id] = updated_embedding
            self._save_embeddings_cache()
            
            self.logger.debug(f"Updated embedding for face {face_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating embedding for {face_id}: {str(e)}")
            return False
    
    def should_register_face(self, face_embedding: np.ndarray) -> bool:
        """
        Determine if face should be registered as new
        
        Args:
            face_embedding: Face embedding vector
            
        Returns:
            True if face should be registered, False otherwise
        """
        if len(self.known_embeddings) == 0:
            return True
        
        # Check similarity with all known faces
        _, best_similarity = self.recognize_face(face_embedding)
        
        # Register if similarity is below registration threshold
        return best_similarity < self.registration_threshold
    
    def get_known_faces_count(self) -> int:
        """
        Get number of registered faces
        
        Returns:
            Number of known faces
        """
        return len(self.known_embeddings)
    
    def remove_face(self, face_id: str) -> bool:
        """
        Remove a registered face
        
        Args:
            face_id: Face identifier to remove
            
        Returns:
            True if removal successful, False otherwise
        """
        try:
            if face_id in self.known_embeddings:
                del self.known_embeddings[face_id]
                self._save_embeddings_cache()
                self.logger.info(f"Removed face: {face_id}")
                return True
            else:
                self.logger.warning(f"Face ID {face_id} not found for removal")
                return False
                
        except Exception as e:
            self.logger.error(f"Error removing face {face_id}: {str(e)}")
            return False
    
    def get_embedding_statistics(self) -> dict:
        """
        Get statistics about stored embeddings
        
        Returns:
            Dictionary with embedding statistics
        """
        if len(self.known_embeddings) == 0:
            return {"count": 0, "dimension": 0}
        
        embeddings = list(self.known_embeddings.values())
        embedding_array = np.array(embeddings)
        
        return {
            "count": len(embeddings),
            "dimension": embedding_array.shape[1] if len(embeddings) > 0 else 0,
            "mean_norm": np.mean([np.linalg.norm(emb) for emb in embeddings]),
            "registered_faces": list(self.known_embeddings.keys())
        }
