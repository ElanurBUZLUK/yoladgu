"""
ML-based Backend Selector Service

This service learns which vector index backend performs best for different query patterns
and automatically selects the optimal backend for each search request.
"""

import os
import pickle
import joblib
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import logging

logger = logging.getLogger(__name__)


class BackendSelector:
    """
    Machine learning model for selecting the best backend based on query characteristics.
    
    Features:
    - Query vector properties
    - Search parameters (k, filters)
    - Index statistics
    - Historical performance data
    """
    
    def __init__(self, model_type: str = "random_forest"):
        self.model_type = model_type
        self.model = None
        self.feature_names = None
        self.backend_labels = None
        self.is_trained = False
        
        # Performance tracking
        self.accuracy_history = []
        self.prediction_counts = {}
        
    def _create_model(self):
        """Create and configure the ML model."""
        if self.model_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def fit(self, X: List[Dict[str, Any]], y: List[str]) -> 'BackendSelector':
        """
        Train the backend selector model.
        
        Args:
            X: List of feature dictionaries
            y: List of backend labels (best performing backend for each query)
            
        Returns:
            Self for chaining
        """
        try:
            if not X or not y:
                logger.warning("No training data provided")
                return self
            
            # Convert features to numpy array
            X_array, self.feature_names = self._features_to_array(X)
            y_array = np.array(y)
            
            # Store backend labels
            self.backend_labels = sorted(set(y))
            
            # Create and train model
            self._create_model()
            
            # Split data for validation
            X_train, X_test, y_train, y_test = train_test_split(
                X_array, y_array, test_size=0.2, random_state=42, stratify=y_array
            )
            
            # Train model
            self.model.fit(X_train, y_train)
            
            # Evaluate performance
            y_pred = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            self.accuracy_history.append(accuracy)
            
            logger.info(f"Backend selector trained successfully. Accuracy: {accuracy:.3f}")
            logger.info(f"Classification report:\\n{classification_report(y_test, y_pred)}")
            
            self.is_trained = True
            
            return self
            
        except Exception as e:
            logger.error(f"Error training backend selector: {e}")
            return self
    
    def predict_backend(self, features: Dict[str, Any]) -> Optional[str]:
        """
        Predict the best backend for a given query.
        
        Args:
            features: Query features dictionary
            
        Returns:
            Predicted backend name or None if model not trained
        """
        try:
            if not self.is_trained or self.model is None:
                logger.warning("Model not trained yet")
                return None
            
            # Convert features to array
            X_array, _ = self._features_to_array([features])
            
            # Make prediction
            prediction = self.model.predict(X_array)[0]
            probability = np.max(self.model.predict_proba(X_array))
            
            # Track predictions
            self.prediction_counts[prediction] = self.prediction_counts.get(prediction, 0) + 1
            
            logger.debug(f"Backend prediction: {prediction} (confidence: {probability:.3f})")
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error predicting backend: {e}")
            return None
    
    def predict_proba(self, features: Dict[str, Any]) -> Optional[Dict[str, float]]:
        """
        Get prediction probabilities for all backends.
        
        Args:
            features: Query features dictionary
            
        Returns:
            Dictionary of backend probabilities or None if model not trained
        """
        try:
            if not self.is_trained or self.model is None:
                return None
            
            # Convert features to array
            X_array, _ = self._features_to_array([features])
            
            # Get probabilities
            probabilities = self.model.predict_proba(X_array)[0]
            
            # Create probability dictionary
            prob_dict = {
                backend: prob for backend, prob in zip(self.backend_labels, probabilities)
            }
            
            return prob_dict
            
        except Exception as e:
            logger.error(f"Error getting prediction probabilities: {e}")
            return None
    
    def _features_to_array(self, features_list: List[Dict[str, Any]]) -> Tuple[np.ndarray, List[str]]:
        """
        Convert feature dictionaries to numpy array.
        
        Args:
            features_list: List of feature dictionaries
            
        Returns:
            Tuple of (feature array, feature names)
        """
        if not features_list:
            return np.array([]), []
        
        # Define feature order (must be consistent)
        feature_order = [
            'query_norm', 'query_std', 'query_max', 'query_min',
            'k_value', 'has_filters', 'filter_count',
            'total_items', 'avg_items_per_backend',
            'hnsw_available', 'faiss_available', 'qdrant_available',
            'recent_hnsw_performance', 'recent_faiss_performance', 'recent_qdrant_performance'
        ]
        
        # Initialize feature array
        X = np.zeros((len(features_list), len(feature_order)))
        
        # Fill feature array
        for i, features in enumerate(features_list):
            for j, feature_name in enumerate(feature_order):
                X[i, j] = features.get(feature_name, 0.0)
        
        return X, feature_order
    
    def save(self, filepath: str) -> bool:
        """Save the trained model to disk."""
        try:
            if not self.is_trained:
                logger.warning("Cannot save untrained model")
                return False
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Save model
            joblib.dump(self, filepath)
            logger.info(f"Model saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False
    
    @classmethod
    def load(cls, filepath: str) -> Optional['BackendSelector']:
        """Load a trained model from disk."""
        try:
            if not os.path.exists(filepath):
                logger.warning(f"Model file not found: {filepath}")
                return None
            
            # Load model
            model = joblib.load(filepath)
            logger.info(f"Model loaded from {filepath}")
            return model
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return None
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for the model."""
        return {
            "is_trained": self.is_trained,
            "model_type": self.model_type,
            "accuracy_history": self.accuracy_history,
            "current_accuracy": self.accuracy_history[-1] if self.accuracy_history else None,
            "prediction_counts": self.prediction_counts,
            "backend_labels": self.backend_labels,
            "feature_count": len(self.feature_names) if self.feature_names else 0
        }


async def collect_training_examples(
    vector_index_manager,
    encoder=None,
    n: int = 200,
    k: int = 10
) -> List[Dict[str, Any]]:
    """
    Collect training examples for the backend selector.
    
    This function generates synthetic queries and tests each backend's performance
    to create training data for the ML model.
    
    Args:
        vector_index_manager: Vector index manager instance
        encoder: Text encoder for generating query vectors
        n: Number of training examples to generate
        k: Number of results to request
        
    Returns:
        List of training examples with features and labels
    """
    try:
        if not vector_index_manager.backends:
            logger.warning("No backends available for training")
            return []
        
        # Check if FAISS is available (needed for training data generation)
        if "faiss" not in vector_index_manager.backends:
            logger.warning("FAISS backend required for training data generation")
            return []
        
        training_examples = []
        
        # Generate synthetic training data
        for i in range(n):
            try:
                # Generate random query vector
                query_vector = np.random.randn(1, vector_index_manager.vector_size).astype(np.float32)
                query_vector = query_vector / np.linalg.norm(query_vector)  # Normalize
                
                # Generate random features
                features = {
                    'query_norm': np.linalg.norm(query_vector),
                    'query_std': np.std(query_vector),
                    'query_max': np.max(query_vector),
                    'query_min': np.min(query_vector),
                    'k_value': np.random.randint(1, 51),  # Random k value
                    'has_filters': np.random.choice([0, 1], p=[0.7, 0.3]),  # 30% have filters
                    'filter_count': np.random.randint(0, 4) if np.random.random() < 0.3 else 0,
                    'total_items': np.random.randint(1000, 100000),
                    'avg_items_per_backend': np.random.randint(500, 50000),
                    'hnsw_available': 1 if "hnsw" in vector_index_manager.backends else 0,
                    'faiss_available': 1 if "faiss" in vector_index_manager.backends else 0,
                    'qdrant_available': 1 if "qdrant" in vector_index_manager.backends else 0,
                    'recent_hnsw_performance': np.random.uniform(0.5, 2.0),
                    'recent_faiss_performance': np.random.uniform(0.1, 1.0),
                    'recent_qdrant_performance': np.random.uniform(0.8, 3.0)
                }
                
                # Test each backend's performance
                backend_performance = {}
                
                for backend_name in vector_index_manager.backends:
                    try:
                        start_time = vector_index_manager._get_time()
                        results = await vector_index_manager.search(
                            query_vector, k, backend_name=backend_name
                        )
                        end_time = vector_index_manager._get_time()
                        
                        response_time = (end_time - start_time) * 1000
                        result_count = len(results)
                        
                        # Calculate performance score (lower is better)
                        performance_score = response_time
                        if result_count < k:
                            performance_score *= 2  # Penalty for fewer results
                        
                        backend_performance[backend_name] = performance_score
                        
                    except Exception as e:
                        logger.debug(f"Error testing {backend_name}: {e}")
                        backend_performance[backend_name] = float('inf')
                
                # Find best performing backend
                if backend_performance:
                    best_backend = min(backend_performance, key=backend_performance.get)
                    
                    training_examples.append({
                        'X': features,
                        'y': best_backend,
                        'performance': backend_performance
                    })
                
            except Exception as e:
                logger.debug(f"Error generating training example {i}: {e}")
                continue
        
        logger.info(f"Generated {len(training_examples)} training examples")
        return training_examples
        
    except Exception as e:
        logger.error(f"Error collecting training examples: {e}")
        return []


async def train_backend_selector(
    vector_index_manager,
    n_examples: int = 200,
    save_path: str = "data/ml/backend_selector.joblib"
) -> Dict[str, Any]:
    """
    Train the backend selector model.
    
    Args:
        vector_index_manager: Vector index manager instance
        n_examples: Number of training examples to generate
        save_path: Path to save the trained model
        
    Returns:
        Training result dictionary
    """
    try:
        logger.info("Starting backend selector training...")
        
        # Collect training examples
        rows = await collect_training_examples(
            vector_index_manager, 
            encoder=None, 
            n=n_examples, 
            k=10
        )
        
        if not rows:
            return {
                "ok": False, 
                "msg": "FAISS gerekli veya veri yetersiz"
            }
        
        # Prepare training data
        X = [r["X"] for r in rows]
        y = [r["y"] for r in rows]
        
        # Train model
        selector = BackendSelector().fit(X, y)
        
        # Save model
        if save_path:
            os.makedirs("data/ml", exist_ok=True)
            selector.save(save_path)
        
        return {
            "ok": True,
            "n": len(X),
            "labels": sorted(set(y)),
            "accuracy": selector.accuracy_history[-1] if selector.accuracy_history else None
        }
        
    except Exception as e:
        logger.error(f"Error training backend selector: {e}")
        return {
            "ok": False,
            "msg": f"Training failed: {str(e)}"
        }
