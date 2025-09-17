"""
ML Backend Selector Training from Retrieval Logs
Trains a model to predict the best backend based on query features
"""

import asyncio
import json
import os
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import structlog

from app.db.session import get_db
from sqlalchemy import text

logger = structlog.get_logger()

class FeatureExtractor:
    """Extract features from query and context for backend selection"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
    def extract_query_features(self, query: str, user_id: str, filters: Dict[str, Any]) -> Dict[str, float]:
        """
        Extract features from query and context
        
        Args:
            query: Search query
            user_id: User ID
            filters: Search filters
            
        Returns:
            Dictionary of features
        """
        features = {}
        
        # Query-based features
        features['query_length'] = len(query)
        features['query_word_count'] = len(query.split())
        features['query_has_numbers'] = 1.0 if any(char.isdigit() for char in query) else 0.0
        features['query_has_special_chars'] = 1.0 if any(not char.isalnum() and char != ' ' for char in query) else 0.0
        features['query_avg_word_length'] = np.mean([len(word) for word in query.split()]) if query.split() else 0.0
        
        # Query complexity features
        features['query_entropy'] = self._calculate_entropy(query)
        features['query_has_questions'] = 1.0 if '?' in query else 0.0
        features['query_has_operators'] = 1.0 if any(op in query.lower() for op in ['and', 'or', 'not', '+', '-']) else 0.0
        
        # Filter-based features
        features['has_difficulty_filter'] = 1.0 if 'difficulty' in filters else 0.0
        features['has_subject_filter'] = 1.0 if 'subject' in filters else 0.0
        features['has_skill_filter'] = 1.0 if 'skills' in filters else 0.0
        features['filter_count'] = len(filters)
        
        # Time-based features (simplified)
        current_hour = datetime.now().hour
        features['is_business_hours'] = 1.0 if 9 <= current_hour <= 17 else 0.0
        features['is_weekend'] = 1.0 if datetime.now().weekday() >= 5 else 0.0
        
        # User-based features (simplified - in production, use actual user stats)
        features['user_id_hash'] = hash(user_id) % 1000 / 1000.0  # Normalized hash
        
        return features
    
    def _calculate_entropy(self, text: str) -> float:
        """Calculate Shannon entropy of text"""
        if not text:
            return 0.0
        
        # Count character frequencies
        char_counts = {}
        for char in text.lower():
            char_counts[char] = char_counts.get(char, 0) + 1
        
        # Calculate entropy
        total_chars = len(text)
        entropy = 0.0
        for count in char_counts.values():
            p = count / total_chars
            if p > 0:
                entropy -= p * np.log2(p)
        
        return entropy

class BackendSelectorTrainer:
    """Train ML model for backend selection"""
    
    def __init__(self):
        self.feature_extractor = FeatureExtractor()
        self.model = None
        self.feature_names = None
        self.backend_labels = ['faiss', 'hnsw', 'qdrant', 'error_aware']
        
    async def load_training_data(self, days: int = 7, min_samples: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load training data from retrieval logs
        
        Args:
            days: Number of days to look back
            min_samples: Minimum samples required per backend
            
        Returns:
            Tuple of (X, y) arrays
        """
        logger.info("Loading training data", days=days, min_samples=min_samples)
        
        try:
            async with get_db() as db:
                # Query retrieval logs
                query_sql = text("""
                    SELECT 
                        user_id,
                        query,
                        filters,
                        bm25_scores,
                        dense_scores,
                        durations,
                        selected_backend,
                        top_k,
                        timestamp
                    FROM retrieval_logs 
                    WHERE timestamp >= NOW() - INTERVAL :days DAY
                    AND selected_backend IS NOT NULL
                    AND durations IS NOT NULL
                    ORDER BY timestamp DESC
                """)
                
                result = await db.execute(query_sql, {"days": days})
                rows = result.fetchall()
                
                if not rows:
                    logger.warning("No retrieval logs found for training")
                    return np.array([]), np.array([])
                
                logger.info("Loaded retrieval logs", count=len(rows))
                
                # Process data
                features_list = []
                labels_list = []
                
                for row in rows:
                    try:
                        # Extract features
                        features = self.feature_extractor.extract_query_features(
                            query=row.query,
                            user_id=row.user_id,
                            filters=row.filters or {}
                        )
                        
                        # Add performance-based features
                        features.update(self._extract_performance_features(
                            durations=row.durations or {},
                            scores={
                                'bm25': row.bm25_scores or {},
                                'dense': row.dense_scores or {}
                            }
                        ))
                        
                        # Determine best backend (label)
                        best_backend = self._determine_best_backend(
                            durations=row.durations or {},
                            scores={
                                'bm25': row.bm25_scores or {},
                                'dense': row.dense_scores or {}
                            }
                        )
                        
                        if best_backend and best_backend in self.backend_labels:
                            features_list.append(features)
                            labels_list.append(best_backend)
                    
                    except Exception as e:
                        logger.warning("Failed to process row", error=str(e))
                        continue
                
                if not features_list:
                    logger.warning("No valid training samples found")
                    return np.array([]), np.array([])
                
                # Convert to arrays
                X = np.array([list(f.values()) for f in features_list])
                y = np.array(labels_list)
                
                # Check minimum samples per class
                unique_labels, counts = np.unique(y, return_counts=True)
                logger.info("Training data distribution", 
                           labels=dict(zip(unique_labels, counts)))
                
                # Filter classes with sufficient samples
                valid_classes = unique_labels[counts >= min_samples]
                if len(valid_classes) < 2:
                    logger.warning("Insufficient samples per class", 
                                 min_samples=min_samples, valid_classes=len(valid_classes))
                    return np.array([]), np.array([])
                
                # Filter data to valid classes
                mask = np.isin(y, valid_classes)
                X = X[mask]
                y = y[mask]
                
                logger.info("Final training data", samples=len(X), features=X.shape[1])
                return X, y
                
        except Exception as e:
            logger.error("Failed to load training data", error=str(e))
            return np.array([]), np.array([])
    
    def _extract_performance_features(self, durations: Dict[str, Any], 
                                    scores: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        """Extract performance-based features"""
        features = {}
        
        # Duration features
        features['avg_duration'] = np.mean([d for d in durations.values() if isinstance(d, (int, float)) and d > 0])
        features['min_duration'] = np.min([d for d in durations.values() if isinstance(d, (int, float)) and d > 0]) if durations else 0.0
        features['max_duration'] = np.max([d for d in durations.values() if isinstance(d, (int, float)) and d > 0]) if durations else 0.0
        features['duration_variance'] = np.var([d for d in durations.values() if isinstance(d, (int, float)) and d > 0]) if durations else 0.0
        
        # Score features
        all_scores = []
        for score_dict in scores.values():
            if isinstance(score_dict, dict):
                all_scores.extend([s for s in score_dict.values() if isinstance(s, (int, float))])
        
        features['avg_score'] = np.mean(all_scores) if all_scores else 0.0
        features['max_score'] = np.max(all_scores) if all_scores else 0.0
        features['score_variance'] = np.var(all_scores) if all_scores else 0.0
        
        return features
    
    def _determine_best_backend(self, durations: Dict[str, Any], 
                              scores: Dict[str, Dict[str, Any]]) -> str:
        """Determine the best backend based on performance metrics"""
        backend_scores = {}
        
        # Calculate composite score for each backend
        for backend in self.backend_labels:
            if backend == 'error_aware':
                continue
            
            # Get duration (lower is better)
            duration = durations.get(backend, float('inf'))
            if duration <= 0 or duration == float('inf'):
                continue
            
            # Get score (higher is better)
            score = 0.0
            if backend in ['faiss', 'hnsw']:
                score = scores.get('dense', {}).get(backend, 0.0)
            else:
                score = scores.get('bm25', {}).get(backend, 0.0)
            
            # Composite score: score / (1 + duration) - balance quality and speed
            composite_score = score / (1 + duration)
            backend_scores[backend] = composite_score
        
        # Return backend with highest composite score
        if backend_scores:
            return max(backend_scores, key=backend_scores.get)
        
        # Fallback to error_aware if no valid backends
        return 'error_aware'
    
    def train_model(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Train the backend selection model
        
        Args:
            X: Feature matrix
            y: Target labels
            
        Returns:
            Training metrics
        """
        if len(X) == 0 or len(y) == 0:
            logger.error("No training data available")
            return {}
        
        logger.info("Starting model training", samples=len(X), features=X.shape[1])
        
        try:
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Scale features
            X_train_scaled = self.feature_extractor.scaler.fit_transform(X_train)
            X_test_scaled = self.feature_extractor.scaler.transform(X_test)
            
            # Encode labels
            y_train_encoded = self.feature_extractor.label_encoder.fit_transform(y_train)
            y_test_encoded = self.feature_extractor.label_encoder.transform(y_test)
            
            # Train Random Forest model
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                class_weight='balanced'
            )
            
            self.model.fit(X_train_scaled, y_train_encoded)
            
            # Evaluate model
            y_pred = self.model.predict(X_test_scaled)
            y_pred_proba = self.model.predict_proba(X_test_scaled)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test_encoded, y_pred)
            
            # Get feature names
            self.feature_names = self._get_feature_names()
            
            # Feature importance
            feature_importance = dict(zip(
                self.feature_names,
                self.model.feature_importances_
            ))
            
            # Classification report
            class_names = self.feature_extractor.label_encoder.classes_
            report = classification_report(
                y_test_encoded, y_pred, 
                target_names=class_names, 
                output_dict=True
            )
            
            # Confusion matrix
            cm = confusion_matrix(y_test_encoded, y_pred)
            
            metrics = {
                'accuracy': accuracy,
                'classification_report': report,
                'confusion_matrix': cm.tolist(),
                'feature_importance': feature_importance,
                'class_names': class_names.tolist(),
                'training_samples': len(X_train),
                'test_samples': len(X_test),
                'n_features': X.shape[1]
            }
            
            logger.info("Model training completed", accuracy=accuracy)
            return metrics
            
        except Exception as e:
            logger.error("Model training failed", error=str(e))
            return {}
    
    def _get_feature_names(self) -> List[str]:
        """Get feature names for the model"""
        # Query features
        query_features = [
            'query_length', 'query_word_count', 'query_has_numbers', 
            'query_has_special_chars', 'query_avg_word_length',
            'query_entropy', 'query_has_questions', 'query_has_operators'
        ]
        
        # Filter features
        filter_features = [
            'has_difficulty_filter', 'has_subject_filter', 'has_skill_filter', 'filter_count'
        ]
        
        # Time features
        time_features = ['is_business_hours', 'is_weekend']
        
        # User features
        user_features = ['user_id_hash']
        
        # Performance features
        performance_features = [
            'avg_duration', 'min_duration', 'max_duration', 'duration_variance',
            'avg_score', 'max_score', 'score_variance'
        ]
        
        return (query_features + filter_features + time_features + 
                user_features + performance_features)
    
    def save_model(self, model_path: str, feature_order_path: str):
        """Save trained model and feature order"""
        try:
            # Create directories if they don't exist
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            
            # Save model
            joblib.dump({
                'model': self.model,
                'scaler': self.feature_extractor.scaler,
                'label_encoder': self.feature_extractor.label_encoder,
                'feature_names': self.feature_names
            }, model_path)
            
            # Save feature order
            with open(feature_order_path, 'w') as f:
                json.dump(self.feature_names, f, indent=2)
            
            logger.info("Model saved", model_path=model_path, feature_order_path=feature_order_path)
            
        except Exception as e:
            logger.error("Failed to save model", error=str(e))
    
    def predict_backend(self, query: str, user_id: str, filters: Dict[str, Any]) -> Tuple[str, float]:
        """
        Predict best backend for a query
        
        Args:
            query: Search query
            user_id: User ID
            filters: Search filters
            
        Returns:
            Tuple of (backend_name, confidence)
        """
        if not self.model:
            logger.warning("Model not trained, returning default backend")
            return 'faiss', 0.0
        
        try:
            # Extract features
            features = self.feature_extractor.extract_query_features(query, user_id, filters)
            
            # Convert to array
            X = np.array([list(features.values())])
            
            # Scale features
            X_scaled = self.feature_extractor.scaler.transform(X)
            
            # Predict
            pred_proba = self.model.predict_proba(X_scaled)[0]
            pred_class_idx = np.argmax(pred_proba)
            confidence = pred_proba[pred_class_idx]
            
            # Decode prediction
            backend = self.feature_extractor.label_encoder.inverse_transform([pred_class_idx])[0]
            
            return backend, confidence
            
        except Exception as e:
            logger.error("Prediction failed", error=str(e))
            return 'faiss', 0.0

async def main():
    """Main training function"""
    logger.info("Starting backend selector training")
    
    try:
        # Initialize trainer
        trainer = BackendSelectorTrainer()
        
        # Load training data
        X, y = await trainer.load_training_data(days=7, min_samples=50)
        
        if len(X) == 0:
            logger.error("No training data available")
            return
        
        # Train model
        metrics = trainer.train_model(X, y)
        
        if not metrics:
            logger.error("Training failed")
            return
        
        # Print results
        print("\n" + "="*60)
        print("BACKEND SELECTOR TRAINING RESULTS")
        print("="*60)
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Training samples: {metrics['training_samples']}")
        print(f"Test samples: {metrics['test_samples']}")
        print(f"Features: {metrics['n_features']}")
        
        print("\nClassification Report:")
        print("-" * 40)
        for class_name in metrics['class_names']:
            if class_name in metrics['classification_report']:
                report = metrics['classification_report'][class_name]
                print(f"{class_name:12} - Precision: {report['precision']:.3f}, "
                      f"Recall: {report['recall']:.3f}, F1: {report['f1-score']:.3f}")
        
        print("\nConfusion Matrix:")
        print("-" * 40)
        print("Actual\\Predicted", end="")
        for name in metrics['class_names']:
            print(f"{name:>10}", end="")
        print()
        
        cm = np.array(metrics['confusion_matrix'])
        for i, name in enumerate(metrics['class_names']):
            print(f"{name:15}", end="")
            for j in range(len(metrics['class_names'])):
                print(f"{cm[i,j]:>10}", end="")
            print()
        
        print("\nTop Feature Importances:")
        print("-" * 40)
        sorted_features = sorted(metrics['feature_importance'].items(), 
                               key=lambda x: x[1], reverse=True)
        for feature, importance in sorted_features[:10]:
            print(f"{feature:25} {importance:.4f}")
        
        # Save model
        model_path = "backend/data/ml/backend_selector.joblib"
        feature_order_path = "backend/data/ml/feature_order.json"
        trainer.save_model(model_path, feature_order_path)
        
        print(f"\nModel saved to: {model_path}")
        print(f"Feature order saved to: {feature_order_path}")
        
        # Test prediction
        test_query = "linear equations algebra"
        test_user = "test_user_123"
        test_filters = {"subject": "math", "difficulty": "intermediate"}
        
        backend, confidence = trainer.predict_backend(test_query, test_user, test_filters)
        print(f"\nTest prediction:")
        print(f"Query: '{test_query}'")
        print(f"Predicted backend: {backend} (confidence: {confidence:.3f})")
        
        print("\n" + "="*60)
        logger.info("Backend selector training completed successfully")
        
    except Exception as e:
        logger.error("Training failed", error=str(e))
        raise

if __name__ == "__main__":
    asyncio.run(main())
