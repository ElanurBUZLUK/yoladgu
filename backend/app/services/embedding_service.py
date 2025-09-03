"""
Advanced embedding service with multiple models and evaluation capabilities.
"""

import asyncio
import time
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_score, recall_score, f1_score
import torch

from app.core.config import settings


class EmbeddingService:
    """
    Advanced embedding service with multiple models and evaluation metrics.
    
    Supported models:
    - paraphrase-multilingual-MiniLM-L12-v2 (384d, multilingual)
    - all-MiniLM-L6-v2 (384d, English)
    - all-mpnet-base-v2 (768d, English)
    - paraphrase-MiniLM-L6-v2 (384d, English)
    - distiluse-base-multilingual-cased-v2 (512d, multilingual)
    """
    
    def __init__(self):
        self.models = {}
        self.current_model = None
        self.vector_size = 384
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Model configurations
        self.model_configs = {
            "paraphrase-multilingual-MiniLM-L12-v2": {
                "dimensions": 384,
                "language": "multilingual",
                "speed": "fast",
                "quality": "high"
            },
            "all-MiniLM-L6-v2": {
                "dimensions": 384,
                "language": "en",
                "speed": "very_fast",
                "quality": "medium"
            },
            "all-mpnet-base-v2": {
                "dimensions": 768,
                "language": "en",
                "quality": "very_high",
                "speed": "medium"
            },
            "paraphrase-MiniLM-L6-v2": {
                "dimensions": 384,
                "language": "en",
                "speed": "very_fast",
                "quality": "medium"
            },
            "distiluse-base-multilingual-cased-v2": {
                "dimensions": 512,
                "language": "multilingual",
                "speed": "fast",
                "quality": "high"
            }
        }
        
        # Performance tracking
        self.stats = {
            "total_encodings": 0,
            "avg_encoding_time_ms": 0.0,
            "model_usage": {},
            "quality_metrics": {}
        }
    
    async def initialize(self, default_model: str = "paraphrase-multilingual-MiniLM-L12-v2") -> bool:
        """Initialize the embedding service with a default model."""
        try:
            await self.load_model(default_model)
            print(f"✅ Embedding service initialized with {default_model}")
            return True
        except Exception as e:
            print(f"❌ Error initializing embedding service: {e}")
            return False
    
    async def load_model(self, model_name: str) -> bool:
        """Load a specific embedding model."""
        try:
            if model_name not in self.model_configs:
                raise ValueError(f"Unsupported model: {model_name}")
            
            if model_name not in self.models:
                print(f"🔄 Loading model: {model_name}")
                start_time = time.time()
                
                # Load model
                model = SentenceTransformer(f'sentence-transformers/{model_name}')
                
                # Move to device if available
                if self.device == "cuda":
                    model = model.to(self.device)
                
                self.models[model_name] = model
                load_time = (time.time() - start_time) * 1000
                print(f"✅ Model {model_name} loaded in {load_time:.2f}ms")
            
            self.current_model = model_name
            self.vector_size = self.model_configs[model_name]["dimensions"]
            
            # Update stats
            if model_name not in self.stats["model_usage"]:
                self.stats["model_usage"][model_name] = 0
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading model {model_name}: {e}")
            return False
    
    def get_current_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        if not self.current_model:
            return {}
        
        config = self.model_configs[self.current_model].copy()
        config.update({
            "name": self.current_model,
            "vector_size": self.vector_size,
            "device": self.device,
            "usage_count": self.stats["model_usage"].get(self.current_model, 0)
        })
        return config
    
    def list_available_models(self) -> Dict[str, Dict[str, Any]]:
        """List all available models with their configurations."""
        return self.model_configs.copy()
    
    async def encode_text(
        self, 
        text: str, 
        model_name: Optional[str] = None,
        normalize: bool = True
    ) -> np.ndarray:
        """Encode text to vector using the specified or current model."""
        try:
            # Load model if specified
            if model_name and model_name != self.current_model:
                await self.load_model(model_name)
            
            if not self.current_model:
                raise RuntimeError("No model loaded")
            
            model = self.models[self.current_model]
            
            # Update usage stats
            self.stats["model_usage"][self.current_model] += 1
            self.stats["total_encodings"] += 1
            
            # Encode
            start_time = time.time()
            embedding = model.encode(text, convert_to_tensor=False)
            encoding_time = (time.time() - start_time) * 1000
            
            # Update performance stats
            sc = self.stats["total_encodings"]
            self.stats["avg_encoding_time_ms"] = (
                (self.stats["avg_encoding_time_ms"] * (sc - 1) + encoding_time) / sc
            )
            
            # Convert to numpy array
            if isinstance(embedding, torch.Tensor):
                embedding = embedding.cpu().numpy()
            
            # Normalize if requested
            if normalize:
                embedding = self._normalize_vector(embedding)
            
            return embedding
            
        except Exception as e:
            print(f"❌ Error encoding text: {e}")
            raise
    
    async def encode_batch(
        self, 
        texts: List[str], 
        model_name: Optional[str] = None,
        normalize: bool = True,
        batch_size: int = 32
    ) -> np.ndarray:
        """Encode multiple texts to vectors in batches."""
        try:
            # Load model if specified
            if model_name and model_name != self.current_model:
                await self.load_model(model_name)
            
            if not self.current_model:
                raise RuntimeError("No model loaded")
            
            model = self.models[self.current_model]
            
            # Update usage stats
            self.stats["model_usage"][self.current_model] += 1
            self.stats["total_encodings"] += len(texts)
            
            # Encode in batches
            start_time = time.time()
            embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                batch_embeddings = model.encode(batch_texts, convert_to_tensor=False)
                
                if isinstance(batch_embeddings, torch.Tensor):
                    batch_embeddings = batch_embeddings.cpu().numpy()
                
                embeddings.append(batch_embeddings)
            
            # Concatenate all batches
            all_embeddings = np.vstack(embeddings)
            
            encoding_time = (time.time() - start_time) * 1000
            
            # Update performance stats
            sc = self.stats["total_encodings"]
            self.stats["avg_encoding_time_ms"] = (
                (self.stats["avg_encoding_time_ms"] * (sc - len(texts)) + encoding_time) / sc
            )
            
            # Normalize if requested
            if normalize:
                all_embeddings = self._normalize_vectors(all_embeddings)
            
            return all_embeddings
            
        except Exception as e:
            print(f"❌ Error encoding batch: {e}")
            raise
    
    def _normalize_vector(self, vector: np.ndarray) -> np.ndarray:
        """Normalize a single vector."""
        norm = np.linalg.norm(vector)
        if norm > 0:
            return vector / norm
        return vector
    
    def _normalize_vectors(self, vectors: np.ndarray) -> np.ndarray:
        """Normalize multiple vectors."""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1.0)
        return vectors / norms
    
    async def evaluate_model_performance(
        self,
        model_name: str,
        test_texts: List[str],
        ground_truth_similarities: Optional[np.ndarray] = None,
        evaluation_metrics: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate model performance on test data.
        
        Args:
            model_name: Name of the model to evaluate
            test_texts: List of test texts
            ground_truth_similarities: Ground truth similarity matrix (optional)
            evaluation_metrics: List of metrics to compute
            
        Returns:
            Dictionary with evaluation results
        """
        if evaluation_metrics is None:
            evaluation_metrics = ["cosine_similarity", "clustering_quality", "semantic_coherence"]
        
        try:
            # Load model
            await self.load_model(model_name)
            
            # Encode test texts
            embeddings = await self.encode_batch(test_texts, normalize=True)
            
            results = {
                "model": model_name,
                "test_size": len(test_texts),
                "vector_size": self.vector_size,
                "metrics": {}
            }
            
            # Compute cosine similarity matrix
            if "cosine_similarity" in evaluation_metrics:
                similarity_matrix = cosine_similarity(embeddings)
                results["metrics"]["cosine_similarity"] = {
                    "mean": float(np.mean(similarity_matrix)),
                    "std": float(np.std(similarity_matrix)),
                    "min": float(np.min(similarity_matrix)),
                    "max": float(np.max(similarity_matrix))
                }
            
            # Clustering quality (using k-means)
            if "clustering_quality" in evaluation_metrics:
                from sklearn.cluster import KMeans
                from sklearn.metrics import silhouette_score
                
                # Use elbow method to find optimal k
                inertias = []
                silhouette_scores = []
                k_range = range(2, min(10, len(test_texts) // 2))
                
                for k in k_range:
                    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                    kmeans.fit(embeddings)
                    inertias.append(kmeans.inertia_)
                    
                    if k > 1:
                        labels = kmeans.labels_
                        silhouette_scores.append(silhouette_score(embeddings, labels))
                    else:
                        silhouette_scores.append(0.0)
                
                # Find optimal k (elbow method)
                optimal_k = k_range[np.argmax(silhouette_scores)]
                
                results["metrics"]["clustering_quality"] = {
                    "optimal_k": int(optimal_k),
                    "max_silhouette_score": float(np.max(silhouette_scores)),
                    "inertias": [float(x) for x in inertias]
                }
            
            # Semantic coherence (using nearest neighbors)
            if "semantic_coherence" in evaluation_metrics:
                from sklearn.neighbors import NearestNeighbors
                
                # Find nearest neighbors for each text
                nbrs = NearestNeighbors(n_neighbors=min(6, len(test_texts)), algorithm='auto')
                nbrs.fit(embeddings)
                
                distances, indices = nbrs.kneighbors(embeddings)
                
                # Calculate average distance to nearest neighbors
                avg_distances = np.mean(distances[:, 1:], axis=1)  # Exclude self
                
                results["metrics"]["semantic_coherence"] = {
                    "mean_neighbor_distance": float(np.mean(avg_distances)),
                    "std_neighbor_distance": float(np.std(avg_distances)),
                    "coherence_score": float(1.0 / (1.0 + np.mean(avg_distances)))
                }
            
            # Compare with ground truth if available
            if ground_truth_similarities is not None and "ground_truth_comparison" in evaluation_metrics:
                predicted_similarities = cosine_similarity(embeddings)
                
                # Flatten matrices for comparison
                gt_flat = ground_truth_similarities.flatten()
                pred_flat = predicted_similarities.flatten()
                
                # Calculate correlation
                correlation = np.corrcoef(gt_flat, pred_flat)[0, 1]
                
                results["metrics"]["ground_truth_comparison"] = {
                    "correlation": float(correlation),
                    "mse": float(np.mean((gt_flat - pred_flat) ** 2)),
                    "mae": float(np.mean(np.abs(gt_flat - pred_flat)))
                }
            
            # Update quality metrics
            self.stats["quality_metrics"][model_name] = results["metrics"]
            
            return results
            
        except Exception as e:
            print(f"❌ Error evaluating model {model_name}: {e}")
            return {"error": str(e)}
    
    async def compare_models(
        self,
        model_names: List[str],
        test_texts: List[str],
        evaluation_metrics: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Compare multiple models on the same test data."""
        results = {}
        
        for model_name in model_names:
            try:
                model_results = await self.evaluate_model_performance(
                    model_name, test_texts, evaluation_metrics=evaluation_metrics
                )
                results[model_name] = model_results
            except Exception as e:
                results[model_name] = {"error": str(e)}
        
        # Add comparison summary
        comparison_summary = self._create_comparison_summary(results)
        results["comparison_summary"] = comparison_summary
        
        return results
    
    def _create_comparison_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Create a summary comparing all models."""
        summary = {
            "best_models": {},
            "performance_ranking": []
        }
        
        # Find best model for each metric
        for model_name, model_results in results.items():
            if "error" in model_results:
                continue
            
            metrics = model_results.get("metrics", {})
            
            for metric_name, metric_values in metrics.items():
                if metric_name not in summary["best_models"]:
                    summary["best_models"][metric_name] = {"model": None, "value": -float('inf')}
                
                # Determine if this is a "higher is better" metric
                if metric_name in ["cosine_similarity", "clustering_quality", "semantic_coherence"]:
                    if "mean" in metric_values:
                        value = metric_values["mean"]
                        if value > summary["best_models"][metric_name]["value"]:
                            summary["best_models"][metric_name] = {"model": model_name, "value": value}
        
        # Create performance ranking
        model_scores = {}
        for model_name, model_results in results.items():
            if "error" in model_results:
                continue
            
            # Calculate overall score
            score = 0.0
            metrics = model_results.get("metrics", {})
            
            for metric_values in metrics.values():
                if "mean" in metric_values:
                    score += metric_values["mean"]
            
            model_scores[model_name] = score
        
        # Sort by score
        sorted_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
        summary["performance_ranking"] = [{"model": name, "score": score} for name, score in sorted_models]
        
        return summary
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive service statistics."""
        return {
            **self.stats,
            "current_model": self.current_model,
            "current_model_info": self.get_current_model_info(),
            "available_models": self.list_available_models(),
            "device": self.device
        }
    
    async def benchmark_models(
        self,
        test_texts: List[str],
        models: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Benchmark multiple models for speed and quality."""
        if models is None:
            models = list(self.model_configs.keys())
        
        benchmark_results = {}
        
        for model_name in models:
            try:
                print(f"🔄 Benchmarking {model_name}...")
                
                # Load model
                await self.load_model(model_name)
                
                # Benchmark encoding speed
                start_time = time.time()
                embeddings = await self.encode_batch(test_texts, normalize=True)
                encoding_time = time.time() - start_time
                
                # Benchmark search speed (cosine similarity)
                search_start = time.time()
                similarity_matrix = cosine_similarity(embeddings)
                search_time = time.time() - search_start
                
                # Calculate quality metrics
                quality_metrics = await self.evaluate_model_performance(
                    model_name, test_texts, 
                    evaluation_metrics=["cosine_similarity", "clustering_quality"]
                )
                
                benchmark_results[model_name] = {
                    "encoding_speed": {
                        "total_time": encoding_time,
                        "time_per_text": encoding_time / len(test_texts),
                        "texts_per_second": len(test_texts) / encoding_time
                    },
                    "search_speed": {
                        "similarity_matrix_time": search_time,
                        "operations_per_second": (len(test_texts) ** 2) / search_time
                    },
                    "quality": quality_metrics.get("metrics", {}),
                    "vector_size": self.vector_size,
                    "memory_usage_mb": embeddings.nbytes / (1024 * 1024)
                }
                
                print(f"✅ {model_name} benchmarked successfully")
                
            except Exception as e:
                print(f"❌ Error benchmarking {model_name}: {e}")
                benchmark_results[model_name] = {"error": str(e)}
        
        return benchmark_results


# Global instance
embedding_service = EmbeddingService()
