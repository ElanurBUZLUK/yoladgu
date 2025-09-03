"""
Advanced FAISS index backend with multiple index types and quantization techniques.
"""

import asyncio
import time
import pickle
import os
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np

try:
    import faiss
except ImportError as e:
    raise ImportError("faiss-cpu gerekli: pip install faiss-cpu") from e

from .base import BaseIndexBackend


class FAISSAdvancedIndexBackend(BaseIndexBackend):
    """
    Advanced FAISS backend with multiple index types:
    - IVF (Inverted File Index) for large-scale approximate search
    - HNSW for fast approximate nearest neighbor
    - PQ (Product Quantization) for memory efficiency
    - SQ (Scalar Quantization) for balanced performance
    """
    
    def __init__(
        self, 
        vector_size: int,
        index_type: str = "ivf",  # "ivf", "hnsw", "pq", "sq", "ivfpq", "ivfsq"
        metric: str = "ip",  # "ip" for cosine, "l2" for euclidean
        nlist: int = 100,  # Number of clusters for IVF
        nprobe: int = 10,  # Number of clusters to probe during search
        m: int = 8,  # Number of sub-vectors for PQ
        bits: int = 8,  # Bits per sub-vector for PQ
        ef_construction: int = 200,  # HNSW construction parameter
        ef_search: int = 128,  # HNSW search parameter
        max_elements: int = 1000000,
        index_path: Optional[str] = None,
        **kwargs
    ):
        super().__init__(vector_size)
        
        self.index_type = index_type
        self.metric = metric
        self.nlist = nlist
        self.nprobe = nprobe
        self.m = m
        self.bits = bits
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.max_elements = max_elements
        self.index_path = index_path
        
        # Index and quantizer
        self.index = None
        self.quantizer = None
        
        # ID mappings
        self.id_to_metadata: Dict[str, Dict[str, Any]] = {}
        self.string_to_int: Dict[str, int] = {}
        self.int_to_string: Dict[int, str] = {}
        self.next_id = 0
        
        # Performance tracking
        self.stats = {
            "total_items": 0,
            "search_count": 0,
            "avg_search_time_ms": 0.0,
            "build_time_ms": 0.0,
            "index_size_mb": 0.0,
            "compression_ratio": 1.0
        }
        
        # Additional parameters
        self._update_params(**kwargs)
    
    def _update_params(self, **kwargs):
        """Update index parameters."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    async def initialize(self) -> bool:
        """Initialize the advanced FAISS index."""
        try:
            start_time = time.time()
            
            # Create quantizer if needed
            if self.index_type.startswith("ivf"):
                self._create_quantizer()
            
            # Create main index
            self._create_index()
            
            # Set search parameters
            self._configure_search()
            
            # Load existing index if path exists
            if self.index_path and os.path.exists(self.index_path):
                await self._load_index()
            
            build_time = (time.time() - start_time) * 1000
            self.stats["build_time_ms"] = build_time
            self._built = True
            
            print(f"✅ FAISS {self.index_type.upper()} index initialized successfully")
            return True
            
        except Exception as e:
            print(f"❌ Error initializing FAISS {self.index_type} index: {e}")
            return False
    
    def _create_quantizer(self):
        """Create quantizer for IVF-based indexes."""
        if self.metric == "ip":
            self.quantizer = faiss.IndexFlatIP(self.vector_size)
        else:
            self.quantizer = faiss.IndexFlatL2(self.vector_size)
    
    def _create_index(self):
        """Create the main FAISS index based on type."""
        if self.index_type == "ivf":
            self.index = faiss.IndexIVFFlat(
                self.quantizer, self.vector_size, self.nlist
            )
            # IVF index needs training
            self.index.is_trained = False
            
        elif self.index_type == "hnsw":
            self.index = faiss.IndexHNSWFlat(
                self.vector_size, self.m, faiss.METRIC_INNER_PRODUCT if self.metric == "ip" else faiss.METRIC_L2
            )
            self.index.hnsw.efConstruction = self.ef_construction
            self.index.hnsw.efSearch = self.ef_search
            
        elif self.index_type == "pq":
            self.index = faiss.IndexPQ(
                self.vector_size, self.m, self.bits
            )
            
        elif self.index_type == "sq":
            self.index = faiss.IndexScalarQuantizer(
                self.vector_size, faiss.ScalarQuantizer.QT_8bit
            )
            
        elif self.index_type == "ivfpq":
            self.index = faiss.IndexIVFPQ(
                self.quantizer, self.vector_size, self.nlist, self.m, self.bits
            )
            
        elif self.index_type == "ivfsq":
            self.index = faiss.IndexIVFScalarQuantizer(
                self.quantizer, self.vector_size, self.nlist, 
                faiss.ScalarQuantizer.QT_8bit
            )
            
        else:
            raise ValueError(f"Unsupported index type: {self.index_type}")
    
    def _configure_search(self):
        """Configure search parameters for the index."""
        if hasattr(self.index, 'nprobe'):
            self.index.nprobe = self.nprobe
        
        if hasattr(self.index, 'hnsw'):
            self.index.hnsw.efSearch = self.ef_search
    
    async def _load_index(self):
        """Load existing index from file."""
        try:
            # Load main index
            self.index = faiss.read_index(self.index_path)
            
            # Load metadata
            metadata_path = self.index_path.replace('.index', '.metadata.pkl')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'rb') as f:
                    data = pickle.load(f)
                    self.id_to_metadata = data.get('metadata', {})
                    self.string_to_int = data.get('string_to_int', {})
                    self.int_to_string = data.get('int_to_string', {})
                    self.next_id = data.get('next_id', 0)
                    self.stats["total_items"] = len(self.id_to_metadata)
            
            print(f"✅ Loaded existing {self.index_type} index from {self.index_path}")
            
        except Exception as e:
            print(f"⚠️ Could not load existing index: {e}")
    
    async def add_items(
        self, 
        vectors: np.ndarray, 
        ids: List[str], 
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> bool:
        """Add items to the advanced FAISS index."""
        try:
            if not self._validate_vector(vectors):
                raise ValueError(f"Vector dimensions must be {self.vector_size}")
            
            # Normalize vectors for cosine similarity
            if self.metric == "ip":
                vectors = self._normalize_vector(vectors).astype(np.float32)
            else:
                vectors = vectors.astype(np.float32)
            
            # Create integer IDs
            int_ids = self._add_ids(ids)
            
            # Add to index
            if hasattr(self.index, 'add_with_ids') and self.index_type not in ["pq", "sq"]:
                # Check if IVF index needs training
                if hasattr(self.index, 'is_trained') and not self.index.is_trained:
                    # Train the index first
                    self.index.train(vectors)
                    self.index.is_trained = True
                    print(f"✅ Trained {self.index_type} index with {len(vectors)} vectors")
                
                self.index.add_with_ids(vectors, int_ids)
            else:
                # For PQ and SQ, use regular add
                if hasattr(self.index, 'is_trained') and not self.index.is_trained:
                    # Train the index first
                    self.index.train(vectors)
                    self.index.is_trained = True
                    print(f"✅ Trained {self.index_type} index with {len(vectors)} vectors")
                
                self.index.add(vectors)
            
            # Store metadata
            if metadata:
                for i, item_id in enumerate(ids):
                    self.id_to_metadata[item_id] = metadata[i] if i < len(metadata) else {}
            
            self.stats["total_items"] += vectors.shape[0]
            
            # Update compression ratio
            self._update_compression_stats()
            
            return True
            
        except Exception as e:
            print(f"❌ Error adding items to FAISS {self.index_type} index: {e}")
            return False
    
    def _add_ids(self, ids: List[str]) -> np.ndarray:
        """Create integer IDs for FAISS."""
        int_ids = []
        for sid in ids:
            if sid not in self.string_to_int:
                self.string_to_int[sid] = self.next_id
                self.int_to_string[self.next_id] = sid
                self.next_id += 1
            int_ids.append(self.string_to_int[sid])
        return np.array(int_ids, dtype=np.int64)
    
    async def search(
        self, 
        query_vector: np.ndarray, 
        k: int = 10, 
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors."""
        try:
            if not self._validate_vector(query_vector):
                raise ValueError(f"Query vector dimensions must be {self.vector_size}")
            
            # Ensure query vector is 2D
            if query_vector.ndim == 1:
                query_vector = query_vector.reshape(1, -1)
            
            # Normalize query vector
            if self.metric == "ip":
                query_vector = self._normalize_vector(query_vector).astype(np.float32)
            else:
                query_vector = query_vector.astype(np.float32)
            
            # Perform search
            start_time = time.time()
            
            if hasattr(self.index, 'search'):
                D, I = self.index.search(query_vector, k)
            else:
                # Fallback for some index types
                D, I = self.index.search(query_vector, k)
            
            search_time = (time.time() - start_time) * 1000
            
            # Update stats
            self.stats["search_count"] += 1
            sc = self.stats["search_count"]
            self.stats["avg_search_time_ms"] = (
                (self.stats["avg_search_time_ms"] * (sc - 1) + search_time) / sc
            )
            
            # Format results
            results = []
            for i in range(len(I[0])):
                idx = I[0][i]
                distance = D[0][i]
                
                if idx < 0:  # No result
                    continue
                
                # Get string ID
                if hasattr(self.index, 'id_map'):
                    # Index has built-in ID mapping
                    string_id = str(idx)
                else:
                    # Use our custom mapping
                    string_id = self.int_to_string.get(idx, str(idx))
                
                # Get metadata
                metadata = self.id_to_metadata.get(string_id, {})
                
                # Convert distance to similarity score
                if self.metric == "ip":
                    score = float(distance)
                    distance = 1.0 - score
                else:
                    score = 1.0 / (1.0 + distance)
                
                results.append({
                    "item_id": string_id,
                    "score": score,
                    "distance": float(distance),
                    "metadata": metadata
                })
            
            # Apply filters
            if filters:
                results = self._apply_filters(results, filters)
            
            # Sort by score (descending)
            results.sort(key=lambda x: x["score"], reverse=True)
            
            return results[:k]
            
        except Exception as e:
            print(f"❌ Error searching FAISS {self.index_type} index: {e}")
            return []
    
    def _apply_filters(self, results: List[Dict[str, Any]], filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Apply metadata filters to search results."""
        filtered_results = []
        
        for result in results:
            metadata = result.get("metadata", {})
            include = True
            
            for key, value in filters.items():
                if key in metadata:
                    if isinstance(value, list):
                        if not any(v in metadata[key] for v in value):
                            include = False
                            break
                    elif metadata[key] != value:
                        include = False
                        break
                else:
                    include = False
                    break
            
            if include:
                filtered_results.append(result)
        
        return filtered_results
    
    def _update_compression_stats(self):
        """Update compression and size statistics."""
        try:
            if self.index_path and os.path.exists(self.index_path):
                size_bytes = os.path.getsize(self.index_path)
                self.stats["index_size_mb"] = size_bytes / (1024 * 1024)
                
                # Calculate compression ratio
                if self.stats["total_items"] > 0:
                    original_size = self.stats["total_items"] * self.vector_size * 4  # float32
                    self.stats["compression_ratio"] = original_size / size_bytes
        except Exception:
            pass
    
    async def delete_items(self, ids: List[str]) -> bool:
        """Delete items from the index."""
        # FAISS doesn't support deletion for most index types
        # Remove from metadata and mappings
        for item_id in ids:
            if item_id in self.string_to_int:
                int_id = self.string_to_int[item_id]
                del self.string_to_int[item_id]
                del self.int_to_string[int_id]
            self.id_to_metadata.pop(item_id, None)
            self.stats["total_items"] = max(0, self.stats["total_items"] - 1)
        
        print(f"⚠️ Items removed from metadata (FAISS {self.index_type} doesn't support deletion)")
        return True
    
    async def save_index(self, path: Optional[str] = None) -> bool:
        """Save index to disk."""
        try:
            save_path = path or self.index_path
            if not save_path:
                return False
            
            # Save main index
            faiss.write_index(self.index, save_path)
            
            # Save metadata
            metadata_path = save_path.replace('.index', '.metadata.pkl')
            metadata_data = {
                'metadata': self.id_to_metadata,
                'string_to_int': self.string_to_int,
                'int_to_string': self.int_to_string,
                'next_id': self.next_id
            }
            
            with open(metadata_path, 'wb') as f:
                pickle.dump(metadata_data, f)
            
            print(f"✅ Saved {self.index_type} index to {save_path}")
            return True
            
        except Exception as e:
            print(f"❌ Error saving {self.index_type} index: {e}")
            return False
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive index statistics."""
        stats = {
            **self.stats,
            "index_type": f"FAISS-{self.index_type.upper()}",
            "metric": self.metric,
            "nlist": getattr(self, 'nlist', None),
            "nprobe": getattr(self, 'nprobe', None),
            "m": getattr(self, 'm', None),
            "bits": getattr(self, 'bits', None),
            "ef_construction": getattr(self, 'ef_construction', None),
            "ef_search": getattr(self, 'ef_search', None),
            "max_elements": getattr(self, 'max_elements', None)
        }
        
        # Update compression stats
        self._update_compression_stats()
        stats.update({
            "index_size_mb": self.stats["index_size_mb"],
            "compression_ratio": self.stats["compression_ratio"]
        })
        
        return stats
    
    async def optimize_hyperparameters(
        self, 
        training_vectors: np.ndarray,
        validation_queries: np.ndarray,
        validation_ground_truth: List[List[str]],
        param_grid: Optional[Dict[str, List[Any]]] = None
    ) -> Dict[str, Any]:
        """
        Optimize hyperparameters using grid search.
        
        Args:
            training_vectors: Vectors for training the index
            validation_queries: Query vectors for validation
            validation_ground_truth: Ground truth results for queries
            param_grid: Parameter grid for optimization
            
        Returns:
            Best parameters and performance metrics
        """
        if param_grid is None:
            param_grid = self._get_default_param_grid()
        
        best_score = 0.0
        best_params = {}
        results = []
        
        print(f"🔍 Starting hyperparameter optimization for {self.index_type}...")
        
        # Grid search
        for params in self._generate_param_combinations(param_grid):
            try:
                # Update parameters
                self._update_params(**params)
                
                # Reinitialize index
                await self.initialize()
                
                # Add training vectors
                dummy_ids = [f"train_{i}" for i in range(len(training_vectors))]
                await self.add_items(training_vectors, dummy_ids)
                
                # Evaluate performance
                score = await self._evaluate_performance(
                    validation_queries, validation_ground_truth
                )
                
                results.append({
                    "params": params,
                    "score": score
                })
                
                if score > best_score:
                    best_score = score
                    best_params = params
                
                print(f"  ✅ Params: {params} -> Score: {score:.4f}")
                
            except Exception as e:
                print(f"  ❌ Params: {params} -> Error: {e}")
                continue
        
        # Apply best parameters
        if best_params:
            self._update_params(**best_params)
            await self.initialize()
            print(f"🎯 Best parameters: {best_params} (Score: {best_score:.4f})")
        
        return {
            "best_params": best_params,
            "best_score": best_score,
            "all_results": results
        }
    
    def _get_default_param_grid(self) -> Dict[str, List[Any]]:
        """Get default parameter grid for optimization."""
        if self.index_type == "ivf":
            return {
                "nlist": [50, 100, 200, 500],
                "nprobe": [1, 5, 10, 20]
            }
        elif self.index_type == "hnsw":
            return {
                "m": [8, 16, 32],
                "ef_construction": [100, 200, 400],
                "ef_search": [64, 128, 256]
            }
        elif self.index_type in ["pq", "ivfpq"]:
            return {
                "m": [4, 8, 16],
                "bits": [4, 8]
            }
        else:
            return {}
    
    def _generate_param_combinations(self, param_grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        """Generate all parameter combinations for grid search."""
        import itertools
        
        keys = param_grid.keys()
        values = param_grid.values()
        combinations = list(itertools.product(*values))
        
        return [dict(zip(keys, combo)) for combo in combinations]
    
    async def _evaluate_performance(
        self, 
        queries: np.ndarray, 
        ground_truth: List[List[str]]
    ) -> float:
        """
        Evaluate index performance using recall@k.
        
        Args:
            queries: Query vectors
            ground_truth: Ground truth results for each query
            
        Returns:
            Average recall@k score
        """
        try:
            total_recall = 0.0
            k = 10  # Evaluate recall@10
            
            for i, query in enumerate(queries):
                if i >= len(ground_truth):
                    break
                
                # Search
                results = await self.search(query.reshape(1, -1), k=k)
                retrieved_ids = [r["item_id"] for r in results]
                
                # Calculate recall
                gt_ids = ground_truth[i]
                if gt_ids:
                    recall = len(set(retrieved_ids) & set(gt_ids)) / len(gt_ids)
                    total_recall += recall
            
            return total_recall / len(queries) if queries.shape[0] > 0 else 0.0
            
        except Exception as e:
            print(f"⚠️ Error evaluating performance: {e}")
            return 0.0
