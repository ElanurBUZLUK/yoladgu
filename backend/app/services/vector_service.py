"""
Vector database service for dense retrieval using multiple backends.
"""

import asyncio
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer

from app.core.config import settings
from app.services.vector_index_manager import vector_index_manager


class VectorService:
    """Vector database service with multiple backend support."""
    
    def __init__(self):
        self.encoder = None
        self.vector_size = 384  # sentence-transformers/all-MiniLM-L6-v2 size
        
        # Initialize index manager
        self._initialized = False
        
        # ML backend selection
        self.ml_selector = None
        self.use_ml_selection = True
        self.dynamic_hybrid_weights = {
            "faiss": 0.4,
            "hnsw": 0.3,
            "qdrant": 0.3
        }
    
    async def _ensure_initialized(self):
        """Ensure the index manager is initialized."""
        if not self._initialized:
            success = await vector_index_manager.initialize()
            if success:
                self._initialized = True
            else:
                raise RuntimeError("Failed to initialize vector index manager")
    
    async def _load_ml_selector(self):
        """Load ML backend selector model if available"""
        if self.ml_selector is None and self.use_ml_selection:
            try:
                import joblib
                import os
                
                model_path = "backend/data/ml/backend_selector.joblib"
                if os.path.exists(model_path):
                    self.ml_selector = joblib.load(model_path)
                    print(f"Loaded ML backend selector from {model_path}")
                else:
                    print("ML backend selector model not found, using rule-based selection")
            except Exception as e:
                print(f"Failed to load ML backend selector: {e}")
                self.ml_selector = None
    
    async def _predict_best_backend(self, query: str, query_vector: np.ndarray, 
                                  filters: Optional[Dict[str, Any]] = None) -> str:
        """Predict best backend using ML model or fallback to dynamic weights"""
        try:
            await self._load_ml_selector()
            
            if self.ml_selector:
                # Extract features for ML prediction
                features = self._extract_query_features(query, query_vector, filters)
                
                # Make prediction
                backend_prediction = self.ml_selector['model'].predict([features])[0]
                backend_name = self.ml_selector['label_encoder'].inverse_transform([backend_prediction])[0]
                
                print(f"ML backend prediction: {backend_name}")
                return backend_name
            else:
                # Fallback to dynamic weights
                return self._select_backend_by_weights(filters)
                
        except Exception as e:
            print(f"ML backend prediction failed: {e}")
            return self._select_backend_by_weights(filters)
    
    def _extract_query_features(self, query: str, query_vector: np.ndarray, 
                               filters: Optional[Dict[str, Any]] = None) -> List[float]:
        """Extract features for ML backend selection"""
        features = []
        
        # Query-based features
        features.append(len(query))  # query_length
        features.append(len(query.split()))  # query_word_count
        features.append(1.0 if any(char.isdigit() for char in query) else 0.0)  # has_numbers
        features.append(1.0 if any(not char.isalnum() and char != ' ' for char in query) else 0.0)  # has_special_chars
        features.append(np.mean([len(word) for word in query.split()]) if query.split() else 0.0)  # avg_word_length
        
        # Query complexity
        features.append(self._calculate_entropy(query))  # entropy
        features.append(1.0 if '?' in query else 0.0)  # has_questions
        features.append(1.0 if any(op in query.lower() for op in ['and', 'or', 'not', '+', '-']) else 0.0)  # has_operators
        
        # Filter features
        features.append(1.0 if filters and 'difficulty' in filters else 0.0)  # has_difficulty_filter
        features.append(1.0 if filters and 'subject' in filters else 0.0)  # has_subject_filter
        features.append(1.0 if filters and 'skills' in filters else 0.0)  # has_skill_filter
        features.append(len(filters) if filters else 0)  # filter_count
        
        # Time features
        current_hour = datetime.now().hour
        features.append(1.0 if 9 <= current_hour <= 17 else 0.0)  # is_business_hours
        features.append(1.0 if datetime.now().weekday() >= 5 else 0.0)  # is_weekend
        
        # Vector features (simplified)
        features.append(np.mean(query_vector))  # vector_mean
        features.append(np.std(query_vector))  # vector_std
        features.append(np.max(query_vector))  # vector_max
        features.append(np.min(query_vector))  # vector_min
        
        return features
    
    def _calculate_entropy(self, text: str) -> float:
        """Calculate Shannon entropy of text"""
        if not text:
            return 0.0
        
        char_counts = {}
        for char in text.lower():
            char_counts[char] = char_counts.get(char, 0) + 1
        
        total_chars = len(text)
        entropy = 0.0
        for count in char_counts.values():
            p = count / total_chars
            if p > 0:
                entropy -= p * np.log2(p)
        
        return entropy
    
    def _select_backend_by_weights(self, filters: Optional[Dict[str, Any]] = None) -> str:
        """Select backend using dynamic weights based on filters"""
        try:
            # Adjust weights based on filters
            weights = self.dynamic_hybrid_weights.copy()
            
            if filters:
                # Prefer FAISS for complex queries with many filters
                if len(filters) > 2:
                    weights["faiss"] += 0.2
                    weights["hnsw"] -= 0.1
                    weights["qdrant"] -= 0.1
                
                # Prefer HNSW for skill-based queries
                if "skills" in filters:
                    weights["hnsw"] += 0.2
                    weights["faiss"] -= 0.1
                    weights["qdrant"] -= 0.1
            
            # Normalize weights
            total = sum(weights.values())
            normalized_weights = {k: v/total for k, v in weights.items()}
            
            # Select backend based on weights
            import random
            backend = random.choices(
                list(normalized_weights.keys()),
                weights=list(normalized_weights.values())
            )[0]
            
            print(f"Dynamic backend selection: {backend}")
            return backend
            
        except Exception as e:
            print(f"Dynamic backend selection failed: {e}")
            return "faiss"  # Safe fallback
    
    def _get_encoder(self) -> SentenceTransformer:
        """Get sentence transformer model (lazy initialization)."""
        if self.encoder is None:
            # Use a truly multilingual model with 384 dimensions
            # Options:
            # - paraphrase-MiniLM-L12-v2: 384d, good multilingual support
            # - paraphrase-multilingual-MiniLM-L12-v2: 384d, better multilingual
            self.encoder = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        return self.encoder
    
    def _enhance_embedding_text(self, text: str, metadata: Dict[str, Any]) -> str:
        """
        Enhance embedding text with metadata for better semantic matching.
        
        Args:
            text: Original text content
            metadata: Item metadata including error tags, skills, etc.
            
        Returns:
            Enhanced text for embedding
        """
        enhanced = text
        
        # Add error tags for English items
        if metadata.get("type") == "english" and metadata.get("error_tags"):
            error_tags = metadata["error_tags"]
            if isinstance(error_tags, list) and error_tags:
                enhanced += f" [ERROR_TAGS]: {', '.join(error_tags)}"
        
        # Add skills for Math items
        elif metadata.get("type") == "math" and metadata.get("skills"):
            skills = metadata["skills"]
            if isinstance(skills, list) and skills:
                enhanced += f" [SKILLS]: {', '.join(skills)}"
        
        # Add CEFR level for English
        if metadata.get("level_cefr"):
            enhanced += f" [LEVEL]: {metadata['level_cefr']}"
        
        # Add topic if available
        if metadata.get("topic"):
            enhanced += f" [TOPIC]: {metadata['topic']}"
        
        return enhanced
    
    async def initialize_collection(self) -> bool:
        """Initialize vector collections (now handled by index manager)."""
        try:
            await self._ensure_initialized()
            return True
        except Exception as e:
            print(f"Error initializing collections: {e}")
            return False
    
    def encode_text(self, text: str) -> List[float]:
        """Encode text to vector using sentence transformer."""
        encoder = self._get_encoder()
        embedding = encoder.encode(text, convert_to_tensor=False)
        return embedding.tolist()
    
    def encode_batch(self, texts: List[str]) -> List[List[float]]:
        """Encode multiple texts to vectors."""
        encoder = self._get_encoder()
        embeddings = encoder.encode(texts, convert_to_tensor=False)
        return embeddings.tolist()
    
    async def add_item(
        self,
        item_id: str,
        text: str,
        metadata: Dict[str, Any],
        backend_name: Optional[str] = None
    ) -> bool:
        """
        Add single item to vector database.
        
        Args:
            item_id: Unique item identifier
            text: Text content to encode
            metadata: Additional metadata for filtering
            backend_name: Specific backend to use (optional)
            
        Returns:
            Success status
        """
        try:
            await self._ensure_initialized()
            
            # Enhanced embedding text with error tags for better semantic matching
            enhanced_text = self._enhance_embedding_text(text, metadata)
            
            # Encode enhanced text
            vector = self.encode_text(enhanced_text)
            vector_array = np.array([vector], dtype=np.float32)
            
            # Add to index manager
            success = await vector_index_manager.add_items(
                vectors=vector_array,
                ids=[item_id],
                metadata=[metadata],
                backend_name=backend_name
            )
            
            return success
            
        except Exception as e:
            print(f"Error adding item: {e}")
            return False
    
    async def add_batch(
        self,
        items: List[Dict[str, Any]],
        backend_name: Optional[str] = None
    ) -> bool:
        """
        Add multiple items to vector database.
        
        Args:
            items: List of dicts with 'id', 'text', and 'metadata' keys
            backend_name: Specific backend to use (optional)
            
        Returns:
            Success status
        """
        try:
            await self._ensure_initialized()
            
            # Prepare data with enhanced texts
            ids = [item['id'] for item in items]
            metadata_list = [item.get('metadata', {}) for item in items]
            
            # Enhance texts with metadata for better semantic matching
            enhanced_texts = [
                self._enhance_embedding_text(item['text'], item.get('metadata', {}))
                for item in items
            ]
            
            # Encode enhanced texts
            vectors = self.encode_batch(enhanced_texts)
            vectors_array = np.array(vectors, dtype=np.float32)
            
            # Add to index manager
            success = await vector_index_manager.add_items(
                vectors=vectors_array,
                ids=ids,
                metadata=metadata_list,
                backend_name=backend_name
            )
            
            return success
            
        except Exception as e:
            print(f"Error adding batch: {e}")
            return False
    
    async def search(
        self,
        query: str,
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        backend_name: Optional[str] = None,
        use_hybrid: bool = False,
        use_mmr: bool = False,
        mmr_lambda: float = 0.7,
        use_ml_selection: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Search for similar items with ML-based backend selection.
        
        Args:
            query: Search query text
            limit: Maximum number of results
            filters: Metadata filters
            backend_name: Specific backend to use (optional)
            use_hybrid: Whether to use hybrid search across backends
            use_mmr: Whether to apply MMR reranking for diversity
            mmr_lambda: MMR lambda parameter (0.7 = relevance, 0.3 = diversity)
            use_ml_selection: Whether to use ML-based backend selection
            
        Returns:
            Search results
        """
        try:
            await self._ensure_initialized()
            
            # Encode query
            query_vector = self.encode_text(query)
            query_array = np.array([query_vector], dtype=np.float32)
            
            # ML-based backend selection
            if use_ml_selection and backend_name in (None, "", "auto"):
                backend = await self._predict_best_backend(query, query_array[0], filters)
            else:
                backend = backend_name
            
            # Perform search
            if use_hybrid and len(vector_index_manager.backends) > 1:
                # Dynamic hybrid weights based on query features
                if use_ml_selection:
                    try:
                        weights = dynamic_hybrid_weights(feats)
                        results = await vector_index_manager.hybrid_search(
                            query_vector=query_array, k=limit, 
                            filters=filters, backend_weights=weights
                        )
                    except Exception as e:
                        print(f"Dynamic weights failed: {e}")
                        results = await vector_index_manager.hybrid_search(
                            query_vector=query_array, k=limit, filters=filters
                        )
                else:
                    results = await vector_index_manager.hybrid_search(
                        query_vector=query_array, k=limit, filters=filters
                    )
            else:
                results = await vector_index_manager.search(
                    query_vector=query_array, k=limit, 
                    filters=filters, backend_name=backend
                )
            
            # Apply MMR reranking if requested
            if use_mmr and results:
                results = self._mmr_rerank(
                    query_vector=query_array[0],
                    candidates=results,
                    k=limit,
                    lambda_param=mmr_lambda
                )
            
            # Format results for backward compatibility
            formatted_results = []
            for result in results:
                formatted_results.append({
                    "item_id": result["item_id"],
                    "score": result["score"],
                    "metadata": result["metadata"]
                })
            
            return formatted_results
            
        except Exception as e:
            print(f"Error searching: {e}")
            return []
    
    def _mmr_rerank(
        self,
        query_vector: np.ndarray,
        candidates: List[Dict[str, Any]],
        k: int = 10,
        lambda_param: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Apply Maximal Marginal Relevance reranking for diversity.
        
        Args:
            query_vector: Query vector
            candidates: List of candidate results
            k: Number of results to return
            lambda_param: Balance between relevance (λ) and diversity (1-λ)
            
        Returns:
            Reranked results
        """
        try:
            from math import inf
            
            if not candidates:
                return []
            
            # Normalize scores
            max_score = max((c["score"] for c in candidates), default=1.0)
            if max_score > 0:
                for c in candidates:
                    c["normalized_score"] = c["score"] / max_score
            else:
                for c in candidates:
                    c["normalized_score"] = c["score"]
            
            # MMR algorithm
            selected = []
            remaining = candidates.copy()
            
            for _ in range(min(k, len(remaining))):
                best_candidate = None
                best_mmr_score = -inf
                
                for candidate in remaining:
                    # Relevance score
                    relevance = candidate["normalized_score"]
                    
                    # Diversity penalty (max similarity to already selected)
                    diversity_penalty = 0.0
                    if selected:
                        # Calculate max similarity to selected items
                        similarities = []
                        for selected_item in selected:
                            # Use metadata skills for diversity if available
                            if "metadata" in candidate and "metadata" in selected_item:
                                candidate_skills = set(candidate["metadata"].get("skills", []))
                                selected_skills = set(selected_item["metadata"].get("skills", []))
                                if candidate_skills and selected_skills:
                                    # Jaccard similarity for skills
                                    intersection = len(candidate_skills & selected_skills)
                                    union = len(candidate_skills | selected_skills)
                                    if union > 0:
                                        similarity = intersection / union
                                        similarities.append(similarity)
                        
                        if similarities:
                            diversity_penalty = max(similarities)
                    
                    # MMR score
                    mmr_score = lambda_param * relevance - (1 - lambda_param) * diversity_penalty
                    
                    if mmr_score > best_mmr_score:
                        best_mmr_score = mmr_score
                        best_candidate = candidate
                
                if best_candidate:
                    selected.append(best_candidate)
                    remaining.remove(best_candidate)
                else:
                    break
            
            # Clean up temporary fields
            for result in selected:
                result.pop("normalized_score", None)
            
            return selected
            
        except Exception as e:
            print(f"Error in MMR reranking: {e}")
            return candidates[:k]
    
    async def search_by_skills(
        self,
        skills: List[str],
        item_type: str,
        lang: str = "tr",
        difficulty_range: Optional[Tuple[float, float]] = None,
        limit: int = 100,
        backend_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search items by skills with metadata filtering.
        
        Args:
            skills: List of skill tags
            item_type: "math" or "english"
            lang: Language code
            difficulty_range: (min_difficulty, max_difficulty) tuple
            limit: Maximum results
            backend_name: Specific backend to use (optional)
            
        Returns:
            Search results
        """
        # Create query from skills
        query = " ".join(skills)
        
        # Build filters
        filters = {
            "type": item_type,
            "lang": lang,
            "status": "active"
        }
        
        if skills:
            filters["skills"] = skills
        
        if difficulty_range and item_type == "math":
            # Note: Range filtering would need to be implemented in the backend
            pass
        
        return await self.search(
            query=query,
            limit=limit,
            filters=filters,
            backend_name=backend_name
        )
    
    async def get_similar_items(
        self,
        item_id: str,
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        backend_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Find items similar to a given item.
        
        Args:
            item_id: Reference item ID
            limit: Maximum results
            filters: Additional filters
            backend_name: Specific backend to use (optional)
            
        Returns:
            Similar items
        """
        try:
            await self._ensure_initialized()
            
            # Get the reference item's vector
            # This would need to be implemented based on how items are stored
            # For now, we'll use a placeholder approach
            
            # TODO: Implement getting reference item vector
            print("get_similar_items not yet implemented with new index manager")
            return []
            
        except Exception as e:
            print(f"Error getting similar items: {e}")
            return []
    
    async def get_stats(self, backend_name: Optional[str] = None) -> Dict[str, Any]:
        """Get vector service statistics."""
        try:
            await self._ensure_initialized()
            
            if backend_name:
                return await vector_index_manager.get_backend_stats(backend_name)
            else:
                return await vector_index_manager.get_manager_stats()
                
        except Exception as e:
            print(f"Error getting stats: {e}")
            return {}
    
    async def health_check(self) -> Dict[str, bool]:
        """Check health of vector backends."""
        try:
            await self._ensure_initialized()
            return await vector_index_manager.health_check()
        except Exception as e:
            print(f"Error in health check: {e}")
            return {}


# Global instance for backward compatibility
vector_service = VectorService()