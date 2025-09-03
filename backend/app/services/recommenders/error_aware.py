"""
Error-Aware Recommendation Service

This service provides personalized question recommendations based on student error profiles
and collaborative filtering using neighbor-lift scoring.
"""

import asyncio
import logging
from collections import defaultdict, Counter
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple, Set, Union
import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix

from app.services.vector_index_manager import vector_index_manager

logger = logging.getLogger(__name__)


@dataclass
class ErrorAwareConfig:
    """Configuration for error-aware recommendation service."""
    window_size: int = 50
    neighbor_count: int = 50
    similarity_weight: float = 0.5
    decay_factor: Optional[float] = 0.1
    min_similarity_threshold: float = 0.1
    max_candidates: int = 100
    use_hnsw: bool = True
    cache_ttl: int = 3600  # 1 hour


class ErrorAwareRecommender:
    """Error-aware recommendation service with HNSW integration."""
    
    def __init__(self, config: Optional[ErrorAwareConfig] = None):
        self.config = config or ErrorAwareConfig()
        self._cache = {}
        self._cache_timestamps = {}
        
        # Performance tracking
        self.stats = {
            "total_recommendations": 0,
            "hnsw_usage": 0,
            "exact_fallback": 0,
            "avg_response_time_ms": 0.0
        }
    
    def _recent_items(self, attempts: List[Dict[str, Any]], student_id: str, last_n: int = 50) -> Set[str]:
        """
        Get recent items attempted by a student.
        
        Args:
            attempts: List of attempt records
            student_id: Target student ID
            last_n: Number of most recent attempts to consider
            
        Returns:
            Set of recently attempted item IDs
        """
        try:
            # Filter attempts for the student
            seq = [a for a in attempts if a.get("user_id") == student_id]
            
            # Sort by timestamp if available
            if seq and any(a.get("created_at") for a in seq):
                seq.sort(key=lambda x: x.get("created_at") or 0)
            
            # Return last N item IDs
            return {a["item_id"] for a in seq[-last_n:]}
            
        except Exception as e:
            logger.error(f"Error getting recent items for student {student_id}: {e}")
            return set()
    
    def build_vocab(self, items_errors: Dict[str, List[str]]) -> List[str]:
        """
        Build vocabulary from all error tags across items.
        
        Args:
            items_errors: Mapping of item_id to list of error tags
            
        Returns:
            Sorted list of unique error tags
        """
        try:
            vocab = set()
            for tags in items_errors.values():
                if tags:  # Check if tags is not None/empty
                    vocab.update([tag for tag in tags if tag and tag.strip()])
            return sorted(list(vocab))
        except Exception as e:
            logger.error(f"Error building vocab: {e}")
            return []
    
    def _l2_normalize(self, X: NDArray[np.float32]) -> NDArray[np.float32]:
        """L2 normalize vectors for cosine similarity."""
        try:
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            norms = np.where(norms > 0, norms, 1.0)
            return X / norms
        except Exception as e:
            logger.error(f"Error in L2 normalization: {e}")
            return X
    
    def student_error_vector(
        self,
        attempts: List[Dict[str, Any]],
        items_errors: Dict[str, List[str]],
        vocab: List[str],
        student_id: str,
        window_size: Optional[int] = None,
        decay: Optional[float] = None
    ) -> NDArray[np.float32]:
        """
        Build error profile vector for a specific student.
        
        Args:
            attempts: List of attempt records
            items_errors: Mapping of item_id to error tags
            vocab: Vocabulary of error tags
            student_id: Target student ID
            window_size: Number of recent attempts to consider
            decay: Time decay factor for older attempts
            
        Returns:
            Normalized error profile vector
        """
        try:
            if not vocab or not attempts:
                return np.zeros((1, len(vocab)), dtype=np.float32)
            
            window_size = window_size or self.config.window_size
            decay = decay or self.config.decay_factor
            
            # Create error tag to index mapping
            error_to_idx = {error: i for i, error in enumerate(vocab)}
            
            # Get student's recent attempts
            student_attempts = [
                a for a in attempts 
                if a.get("user_id") == student_id
            ][-window_size:]
            
            # Build error count vector
            error_counts = Counter()
            for attempt in student_attempts:
                if attempt.get("correct", True):  # Skip correct attempts
                    continue
                
                # Calculate weight with time decay
                weight = 1.0
                if decay:
                    age_days = None
                    
                    # Try created_at first
                    if attempt.get("created_at") is not None:
                        try:
                            import datetime
                            if isinstance(attempt["created_at"], str):
                                created_at = datetime.datetime.fromisoformat(attempt["created_at"].replace('Z', '+00:00'))
                            else:
                                created_at = attempt["created_at"]
                            
                            now = datetime.datetime.utcnow()
                            if created_at.tzinfo:
                                now = now.replace(tzinfo=datetime.timezone.utc)
                            
                            dt = now - created_at
                            age_days = max(0.0, dt.total_seconds() / 86400.0)
                        except (ValueError, TypeError, AttributeError) as e:
                            logger.debug(f"Could not parse created_at for attempt {attempt.get('id')}: {e}")
                            age_days = None
                    
                    # Fallback to age_days field if created_at failed
                    if age_days is None and attempt.get("age_days") is not None:
                        try:
                            age_days = float(attempt["age_days"])
                        except (ValueError, TypeError):
                            age_days = None
                    
                    # Apply decay if we have age_days
                    if age_days is not None:
                        weight = np.exp(-decay * age_days)
                
                # Add error tags with weight
                item_errors = items_errors.get(attempt.get("item_id"), [])
                for error in item_errors:
                    if error in error_to_idx:
                        error_counts[error] += weight
            
            # Create vector
            vector = np.zeros((1, len(vocab)), dtype=np.float32)
            for error, count in error_counts.items():
                if error in error_to_idx:
                    vector[0, error_to_idx[error]] = count
            
            return self._l2_normalize(vector)
            
        except Exception as e:
            logger.error(f"Error building student vector for {student_id}: {e}")
            return np.zeros((1, len(vocab)), dtype=np.float32)
    
    def question_error_matrix(
        self,
        items_errors: Dict[str, List[str]],
        vocab: List[str],
        use_sparse: bool = True
    ) -> Tuple[Union[NDArray[np.float32], csr_matrix], List[str]]:
        """
        Build error profile matrix for all questions.
        
        Args:
            items_errors: Mapping of item_id to error tags
            vocab: Vocabulary of error tags
            use_sparse: Whether to use sparse matrix for memory efficiency
            
        Returns:
            Error profile matrix and list of question IDs
        """
        try:
            if not vocab or not items_errors:
                return np.zeros((0, len(vocab)), dtype=np.float32), []
            
            question_ids = list(items_errors.keys())
            error_to_idx = {error: i for i, error in enumerate(vocab)}
            
            if use_sparse and len(question_ids) > 1000:  # Use sparse for large datasets
                return self._build_sparse_matrix(items_errors, vocab, question_ids, error_to_idx)
            else:
                return self._build_dense_matrix(items_errors, vocab, question_ids, error_to_idx)
                
        except Exception as e:
            logger.error(f"Error building question matrix: {e}")
            return np.zeros((0, len(vocab)), dtype=np.float32), []
    
    def _build_sparse_matrix(
        self,
        items_errors: Dict[str, List[str]],
        vocab: List[str],
        question_ids: List[str],
        error_to_idx: Dict[str, int]
    ) -> Tuple[csr_matrix, List[str]]:
        """Build sparse matrix for memory efficiency."""
        rows, cols, data = [], [], []
        
        for qi, qid in enumerate(question_ids):
            for error in items_errors.get(qid, []):
                if error in error_to_idx:
                    rows.append(qi)
                    cols.append(error_to_idx[error])
                    data.append(1.0)
        
        matrix = csr_matrix((data, (rows, cols)), shape=(len(question_ids), len(vocab)))
        return matrix, question_ids
    
    def _build_dense_matrix(
        self,
        items_errors: Dict[str, List[str]],
        vocab: List[str],
        question_ids: List[str],
        error_to_idx: Dict[str, int]
    ) -> Tuple[NDArray[np.float32], List[str]]:
        """Build dense matrix for smaller datasets."""
        matrix = np.zeros((len(question_ids), len(vocab)), dtype=np.float32)
        
        for qi, qid in enumerate(question_ids):
            for error in items_errors.get(qid, []):
                if error in error_to_idx:
                    matrix[qi, error_to_idx[error]] += 1.0
        
        return self._l2_normalize(matrix), question_ids
    
    def cosine_similarity(
        self,
        query_vector: NDArray[np.float32],
        candidate_matrix: Union[NDArray[np.float32], csr_matrix]
    ) -> NDArray[np.float32]:
        """Calculate cosine similarity between query and candidates."""
        try:
            if isinstance(candidate_matrix, csr_matrix):
                # Sparse matrix multiplication
                return (query_vector @ candidate_matrix.T).toarray()[0]
            else:
                # Dense matrix multiplication
                return (query_vector @ candidate_matrix.T)[0]
        except Exception as e:
            logger.error(f"Error calculating cosine similarity: {e}")
            return np.array([])
    
    def find_similar_questions(
        self,
        student_vector: NDArray[np.float32],
        question_matrix: Union[NDArray[np.float32], csr_matrix],
        question_ids: List[str],
        k: int = 20
    ) -> List[str]:
        """
        Find questions similar to student's error profile.
        
        Args:
            student_vector: Student's error profile vector
            question_matrix: Matrix of question error profiles
            question_ids: List of question IDs
            k: Number of similar questions to return
            
        Returns:
            List of similar question IDs
        """
        try:
            if question_matrix.size == 0 or not question_ids:
                return []
            
            similarities = self.cosine_similarity(student_vector, question_matrix)
            if len(similarities) == 0:
                return []
            
            # Get top-k similar questions
            top_indices = np.argsort(-similarities)[:k]
            return [question_ids[i] for i in top_indices if i < len(question_ids)]
            
        except Exception as e:
            logger.error(f"Error finding similar questions: {e}")
            return []
    
    def find_neighbor_students(
        self,
        student_vectors: NDArray[np.float32],
        student_ids: List[str],
        target_student_id: str,
        k: int = 50
    ) -> List[str]:
        """
        Find similar students using FAISS for performance with exact fallback.
        
        Args:
            student_vectors: Matrix of student error profile vectors
            student_ids: List of student IDs
            target_student_id: Target student ID
            k: Number of neighbors to find
            
        Returns:
            List of neighbor student IDs
        """
        try:
            # L2 normalize
            X = self._l2_normalize(student_vectors.astype(np.float32))
            
            try:
                # Try FAISS first for fast search
                import faiss
                index = faiss.IndexFlatIP(X.shape[1])
                index.add(X)
                
                # Find target student index
                idx = student_ids.index(target_student_id)
                
                # Search for k+1 neighbors (including self)
                D, I = index.search(X[idx:idx+1], k+1)
                
                # Return neighbors excluding self
                neighbors = [student_ids[j] for j in I[0] if j != idx][:k]
                logger.debug(f"FAISS neighbor search found {len(neighbors)} neighbors")
                return neighbors
                
            except Exception as faiss_error:
                logger.warning(f"FAISS search failed, falling back to exact: {faiss_error}")
                
                # Exact fallback using numpy
                idx = student_ids.index(target_student_id)
                sims = (X[idx:idx+1] @ X.T)[0]
                order = np.argsort(-sims)
                
                # Return neighbors excluding self
                neighbors = [student_ids[j] for j in order if j != idx][:k]
                logger.debug(f"Exact neighbor search found {len(neighbors)} neighbors")
                return neighbors
                
        except Exception as e:
            logger.error(f"Error finding neighbor students: {e}")
            return []
    
    async def find_neighbor_students_hnsw(
        self,
        student_vectors: NDArray[np.float32],
        student_ids: List[str],
        target_student_id: str,
        k: int = 50
    ) -> List[str]:
        """
        Find similar students using HNSW for performance.
        
        Args:
            student_vectors: Matrix of student error profile vectors
            student_ids: List of student IDs
            target_student_id: Target student ID
            k: Number of neighbors to find
            
        Returns:
            List of neighbor student IDs
        """
        try:
            if not self.config.use_hnsw:
                return self._find_neighbors_exact(student_vectors, student_ids, target_student_id, k)
            
            # Find target student index
            try:
                target_idx = student_ids.index(target_student_id)
            except ValueError:
                logger.warning(f"Student {target_student_id} not found in student list")
                return []
            
            # Get target student vector
            target_vector = student_vectors[target_idx:target_idx+1]
            
            # Use HNSW for fast neighbor search
            try:
                results = await vector_index_manager.search(
                    query_vector=target_vector,
                    k=k+1,  # +1 to exclude self
                    backend_name="hnsw"
                )
                
                self.stats["hnsw_usage"] += 1
                
                # Filter out self and return neighbor IDs
                neighbor_ids = [
                    r["item_id"] for r in results 
                    if r["item_id"] != target_student_id
                ]
                return neighbor_ids[:k]
                
            except Exception as e:
                logger.warning(f"HNSW search failed, falling back to exact: {e}")
                self.stats["exact_fallback"] += 1
                return self._find_neighbors_exact(student_vectors, student_ids, target_student_id, k)
                
        except Exception as e:
            logger.error(f"Error finding neighbor students: {e}")
            return []
    
    def _find_neighbors_exact(
        self,
        student_vectors: NDArray[np.float32],
        student_ids: List[str],
        target_student_id: str,
        k: int = 50
    ) -> List[str]:
        """Fallback to exact neighbor search."""
        try:
            target_idx = student_ids.index(target_student_id)
            target_vector = student_vectors[target_idx:target_idx+1]
            
            # Calculate similarities with all students
            similarities = (target_vector @ student_vectors.T)[0]
            
            # Get top-k neighbors (excluding self)
            top_indices = np.argsort(-similarities)
            neighbors = [
                student_ids[i] for i in top_indices 
                if i != target_idx and i < len(student_ids)
            ]
            
            return neighbors[:k]
            
        except Exception as e:
            logger.error(f"Error in exact neighbor search: {e}")
            return []
    
    def calculate_lift_scores(
        self,
        attempts: List[Dict[str, Any]],
        neighbor_ids: List[str]
    ) -> Dict[str, float]:
        """
        Calculate improvement scores for questions based on neighbor performance.
        
        Args:
            attempts: List of attempt records
            neighbor_ids: List of neighbor student IDs
            
        Returns:
            Mapping of question_id to improvement score
        """
        try:
            if not neighbor_ids:
                return {}
            
            # Group attempts by (student, question) with timestamps
            student_question_attempts = defaultdict(list)
            for attempt in attempts:
                if attempt.get("user_id") in neighbor_ids:
                    key = (attempt["user_id"], attempt["item_id"])
                    ts = attempt.get("created_at") or 0
                    is_correct = int(attempt.get("correct", True))
                    student_question_attempts[key].append((ts, is_correct))
            
            # Calculate improvement scores with proper time ordering
            improvement_scores = defaultdict(int)
            for (student_id, question_id), seq in student_question_attempts.items():
                # Sort by timestamp to ensure correct order
                seq.sort(key=lambda x: x[0])
                
                seen_wrong = False
                for ts, is_correct in seq:
                    if is_correct == 0:  # Wrong answer
                        seen_wrong = True
                    elif seen_wrong and is_correct == 1:  # Correct after wrong
                        # Student learned from mistake
                        improvement_scores[question_id] += 1
                        break
            
            # Normalize by number of neighbors
            if neighbor_ids:
                normalized_scores = {
                    qid: score / len(neighbor_ids)
                    for qid, score in improvement_scores.items()
                }
            else:
                normalized_scores = {}
            
            return normalized_scores
            
        except Exception as e:
            logger.error(f"Error calculating lift scores: {e}")
            return {}
    
    async def recommend_error_aware(
        self,
        attempts: List[Dict[str, Any]],
        items_errors: Dict[str, List[str]],
        student_ids: List[str],
        student_vectors: NDArray[np.float32],
        target_student_id: str,
        alpha: Optional[float] = None,
        k: int = 10
    ) -> List[str]:
        """
        Generate error-aware recommendations for a student.
        
        Args:
            attempts: List of attempt records
            items_errors: Mapping of item_id to error tags
            student_ids: List of student IDs
            student_vectors: Matrix of student error profile vectors
            target_student_id: Target student ID
            alpha: Weight for similarity vs lift scoring
            k: Number of recommendations to return
            
        Returns:
            List of recommended question IDs
        """
        try:
            start_time = asyncio.get_event_loop().time()
            
            alpha = alpha or self.config.similarity_weight
            
            # Build vocabulary and matrices
            vocab = self.build_vocab(items_errors)
            if not vocab:
                logger.warning("No error vocabulary found")
                return []
            
            question_matrix, question_ids = self.question_error_matrix(items_errors, vocab)
            if question_matrix.size == 0:
                logger.warning("No question error profiles found")
                return []
            
            # Build target student vector
            student_vector = self.student_error_vector(
                attempts, items_errors, vocab, target_student_id
            )
            
            # Get threshold-filtered and recent-filtered candidates
            threshold = getattr(self.config, "min_similarity_threshold", 0.1)
            
            # Calculate similarities and apply threshold
            similarities = self.cosine_similarity(student_vector, question_matrix)
            order = np.argsort(-similarities)
            candidates_with_scores = [(i, float(similarities[i])) for i in order if similarities[i] >= threshold]
            
            # Get recent items to exclude
            recent_items = self._recent_items(attempts, target_student_id, last_n=self.config.window_size)
            
            # Filter out recent items and get question IDs
            candidate_ids = [
                question_ids[i] for i, _ in candidates_with_scores 
                if question_ids[i] not in recent_items
            ]
            
            # Limit candidates
            max_candidates = getattr(self.config, "max_candidates", 100)
            candidate_ids = candidate_ids[:max_candidates]
            
            if not candidate_ids:
                logger.warning(f"No valid candidates found for student {target_student_id}")
                return []
            
            # Use FAISS-based neighbor finding instead of HNSW
            neighbor_ids = self.find_neighbor_students(
                student_vectors, student_ids, target_student_id, 
                k=self.config.neighbor_count
            )
            
            # Calculate lift scores
            lift_scores = self.calculate_lift_scores(attempts, neighbor_ids)
            
            # Combine similarity and lift scores
            final_scores = []
            for question_id in candidate_ids:
                # Similarity score (rank-based)
                similarity_score = 1.0 / (1.0 + candidate_ids.index(question_id))
                
                # Lift score
                lift_score = lift_scores.get(question_id, 0.0)
                
                # Combined score
                combined_score = alpha * similarity_score + (1 - alpha) * lift_score
                
                final_scores.append((question_id, combined_score))
            
            # Sort by score and return top-k
            final_scores.sort(key=lambda x: -x[1])
            recommendations = [qid for qid, _ in final_scores[:k]]
            
            # Update statistics
            response_time = (asyncio.get_event_loop().time() - start_time) * 1000
            self.stats["total_recommendations"] += 1
            self.stats["avg_response_time_ms"] = (
                (self.stats["avg_response_time_ms"] * (self.stats["total_recommendations"] - 1) + response_time)
                / self.stats["total_recommendations"]
            )
            
            logger.info(f"Generated {len(recommendations)} recommendations for student {target_student_id}")
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations for {target_student_id}: {e}")
            return []
    
    def _mmr_diversify(
        self,
        candidates: List[str],
        student_vector: NDArray[np.float32],
        question_matrix: Union[NDArray[np.float32], csr_matrix],
        question_ids: List[str],
        lift_scores: Dict[str, float],
        k: int,
        mmr_lambda: float,
        alpha: float
    ) -> List[str]:
        """
        Apply Maximal Marginal Relevance diversification to select diverse recommendations.
        
        Args:
            candidates: List of candidate question IDs
            student_vector: Student's error profile vector
            question_matrix: Matrix of question error profiles
            question_ids: List of all question IDs
            lift_scores: Mapping of question_id to lift score
            k: Number of recommendations to return
            mmr_lambda: MMR parameter (0.0 = relevance only, 1.0 = diversity only)
            alpha: Weight for similarity vs lift scoring
            
        Returns:
            List of diversified question IDs
        """
        try:
            if not candidates:
                return []
            
            # Get candidate matrix indices
            candidate_indices = [question_ids.index(qid) for qid in candidates if qid in question_ids]
            if not candidate_indices:
                return []
            
            # Extract candidate matrix
            if isinstance(question_matrix, csr_matrix):
                candidate_matrix = question_matrix[candidate_indices]
            else:
                candidate_matrix = question_matrix[candidate_indices]
            
            # Calculate relevance scores
            relevance_scores = []
            for i, qid in enumerate(candidates):
                if qid in question_ids:
                    # Similarity score (rank-based)
                    similarity_score = 1.0 / (1.0 + candidates.index(qid))
                    
                    # Lift score
                    lift_score = lift_scores.get(qid, 0.0)
                    
                    # Combined score
                    combined_score = alpha * similarity_score + (1 - alpha) * lift_score
                    relevance_scores.append((i, combined_score))
                else:
                    relevance_scores.append((i, 0.0))
            
            # Sort by relevance
            relevance_scores.sort(key=lambda x: -x[1])
            
            # MMR selection
            selected = []
            remaining = list(range(len(candidates)))
            
            # Select first item (most relevant)
            if remaining:
                first_idx = relevance_scores[0][0]
                selected.append(first_idx)
                remaining.remove(first_idx)
            
            # Select remaining items using MMR
            while remaining and len(selected) < k:
                best_idx = None
                best_score = -float('inf')
                
                for idx in remaining:
                    # Relevance score
                    rel_score = next(score for i, score in relevance_scores if i == idx)
                    
                    # Diversity penalty (max similarity to already selected)
                    div_penalty = 0.0
                    if selected:
                        # Calculate similarity to selected items
                        similarities = []
                        for sel_idx in selected:
                            if sel_idx < len(candidate_matrix) and idx < len(candidate_matrix):
                                # Cosine similarity between candidate and selected
                                sim = self._cosine_similarity_single(
                                    candidate_matrix[idx:idx+1], 
                                    candidate_matrix[sel_idx:sel_idx+1]
                                )
                                similarities.append(sim)
                        
                        if similarities:
                            div_penalty = max(similarities)
                    
                    # MMR score
                    mmr_score = mmr_lambda * rel_score - (1.0 - mmr_lambda) * div_penalty
                    
                    if mmr_score > best_score:
                        best_score = mmr_score
                        best_idx = idx
                
                if best_idx is not None:
                    selected.append(best_idx)
                    remaining.remove(best_idx)
                else:
                    break
            
            # Return selected question IDs
            return [candidates[i] for i in selected if i < len(candidates)]
            
        except Exception as e:
            logger.error(f"Error in MMR diversification: {e}")
            # Fallback to simple ranking
            return candidates[:k]
    
    def _cosine_similarity_single(
        self, 
        vec1: Union[NDArray[np.float32], csr_matrix], 
        vec2: Union[NDArray[np.float32], csr_matrix]
    ) -> float:
        """Calculate cosine similarity between two single vectors."""
        try:
            if isinstance(vec1, csr_matrix) and isinstance(vec2, csr_matrix):
                # Sparse matrix multiplication
                dot_product = (vec1 @ vec2.T).toarray()[0, 0]
                norm1 = np.linalg.norm(vec1.toarray())
                norm2 = np.linalg.norm(vec2.toarray())
            else:
                # Dense matrix multiplication
                dot_product = (vec1 @ vec2.T)[0, 0]
                norm1 = np.linalg.norm(vec1)
                norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return float(dot_product / (norm1 * norm2))
            
        except Exception as e:
            logger.error(f"Error calculating single cosine similarity: {e}")
            return 0.0
    
    def _get_question_metadata(self, question_id: str) -> Dict[str, Any]:
        """
        Get metadata for a question. Override this method based on your data structure.
        
        Args:
            question_id: Question ID
            
        Returns:
            Dictionary containing question metadata
        """
        # This is a placeholder - implement based on your data structure
        # Example: return {"difficulty": "medium", "skills": ["algebra", "equations"]}
        return {}
    
    def _passes_metadata_filters(
        self, 
        metadata: Dict[str, Any], 
        difficulty_filter: Optional[str], 
        skills_filter: Optional[List[str]]
    ) -> bool:
        """
        Check if question metadata passes the specified filters.
        
        Args:
            metadata: Question metadata
            difficulty_filter: Comma-separated difficulty levels
            skills_filter: Required skills
            
        Returns:
            True if question passes all filters
        """
        try:
            # Check difficulty filter
            if difficulty_filter:
                allowed_difficulties = set(d.strip() for d in difficulty_filter.split(","))
                question_difficulty = metadata.get("difficulty", "").strip()
                if question_difficulty not in allowed_difficulties:
                    return False
            
            # Check skills filter
            if skills_filter:
                required_skills = set(s.strip() for s in skills_filter)
                question_skills = set(s.strip() for s in metadata.get("skills", []))
                if not (required_skills & question_skills):  # No intersection
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking metadata filters: {e}")
            return True  # Default to passing if there's an error
    
    def get_stats(self) -> Dict[str, Any]:
        """Get service statistics."""
        return {
            **self.stats,
            "config": {
                "window_size": self.config.window_size,
                "neighbor_count": self.config.neighbor_count,
                "similarity_weight": self.config.similarity_weight,
                "use_hnsw": self.config.use_hnsw
            }
        }
    
    def clear_cache(self) -> None:
        """Clear recommendation cache."""
        self._cache.clear()
        self._cache_timestamps.clear()
        logger.info("Recommendation cache cleared")


# Global instance for backward compatibility
error_aware_recommender = ErrorAwareRecommender()


# =============================================================================
# BACKWARD COMPATIBILITY FUNCTIONS
# =============================================================================

def build_vocab(items_errors: Dict[str, List[str]]) -> List[str]:
    """Backward compatibility function."""
    return error_aware_recommender.build_vocab(items_errors)


def student_vec(
    attempts: List[Dict[str, Any]], 
    items_errors: Dict[str, List[str]], 
    vocab: List[str], 
    sid: str, 
    W: int = 50, 
    decay: Optional[float] = None
) -> NDArray[np.float32]:
    """Backward compatibility function."""
    return error_aware_recommender.student_error_vector(
        attempts, items_errors, vocab, sid, W, decay
    )


def question_matrix(
    items_errors: Dict[str, List[str]], 
    vocab: List[str]
) -> Tuple[NDArray[np.float32], List[str]]:
    """Backward compatibility function."""
    matrix, qids = error_aware_recommender.question_error_matrix(items_errors, vocab, use_sparse=False)
    return matrix, qids


def cosine(a: NDArray[np.float32], B: NDArray[np.float32]) -> NDArray[np.float32]:
    """Backward compatibility function."""
    return error_aware_recommender.cosine_similarity(a, B)


def topk_similar_questions(
    h_s: NDArray[np.float32], 
    QX: NDArray[np.float32], 
    qids: List[str], 
    k: int = 20
) -> List[str]:
    """Backward compatibility function."""
    return error_aware_recommender.find_similar_questions(h_s, QX, qids, k)


def neighbor_ids(
    h_students: NDArray[np.float32], 
    sids: List[str], 
    sid: str, 
    k: int = 50
) -> List[str]:
    """Backward compatibility function (synchronous)."""
    # Note: This is synchronous, for async version use find_neighbor_students_hnsw
    return error_aware_recommender._find_neighbors_exact(h_students, sids, sid, k)


def lift_scores(attempts: List[Dict[str, Any]], neigh: List[str]) -> Dict[str, float]:
    """Backward compatibility function."""
    return error_aware_recommender.calculate_lift_scores(attempts, neigh)


async def recommend_error_aware(
    attempts: List[Dict[str, Any]],
    items_errors: Dict[str, List[str]],
    sids: List[str],
    h_students: NDArray[np.float32],
    sid: str,
    alpha: float = 0.5,
    k: int = 10
) -> List[str]:
    """Backward compatibility function (async)."""
    return await error_aware_recommender.recommend_error_aware(
        attempts, items_errors, sids, h_students, sid, alpha, k
    )


# Synchronous version for backward compatibility
def recommend_error_aware_sync(
    attempts: List[Dict[str, Any]],
    items_errors: Dict[str, List[str]],
    sids: List[str],
    h_students: NDArray[np.float32],
    sid: str,
    alpha: float = 0.5,
    k: int = 10
) -> List[str]:
    """Synchronous version for backward compatibility."""
    try:
        # Run async function in sync context
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If we're already in an async context, create a new loop
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run,
                    error_aware_recommender.recommend_error_aware(
                        attempts, items_errors, sids, h_students, sid, alpha, k
                    )
                )
                return future.result()
        else:
            return loop.run_until_complete(
                error_aware_recommender.recommend_error_aware(
                    attempts, items_errors, sids, h_students, sid, alpha, k
                )
            )
    except Exception as e:
        logger.error(f"Error in sync recommendation: {e}")
        return []
