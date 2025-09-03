"""
Feature Extraction Service for ML-based Backend Selection

This service extracts relevant features from search queries and index statistics
to help the ML model make optimal backend selection decisions.
"""

import numpy as np
from typing import Dict, Any, List, Optional, Union
import logging

logger = logging.getLogger(__name__)


def extract_query_features(
    query_text: str,
    query_vector: np.ndarray,
    k: int,
    filters: Optional[Dict[str, Any]] = None,
    index_stats: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Extract features from a search query for backend selection.
    
    Args:
        query_text: Original query text
        query_vector: Encoded query vector
        k: Number of results requested
        filters: Optional filters to apply
        index_stats: Current index statistics
        
    Returns:
        Dictionary of extracted features
    """
    try:
        features = {}
        
        # Query vector properties
        features.update(_extract_vector_features(query_vector))
        
        # Search parameters
        features.update(_extract_search_features(k, filters))
        
        # Index statistics
        if index_stats:
            features.update(_extract_index_features(index_stats))
        else:
            # Default values if no stats available
            features.update(_get_default_index_features())
        
        # Query text features
        features.update(_extract_text_features(query_text))
        
        return features
        
    except Exception as e:
        logger.error(f"Error extracting query features: {e}")
        return _get_default_features()


def _extract_vector_features(query_vector: np.ndarray) -> Dict[str, float]:
    """Extract features from the query vector."""
    try:
        # Ensure query_vector is 2D
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
        
        vector = query_vector[0]  # Get first vector
        
        return {
            'query_norm': float(np.linalg.norm(vector)),
            'query_std': float(np.std(vector)),
            'query_max': float(np.max(vector)),
            'query_min': float(np.min(vector)),
            'query_mean': float(np.mean(vector)),
            'query_median': float(np.median(vector)),
            'query_range': float(np.max(vector) - np.min(vector)),
            'query_sparsity': float(np.sum(vector == 0) / len(vector)),
            'query_entropy': float(_calculate_entropy(vector))
        }
        
    except Exception as e:
        logger.error(f"Error extracting vector features: {e}")
        return {
            'query_norm': 1.0,
            'query_std': 0.0,
            'query_max': 0.0,
            'query_min': 0.0,
            'query_mean': 0.0,
            'query_median': 0.0,
            'query_range': 0.0,
            'query_sparsity': 0.0,
            'query_entropy': 0.0
        }


def _extract_search_features(k: int, filters: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
    """Extract features from search parameters."""
    try:
        features = {
            'k_value': float(k),
            'k_log': float(np.log(k + 1)),  # Log scale for large k values
            'k_squared': float(k ** 2),     # Quadratic scale for very large k
            'has_filters': 1.0 if filters else 0.0,
            'filter_count': float(len(filters)) if filters else 0.0,
            'filter_depth': float(_calculate_filter_depth(filters)) if filters else 0.0,
            'complex_filters': float(_is_complex_filters(filters)) if filters else 0.0
        }
        
        # K value categories
        features['k_small'] = 1.0 if k <= 10 else 0.0
        features['k_medium'] = 1.0 if 10 < k <= 50 else 0.0
        features['k_large'] = 1.0 if k > 50 else 0.0
        
        return features
        
    except Exception as e:
        logger.error(f"Error extracting search features: {e}")
        return {
            'k_value': float(k),
            'k_log': float(np.log(k + 1)),
            'k_squared': float(k ** 2),
            'has_filters': 0.0,
            'filter_count': 0.0,
            'filter_depth': 0.0,
            'complex_filters': 0.0,
            'k_small': 1.0 if k <= 10 else 0.0,
            'k_medium': 1.0 if 10 < k <= 50 else 0.0,
            'k_large': 1.0 if k > 50 else 0.0
        }


def _extract_index_features(index_stats: Dict[str, Any]) -> Dict[str, float]:
    """Extract features from index statistics."""
    try:
        features = {}
        
        # Basic index stats
        features['total_items'] = float(index_stats.get('total_items', 0))
        features['total_backends'] = float(len(index_stats.get('available_backends', [])))
        
        # Backend availability
        available_backends = index_stats.get('available_backends', [])
        features['hnsw_available'] = 1.0 if 'hnsw' in available_backends else 0.0
        features['faiss_available'] = 1.0 if 'faiss' in available_backends else 0.0
        features['qdrant_available'] = 1.0 if 'qdrant' in available_backends else 0.0
        
        # Backend details
        backend_details = index_stats.get('backend_details', {})
        
        # Calculate average items per backend
        total_items = 0
        backend_count = 0
        for backend_name, backend_stats in backend_details.items():
            if isinstance(backend_stats, dict):
                items = backend_stats.get('total_items', 0)
                total_items += items
                backend_count += 1
        
        features['avg_items_per_backend'] = float(total_items / max(backend_count, 1))
        
        # Recent performance metrics
        features['recent_hnsw_performance'] = float(
            backend_details.get('hnsw', {}).get('avg_search_time_ms', 1.0)
        )
        features['recent_faiss_performance'] = float(
            backend_details.get('faiss', {}).get('avg_search_time_ms', 1.0)
        )
        features['recent_qdrant_performance'] = float(
            backend_details.get('qdrant', {}).get('avg_search_time_ms', 1.0)
        )
        
        # Performance ratios
        if features['recent_hnsw_performance'] > 0 and features['recent_faiss_performance'] > 0:
            features['hnsw_faiss_ratio'] = features['recent_hnsw_performance'] / features['recent_faiss_performance']
        else:
            features['hnsw_faiss_ratio'] = 1.0
            
        if features['recent_qdrant_performance'] > 0 and features['recent_hnsw_performance'] > 0:
            features['qdrant_hnsw_ratio'] = features['recent_qdrant_performance'] / features['recent_hnsw_performance']
        else:
            features['qdrant_hnsw_ratio'] = 1.0
        
        return features
        
    except Exception as e:
        logger.error(f"Error extracting index features: {e}")
        return _get_default_index_features()


def _extract_text_features(query_text: str) -> Dict[str, float]:
    """Extract features from query text."""
    try:
        if not query_text:
            return _get_default_text_features()
        
        # Text length features
        text_length = len(query_text)
        word_count = len(query_text.split())
        
        features = {
            'text_length': float(text_length),
            'word_count': float(word_count),
            'avg_word_length': float(text_length / max(word_count, 1)),
            'text_length_log': float(np.log(text_length + 1)),
            'word_count_log': float(np.log(word_count + 1))
        }
        
        # Text complexity features
        features['has_numbers'] = 1.0 if any(c.isdigit() for c in query_text) else 0.0
        features['has_special_chars'] = 1.0 if any(not c.isalnum() and not c.isspace() for c in query_text) else 0.0
        features['has_uppercase'] = 1.0 if any(c.isupper() for c in query_text) else 0.0
        
        # Language features (simple heuristics)
        features['likely_english'] = 1.0 if _is_likely_english(query_text) else 0.0
        features['likely_turkish'] = 1.0 if _is_likely_turkish(query_text) else 0.0
        
        return features
        
    except Exception as e:
        logger.error(f"Error extracting text features: {e}")
        return _get_default_text_features()


def _calculate_filter_depth(filters: Dict[str, Any]) -> int:
    """Calculate the depth of nested filters."""
    try:
        if not filters:
            return 0
        
        max_depth = 1
        for value in filters.values():
            if isinstance(value, dict):
                max_depth = max(max_depth, 1 + _calculate_filter_depth(value))
            elif isinstance(value, list):
                max_depth = max(max_depth, 2)  # List adds one level of complexity
        
        return max_depth
        
    except Exception:
        return 1


def _is_complex_filters(filters: Dict[str, Any]) -> bool:
    """Determine if filters are complex (require advanced querying)."""
    try:
        if not filters:
            return False
        
        # Check for complex filter patterns
        for key, value in filters.items():
            if isinstance(value, list) and len(value) > 3:
                return True  # Multiple values
            if isinstance(value, dict):
                return True  # Nested filters
            if key in ['range', 'geo', 'date']:
                return True  # Special filter types
        
        return False
        
    except Exception:
        return False


def _calculate_entropy(vector: np.ndarray) -> float:
    """Calculate entropy of the vector distribution."""
    try:
        # Normalize to probabilities
        vector_abs = np.abs(vector)
        if np.sum(vector_abs) == 0:
            return 0.0
        
        probs = vector_abs / np.sum(vector_abs)
        probs = probs[probs > 0]  # Remove zero probabilities
        
        # Calculate entropy: -sum(p * log(p))
        entropy = -np.sum(probs * np.log(probs))
        return float(entropy)
        
    except Exception:
        return 0.0


def _is_likely_english(text: str) -> bool:
    """Simple heuristic to detect English text."""
    try:
        english_chars = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ')
        text_chars = set(text.lower())
        english_ratio = len(text_chars.intersection(english_chars)) / max(len(text_chars), 1)
        return english_ratio > 0.7
        
    except Exception:
        return False


def _is_likely_turkish(text: str) -> bool:
    """Simple heuristic to detect Turkish text."""
    try:
        turkish_chars = set('çğıöşüÇĞIİÖŞÜ')
        text_chars = set(text)
        turkish_ratio = len(text_chars.intersection(turkish_chars)) / max(len(text), 1)
        return turkish_ratio > 0.1
        
    except Exception:
        return False


def _get_default_features() -> Dict[str, float]:
    """Get default feature values when extraction fails."""
    return {
        **_get_default_vector_features(),
        **_get_default_search_features(),
        **_get_default_index_features(),
        **_get_default_text_features()
    }


def _get_default_vector_features() -> Dict[str, float]:
    """Default vector features."""
    return {
        'query_norm': 1.0,
        'query_std': 0.0,
        'query_max': 0.0,
        'query_min': 0.0,
        'query_mean': 0.0,
        'query_median': 0.0,
        'query_range': 0.0,
        'query_sparsity': 0.0,
        'query_entropy': 0.0
    }


def _get_default_search_features() -> Dict[str, float]:
    """Default search features."""
    return {
        'k_value': 10.0,
        'k_log': np.log(11),
        'k_squared': 100.0,
        'has_filters': 0.0,
        'filter_count': 0.0,
        'filter_depth': 0.0,
        'complex_filters': 0.0,
        'k_small': 1.0,
        'k_medium': 0.0,
        'k_large': 0.0
    }


def _get_default_index_features() -> Dict[str, float]:
    """Default index features."""
    return {
        'total_items': 10000.0,
        'total_backends': 3.0,
        'hnsw_available': 1.0,
        'faiss_available': 1.0,
        'qdrant_available': 1.0,
        'avg_items_per_backend': 3333.0,
        'recent_hnsw_performance': 1.0,
        'recent_faiss_performance': 1.0,
        'recent_qdrant_performance': 1.0,
        'hnsw_faiss_ratio': 1.0,
        'qdrant_hnsw_ratio': 1.0
    }


def _get_default_text_features() -> Dict[str, float]:
    """Default text features."""
    return {
        'text_length': 10.0,
        'word_count': 2.0,
        'avg_word_length': 5.0,
        'text_length_log': np.log(11),
        'word_count_log': np.log(3),
        'has_numbers': 0.0,
        'has_special_chars': 0.0,
        'has_uppercase': 0.0,
        'likely_english': 1.0,
        'likely_turkish': 0.0
    }


def dynamic_hybrid_weights(
    features: Dict[str, Any],
    base_weights: Optional[Dict[str, float]] = None
) -> Dict[str, float]:
    """
    Generate dynamic hybrid search weights based on query features.
    
    Args:
        features: Extracted query features
        base_weights: Base weights to adjust
        
    Returns:
        Dictionary of backend weights
    """
    try:
        # Default base weights
        if base_weights is None:
            base_weights = {
                'hnsw': 0.4,
                'faiss': 0.3,
                'qdrant': 0.3
            }
        
        # Start with base weights
        weights = base_weights.copy()
        
        # Adjust based on query characteristics
        
        # K value adjustments
        k_value = features.get('k_value', 10)
        if k_value > 50:
            # Large k: favor FAISS for accuracy
            weights['faiss'] = min(weights['faiss'] * 1.5, 0.6)
            weights['hnsw'] = max(weights['hnsw'] * 0.8, 0.2)
        elif k_value < 10:
            # Small k: favor HNSW for speed
            weights['hnsw'] = min(weights['hnsw'] * 1.3, 0.6)
            weights['faiss'] = max(weights['faiss'] * 0.9, 0.2)
        
        # Filter adjustments
        if features.get('has_filters', 0) > 0:
            # Filters: favor Qdrant
            weights['qdrant'] = min(weights['qdrant'] * 1.4, 0.6)
            weights['hnsw'] = max(weights['hnsw'] * 0.7, 0.2)
        
        # Performance-based adjustments
        hnsw_perf = features.get('recent_hnsw_performance', 1.0)
        faiss_perf = features.get('recent_faiss_performance', 1.0)
        qdrant_perf = features.get('recent_qdrant_performance', 1.0)
        
        # Normalize performance (lower is better)
        total_perf = hnsw_perf + faiss_perf + qdrant_perf
        if total_perf > 0:
            hnsw_weight = (total_perf - hnsw_perf) / total_perf
            faiss_weight = (total_perf - faiss_perf) / total_perf
            qdrant_weight = (total_perf - qdrant_perf) / total_perf
            
            # Blend with current weights
            weights['hnsw'] = 0.7 * weights['hnsw'] + 0.3 * hnsw_weight
            weights['faiss'] = 0.7 * weights['faiss'] + 0.3 * faiss_weight
            weights['qdrant'] = 0.7 * weights['qdrant'] + 0.3 * qdrant_weight
        
        # Normalize weights to sum to 1.0
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}
        
        return weights
        
    except Exception as e:
        logger.error(f"Error calculating dynamic weights: {e}")
        # Return base weights on error
        return base_weights or {'hnsw': 0.4, 'faiss': 0.3, 'qdrant': 0.3}
