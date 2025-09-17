"""
Hybrid Search Index - BM25 + Dense Vector Index
Supports both lexical (BM25) and semantic (E5) search
"""

import faiss
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import structlog
from collections import defaultdict
import json
import pickle
import os
from dataclasses import dataclass

logger = structlog.get_logger()

@dataclass
class SearchResult:
    """Search result with metadata"""
    doc_id: str
    score: float
    text: str
    metadata: Dict[str, Any]

class BM25Index:
    """Simple BM25 implementation for lexical search"""
    
    def __init__(self):
        self.documents = {}
        self.vocab = set()
        self.doc_freqs = defaultdict(int)
        self.doc_lengths = {}
        self.avg_doc_length = 0.0
        self.k1 = 1.2
        self.b = 0.75
        
    def add_documents(self, documents: List[Dict[str, Any]]):
        """Add documents to BM25 index"""
        for doc in documents:
            doc_id = doc["id"]
            text = doc["text"]
            
            # Tokenize (simple whitespace tokenization)
            tokens = text.lower().split()
            
            self.documents[doc_id] = {
                "text": text,
                "tokens": tokens,
                "metadata": doc.get("metadata", {})
            }
            
            self.vocab.update(tokens)
            self.doc_lengths[doc_id] = len(tokens)
            
            # Count document frequencies
            for token in set(tokens):
                self.doc_freqs[token] += 1
        
        # Calculate average document length
        if self.doc_lengths:
            self.avg_doc_length = sum(self.doc_lengths.values()) / len(self.doc_lengths)
        
        logger.info("BM25 index built with %d documents", len(self.documents))
    
    def search(self, query: str, k: int = 10) -> List[SearchResult]:
        """Search using BM25"""
        if not self.documents:
            return []
        
        query_tokens = query.lower().split()
        scores = {}
        
        for doc_id, doc_data in self.documents.items():
            score = 0.0
            doc_length = self.doc_lengths[doc_id]
            
            for token in query_tokens:
                if token in doc_data["tokens"]:
                    tf = doc_data["tokens"].count(token)
                    idf = np.log((len(self.documents) - self.doc_freqs[token] + 0.5) / 
                               (self.doc_freqs[token] + 0.5))
                    
                    # BM25 formula
                    score += idf * (tf * (self.k1 + 1)) / \
                            (tf + self.k1 * (1 - self.b + self.b * doc_length / self.avg_doc_length))
            
            scores[doc_id] = score
        
        # Sort by score and return top k
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
        
        results = []
        for doc_id, score in sorted_docs:
            results.append(SearchResult(
                doc_id=doc_id,
                score=score,
                text=self.documents[doc_id]["text"],
                metadata=self.documents[doc_id]["metadata"]
            ))
        
        return results

class DenseVectorIndex:
    """FAISS-based dense vector index for semantic search"""
    
    def __init__(self, vector_size: int = 1024, index_type: str = "hnsw"):
        self.vector_size = vector_size
        self.index_type = index_type
        self.index = None
        self.documents = {}
        self._built = False
        
    def initialize(self):
        """Initialize FAISS index"""
        try:
            if self.index_type == "hnsw":
                # HNSW for fast approximate search
                self.index = faiss.IndexHNSWFlat(self.vector_size, 32)
                self.index.hnsw.efConstruction = 400
                self.index.hnsw.efSearch = 128
            elif self.index_type == "ivf":
                # IVF for large datasets
                quantizer = faiss.IndexFlatIP(self.vector_size)
                self.index = faiss.IndexIVFFlat(quantizer, self.vector_size, 100)
                self.index.nprobe = 10
            else:
                # Flat index for exact search
                self.index = faiss.IndexFlatIP(self.vector_size)
            
            self._built = True
            logger.info("Dense vector index initialized: %s", self.index_type)
            return True
            
        except Exception as e:
            logger.error("Failed to initialize dense index: %s", e)
            return False
    
    def add_vectors(self, vectors: np.ndarray, documents: List[Dict[str, Any]]):
        """Add vectors and documents to index"""
        if not self._built:
            raise RuntimeError("Index not initialized")
        
        # Store documents
        for i, doc in enumerate(documents):
            self.documents[i] = doc
        
        # Add vectors to index
        if hasattr(self.index, 'is_trained') and not self.index.is_trained:
            # Train IVF index if needed
            self.index.train(vectors)
        
        self.index.add(vectors.astype(np.float32))
        logger.info("Added %d vectors to dense index", len(vectors))
    
    def search(self, query_vector: np.ndarray, k: int = 10) -> List[SearchResult]:
        """Search using dense vectors"""
        if not self._built or self.index.ntotal == 0:
            return []
        
        # Ensure query vector is 2D
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
        
        # Search
        scores, indices = self.index.search(query_vector.astype(np.float32), k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(self.documents):  # Check for valid index
                doc = self.documents[idx]
                results.append(SearchResult(
                    doc_id=doc["id"],
                    score=float(score),
                    text=doc["text"],
                    metadata=doc.get("metadata", {})
                ))
        
        return results

class HybridSearchIndex:
    """Combined BM25 + Dense Vector Index for hybrid search"""
    
    def __init__(self, vector_size: int = 1024, index_type: str = "hnsw"):
        self.bm25_index = BM25Index()
        self.dense_index = DenseVectorIndex(vector_size, index_type)
        self.vector_size = vector_size
        self._initialized = False
        
    async def initialize(self) -> bool:
        """Initialize both indices"""
        try:
            success = self.dense_index.initialize()
            if success:
                self._initialized = True
                logger.info("Hybrid search index initialized")
            return success
        except Exception as e:
            logger.error("Failed to initialize hybrid index: %s", e)
            return False
    
    def add_documents(self, documents: List[Dict[str, Any]], vectors: np.ndarray):
        """Add documents to both indices"""
        # Add to BM25
        self.bm25_index.add_documents(documents)
        
        # Add to dense index
        self.dense_index.add_vectors(vectors, documents)
        
        logger.info("Added %d documents to hybrid index", len(documents))
    
    def search_bm25(self, query: str, k: int = 100) -> List[SearchResult]:
        """Search using BM25 only"""
        return self.bm25_index.search(query, k)
    
    def search_dense(self, query_vector: np.ndarray, k: int = 100) -> List[SearchResult]:
        """Search using dense vectors only"""
        return self.dense_index.search(query_vector, k)
    
    def search_hybrid(self, query: str, query_vector: np.ndarray, k: int = 10, 
                     bm25_weight: float = 0.5, dense_weight: float = 0.5) -> List[SearchResult]:
        """Hybrid search combining BM25 and dense results"""
        # Get results from both indices
        bm25_results = self.search_bm25(query, k * 2)
        dense_results = self.search_dense(query_vector, k * 2)
        
        # Combine using weighted scores
        combined_scores = {}
        
        # Add BM25 scores
        for result in bm25_results:
            combined_scores[result.doc_id] = {
                "score": result.score * bm25_weight,
                "bm25_score": result.score,
                "dense_score": 0.0,
                "text": result.text,
                "metadata": result.metadata
            }
        
        # Add dense scores
        for result in dense_results:
            if result.doc_id in combined_scores:
                combined_scores[result.doc_id]["score"] += result.score * dense_weight
                combined_scores[result.doc_id]["dense_score"] = result.score
            else:
                combined_scores[result.doc_id] = {
                    "score": result.score * dense_weight,
                    "bm25_score": 0.0,
                    "dense_score": result.score,
                    "text": result.text,
                    "metadata": result.metadata
                }
        
        # Sort by combined score
        sorted_results = sorted(combined_scores.items(), key=lambda x: x[1]["score"], reverse=True)
        
        # Return top k results
        results = []
        for doc_id, data in sorted_results[:k]:
            results.append(SearchResult(
                doc_id=doc_id,
                score=data["score"],
                text=data["text"],
                metadata=data["metadata"]
            ))
        
        return results
    
    @property
    def is_initialized(self) -> bool:
        """Check if index is initialized"""
        return self._initialized
    
    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics"""
        return {
            "bm25_documents": len(self.bm25_index.documents),
            "dense_vectors": self.dense_index.index.ntotal if self.dense_index.index else 0,
            "vector_size": self.vector_size,
            "index_type": self.dense_index.index_type
        }
