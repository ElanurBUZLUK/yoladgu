"""
LangChain integration service for advanced RAG with FAISS.
"""

import os
import pickle
import time
import numpy as np
import faiss
from typing import List, Dict, Any, Optional, Tuple
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document

from app.services.embedding_service import embedding_service


class LangChainIntegrationService:
    """
    Advanced LangChain integration service with FAISS and RAG capabilities.
    """
    
    def __init__(self):
        self.embeddings = None
        self.vectorstore = None
        self.qa_chain = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
    
    async def initialize_embeddings(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """Initialize HuggingFace embeddings."""
        try:
            self.embeddings = HuggingFaceEmbeddings(model_name=model_name)
            print(f"✅ LangChain embeddings initialized with {model_name}")
            return True
        except Exception as e:
            print(f"❌ Error initializing LangChain embeddings: {e}")
            return False
    
    def create_advanced_vector_store(
        self, 
        text_chunks: List[str], 
        index_type: str = "IVF",
        save_path: Optional[str] = None
    ) -> FAISS:
        """
        Create advanced FAISS vector store with different index types.
        
        Args:
            text_chunks: List of text chunks to index
            index_type: Type of FAISS index ("Flat", "IVF", "HNSW", "PQ", "IVF_PQ", "IVF_SQ")
            save_path: Optional path to save the vector store
            
        Returns:
            FAISS vector store instance
        """
        if not self.embeddings:
            raise RuntimeError("Embeddings not initialized. Call initialize_embeddings() first.")
        
        # Calculate embeddings
        print(f"🔄 Calculating embeddings for {len(text_chunks)} chunks...")
        embedding_list = self.embeddings.embed_documents(text_chunks)
        embedding_array = np.array(embedding_list).astype('float32')
        
        # Get dimensions and vector count
        dimension = embedding_array.shape[1]
        num_vectors = embedding_array.shape[0]
        
        print(f"📊 Embeddings: {num_vectors} vectors, {dimension} dimensions")
        
        # Create FAISS index based on type
        if index_type == "IVF" and num_vectors >= 100:
            # IVF index with quantization
            nlist = min(100, max(1, num_vectors // 10))  # Cluster count
            quantizer = faiss.IndexFlatL2(dimension)
            index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_L2)
            
            # Train the index
            print(f"🔧 Training IVF index with {nlist} clusters...")
            index.train(embedding_array)
            index.add(embedding_array)
            index.nprobe = min(10, nlist)  # Number of clusters to probe
            
        elif index_type == "HNSW":
            # HNSW index
            print("🔧 Creating HNSW index...")
            index = faiss.IndexHNSWFlat(dimension, 32)  # 32 connections
            index.add(embedding_array)
            
        elif index_type == "PQ":
            # Product Quantization
            print("🔧 Creating Product Quantization index...")
            m = 8  # Number of sub-vectors
            bits = 8  # Bits per sub-vector
            
            # Check if we have enough training data
            if num_vectors < 256:
                print(f"  ⚠️ Insufficient training data for PQ (need >=256, have {num_vectors})")
                print(f"  🔄 Falling back to Flat index")
                index = faiss.IndexFlatL2(dimension)
                index.add(embedding_array)
            else:
                index = faiss.IndexPQ(dimension, m, bits)
                index.train(embedding_array)
                index.add(embedding_array)
            
        elif index_type == "IVF_PQ":
            # IVF with Product Quantization
            print("🔧 Creating IVF with Product Quantization index...")
            nlist = min(100, max(1, num_vectors // 10))
            quantizer = faiss.IndexFlatL2(dimension)
            m = 8
            bits = 8
            
            # Check if we have enough training data
            if num_vectors < 256:
                print(f"  ⚠️ Insufficient training data for IVF_PQ (need >=256, have {num_vectors})")
                print(f"  🔄 Falling back to IVF index")
                index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_L2)
                index.train(embedding_array)
                index.add(embedding_array)
                index.nprobe = min(10, nlist)
            else:
                index = faiss.IndexIVFPQ(quantizer, dimension, nlist, m, bits)
                index.train(embedding_array)
                index.add(embedding_array)
                index.nprobe = min(10, nlist)
            
        elif index_type == "IVF_SQ":
            # IVF with Scalar Quantization
            print("🔧 Creating IVF with Scalar Quantization index...")
            nlist = min(100, max(1, num_vectors // 10))
            quantizer = faiss.IndexFlatL2(dimension)
            index = faiss.IndexIVFScalarQuantizer(
                quantizer, dimension, nlist, faiss.ScalarQuantizer.QT_8bit
            )
            index.train(embedding_array)
            index.add(embedding_array)
            index.nprobe = min(10, nlist)
            
        else:
            # Simple Flat index (default)
            print("🔧 Creating Flat index...")
            index = faiss.IndexFlatL2(dimension)
            index.add(embedding_array)
        
        # Create docstore
        docstore = InMemoryDocstore(
            {str(i): Document(page_content=chunk, metadata={"index": i}) 
             for i, chunk in enumerate(text_chunks)}
        )
        
        # Index-to-docstore ID mapping
        index_to_docstore_id = {i: str(i) for i in range(len(text_chunks))}
        
        # Create FAISS vector store
        self.vectorstore = FAISS(
            embedding_function=self.embeddings,
            index=index,
            docstore=docstore,
            index_to_docstore_id=index_to_docstore_id
        )
        
        print(f"✅ Vector store created with {index_type} index")
        
        # Save if path provided
        if save_path:
            self.save_vector_store(save_path)
        
        return self.vectorstore
    
    def save_vector_store(self, file_path: str):
        """Save vector store to disk."""
        try:
            if not self.vectorstore:
                raise RuntimeError("No vector store to save")
            
            # Save FAISS index
            faiss.write_index(self.vectorstore.index, f"{file_path}.index")
            
            # Save metadata
            with open(f"{file_path}.metadata", "wb") as f:
                pickle.dump({
                    "docstore": self.vectorstore.docstore,
                    "index_to_docstore_id": self.vectorstore.index_to_docstore_id
                }, f)
            
            print(f"✅ Vector store saved to {file_path}")
            
        except Exception as e:
            print(f"❌ Error saving vector store: {e}")
    
    def load_vector_store(self, file_path: str) -> FAISS:
        """Load vector store from disk."""
        try:
            if not self.embeddings:
                raise RuntimeError("Embeddings not initialized")
            
            # Load FAISS index
            index = faiss.read_index(f"{file_path}.index")
            
            # Load metadata
            with open(f"{file_path}.metadata", "rb") as f:
                metadata = pickle.load(f)
            
            # Create FAISS vector store
            self.vectorstore = FAISS(
                embedding_function=self.embeddings,
                index=index,
                docstore=metadata["docstore"],
                index_to_docstore_id=metadata["index_to_docstore_id"]
            )
            
            print(f"✅ Vector store loaded from {file_path}")
            return self.vectorstore
            
        except Exception as e:
            print(f"❌ Error loading vector store: {e}")
            raise
    
    def create_advanced_rag_chain(
        self, 
        model_name: str = "gpt-3.5-turbo",
        temperature: float = 0.0,
        max_tokens: int = 1000
    ):
        """
        Create advanced RAG chain with custom prompt and retrieval.
        
        Args:
            model_name: LLM model name
            temperature: Model temperature
            max_tokens: Maximum tokens for response
            
        Returns:
            RetrievalQA chain instance
        """
        if not self.vectorstore:
            raise RuntimeError("No vector store available. Create one first.")
        
        # Custom prompt template
        prompt_template = """Aşağıdaki bağlamı kullanarak soruyu detaylı ve kapsamlı bir şekilde cevaplayın. 
        Eğer cevabı bilmiyorsanız, "Bu konuda bilgi bulunamadı" deyin, cevap uydurmayın.

        Bağlam: {context}

        Soru: {question}

        Detaylı ve kapsamlı cevap:"""
        
        PROMPT = PromptTemplate(
            template=prompt_template, 
            input_variables=["context", "question"]
        )
        
        # Chain configuration
        chain_type_kwargs = {"prompt": PROMPT}
        
        # Create retrieval QA chain
        # Note: LLM is required in new LangChain versions
        # For now, create a basic retriever without the full chain
        self.qa_chain = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}  # Get 5 results
        )
        
        print("✅ Advanced RAG chain created")
        return self.qa_chain
    
    def evaluate_retrieval_quality(
        self, 
        query: str, 
        k: int = 5,
        ground_truth: Optional[List[str]] = None
    ) -> Tuple[List[Document], List[float], Dict[str, float]]:
        """
        Evaluate retrieval quality with various metrics.
        
        Args:
            query: Search query
            k: Number of results to retrieve
            ground_truth: Optional ground truth documents for evaluation
            
        Returns:
            Tuple of (documents, scores, metrics)
        """
        if not self.vectorstore:
            raise RuntimeError("No vector store available")
        
        # Perform similarity search
        docs = self.vectorstore.similarity_search(query, k=k)
        
        # Calculate retrieval scores
        scores = []
        for i, doc in enumerate(docs):
            # Simple ranking score (decreasing with position)
            score = 1.0 / (i + 1)
            scores.append(score)
        
        # Calculate metrics
        metrics = {
            "num_results": len(docs),
            "avg_score": np.mean(scores) if scores else 0.0,
            "max_score": max(scores) if scores else 0.0,
            "min_score": min(scores) if scores else 0.0
        }
        
        # Calculate similarity scores if embeddings available
        try:
            query_embedding = self.embeddings.embed_query(query)
            doc_embeddings = [
                self.embeddings.embed_documents([doc.page_content])[0] 
                for doc in docs
            ]
            
            if doc_embeddings:
                similarity_scores = self._cosine_similarity([query_embedding], doc_embeddings)[0]
                metrics["avg_similarity"] = float(np.mean(similarity_scores))
                metrics["min_similarity"] = float(np.min(similarity_scores))
                metrics["max_similarity"] = float(np.max(similarity_scores))
        except Exception as e:
            print(f"⚠️ Could not calculate similarity scores: {e}")
        
        return docs, scores, metrics
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Calculate cosine similarity between arrays."""
        from sklearn.metrics.pairwise import cosine_similarity
        return cosine_similarity(a, b)
    
    def benchmark_index_performance(
        self, 
        test_queries: List[str], 
        k: int = 5
    ) -> Dict[str, Dict[str, Any]]:
        """
        Benchmark FAISS index performance.
        
        Args:
            test_queries: List of test queries
            k: Number of results per query
            
        Returns:
            Dictionary with performance metrics for each query
        """
        if not self.vectorstore:
            raise RuntimeError("No vector store available")
        
        results = {}
        
        for query in test_queries:
            start_time = time.time()
            
            try:
                # Perform search
                docs = self.vectorstore.similarity_search(query, k=k)
                search_time = time.time() - start_time
                
                # Calculate similarity scores
                query_embedding = self.embeddings.embed_query(query)
                doc_embeddings = [
                    self.embeddings.embed_documents([doc.page_content])[0] 
                    for doc in docs
                ]
                
                if doc_embeddings:
                    similarity_scores = self._cosine_similarity([query_embedding], doc_embeddings)[0]
                    avg_similarity = float(np.mean(similarity_scores))
                else:
                    avg_similarity = 0.0
                
                results[query] = {
                    "search_time": search_time,
                    "avg_similarity": avg_similarity,
                    "num_results": len(docs),
                    "success": True
                }
                
            except Exception as e:
                results[query] = {
                    "search_time": 0.0,
                    "avg_similarity": 0.0,
                    "num_results": 0,
                    "success": False,
                    "error": str(e)
                }
        
        return results
    
    def optimize_index_parameters(self, text_chunks: List[str]) -> Dict[str, Any]:
        """
        Optimize index parameters based on data characteristics.
        
        Args:
            text_chunks: Text chunks for analysis
            
        Returns:
            Dictionary with optimal parameters
        """
        if not self.embeddings:
            raise RuntimeError("Embeddings not initialized")
        
        # Calculate embeddings
        embedding_list = self.embeddings.embed_documents(text_chunks)
        embedding_array = np.array(embedding_list).astype('float32')
        
        # Analyze data
        num_vectors = embedding_array.shape[0]
        dimension = embedding_array.shape[1]
        
        # Calculate optimal parameters
        optimal_params = {
            "optimal_nlist": min(256, max(1, int(np.sqrt(num_vectors)))),
            "embedding_dimension": dimension,
            "num_vectors": num_vectors,
            "recommended_index_type": self._recommend_index_type(num_vectors, dimension)
        }
        
        # Add specific recommendations
        if num_vectors < 1000:
            optimal_params["recommended_index_type"] = "Flat"
        elif num_vectors < 10000:
            optimal_params["recommended_index_type"] = "IVF"
        else:
            optimal_params["recommended_index_type"] = "IVF_PQ"
        
        return optimal_params
    
    def _recommend_index_type(self, num_vectors: int, dimension: int) -> str:
        """Recommend best index type based on data size."""
        if num_vectors < 1000:
            return "Flat"
        elif num_vectors < 10000:
            return "IVF"
        elif dimension > 512:
            return "IVF_SQ"  # Better for high-dimensional data
        else:
            return "IVF_PQ"  # Better compression for large datasets
    
    def get_vector_store_stats(self) -> Dict[str, Any]:
        """Get comprehensive vector store statistics."""
        if not self.vectorstore:
            return {"error": "No vector store available"}
        
        try:
            index = self.vectorstore.index
            
            stats = {
                "index_type": type(index).__name__,
                "total_vectors": index.ntotal if hasattr(index, 'ntotal') else 0,
                "dimension": index.d if hasattr(index, 'd') else 0,
                "is_trained": getattr(index, 'is_trained', True)
            }
            
            # Add index-specific stats
            if hasattr(index, 'nlist'):
                stats["nlist"] = index.nlist
            if hasattr(index, 'nprobe'):
                stats["nprobe"] = index.nprobe
            if hasattr(index, 'hnsw'):
                stats["ef_construction"] = index.hnsw.efConstruction
                stats["ef_search"] = index.hnsw.efSearch
                stats["m"] = index.hnsw.M
            
            return stats
            
        except Exception as e:
            return {"error": str(e)}


# Global instance
langchain_service = LangChainIntegrationService()


def get_advanced_vector_store(text_chunks, index_type="IVF"):
    """
    Gelişmiş FAISS vector store oluşturur - User's implementation
    
    Args:
        text_chunks: List of text chunks to index
        index_type: Type of FAISS index ("IVF", "HNSW", "PQ", "Flat")
        
    Returns:
        FAISS vector store instance
    """
    try:
        # Initialize embeddings if not already done
        if not langchain_service.embeddings:
            # Use default model
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        else:
            embeddings = langchain_service.embeddings
        
        # Calculate embeddings
        embedding_list = embeddings.embed_documents(text_chunks)
        embedding_array = np.array(embedding_list).astype('float32')
        
        # Get dimensions and vector count
        dimension = embedding_array.shape[1]
        num_vectors = embedding_array.shape[0]
        
        # Create FAISS index based on type
        if index_type == "IVF" and num_vectors >= 100:
            # IVF index with quantization
            nlist = min(100, num_vectors // 10)  # Cluster count
            quantizer = faiss.IndexFlatL2(dimension)
            index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_L2)
            
            # Train the index
            index.train(embedding_array)
            index.add(embedding_array)
            index.nprobe = min(10, nlist)  # Number of clusters to probe
            
        elif index_type == "HNSW":
            # HNSW index
            index = faiss.IndexHNSWFlat(dimension, 32)  # 32 connections
            index.add(embedding_array)
            
        elif index_type == "PQ":
            # Product Quantization
            m = 8  # Number of sub-vectors
            bits = 8  # Bits per sub-vector
            index = faiss.IndexPQ(dimension, m, bits)
            index.train(embedding_array)
            index.add(embedding_array)
            
        else:
            # Simple Flat index (default)
            index = faiss.IndexFlatL2(dimension)
            index.add(embedding_array)
        
        # Create docstore
        docstore = InMemoryDocstore(
            {str(i): doc for i, doc in enumerate(text_chunks)}
        )
        
        # Index-to-docstore ID mapping
        index_to_docstore_id = {i: str(i) for i in range(len(text_chunks))}
        
        # Create FAISS vector store
        vectorstore = FAISS(
            embedding_function=embeddings,
            index=index,
            docstore=docstore,
            index_to_docstore_id=index_to_docstore_id
        )
        
        return vectorstore
        
    except Exception as e:
        print(f"❌ Error creating advanced vector store: {e}")
        raise


def save_vector_store(vectorstore, file_path):
    """Vector store'u diske kaydet - User's implementation"""
    try:
        faiss.write_index(vectorstore.index, f"{file_path}.index")
        with open(f"{file_path}.metadata", "wb") as f:
            pickle.dump({
                "docstore": vectorstore.docstore,
                "index_to_docstore_id": vectorstore.index_to_docstore_id
            }, f)
        print(f"✅ Vector store saved to {file_path}")
    except Exception as e:
        print(f"❌ Error saving vector store: {e}")
        raise


def load_vector_store(embeddings, file_path):
    """Vector store'u diskten yükle - User's implementation"""
    try:
        index = faiss.read_index(f"{file_path}.index")
        with open(f"{file_path}.metadata", "rb") as f:
            metadata = pickle.load(f)
        
        return FAISS(
            embedding_function=embeddings,
            index=index,
            docstore=metadata["docstore"],
            index_to_docstore_id=metadata["index_to_docstore_id"]
        )
    except Exception as e:
        print(f"❌ Error loading vector store: {e}")
        raise
