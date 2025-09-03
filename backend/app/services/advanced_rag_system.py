"""
Advanced RAG System with FAISS backend integration.
"""

import asyncio
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from sentence_transformers import SentenceTransformer

from app.services.index_backends.faiss_advanced_index import FAISSAdvancedIndexBackend


class AdvancedRAGSystem:
    """
    Advanced RAG System with FAISS backend integration.
    """
    
    def __init__(self):
        self.vector_store = None
        self.qa_chain = None
        self.embedding_model = None
        
    async def initialize_vector_store(self, index_type="ivf"):
        """Initialize the advanced FAISS vector store"""
        # Embedding boyutunu belirle (all-MiniLM-L6-v2 için 384)
        embedding_size = 384
        
        # Adjust nlist based on expected document count
        if index_type == "ivf":
            nlist = min(50, max(1, 100 // 10))  # Smaller nlist for small datasets
            nprobe = min(5, nlist)
        else:
            nlist = 100
            nprobe = 10
        
        self.vector_store = FAISSAdvancedIndexBackend(
            vector_size=embedding_size,
            index_type=index_type,
            metric="ip",  # Inner product for cosine similarity
            nlist=nlist,
            nprobe=nprobe,
            index_path=f"data/faiss_{index_type}.index"
        )
        
        await self.vector_store.initialize()
        return self.vector_store
    
    async def add_documents(self, documents: List[str], metadata_list: Optional[List[Dict[str, Any]]] = None):
        """Add documents to the vector store"""
        if metadata_list is None:
            metadata_list = [{"content": doc} for doc in documents]
        else:
            # Ensure each metadata has content
            for i, metadata in enumerate(metadata_list):
                if "content" not in metadata:
                    metadata["content"] = documents[i]
        
        # Document'ları vektörlere dönüştür
        embeddings = self._generate_embeddings(documents)
        
        # ID'ler oluştur
        ids = [f"doc_{i}" for i in range(len(documents))]
        
        # Vector store'a ekle
        success = await self.vector_store.add_items(embeddings, ids, metadata_list)
        return success
    
    def _generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for texts"""
        if self.embedding_model is None:
            self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        # Return numpy array for FAISS compatibility
        embeddings = self.embedding_model.encode(texts)
        return embeddings
    
    async def query(self, question: str, k: int = 5) -> Tuple[str, List[Dict[str, Any]]]:
        """Query the RAG system"""
        # Soruyu vektöre dönüştür
        query_embedding = self._generate_embeddings([question])[0]
        
        # Benzer dokümanları bul
        results = await self.vector_store.search(query_embedding, k=k)
        
        # RAG cevabı oluştur
        context = "\n\n".join([f"Document {i+1}: {r['metadata'].get('content', '')}" 
                              for i, r in enumerate(results)])
        
        answer = await self._generate_answer(question, context)
        return answer, results
    
    async def _generate_answer(self, question: str, context: str) -> str:
        """Generate answer using LLM"""
        try:
            from langchain_community.chat_models import ChatOpenAI
            from langchain.schema import HumanMessage, SystemMessage
            
            llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
            
            prompt = f"""
            Aşağıdaki bağlamı kullanarak soruyu cevaplayın. Eğer cevabı bilmiyorsanız, 
            "Bilmiyorum" deyin, cevap uydurmayın.

            Bağlam: {context}

            Soru: {question}
            Detaylı ve kapsamlı cevap:
            """
            
            response = await llm.agenerate([[
                SystemMessage(content="Sen bir yardımcı asistansın."),
                HumanMessage(content=prompt)
            ]])
            
            return response.generations[0][0].text
            
        except ImportError:
            # Fallback if LangChain is not available
            return f"LLM integration not available. Retrieved {len(context.split('Document'))} relevant documents."
        except Exception as e:
            return f"Error generating answer: {str(e)}"
    
    async def get_vector_store_stats(self) -> Dict[str, Any]:
        """Get vector store statistics"""
        if self.vector_store:
            return await self.vector_store.get_stats()
        return {"error": "Vector store not initialized"}
    
    async def save_vector_store(self, path: str) -> bool:
        """Save vector store to disk"""
        if self.vector_store:
            try:
                await self.vector_store.save_index(path)
                return True
            except Exception as e:
                print(f"Error saving vector store: {e}")
                return False
        return False
    
    async def load_vector_store(self, path: str) -> bool:
        """Load vector store from disk"""
        try:
            # Try to load existing index
            await self.vector_store.load_index(path)
            return True
        except Exception as e:
            print(f"Error loading vector store: {e}")
            return False
    
    async def optimize_index(self, documents: List[str]) -> Dict[str, Any]:
        """Optimize index parameters based on document characteristics"""
        if not documents:
            return {"error": "No documents provided"}
        
        # Analyze document characteristics
        total_chars = sum(len(doc) for doc in documents)
        avg_doc_length = total_chars / len(documents)
        
        # Recommend optimal parameters
        recommendations = {
            "num_documents": len(documents),
            "total_characters": total_chars,
            "average_document_length": avg_doc_length,
            "recommended_chunk_size": min(1000, max(200, int(avg_doc_length * 0.8))),
            "recommended_index_type": self._recommend_index_type(len(documents)),
            "memory_estimate_mb": self._estimate_memory_usage(len(documents))
        }
        
        return recommendations
    
    def _recommend_index_type(self, num_documents: int) -> str:
        """Recommend best index type based on document count"""
        if num_documents < 1000:
            return "flat"  # Simple and fast for small datasets
        elif num_documents < 10000:
            return "ivf"   # Good balance for medium datasets
        elif num_documents < 100000:
            return "ivf_pq"  # Memory efficient for large datasets
        else:
            return "ivf_sq"  # Most memory efficient for very large datasets
    
    def _estimate_memory_usage(self, num_documents: int) -> float:
        """Estimate memory usage in MB"""
        # Rough estimation: 384 dimensions * 4 bytes per float * num_documents
        vector_memory = num_documents * 384 * 4 / (1024 * 1024)  # MB
        
        # Index overhead (varies by type)
        index_overhead = {
            "flat": 1.0,
            "ivf": 1.2,
            "ivf_pq": 0.3,
            "ivf_sq": 0.5
        }
        
        # Use average overhead for estimation
        avg_overhead = sum(index_overhead.values()) / len(index_overhead)
        
        return vector_memory * avg_overhead
    
    async def batch_process_documents(self, documents: List[str], batch_size: int = 100) -> bool:
        """Process documents in batches for large datasets"""
        if not documents:
            return False
        
        try:
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                batch_metadata = [{"content": doc, "batch": i // batch_size} for doc in batch]
                
                success = await self.add_documents(batch, batch_metadata)
                if not success:
                    print(f"Failed to process batch {i // batch_size}")
                    return False
                
                print(f"Processed batch {i // batch_size + 1}/{(len(documents) + batch_size - 1) // batch_size}")
            
            return True
            
        except Exception as e:
            print(f"Error in batch processing: {e}")
            return False
    
    async def search_similar_documents(self, query: str, k: int = 5, threshold: float = 0.5) -> List[Dict[str, Any]]:
        """Search for similar documents with similarity threshold"""
        query_embedding = self._generate_embeddings([query])[0]
        
        # Get more results than needed to filter by threshold
        results = await self.vector_store.search(query_embedding, k=k*2)
        
        # Filter by similarity threshold
        filtered_results = [
            result for result in results 
            if result.get('score', 0) >= threshold
        ][:k]
        
        return filtered_results
    
    async def get_document_by_id(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get document by its ID"""
        if not self.vector_store:
            return None
        
        try:
            # This would need to be implemented in the FAISS backend
            # For now, return None
            return None
        except Exception as e:
            print(f"Error retrieving document {doc_id}: {e}")
            return None


# Global instance
advanced_rag_system = AdvancedRAGSystem()
