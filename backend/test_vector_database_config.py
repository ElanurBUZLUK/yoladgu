#!/usr/bin/env python3
"""
Vector Database Configuration Test Script

Bu script vector database yapılandırmasını, embedding model'ini ve
indexleme mekanizmasını test eder.
"""

import asyncio
import numpy as np
import time
from typing import List, Dict, Any
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

async def test_vector_database_config():
    """Test vector database configuration and functionality."""
    
    print("🚀 Vector Database Configuration Test")
    print("=" * 50)
    
    # Test 1: Import and Dependencies
    print("\n1️⃣ Testing Dependencies...")
    try:
        from sentence_transformers import SentenceTransformer
        print("  ✅ sentence-transformers imported successfully")
        
        from app.services.vector_service import vector_service
        print("  ✅ vector_service imported successfully")
        
        from app.services.vector_index_manager import vector_index_manager
        print("  ✅ vector_index_manager imported successfully")
        
    except ImportError as e:
        print(f"  ❌ Import error: {e}")
        return False
    
    # Test 2: Embedding Model Configuration
    print("\n2️⃣ Testing Embedding Model...")
    try:
        # Test encoder initialization
        encoder = vector_service._get_encoder()
        print(f"  ✅ Encoder initialized: {type(encoder).__name__}")
        
        # Test model info
        model_name = encoder.model_name if hasattr(encoder, 'model_name') else "Unknown"
        print(f"  ✅ Model name: {model_name}")
        
        # Test vector dimensions
        test_text = "This is a test sentence for embedding."
        test_vector = vector_service.encode_text(test_text)
        print(f"  ✅ Vector dimensions: {len(test_vector)}")
        print(f"  ✅ Vector type: {type(test_vector[0])}")
        
        # Test batch encoding
        test_texts = ["First sentence.", "Second sentence.", "Third sentence."]
        test_vectors = vector_service.encode_batch(test_texts)
        print(f"  ✅ Batch encoding: {len(test_vectors)} vectors")
        
    except Exception as e:
        print(f"  ❌ Embedding test failed: {e}")
        return False
    
    # Test 3: Vector Index Manager
    print("\n3️⃣ Testing Vector Index Manager...")
    try:
        # Initialize manager
        await vector_index_manager.initialize()
        print("  ✅ Vector index manager initialized")
        
        # Check available backends
        backends = list(vector_index_manager.backends.keys())
        print(f"  ✅ Available backends: {backends}")
        
        # Check backend stats
        stats = await vector_index_manager.get_manager_stats()
        print(f"  ✅ Manager stats: {stats}")
        
    except Exception as e:
        print(f"  ❌ Index manager test failed: {e}")
        return False
    
    # Test 4: Backend Health Check
    print("\n4️⃣ Testing Backend Health...")
    try:
        health_status = await vector_index_manager.health_check()
        print("  ✅ Health check results:")
        for backend, status in health_status.items():
            print(f"    - {backend}: {'🟢 Healthy' if status else '🔴 Unhealthy'}")
        
    except Exception as e:
        print(f"  ❌ Health check failed: {e}")
        return False
    
    # Test 5: Vector Operations
    print("\n5️⃣ Testing Vector Operations...")
    try:
        # Test data
        test_items = [
            {
                "id": "test_001",
                "text": "Mathematics is the study of numbers and shapes.",
                "metadata": {
                    "type": "math",
                    "skills": ["algebra", "geometry"],
                    "topic": "mathematics",
                    "difficulty": "medium"
                }
            },
            {
                "id": "test_002", 
                "text": "English grammar includes prepositions and articles.",
                "metadata": {
                    "type": "english",
                    "error_tags": ["prepositions", "articles"],
                    "level_cefr": "B1",
                    "topic": "grammar"
                }
            }
        ]
        
        # Add items
        for item in test_items:
            success = await vector_service.add_item(
                item_id=item["id"],
                text=item["text"],
                metadata=item["metadata"]
            )
            if success:
                print(f"  ✅ Added item: {item['id']}")
            else:
                print(f"  ❌ Failed to add item: {item['id']}")
        
        # Wait for indexing
        await asyncio.sleep(2)
        
        # Test search
        search_results = await vector_service.search(
            query="mathematics numbers",
            limit=5,
            use_hybrid=True
        )
        print(f"  ✅ Search results: {len(search_results)} items found")
        
        # Test search with filters
        filtered_results = await vector_service.search(
            query="grammar prepositions",
            limit=5,
            filters={"type": "english", "level_cefr": "B1"}
        )
        print(f"  ✅ Filtered search: {len(filtered_results)} items found")
        
    except Exception as e:
        print(f"  ❌ Vector operations test failed: {e}")
        return False
    
    # Test 6: Performance Metrics
    print("\n6️⃣ Testing Performance Metrics...")
    try:
        # Test search performance
        start_time = time.time()
        for i in range(5):
            await vector_service.search(
                query=f"test query {i}",
                limit=3
            )
        end_time = time.time()
        
        avg_search_time = (end_time - start_time) / 5 * 1000  # Convert to ms
        print(f"  ✅ Average search time: {avg_search_time:.2f} ms")
        
        # Get performance stats
        stats = await vector_index_manager.get_manager_stats()
        if "avg_search_time_ms" in stats:
            print(f"  ✅ Manager avg search time: {stats['avg_search_time_ms']:.2f} ms")
        
    except Exception as e:
        print(f"  ❌ Performance test failed: {e}")
        return False
    
    # Test 7: Enhanced Embedding
    print("\n7️⃣ Testing Enhanced Embedding...")
    try:
        # Test text enhancement
        original_text = "Students learn mathematics."
        metadata = {
            "type": "math",
            "skills": ["algebra", "calculus"],
            "topic": "education",
            "difficulty": "advanced"
        }
        
        enhanced_text = vector_service._enhance_embedding_text(original_text, metadata)
        print(f"  ✅ Original text: {original_text}")
        print(f"  ✅ Enhanced text: {enhanced_text}")
        
        # Test vector similarity
        original_vector = vector_service.encode_text(original_text)
        enhanced_vector = vector_service.encode_text(enhanced_text)
        
        # Calculate cosine similarity
        original_norm = np.linalg.norm(original_vector)
        enhanced_norm = np.linalg.norm(enhanced_vector)
        
        if original_norm > 0 and enhanced_norm > 0:
            similarity = np.dot(original_vector, enhanced_vector) / (original_norm * enhanced_norm)
            print(f"  ✅ Vector similarity: {similarity:.4f}")
        
    except Exception as e:
        print(f"  ❌ Enhanced embedding test failed: {e}")
        return False
    
    print("\n🎉 All tests completed successfully!")
    return True

async def test_individual_backends():
    """Test individual backend configurations."""
    
    print("\n🔧 Individual Backend Tests")
    print("=" * 40)
    
    try:
        from app.services.index_backends.faiss_flat_index import FAISSFlatIndexBackend
        from app.services.index_backends.hnsw_index import HNSWIndexBackend
        from app.services.index_backends.qdrant_index import QdrantIndexBackend
        
        # Test FAISS
        print("\n📊 Testing FAISS Backend...")
        try:
            faiss_backend = FAISSFlatIndexBackend(vector_size=384, metric="ip")
            success = await faiss_backend.initialize()
            if success:
                print("  ✅ FAISS backend initialized successfully")
                
                # Test add and search
                test_vector = np.random.random(384).astype(np.float32)
                add_success = await faiss_backend.add_items(
                    vectors=np.array([test_vector]),
                    ids=["test_faiss"],
                    metadata=[{"type": "test"}]
                )
                print(f"  ✅ FAISS add items: {add_success}")
                
                # Test search
                search_results = await faiss_backend.search(
                    query_vector=test_vector,
                    k=1
                )
                print(f"  ✅ FAISS search: {len(search_results)} results")
                
            else:
                print("  ❌ FAISS backend initialization failed")
                
        except Exception as e:
            print(f"  ❌ FAISS test failed: {e}")
        
        # Test HNSW
        print("\n🔗 Testing HNSW Backend...")
        try:
            hnsw_backend = HNSWIndexBackend(
                vector_size=384,
                max_elements=1000,
                ef_construction=200,
                m=16
            )
            success = await hnsw_backend.initialize()
            if success:
                print("  ✅ HNSW backend initialized successfully")
                
                # Test add and search
                test_vector = np.random.random(384).astype(np.float32)
                add_success = await hnsw_backend.add_items(
                    vectors=np.array([test_vector]),
                    ids=["test_hnsw"],
                    metadata=[{"type": "test"}]
                )
                print(f"  ✅ HNSW add items: {add_success}")
                
                # Test search
                search_results = await hnsw_backend.search(
                    query_vector=test_vector,
                    k=1
                )
                print(f"  ✅ HNSW search: {len(search_results)} results")
                
            else:
                print("  ❌ HNSW backend initialization failed")
                
        except Exception as e:
            print(f"  ❌ HNSW test failed: {e}")
        
        # Test Qdrant
        print("\n🗄️ Testing Qdrant Backend...")
        try:
            qdrant_backend = QdrantIndexBackend(
                collection_name="test_collection",
                vector_size=384,
                url="http://localhost:6333"
            )
            success = await qdrant_backend.initialize()
            if success:
                print("  ✅ Qdrant backend initialized successfully")
                
                # Test add and search
                test_vector = np.random.random(384).astype(np.float32)
                add_success = await qdrant_backend.add_items(
                    vectors=np.array([test_vector]),
                    ids=["test_qdrant"],
                    metadata=[{"type": "test"}]
                )
                print(f"  ✅ Qdrant add items: {add_success}")
                
                # Test search
                search_results = await qdrant_backend.search(
                    query_vector=test_vector,
                    k=1
                )
                print(f"  ✅ Qdrant search: {len(search_results)} results")
                
            else:
                print("  ❌ Qdrant backend initialization failed")
                
        except Exception as e:
            print(f"  ❌ Qdrant test failed: {e}")
            
    except ImportError as e:
        print(f"  ❌ Backend import failed: {e}")

async def main():
    """Main test function."""
    print("🚀 Starting Vector Database Configuration Tests...")
    
    # Run main tests
    success = await test_vector_database_config()
    
    # Run individual backend tests
    await test_individual_backends()
    
    if success:
        print("\n🎉 All tests passed! Vector database is properly configured.")
    else:
        print("\n❌ Some tests failed. Please check the configuration.")
    
    return success

if __name__ == "__main__":
    # Run async tests
    try:
        result = asyncio.run(main())
        sys.exit(0 if result else 1)
    except KeyboardInterrupt:
        print("\n⏹️ Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1)
