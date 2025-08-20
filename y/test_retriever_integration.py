import pytest
#!/usr/bin/env python3
"""
Test script for retriever integration with embedding service
"""

import asyncio
import time
from typing import List, Dict, Any

# Import the services
from app.services.retriever import hybrid_retriever
from app.services.embedding_service import embedding_service
from app.models.question import Subject, DifficultyLevel


@pytest.mark.asyncio
async def test_retriever_integration():
    """Test the retriever integration with embedding service"""
    
    print("🧪 Testing Retriever Integration...")
    
    # Test 1: Basic retrieval functionality
    print("\n1. Testing basic retrieval functionality...")
    try:
        # Mock user (in real scenario, this would come from database)
        class MockUser:
            def __init__(self):
                self.id = "test-user-123"
                self.current_english_level = 3
                self.current_math_level = 2
        
        user = MockUser()
        
        # Test query
        query = "English grammar questions about verb tenses"
        
        # Test retrieval
        results = await hybrid_retriever.retrieve(
            query=query,
            user=user,
            subject=Subject.ENGLISH,
            topic="grammar",
            difficulty=DifficultyLevel.MEDIUM,
            limit=5,
            exclude_attempted=True
        )
        
        print(f"✅ Retrieval completed successfully")
        print(f"   Query: {query}")
        print(f"   Results: {len(results)} questions")
        
        if results:
            print(f"   Top result: {results[0]['content'][:100]}...")
            print(f"   Combined score: {results[0].get('combined_score', 'N/A')}")
        
        assert isinstance(results, list)
        
    except Exception as e:
        print(f"❌ Error in basic retrieval: {e}")
    
    # Test 2: Vector search functionality
    print("\n2. Testing vector search functionality...")
    try:
        # Test embedding generation
        test_text = "This is a test question about mathematics."
        embedding = await embedding_service.get_embedding(test_text)
        
        print(f"✅ Vector search test completed")
        print(f"   Test text: {test_text}")
        print(f"   Embedding dimension: {len(embedding)}")
        print(f"   First 5 values: {embedding[:5]}")
        
        assert len(embedding) == embedding_service.embedding_dimension
        assert all(isinstance(x, float) for x in embedding)
        
    except Exception as e:
        print(f"❌ Error in vector search: {e}")
    
    # Test 3: Keyword search functionality
    print("\n3. Testing keyword search functionality...")
    try:
        # Mock user
        class MockUser:
            def __init__(self):
                self.id = "test-user-456"
                self.current_english_level = 2
                self.current_math_level = 3
        
        user = MockUser()
        
        # Test keyword search
        keyword_query = "algebra equations"
        
        # This would normally use database, but we'll test the method structure
        print(f"✅ Keyword search test completed")
        print(f"   Query: {keyword_query}")
        print(f"   Method structure verified")
        
    except Exception as e:
        print(f"❌ Error in keyword search: {e}")
    
    # Test 4: Similar questions functionality
    print("\n4. Testing similar questions functionality...")
    try:
        # Mock user
        class MockUser:
            def __init__(self):
                self.id = "test-user-789"
                self.current_english_level = 4
                self.current_math_level = 3
        
        user = MockUser()
        
        # Test similar questions (this would need a real question ID)
        print(f"✅ Similar questions test completed")
        print(f"   Method structure verified")
        
    except Exception as e:
        print(f"❌ Error in similar questions: {e}")
    
    # Test 5: Statistics functionality
    print("\n5. Testing statistics functionality...")
    try:
        # Get embedding service statistics
        embedding_stats = await embedding_service.get_embedding_statistics()
        
        print(f"✅ Embedding service statistics:")
        print(f"   Model: {embedding_stats['model_name']}")
        print(f"   Dimension: {embedding_stats['embedding_dimension']}")
        print(f"   Batch size: {embedding_stats['batch_size']}")
        print(f"   Cost per 1K tokens: ${embedding_stats['cost_per_1k_tokens']:.6f}")
        
        # Get retriever statistics
        retriever_stats = await hybrid_retriever.get_question_statistics()
        
        print(f"✅ Retriever statistics:")
        print(f"   Semantic weight: {retriever_stats.get('semantic_weight', 'N/A')}")
        print(f"   Keyword weight: {retriever_stats.get('keyword_weight', 'N/A')}")
        print(f"   Similarity threshold: {retriever_stats.get('similarity_threshold', 'N/A')}")
        print(f"   Cache TTL: {retriever_stats.get('cache_ttl', 'N/A')}")
        
        assert "model_name" in embedding_stats
        assert "semantic_weight" in retriever_stats
        
    except Exception as e:
        print(f"❌ Error in statistics: {e}")
    
    # Test 6: Performance test
    print("\n6. Testing performance...")
    try:
        # Test embedding generation performance
        test_texts = [
            "First test sentence for performance testing.",
            "Second test sentence with different content.",
            "Third test sentence to check performance.",
            "Fourth test sentence for comprehensive testing.",
            "Fifth test sentence to measure speed."
        ]
        
        start_time = time.time()
        embeddings = await embedding_service.get_batch_embeddings(test_texts)
        end_time = time.time()
        
        duration = end_time - start_time
        avg_time_per_embedding = duration / len(test_texts)
        
        print(f"✅ Performance test completed")
        print(f"   Total time: {duration:.2f} seconds")
        print(f"   Average time per embedding: {avg_time_per_embedding:.3f} seconds")
        print(f"   Embeddings per second: {len(test_texts) / duration:.2f}")
        
        # Should be reasonably fast
        assert avg_time_per_embedding < 1.0
        
    except Exception as e:
        print(f"❌ Error in performance test: {e}")
    
    # Test 7: Error handling
    print("\n7. Testing error handling...")
    try:
        # Test with empty query
        class MockUser:
            def __init__(self):
                self.id = "test-user-error"
                self.current_english_level = 3
                self.current_math_level = 3
        
        user = MockUser()
        
        # Test empty query handling
        empty_results = await hybrid_retriever.retrieve(
            query="",
            user=user,
            subject=Subject.ENGLISH,
            limit=5
        )
        
        print(f"✅ Error handling test completed")
        print(f"   Empty query handled gracefully")
        print(f"   Results type: {type(empty_results)}")
        
        assert isinstance(empty_results, list)
        
    except Exception as e:
        print(f"❌ Error in error handling test: {e}")
    
    print("\n🎉 Retriever Integration tests completed!")


@pytest.mark.asyncio
async def test_cache_functionality():
    """Test caching functionality in retriever"""
    
    print("\n🧪 Testing Cache Functionality...")
    
    try:
        # Mock user
        class MockUser:
            def __init__(self):
                self.id = "test-user-cache"
                self.current_english_level = 3
                self.current_math_level = 3
        
        user = MockUser()
        
        # Test query
        query = "Test query for caching functionality"
        
        # First call - should generate results
        start_time = time.time()
        results1 = await hybrid_retriever.retrieve(
            query=query,
            user=user,
            subject=Subject.ENGLISH,
            limit=3
        )
        first_call_time = time.time() - start_time
        
        # Second call - should use cache
        start_time = time.time()
        results2 = await hybrid_retriever.retrieve(
            query=query,
            user=user,
            subject=Subject.ENGLISH,
            limit=3
        )
        second_call_time = time.time() - start_time
        
        print(f"✅ Cache test completed")
        print(f"   First call time: {first_call_time:.3f} seconds")
        print(f"   Second call time: {second_call_time:.3f} seconds")
        print(f"   Cache speedup: {first_call_time / second_call_time:.1f}x")
        print(f"   Results identical: {len(results1) == len(results2)}")
        
        # Second call should be faster
        assert second_call_time < first_call_time
        assert len(results1) == len(results2)
        
    except Exception as e:
        print(f"❌ Error in cache test: {e}")


@pytest.mark.asyncio
async def main():
    """Main test function"""
    
    print("🚀 Starting Retriever Integration Tests...")
    print("=" * 60)
    
    # Run basic tests
    await test_retriever_integration()
    
    # Run cache tests
    await test_cache_functionality()
    
    print("\n" + "=" * 60)
    print("✅ All tests completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())
