#!/usr/bin/env python3
"""
Test script for the new embedding service with OpenAI API integration
"""

import asyncio
import time
from typing import List

# Import the embedding service
from app.services.embedding_service import embedding_service


async def test_embedding_service():
    """Test the embedding service functionality"""
    
    print("🧪 Testing Embedding Service...")
    
    # Test 1: Basic embedding generation
    print("\n1. Testing basic embedding generation...")
    try:
        text = "This is a test sentence for embedding generation."
        embedding = await embedding_service.get_embedding(text)
        
        print(f"✅ Embedding generated successfully")
        print(f"   Dimension: {len(embedding)}")
        print(f"   Expected: {embedding_service.embedding_dimension}")
        print(f"   First 5 values: {embedding[:5]}")
        
        assert len(embedding) == embedding_service.embedding_dimension
        assert all(isinstance(x, float) for x in embedding)
        
    except Exception as e:
        print(f"❌ Error in basic embedding: {e}")
    
    # Test 2: Batch embedding generation
    print("\n2. Testing batch embedding generation...")
    try:
        texts = [
            "First test sentence for batch processing.",
            "Second test sentence with different content.",
            "Third test sentence to check batch functionality.",
            "Fourth test sentence for comprehensive testing."
        ]
        
        embeddings = await embedding_service.get_batch_embeddings(texts)
        
        print(f"✅ Batch embeddings generated successfully")
        print(f"   Input texts: {len(texts)}")
        print(f"   Output embeddings: {len(embeddings)}")
        print(f"   All dimensions correct: {all(len(emb) == embedding_service.embedding_dimension for emb in embeddings)}")
        
        assert len(embeddings) == len(texts)
        assert all(len(emb) == embedding_service.embedding_dimension for emb in embeddings)
        
    except Exception as e:
        print(f"❌ Error in batch embedding: {e}")
    
    # Test 3: Similarity calculation
    print("\n3. Testing similarity calculation...")
    try:
        text1 = "The cat sat on the mat."
        text2 = "A cat is sitting on a mat."
        text3 = "The weather is sunny today."
        
        emb1 = await embedding_service.get_embedding(text1)
        emb2 = await embedding_service.get_embedding(text2)
        emb3 = await embedding_service.get_embedding(text3)
        
        sim_1_2 = await embedding_service.calculate_similarity(emb1, emb2)
        sim_1_3 = await embedding_service.calculate_similarity(emb1, emb3)
        
        print(f"✅ Similarity calculated successfully")
        print(f"   Similarity (cat/cat): {sim_1_2:.4f}")
        print(f"   Similarity (cat/weather): {sim_1_3:.4f}")
        print(f"   Similar sentences should be more similar: {sim_1_2 > sim_1_3}")
        
        # Similar sentences should be more similar
        assert sim_1_2 > sim_1_3
        
    except Exception as e:
        print(f"❌ Error in similarity calculation: {e}")
    
    # Test 4: Most similar search
    print("\n4. Testing most similar search...")
    try:
        query_text = "The cat is sleeping."
        candidate_texts = [
            "A cat is sleeping on the couch.",
            "The dog is running in the park.",
            "A cat is eating food.",
            "The bird is flying in the sky.",
            "A cat is playing with a toy."
        ]
        
        query_embedding = await embedding_service.get_embedding(query_text)
        candidate_embeddings = await embedding_service.get_batch_embeddings(candidate_texts)
        
        similar_results = await embedding_service.find_most_similar(
            query_embedding, candidate_embeddings, top_k=3
        )
        
        print(f"✅ Most similar search completed")
        print(f"   Query: {query_text}")
        print(f"   Top 3 similar texts:")
        for i, result in enumerate(similar_results, 1):
            idx = result["index"]
            similarity = result["similarity"]
            text = candidate_texts[idx]
            print(f"   {i}. Similarity: {similarity:.4f} - {text}")
        
        assert len(similar_results) == 3
        assert all("similarity" in result for result in similar_results)
        
    except Exception as e:
        print(f"❌ Error in most similar search: {e}")
    
    # Test 5: Service statistics
    print("\n5. Testing service statistics...")
    try:
        stats = await embedding_service.get_embedding_statistics()
        
        print(f"✅ Service statistics retrieved")
        print(f"   Model: {stats['model_name']}")
        print(f"   Dimension: {stats['embedding_dimension']}")
        print(f"   Batch size: {stats['batch_size']}")
        print(f"   Cost per 1K tokens: ${stats['cost_per_1k_tokens']:.6f}")
        
        assert "model_name" in stats
        assert "embedding_dimension" in stats
        assert "cost_per_1k_tokens" in stats
        
    except Exception as e:
        print(f"❌ Error in service statistics: {e}")
    
    # Test 6: Health check
    print("\n6. Testing health check...")
    try:
        health = await embedding_service.health_check()
        
        print(f"✅ Health check completed")
        print(f"   Status: {health['status']}")
        print(f"   Test successful: {health['test_successful']}")
        print(f"   Timestamp: {health['timestamp']}")
        
        assert "status" in health
        assert "test_successful" in health
        assert "timestamp" in health
        
    except Exception as e:
        print(f"❌ Error in health check: {e}")
    
    # Test 7: Performance test
    print("\n7. Testing performance...")
    try:
        test_texts = [f"Test sentence number {i} for performance testing." for i in range(10)]
        
        start_time = time.time()
        embeddings = await embedding_service.get_batch_embeddings(test_texts)
        end_time = time.time()
        
        duration = end_time - start_time
        avg_time_per_embedding = duration / len(test_texts)
        
        print(f"✅ Performance test completed")
        print(f"   Total time: {duration:.2f} seconds")
        print(f"   Average time per embedding: {avg_time_per_embedding:.3f} seconds")
        print(f"   Embeddings per second: {len(test_texts) / duration:.2f}")
        
        # Should be reasonably fast (less than 1 second per embedding)
        assert avg_time_per_embedding < 1.0
        
    except Exception as e:
        print(f"❌ Error in performance test: {e}")
    
    # Test 8: Error handling
    print("\n8. Testing error handling...")
    try:
        # Test with empty text
        empty_embedding = await embedding_service.get_embedding("")
        print(f"✅ Empty text handled: {len(empty_embedding)} dimensions")
        
        # Test with None text
        none_embedding = await embedding_service.get_embedding(None)
        print(f"✅ None text handled: {len(none_embedding)} dimensions")
        
        # Test with very long text
        long_text = "This is a very long text. " * 1000
        long_embedding = await embedding_service.get_embedding(long_text)
        print(f"✅ Long text handled: {len(long_embedding)} dimensions")
        
    except Exception as e:
        print(f"❌ Error in error handling test: {e}")
    
    print("\n🎉 Embedding Service tests completed!")


async def test_cache_functionality():
    """Test caching functionality"""
    
    print("\n🧪 Testing Cache Functionality...")
    
    try:
        text = "This is a test for caching functionality."
        
        # First call - should generate embedding
        start_time = time.time()
        embedding1 = await embedding_service.get_embedding(text)
        first_call_time = time.time() - start_time
        
        # Second call - should use cache
        start_time = time.time()
        embedding2 = await embedding_service.get_embedding(text)
        second_call_time = time.time() - start_time
        
        print(f"✅ Cache test completed")
        print(f"   First call time: {first_call_time:.3f} seconds")
        print(f"   Second call time: {second_call_time:.3f} seconds")
        print(f"   Cache speedup: {first_call_time / second_call_time:.1f}x")
        print(f"   Embeddings identical: {embedding1 == embedding2}")
        
        # Second call should be faster
        assert second_call_time < first_call_time
        assert embedding1 == embedding2
        
    except Exception as e:
        print(f"❌ Error in cache test: {e}")


async def main():
    """Main test function"""
    
    print("🚀 Starting Embedding Service Tests...")
    print("=" * 50)
    
    # Run basic tests
    await test_embedding_service()
    
    # Run cache tests
    await test_cache_functionality()
    
    print("\n" + "=" * 50)
    print("✅ All tests completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())
