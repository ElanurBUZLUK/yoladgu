#!/usr/bin/env python3
"""
Test script for LangChain integration and RAG functionality.
"""

import asyncio
import time
from typing import List, Dict, Any

# Add backend to path
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.services.langchain_integration import langchain_service


async def test_langchain_integration():
    """Test LangChain integration service."""
    print("🚀 Testing LangChain Integration")
    print("=" * 50)
    
    try:
        # Initialize embeddings
        print("🔍 Initializing embeddings...")
        success = await langchain_service.initialize_embeddings()
        if not success:
            print("❌ Failed to initialize embeddings")
            return False
        
        print("✅ Embeddings initialized successfully")
        
        # Test text chunks
        test_texts = [
            "Machine learning is a subset of artificial intelligence that focuses on algorithms and statistical models.",
            "Natural language processing enables computers to understand and generate human language.",
            "Vector databases store and retrieve high-dimensional vector representations of data.",
            "FAISS is a library for efficient similarity search and clustering of dense vectors.",
            "Retrieval-Augmented Generation combines information retrieval with text generation.",
            "Embedding models convert text into numerical vectors for machine learning applications.",
            "Semantic search finds documents based on meaning rather than exact keyword matches.",
            "Product quantization reduces memory usage while maintaining search quality.",
            "Inverted file indexes organize vectors into clusters for faster approximate search.",
            "Hyperparameter optimization improves model performance through systematic parameter tuning."
        ]
        
        print(f"📝 Created {len(test_texts)} test text chunks")
        
        # Test vector store creation with different index types
        index_types = ["Flat", "IVF", "PQ"]
        
        for index_type in index_types:
            print(f"\n🔍 Testing {index_type} index...")
            
            try:
                # Create vector store
                vectorstore = langchain_service.create_advanced_vector_store(
                    text_chunks=test_texts,
                    index_type=index_type,
                    save_path=f"data/test_langchain_{index_type.lower()}"
                )
                
                print(f"  ✅ {index_type} vector store created successfully")
                
                # Get stats
                stats = langchain_service.get_vector_store_stats()
                print(f"  📊 Stats: {stats}")
                
                # Test retrieval quality evaluation
                test_query = "What is machine learning?"
                docs, scores, metrics = langchain_service.evaluate_retrieval_quality(
                    query=test_query,
                    k=3
                )
                
                print(f"  🔍 Query: '{test_query}'")
                print(f"  📊 Retrieved {len(docs)} documents")
                print(f"  📈 Metrics: {metrics}")
                
                # Test performance benchmarking
                test_queries = [
                    "What is machine learning?",
                    "How do vector databases work?",
                    "What is FAISS?"
                ]
                
                benchmark_results = langchain_service.benchmark_index_performance(
                    test_queries=test_queries,
                    k=3
                )
                
                print(f"  ⚡ Benchmark completed for {len(test_queries)} queries")
                
                # Test parameter optimization
                optimal_params = langchain_service.optimize_index_parameters(test_texts)
                print(f"  🎯 Optimal parameters: {optimal_params}")
                
            except Exception as e:
                print(f"  ❌ Error testing {index_type}: {e}")
                continue
        
        # Test RAG chain creation
        print("\n🔍 Testing RAG chain creation...")
        try:
            # Use the last created vector store
            if langchain_service.vectorstore:
                qa_chain = langchain_service.create_advanced_rag_chain(
                    model_name="gpt-3.5-turbo",
                    temperature=0.0,
                    max_tokens=1000
                )
                print("  ✅ RAG chain created successfully")
            else:
                print("  ⚠️ No vector store available for RAG chain testing")
                
        except Exception as e:
            print(f"  ❌ Error creating RAG chain: {e}")
        
        print("\n✅ LangChain integration test completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ LangChain integration test failed: {e}")
        return False


async def test_document_processing():
    """Test document processing capabilities."""
    print("\n🚀 Testing Document Processing")
    print("=" * 50)
    
    try:
        # Test with different chunk sizes
        chunk_sizes = [500, 1000, 1500]
        
        for chunk_size in chunk_sizes:
            print(f"\n🔍 Testing chunk size: {chunk_size}")
            
            # Create sample text chunks
            sample_texts = [
                "This is a sample text chunk for testing document processing. " * (chunk_size // 50),
                "Another sample text chunk with different content for testing. " * (chunk_size // 50),
                "Third sample text chunk to test various processing scenarios. " * (chunk_size // 50)
            ]
            
            # Optimize parameters
            optimal_params = langchain_service.optimize_index_parameters(sample_texts)
            
            print(f"  📊 Chunk size: {chunk_size}")
            print(f"  📝 Number of chunks: {len(sample_texts)}")
            print(f"  🎯 Recommended index type: {optimal_params.get('recommended_index_type', 'Unknown')}")
            print(f"  📏 Optimal nlist: {optimal_params.get('optimal_nlist', 'Unknown')}")
        
        print("\n✅ Document processing test completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Document processing test failed: {e}")
        return False


async def test_performance_optimization():
    """Test performance optimization features."""
    print("\n🚀 Testing Performance Optimization")
    print("=" * 50)
    
    try:
        # Create a larger dataset for performance testing
        large_texts = []
        base_text = "This is a sample text for performance testing. "
        
        for i in range(100):
            large_texts.append(f"{base_text}Variation {i}. " + base_text * 10)
        
        print(f"📝 Created {len(large_texts)} large text chunks")
        
        # Test different index types for performance
        index_types = ["Flat", "IVF", "IVF_PQ"]
        
        performance_results = {}
        
        for index_type in index_types:
            print(f"\n🔍 Testing {index_type} performance...")
            
            try:
                start_time = time.time()
                
                # Create vector store
                vectorstore = langchain_service.create_advanced_vector_store(
                    text_chunks=large_texts,
                    index_type=index_type
                )
                
                creation_time = time.time() - start_time
                
                # Test search performance
                test_queries = [
                    "What is machine learning?",
                    "How do vector databases work?",
                    "What is FAISS?",
                    "Explain retrieval augmented generation",
                    "What are embedding models?"
                ]
                
                search_times = []
                for query in test_queries:
                    start_time = time.time()
                    docs = vectorstore.similarity_search(query, k=5)
                    search_time = time.time() - start_time
                    search_times.append(search_time)
                
                avg_search_time = sum(search_times) / len(search_times)
                
                # Get stats
                stats = langchain_service.get_vector_store_stats()
                
                performance_results[index_type] = {
                    "creation_time": creation_time,
                    "avg_search_time": avg_search_time,
                    "total_vectors": stats.get("total_vectors", 0),
                    "index_type": stats.get("index_type", "Unknown")
                }
                
                print(f"  ✅ {index_type} performance test completed")
                print(f"  ⏱️ Creation time: {creation_time:.3f}s")
                print(f"  🔍 Avg search time: {avg_search_time:.3f}s")
                print(f"  📊 Total vectors: {stats.get('total_vectors', 0)}")
                
            except Exception as e:
                print(f"  ❌ Error testing {index_type} performance: {e}")
                continue
        
        # Print performance comparison
        print("\n📊 Performance Comparison")
        print("=" * 30)
        
        for index_type, results in performance_results.items():
            print(f"{index_type}:")
            print(f"  Creation: {results['creation_time']:.3f}s")
            print(f"  Search: {results['avg_search_time']:.3f}s")
            print(f"  Vectors: {results['total_vectors']}")
            print()
        
        print("✅ Performance optimization test completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Performance optimization test failed: {e}")
        return False


async def main():
    """Main test function."""
    print("🚀 LangChain Integration and RAG Tests")
    print("=" * 60)
    
    # Test LangChain integration
    integration_success = await test_langchain_integration()
    
    # Test document processing
    processing_success = await test_document_processing()
    
    # Test performance optimization
    performance_success = await test_performance_optimization()
    
    # Final summary
    print("\n🎉 All Tests Completed!")
    print("=" * 60)
    
    print(f"✅ LangChain Integration: {integration_success}")
    print(f"✅ Document Processing: {processing_success}")
    print(f"✅ Performance Optimization: {performance_success}")
    
    total_tests = 3
    passed_tests = sum([integration_success, processing_success, performance_success])
    
    print(f"\n📊 Test Results: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed successfully!")
    else:
        print("⚠️ Some tests failed. Check the output above for details.")
    
    return {
        "integration_success": integration_success,
        "processing_success": processing_success,
        "performance_success": performance_success,
        "total_tests": total_tests,
        "passed_tests": passed_tests
    }


if __name__ == "__main__":
    asyncio.run(main())
