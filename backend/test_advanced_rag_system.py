#!/usr/bin/env python3
"""
Test script for Advanced RAG System.
"""

import asyncio
import time
from typing import List, Dict, Any

# Add backend to path
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.services.advanced_rag_system import advanced_rag_system


async def test_advanced_rag_system():
    """Test the Advanced RAG System."""
    print("🚀 Testing Advanced RAG System")
    print("=" * 50)
    
    try:
        # Test documents
        test_documents = [
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
        
        print(f"📝 Created {len(test_documents)} test documents")
        
        # Test 1: Initialize vector store
        print("\n🔍 Test 1: Initialize Vector Store")
        print("-" * 30)
        
        vector_store = await advanced_rag_system.initialize_vector_store(index_type="ivf")
        print(f"✅ Vector store initialized: {vector_store is not None}")
        
        # Test 2: Add documents
        print("\n🔍 Test 2: Add Documents")
        print("-" * 30)
        
        success = await advanced_rag_system.add_documents(test_documents)
        print(f"✅ Documents added: {success}")
        
        # Test 3: Get stats
        print("\n🔍 Test 3: Get Vector Store Stats")
        print("-" * 30)
        
        stats = await advanced_rag_system.get_vector_store_stats()
        print(f"📊 Stats: {stats}")
        
        # Test 4: Query RAG system
        print("\n🔍 Test 4: Query RAG System")
        print("-" * 30)
        
        test_questions = [
            "What is machine learning?",
            "How do vector databases work?",
            "What is FAISS?"
        ]
        
        for question in test_questions:
            print(f"\n🔍 Question: {question}")
            answer, results = await advanced_rag_system.query(question, k=3)
            print(f"📝 Answer: {answer[:200]}...")
            print(f"📊 Results: {len(results)} documents found")
        
        # Test 5: Similarity search with threshold
        print("\n🔍 Test 5: Similarity Search with Threshold")
        print("-" * 30)
        
        query = "What is machine learning?"
        similar_docs = await advanced_rag_system.search_similar_documents(
            query=query,
            k=3,
            threshold=0.3
        )
        print(f"🔍 Query: {query}")
        print(f"📊 Similar documents found: {len(similar_docs)}")
        
        # Test 6: Optimize index parameters
        print("\n🔍 Test 6: Optimize Index Parameters")
        print("-" * 30)
        
        recommendations = await advanced_rag_system.optimize_index(test_documents)
        print(f"🎯 Recommendations: {recommendations}")
        
        # Test 7: Batch processing
        print("\n🔍 Test 7: Batch Processing")
        print("-" * 30)
        
        # Create larger dataset for batch testing
        large_documents = []
        base_text = "This is a sample document for batch processing testing. "
        
        for i in range(50):
            large_documents.append(f"{base_text}Variation {i}. " + base_text * 5)
        
        print(f"📝 Created {len(large_documents)} large documents")
        
        success = await advanced_rag_system.batch_process_documents(
            documents=large_documents,
            batch_size=20
        )
        print(f"✅ Batch processing: {success}")
        
        # Test 8: Save and load vector store
        print("\n🔍 Test 8: Save and Load Vector Store")
        print("-" * 30)
        
        save_path = "data/test_advanced_rag_system"
        save_success = await advanced_rag_system.save_vector_store(save_path)
        print(f"💾 Save vector store: {save_success}")
        
        if save_success:
            print(f"📁 Vector store saved to: {save_path}")
        
        print("\n✅ Advanced RAG System test completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Advanced RAG System test failed: {e}")
        return False


async def test_performance():
    """Test performance of the RAG system."""
    print("\n🚀 Testing Performance")
    print("=" * 50)
    
    try:
        # Create test documents
        test_docs = []
        base_text = "This is a performance test document. "
        
        for i in range(100):
            test_docs.append(f"{base_text}Document {i}. " + base_text * 10)
        
        print(f"📝 Created {len(test_docs)} test documents")
        
        # Initialize with different index types
        index_types = ["flat", "ivf"]
        
        for index_type in index_types:
            print(f"\n🔍 Testing {index_type.upper()} index performance...")
            
            # Initialize
            start_time = time.time()
            await advanced_rag_system.initialize_vector_store(index_type=index_type)
            init_time = time.time() - start_time
            
            # Add documents
            start_time = time.time()
            success = await advanced_rag_system.add_documents(test_docs)
            add_time = time.time() - start_time
            
            if success:
                # Test queries
                test_queries = [
                    "What is machine learning?",
                    "How do vector databases work?",
                    "What is FAISS?",
                    "Explain retrieval augmented generation",
                    "What are embedding models?"
                ]
                
                query_times = []
                for query in test_queries:
                    start_time = time.time()
                    answer, results = await advanced_rag_system.query(query, k=5)
                    query_time = time.time() - start_time
                    query_times.append(query_time)
                
                avg_query_time = sum(query_times) / len(query_times)
                
                print(f"  ✅ {index_type.upper()} performance test completed")
                print(f"  ⏱️ Initialization time: {init_time:.3f}s")
                print(f"  ⏱️ Document addition time: {add_time:.3f}s")
                print(f"  🔍 Average query time: {avg_query_time:.3f}s")
                print(f"  📊 Total documents: {len(test_docs)}")
                
            else:
                print(f"  ❌ Failed to add documents for {index_type}")
        
        print("\n✅ Performance test completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        return False


async def main():
    """Main test function."""
    print("🚀 Advanced RAG System Tests")
    print("=" * 60)
    
    # Test basic functionality
    basic_success = await test_advanced_rag_system()
    
    # Test performance
    performance_success = await test_performance()
    
    # Final summary
    print("\n🎉 All Tests Completed!")
    print("=" * 60)
    
    print(f"✅ Basic Functionality: {basic_success}")
    print(f"✅ Performance: {performance_success}")
    
    total_tests = 2
    passed_tests = sum([basic_success, performance_success])
    
    print(f"\n📊 Test Results: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed successfully!")
    else:
        print("⚠️ Some tests failed. Check the output above for details.")
    
    return {
        "basic_success": basic_success,
        "performance_success": performance_success,
        "total_tests": total_tests,
        "passed_tests": passed_tests
    }


if __name__ == "__main__":
    asyncio.run(main())
