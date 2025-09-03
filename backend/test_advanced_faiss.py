#!/usr/bin/env python3
"""
Test script for advanced FAISS indexes and embedding models.
"""

import asyncio
import numpy as np
import time
from typing import List, Dict, Any

# Add backend to path
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.services.index_backends.faiss_advanced_index import FAISSAdvancedIndexBackend
from app.services.embedding_service import embedding_service


async def test_advanced_faiss_indexes():
    """Test advanced FAISS index types."""
    print("🚀 Testing Advanced FAISS Indexes")
    print("=" * 50)
    
    # Test data
    vector_size = 384
    num_vectors = 1000
    test_vectors = np.random.random((num_vectors, vector_size)).astype(np.float32)
    test_ids = [f"test_{i}" for i in range(num_vectors)]
    test_metadata = [{"type": "test", "index": i} for i in range(num_vectors)]
    
    # Test different index types
    index_types = ["ivf", "hnsw", "pq", "sq", "ivfpq", "ivfsq"]
    
    results = {}
    
    for index_type in index_types:
        print(f"\n🔍 Testing {index_type.upper()} index...")
        
        try:
            # Create index
            index = FAISSAdvancedIndexBackend(
                vector_size=vector_size,
                index_type=index_type,
                metric="ip",
                nlist=100 if "ivf" in index_type else None,
                nprobe=10 if "ivf" in index_type else None,
                m=8 if "pq" in index_type else None,
                bits=8 if "pq" in index_type else None,
                ef_construction=200 if "hnsw" in index_type else None,
                ef_search=128 if "hnsw" in index_type else None,
                index_path=f"data/test_faiss_{index_type}.index"
            )
            
            # Initialize
            start_time = time.time()
            success = await index.initialize()
            init_time = time.time() - start_time
            
            if not success:
                print(f"  ❌ Failed to initialize {index_type} index")
                continue
            
            print(f"  ✅ Initialized in {init_time:.3f}s")
            
            # Add vectors
            start_time = time.time()
            add_success = await index.add_items(test_vectors, test_ids, test_metadata)
            add_time = time.time() - start_time
            
            if not add_success:
                print(f"  ❌ Failed to add vectors to {index_type} index")
                continue
            
            print(f"  ✅ Added {num_vectors} vectors in {add_time:.3f}s")
            
            # Test search
            query_vector = np.random.random((1, vector_size)).astype(np.float32)
            
            start_time = time.time()
            search_results = await index.search(query_vector, k=10)
            search_time = time.time() - start_time
            
            print(f"  ✅ Search completed in {search_time:.3f}s, found {len(search_results)} results")
            
            # Get stats
            stats = await index.get_stats()
            
            results[index_type] = {
                "init_time": init_time,
                "add_time": add_time,
                "search_time": search_time,
                "search_results": len(search_results),
                "stats": stats
            }
            
            # Save index
            await index.save_index()
            print(f"  ✅ Index saved successfully")
            
        except Exception as e:
            print(f"  ❌ Error testing {index_type}: {e}")
            results[index_type] = {"error": str(e)}
    
    # Print summary
    print("\n📊 Test Results Summary")
    print("=" * 50)
    
    for index_type, result in results.items():
        if "error" in result:
            print(f"{index_type.upper()}: ❌ {result['error']}")
        else:
            print(f"{index_type.upper()}: ✅ Init: {result['init_time']:.3f}s, "
                  f"Add: {result['add_time']:.3f}s, Search: {result['search_time']:.3f}s, "
                  f"Results: {result['search_results']}")
    
    return results


async def test_embedding_models():
    """Test different embedding models."""
    print("\n🚀 Testing Embedding Models")
    print("=" * 50)
    
    # Test texts
    test_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "Vector databases enable efficient similarity search.",
        "Natural language processing helps computers understand text.",
        "Deep learning models can learn complex patterns.",
        "Semantic search improves search relevance.",
        "Embedding models convert text to numerical vectors.",
        "FAISS provides fast similarity search algorithms.",
        "Quantization reduces memory usage in vector indexes.",
        "Hyperparameter optimization improves model performance."
    ]
    
    # Initialize embedding service
    await embedding_service.initialize()
    
    # Test different models
    models = [
        "paraphrase-multilingual-MiniLM-L12-v2",
        "all-MiniLM-L6-v2",
        "paraphrase-MiniLM-L6-v2"
    ]
    
    results = {}
    
    for model_name in models:
        print(f"\n🔍 Testing {model_name}...")
        
        try:
            # Load model
            start_time = time.time()
            success = await embedding_service.load_model(model_name)
            load_time = time.time() - start_time
            
            if not success:
                print(f"  ❌ Failed to load {model_name}")
                continue
            
            print(f"  ✅ Loaded in {load_time:.3f}s")
            
            # Encode texts
            start_time = time.time()
            embeddings = await embedding_service.encode_batch(test_texts)
            encode_time = time.time() - start_time
            
            print(f"  ✅ Encoded {len(test_texts)} texts in {encode_time:.3f}s")
            
            # Evaluate performance
            start_time = time.time()
            evaluation = await embedding_service.evaluate_model_performance(
                model_name=model_name,
                test_texts=test_texts,
                evaluation_metrics=["cosine_similarity", "clustering_quality", "semantic_coherence"]
            )
            eval_time = time.time() - start_time
            
            print(f"  ✅ Evaluated in {eval_time:.3f}s")
            
            # Get model info
            model_info = embedding_service.get_current_model_info()
            
            results[model_name] = {
                "load_time": load_time,
                "encode_time": encode_time,
                "eval_time": eval_time,
                "vector_size": model_info.get("vector_size", 0),
                "evaluation": evaluation,
                "model_info": model_info
            }
            
        except Exception as e:
            print(f"  ❌ Error testing {model_name}: {e}")
            results[model_name] = {"error": str(e)}
    
    # Print summary
    print("\n📊 Embedding Model Results")
    print("=" * 50)
    
    for model_name, result in results.items():
        if "error" in result:
            print(f"{model_name}: ❌ {result['error']}")
        else:
            print(f"{model_name}: ✅ Load: {result['load_time']:.3f}s, "
                  f"Encode: {result['encode_time']:.3f}s, "
                  f"Eval: {result['eval_time']:.3f}s, "
                  f"Size: {result['vector_size']}d")
    
    return results


async def test_hyperparameter_optimization():
    """Test hyperparameter optimization."""
    print("\n🚀 Testing Hyperparameter Optimization")
    print("=" * 50)
    
    try:
        # Create test data
        vector_size = 384
        num_training = 500
        num_validation = 100
        
        training_vectors = np.random.random((num_training, vector_size)).astype(np.float32)
        validation_queries = np.random.random((num_validation, vector_size)).astype(np.float32)
        
        # Create ground truth (simulate similar vectors)
        validation_ground_truth = []
        for i in range(num_validation):
            # Find 5 most similar training vectors
            similarities = np.dot(validation_queries[i], training_vectors.T)
            top_indices = np.argsort(similarities)[-5:]
            validation_ground_truth.append([f"train_{idx}" for idx in top_indices])
        
        # Create IVF index for optimization
        index = FAISSAdvancedIndexBackend(
            vector_size=vector_size,
            index_type="ivf",
            metric="ip",
            nlist=50,
            nprobe=5
        )
        
        await index.initialize()
        
        # Add training vectors
        training_ids = [f"train_{i}" for i in range(num_training)]
        await index.add_items(training_vectors, training_ids)
        
        print("✅ Created test index with training data")
        
        # Test optimization
        print("🔍 Running hyperparameter optimization...")
        start_time = time.time()
        
        optimization_result = await index.optimize_hyperparameters(
            training_vectors=training_vectors,
            validation_queries=validation_queries,
            validation_ground_truth=validation_ground_truth,
            param_grid={
                "nlist": [50, 100, 200],
                "nprobe": [1, 5, 10]
            }
        )
        
        opt_time = time.time() - start_time
        
        print(f"✅ Optimization completed in {opt_time:.3f}s")
        print(f"🎯 Best parameters: {optimization_result.get('best_params', {})}")
        print(f"🏆 Best score: {optimization_result.get('best_score', 0.0):.4f}")
        
        return optimization_result
        
    except Exception as e:
        print(f"❌ Error testing hyperparameter optimization: {e}")
        return {"error": str(e)}


async def test_benchmark():
    """Test model benchmarking."""
    print("\n🚀 Testing Model Benchmarking")
    print("=" * 50)
    
    try:
        # Test texts
        test_texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is a subset of artificial intelligence.",
            "Vector databases enable efficient similarity search.",
            "Natural language processing helps computers understand text.",
            "Deep learning models can learn complex patterns."
        ]
        
        # Run benchmark
        print("🔍 Running benchmark...")
        start_time = time.time()
        
        benchmark_results = await embedding_service.benchmark_models(
            test_texts=test_texts,
            models=["paraphrase-multilingual-MiniLM-L12-v2", "all-MiniLM-L6-v2"]
        )
        
        benchmark_time = time.time() - start_time
        
        print(f"✅ Benchmark completed in {benchmark_time:.3f}s")
        
        # Print results
        for model_name, results in benchmark_results.items():
            if "error" not in results:
                encoding_speed = results.get("encoding_speed", {})
                search_speed = results.get("search_speed", {})
                
                print(f"\n📊 {model_name}:")
                print(f"  Encoding: {encoding_speed.get('texts_per_second', 0):.2f} texts/sec")
                print(f"  Search: {search_speed.get('operations_per_second', 0):.2f} ops/sec")
                print(f"  Memory: {results.get('memory_usage_mb', 0):.2f} MB")
        
        return benchmark_results
        
    except Exception as e:
        print(f"❌ Error testing benchmark: {e}")
        return {"error": str(e)}


async def main():
    """Main test function."""
    print("🚀 Advanced FAISS and Embedding Model Tests")
    print("=" * 60)
    
    # Test advanced FAISS indexes
    faiss_results = await test_advanced_faiss_indexes()
    
    # Test embedding models
    embedding_results = await test_embedding_models()
    
    # Test hyperparameter optimization
    optimization_results = await test_hyperparameter_optimization()
    
    # Test benchmarking
    benchmark_results = await test_benchmark()
    
    # Final summary
    print("\n🎉 All Tests Completed!")
    print("=" * 60)
    
    print(f"✅ FAISS Indexes: {len([r for r in faiss_results.values() if 'error' not in r])}/{len(faiss_results)}")
    print(f"✅ Embedding Models: {len([r for r in embedding_results.values() if 'error' not in r])}/{len(embedding_results)}")
    print(f"✅ Hyperparameter Optimization: {'error' not in optimization_results}")
    print(f"✅ Benchmarking: {'error' not in benchmark_results}")
    
    return {
        "faiss_results": faiss_results,
        "embedding_results": embedding_results,
        "optimization_results": optimization_results,
        "benchmark_results": benchmark_results
    }


if __name__ == "__main__":
    asyncio.run(main())
