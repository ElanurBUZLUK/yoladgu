"""
Demo: Hybrid Search System (BM25 + E5-Large-v2 + RRF)
Advanced search combining lexical and semantic approaches
"""

import asyncio
import time
import numpy as np
from typing import List, Dict, Any
import structlog

# Import our hybrid search components
from ml.nlp.hybrid import hybrid_search_engine
from ml.nlp.embedder import e5_embedder

logger = structlog.get_logger()

# Sample documents for testing
SAMPLE_DOCUMENTS = [
    {
        "id": "doc_1",
        "text": "English grammar is the structure of expressions in the English language. This includes the rules of syntax, morphology, and semantics.",
        "metadata": {
            "type": "grammar",
            "level": "A2",
            "topics": ["grammar", "structure", "syntax"],
            "timestamp": time.time()
        }
    },
    {
        "id": "doc_2",
        "text": "Vocabulary learning is essential for language proficiency. Students should practice with flashcards, context clues, and repetition.",
        "metadata": {
            "type": "vocabulary",
            "level": "B1", 
            "topics": ["vocabulary", "learning", "practice"],
            "timestamp": time.time()
        }
    },
    {
        "id": "doc_3",
        "text": "Reading comprehension improves with practice and exposure to diverse texts. Students should read regularly and analyze passages.",
        "metadata": {
            "type": "reading",
            "level": "B2",
            "topics": ["reading", "comprehension", "analysis"],
            "timestamp": time.time()
        }
    },
    {
        "id": "doc_4",
        "text": "Writing skills develop through practice, feedback, and revision. Focus on clarity, coherence, and proper grammar usage.",
        "metadata": {
            "type": "writing",
            "level": "B1",
            "topics": ["writing", "skills", "revision"],
            "timestamp": time.time()
        }
    },
    {
        "id": "doc_5",
        "text": "Speaking practice helps improve pronunciation, fluency, and confidence. Use conversation partners and record yourself.",
        "metadata": {
            "type": "speaking",
            "level": "A2",
            "topics": ["speaking", "pronunciation", "fluency"],
            "timestamp": time.time()
        }
    },
    {
        "id": "doc_6",
        "text": "Listening comprehension requires active engagement with audio materials. Practice with podcasts, videos, and conversations.",
        "metadata": {
            "type": "listening",
            "level": "B1",
            "topics": ["listening", "comprehension", "audio"],
            "timestamp": time.time()
        }
    }
]

async def test_e5_embedder():
    """Test E5 embedding service"""
    print("\n🔤 E5 Embedding Service Test")
    print("=" * 50)
    
    try:
        # Initialize E5 embedder
        success = await e5_embedder.initialize()
        if not success:
            print("❌ Failed to initialize E5 embedder")
            return False
        
        print("✅ E5 embedder initialized successfully")
        print(f"   Model: {e5_embedder.model_name}")
        print(f"   Device: {e5_embedder.device}")
        print(f"   Embedding dimension: {e5_embedder.embedding_dim}")
        
        # Test query encoding
        test_queries = ["English grammar rules", "vocabulary learning methods"]
        query_embeddings = e5_embedder.encode_queries(test_queries)
        print(f"   Query embeddings shape: {query_embeddings.shape}")
        
        # Test passage encoding
        test_passages = [doc["text"] for doc in SAMPLE_DOCUMENTS[:2]]
        passage_embeddings = e5_embedder.encode_passages(test_passages)
        print(f"   Passage embeddings shape: {passage_embeddings.shape}")
        
        # Test similarity
        similarity = np.dot(query_embeddings[0], passage_embeddings[0])
        print(f"   Query-Passage similarity: {similarity:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ E5 embedder test failed: {e}")
        return False

async def test_hybrid_search_engine():
    """Test hybrid search engine"""
    print("\n🔍 Hybrid Search Engine Test")
    print("=" * 50)
    
    try:
        # Initialize hybrid search engine
        success = await hybrid_search_engine.initialize()
        if not success:
            print("❌ Failed to initialize hybrid search engine")
            return False
        
        print("✅ Hybrid search engine initialized successfully")
        
        # Add sample documents
        await hybrid_search_engine.add_documents(SAMPLE_DOCUMENTS)
        print(f"   Added {len(SAMPLE_DOCUMENTS)} documents")
        
        # Get stats
        stats = hybrid_search_engine.get_stats()
        print(f"   Index stats: {stats['index_stats']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Hybrid search engine test failed: {e}")
        return False

async def test_search_types():
    """Test different search types"""
    print("\n🎯 Search Types Comparison")
    print("=" * 50)
    
    test_queries = [
        "English grammar learning",
        "vocabulary practice methods", 
        "reading comprehension skills"
    ]
    
    search_types = ["bm25", "dense", "hybrid", "rrf"]
    
    for query in test_queries:
        print(f"\n📝 Query: '{query}'")
        print("-" * 30)
        
        for search_type in search_types:
            try:
                start_time = time.time()
                
                # Create user context for personalization
                user_context = {
                    "user_id": "demo_user",
                    "user_level": "B1",
                    "user_topics": ["grammar", "vocabulary"]
                }
                
                results = await hybrid_search_engine.search(
                    query=query,
                    k=3,
                    user_context=user_context,
                    search_type=search_type
                )
                
                processing_time = (time.time() - start_time) * 1000
                
                print(f"   {search_type.upper():>6}: {len(results)} results ({processing_time:.1f}ms)")
                
                for i, result in enumerate(results, 1):
                    print(f"      {i}. {result.doc_id} (score: {result.score:.4f})")
                    print(f"         {result.text[:60]}...")
                    
            except Exception as e:
                print(f"   {search_type.upper():>6}: Error - {e}")

async def test_rrf_fusion():
    """Test Reciprocal Rank Fusion specifically"""
    print("\n🔄 Reciprocal Rank Fusion Test")
    print("=" * 50)
    
    try:
        query = "English learning methods"
        
        # Test RRF with different k values
        rrf_k_values = [30, 60, 100]
        
        for k in rrf_k_values:
            print(f"\n   RRF with k={k}")
            
            # Update RRF parameter
            hybrid_search_engine.rrf.k = k
            
            results = await hybrid_search_engine.search(
                query=query,
                k=5,
                search_type="rrf"
            )
            
            print(f"      Results: {len(results)}")
            for i, result in enumerate(results, 1):
                print(f"         {i}. {result.doc_id} (score: {result.score:.4f})")
        
    except Exception as e:
        print(f"❌ RRF test failed: {e}")

async def test_personalization():
    """Test personalized reranking"""
    print("\n👤 Personalization Test")
    print("=" * 50)
    
    try:
        query = "language learning practice"
        
        # Test with different user contexts
        user_contexts = [
            {
                "user_id": "beginner",
                "user_level": "A2",
                "user_topics": ["grammar", "vocabulary"]
            },
            {
                "user_id": "intermediate", 
                "user_level": "B1",
                "user_topics": ["reading", "writing"]
            },
            {
                "user_id": "advanced",
                "user_level": "B2", 
                "user_topics": ["comprehension", "analysis"]
            }
        ]
        
        for context in user_contexts:
            print(f"\n   User: {context['user_id']} (Level: {context['user_level']})")
            
            results = await hybrid_search_engine.search(
                query=query,
                k=3,
                user_context=context,
                search_type="rrf"
            )
            
            for i, result in enumerate(results, 1):
                print(f"      {i}. {result.doc_id} (score: {result.score:.4f})")
                print(f"         Level: {result.metadata.get('level', 'Unknown')}")
                print(f"         Topics: {result.metadata.get('topics', [])}")
        
    except Exception as e:
        print(f"❌ Personalization test failed: {e}")

async def performance_benchmark():
    """Performance benchmark"""
    print("\n⚡ Performance Benchmark")
    print("=" * 50)
    
    try:
        query = "English language learning"
        iterations = 10
        
        search_types = ["bm25", "dense", "hybrid", "rrf"]
        
        for search_type in search_types:
            times = []
            
            for _ in range(iterations):
                start_time = time.time()
                
                await hybrid_search_engine.search(
                    query=query,
                    k=10,
                    search_type=search_type
                )
                
                processing_time = (time.time() - start_time) * 1000
                times.append(processing_time)
            
            avg_time = np.mean(times)
            std_time = np.std(times)
            
            print(f"   {search_type.upper():>6}: {avg_time:.1f}±{std_time:.1f}ms (avg±std)")
        
    except Exception as e:
        print(f"❌ Performance benchmark failed: {e}")

async def main():
    """Main demo function"""
    print("🚀 Hybrid Search System Demo")
    print("=" * 60)
    print("BM25 (Lexical) + E5-Large-v2 (Semantic) + RRF (Fusion)")
    print("=" * 60)
    
    try:
        # Test E5 embedder
        embedder_success = await test_e5_embedder()
        if not embedder_success:
            print("❌ E5 embedder test failed, stopping demo")
            return
        
        # Test hybrid search engine
        engine_success = await test_hybrid_search_engine()
        if not engine_success:
            print("❌ Hybrid search engine test failed, stopping demo")
            return
        
        # Test different search types
        await test_search_types()
        
        # Test RRF fusion
        await test_rrf_fusion()
        
        # Test personalization
        await test_personalization()
        
        # Performance benchmark
        await performance_benchmark()
        
        print("\n✅ Demo completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
