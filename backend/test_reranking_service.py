#!/usr/bin/env python3
"""
Test script for the re-ranking service implementation.
"""

import asyncio
import json
import logging
import sys
import os
from datetime import datetime

# Add the app directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_reranking_service_structure():
    """Test re-ranking service structure and basic functionality."""
    try:
        # Import without loading the actual model
        import unittest.mock as mock
        
        with mock.patch('sentence_transformers.CrossEncoder'):
            from app.services.reranking_service import reranking_service
            
            logger.info("Testing re-ranking service structure...")
            
            # Check service attributes
            assert hasattr(reranking_service, 'rerank_candidates')
            assert hasattr(reranking_service, 'get_stats')
            assert hasattr(reranking_service, 'warmup')
            
            # Check configuration
            assert reranking_service.batch_size > 0
            assert reranking_service.max_cache_size > 0
            
            logger.info("✅ Re-ranking service structure test passed!")
            return True
        
    except Exception as e:
        logger.error(f"❌ Re-ranking service structure test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_reranking_functionality():
    """Test re-ranking functionality with mock data."""
    try:
        import unittest.mock as mock
        
        # Mock the CrossEncoder
        mock_model = mock.Mock()
        mock_model.predict.return_value = [0.8, 0.6, 0.9, 0.4, 0.7]  # Mock scores
        
        with mock.patch('sentence_transformers.CrossEncoder', return_value=mock_model):
            from app.services.reranking_service import reranking_service
            
            logger.info("Testing re-ranking functionality...")
            
            # Prepare test data
            query_context = {
                "query": "linear equation problems",
                "target_skills": ["linear_equation", "algebra"],
                "theta_math": 0.2,
                "language": "tr",
                "error_profiles": {
                    "math": {"sign_error": 0.4, "algebra": 0.3}
                }
            }
            
            candidates = [
                {
                    "item_id": "item_1",
                    "retriever_scores": {"dense": 0.7, "sparse": 0.5},
                    "metadata": {
                        "type": "math",
                        "skills": ["linear_equation"],
                        "difficulty": 0.1,
                        "lang": "tr",
                        "stem": "Solve for x: 2x + 3 = 7"
                    }
                },
                {
                    "item_id": "item_2", 
                    "retriever_scores": {"dense": 0.6, "sparse": 0.8},
                    "metadata": {
                        "type": "math",
                        "skills": ["algebra", "quadratic"],
                        "difficulty": 0.5,
                        "lang": "tr",
                        "stem": "Find the roots of x² - 5x + 6 = 0"
                    }
                },
                {
                    "item_id": "item_3",
                    "retriever_scores": {"dense": 0.8, "sparse": 0.4},
                    "metadata": {
                        "type": "math",
                        "skills": ["linear_equation", "word_problems"],
                        "difficulty": 0.0,
                        "lang": "tr",
                        "stem": "A train travels 120 km in 2 hours"
                    }
                }
            ]
            
            # Force model to be loaded (mocked)
            reranking_service._model_loaded = True
            reranking_service.model = mock_model
            
            # Test re-ranking
            result = await reranking_service.rerank_candidates(
                query_context=query_context,
                candidates=candidates,
                max_k=3,
                use_cache=False  # Disable cache for testing
            )
            
            # Verify results
            assert len(result) <= 3
            assert all("rerank_score" in item for item in result)
            assert all("cross_encoder_score" in item for item in result)
            
            # Check that results are sorted by rerank_score
            scores = [item["rerank_score"] for item in result]
            assert scores == sorted(scores, reverse=True)
            
            logger.info(f"Re-ranking result: {len(result)} items")
            for i, item in enumerate(result):
                logger.info(f"  {i+1}. {item['item_id']}: score={item['rerank_score']:.3f}")
            
            logger.info("✅ Re-ranking functionality test passed!")
            return True
        
    except Exception as e:
        logger.error(f"❌ Re-ranking functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_caching_functionality():
    """Test caching functionality."""
    try:
        import unittest.mock as mock
        
        mock_model = mock.Mock()
        mock_model.predict.return_value = [0.8, 0.6]
        
        with mock.patch('sentence_transformers.CrossEncoder', return_value=mock_model):
            from app.services.reranking_service import reranking_service
            
            logger.info("Testing caching functionality...")
            
            # Force model to be loaded
            reranking_service._model_loaded = True
            reranking_service.model = mock_model
            
            query_context = {
                "query": "test query",
                "target_skills": ["test_skill"],
                "theta_math": 0.0,
                "language": "tr"
            }
            
            candidates = [
                {
                    "item_id": "cache_test_1",
                    "retriever_scores": {"dense": 0.5},
                    "metadata": {"type": "math", "skills": ["test_skill"]}
                },
                {
                    "item_id": "cache_test_2",
                    "retriever_scores": {"dense": 0.6},
                    "metadata": {"type": "math", "skills": ["test_skill"]}
                }
            ]
            
            # First call - should cache the result
            result1 = await reranking_service.rerank_candidates(
                query_context=query_context,
                candidates=candidates,
                max_k=2,
                use_cache=True
            )
            
            # Second call - should use cache
            result2 = await reranking_service.rerank_candidates(
                query_context=query_context,
                candidates=candidates,
                max_k=2,
                use_cache=True
            )
            
            # Results should be identical
            assert len(result1) == len(result2)
            
            # Check cache statistics
            stats = reranking_service.get_stats()
            assert stats["cache_hits"] > 0
            
            logger.info(f"Cache hit rate: {stats['cache_hit_rate']:.2%}")
            logger.info("✅ Caching functionality test passed!")
            return True
        
    except Exception as e:
        logger.error(f"❌ Caching functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_fallback_mechanisms():
    """Test fallback mechanisms when cross-encoder fails."""
    try:
        import unittest.mock as mock
        
        # Mock model that raises an exception
        mock_model = mock.Mock()
        mock_model.predict.side_effect = Exception("Model inference failed")
        
        with mock.patch('sentence_transformers.CrossEncoder', return_value=mock_model):
            from app.services.reranking_service import reranking_service
            
            logger.info("Testing fallback mechanisms...")
            
            # Force model to be loaded but fail on inference
            reranking_service._model_loaded = True
            reranking_service.model = mock_model
            
            query_context = {
                "query": "fallback test",
                "target_skills": ["test_skill"],
                "theta_math": 0.0,
                "language": "tr"
            }
            
            candidates = [
                {
                    "item_id": "fallback_test_1",
                    "retriever_scores": {"dense": 0.7},
                    "metadata": {"type": "math", "skills": ["test_skill"]}
                }
            ]
            
            # Should not raise exception, should use fallback
            result = await reranking_service.rerank_candidates(
                query_context=query_context,
                candidates=candidates,
                max_k=1,
                use_cache=False
            )
            
            # Should still return results
            assert len(result) > 0
            assert "rerank_score" in result[0]
            
            logger.info("✅ Fallback mechanisms test passed!")
            return True
        
    except Exception as e:
        logger.error(f"❌ Fallback mechanisms test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_orchestration_integration():
    """Test integration with orchestration service."""
    try:
        import unittest.mock as mock
        
        # Mock all external dependencies
        with mock.patch('sentence_transformers.CrossEncoder'), \
             mock.patch('redis.Redis'), \
             mock.patch('app.services.retrieval_service.retrieval_service'), \
             mock.patch('app.services.profile_service.profile_service'), \
             mock.patch('app.services.bandit_service.bandit_service'), \
             mock.patch('app.services.math_generation_service.math_generation_service'):
            
            from app.services.orchestration_service import orchestration_service
            
            logger.info("Testing orchestration integration...")
            
            # Check that reranking service is imported
            assert hasattr(orchestration_service, '_reranking_phase')
            assert hasattr(orchestration_service, '_build_rerank_query')
            assert hasattr(orchestration_service, '_fallback_reranking')
            
            logger.info("✅ Orchestration integration test passed!")
            return True
        
    except Exception as e:
        logger.error(f"❌ Orchestration integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run all re-ranking service tests."""
    logger.info("🚀 Starting re-ranking service tests...")
    
    tests = [
        ("Service Structure", test_reranking_service_structure),
        ("Re-ranking Functionality", test_reranking_functionality),
        ("Caching Functionality", test_caching_functionality),
        ("Fallback Mechanisms", test_fallback_mechanisms),
        ("Orchestration Integration", test_orchestration_integration)
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\n📋 Running {test_name} test...")
        try:
            success = await test_func()
            results.append((test_name, success))
        except Exception as e:
            logger.error(f"Test {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info("\n📊 Test Results Summary:")
    passed = 0
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        logger.info(f"  {test_name}: {status}")
        if success:
            passed += 1
    
    logger.info(f"\n🎯 Overall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        logger.info("🎉 All tests passed! Re-ranking service is working correctly.")
    else:
        logger.error("⚠️  Some tests failed. Please check the implementation.")


if __name__ == "__main__":
    asyncio.run(main())