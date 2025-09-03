#!/usr/bin/env python3
"""
Test script for the recommendation pipeline implementation.
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


async def test_orchestration_service():
    """Test the orchestration service directly."""
    try:
        # Mock the external dependencies that might not be available
        import unittest.mock as mock
        
        # Mock Redis and other external services
        with mock.patch('redis.Redis'), \
             mock.patch('app.services.retrieval_service.retrieval_service'), \
             mock.patch('app.services.profile_service.profile_service'), \
             mock.patch('app.services.bandit_service.bandit_service'), \
             mock.patch('app.services.math_generation_service.math_generation_service'):
            
            from app.services.orchestration_service import orchestration_service
            
            logger.info("Testing orchestration service...")
            
            # Mock the dependencies to return test data
            orchestration_service.user_repo = mock.Mock()
            orchestration_service.item_repo = mock.Mock()
            orchestration_service.attempt_repo = mock.Mock()
            
            # Test recommendation pipeline
            result = await orchestration_service.recommend_next_questions(
                user_id="test_user_123",
                target_skills=["linear_equation", "algebra"],
                constraints={"language": "tr"},
                personalization={"difficulty_preference": "adaptive"}
            )
            
            logger.info(f"Pipeline result keys: {list(result.keys())}")
            
            # Verify result structure
            assert "items" in result
            assert "metadata" in result
            
            logger.info("✅ Orchestration service test passed!")
            return True
        
    except Exception as e:
        logger.error(f"❌ Orchestration service test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_api_integration():
    """Test API integration (mock test)."""
    try:
        import unittest.mock as mock
        
        # Mock external dependencies
        with mock.patch('redis.Redis'), \
             mock.patch('app.services.retrieval_service.retrieval_service'), \
             mock.patch('app.services.profile_service.profile_service'), \
             mock.patch('app.services.bandit_service.bandit_service'), \
             mock.patch('app.services.math_generation_service.math_generation_service'):
            
            from app.models.request import RecommendRequest
            from app.api.v1.recommend import recommend_next_questions
            
            logger.info("Testing API integration...")
            
            # Create mock request
            request = RecommendRequest(
                user_id="test_user_123",
                target_skills=["linear_equation"],
                constraints={"language": "tr"}
            )
            
            # Mock FastAPI request object
            mock_req = mock.Mock()
            mock_req.headers = {}
            
            # Mock token (in real implementation, this would be validated)
            mock_token = "mock_jwt_token"
            
            # Test the endpoint
            response = await recommend_next_questions(request, mock_req, mock_token)
            
            logger.info(f"API response type: {type(response)}")
            
            # Verify response structure
            assert hasattr(response, 'items')
            assert hasattr(response, 'policy_id')
            assert hasattr(response, 'bandit_version')
            assert hasattr(response, 'request_id')
            
            logger.info("✅ API integration test passed!")
            return True
        
    except Exception as e:
        logger.error(f"❌ API integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_error_handling():
    """Test error handling and fallback mechanisms."""
    try:
        import unittest.mock as mock
        
        # Mock external dependencies
        with mock.patch('redis.Redis'), \
             mock.patch('app.services.retrieval_service.retrieval_service'), \
             mock.patch('app.services.profile_service.profile_service'), \
             mock.patch('app.services.bandit_service.bandit_service'), \
             mock.patch('app.services.math_generation_service.math_generation_service'):
            
            from app.services.orchestration_service import orchestration_service
            
            logger.info("Testing error handling...")
            
            # Mock the dependencies to return test data
            orchestration_service.user_repo = mock.Mock()
            orchestration_service.item_repo = mock.Mock()
            orchestration_service.attempt_repo = mock.Mock()
            
            # Test with invalid user ID
            result = await orchestration_service.recommend_next_questions(
                user_id="",  # Invalid user ID
                target_skills=["nonexistent_skill"],
                constraints={}
            )
            
            # Should still return a result (fallback)
            assert "items" in result
            assert "metadata" in result
            
            logger.info("✅ Error handling test passed!")
            return True
        
    except Exception as e:
        logger.error(f"❌ Error handling test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run all tests."""
    logger.info("🚀 Starting recommendation pipeline tests...")
    
    tests = [
        ("Orchestration Service", test_orchestration_service),
        ("API Integration", test_api_integration),
        ("Error Handling", test_error_handling)
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
        logger.info("🎉 All tests passed! Recommendation pipeline is working correctly.")
    else:
        logger.error("⚠️  Some tests failed. Please check the implementation.")


if __name__ == "__main__":
    asyncio.run(main())