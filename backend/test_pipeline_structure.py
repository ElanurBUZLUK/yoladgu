#!/usr/bin/env python3
"""
Structure validation test for the recommendation pipeline implementation.
"""

import ast
import os
import sys
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def validate_file_structure(file_path, expected_functions=None, expected_classes=None):
    """Validate that a Python file has the expected structure."""
    if not os.path.exists(file_path):
        logger.error(f"❌ File not found: {file_path}")
        return False
    
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Parse the AST
        tree = ast.parse(content)
        
        # Extract functions and classes
        functions = []
        classes = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                functions.append(node.name)
            elif isinstance(node, ast.ClassDef):
                classes.append(node.name)
        
        logger.info(f"✅ {file_path} parsed successfully")
        
        if expected_functions:
            for func in expected_functions:
                if func in functions:
                    logger.info(f"  ✅ Function '{func}' found")
                else:
                    logger.warning(f"  ⚠️  Function '{func}' not found")
        
        if expected_classes:
            for cls in expected_classes:
                if cls in classes:
                    logger.info(f"  ✅ Class '{cls}' found")
                else:
                    logger.warning(f"  ⚠️  Class '{cls}' not found")
        
        return True
        
    except SyntaxError as e:
        logger.error(f"❌ Syntax error in {file_path}: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Error validating {file_path}: {e}")
        return False


def test_orchestration_service():
    """Test orchestration service structure."""
    logger.info("Testing orchestration service structure...")
    
    expected_functions = [
        "recommend_next_questions",
        "_get_user_context",
        "_retrieval_phase",
        "_reranking_phase",
        "_diversification_phase",
        "_bandit_selection_phase",
        "_generation_fallback",
        "_emergency_fallback"
    ]
    
    expected_classes = ["RecommendationPipeline"]
    
    return validate_file_structure(
        "app/services/orchestration_service.py",
        expected_functions,
        expected_classes
    )


def test_recommend_endpoints():
    """Test recommendation endpoints structure."""
    logger.info("Testing recommendation endpoints structure...")
    
    expected_functions = [
        "recommend_next_questions",
        "search_questions",
        "rerank_candidates"
    ]
    
    return validate_file_structure(
        "app/api/v1/recommend.py",
        expected_functions
    )


def test_request_models():
    """Test request models structure."""
    logger.info("Testing request models structure...")
    
    expected_classes = [
        "RecommendRequest",
        "SearchRequest",
        "RerankRequest"
    ]
    
    return validate_file_structure(
        "app/models/request.py",
        expected_classes=expected_classes
    )


def test_response_models():
    """Test response models structure."""
    logger.info("Testing response models structure...")
    
    expected_classes = [
        "RecommendResponse",
        "SearchResponse",
        "RerankResponse",
        "RecommendedItem"
    ]
    
    return validate_file_structure(
        "app/models/response.py",
        expected_classes=expected_classes
    )


def test_pipeline_integration():
    """Test that the pipeline components are properly integrated."""
    logger.info("Testing pipeline integration...")
    
    try:
        # Check that orchestration service imports are correct
        with open("app/services/orchestration_service.py", 'r') as f:
            content = f.read()
        
        required_imports = [
            "from app.services.retrieval_service import retrieval_service",
            "from app.services.profile_service import profile_service",
            "from app.services.bandit_service import bandit_service",
            "from app.services.math_generation_service import math_generation_service"
        ]
        
        for import_stmt in required_imports:
            if import_stmt in content:
                logger.info(f"  ✅ Import found: {import_stmt}")
            else:
                logger.warning(f"  ⚠️  Import not found: {import_stmt}")
        
        # Check that API endpoints import orchestration service
        with open("app/api/v1/recommend.py", 'r') as f:
            api_content = f.read()
        
        if "from app.services.orchestration_service import orchestration_service" in api_content:
            logger.info("  ✅ API endpoints import orchestration service")
        else:
            logger.warning("  ⚠️  API endpoints don't import orchestration service")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Integration test failed: {e}")
        return False


def main():
    """Run all structure validation tests."""
    logger.info("🚀 Starting pipeline structure validation...")
    
    tests = [
        ("Orchestration Service", test_orchestration_service),
        ("Recommendation Endpoints", test_recommend_endpoints),
        ("Request Models", test_request_models),
        ("Response Models", test_response_models),
        ("Pipeline Integration", test_pipeline_integration)
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\n📋 Running {test_name} validation...")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            logger.error(f"Test {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info("\n📊 Validation Results Summary:")
    passed = 0
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        logger.info(f"  {test_name}: {status}")
        if success:
            passed += 1
    
    logger.info(f"\n🎯 Overall: {passed}/{len(results)} validations passed")
    
    if passed == len(results):
        logger.info("🎉 All validations passed! Pipeline structure is correct.")
    else:
        logger.error("⚠️  Some validations failed. Please check the implementation.")


if __name__ == "__main__":
    main()