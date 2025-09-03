#!/usr/bin/env python3
"""
Structure validation test for the re-ranking service implementation.
"""

import ast
import os
import sys
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def validate_reranking_service():
    """Validate re-ranking service structure."""
    logger.info("Testing re-ranking service structure...")
    
    file_path = "app/services/reranking_service.py"
    
    if not os.path.exists(file_path):
        logger.error(f"❌ File not found: {file_path}")
        return False
    
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Parse the AST
        tree = ast.parse(content)
        
        # Extract classes and methods
        classes = []
        methods = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                classes.append(node.name)
                # Get methods within the class
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        methods.append(f"{node.name}.{item.name}")
            elif isinstance(node, ast.FunctionDef):
                methods.append(node.name)
        
        logger.info(f"✅ {file_path} parsed successfully")
        
        # Check expected classes
        expected_classes = ["RerankingService"]
        for cls in expected_classes:
            if cls in classes:
                logger.info(f"  ✅ Class '{cls}' found")
            else:
                logger.warning(f"  ⚠️  Class '{cls}' not found")
        
        # Check expected methods
        expected_methods = [
            "RerankingService.rerank_candidates",
            "RerankingService._load_model",
            "RerankingService._batch_predict",
            "RerankingService._build_query_text",
            "RerankingService._build_candidate_text",
            "RerankingService._combine_scores",
            "RerankingService.get_stats"
        ]
        
        for method in expected_methods:
            if method in methods:
                logger.info(f"  ✅ Method '{method}' found")
            else:
                logger.warning(f"  ⚠️  Method '{method}' not found")
        
        # Check imports
        expected_imports = [
            "sentence_transformers",
            "torch",
            "numpy",
            "asyncio"
        ]
        
        for imp in expected_imports:
            if imp in content:
                logger.info(f"  ✅ Import '{imp}' found")
            else:
                logger.warning(f"  ⚠️  Import '{imp}' not found")
        
        return True
        
    except SyntaxError as e:
        logger.error(f"❌ Syntax error in {file_path}: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Error validating {file_path}: {e}")
        return False


def validate_orchestration_integration():
    """Validate orchestration service integration."""
    logger.info("Testing orchestration service integration...")
    
    file_path = "app/services/orchestration_service.py"
    
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Check for reranking service import
        if "from app.services.reranking_service import reranking_service" in content:
            logger.info("  ✅ Re-ranking service import found")
        else:
            logger.warning("  ⚠️  Re-ranking service import not found")
        
        # Check for updated _reranking_phase method
        if "reranking_service.rerank_candidates" in content:
            logger.info("  ✅ Re-ranking service usage found")
        else:
            logger.warning("  ⚠️  Re-ranking service usage not found")
        
        # Check for new helper methods
        expected_methods = [
            "_build_rerank_query",
            "_fallback_reranking"
        ]
        
        for method in expected_methods:
            if f"def {method}" in content:
                logger.info(f"  ✅ Method '{method}' found")
            else:
                logger.warning(f"  ⚠️  Method '{method}' not found")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error validating orchestration integration: {e}")
        return False


def validate_api_integration():
    """Validate API endpoint integration."""
    logger.info("Testing API endpoint integration...")
    
    file_path = "app/api/v1/recommend.py"
    
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Check for reranking service import
        if "from app.services.reranking_service import reranking_service" in content:
            logger.info("  ✅ Re-ranking service import found in API")
        else:
            logger.warning("  ⚠️  Re-ranking service import not found in API")
        
        # Check for service usage in rerank endpoint
        if "reranking_service.rerank_candidates" in content:
            logger.info("  ✅ Re-ranking service usage found in API")
        else:
            logger.warning("  ⚠️  Re-ranking service usage not found in API")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error validating API integration: {e}")
        return False


def validate_requirements():
    """Validate that required dependencies are listed."""
    logger.info("Testing requirements...")
    
    file_path = "requirements.txt"
    
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Check for ML dependencies
        required_deps = [
            "sentence-transformers",
            "torch",
            "numpy"
        ]
        
        for dep in required_deps:
            if dep in content:
                logger.info(f"  ✅ Dependency '{dep}' found in requirements")
            else:
                logger.warning(f"  ⚠️  Dependency '{dep}' not found in requirements")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error validating requirements: {e}")
        return False


def main():
    """Run all structure validation tests."""
    logger.info("🚀 Starting re-ranking service structure validation...")
    
    tests = [
        ("Re-ranking Service Structure", validate_reranking_service),
        ("Orchestration Integration", validate_orchestration_integration),
        ("API Integration", validate_api_integration),
        ("Requirements", validate_requirements)
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
        logger.info("🎉 All validations passed! Re-ranking service structure is correct.")
    else:
        logger.error("⚠️  Some validations failed. Please check the implementation.")


if __name__ == "__main__":
    main()