#!/usr/bin/env python3
"""
Test script for enhanced QuestionGenerator with orchestration, caching, and logging
"""

import asyncio
import logging
from datetime import datetime
from app.services.question_generator import question_generator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

async def test_question_generation():
    """Test the enhanced question generation system"""
    print("🚀 Testing Enhanced QuestionGenerator System")
    print("=" * 50)
    
    # Test 1: Template-based generation
    print("\n📝 Test 1: Template-based Generation")
    print("-" * 30)
    
    question1 = await question_generator.generate_question(
        subject="english",
        error_type="present_perfect_error",
        difficulty_level=3
    )
    
    print(f"Generated Question: {question1['question']}")
    print(f"Answer: {question1['answer']}")
    print(f"Method: {question1['metadata']['generation_method']}")
    print(f"Strategy: {question1['metadata']['orchestration_strategy']}")
    print(f"Generation Time: {question1['metadata']['generation_time_ms']:.2f}ms")
    
    # Test 2: Same request (should hit cache)
    print("\n🔄 Test 2: Cache Hit Test")
    print("-" * 30)
    
    question2 = await question_generator.generate_question(
        subject="english",
        error_type="present_perfect_error",
        difficulty_level=3
    )
    
    print(f"Cache Hit: {question2['metadata']['cache_key'] == question1['metadata']['cache_key']}")
    print(f"Generation Time: {question2['metadata']['generation_time_ms']:.2f}ms")
    
    # Test 3: Math question
    print("\n🔢 Test 3: Math Question Generation")
    print("-" * 30)
    
    question3 = await question_generator.generate_question(
        subject="math",
        error_type="algebra_equation",
        difficulty_level=4
    )
    
    print(f"Generated Question: {question3['question']}")
    print(f"Answer: {question3['answer']}")
    print(f"Method: {question3['metadata']['generation_method']}")
    
    # Test 4: Non-existent error type (should use fallback)
    print("\n⚠️ Test 4: Non-existent Error Type (Fallback)")
    print("-" * 30)
    
    question4 = await question_generator.generate_question(
        subject="english",
        error_type="non_existent_error_type",
        difficulty_level=2
    )
    
    print(f"Generated Question: {question4['question']}")
    print(f"Method: {question4['metadata']['generation_method']}")
    
    # Test 5: Get comprehensive statistics
    print("\n📊 Test 5: Generation Statistics")
    print("-" * 30)
    
    stats = question_generator.get_generation_stats()
    print(f"Today's GPT Usage: {stats['today_gpt_usage']}/{stats['daily_gpt_limit']}")
    print(f"Templates Loaded: {stats['templates_loaded']}")
    print(f"Cache Performance: {stats['cache_performance']}")
    print(f"Orchestration Settings: {stats['orchestration_settings']}")
    
    # Test 6: Cache management
    print("\n🧹 Test 6: Cache Management")
    print("-" * 30)
    
    print(f"Cache size before cleanup: {len(question_generator.question_cache)}")
    expired_cleared = question_generator.clear_expired_cache()
    print(f"Expired entries cleared: {expired_cleared}")
    print(f"Cache size after cleanup: {len(question_generator.question_cache)}")
    
    print("\n✅ All tests completed successfully!")
    print("=" * 50)

def test_orchestration_logic():
    """Test the orchestration logic separately"""
    print("\n🎯 Testing Orchestration Logic")
    print("-" * 30)
    
    # Test different scenarios
    test_cases = [
        ("english", "present_perfect_error", 3),
        ("math", "algebra_equation", 4),
        ("english", "non_existent_type", 2),
    ]
    
    for subject, error_type, difficulty in test_cases:
        strategy = question_generator._orchestrate_generation_strategy(
            subject, error_type, difficulty
        )
        template_exists = question_generator._has_template_for_error_type(subject, error_type)
        
        print(f"{subject}/{error_type} (difficulty {difficulty}):")
        print(f"  Template exists: {template_exists}")
        print(f"  Strategy chosen: {strategy}")
        print()

if __name__ == "__main__":
    print("🔧 QuestionGenerator Test Suite")
    print("=" * 50)
    
    # Test orchestration logic
    test_orchestration_logic()
    
    # Test async question generation
    try:
        asyncio.run(test_question_generation())
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        print("This might be due to missing configuration or services")
        print("Check that the backend is properly configured")
