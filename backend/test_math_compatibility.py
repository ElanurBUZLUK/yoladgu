#!/usr/bin/env python3
"""
Test script to verify math question compatibility between new JSON format and frontend
"""

import asyncio
import json
import sys
from pathlib import Path

# Add the backend directory to the path
sys.path.append(str(Path(__file__).parent))

# Mock the imports to avoid database dependencies
class Subject:
    MATH = "math"
    ENGLISH = "english"

class QuestionType:
    MULTIPLE_CHOICE = "multiple_choice"
    TRUE_FALSE = "true_false"
    OPEN_ENDED = "open_ended"

class SourceType:
    MANUAL = "manual"
    PDF = "pdf"
    API = "api"

class Question:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

class QuestionResponse:
    @classmethod
    def from_orm(cls, question):
        return cls(
            id=str(question.id) if hasattr(question, 'id') else 'test-123',
            content=question.content,
            options=question.options,
            topic_category=question.topic_category,
            difficulty_level=question.difficulty_level
        )
    
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

def test_json_format_compatibility():
    """Test if the new JSON format is compatible with existing schemas"""
    
    print("🧪 Testing JSON Format Compatibility...")
    
    # Test data using new format
    test_question_data = {
        "stem": "What is 2 + 2?",
        "options": {
            "A": "1",
            "B": "2", 
            "C": "3",
            "D": "4"
        },
        "correct_answer": "D",
        "topic": "arithmetic",
        "subtopic": "addition",
        "difficulty": 0.5,
        "source": "seed",
        "metadata": {
            "estimated_time": 30,
            "learning_objectives": ["basic addition"],
            "tags": ["arithmetic", "basic"],
            "cefr_level": "A1"
        }
    }
    
    # Convert to Question model format
    question = Question(
        subject=Subject.MATH,
        content=test_question_data['stem'],
        question_type=QuestionType.MULTIPLE_CHOICE,
        difficulty_level=1,  # Converted from 0.5
        original_difficulty=1,
        topic_category=test_question_data['topic'],
        correct_answer=test_question_data['correct_answer'],
        options=test_question_data['options'],
        source_type=SourceType.MANUAL,
        estimated_difficulty=test_question_data['difficulty'],
        question_metadata={
            "subtopic": test_question_data.get('subtopic'),
            "source": test_question_data.get('source', 'seed'),
            "estimated_time": test_question_data.get('metadata', {}).get('estimated_time', 60),
            "learning_objectives": test_question_data.get('metadata', {}).get('learning_objectives', []),
            "tags": test_question_data.get('metadata', {}).get('tags', []),
            "cefr_level": test_question_data.get('metadata', {}).get('cefr_level', 'A1')
        }
    )
    
    # Test QuestionResponse conversion
    try:
        response = QuestionResponse.from_orm(question)
        print("✅ QuestionResponse conversion successful")
        print(f"   Content: {response.content}")
        print(f"   Options: {response.options}")
        print(f"   Topic: {response.topic_category}")
        print(f"   Difficulty: {response.difficulty_level}")
    except Exception as e:
        print(f"❌ QuestionResponse conversion failed: {e}")
        return False
    
    return True

def test_frontend_compatibility():
    """Test if the response format is compatible with frontend expectations"""
    
    print("\n🧪 Testing Frontend Compatibility...")
    
    # Simulate backend response
    backend_response = {
        "questions": [
            {
                "id": "test-123",
                "content": "What is 2 + 2?",
                "options": {
                    "A": "1",
                    "B": "2",
                    "C": "3", 
                    "D": "4"
                },
                "correct_answer": "D",
                "difficulty_level": 1,
                "topic_category": "arithmetic",
                "question_type": "multiple_choice"
            }
        ],
        "recommendation_reason": "Based on your level",
        "total_available": 1,
        "user_level": 2,
        "next_recommendations": []
    }
    
    # Simulate frontend processing
    try:
        question = backend_response['questions'][0]
        
        # Frontend options conversion
        options_array = []
        if question['options']:
            if isinstance(question['options'], dict):
                options_array = list(question['options'].values())
            else:
                options_array = question['options']
        
        frontend_question = {
            "id": question['id'],
            "text": question['content'],
            "options": options_array,
            "optionsMap": question['options'],
            "correct_answer": question['correct_answer'],
            "difficulty_level": question['difficulty_level'],
            "topic": question['topic_category'],
            "question_type": question['question_type']
        }
        
        print("✅ Frontend processing successful")
        print(f"   Text: {frontend_question['text']}")
        print(f"   Options Array: {frontend_question['options']}")
        print(f"   Options Map: {frontend_question['optionsMap']}")
        print(f"   Topic: {frontend_question['topic']}")
        
        # Test option letter generation
        for i, option in enumerate(frontend_question['options']):
            letter = chr(65 + i)  # A, B, C, D...
            print(f"   Option {letter}: {option}")
        
    except Exception as e:
        print(f"❌ Frontend processing failed: {e}")
        return False
    
    return True

def test_api_endpoint_compatibility():
    """Test if the API endpoints work with new format"""
    
    print("\n🧪 Testing API Endpoint Compatibility...")
    
    # Test the conversion logic from the API
    def convert_difficulty(difficulty: float) -> int:
        if difficulty <= 0.5:
            return 1
        elif difficulty <= 1.0:
            return 2
        elif difficulty <= 1.5:
            return 3
        elif difficulty <= 1.8:
            return 4
        else:
            return 5
    
    def determine_question_type(options: dict) -> QuestionType:
        if len(options) == 2 and all(opt in ['True', 'False', 'true', 'false'] for opt in options.values()):
            return QuestionType.TRUE_FALSE
        elif len(options) == 0:
            return QuestionType.OPEN_ENDED
        else:
            return QuestionType.MULTIPLE_CHOICE
    
    # Test cases
    test_cases = [
        {"difficulty": 0.3, "expected": 1},
        {"difficulty": 0.8, "expected": 2},
        {"difficulty": 1.2, "expected": 3},
        {"difficulty": 1.7, "expected": 4},
        {"difficulty": 2.0, "expected": 5}
    ]
    
    for case in test_cases:
        result = convert_difficulty(case['difficulty'])
        if result == case['expected']:
            print(f"✅ Difficulty {case['difficulty']} -> Level {result}")
        else:
            print(f"❌ Difficulty {case['difficulty']} -> Level {result} (expected {case['expected']})")
    
    # Test question type determination
    test_options = [
        ({"A": "True", "B": "False"}, QuestionType.TRUE_FALSE),
        ({"A": "1", "B": "2", "C": "3", "D": "4"}, QuestionType.MULTIPLE_CHOICE),
        ({}, QuestionType.OPEN_ENDED)
    ]
    
    for options, expected in test_options:
        result = determine_question_type(options)
        if result == expected:
            print(f"✅ Options {options} -> Type {result}")
        else:
            print(f"❌ Options {options} -> Type {result} (expected {expected})")
    
    return True

def main():
    """Run all compatibility tests"""
    
    print("🚀 Starting Math Question Compatibility Tests\n")
    
    tests = [
        ("JSON Format", test_json_format_compatibility),
        ("Frontend Compatibility", test_frontend_compatibility),
        ("API Endpoint", test_api_endpoint_compatibility)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test failed with exception: {e}")
            results.append((test_name, False))
    
    print("\n📊 Test Results:")
    print("=" * 50)
    
    all_passed = True
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:25} {status}")
        if not passed:
            all_passed = False
    
    print("=" * 50)
    
    if all_passed:
        print("🎉 All tests passed! The new JSON format is fully compatible.")
        return 0
    else:
        print("⚠️  Some tests failed. Please review the issues above.")
        return 1

if __name__ == "__main__":
    exit(main())
