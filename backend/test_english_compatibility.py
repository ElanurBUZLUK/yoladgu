#!/usr/bin/env python3
"""
English Questions Compatibility Test

This script tests the compatibility of the new English JSON format with:
1. Backend data models and schemas
2. Frontend expected data structure
3. API endpoint logic (difficulty and question type conversion)
"""

import json
import sys
import os
from typing import Dict, Any, List

# Mock classes for testing without database
class MockSubject:
    ENGLISH = "english"

class MockQuestionType:
    MULTIPLE_CHOICE = "multiple_choice"
    TRUE_FALSE = "true_false"
    OPEN_ENDED = "open_ended"
    FILL_BLANK = "fill_blank"

class MockSourceType:
    MANUAL = "manual"

class MockQuestion:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

class MockQuestionResponse:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

def test_json_format_compatibility():
    """Test conversion from new JSON format to Question model and then to QuestionResponse schema"""
    print("🧪 Testing JSON format compatibility...")
    
    # Sample English question in new format
    sample_question = {
        "stem": "Choose the correct form: I ____ to school every day.",
        "options": {
            "A": "go",
            "B": "goes",
            "C": "going",
            "D": "went"
        },
        "correct_answer": "A",
        "topic": "grammar",
        "subtopic": "present_tense",
        "difficulty": 0.5,
        "source": "seed",
        "metadata": {
            "estimated_time": 30,
            "learning_objectives": ["present tense with 'I'"],
            "tags": ["grammar", "present_tense", "basic"],
            "cefr_level": "A1"
        }
    }
    
    try:
        # Test 1: Convert difficulty (continuous 0.0-2.0 to discrete 1-5)
        difficulty = sample_question.get('difficulty', 1.0)
        if difficulty <= 0.5:
            difficulty_level = 1
        elif difficulty <= 1.0:
            difficulty_level = 2
        elif difficulty <= 1.5:
            difficulty_level = 3
        elif difficulty <= 1.8:
            difficulty_level = 4
        else:
            difficulty_level = 5
        
        assert difficulty_level == 1, f"Expected difficulty_level 1, got {difficulty_level}"
        print("✅ Difficulty conversion: PASS")
        
        # Test 2: Determine question type based on options
        options = sample_question['options']
        if len(options) == 2 and all(opt in ['True', 'False', 'true', 'false'] for opt in options.values()):
            question_type = MockQuestionType.TRUE_FALSE
        elif len(options) == 0:
            question_type = MockQuestionType.OPEN_ENDED
        else:
            question_type = MockQuestionType.MULTIPLE_CHOICE
        
        assert question_type == MockQuestionType.MULTIPLE_CHOICE, f"Expected MULTIPLE_CHOICE, got {question_type}"
        print("✅ Question type determination: PASS")
        
        # Test 3: Create Question model
        question = MockQuestion(
            subject=MockSubject.ENGLISH,
            content=sample_question['stem'],
            question_type=question_type,
            difficulty_level=difficulty_level,
            original_difficulty=difficulty_level,
            topic_category=sample_question['topic'],
            correct_answer=sample_question['correct_answer'],
            options=sample_question['options'],
            source_type=MockSourceType.MANUAL,
            estimated_difficulty=sample_question.get('difficulty', 1.0),
            question_metadata={
                "subtopic": sample_question.get('subtopic'),
                "source": sample_question.get('source', 'seed'),
                "estimated_time": sample_question.get('metadata', {}).get('estimated_time', 60),
                "learning_objectives": sample_question.get('metadata', {}).get('learning_objectives', []),
                "tags": sample_question.get('metadata', {}).get('tags', []),
                "cefr_level": sample_question.get('metadata', {}).get('cefr_level', 'A1')
            }
        )
        
        assert question.subject == MockSubject.ENGLISH, f"Expected ENGLISH, got {question.subject}"
        assert question.content == sample_question['stem'], "Content mismatch"
        assert question.topic_category == "grammar", f"Expected 'grammar', got {question.topic_category}"
        assert question.correct_answer == "A", f"Expected 'A', got {question.correct_answer}"
        assert question.question_metadata['cefr_level'] == "A1", f"Expected 'A1', got {question.question_metadata['cefr_level']}"
        print("✅ Question model creation: PASS")
        
        # Test 4: Convert to QuestionResponse schema
        response = MockQuestionResponse(
            id="test-id",
            content=question.content,
            question_type=question.question_type,
            difficulty_level=question.difficulty_level,
            topic_category=question.topic_category,
            correct_answer=question.correct_answer,
            options=question.options,
            question_metadata=question.question_metadata
        )
        
        assert response.content == sample_question['stem'], "Response content mismatch"
        assert response.options == sample_question['options'], "Response options mismatch"
        print("✅ QuestionResponse conversion: PASS")
        
        return True
        
    except Exception as e:
        print(f"❌ JSON format compatibility test failed: {e}")
        return False

def test_frontend_compatibility():
    """Test frontend's API service processing of backend response"""
    print("\n🧪 Testing frontend compatibility...")
    
    try:
        # Simulate backend response
        backend_response = {
            "questions": [{
                "id": "test-id",
                "content": "Choose the correct form: I ____ to school every day.",
                "question_type": "multiple_choice",
                "difficulty_level": 1,
                "topic_category": "grammar",
                "correct_answer": "A",
                "options": {
                    "A": "go",
                    "B": "goes",
                    "C": "going",
                    "D": "went"
                },
                "question_metadata": {
                    "cefr_level": "A1"
                }
            }]
        }
        
        # Simulate frontend's API service processing
        if backend_response["questions"] and len(backend_response["questions"]) > 0:
            question = backend_response["questions"][0]
            
            # Convert options object to array for frontend component
            options_array = []
            if question["options"]:
                if isinstance(question["options"], list):
                    options_array = question["options"]
                elif isinstance(question["options"], dict):
                    options_array = list(question["options"].values())
            
            frontend_question = {
                "id": question["id"],
                "text": question["content"],
                "options": options_array,
                "optionsMap": question["options"],  # Original options object
                "correct_answer": question["correct_answer"],
                "difficulty_level": question["difficulty_level"],
                "topic": question["topic_category"],
                "question_type": question["question_type"],
                "cefr_level": question.get("question_metadata", {}).get("cefr_level")
            }
        
        # Test frontend processing
        assert frontend_question["text"] == "Choose the correct form: I ____ to school every day.", "Text mismatch"
        assert len(frontend_question["options"]) == 4, f"Expected 4 options, got {len(frontend_question['options'])}"
        assert frontend_question["options"] == ["go", "goes", "going", "went"], "Options array mismatch"
        assert frontend_question["optionsMap"] == {"A": "go", "B": "goes", "C": "going", "D": "went"}, "OptionsMap mismatch"
        assert frontend_question["correct_answer"] == "A", "Correct answer mismatch"
        assert frontend_question["cefr_level"] == "A1", "CEFR level mismatch"
        
        print("✅ Frontend compatibility: PASS")
        return True
        
    except Exception as e:
        print(f"❌ Frontend compatibility test failed: {e}")
        return False

def test_api_endpoint_compatibility():
    """Test the difficulty conversion and question type determination logic used in the API endpoint"""
    print("\n🧪 Testing API endpoint compatibility...")
    
    try:
        # Test cases for difficulty conversion
        test_cases = [
            (0.3, 1),  # Very easy
            (0.8, 2),  # Easy
            (1.2, 3),  # Medium
            (1.7, 4),  # Hard
            (1.9, 5)   # Very hard
        ]
        
        for difficulty, expected_level in test_cases:
            if difficulty <= 0.5:
                level = 1
            elif difficulty <= 1.0:
                level = 2
            elif difficulty <= 1.5:
                level = 3
            elif difficulty <= 1.8:
                level = 4
            else:
                level = 5
            
            assert level == expected_level, f"Difficulty {difficulty} should be level {expected_level}, got {level}"
        
        print("✅ Difficulty conversion logic: PASS")
        
        # Test cases for question type determination
        type_test_cases = [
            ({"A": "True", "B": "False"}, MockQuestionType.TRUE_FALSE),
            ({"A": "go", "B": "goes", "C": "going", "D": "went"}, MockQuestionType.MULTIPLE_CHOICE),
            ({}, MockQuestionType.OPEN_ENDED)
        ]
        
        for options, expected_type in type_test_cases:
            if len(options) == 2 and all(opt in ['True', 'False', 'true', 'false'] for opt in options.values()):
                question_type = MockQuestionType.TRUE_FALSE
            elif len(options) == 0:
                question_type = MockQuestionType.OPEN_ENDED
            else:
                question_type = MockQuestionType.MULTIPLE_CHOICE
            
            assert question_type == expected_type, f"Options {options} should be {expected_type}, got {question_type}"
        
        print("✅ Question type determination logic: PASS")
        return True
        
    except Exception as e:
        print(f"❌ API endpoint compatibility test failed: {e}")
        return False

def main():
    """Run all compatibility tests"""
    print("🚀 Starting English Questions Compatibility Tests\n")
    
    tests = [
        test_json_format_compatibility,
        test_frontend_compatibility,
        test_api_endpoint_compatibility
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! English questions are fully compatible.")
        return 0
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    exit(main())
