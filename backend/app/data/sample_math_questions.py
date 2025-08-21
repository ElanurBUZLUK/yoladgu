"""
Sample mathematics questions for seeding the database
"""

import json
import os
from typing import List, Dict, Any
from app.models.question import Subject, QuestionType, SourceType

def load_questions_from_json(file_path: str) -> List[Dict[str, Any]]:
    """
    Load questions from a JSON file
    
    Expected JSON format:
    [
        {
            "subject": "math",
            "content": "What is 5 + 3?",
            "question_type": "multiple_choice",
            "difficulty_level": 1,
            "topic_category": "addition",
            "correct_answer": "8",
            "options": ["6", "7", "8", "9"],
            "source_type": "manual",
            "question_metadata": {
                "estimated_time": 30,
                "learning_objectives": ["basic addition"],
                "tags": ["arithmetic", "basic"]
            }
        }
    ]
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            questions_data = json.load(f)
        
        # Convert string values to enum types
        converted_questions = []
        for question in questions_data:
            converted_question = question.copy()
            
            # Convert subject
            if isinstance(question.get('subject'), str):
                converted_question['subject'] = Subject.MATH if question['subject'].lower() == 'math' else Subject.ENGLISH
            
            # Convert question_type
            if isinstance(question.get('question_type'), str):
                question_type_map = {
                    'multiple_choice': QuestionType.MULTIPLE_CHOICE,
                    'fill_blank': QuestionType.FILL_BLANK,
                    'open_ended': QuestionType.OPEN_ENDED,
                    'true_false': QuestionType.TRUE_FALSE
                }
                converted_question['question_type'] = question_type_map.get(
                    question['question_type'].lower(), QuestionType.MULTIPLE_CHOICE
                )
            
            # Convert source_type
            if isinstance(question.get('source_type'), str):
                source_type_map = {
                    'manual': SourceType.MANUAL,
                    'pdf': SourceType.PDF,
                    'api': SourceType.API
                }
                converted_question['source_type'] = source_type_map.get(
                    question['source_type'].lower(), SourceType.MANUAL
                )
            
            converted_questions.append(converted_question)
        
        return converted_questions
    
    except FileNotFoundError:
        print(f"❌ JSON file not found: {file_path}")
        return []
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON format in {file_path}: {e}")
        return []
    except Exception as e:
        print(f"❌ Error loading questions from {file_path}: {e}")
        return []

def get_questions_from_json_directory(directory_path: str = "data/questions") -> List[Dict[str, Any]]:
    """
    Load all JSON question files from a directory
    """
    all_questions = []
    
    if not os.path.exists(directory_path):
        print(f"📁 Creating directory: {directory_path}")
        os.makedirs(directory_path, exist_ok=True)
        return all_questions
    
    for filename in os.listdir(directory_path):
        if filename.endswith('.json'):
            file_path = os.path.join(directory_path, filename)
            print(f"📄 Loading questions from: {filename}")
            questions = load_questions_from_json(file_path)
            all_questions.extend(questions)
            print(f"✅ Loaded {len(questions)} questions from {filename}")
    
    return all_questions

# Load questions from JSON files if they exist
JSON_QUESTIONS = get_questions_from_json_directory()

# Combine JSON questions with sample questions
SAMPLE_MATH_QUESTIONS = JSON_QUESTIONS + [
    # Level 1 - Basic Arithmetic
    {
        "subject": Subject.MATH,
        "content": "What is 5 + 3?",
        "question_type": QuestionType.MULTIPLE_CHOICE,
        "difficulty_level": 1,
        "topic_category": "addition",
        "correct_answer": "8",
        "options": ["6", "7", "8", "9"],
        "source_type": SourceType.MANUAL,
        "question_metadata": {
            "estimated_time": 30,
            "learning_objectives": ["basic addition"],
            "tags": ["arithmetic", "basic"]
        }
    },
    {
        "subject": Subject.MATH,
        "content": "What is 12 - 7?",
        "question_type": QuestionType.MULTIPLE_CHOICE,
        "difficulty_level": 1,
        "topic_category": "subtraction",
        "correct_answer": "5",
        "options": ["4", "5", "6", "7"],
        "source_type": SourceType.MANUAL,
        "question_metadata": {
            "estimated_time": 30,
            "learning_objectives": ["basic subtraction"],
            "tags": ["arithmetic", "basic"]
        }
    },
    {
        "subject": Subject.MATH,
        "content": "Fill in the blank: 4 × 3 = ____",
        "question_type": QuestionType.FILL_BLANK,
        "difficulty_level": 1,
        "topic_category": "multiplication",
        "correct_answer": "12",
        "source_type": SourceType.MANUAL,
        "question_metadata": {
            "estimated_time": 45,
            "learning_objectives": ["basic multiplication"],
            "tags": ["arithmetic", "multiplication"]
        }
    },
    
    # Level 2 - Intermediate Arithmetic
    {
        "subject": Subject.MATH,
        "content": "What is 15 × 8?",
        "question_type": QuestionType.MULTIPLE_CHOICE,
        "difficulty_level": 2,
        "topic_category": "multiplication",
        "correct_answer": "120",
        "options": ["110", "115", "120", "125"],
        "source_type": SourceType.MANUAL,
        "question_metadata": {
            "estimated_time": 60,
            "learning_objectives": ["two-digit multiplication"],
            "tags": ["arithmetic", "multiplication"]
        }
    },
    {
        "subject": Subject.MATH,
        "content": "What is 144 ÷ 12?",
        "question_type": QuestionType.MULTIPLE_CHOICE,
        "difficulty_level": 2,
        "topic_category": "division",
        "correct_answer": "12",
        "options": ["10", "11", "12", "13"],
        "source_type": SourceType.MANUAL,
        "question_metadata": {
            "estimated_time": 60,
            "learning_objectives": ["division with larger numbers"],
            "tags": ["arithmetic", "division"]
        }
    },
    {
        "subject": Subject.MATH,
        "content": "Solve: 25 + 37 - 18 = ?",
        "question_type": QuestionType.OPEN_ENDED,
        "difficulty_level": 2,
        "topic_category": "mixed_operations",
        "correct_answer": "44",
        "source_type": SourceType.MANUAL,
        "question_metadata": {
            "estimated_time": 90,
            "learning_objectives": ["order of operations", "mixed arithmetic"],
            "tags": ["arithmetic", "mixed_operations"]
        }
    },
    
    # Level 3 - Fractions and Decimals
    {
        "subject": Subject.MATH,
        "content": "What is 1/2 + 1/4?",
        "question_type": QuestionType.MULTIPLE_CHOICE,
        "difficulty_level": 3,
        "topic_category": "fractions",
        "correct_answer": "3/4",
        "options": ["1/2", "2/4", "3/4", "4/4"],
        "source_type": SourceType.MANUAL,
        "question_metadata": {
            "estimated_time": 120,
            "learning_objectives": ["fraction addition", "common denominators"],
            "tags": ["fractions", "addition"]
        }
    },
    {
        "subject": Subject.MATH,
        "content": "Convert 0.75 to a fraction in its simplest form.",
        "question_type": QuestionType.OPEN_ENDED,
        "difficulty_level": 3,
        "topic_category": "decimals_fractions",
        "correct_answer": "3/4",
        "source_type": SourceType.MANUAL,
        "question_metadata": {
            "estimated_time": 150,
            "learning_objectives": ["decimal to fraction conversion", "simplifying fractions"],
            "tags": ["decimals", "fractions", "conversion"]
        }
    },
    {
        "subject": Subject.MATH,
        "content": "What is 2.5 × 4.2?",
        "question_type": QuestionType.MULTIPLE_CHOICE,
        "difficulty_level": 3,
        "topic_category": "decimal_multiplication",
        "correct_answer": "10.5",
        "options": ["10.0", "10.5", "11.0", "11.5"],
        "source_type": SourceType.MANUAL,
        "question_metadata": {
            "estimated_time": 120,
            "learning_objectives": ["decimal multiplication"],
            "tags": ["decimals", "multiplication"]
        }
    },
    
    # Level 4 - Algebra Basics
    {
        "subject": Subject.MATH,
        "content": "Solve for x: 2x + 5 = 13",
        "question_type": QuestionType.MULTIPLE_CHOICE,
        "difficulty_level": 4,
        "topic_category": "linear_equations",
        "correct_answer": "x = 4",
        "options": ["x = 3", "x = 4", "x = 5", "x = 6"],
        "source_type": SourceType.MANUAL,
        "question_metadata": {
            "estimated_time": 180,
            "learning_objectives": ["solving linear equations", "algebraic manipulation"],
            "tags": ["algebra", "equations", "solving"]
        }
    },
    {
        "subject": Subject.MATH,
        "content": "What is the value of 3x² when x = 4?",
        "question_type": QuestionType.MULTIPLE_CHOICE,
        "difficulty_level": 4,
        "topic_category": "algebraic_expressions",
        "correct_answer": "48",
        "options": ["36", "42", "48", "54"],
        "source_type": SourceType.MANUAL,
        "question_metadata": {
            "estimated_time": 120,
            "learning_objectives": ["evaluating algebraic expressions", "exponents"],
            "tags": ["algebra", "expressions", "exponents"]
        }
    },
    {
        "subject": Subject.MATH,
        "content": "Simplify: 4x + 3x - 2x",
        "question_type": QuestionType.OPEN_ENDED,
        "difficulty_level": 4,
        "topic_category": "algebraic_simplification",
        "correct_answer": "5x",
        "source_type": SourceType.MANUAL,
        "question_metadata": {
            "estimated_time": 90,
            "learning_objectives": ["combining like terms", "algebraic simplification"],
            "tags": ["algebra", "simplification", "like_terms"]
        }
    },
    
    # Level 5 - Advanced Topics
    {
        "subject": Subject.MATH,
        "content": "Solve the quadratic equation: x² - 5x + 6 = 0",
        "question_type": QuestionType.MULTIPLE_CHOICE,
        "difficulty_level": 5,
        "topic_category": "quadratic_equations",
        "correct_answer": "x = 2, x = 3",
        "options": ["x = 1, x = 6", "x = 2, x = 3", "x = -2, x = -3", "x = 0, x = 5"],
        "source_type": SourceType.MANUAL,
        "question_metadata": {
            "estimated_time": 300,
            "learning_objectives": ["solving quadratic equations", "factoring"],
            "tags": ["algebra", "quadratic", "factoring"]
        }
    },
    {
        "subject": Subject.MATH,
        "content": "Find the area of a circle with radius 5 cm. (Use π ≈ 3.14)",
        "question_type": QuestionType.MULTIPLE_CHOICE,
        "difficulty_level": 5,
        "topic_category": "geometry",
        "correct_answer": "78.5 cm²",
        "options": ["62.8 cm²", "78.5 cm²", "94.2 cm²", "157 cm²"],
        "source_type": SourceType.MANUAL,
        "question_metadata": {
            "estimated_time": 180,
            "learning_objectives": ["circle area formula", "geometry calculations"],
            "tags": ["geometry", "circles", "area"]
        }
    },
    {
        "subject": Subject.MATH,
        "content": "If f(x) = 2x + 3, what is f(5)?",
        "question_type": QuestionType.OPEN_ENDED,
        "difficulty_level": 5,
        "topic_category": "functions",
        "correct_answer": "13",
        "source_type": SourceType.MANUAL,
        "question_metadata": {
            "estimated_time": 120,
            "learning_objectives": ["function evaluation", "function notation"],
            "tags": ["functions", "evaluation", "algebra"]
        }
    },
    
    # Additional questions for variety
    {
        "subject": Subject.MATH,
        "content": "True or False: 7 × 8 = 56",
        "question_type": QuestionType.TRUE_FALSE,
        "difficulty_level": 2,
        "topic_category": "multiplication",
        "correct_answer": "True",
        "options": ["True", "False"],
        "source_type": SourceType.MANUAL,
        "question_metadata": {
            "estimated_time": 30,
            "learning_objectives": ["multiplication facts"],
            "tags": ["arithmetic", "multiplication", "facts"]
        }
    },
    {
        "subject": Subject.MATH,
        "content": "What is the next number in the sequence: 2, 4, 6, 8, ___?",
        "question_type": QuestionType.FILL_BLANK,
        "difficulty_level": 2,
        "topic_category": "patterns",
        "correct_answer": "10",
        "source_type": SourceType.MANUAL,
        "question_metadata": {
            "estimated_time": 60,
            "learning_objectives": ["number patterns", "sequences"],
            "tags": ["patterns", "sequences", "arithmetic"]
        }
    },
    {
        "subject": Subject.MATH,
        "content": "A rectangle has a length of 8 cm and width of 5 cm. What is its perimeter?",
        "question_type": QuestionType.MULTIPLE_CHOICE,
        "difficulty_level": 3,
        "topic_category": "geometry",
        "correct_answer": "26 cm",
        "options": ["18 cm", "22 cm", "26 cm", "40 cm"],
        "source_type": SourceType.MANUAL,
        "question_metadata": {
            "estimated_time": 120,
            "learning_objectives": ["perimeter calculation", "rectangle properties"],
            "tags": ["geometry", "perimeter", "rectangles"]
        }
    }
]