"""
Sample mathematics questions for seeding the database
"""

from app.models.question import Subject, QuestionType, SourceType

SAMPLE_MATH_QUESTIONS = [
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