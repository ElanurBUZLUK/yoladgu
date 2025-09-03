"""
Test distractor generation functionality in math generation service.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from services.math_generation_service import math_generation_service, distractor_generator

def test_distractor_generation():
    """Test misconception-based distractor generation."""
    
    print("=== Testing Linear Equation Distractors ===")
    
    # Test linear equation distractors
    params = {"a": 2, "b": 3, "c": 7}
    correct_answer = 2  # Solution to 2x + 3 = 7
    
    distractors = math_generation_service.generate_distractors_for_question(
        template_id="linear_equation_v1",
        params=params,
        correct_answer=correct_answer,
        num_distractors=3,
        analysis=True
    )
    
    print(f"Linear equation: 2x + 3 = 7")
    print(f"Correct answer: x = {correct_answer}")
    print(f"Generated distractors: {distractors['distractors']}")
    print(f"Quality analysis: {distractors['quality_analysis']}")
    print()
    
    # Test individual misconceptions
    print("=== Testing Individual Misconceptions ===")
    
    misconceptions = math_generation_service.get_misconception_patterns("linear_equation_v1")
    print(f"Available misconceptions for linear equations: {misconceptions}")
    
    for misconception in misconceptions[:3]:  # Test first 3
        result = math_generation_service.test_misconception_pattern(
            template_id="linear_equation_v1",
            misconception_name=misconception,
            params=params,
            correct_answer=correct_answer
        )
        print(f"Misconception '{misconception}': {result}")
    print()
    
    print("=== Testing Quadratic Equation Distractors ===")
    
    # Test quadratic equation distractors
    quad_params = {"a": 1, "b": -5, "c": 6, "roots": [2, 3]}
    correct_roots = [2, 3]
    
    quad_distractors = math_generation_service.generate_distractors_for_question(
        template_id="quadratic_equation_v1",
        params=quad_params,
        correct_answer=correct_roots,
        num_distractors=3,
        analysis=True
    )
    
    print(f"Quadratic equation: x² - 5x + 6 = 0")
    print(f"Correct roots: {correct_roots}")
    print(f"Generated distractors: {quad_distractors['distractors']}")
    print(f"Quality analysis: {quad_distractors['quality_analysis']}")
    print()
    
    print("=== Testing Ratio Proportion Distractors ===")
    
    # Test ratio proportion distractors
    ratio_params = {"a": 3, "b": 4, "c": 9, "x": 12}
    correct_x = 12  # Solution to 3/4 = 9/x
    
    ratio_distractors = math_generation_service.generate_distractors_for_question(
        template_id="ratio_proportion_v1",
        params=ratio_params,
        correct_answer=correct_x,
        num_distractors=3,
        analysis=True
    )
    
    print(f"Ratio proportion: 3/4 = 9/x")
    print(f"Correct answer: x = {correct_x}")
    print(f"Generated distractors: {ratio_distractors['distractors']}")
    print(f"Quality analysis: {ratio_distractors['quality_analysis']}")
    print()
    
    print("=== Testing Distractor Quality Validation ===")
    
    # Test quality validation
    test_distractors = [1, 4, 7]  # Some test distractors
    quality = math_generation_service.validate_distractor_quality(
        template_id="linear_equation_v1",
        distractors=test_distractors,
        correct_answer=correct_answer
    )
    
    print(f"Test distractors: {test_distractors}")
    print(f"Quality validation: {quality}")
    print()
    
    print("=== Testing Full Question Generation with Distractors ===")
    
    # Test full question generation
    question = math_generation_service.generate_question("linear_equation_v1", language="tr")
    
    print(f"Generated question: {question['item']['stem']}")
    print(f"Choices: {question['item']['choices']}")
    print(f"Answer key: {question['item']['answer_key']}")
    print(f"QA checks: {question['item']['qa_checks']}")
    print()
    
    # Test quadratic with multiple choice
    quad_question = math_generation_service.generate_question("quadratic_equation_v1", language="tr")
    
    print(f"Quadratic question: {quad_question['item']['stem']}")
    if quad_question['item'].get('choices'):
        print(f"Choices: {quad_question['item']['choices']}")
    print(f"Answer key: {quad_question['item']['answer_key']}")
    print(f"Type: {quad_question['item'].get('type', 'unknown')}")
    print()
    
    print("All distractor generation tests completed successfully!")

def test_misconception_patterns():
    """Test specific misconception patterns in detail."""
    
    print("=== Detailed Misconception Pattern Testing ===")
    
    # Test linear equation misconceptions
    params = {"a": 3, "b": -6, "c": 9}  # 3x - 6 = 9, solution x = 5
    correct = 5
    
    print(f"Testing equation: 3x - 6 = 9 (correct answer: x = {correct})")
    
    from services.math_generation_service import LinearEquationMisconceptions
    
    # Test each misconception
    misconceptions = [
        ("Forget to isolate", LinearEquationMisconceptions.forget_to_isolate_variable),
        ("Wrong sign", LinearEquationMisconceptions.wrong_sign_operation),
        ("Wrong coefficient", LinearEquationMisconceptions.divide_by_wrong_coefficient),
        ("Forget negative", LinearEquationMisconceptions.forget_negative_sign),
        ("Arithmetic error", LinearEquationMisconceptions.arithmetic_error)
    ]
    
    for name, func in misconceptions:
        try:
            distractor = func(params, correct)
            print(f"  {name}: {distractor}")
        except Exception as e:
            print(f"  {name}: Error - {e}")
    
    print()
    
    # Test quadratic misconceptions
    quad_params = {"a": 1, "b": -3, "c": 2}  # x² - 3x + 2 = 0, roots [1, 2]
    correct_roots = [1, 2]
    
    print(f"Testing quadratic: x² - 3x + 2 = 0 (correct roots: {correct_roots})")
    
    from services.math_generation_service import QuadraticMisconceptions
    
    quad_misconceptions = [
        ("Wrong factoring", QuadraticMisconceptions.wrong_factoring),
        ("Discriminant error", QuadraticMisconceptions.discriminant_error),
        ("Formula sign error", QuadraticMisconceptions.quadratic_formula_sign_error)
    ]
    
    for name, func in quad_misconceptions:
        try:
            distractor = func(quad_params, correct_roots)
            print(f"  {name}: {distractor}")
        except Exception as e:
            print(f"  {name}: Error - {e}")
    
    print()

if __name__ == "__main__":
    test_distractor_generation()
    print("\n" + "="*50 + "\n")
    test_misconception_patterns()