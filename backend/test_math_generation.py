"""
Test math generation service with solver integration.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from services.math_generation_service import math_generation_service

def test_math_generation_with_solver():
    """Test math generation service with SymPy solver integration."""
    
    print("Testing Linear Equation Generation with Solver...")
    question = math_generation_service.generate_question("linear_equation_v1", language="tr")
    
    print(f"Generated question: {question['item']['stem']}")
    print(f"Answer key: {question['item']['answer_key']}")
    print(f"Solution steps: {question['item']['solution']}")
    print(f"QA checks: {question['item']['qa_checks']}")
    print(f"Generator params: {question['generator']['params']}")
    print()
    
    print("Testing Quadratic Equation Generation...")
    question = math_generation_service.generate_question("quadratic_equation_v1", language="tr")
    
    print(f"Generated question: {question['item']['stem']}")
    print(f"Answer key: {question['item']['answer_key']}")
    print(f"Solution steps: {question['item']['solution']}")
    print(f"QA checks: {question['item']['qa_checks']}")
    print()
    
    print("Testing Ratio Proportion Generation...")
    question = math_generation_service.generate_question("ratio_proportion_v1", language="tr")
    
    print(f"Generated question: {question['item']['stem']}")
    print(f"Answer key: {question['item']['answer_key']}")
    print(f"Solution steps: {question['item']['solution']}")
    print(f"QA checks: {question['item']['qa_checks']}")
    print()
    
    print("Testing Arbitrary Equation Solving...")
    result = math_generation_service.solve_arbitrary_equation("3*x - 9 = 12", "x", "tr")
    print(f"Equation: 3*x - 9 = 12")
    print(f"Success: {result['success']}")
    print(f"Solutions: {result.get('solutions', [])}")
    print(f"Explanation: {result.get('explanation', 'No explanation')}")
    print()
    
    print("Testing Solution Validation...")
    validation = math_generation_service.validate_equation_solution("2*x + 4 = 10", 3.0, "x")
    print(f"Equation: 2*x + 4 = 10, Proposed solution: x = 3")
    print(f"Is valid: {validation['is_valid']}")
    print(f"Left result: {validation.get('left_result', 'N/A')}")
    print(f"Right result: {validation.get('right_result', 'N/A')}")
    print()
    
    print("Testing Single Solution Guarantee...")
    params = {"a": 2, "b": 3, "c": 7}
    has_single = math_generation_service.check_single_solution_guarantee("linear_equation_v1", params)
    print(f"Linear equation with params {params} has single solution: {has_single}")
    
    params = {"a": 1, "b": -5, "c": 6, "roots": [2, 3]}
    has_single = math_generation_service.check_single_solution_guarantee("quadratic_equation_v1", params)
    print(f"Quadratic equation with params {params} has single solution: {has_single}")
    print()
    
    print("Testing Solution Steps Generation...")
    steps = math_generation_service.generate_solution_steps("4*x - 8 = 16", "x", "tr")
    print(f"Equation: 4*x - 8 = 16")
    print("Solution steps:")
    for i, step in enumerate(steps, 1):
        print(f"  {i}. {step}")
    print()
    
    print("All math generation tests completed successfully!")

if __name__ == "__main__":
    test_math_generation_with_solver()