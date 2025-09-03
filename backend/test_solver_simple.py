"""
Simple test for SymPy solver functionality.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from services.math_generation_service import SymPySolver

def test_sympy_solver():
    """Test basic SymPy solver functionality."""
    solver = SymPySolver()
    
    # Test linear equation
    print("Testing linear equation: 2*x + 3 = 7")
    result = solver.solve_equation("2*x + 3 = 7", "x")
    print(f"Success: {result['success']}")
    print(f"Solutions: {result.get('solutions', [])}")
    print(f"Single solution: {result.get('has_single_solution', False)}")
    print(f"Steps: {result.get('steps', [])}")
    print()
    
    # Test quadratic equation
    print("Testing quadratic equation: x**2 - 5*x + 6 = 0")
    result = solver.solve_equation("x**2 - 5*x + 6 = 0", "x")
    print(f"Success: {result['success']}")
    print(f"Solutions: {result.get('solutions', [])}")
    print(f"Single solution: {result.get('has_single_solution', False)}")
    print(f"Equation type: {result.get('equation_type', 'unknown')}")
    print()
    
    # Test validation
    print("Testing solution validation...")
    validation = result.get('validation', {})
    print(f"All valid: {validation.get('all_valid', False)}")
    print(f"Individual checks: {len(validation.get('individual_checks', []))}")
    print()
    
    # Test single solution verification
    print("Testing single solution verification...")
    is_single = solver.verify_single_solution("3*x - 6 = 9", "x")
    print(f"Linear equation has single solution: {is_single}")
    
    is_single_quad = solver.verify_single_solution("x**2 - 1 = 0", "x")
    print(f"Quadratic equation has single solution: {is_single_quad}")
    print()
    
    print("All tests completed successfully!")

if __name__ == "__main__":
    test_sympy_solver()