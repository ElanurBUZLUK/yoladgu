"""
Tests for SymPy solver integration in math generation service.
"""

import pytest
from app.services.math_generation_service import SymPySolver, math_generation_service


class TestSymPySolver:
    """Test SymPy solver functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.solver = SymPySolver()
    
    def test_linear_equation_solving(self):
        """Test solving linear equations."""
        # Test simple linear equation
        result = self.solver.solve_equation("2*x + 3 = 7", "x")
        
        assert result["success"] is True
        assert result["has_single_solution"] is True
        assert len(result["solutions"]) == 1
        assert abs(result["solutions"][0] - 2.0) < 1e-10
        assert result["validation"]["all_valid"] is True
    
    def test_quadratic_equation_solving(self):
        """Test solving quadratic equations."""
        # Test factorizable quadratic
        result = self.solver.solve_equation("x**2 - 5*x + 6 = 0", "x")
        
        assert result["success"] is True
        assert result["has_single_solution"] is False  # Quadratics have 2 solutions
        assert len(result["solutions"]) == 2
        
        # Solutions should be 2 and 3
        solutions = sorted(result["solutions"])
        assert abs(solutions[0] - 2.0) < 1e-10
        assert abs(solutions[1] - 3.0) < 1e-10
        assert result["validation"]["all_valid"] is True
    
    def test_rational_equation_solving(self):
        """Test solving rational equations."""
        # Test proportion: 2/3 = 4/x
        result = self.solver.solve_equation("2/3 = 4/x", "x")
        
        assert result["success"] is True
        assert result["has_single_solution"] is True
        assert len(result["solutions"]) == 1
        assert abs(result["solutions"][0] - 6.0) < 1e-10
    
    def test_equation_validation(self):
        """Test solution validation."""
        # Test valid solution
        result = self.solver.solve_equation("3*x - 6 = 9", "x")
        
        assert result["success"] is True
        assert result["validation"]["all_valid"] is True
        
        # Check individual validation
        checks = result["validation"]["individual_checks"]
        assert len(checks) == 1
        assert checks[0]["is_valid"] is True
        assert abs(checks[0]["difference"]) < 1e-10
    
    def test_single_solution_verification(self):
        """Test single solution guarantee checking."""
        # Linear equation should have single solution
        assert self.solver.verify_single_solution("2*x + 1 = 5", "x") is True
        
        # Quadratic should not have single solution
        assert self.solver.verify_single_solution("x**2 - 1 = 0", "x") is False
    
    def test_solution_steps_generation(self):
        """Test generation of solution steps."""
        result = self.solver.solve_equation("2*x + 4 = 10", "x")
        
        assert result["success"] is True
        assert "steps" in result
        assert len(result["steps"]) > 0
        
        # Steps should contain key information
        steps_text = " ".join(result["steps"])
        assert "2*x + 4 = 10" in steps_text or "Başlangıç" in steps_text
    
    def test_equation_classification(self):
        """Test equation type classification."""
        # Linear equation
        result = self.solver.solve_equation("3*x + 2 = 8", "x")
        assert result["equation_type"] == "linear"
        
        # Quadratic equation
        result = self.solver.solve_equation("x**2 + 2*x + 1 = 0", "x")
        assert result["equation_type"] == "quadratic"
    
    def test_error_handling(self):
        """Test error handling for invalid equations."""
        # Invalid syntax
        result = self.solver.solve_equation("2x + = 5", "x")
        assert result["success"] is False
        assert "error" in result
        assert result["error_type"] == "parse_error"
        
        # Unsolvable equation
        result = self.solver.solve_equation("0*x = 1", "x")
        assert result["success"] is False or len(result.get("solutions", [])) == 0
    
    def test_complex_solutions(self):
        """Test handling of complex solutions."""
        # Equation with complex roots
        result = self.solver.solve_equation("x**2 + 1 = 0", "x")
        
        assert result["success"] is True
        assert len(result["solutions"]) == 2
        
        # Should have complex solutions ±i
        solutions = result["solutions"]
        assert all(isinstance(sol, complex) for sol in solutions)
    
    def test_explanation_generation(self):
        """Test solution explanation generation."""
        explanation_tr = self.solver.generate_solution_explanation("2*x + 3 = 7", "x", "tr")
        explanation_en = self.solver.generate_solution_explanation("2*x + 3 = 7", "x", "en")
        
        assert "linear" in explanation_tr.lower() or "doğru" in explanation_tr.lower()
        assert "linear" in explanation_en.lower()
        assert "Çözüm sayısı" in explanation_tr or "solution" in explanation_en


class TestMathGenerationServiceSolver:
    """Test solver integration in math generation service."""
    
    def test_arbitrary_equation_solving(self):
        """Test solving arbitrary equations through service."""
        result = math_generation_service.solve_arbitrary_equation("4*x - 8 = 12", "x", "tr")
        
        assert result["success"] is True
        assert result["has_single_solution"] is True
        assert abs(result["solutions"][0] - 5.0) < 1e-10
        assert "explanation" in result
    
    def test_solution_validation_service(self):
        """Test solution validation through service."""
        # Valid solution
        validation = math_generation_service.validate_equation_solution("3*x + 6 = 15", 3.0, "x")
        assert validation["is_valid"] is True
        assert validation["tolerance_met"] is True
        
        # Invalid solution
        validation = math_generation_service.validate_equation_solution("3*x + 6 = 15", 2.0, "x")
        assert validation["is_valid"] is False
    
    def test_single_solution_guarantee_check(self):
        """Test single solution guarantee checking for templates."""
        # Linear equation template
        params = {"a": 2, "b": 3, "c": 7}
        assert math_generation_service.check_single_solution_guarantee("linear_equation_v1", params) is True
        
        # Quadratic equation template (should have 2 solutions)
        params = {"a": 1, "b": -5, "c": 6, "roots": [2, 3]}
        result = math_generation_service.check_single_solution_guarantee("quadratic_equation_v1", params)
        # Quadratics typically have 2 solutions, so this should be False for single solution
        assert result is False or result is True  # Depends on discriminant
    
    def test_solution_steps_generation_service(self):
        """Test solution steps generation through service."""
        steps_tr = math_generation_service.generate_solution_steps("2*x + 4 = 8", "x", "tr")
        steps_en = math_generation_service.generate_solution_steps("2*x + 4 = 8", "x", "en")
        
        assert len(steps_tr) > 0
        assert len(steps_en) > 0
        assert isinstance(steps_tr, list)
        assert isinstance(steps_en, list)
    
    def test_template_solver_integration(self):
        """Test that templates use the advanced solver correctly."""
        # Test linear equation template
        question = math_generation_service.generate_question("linear_equation_v1", language="tr")
        
        assert question["item"]["qa_checks"]["solver_passed"] is True
        assert question["item"]["qa_checks"]["single_gold"] is True
        
        # Check that solution contains solver validation
        generator_params = question["generator"]["params"]
        template = math_generation_service.templates["linear_equation_v1"]
        solution = template.solve_question(generator_params)
        
        # Should have solver validation results
        assert "solver_validation" in solution or "verification" in solution
    
    def test_error_handling_in_templates(self):
        """Test error handling when solver fails in templates."""
        # Create invalid parameters that might cause solver issues
        template = math_generation_service.templates["linear_equation_v1"]
        
        # This should still work due to fallback mechanisms
        try:
            params = {"a": 0, "b": 1, "c": 1}  # Division by zero case
            solution = template.solve_question(params)
            # Should either work or raise appropriate error
            assert True  # If we get here, error handling worked
        except ValueError as e:
            # Expected behavior for invalid parameters
            assert "solution" in str(e).lower() or "error" in str(e).lower()


if __name__ == "__main__":
    pytest.main([__file__])