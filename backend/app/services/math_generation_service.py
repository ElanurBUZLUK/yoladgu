"""
Math question generation service with parametric templates and solver validation.
"""

import random
import sympy as sp
from typing import Dict, Any, List, Optional, Tuple
from sympy import symbols, Eq, solve, simplify, expand, factor, latex, sympify
from sympy.parsing.sympy_parser import parse_expr
from sympy.solvers import solve as sympy_solve
from sympy.core.sympify import SympifyError
import json
import logging

from app.core.config import settings

logger = logging.getLogger(__name__)


class MathMisconception:
    """Represents a mathematical misconception for distractor generation."""
    
    def __init__(self, name: str, description: str, error_type: str, frequency: float = 1.0):
        self.name = name
        self.description = description
        self.error_type = error_type  # "algebraic", "arithmetic", "conceptual"
        self.frequency = frequency  # How common this misconception is (0.0-1.0)
    
    def apply(self, params: Dict[str, Any], correct_answer: Any) -> Optional[Any]:
        """Apply misconception to generate distractor. Override in subclasses."""
        raise NotImplementedError


class LinearEquationMisconceptions:
    """Common misconceptions for linear equations."""
    
    @staticmethod
    def forget_to_isolate_variable(params: Dict[str, Any], correct_answer: int) -> Optional[int]:
        """Student forgets to subtract constant term."""
        a, b, c = params["a"], params["b"], params["c"]
        if a != 0:
            return c // a
        return None
    
    @staticmethod
    def wrong_sign_operation(params: Dict[str, Any], correct_answer: int) -> Optional[int]:
        """Student uses wrong sign when moving terms."""
        a, b, c = params["a"], params["b"], params["c"]
        if a != 0:
            return (c + b) // a
        return None
    
    @staticmethod
    def divide_by_wrong_coefficient(params: Dict[str, Any], correct_answer: int) -> Optional[int]:
        """Student divides by constant instead of coefficient."""
        a, b, c = params["a"], params["b"], params["c"]
        if b != 0:
            return (c - b) // b
        return None
    
    @staticmethod
    def forget_negative_sign(params: Dict[str, Any], correct_answer: int) -> Optional[int]:
        """Student forgets negative sign in final answer."""
        return abs(correct_answer) if correct_answer < 0 else -abs(correct_answer)
    
    @staticmethod
    def arithmetic_error(params: Dict[str, Any], correct_answer: int) -> Optional[int]:
        """Student makes basic arithmetic mistake."""
        # Common arithmetic errors: off by 1, wrong multiplication
        errors = [correct_answer + 1, correct_answer - 1, correct_answer * 2, correct_answer // 2]
        return random.choice([e for e in errors if e != correct_answer])


class QuadraticMisconceptions:
    """Common misconceptions for quadratic equations."""
    
    @staticmethod
    def wrong_factoring(params: Dict[str, Any], correct_roots: List) -> List:
        """Student factors incorrectly."""
        if len(correct_roots) == 2:
            r1, r2 = correct_roots
            # Common mistake: wrong signs in factoring
            return [-r1, r2] if random.choice([True, False]) else [r1, -r2]
        return correct_roots
    
    @staticmethod
    def discriminant_error(params: Dict[str, Any], correct_roots: List) -> List:
        """Student makes error in discriminant calculation."""
        a, b, c = params["a"], params["b"], params["c"]
        # Wrong discriminant: b² + 4ac instead of b² - 4ac
        wrong_discriminant = b**2 + 4*a*c
        if wrong_discriminant >= 0:
            import math
            sqrt_d = math.sqrt(wrong_discriminant)
            r1 = (-b + sqrt_d) / (2*a)
            r2 = (-b - sqrt_d) / (2*a)
            return [r1, r2]
        return correct_roots
    
    @staticmethod
    def quadratic_formula_sign_error(params: Dict[str, Any], correct_roots: List) -> List:
        """Student uses wrong sign in quadratic formula."""
        if len(correct_roots) == 2:
            r1, r2 = correct_roots
            # Swap signs
            return [-r1, -r2]
        return correct_roots


class RatioMisconceptions:
    """Common misconceptions for ratio and proportion problems."""
    
    @staticmethod
    def cross_multiply_error(params: Dict[str, Any], correct_answer: int) -> Optional[int]:
        """Student cross multiplies incorrectly."""
        a, b, c = params["a"], params["b"], params["c"]
        # Wrong cross multiplication: a*c = b*x instead of a*x = b*c
        if c != 0:
            return (a * c) // b if b != 0 else None
        return None
    
    @staticmethod
    def proportion_inversion(params: Dict[str, Any], correct_answer: int) -> Optional[int]:
        """Student inverts the proportion."""
        a, b, c = params["a"], params["b"], params["c"]
        # Solve c/a = b/x instead of a/b = c/x
        if a != 0:
            return (c * b) // a
        return None
    
    @staticmethod
    def additive_thinking(params: Dict[str, Any], correct_answer: int) -> Optional[int]:
        """Student uses additive instead of multiplicative reasoning."""
        a, b, c = params["a"], params["b"], params["c"]
        # Think: if a increases by (c-a), then b increases by same amount
        difference = c - a
        return b + difference


class DistractorGenerator:
    """Advanced distractor generator using misconception patterns."""
    
    def __init__(self):
        self.misconception_registry = {
            "linear_equation_v1": [
                LinearEquationMisconceptions.forget_to_isolate_variable,
                LinearEquationMisconceptions.wrong_sign_operation,
                LinearEquationMisconceptions.divide_by_wrong_coefficient,
                LinearEquationMisconceptions.forget_negative_sign,
                LinearEquationMisconceptions.arithmetic_error
            ],
            "quadratic_equation_v1": [
                QuadraticMisconceptions.wrong_factoring,
                QuadraticMisconceptions.discriminant_error,
                QuadraticMisconceptions.quadratic_formula_sign_error
            ],
            "ratio_proportion_v1": [
                RatioMisconceptions.cross_multiply_error,
                RatioMisconceptions.proportion_inversion,
                RatioMisconceptions.additive_thinking
            ]
        }
    
    def generate_distractors(
        self,
        template_id: str,
        params: Dict[str, Any],
        correct_answer: Any,
        num_distractors: int = 3,
        ensure_uniqueness: bool = True,
        plausibility_check: bool = True
    ) -> List[Any]:
        """
        Generate distractors using misconception patterns.
        
        Args:
            template_id: Template identifier
            params: Template parameters
            correct_answer: Correct answer to avoid
            num_distractors: Number of distractors to generate
            ensure_uniqueness: Ensure all distractors are unique
            plausibility_check: Check if distractors are plausible
            
        Returns:
            List of distractor values
        """
        distractors = []
        
        if template_id not in self.misconception_registry:
            return self._generate_random_distractors(correct_answer, num_distractors)
        
        misconceptions = self.misconception_registry[template_id]
        
        # Apply misconceptions to generate distractors
        for misconception_func in misconceptions:
            try:
                distractor = misconception_func(params, correct_answer)
                
                if distractor is not None:
                    # Handle different answer types
                    if isinstance(correct_answer, list):  # For quadratics
                        if isinstance(distractor, list) and distractor != correct_answer:
                            distractors.extend(distractor)
                    else:  # For single answers
                        if distractor != correct_answer:
                            distractors.append(distractor)
                
                if len(distractors) >= num_distractors:
                    break
                    
            except Exception as e:
                logger.warning(f"Misconception function failed: {e}")
                continue
        
        # Remove duplicates if requested
        if ensure_uniqueness:
            distractors = self._ensure_uniqueness(distractors, correct_answer)
        
        # Check plausibility
        if plausibility_check:
            distractors = self._check_plausibility(distractors, correct_answer, template_id)
        
        # Fill remaining slots with random distractors if needed
        while len(distractors) < num_distractors:
            random_distractor = self._generate_single_random_distractor(correct_answer)
            if random_distractor not in distractors and random_distractor != correct_answer:
                distractors.append(random_distractor)
        
        return distractors[:num_distractors]
    
    def _ensure_uniqueness(self, distractors: List[Any], correct_answer: Any) -> List[Any]:
        """Remove duplicate distractors and correct answer."""
        unique_distractors = []
        seen = set()
        
        for distractor in distractors:
            # Convert to hashable type for set operations
            key = str(distractor) if not isinstance(distractor, (int, float, str)) else distractor
            
            if key not in seen and distractor != correct_answer:
                unique_distractors.append(distractor)
                seen.add(key)
        
        return unique_distractors
    
    def _check_plausibility(self, distractors: List[Any], correct_answer: Any, template_id: str) -> List[Any]:
        """Check if distractors are plausible (not too far from correct answer)."""
        plausible_distractors = []
        
        for distractor in distractors:
            if self._is_plausible(distractor, correct_answer, template_id):
                plausible_distractors.append(distractor)
        
        return plausible_distractors
    
    def _is_plausible(self, distractor: Any, correct_answer: Any, template_id: str) -> bool:
        """Check if a single distractor is plausible."""
        try:
            if isinstance(correct_answer, (int, float)) and isinstance(distractor, (int, float)):
                # For numerical answers, check if distractor is within reasonable range
                if correct_answer == 0:
                    return abs(distractor) <= 20  # Reasonable range around 0
                else:
                    ratio = abs(distractor / correct_answer) if correct_answer != 0 else float('inf')
                    return 0.1 <= ratio <= 10  # Within order of magnitude
            
            elif isinstance(correct_answer, list) and isinstance(distractor, list):
                # For multiple answers (like quadratic roots), check individual elements
                return all(self._is_plausible(d, c, template_id) 
                          for d, c in zip(distractor, correct_answer))
            
            return True  # Default to plausible for other types
            
        except (ZeroDivisionError, TypeError):
            return False
    
    def _generate_random_distractors(self, correct_answer: Any, num_distractors: int) -> List[Any]:
        """Generate random distractors as fallback."""
        distractors = []
        
        for _ in range(num_distractors):
            distractor = self._generate_single_random_distractor(correct_answer)
            if distractor != correct_answer and distractor not in distractors:
                distractors.append(distractor)
        
        return distractors
    
    def _generate_single_random_distractor(self, correct_answer: Any) -> Any:
        """Generate a single random distractor."""
        if isinstance(correct_answer, int):
            # Generate random integer near correct answer
            offset = random.randint(-5, 5)
            return correct_answer + offset if offset != 0 else correct_answer + 1
        
        elif isinstance(correct_answer, float):
            # Generate random float near correct answer
            offset = random.uniform(-2.0, 2.0)
            return correct_answer + offset if abs(offset) > 0.1 else correct_answer + 1.0
        
        elif isinstance(correct_answer, list):
            # For lists (like quadratic roots), modify each element
            return [self._generate_single_random_distractor(item) for item in correct_answer]
        
        else:
            return str(correct_answer) + "_alt"  # Fallback for other types
    
    def analyze_distractor_quality(
        self,
        distractors: List[Any],
        correct_answer: Any,
        template_id: str
    ) -> Dict[str, Any]:
        """Analyze the quality of generated distractors."""
        analysis = {
            "total_distractors": len(distractors),
            "uniqueness_score": len(set(str(d) for d in distractors)) / len(distractors) if distractors else 0,
            "plausibility_scores": [],
            "difficulty_distribution": {},
            "misconception_coverage": {}
        }
        
        # Analyze each distractor
        for i, distractor in enumerate(distractors):
            plausible = self._is_plausible(distractor, correct_answer, template_id)
            analysis["plausibility_scores"].append(plausible)
        
        # Calculate overall plausibility
        analysis["overall_plausibility"] = (
            sum(analysis["plausibility_scores"]) / len(analysis["plausibility_scores"])
            if analysis["plausibility_scores"] else 0
        )
        
        return analysis


# Create global distractor generator instance
distractor_generator = DistractorGenerator()


class MathTemplate:
    """Base class for math question templates."""
    
    def __init__(self, template_id: str, name: str, skills: List[str], bloom_level: str = "apply"):
        self.template_id = template_id
        self.name = name
        self.skills = skills
        self.bloom_level = bloom_level
        self.difficulty_range = (0.0, 1.0)  # Default difficulty range
    
    def generate_parameters(self, constraints: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate random parameters for the template."""
        raise NotImplementedError
    
    def create_question(self, params: Dict[str, Any], lang: str = "tr") -> Dict[str, Any]:
        """Create question from parameters."""
        raise NotImplementedError
    
    def solve_question(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Solve the question programmatically."""
        raise NotImplementedError
    
    def validate_solution(self, params: Dict[str, Any], solution: Dict[str, Any]) -> bool:
        """Validate that the solution is correct and unique."""
        raise NotImplementedError


class LinearEquationTemplate(MathTemplate):
    """Template for linear equations: ax + b = c"""
    
    def __init__(self):
        super().__init__(
            template_id="linear_equation_v1",
            name="Linear Equation",
            skills=["linear_equation", "algebra"],
            bloom_level="apply"
        )
        self.difficulty_range = (-0.5, 0.5)
    
    def generate_parameters(self, constraints: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate parameters for ax + b = c"""
        constraints = constraints or {}
        
        # Default ranges
        a_range = constraints.get("a", {"min": 1, "max": 9, "exclude": [0]})
        b_range = constraints.get("b", {"min": -10, "max": 10})
        c_range = constraints.get("c", {"min": -10, "max": 10})
        
        # Generate a (coefficient of x)
        a_min, a_max = a_range["min"], a_range["max"]
        a_exclude = a_range.get("exclude", [0])
        
        while True:
            a = random.randint(a_min, a_max)
            if a not in a_exclude:
                break
        
        # Generate b (constant term)
        b = random.randint(b_range["min"], b_range["max"])
        
        # Generate c (right side)
        c = random.randint(c_range["min"], c_range["max"])
        
        # Ensure solution is reasonable (integer)
        x_solution = (c - b) / a
        if not x_solution.is_integer() or abs(x_solution) > 20:
            # Regenerate c to get integer solution
            x_target = random.randint(-10, 10)
            c = a * x_target + b
        
        return {"a": a, "b": b, "c": c}
    
    def create_question(self, params: Dict[str, Any], lang: str = "tr") -> Dict[str, Any]:
        """Create linear equation question."""
        a, b, c = params["a"], params["b"], params["c"]
        
        # Create equation string
        if b >= 0:
            equation = f"{a}x + {b} = {c}"
        else:
            equation = f"{a}x - {abs(b)} = {c}"
        
        # Question text based on language
        if lang == "tr":
            stem = f"Aşağıdaki denklemi çözünüz: {equation}"
            choices_prefix = "x = "
        else:
            stem = f"Solve the following equation: {equation}"
            choices_prefix = "x = "
        
        # Generate solution and distractors
        solution_data = self.solve_question(params)
        correct_answer = solution_data["x"]
        
        # Generate distractors (common mistakes)
        distractors = self._generate_distractors(params, correct_answer)
        
        # Create choices
        choices = [f"{choices_prefix}{correct_answer}"] + [f"{choices_prefix}{d}" for d in distractors]
        random.shuffle(choices)
        
        # Find correct choice index
        correct_choice = f"{choices_prefix}{correct_answer}"
        answer_key = correct_choice
        
        return {
            "stem": stem,
            "choices": choices,
            "answer_key": answer_key,
            "solution": solution_data["steps"],
            "equation": equation
        }
    
    def solve_question(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Solve ax + b = c programmatically using SymPy solver."""
        a, b, c = params["a"], params["b"], params["c"]
        
        # Create equation string for SymPy solver
        if b >= 0:
            equation_str = f"{a}*x + {b} = {c}"
        else:
            equation_str = f"{a}*x - {abs(b)} = {c}"
        
        # Use advanced SymPy solver
        from app.services.math_generation_service import SymPySolver
        solver = SymPySolver()
        result = solver.solve_equation(equation_str, 'x')
        
        if not result.get("success", False):
            raise ValueError(f"Solver failed: {result.get('error', 'Unknown error')}")
        
        if not result.get("has_single_solution", False):
            raise ValueError("Equation does not have exactly one solution")
        
        solutions = result.get("solutions", [])
        if not solutions:
            raise ValueError("No solution found")
        
        x_value = solutions[0]
        
        # Ensure integer solution if possible
        if isinstance(x_value, complex):
            if abs(x_value.imag) < 1e-10:  # Essentially real
                x_value = x_value.real
        
        if isinstance(x_value, (int, float)) and abs(x_value - round(x_value)) < 1e-10:
            x_value = int(round(x_value))
        
        # Generate detailed steps using solver
        solver_steps = result.get("steps", [])
        
        # Fallback to manual steps if solver steps are not available
        if not solver_steps:
            steps = [
                f"{a}x + {b} = {c}",
                f"{a}x = {c} - {b}",
                f"{a}x = {c - b}",
                f"x = {c - b}/{a}",
                f"x = {x_value}"
            ]
        else:
            steps = solver_steps
        
        return {
            "x": x_value,
            "steps": " → ".join(steps) if isinstance(steps, list) else str(steps),
            "verification": f"Doğrulama: {a}×{x_value} + {b} = {a*x_value + b} = {c}",
            "solver_validation": result.get("validation", {}),
            "equation_type": result.get("equation_type", "linear")
        }
    
    def validate_solution(self, params: Dict[str, Any], solution: Dict[str, Any]) -> bool:
        """Validate solution is correct and unique."""
        try:
            a, b, c = params["a"], params["b"], params["c"]
            x_value = solution["x"]
            
            # Check if solution satisfies equation
            result = a * x_value + b
            return abs(result - c) < 1e-10
        except:
            return False
    
    def _generate_distractors(self, params: Dict[str, Any], correct_answer: int) -> List[int]:
        """Generate misconception-based distractors using advanced generator."""
        return distractor_generator.generate_distractors(
            template_id=self.template_id,
            params=params,
            correct_answer=correct_answer,
            num_distractors=3,
            ensure_uniqueness=True,
            plausibility_check=True
        )


class QuadraticEquationTemplate(MathTemplate):
    """Template for quadratic equations: ax² + bx + c = 0"""
    
    def __init__(self):
        super().__init__(
            template_id="quadratic_equation_v1",
            name="Quadratic Equation",
            skills=["quadratic_equation", "algebra", "factoring"],
            bloom_level="analyze"
        )
        self.difficulty_range = (0.2, 0.8)
    
    def generate_parameters(self, constraints: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate parameters for ax² + bx + c = 0"""
        constraints = constraints or {}
        
        # For simplicity, generate factorable quadratics
        # (x - r1)(x - r2) = x² - (r1+r2)x + r1*r2
        
        r1 = random.randint(-5, 5)
        r2 = random.randint(-5, 5)
        
        # Ensure roots are different for interesting problem
        while r1 == r2:
            r2 = random.randint(-5, 5)
        
        # Calculate coefficients
        a = 1  # Keep it simple
        b = -(r1 + r2)
        c = r1 * r2
        
        return {"a": a, "b": b, "c": c, "roots": [r1, r2]}
    
    def create_question(self, params: Dict[str, Any], lang: str = "tr") -> Dict[str, Any]:
        """Create quadratic equation question."""
        a, b, c = params["a"], params["b"], params["c"]
        
        # Create equation string
        equation_parts = []
        if a != 1:
            equation_parts.append(f"{a}x²")
        else:
            equation_parts.append("x²")
        
        if b > 0:
            equation_parts.append(f"+ {b}x")
        elif b < 0:
            equation_parts.append(f"- {abs(b)}x")
        elif b == 0:
            pass  # Don't add zero term
        
        if c > 0:
            equation_parts.append(f"+ {c}")
        elif c < 0:
            equation_parts.append(f"- {abs(c)}")
        elif c == 0:
            pass  # Don't add zero term
        
        equation = " ".join(equation_parts) + " = 0"
        
        # Question text
        if lang == "tr":
            stem = f"Aşağıdaki ikinci dereceden denklemi çözünüz: {equation}"
        else:
            stem = f"Solve the quadratic equation: {equation}"
        
        # Get solution
        solution_data = self.solve_question(params)
        roots = solution_data["roots"]
        
        # Generate distractors for multiple choice
        distractors = self._generate_distractors(params, roots)
        
        # Format choices
        choices = []
        if lang == "tr":
            correct_choice = f"x₁ = {roots[0]}, x₂ = {roots[1]}"
            choices.append(correct_choice)
            
            for distractor in distractors:
                if isinstance(distractor, list) and len(distractor) == 2:
                    choice = f"x₁ = {distractor[0]}, x₂ = {distractor[1]}"
                    choices.append(choice)
        else:
            correct_choice = f"x₁ = {roots[0]}, x₂ = {roots[1]}"
            choices.append(correct_choice)
            
            for distractor in distractors:
                if isinstance(distractor, list) and len(distractor) == 2:
                    choice = f"x₁ = {distractor[0]}, x₂ = {distractor[1]}"
                    choices.append(choice)
        
        # Shuffle choices
        random.shuffle(choices)
        answer_key = correct_choice
        
        result = {
            "stem": stem,
            "answer_key": answer_key,
            "solution": solution_data["steps"],
            "equation": equation
        }
        
        # Add choices and type if we have multiple choice
        if len(choices) > 1:
            result["choices"] = choices
            result["type"] = "multiple_choice"
        else:
            result["type"] = "open_ended"
        
        return result
    
    def _generate_distractors(self, params: Dict[str, Any], correct_roots: List) -> List[List]:
        """Generate misconception-based distractors for quadratic equations."""
        return distractor_generator.generate_distractors(
            template_id=self.template_id,
            params=params,
            correct_answer=correct_roots,
            num_distractors=3,
            ensure_uniqueness=True,
            plausibility_check=True
        )
    
    def solve_question(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Solve quadratic equation using advanced SymPy solver."""
        a, b, c = params["a"], params["b"], params["c"]
        
        # Create equation string
        equation_str = f"{a}*x**2 + {b}*x + {c} = 0"
        
        # Use advanced SymPy solver
        from app.services.math_generation_service import SymPySolver
        solver = SymPySolver()
        result = solver.solve_equation(equation_str, 'x')
        
        if not result.get("success", False):
            raise ValueError(f"Solver failed: {result.get('error', 'Unknown error')}")
        
        solutions = result.get("solutions", [])
        if len(solutions) != 2:
            raise ValueError("Quadratic equation should have exactly 2 solutions")
        
        # Sort solutions and convert to appropriate type
        roots = []
        for sol in solutions:
            if isinstance(sol, complex):
                if abs(sol.imag) < 1e-10:  # Essentially real
                    val = sol.real
                    roots.append(int(val) if abs(val - round(val)) < 1e-10 else val)
                else:
                    roots.append(sol)  # Keep as complex
            else:
                roots.append(int(sol) if abs(sol - round(sol)) < 1e-10 else sol)
        
        roots.sort(key=lambda x: (x.real if isinstance(x, complex) else x))
        
        # Generate factored form using SymPy
        x = symbols('x')
        poly = a*x**2 + b*x + c
        factored_expr = factor(poly)
        
        if factored_expr != poly:
            factored = f"{factored_expr} = 0"
        else:
            # Manual factoring for display
            if len(roots) == 2 and all(isinstance(r, (int, float)) for r in roots):
                r1, r2 = roots
                if r1 >= 0 and r2 >= 0:
                    factored = f"(x - {r1})(x - {r2}) = 0"
                elif r1 < 0 and r2 < 0:
                    factored = f"(x + {abs(r1)})(x + {abs(r2)}) = 0"
                else:
                    factored = f"(x - {r1})(x - {r2}) = 0"
            else:
                factored = "Karmaşık kökler - faktörleme zor"
        
        # Use solver steps or generate manual steps
        solver_steps = result.get("steps", [])
        if solver_steps:
            steps = solver_steps
        else:
            steps = [
                f"x² + {b}x + {c} = 0",
                f"Faktör: {factored}",
                f"Kökler: x₁ = {roots[0]}, x₂ = {roots[1]}"
            ]
        
        return {
            "roots": roots,
            "steps": " → ".join(steps) if isinstance(steps, list) else str(steps),
            "factored_form": factored,
            "solver_validation": result.get("validation", {}),
            "equation_type": result.get("equation_type", "quadratic"),
            "discriminant": b**2 - 4*a*c
        }
    
    def validate_solution(self, params: Dict[str, Any], solution: Dict[str, Any]) -> bool:
        """Validate quadratic solution."""
        try:
            a, b, c = params["a"], params["b"], params["c"]
            roots = solution["roots"]
            
            # Check each root
            for root in roots:
                result = a * root**2 + b * root + c
                if abs(result) > 1e-10:
                    return False
            return True
        except:
            return False


class RatioProportionTemplate(MathTemplate):
    """Template for ratio and proportion problems."""
    
    def __init__(self):
        super().__init__(
            template_id="ratio_proportion_v1",
            name="Ratio and Proportion",
            skills=["ratio", "proportion", "cross_multiplication"],
            bloom_level="apply"
        )
        self.difficulty_range = (-0.2, 0.4)
    
    def generate_parameters(self, constraints: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate ratio proportion parameters."""
        # Create a/b = c/x type problem
        
        # Choose base ratio
        a = random.randint(2, 12)
        b = random.randint(2, 12)
        
        # Ensure a and b are coprime for cleaner ratios
        from math import gcd
        while gcd(a, b) > 1:
            b = random.randint(2, 12)
        
        # Choose multiplier for second ratio
        multiplier = random.randint(2, 8)
        c = a * multiplier
        x = b * multiplier  # This is what we solve for
        
        return {"a": a, "b": b, "c": c, "x": x}
    
    def create_question(self, params: Dict[str, Any], lang: str = "tr") -> Dict[str, Any]:
        """Create ratio proportion question."""
        a, b, c = params["a"], params["b"], params["c"]
        
        if lang == "tr":
            stem = f"Aşağıdaki orantıyı çözünüz: {a}/{b} = {c}/x"
        else:
            stem = f"Solve the proportion: {a}/{b} = {c}/x"
        
        solution_data = self.solve_question(params)
        correct_answer = solution_data["x"]
        
        # Generate misconception-based distractors
        distractors = self._generate_distractors(params, correct_answer)
        
        choices = [f"x = {correct_answer}"] + [f"x = {d}" for d in distractors[:3]]
        random.shuffle(choices)
        
        return {
            "stem": stem,
            "choices": choices,
            "answer_key": f"x = {correct_answer}",
            "solution": solution_data["steps"]
        }
    
    def solve_question(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Solve proportion using cross multiplication and SymPy validation."""
        a, b, c = params["a"], params["b"], params["c"]
        
        # Create equation string for SymPy solver
        equation_str = f"{a}/({b}) = {c}/x"
        
        # Use advanced SymPy solver for validation
        from app.services.math_generation_service import SymPySolver
        solver = SymPySolver()
        
        # Convert to standard equation form: a*x = b*c
        standard_eq = f"{a}*x = {b*c}"
        result = solver.solve_equation(standard_eq, 'x')
        
        if not result.get("success", False):
            # Fallback to manual calculation
            x = (b * c) // a
        else:
            solutions = result.get("solutions", [])
            if solutions:
                x_val = solutions[0]
                # Handle complex numbers
                if isinstance(x_val, complex):
                    if abs(x_val.imag) < 1e-10:  # Essentially real
                        x_val = x_val.real
                    else:
                        x = (b * c) // a  # Fallback for complex
                        
                if isinstance(x_val, (int, float)):
                    x = int(x_val) if abs(x_val - round(x_val)) < 1e-10 else x_val
                else:
                    x = (b * c) // a  # Fallback
            else:
                x = (b * c) // a
        
        # Generate detailed steps
        steps = [
            f"{a}/{b} = {c}/x",
            f"Çapraz çarpım: {a} × x = {b} × {c}",
            f"{a}x = {b * c}",
            f"x = {b * c}/{a}",
            f"x = {x}"
        ]
        
        # Add verification
        verification = f"Doğrulama: {a}/{b} = {float(a)/float(b):.3f}, {c}/{x} = {float(c)/float(x):.3f}"
        
        return {
            "x": x,
            "steps": " → ".join(steps),
            "verification": verification,
            "solver_validation": result.get("validation", {}) if result.get("success") else None,
            "cross_multiplication": f"{a} × {x} = {b} × {c} = {b*c}"
        }
    
    def validate_solution(self, params: Dict[str, Any], solution: Dict[str, Any]) -> bool:
        """Validate proportion solution."""
        try:
            a, b, c = params["a"], params["b"], params["c"]
            x = solution["x"]
            
            # Check if a/b = c/x
            return abs(a/b - c/x) < 1e-10
        except:
            return False
    
    def _generate_distractors(self, params: Dict[str, Any], correct_answer: int) -> List[int]:
        """Generate misconception-based distractors for ratio problems."""
        return distractor_generator.generate_distractors(
            template_id=self.template_id,
            params=params,
            correct_answer=correct_answer,
            num_distractors=3,
            ensure_uniqueness=True,
            plausibility_check=True
        )


class SymPySolver:
    """Advanced SymPy-based equation solver with validation and step generation."""
    
    def __init__(self):
        self.supported_functions = {
            'sin', 'cos', 'tan', 'log', 'ln', 'sqrt', 'exp', 'abs',
            'floor', 'ceil', 'factorial'
        }
    
    def solve_equation(self, equation_str: str, variable: str = 'x') -> Dict[str, Any]:
        """
        Solve equation using SymPy with comprehensive validation.
        
        Args:
            equation_str: String representation of equation (e.g., "2*x + 3 = 7")
            variable: Variable to solve for
            
        Returns:
            Dictionary with solutions, steps, and validation results
        """
        try:
            # Parse equation
            if '=' in equation_str:
                left_str, right_str = equation_str.split('=', 1)
                left_expr = sympify(left_str.strip())
                right_expr = sympify(right_str.strip())
                equation = Eq(left_expr, right_expr)
            else:
                # Assume equation equals zero
                equation = Eq(sympify(equation_str.strip()), 0)
            
            # Define variable
            var = symbols(variable)
            
            # Solve equation
            solutions = sympy_solve(equation, var)
            
            # Validate solutions
            validation_results = self._validate_solutions(equation, var, solutions)
            
            # Generate solution steps
            steps = self._generate_solution_steps(equation, var, solutions)
            
            # Check for single solution guarantee
            has_single_solution = len(solutions) == 1
            
            return {
                "solutions": [complex(sol) if sol.is_complex else float(sol) for sol in solutions],
                "solution_count": len(solutions),
                "has_single_solution": has_single_solution,
                "steps": steps,
                "validation": validation_results,
                "equation_type": self._classify_equation(equation, var),
                "latex_form": latex(equation),
                "success": True
            }
            
        except SympifyError as e:
            logger.error(f"Failed to parse equation '{equation_str}': {e}")
            return {
                "success": False,
                "error": f"Parse error: {str(e)}",
                "error_type": "parse_error"
            }
        except Exception as e:
            logger.error(f"Solver error for '{equation_str}': {e}")
            return {
                "success": False,
                "error": f"Solver error: {str(e)}",
                "error_type": "solver_error"
            }
    
    def _validate_solutions(self, equation: Eq, variable: symbols, solutions: List) -> Dict[str, Any]:
        """Validate that solutions actually satisfy the equation."""
        validation_results = {
            "all_valid": True,
            "individual_checks": [],
            "verification_errors": []
        }
        
        for i, solution in enumerate(solutions):
            try:
                # Substitute solution back into equation
                left_result = equation.lhs.subs(variable, solution)
                right_result = equation.rhs.subs(variable, solution)
                
                # Simplify and check equality
                left_simplified = simplify(left_result)
                right_simplified = simplify(right_result)
                
                # Check if equal (with numerical tolerance)
                difference = simplify(left_simplified - right_simplified)
                is_valid = abs(complex(difference)) < 1e-10
                
                validation_results["individual_checks"].append({
                    "solution_index": i,
                    "solution_value": complex(solution),
                    "left_result": complex(left_simplified),
                    "right_result": complex(right_simplified),
                    "difference": complex(difference),
                    "is_valid": is_valid
                })
                
                if not is_valid:
                    validation_results["all_valid"] = False
                    validation_results["verification_errors"].append(
                        f"Solution {solution} does not satisfy equation"
                    )
                    
            except Exception as e:
                validation_results["all_valid"] = False
                validation_results["verification_errors"].append(
                    f"Validation error for solution {solution}: {str(e)}"
                )
        
        return validation_results
    
    def _generate_solution_steps(self, equation: Eq, variable: symbols, solutions: List) -> List[str]:
        """Generate human-readable solution steps."""
        steps = []
        
        try:
            # Start with original equation
            steps.append(f"Başlangıç: {equation}")
            
            # Classify equation type and generate appropriate steps
            eq_type = self._classify_equation(equation, variable)
            
            if eq_type == "linear":
                steps.extend(self._generate_linear_steps(equation, variable, solutions))
            elif eq_type == "quadratic":
                steps.extend(self._generate_quadratic_steps(equation, variable, solutions))
            elif eq_type == "rational":
                steps.extend(self._generate_rational_steps(equation, variable, solutions))
            else:
                # Generic steps
                steps.append(f"Çözüm: {variable} = {solutions}")
            
            # Add verification step
            if solutions:
                verification = f"Doğrulama: "
                for sol in solutions:
                    left_check = equation.lhs.subs(variable, sol)
                    right_check = equation.rhs.subs(variable, sol)
                    verification += f"{variable}={sol} → {simplify(left_check)} = {simplify(right_check)}"
                steps.append(verification)
            
        except Exception as e:
            logger.error(f"Step generation error: {e}")
            steps.append(f"Çözüm: {solutions}")
        
        return steps
    
    def _classify_equation(self, equation: Eq, variable: symbols) -> str:
        """Classify the type of equation."""
        try:
            # Get the polynomial degree
            poly_expr = equation.lhs - equation.rhs
            degree = sp.degree(poly_expr, variable)
            
            if degree == 1:
                return "linear"
            elif degree == 2:
                return "quadratic"
            elif poly_expr.has(1/variable):
                return "rational"
            elif degree > 2:
                return f"polynomial_degree_{degree}"
            else:
                return "other"
        except:
            return "unknown"
    
    def _generate_linear_steps(self, equation: Eq, variable: symbols, solutions: List) -> List[str]:
        """Generate steps for linear equations."""
        steps = []
        
        try:
            # Convert to standard form ax + b = c
            poly_expr = equation.lhs - equation.rhs
            coeffs = sp.Poly(poly_expr, variable).all_coeffs()
            
            if len(coeffs) == 2:  # ax + b = 0
                a, b = coeffs
                steps.append(f"Standart form: {a}*{variable} + {b} = 0")
                steps.append(f"İzole et: {a}*{variable} = {-b}")
                if a != 1:
                    steps.append(f"Böl: {variable} = {-b}/{a}")
                steps.append(f"Sonuç: {variable} = {solutions[0]}")
            
        except Exception as e:
            logger.error(f"Linear steps error: {e}")
            steps.append(f"Doğrudan çözüm: {variable} = {solutions}")
        
        return steps
    
    def _generate_quadratic_steps(self, equation: Eq, variable: symbols, solutions: List) -> List[str]:
        """Generate steps for quadratic equations."""
        steps = []
        
        try:
            # Try to factor first
            poly_expr = equation.lhs - equation.rhs
            factored = factor(poly_expr)
            
            if factored != poly_expr:
                steps.append(f"Faktörle: {factored} = 0")
                steps.append(f"Kökler: {solutions}")
            else:
                # Use quadratic formula
                coeffs = sp.Poly(poly_expr, variable).all_coeffs()
                if len(coeffs) == 3:
                    a, b, c = coeffs
                    discriminant = b**2 - 4*a*c
                    steps.append(f"Karesel formül: {variable} = (-{b} ± √({discriminant})) / (2*{a})")
                    steps.append(f"Diskriminant: Δ = {discriminant}")
                    steps.append(f"Kökler: {solutions}")
        
        except Exception as e:
            logger.error(f"Quadratic steps error: {e}")
            steps.append(f"Karesel çözüm: {solutions}")
        
        return steps
    
    def _generate_rational_steps(self, equation: Eq, variable: symbols, solutions: List) -> List[str]:
        """Generate steps for rational equations."""
        steps = []
        steps.append("Rasyonel denklem çözümü")
        steps.append("Ortak payda bulun ve çarpın")
        steps.append(f"Çözüm: {solutions}")
        return steps
    
    def verify_single_solution(self, equation_str: str, variable: str = 'x') -> bool:
        """Verify that equation has exactly one solution."""
        result = self.solve_equation(equation_str, variable)
        return result.get("success", False) and result.get("has_single_solution", False)
    
    def generate_solution_explanation(self, equation_str: str, variable: str = 'x', language: str = 'tr') -> str:
        """Generate detailed explanation of solution process."""
        result = self.solve_equation(equation_str, variable)
        
        if not result.get("success", False):
            return f"Denklem çözülemedi: {result.get('error', 'Bilinmeyen hata')}"
        
        explanation = []
        
        if language == 'tr':
            explanation.append(f"Denklem türü: {result.get('equation_type', 'bilinmeyen')}")
            explanation.append(f"Çözüm sayısı: {result.get('solution_count', 0)}")
            explanation.append("Çözüm adımları:")
            explanation.extend(result.get("steps", []))
            
            if result.get("validation", {}).get("all_valid", False):
                explanation.append("✓ Tüm çözümler doğrulandı")
            else:
                explanation.append("⚠ Bazı çözümler doğrulanamadı")
        else:
            explanation.append(f"Equation type: {result.get('equation_type', 'unknown')}")
            explanation.append(f"Number of solutions: {result.get('solution_count', 0)}")
            explanation.append("Solution steps:")
            explanation.extend(result.get("steps", []))
            
            if result.get("validation", {}).get("all_valid", False):
                explanation.append("✓ All solutions verified")
            else:
                explanation.append("⚠ Some solutions could not be verified")
        
        return "\n".join(explanation)


class MathGenerationService:
    """Service for generating math questions using templates."""
    
    def __init__(self):
        self.templates = {
            "linear_equation_v1": LinearEquationTemplate(),
            "quadratic_equation_v1": QuadraticEquationTemplate(),
            "ratio_proportion_v1": RatioProportionTemplate()
        }
        self.solver = SymPySolver()
    
    def get_available_templates(self) -> List[Dict[str, Any]]:
        """Get list of available templates."""
        return [
            {
                "template_id": template.template_id,
                "name": template.name,
                "skills": template.skills,
                "bloom_level": template.bloom_level,
                "difficulty_range": template.difficulty_range
            }
            for template in self.templates.values()
        ]
    
    def generate_question(
        self,
        template_id: str,
        params_hint: Optional[Dict[str, Any]] = None,
        target_difficulty: Optional[float] = None,
        language: str = "tr",
        rationale_required: bool = False
    ) -> Dict[str, Any]:
        """
        Generate a math question using specified template.
        
        Args:
            template_id: Template identifier
            params_hint: Parameter constraints/hints
            target_difficulty: Target difficulty level
            language: Question language
            rationale_required: Whether to include detailed rationale
            
        Returns:
            Generated question with validation results
        """
        if template_id not in self.templates:
            raise ValueError(f"Unknown template: {template_id}")
        
        template = self.templates[template_id]
        
        # Generate parameters
        max_attempts = 10
        for attempt in range(max_attempts):
            try:
                params = template.generate_parameters(params_hint)
                
                # Check if difficulty matches target
                if target_difficulty is not None:
                    estimated_difficulty = self._estimate_difficulty(template_id, params)
                    if abs(estimated_difficulty - target_difficulty) > 0.3:
                        continue  # Try again
                
                # Create question
                question = template.create_question(params, language)
                
                # Solve and validate
                solution = template.solve_question(params)
                is_valid = template.validate_solution(params, solution)
                
                if not is_valid:
                    continue  # Try again
                
                # Build result
                result = {
                    "item": {
                        "stem": question["stem"],
                        "choices": question.get("choices"),
                        "answer_key": question["answer_key"],
                        "solution": question.get("solution"),
                        "skills": template.skills,
                        "bloom_level": template.bloom_level,
                        "difficulty_estimate": {
                            "a": 1.0,  # Default discrimination
                            "b": self._estimate_difficulty(template_id, params)
                        },
                        "qa_checks": {
                            "solver_passed": is_valid,
                            "single_gold": True,  # Templates ensure single solution
                            "params_valid": True
                        }
                    },
                    "generator": {
                        "template_id": template_id,
                        "version": "1.0.0",
                        "params": params,
                        "language": language
                    }
                }
                
                # Add rationale if requested
                if rationale_required:
                    result["item"]["rationale"] = self._generate_rationale(
                        template, params, solution, language
                    )
                
                return result
                
            except Exception as e:
                print(f"Generation attempt {attempt + 1} failed: {e}")
                continue
        
        raise ValueError(f"Failed to generate valid question after {max_attempts} attempts")
    
    def _estimate_difficulty(self, template_id: str, params: Dict[str, Any]) -> float:
        """Estimate difficulty based on template and parameters."""
        if template_id == "linear_equation_v1":
            # Difficulty based on coefficient size and solution complexity
            a, b, c = params["a"], params["b"], params["c"]
            solution = (c - b) / a
            
            # Factors affecting difficulty
            coeff_complexity = (abs(a) + abs(b) + abs(c)) / 30.0
            solution_complexity = abs(solution) / 20.0
            
            difficulty = min(coeff_complexity + solution_complexity - 0.2, 1.0)
            return max(difficulty, -1.0)
        
        elif template_id == "quadratic_equation_v1":
            # Quadratics are generally harder
            return 0.5
        
        elif template_id == "ratio_proportion_v1":
            # Difficulty based on numbers involved
            a, b, c = params["a"], params["b"], params["c"]
            complexity = (a + b + c) / 30.0
            return min(complexity, 0.8) - 0.2
        
        return 0.0  # Default
    
    def _generate_rationale(
        self,
        template: MathTemplate,
        params: Dict[str, Any],
        solution: Dict[str, Any],
        language: str
    ) -> str:
        """Generate detailed rationale for the solution."""
        if language == "tr":
            rationale = f"Bu {template.name.lower()} sorusunu çözmek için:\n"
            rationale += f"1. Verilen denklemi analiz edelim\n"
            rationale += f"2. Adım adım çözüm: {solution.get('steps', '')}\n"
            rationale += f"3. Sonucu doğrulayalım: {solution.get('verification', '')}"
        else:
            rationale = f"To solve this {template.name.lower()} problem:\n"
            rationale += f"1. Analyze the given equation\n"
            rationale += f"2. Step-by-step solution: {solution.get('steps', '')}\n"
            rationale += f"3. Verify the result: {solution.get('verification', '')}"
        
        return rationale
    
    def validate_generated_question(self, question_data: Dict[str, Any]) -> Dict[str, bool]:
        """Validate a generated question."""
        checks = {
            "has_stem": bool(question_data.get("stem")),
            "has_answer_key": bool(question_data.get("answer_key")),
            "has_skills": bool(question_data.get("skills")),
            "solver_validated": question_data.get("qa_checks", {}).get("solver_passed", False),
            "single_solution": question_data.get("qa_checks", {}).get("single_gold", False)
        }
        
        return checks
    
    def batch_generate(
        self,
        template_id: str,
        count: int,
        params_hint: Optional[Dict[str, Any]] = None,
        language: str = "tr"
    ) -> List[Dict[str, Any]]:
        """Generate multiple questions in batch."""
        questions = []
        
        for i in range(count):
            try:
                question = self.generate_question(
                    template_id=template_id,
                    params_hint=params_hint,
                    language=language
                )
                questions.append(question)
            except Exception as e:
                print(f"Failed to generate question {i + 1}: {e}")
        
        return questions
    
    def solve_arbitrary_equation(
        self,
        equation_str: str,
        variable: str = 'x',
        language: str = 'tr'
    ) -> Dict[str, Any]:
        """
        Solve any mathematical equation using SymPy solver.
        
        Args:
            equation_str: String representation of equation
            variable: Variable to solve for
            language: Language for explanation
            
        Returns:
            Complete solution with steps and validation
        """
        result = self.solver.solve_equation(equation_str, variable)
        
        if result.get("success", False):
            # Add language-specific formatting
            if language == 'tr':
                result["explanation"] = self.solver.generate_solution_explanation(
                    equation_str, variable, 'tr'
                )
            else:
                result["explanation"] = self.solver.generate_solution_explanation(
                    equation_str, variable, 'en'
                )
        
        return result
    
    def validate_equation_solution(
        self,
        equation_str: str,
        proposed_solution: float,
        variable: str = 'x'
    ) -> Dict[str, bool]:
        """
        Validate if a proposed solution satisfies the equation.
        
        Args:
            equation_str: String representation of equation
            proposed_solution: Proposed solution value
            variable: Variable name
            
        Returns:
            Validation results
        """
        try:
            # Parse equation
            if '=' in equation_str:
                left_str, right_str = equation_str.split('=', 1)
                left_expr = sympify(left_str.strip())
                right_expr = sympify(right_str.strip())
            else:
                left_expr = sympify(equation_str.strip())
                right_expr = 0
            
            # Substitute proposed solution
            var = symbols(variable)
            left_result = left_expr.subs(var, proposed_solution)
            right_result = right_expr.subs(var, proposed_solution)
            
            # Check equality with tolerance
            difference = abs(complex(left_result - right_result))
            is_valid = difference < 1e-10
            
            return {
                "is_valid": is_valid,
                "left_result": float(left_result),
                "right_result": float(right_result),
                "difference": float(difference),
                "tolerance_met": difference < 1e-10
            }
            
        except Exception as e:
            logger.error(f"Validation error: {e}")
            return {
                "is_valid": False,
                "error": str(e)
            }
    
    def check_single_solution_guarantee(
        self,
        template_id: str,
        params: Dict[str, Any]
    ) -> bool:
        """
        Check if a template with given parameters guarantees a single solution.
        
        Args:
            template_id: Template identifier
            params: Template parameters
            
        Returns:
            True if single solution is guaranteed
        """
        if template_id not in self.templates:
            return False
        
        template = self.templates[template_id]
        
        try:
            # Generate question and solve
            solution = template.solve_question(params)
            
            # Check validation results
            solver_validation = solution.get("solver_validation", {})
            if solver_validation:
                return (
                    solver_validation.get("all_valid", False) and
                    len(solver_validation.get("individual_checks", [])) == 1
                )
            
            # Fallback checks based on template type
            if template_id == "linear_equation_v1":
                return True  # Linear equations always have single solution (if solvable)
            elif template_id == "quadratic_equation_v1":
                discriminant = solution.get("discriminant", 0)
                return discriminant > 0  # Two distinct real roots
            elif template_id == "ratio_proportion_v1":
                return True  # Proportions have single solution
            
            return False
            
        except Exception as e:
            logger.error(f"Single solution check failed: {e}")
            return False
    
    def generate_solution_steps(
        self,
        equation_str: str,
        variable: str = 'x',
        language: str = 'tr'
    ) -> List[str]:
        """
        Generate detailed solution steps for any equation.
        
        Args:
            equation_str: String representation of equation
            variable: Variable to solve for
            language: Language for steps
            
        Returns:
            List of solution steps
        """
        result = self.solver.solve_equation(equation_str, variable)
        
        if result.get("success", False):
            return result.get("steps", [])
        else:
            if language == 'tr':
                return [f"Denklem çözülemedi: {result.get('error', 'Bilinmeyen hata')}"]
            else:
                return [f"Could not solve equation: {result.get('error', 'Unknown error')}"]
    
    def generate_distractors_for_question(
        self,
        template_id: str,
        params: Dict[str, Any],
        correct_answer: Any,
        num_distractors: int = 3,
        analysis: bool = False
    ) -> Dict[str, Any]:
        """
        Generate distractors for a specific question with optional analysis.
        
        Args:
            template_id: Template identifier
            params: Template parameters
            correct_answer: Correct answer
            num_distractors: Number of distractors to generate
            analysis: Whether to include quality analysis
            
        Returns:
            Dictionary with distractors and optional analysis
        """
        distractors = distractor_generator.generate_distractors(
            template_id=template_id,
            params=params,
            correct_answer=correct_answer,
            num_distractors=num_distractors,
            ensure_uniqueness=True,
            plausibility_check=True
        )
        
        result = {
            "distractors": distractors,
            "count": len(distractors),
            "template_id": template_id
        }
        
        if analysis:
            quality_analysis = distractor_generator.analyze_distractor_quality(
                distractors, correct_answer, template_id
            )
            result["quality_analysis"] = quality_analysis
        
        return result
    
    def validate_distractor_quality(
        self,
        template_id: str,
        distractors: List[Any],
        correct_answer: Any
    ) -> Dict[str, Any]:
        """
        Validate the quality of provided distractors.
        
        Args:
            template_id: Template identifier
            distractors: List of distractors to validate
            correct_answer: Correct answer for comparison
            
        Returns:
            Quality analysis results
        """
        return distractor_generator.analyze_distractor_quality(
            distractors, correct_answer, template_id
        )
    
    def get_misconception_patterns(self, template_id: str) -> List[str]:
        """
        Get available misconception patterns for a template.
        
        Args:
            template_id: Template identifier
            
        Returns:
            List of misconception pattern names
        """
        if template_id in distractor_generator.misconception_registry:
            misconceptions = distractor_generator.misconception_registry[template_id]
            return [func.__name__ for func in misconceptions]
        return []
    
    def test_misconception_pattern(
        self,
        template_id: str,
        misconception_name: str,
        params: Dict[str, Any],
        correct_answer: Any
    ) -> Dict[str, Any]:
        """
        Test a specific misconception pattern.
        
        Args:
            template_id: Template identifier
            misconception_name: Name of misconception to test
            params: Template parameters
            correct_answer: Correct answer
            
        Returns:
            Test results including generated distractor
        """
        if template_id not in distractor_generator.misconception_registry:
            return {"error": f"No misconceptions registered for template {template_id}"}
        
        misconceptions = distractor_generator.misconception_registry[template_id]
        
        # Find the misconception function by name
        target_func = None
        for func in misconceptions:
            if func.__name__ == misconception_name:
                target_func = func
                break
        
        if target_func is None:
            return {"error": f"Misconception '{misconception_name}' not found"}
        
        try:
            distractor = target_func(params, correct_answer)
            
            return {
                "misconception_name": misconception_name,
                "generated_distractor": distractor,
                "is_different_from_correct": distractor != correct_answer,
                "is_plausible": distractor_generator._is_plausible(
                    distractor, correct_answer, template_id
                ) if distractor is not None else False
            }
            
        except Exception as e:
            return {
                "error": f"Failed to apply misconception: {str(e)}",
                "misconception_name": misconception_name
            }


# Create service instance
math_generation_service = MathGenerationService()