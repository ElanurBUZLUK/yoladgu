"""
Demo: Math Recommendation System (IRT + Multi-Skill Elo)
Adaptive math learning with intelligent question selection
"""

import asyncio
import time
import numpy as np
from typing import List, Dict, Any
import structlog

# Import our math recommendation components
from ml.math.irt import irt_model, ItemParams, StudentAbility
from ml.math.multiskill_elo import multiskill_elo
from ml.math.selector import math_selector

logger = structlog.get_logger()

# Sample questions for testing
SAMPLE_QUESTIONS = [
    {
        "id": "math_001",
        "type": "algebra",
        "question": "Solve for x: 2x + 5 = 13",
        "options": ["x = 4", "x = 3", "x = 5", "x = 6"],
        "correct_answer": 0,
        "required_skills": ["linear_equations", "algebra"],
        "skill_weights": {"linear_equations": 0.7, "algebra": 0.3},
        "difficulty_level": "beginner",
        "irt_params": {
            "item_id": "math_001",
            "a": 1.2,
            "b": -0.5,
            "c": 0.0,
            "skill_weights": {"linear_equations": 0.7, "algebra": 0.3}
        }
    },
    {
        "id": "math_002",
        "type": "geometry",
        "question": "Find the area of a circle with radius 7 cm",
        "options": ["154 cm²", "44 cm²", "22 cm²", "77 cm²"],
        "correct_answer": 0,
        "required_skills": ["circle_area", "geometry"],
        "skill_weights": {"circle_area": 0.8, "geometry": 0.2},
        "difficulty_level": "intermediate",
        "irt_params": {
            "item_id": "math_002",
            "a": 1.0,
            "b": 0.2,
            "c": 0.0,
            "skill_weights": {"circle_area": 0.8, "geometry": 0.2}
        }
    },
    {
        "id": "math_003",
        "type": "calculus",
        "question": "Find the derivative of f(x) = x³ + 2x² - 5x + 1",
        "options": ["3x² + 4x - 5", "3x² + 4x + 5", "x² + 4x - 5", "3x² - 4x - 5"],
        "correct_answer": 0,
        "required_skills": ["derivatives", "polynomials", "calculus"],
        "skill_weights": {"derivatives": 0.6, "polynomials": 0.3, "calculus": 0.1},
        "difficulty_level": "advanced",
        "irt_params": {
            "item_id": "math_003",
            "a": 1.5,
            "b": 1.2,
            "c": 0.0,
            "skill_weights": {"derivatives": 0.6, "polynomials": 0.3, "calculus": 0.1}
        }
    },
    {
        "id": "math_004",
        "type": "statistics",
        "question": "What is the mean of the numbers: 2, 4, 6, 8, 10?",
        "options": ["6", "5", "7", "8"],
        "correct_answer": 0,
        "required_skills": ["mean", "statistics"],
        "skill_weights": {"mean": 0.9, "statistics": 0.1},
        "difficulty_level": "beginner",
        "irt_params": {
            "item_id": "math_004",
            "a": 1.1,
            "b": -1.0,
            "c": 0.0,
            "skill_weights": {"mean": 0.9, "statistics": 0.1}
        }
    },
    {
        "id": "math_005",
        "type": "trigonometry",
        "question": "What is sin(30°)?",
        "options": ["1/2", "√3/2", "1", "0"],
        "correct_answer": 0,
        "required_skills": ["trigonometry", "special_angles"],
        "skill_weights": {"trigonometry": 0.4, "special_angles": 0.6},
        "difficulty_level": "intermediate",
        "irt_params": {
            "item_id": "math_005",
            "a": 1.3,
            "b": 0.5,
            "c": 0.0,
            "skill_weights": {"trigonometry": 0.4, "special_angles": 0.6}
        }
    }
]

async def test_irt_model():
    """Test IRT model functionality"""
    print("\n📊 IRT Model Test")
    print("=" * 50)
    
    try:
        # Add sample items
        for question in SAMPLE_QUESTIONS:
            item_params = ItemParams.from_dict(question["irt_params"])
            irt_model.add_item(item_params)
        
        print(f"✅ Added {len(SAMPLE_QUESTIONS)} items to IRT model")
        
        # Test probability calculations
        test_theta = 0.5
        for item_id, item in irt_model.items.items():
            prob = irt_model.p_correct(test_theta, item)
            print(f"   Item {item_id}: P(correct|θ={test_theta}) = {prob:.3f}")
        
        # Test multi-skill probability
        skill_thetas = {"linear_equations": 0.8, "algebra": 0.6}
        item = irt_model.items["math_001"]
        multi_prob = irt_model.p_correct_multi_skill(skill_thetas, item)
        print(f"   Multi-skill prob: {multi_prob:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ IRT model test failed: {e}")
        return False

async def test_multiskill_elo():
    """Test Multi-Skill Elo system"""
    print("\n🏆 Multi-Skill Elo Test")
    print("=" * 50)
    
    try:
        # Add items to Elo system
        for question in SAMPLE_QUESTIONS:
            multiskill_elo.add_item(
                question["id"], 
                question["required_skills"]
            )
        
        print(f"✅ Added {len(SAMPLE_QUESTIONS)} items to Elo system")
        
        # Test with sample users
        test_users = ["student_1", "student_2", "student_3"]
        
        for user_id in test_users:
            # Add user skills
            for question in SAMPLE_QUESTIONS:
                for skill in question["required_skills"]:
                    multiskill_elo.add_user_skill(user_id, skill)
            
            print(f"   Added user {user_id} with skills")
        
        # Test expected score calculation
        user_id = "student_1"
        item_id = "math_001"
        expected_score = multiskill_elo.calculate_expected_score(user_id, item_id)
        print(f"   Expected score for {user_id} on {item_id}: {expected_score:.3f}")
        
        # Test rating updates
        expected, actual = multiskill_elo.update_ratings(
            user_id=user_id,
            item_id=item_id,
            correct=True
        )
        print(f"   Rating update: expected={expected:.3f}, actual={actual:.3f}")
        
        # Get user stats
        user_stats = multiskill_elo.get_user_stats(user_id)
        print(f"   User {user_id} overall rating: {user_stats['overall_rating']:.1f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Multi-skill Elo test failed: {e}")
        return False

async def test_question_selector():
    """Test question selector"""
    print("\n🎯 Question Selector Test")
    print("=" * 50)
    
    try:
        # Test question selection for different users
        test_users = ["beginner_student", "intermediate_student", "advanced_student"]
        
        for user_id in test_users:
            print(f"\n   Testing user: {user_id}")
            
            # Initialize user in both systems
            if user_id not in irt_model.students:
                student = StudentAbility(user_id=user_id, theta=0.0)
                irt_model.add_student(student)
            
            # Add skills to Elo system
            for question in SAMPLE_QUESTIONS:
                for skill in question["required_skills"]:
                    multiskill_elo.add_user_skill(user_id, skill)
            
            # Select next question
            selected_question = math_selector.select_next_question(
                user_id=user_id,
                available_questions=SAMPLE_QUESTIONS
            )
            
            if selected_question:
                print(f"      Selected: {selected_question['id']} ({selected_question['type']})")
                print(f"      Skills: {selected_question['required_skills']}")
                print(f"      Difficulty: {selected_question['difficulty_level']}")
                
                # Calculate expected probability
                item_params = ItemParams.from_dict(selected_question["irt_params"])
                expected_prob = math_selector.calculate_expected_probability(
                    user_id, selected_question["id"], item_params, selected_question["required_skills"]
                )
                print(f"      Expected probability: {expected_prob:.3f}")
                
                # Calculate learning gain
                learning_gain = math_selector.calculate_learning_gain(
                    user_id, item_params, selected_question["required_skills"], selected_question["skill_weights"]
                )
                print(f"      Learning gain: {learning_gain:.3f}")
            else:
                print(f"      No suitable question found")
        
        return True
        
    except Exception as e:
        print(f"❌ Question selector test failed: {e}")
        return False

async def test_adaptive_learning_simulation():
    """Simulate adaptive learning session"""
    print("\n🎓 Adaptive Learning Simulation")
    print("=" * 50)
    
    try:
        user_id = "adaptive_student"
        
        # Initialize student
        student = StudentAbility(user_id=user_id, theta=0.0)
        irt_model.add_student(student)
        
        # Add skills
        for question in SAMPLE_QUESTIONS:
            for skill in question["required_skills"]:
                multiskill_elo.add_user_skill(user_id, skill)
        
        print(f"   Simulating learning session for {user_id}")
        
        # Simulate 10 questions
        session_results = []
        
        for i in range(10):
            # Select next question
            selected_question = math_selector.select_next_question(
                user_id=user_id,
                available_questions=SAMPLE_QUESTIONS,
                recent_questions=[r["item_id"] for r in session_results[-3:]]  # Avoid last 3
            )
            
            if not selected_question:
                print(f"      No more suitable questions")
                break
            
            # Simulate answer (higher probability for easier questions)
            difficulty = selected_question["difficulty_level"]
            if difficulty == "beginner":
                correct_prob = 0.8
            elif difficulty == "intermediate":
                correct_prob = 0.6
            else:  # advanced
                correct_prob = 0.4
            
            # Add some randomness based on user ability
            user_ability = irt_model.students[user_id].theta
            correct_prob = min(0.95, max(0.05, correct_prob + user_ability * 0.1))
            
            correct = np.random.random() < correct_prob
            
            # Process answer
            item_params = ItemParams.from_dict(selected_question["irt_params"])
            
            # Update IRT
            irt_model.add_response(
                user_id=user_id,
                item_id=selected_question["id"],
                correct=correct
            )
            
            # Update Elo
            expected, actual = multiskill_elo.update_ratings(
                user_id=user_id,
                item_id=selected_question["id"],
                correct=correct,
                skill_weights=selected_question["skill_weights"]
            )
            
            # Update IRT ability
            user_responses = [r for r in irt_model.responses if r["user_id"] == user_id]
            new_theta = irt_model.estimate_ability(user_id, user_responses)
            irt_model.students[user_id].theta = new_theta
            
            session_results.append({
                "item_id": selected_question["id"],
                "correct": correct,
                "expected_prob": expected,
                "theta": new_theta
            })
            
            print(f"      Q{i+1}: {selected_question['id']} - {'✓' if correct else '✗'} "
                  f"(expected: {expected:.3f}, θ: {new_theta:.3f})")
        
        # Analyze session
        total_correct = sum(1 for r in session_results if r["correct"])
        accuracy = total_correct / len(session_results)
        theta_improvement = session_results[-1]["theta"] - session_results[0]["theta"]
        
        print(f"\n   Session Summary:")
        print(f"      Questions attempted: {len(session_results)}")
        print(f"      Accuracy: {accuracy:.3f}")
        print(f"      Ability improvement: {theta_improvement:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Adaptive learning simulation failed: {e}")
        return False

async def test_performance_benchmarks():
    """Performance benchmarks"""
    print("\n⚡ Performance Benchmarks")
    print("=" * 50)
    
    try:
        # Benchmark question selection
        user_id = "benchmark_user"
        student = StudentAbility(user_id=user_id, theta=0.0)
        irt_model.add_student(student)
        
        for question in SAMPLE_QUESTIONS:
            for skill in question["required_skills"]:
                multiskill_elo.add_user_skill(user_id, skill)
        
        # Time question selection
        times = []
        for _ in range(100):
            start_time = time.time()
            math_selector.select_next_question(user_id, SAMPLE_QUESTIONS)
            times.append((time.time() - start_time) * 1000)
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        
        print(f"   Question selection: {avg_time:.2f}±{std_time:.2f}ms")
        
        # Benchmark rating updates
        times = []
        for _ in range(100):
            start_time = time.time()
            multiskill_elo.update_ratings(user_id, "math_001", True)
            times.append((time.time() - start_time) * 1000)
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        
        print(f"   Rating updates: {avg_time:.2f}±{std_time:.2f}ms")
        
        # Benchmark IRT probability calculation
        times = []
        item = irt_model.items["math_001"]
        for _ in range(1000):
            start_time = time.time()
            irt_model.p_correct(0.5, item)
            times.append((time.time() - start_time) * 1000)
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        
        print(f"   IRT probability: {avg_time:.3f}±{std_time:.3f}ms")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance benchmarks failed: {e}")
        return False

async def main():
    """Main demo function"""
    print("🧮 Math Recommendation System Demo")
    print("=" * 60)
    print("IRT + Multi-Skill Elo for Adaptive Math Learning")
    print("=" * 60)
    
    try:
        # Test IRT model
        irt_success = await test_irt_model()
        if not irt_success:
            print("❌ IRT model test failed, stopping demo")
            return
        
        # Test Multi-Skill Elo
        elo_success = await test_multiskill_elo()
        if not elo_success:
            print("❌ Multi-skill Elo test failed, stopping demo")
            return
        
        # Test question selector
        selector_success = await test_question_selector()
        if not selector_success:
            print("❌ Question selector test failed, stopping demo")
            return
        
        # Test adaptive learning simulation
        simulation_success = await test_adaptive_learning_simulation()
        if not simulation_success:
            print("❌ Adaptive learning simulation failed, stopping demo")
            return
        
        # Performance benchmarks
        await test_performance_benchmarks()
        
        print("\n✅ Demo completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
