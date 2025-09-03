"""
Test generation API endpoints.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

import asyncio
from fastapi.testclient import TestClient
from app.main import app

# Create test client
client = TestClient(app)

def test_generate_endpoints():
    """Test math generation API endpoints."""
    
    print("=== Testing Math Generation Endpoints ===")
    
    # Test list math templates
    print("\n1. Testing GET /api/v1/generate/templates/math")
    response = client.get(
        "/api/v1/generate/templates/math",
        headers={"Authorization": "Bearer test-token"}
    )
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"Templates found: {data['total_count']}")
        for template in data['templates']:
            print(f"  - {template['template_id']}: {template['name']}")
    else:
        print(f"Error: {response.json()}")
    
    # Test generate math question
    print("\n2. Testing POST /api/v1/generate/math")
    request_data = {
        "template_id": "linear_equation_v1",
        "language": "tr",
        "rationale_required": False
    }
    
    response = client.post(
        "/api/v1/generate/math",
        json=request_data,
        headers={"Authorization": "Bearer test-token"}
    )
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"Generated question: {data['item']['stem']}")
        print(f"Answer key: {data['item']['answer_key']}")
        print(f"Choices: {data['item']['choices']}")
        print(f"QA checks: {data['item']['qa_checks']}")
    else:
        print(f"Error: {response.json()}")
    
    # Test with target difficulty
    print("\n3. Testing with target difficulty")
    request_data = {
        "template_id": "quadratic_equation_v1",
        "target_difficulty": 0.5,
        "language": "tr"
    }
    
    response = client.post(
        "/api/v1/generate/math",
        json=request_data,
        headers={"Authorization": "Bearer test-token"}
    )
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"Quadratic question: {data['item']['stem']}")
        print(f"Answer: {data['item']['answer_key']}")
        print(f"Type: {data['item'].get('type', 'unknown')}")
    else:
        print(f"Error: {response.json()}")
    
    # Test batch generation
    print("\n4. Testing POST /api/v1/generate/math/batch")
    response = client.post(
        "/api/v1/generate/math/batch?template_id=ratio_proportion_v1&count=3&language=tr",
        headers={"Authorization": "Bearer test-token"}
    )
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"Generated {data['generated_count']} questions")
        for i, question in enumerate(data['questions'][:2], 1):  # Show first 2
            print(f"  Question {i}: {question['item']['stem']}")
    else:
        print(f"Error: {response.json()}")
    
    # Test equation solving
    print("\n5. Testing POST /api/v1/generate/math/solve")
    response = client.post(
        "/api/v1/generate/math/solve",
        params={"equation": "3*x + 6 = 15", "language": "tr"},
        headers={"Authorization": "Bearer test-token"}
    )
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"Equation: {data['equation']}")
        print(f"Success: {data['result']['success']}")
        print(f"Solutions: {data['result'].get('solutions', [])}")
    else:
        print(f"Error: {response.json()}")
    
    # Test distractor generation
    print("\n6. Testing POST /api/v1/generate/math/distractors")
    distractor_data = {
        "template_id": "linear_equation_v1",
        "params": {"a": 2, "b": 3, "c": 7},
        "correct_answer": 2,
        "num_distractors": 3,
        "analysis": True
    }
    
    response = client.post(
        "/api/v1/generate/math/distractors",
        json=distractor_data,
        headers={"Authorization": "Bearer test-token"}
    )
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"Distractors: {data['distractors']}")
        print(f"Quality score: {data.get('quality_analysis', {}).get('overall_plausibility', 'N/A')}")
    else:
        print(f"Error: {response.json()}")
    
    # Test misconception listing
    print("\n7. Testing GET /api/v1/generate/math/misconceptions/linear_equation_v1")
    response = client.get(
        "/api/v1/generate/math/misconceptions/linear_equation_v1",
        headers={"Authorization": "Bearer test-token"}
    )
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"Misconceptions for {data['template_id']}: {data['misconceptions']}")
    else:
        print(f"Error: {response.json()}")
    
    # Test error handling - invalid template
    print("\n8. Testing error handling - invalid template")
    request_data = {
        "template_id": "invalid_template",
        "language": "tr"
    }
    
    response = client.post(
        "/api/v1/generate/math",
        json=request_data,
        headers={"Authorization": "Bearer test-token"}
    )
    print(f"Status: {response.status_code}")
    print(f"Error response: {response.json()}")
    
    # Test English templates (placeholder)
    print("\n9. Testing GET /api/v1/generate/templates/english")
    response = client.get(
        "/api/v1/generate/templates/english",
        headers={"Authorization": "Bearer test-token"}
    )
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"English status: {data['status']}")
        print(f"Available error types: {data['available_error_types']}")
    else:
        print(f"Error: {response.json()}")
    
    print("\nAll generation endpoint tests completed!")

def test_validation_endpoints():
    """Test validation and quality assurance endpoints."""
    
    print("\n=== Testing Validation Endpoints ===")
    
    # Test question validation
    print("\n1. Testing POST /api/v1/generate/math/validate")
    
    # First generate a question to validate
    response = client.post(
        "/api/v1/generate/math",
        json={"template_id": "linear_equation_v1", "language": "tr"},
        headers={"Authorization": "Bearer test-token"}
    )
    
    if response.status_code == 200:
        question_data = response.json()["item"]
        
        # Now validate it
        validation_response = client.post(
            "/api/v1/generate/math/validate",
            json=question_data.dict() if hasattr(question_data, 'dict') else question_data,
            headers={"Authorization": "Bearer test-token"}
        )
        
        print(f"Validation status: {validation_response.status_code}")
        if validation_response.status_code == 200:
            validation_data = validation_response.json()
            print(f"Overall valid: {validation_data['overall_valid']}")
            print(f"Validation results: {validation_data['validation_results']}")
        else:
            print(f"Validation error: {validation_response.json()}")
    else:
        print("Could not generate question for validation test")

if __name__ == "__main__":
    test_generate_endpoints()
    test_validation_endpoints()