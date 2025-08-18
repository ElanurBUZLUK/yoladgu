#!/usr/bin/env python3
"""
Comprehensive Answer Evaluation API test script
"""
import asyncio
import httpx
import json

BASE_URL = "http://localhost:8000"

async def test_answer_evaluation_api():
    """Test all answer evaluation API endpoints"""
    
    async with httpx.AsyncClient() as client:
        print("🧪 Testing Answer Evaluation API Endpoints...")
        
        # First, login to get token
        print("\n1. Logging in...")
        login_data = {
            "username": "testuser",
            "password": "testpass123"
        }
        
        try:
            response = await client.post(f"{BASE_URL}/api/v1/users/login", json=login_data)
            if response.status_code == 200:
                token_data = response.json()
                access_token = token_data["access_token"]
                headers = {"Authorization": f"Bearer {access_token}"}
                print("✅ Login successful")
            else:
                print(f"❌ Login failed: {response.status_code}")
                return
        except Exception as e:
            print(f"❌ Login error: {e}")
            return
        
        # Get a question to test with
        print("\n2. Getting a test question...")
        question_id = None
        try:
            response = await client.get(
                f"{BASE_URL}/api/v1/math/questions/search?limit=1",
                headers=headers
            )
            
            if response.status_code == 200:
                search_data = response.json()
                if search_data.get('questions'):
                    question_id = search_data['questions'][0]['id']
                    question_content = search_data['questions'][0]['content']
                    correct_answer = search_data['questions'][0]['correct_answer']
                    print(f"✅ Got test question: {question_content[:50]}...")
                    print(f"   Question ID: {question_id}")
                    print(f"   Correct answer: {correct_answer}")
                else:
                    print("❌ No questions found for testing")
                    return
            else:
                print(f"❌ Failed to get test question: {response.status_code}")
                return
        except Exception as e:
            print(f"❌ Error getting test question: {e}")
            return
        
        # Test answer evaluation (without saving)
        print("\n3. Testing answer evaluation...")
        try:
            evaluation_data = {
                "question_id": question_id,
                "student_answer": correct_answer,  # Use correct answer first
                "use_llm": False  # Use rule-based for testing
            }
            
            response = await client.post(
                f"{BASE_URL}/api/v1/answers/evaluate",
                json=evaluation_data,
                headers=headers
            )
            
            if response.status_code == 200:
                print("✅ Answer evaluation successful")
                evaluation = response.json()
                print(f"   Is correct: {evaluation.get('is_correct')}")
                print(f"   Score: {evaluation.get('score')}")
                print(f"   Feedback: {evaluation.get('feedback', '')[:50]}...")
                
                if evaluation.get('recommendations'):
                    print(f"   Recommendations: {len(evaluation['recommendations'])} items")
            else:
                print(f"❌ Answer evaluation failed: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"❌ Answer evaluation error: {e}")
        
        # Test answer submission (with saving)
        print("\n4. Testing answer submission...")
        try:
            submission_data = {
                "question_id": question_id,
                "student_answer": "wrong answer",  # Intentionally wrong
                "time_spent": 45
            }
            
            response = await client.post(
                f"{BASE_URL}/api/v1/answers/submit",
                json=submission_data,
                headers=headers
            )
            
            if response.status_code == 200:
                print("✅ Answer submission successful")
                result = response.json()
                print(f"   Success: {result.get('success')}")
                print(f"   Message: {result.get('message')}")
                
                evaluation = result.get('evaluation', {})
                print(f"   Evaluation - Correct: {evaluation.get('is_correct')}")
                print(f"   Evaluation - Score: {evaluation.get('score')}")
                
                attempt = result.get('attempt', {})
                print(f"   Attempt ID: {attempt.get('id')}")
                print(f"   Time spent: {attempt.get('time_spent')} seconds")
            else:
                print(f"❌ Answer submission failed: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"❌ Answer submission error: {e}")
        
        # Submit a few more answers to build performance data
        print("\n5. Submitting additional answers for performance data...")
        for i in range(3):
            try:
                # Alternate between correct and incorrect answers
                answer = correct_answer if i % 2 == 0 else "incorrect"
                
                submission_data = {
                    "question_id": question_id,
                    "student_answer": answer,
                    "time_spent": 30 + (i * 10)
                }
                
                response = await client.post(
                    f"{BASE_URL}/api/v1/answers/submit",
                    json=submission_data,
                    headers=headers
                )
                
                if response.status_code == 200:
                    result = response.json()
                    is_correct = result.get('evaluation', {}).get('is_correct')
                    print(f"   Submission {i+1}: {'✅' if is_correct else '❌'}")
                
            except Exception as e:
                print(f"   Submission {i+1} error: {e}")
        
        # Test performance metrics
        print("\n6. Testing performance metrics...")
        try:
            response = await client.get(
                f"{BASE_URL}/api/v1/answers/performance?days=30",
                headers=headers
            )
            
            if response.status_code == 200:
                print("✅ Performance metrics successful")
                metrics = response.json()
                print(f"   Total attempts: {metrics.get('total_attempts', 0)}")
                print(f"   Correct attempts: {metrics.get('correct_attempts', 0)}")
                print(f"   Accuracy rate: {metrics.get('accuracy_rate', 0):.1f}%")
                print(f"   Current streak: {metrics.get('current_streak', 0)}")
                print(f"   Best streak: {metrics.get('best_streak', 0)}")
                print(f"   Average time: {metrics.get('average_time_per_question', 0):.1f}s")
            else:
                print(f"❌ Performance metrics failed: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"❌ Performance metrics error: {e}")
        
        # Test error analysis
        print("\n7. Testing error analysis...")
        try:
            response = await client.get(
                f"{BASE_URL}/api/v1/answers/error-analysis",
                headers=headers
            )
            
            if response.status_code == 200:
                print("✅ Error analysis successful")
                errors = response.json()
                print(f"   Error patterns found: {len(errors)}")
                
                for error in errors[:3]:  # Show first 3 errors
                    print(f"   - {error.get('error_type')}: {error.get('frequency')} times")
                    print(f"     Subject: {error.get('subject')}")
                    print(f"     Last occurrence: {error.get('last_occurrence', '')[:10]}")
            else:
                print(f"❌ Error analysis failed: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"❌ Error analysis error: {e}")
        
        # Test detailed error analysis for math
        print("\n8. Testing detailed math error analysis...")
        try:
            response = await client.get(
                f"{BASE_URL}/api/v1/answers/detailed-error-analysis/math",
                headers=headers
            )
            
            if response.status_code == 200:
                print("✅ Detailed math error analysis successful")
                analysis = response.json()
                
                math_errors = analysis.get('math_errors', {})
                print(f"   Math errors: {len(math_errors)} types")
                
                most_common = analysis.get('most_common_errors', [])
                if most_common:
                    print(f"   Most common errors: {most_common[:3]}")
            else:
                print(f"❌ Detailed math error analysis failed: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"❌ Detailed math error analysis error: {e}")
        
        # Test level adjustment recommendation
        print("\n9. Testing level adjustment recommendation...")
        try:
            response = await client.get(
                f"{BASE_URL}/api/v1/answers/level-recommendation/math",
                headers=headers
            )
            
            if response.status_code == 200:
                print("✅ Level recommendation successful")
                recommendation = response.json()
                print(f"   Has recommendation: {recommendation.get('has_recommendation')}")
                print(f"   Message: {recommendation.get('message')}")
                
                if recommendation.get('has_recommendation'):
                    rec_data = recommendation.get('recommendation', {})
                    print(f"   Current level: {rec_data.get('current_level')}")
                    print(f"   Recommended level: {rec_data.get('recommended_level')}")
                    print(f"   Reason: {rec_data.get('reason', '')[:50]}...")
            else:
                print(f"❌ Level recommendation failed: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"❌ Level recommendation error: {e}")
        
        # Test user attempts history
        print("\n10. Testing user attempts history...")
        try:
            response = await client.get(
                f"{BASE_URL}/api/v1/answers/attempts?limit=5",
                headers=headers
            )
            
            if response.status_code == 200:
                print("✅ User attempts history successful")
                data = response.json()
                attempts = data.get('attempts', [])
                print(f"   Attempts returned: {len(attempts)}")
                print(f"   Total returned: {data.get('total_returned', 0)}")
                
                if attempts:
                    first_attempt = attempts[0]
                    print(f"   Latest attempt:")
                    print(f"     Question: {first_attempt.get('question_content', '')[:30]}...")
                    print(f"     Answer: {first_attempt.get('student_answer', '')[:20]}...")
                    print(f"     Correct: {first_attempt.get('is_correct')}")
                    print(f"     Date: {first_attempt.get('attempt_date', '')[:10]}")
            else:
                print(f"❌ User attempts history failed: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"❌ User attempts history error: {e}")
        
        # Test statistics summary
        print("\n11. Testing statistics summary...")
        try:
            response = await client.get(
                f"{BASE_URL}/api/v1/answers/statistics/summary",
                headers=headers
            )
            
            if response.status_code == 200:
                print("✅ Statistics summary successful")
                stats = response.json()
                
                overall = stats.get('overall', {})
                print(f"   Overall accuracy: {overall.get('accuracy_rate', 0):.1f}%")
                print(f"   Total attempts: {overall.get('total_attempts', 0)}")
                print(f"   Current streak: {overall.get('current_streak', 0)}")
                
                by_subject = stats.get('by_subject', {})
                for subject, data in by_subject.items():
                    print(f"   {subject.title()}: {data.get('accuracy', 0):.1f}% ({data.get('attempts', 0)} attempts)")
                
                improvement_areas = stats.get('improvement_areas', [])
                if improvement_areas:
                    print(f"   Improvement areas: {', '.join(improvement_areas[:3])}")
            else:
                print(f"❌ Statistics summary failed: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"❌ Statistics summary error: {e}")
        
        # Test personalized feedback
        print("\n12. Testing personalized feedback...")
        try:
            response = await client.get(
                f"{BASE_URL}/api/v1/answers/feedback/{question_id}",
                headers=headers
            )
            
            if response.status_code == 200:
                print("✅ Personalized feedback successful")
                feedback_data = response.json()
                print(f"   Question ID: {feedback_data.get('question_id')}")
                print(f"   Is correct: {feedback_data.get('is_correct')}")
                
                feedback = feedback_data.get('personalized_feedback', {})
                positive = feedback.get('positive_feedback', [])
                constructive = feedback.get('constructive_feedback', [])
                
                if positive:
                    print(f"   Positive feedback: {positive[0][:50]}...")
                if constructive:
                    print(f"   Constructive feedback: {constructive[0][:50]}...")
            else:
                print(f"❌ Personalized feedback failed: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"❌ Personalized feedback error: {e}")
        
        print("\n🎉 Answer Evaluation API test completed!")

