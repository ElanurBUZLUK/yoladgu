#!/usr/bin/env python3
"""
Error Pattern Analytics test script
"""
import asyncio
import httpx
import json

BASE_URL = "http://localhost:8000"

async def test_error_pattern_analytics():
    """Test error pattern tracking and analytics"""
    
    async with httpx.AsyncClient() as client:
        print("🧪 Testing Error Pattern Analytics...")
        
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
        
        # Submit some wrong answers to create error patterns
        print("\n2. Creating error patterns by submitting wrong answers...")
        
        # Get a math question
        try:
            response = await client.get(
                f"{BASE_URL}/api/v1/math/questions/search?limit=1",
                headers=headers
            )
            
            if response.status_code == 200:
                search_data = response.json()
                if search_data.get('questions'):
                    question_id = search_data['questions'][0]['id']
                    print(f"✅ Got math question: {question_id}")
                    
                    # Submit several wrong answers to create error patterns
                    error_submissions = [
                        {"student_answer": "wrong1", "error_type": "calculation_error"},
                        {"student_answer": "wrong2", "error_type": "calculation_error"},
                        {"student_answer": "wrong3", "error_type": "concept_error"},
                    ]
                    
                    for i, submission in enumerate(error_submissions):
                        submission_data = {
                            "question_id": question_id,
                            "student_answer": submission["student_answer"],
                            "time_spent": 30 + (i * 10)
                        }
                        
                        response = await client.post(
                            f"{BASE_URL}/api/v1/answers/submit",
                            json=submission_data,
                            headers=headers
                        )
                        
                        if response.status_code == 200:
                            print(f"   Submitted wrong answer {i+1}: ✅")
                        else:
                            print(f"   Submission {i+1} failed: {response.status_code}")
                
                else:
                    print("❌ No math questions found")
                    return
            else:
                print(f"❌ Failed to get math question: {response.status_code}")
                return
        except Exception as e:
            print(f"❌ Error creating error patterns: {e}")
        
        # Test manual error pattern tracking
        print("\n3. Testing manual error pattern tracking...")
        try:
            error_data = {
                "subject": "math",
                "error_type": "addition_error",
                "topic_category": "arithmetic",
                "difficulty_level": 2,
                "additional_context": {"test": "manual_tracking"}
            }
            
            response = await client.post(
                f"{BASE_URL}/api/v1/answers/error-patterns/track",
                json=error_data,
                headers=headers
            )
            
            if response.status_code == 200:
                print("✅ Manual error pattern tracking successful")
                result = response.json()
                print(f"   Error type: {result['error_pattern']['error_type']}")
                print(f"   Error count: {result['error_pattern']['error_count']}")
                print(f"   Subject: {result['error_pattern']['subject']}")
            else:
                print(f"❌ Manual error tracking failed: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"❌ Manual error tracking error: {e}")
        
        # Test error trend analysis
        print("\n4. Testing error trend analysis...")
        try:
            response = await client.get(
                f"{BASE_URL}/api/v1/answers/error-patterns/trend-analysis/math?days=30",
                headers=headers
            )
            
            if response.status_code == 200:
                print("✅ Error trend analysis successful")
                trend = response.json()
                print(f"   Total errors: {trend.get('total_errors', 0)}")
                print(f"   Error rate: {trend.get('error_rate', 0):.1f}%")
                print(f"   Trend: {trend.get('trend', 'unknown')}")
                
                weekly_breakdown = trend.get('weekly_breakdown', [])
                if weekly_breakdown:
                    print(f"   Weekly data points: {len(weekly_breakdown)}")
                
                frequent_errors = trend.get('most_frequent_errors', [])
                if frequent_errors:
                    print(f"   Most frequent error: {frequent_errors[0]['error_type']} ({frequent_errors[0]['count']} times)")
            else:
                print(f"❌ Error trend analysis failed: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"❌ Error trend analysis error: {e}")
        
        # Test intervention recommendations
        print("\n5. Testing intervention recommendations...")
        try:
            response = await client.get(
                f"{BASE_URL}/api/v1/answers/error-patterns/interventions/math",
                headers=headers
            )
            
            if response.status_code == 200:
                print("✅ Intervention recommendations successful")
                interventions = response.json()
                print(f"   Total recommendations: {interventions.get('total_recommendations', 0)}")
                
                for i, intervention in enumerate(interventions.get('interventions', [])[:3], 1):
                    print(f"   Intervention {i}:")
                    print(f"     Error type: {intervention.get('error_type')}")
                    print(f"     Priority: {intervention.get('priority')}")
                    print(f"     Type: {intervention.get('intervention_type')}")
                    print(f"     Description: {intervention.get('description', '')[:50]}...")
            else:
                print(f"❌ Intervention recommendations failed: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"❌ Intervention recommendations error: {e}")
        
        # Test similar students
        print("\n6. Testing similar students analysis...")
        try:
            response = await client.get(
                f"{BASE_URL}/api/v1/answers/error-patterns/similar-students?subject=math&limit=5",
                headers=headers
            )
            
            if response.status_code == 200:
                print("✅ Similar students analysis successful")
                similar = response.json()
                print(f"   Similar students found: {similar.get('total_found', 0)}")
                
                for student in similar.get('similar_students', [])[:3]:
                    print(f"   Student: {student.get('username')}")
                    print(f"     Common errors: {student.get('common_error_count')}")
                    print(f"     Similarity: {student.get('similarity_score', 0):.2f}")
                    print(f"     Learning style: {student.get('learning_style')}")
            else:
                print(f"❌ Similar students analysis failed: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"❌ Similar students analysis error: {e}")
        
        # Test class error analytics (if user has teacher permissions)
        print("\n7. Testing class error analytics...")
        try:
            response = await client.get(
                f"{BASE_URL}/api/v1/answers/error-patterns/class-analytics/math?days=30&min_students=1",
                headers=headers
            )
            
            if response.status_code == 200:
                print("✅ Class error analytics successful")
                analytics = response.json()
                print(f"   Total error types: {analytics.get('total_error_types', 0)}")
                print(f"   Significant errors: {analytics.get('significant_error_types', 0)}")
                print(f"   Affected students: {analytics.get('affected_students', 0)}")
                
                common_errors = analytics.get('most_common_errors', [])
                if common_errors:
                    print(f"   Most common error: {common_errors[0]['error_type']}")
                    print(f"     Affects {common_errors[0]['student_count']} students")
            elif response.status_code == 403:
                print("⚠️  Class analytics requires teacher permissions (expected for student user)")
            else:
                print(f"❌ Class error analytics failed: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"❌ Class error analytics error: {e}")
        
        # Test error analytics overview (if user has teacher permissions)
        print("\n8. Testing error analytics overview...")
        try:
            response = await client.get(
                f"{BASE_URL}/api/v1/answers/error-patterns/analytics/overview?days=30",
                headers=headers
            )
            
            if response.status_code == 200:
                print("✅ Error analytics overview successful")
                overview = response.json()
                
                overall = overview.get('overview', {})
                print(f"   Total error types: {overall.get('total_error_types', 0)}")
                print(f"   Affected students: {overall.get('affected_students', 0)}")
                
                by_subject = overview.get('by_subject', {})
                for subject, data in by_subject.items():
                    print(f"   {subject.title()}: {data.get('error_types', 0)} error types, {data.get('affected_students', 0)} students")
                
                critical_errors = overview.get('critical_errors', [])
                if critical_errors:
                    print(f"   Most critical error: {critical_errors[0]['error_type']} ({critical_errors[0]['subject']})")
            elif response.status_code == 403:
                print("⚠️  Analytics overview requires teacher permissions (expected for student user)")
            else:
                print(f"❌ Error analytics overview failed: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"❌ Error analytics overview error: {e}")
        
        # Test existing error analysis endpoint for comparison
        print("\n9. Testing existing error analysis endpoint...")
        try:
            response = await client.get(
                f"{BASE_URL}/api/v1/answers/error-analysis?subject=math",
                headers=headers
            )
            
            if response.status_code == 200:
                print("✅ Existing error analysis successful")
                errors = response.json()
                print(f"   Error patterns found: {len(errors)}")
                
                for error in errors[:3]:
                    print(f"   - {error.get('error_type')}: {error.get('frequency')} times")
                    print(f"     Description: {error.get('description', '')[:50]}...")
                    
                    recommendations = error.get('practice_recommendations', [])
                    if recommendations:
                        print(f"     Recommendation: {recommendations[0][:40]}...")
            else:
                print(f"❌ Existing error analysis failed: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"❌ Existing error analysis error: {e}")
        
        print("\n🎉 Error Pattern Analytics test completed!")

if __name__ == "__main__":
    asyncio.run(test_error_pattern_analytics())