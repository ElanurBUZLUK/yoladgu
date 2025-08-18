#!/usr/bin/env python3
"""
Question service test script
"""
import asyncio
import httpx
import json

BASE_URL = "http://localhost:8000"

async def test_question_service():
    """Test question service and APIs"""
    
    async with httpx.AsyncClient() as client:
        print("🧪 Testing Question Service...")
        
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
        
        # Test question recommendation
        print("\n2. Testing question recommendation...")
        try:
            recommendation_data = {
                "subject": "math",
                "user_level": 2,
                "preferred_difficulty": 2,
                "limit": 5,
                "exclude_recent": True,
                "learning_style": "mixed"
            }
            
            response = await client.post(
                f"{BASE_URL}/api/v1/math/questions/recommend", 
                json=recommendation_data, 
                headers=headers
            )
            
            if response.status_code == 200:
                print("✅ Question recommendation successful")
                recommendations = response.json()
                print(f"   Recommended questions: {len(recommendations.get('questions', []))}")
                print(f"   Reason: {recommendations.get('recommendation_reason', 'N/A')}")
                print(f"   User level: {recommendations.get('user_level', 'N/A')}")
                print(f"   Total available: {recommendations.get('total_available', 'N/A')}")
            else:
                print(f"❌ Question recommendation failed: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"❌ Question recommendation error: {e}")
        
        # Test question search
        print("\n3. Testing question search...")
        try:
            search_params = {
                "subject": "math",
                "difficulty_level": 2,
                "limit": 3
            }
            
            response = await client.get(
                f"{BASE_URL}/api/v1/math/questions/search",
                params=search_params,
                headers=headers
            )
            
            if response.status_code == 200:
                print("✅ Question search successful")
                search_results = response.json()
                print(f"   Found questions: {len(search_results.get('questions', []))}")
                
                # Show first question details
                if search_results.get('questions'):
                    first_q = search_results['questions'][0]
                    print(f"   First question: {first_q.get('content', '')[:50]}...")
                    print(f"   Difficulty: {first_q.get('difficulty_level')}")
                    print(f"   Type: {first_q.get('question_type')}")
            else:
                print(f"❌ Question search failed: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"❌ Question search error: {e}")
        
        # Test question pool
        print("\n4. Testing question pool...")
        try:
            response = await client.get(
                f"{BASE_URL}/api/v1/math/questions/pool?subject=math&difficulty_level=2",
                headers=headers
            )
            
            if response.status_code == 200:
                print("✅ Question pool retrieval successful")
                pool_data = response.json()
                print(f"   Pool size: {pool_data.get('pool_size', 0)}")
                print(f"   Subject: {pool_data.get('subject')}")
                print(f"   Difficulty level: {pool_data.get('difficulty_level')}")
                print(f"   Last updated: {pool_data.get('last_updated', 'N/A')}")
            else:
                print(f"❌ Question pool failed: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"❌ Question pool error: {e}")
        
        # Test question statistics
        print("\n5. Testing question statistics...")
        try:
            response = await client.get(f"{BASE_URL}/api/v1/math/questions/stats", headers=headers)
            
            if response.status_code == 200:
                print("✅ Question statistics successful")
                stats = response.json()
                print(f"   Total questions: {stats.get('total_questions', 0)}")
                print(f"   Average difficulty: {stats.get('average_difficulty', 0):.1f}")
                
                by_subject = stats.get('by_subject', {})
                print(f"   Math questions: {by_subject.get('math', 0)}")
                
                by_difficulty = stats.get('by_difficulty', {})
                for level in range(1, 6):
                    count = by_difficulty.get(str(level), 0)
                    print(f"   Level {level}: {count} questions")
            else:
                print(f"❌ Question statistics failed: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"❌ Question statistics error: {e}")
        
        # Test question creation (admin functionality)
        print("\n6. Testing question creation...")
        try:
            new_question = {
                "subject": "math",
                "content": "What is 10 + 15?",
                "question_type": "multiple_choice",
                "difficulty_level": 1,
                "topic_category": "addition",
                "correct_answer": "25",
                "options": ["20", "23", "25", "30"],
                "source_type": "manual",
                "question_metadata": {
                    "estimated_time": 30,
                    "tags": ["test", "addition"]
                }
            }
            
            response = await client.post(
                f"{BASE_URL}/api/v1/math/questions/",
                json=new_question,
                headers=headers
            )
            
            if response.status_code == 201:
                print("✅ Question creation successful")
                created_question = response.json()
                print(f"   Created question ID: {created_question.get('id')}")
                print(f"   Content: {created_question.get('content')}")
            else:
                print(f"❌ Question creation failed: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"❌ Question creation error: {e}")
        
        # Test difficulty adjustment
        print("\n7. Testing difficulty adjustment...")
        try:
            # First get a question ID from search
            search_response = await client.get(
                f"{BASE_URL}/api/v1/math/questions/search?subject=math&limit=1",
                headers=headers
            )
            
            if search_response.status_code == 200:
                search_data = search_response.json()
                if search_data.get('questions'):
                    question_id = search_data['questions'][0]['id']
                    
                    adjustment_data = {
                        "new_difficulty": 3,
                        "reason": "Question seems too easy for level 2",
                        "adjusted_by": "test_user"
                    }
                    
                    response = await client.put(
                        f"{BASE_URL}/api/v1/math/questions/{question_id}/difficulty",
                        json=adjustment_data,
                        headers=headers
                    )
                    
                    if response.status_code == 200:
                        print("✅ Difficulty adjustment successful")
                        adjustment = response.json()
                        print(f"   Old difficulty: {adjustment.get('old_difficulty')}")
                        print(f"   New difficulty: {adjustment.get('new_difficulty')}")
                        print(f"   Reason: {adjustment.get('reason')}")
                    else:
                        print(f"❌ Difficulty adjustment failed: {response.status_code} - {response.text}")
                else:
                    print("⚠️  No questions found for difficulty adjustment test")
            else:
                print("⚠️  Could not get question for difficulty adjustment test")
        except Exception as e:
            print(f"❌ Difficulty adjustment error: {e}")
        
        print("\n🎉 Question service test completed!")

if __name__ == "__main__":
    asyncio.run(test_question_service())