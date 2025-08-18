#!/usr/bin/env python3
"""
Comprehensive Math API test script
"""
import asyncio
import httpx
import json

BASE_URL = "http://localhost:8000"

async def test_math_api():
    """Test all math API endpoints"""
    
    async with httpx.AsyncClient() as client:
        print("🧪 Testing Math API Endpoints...")
        
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
        
        # Test math question recommendation
        print("\n2. Testing math question recommendation...")
        try:
            recommendation_data = {
                "subject": "math",
                "user_level": 2,
                "preferred_difficulty": 2,
                "limit": 3,
                "exclude_recent": True,
                "learning_style": "mixed"
            }
            
            response = await client.post(
                f"{BASE_URL}/api/v1/math/questions/recommend", 
                json=recommendation_data, 
                headers=headers
            )
            
            if response.status_code == 200:
                print("✅ Math question recommendation successful")
                recommendations = response.json()
                print(f"   Recommended questions: {len(recommendations.get('questions', []))}")
                print(f"   Reason: {recommendations.get('recommendation_reason', 'N/A')}")
                print(f"   User level: {recommendations.get('user_level', 'N/A')}")
                
                # Show first question
                if recommendations.get('questions'):
                    first_q = recommendations['questions'][0]
                    print(f"   First question: {first_q.get('content', '')[:50]}...")
            else:
                print(f"❌ Math question recommendation failed: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"❌ Math question recommendation error: {e}")
        
        # Test math question search
        print("\n3. Testing math question search...")
        try:
            search_params = {
                "difficulty_level": 2,
                "limit": 3
            }
            
            response = await client.get(
                f"{BASE_URL}/api/v1/math/questions/search",
                params=search_params,
                headers=headers
            )
            
            if response.status_code == 200:
                print("✅ Math question search successful")
                search_results = response.json()
                print(f"   Found questions: {len(search_results.get('questions', []))}")
                print(f"   Search criteria: {search_results.get('search_criteria', {})}")
            else:
                print(f"❌ Math question search failed: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"❌ Math question search error: {e}")
        
        # Test questions by level
        print("\n4. Testing questions by level...")
        try:
            response = await client.get(
                f"{BASE_URL}/api/v1/math/questions/by-level/2?limit=3",
                headers=headers
            )
            
            if response.status_code == 200:
                print("✅ Questions by level successful")
                level_data = response.json()
                print(f"   Level: {level_data.get('level')}")
                print(f"   Questions count: {level_data.get('count')}")
                print(f"   User current level: {level_data.get('user_current_level')}")
            else:
                print(f"❌ Questions by level failed: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"❌ Questions by level error: {e}")
        
        # Test math question pool
        print("\n5. Testing math question pool...")
        try:
            response = await client.get(
                f"{BASE_URL}/api/v1/math/questions/pool?difficulty_level=2",
                headers=headers
            )
            
            if response.status_code == 200:
                print("✅ Math question pool successful")
                pool_data = response.json()
                print(f"   Pool size: {pool_data.get('pool_size', 0)}")
                print(f"   Subject: {pool_data.get('subject')}")
                print(f"   Difficulty level: {pool_data.get('difficulty_level')}")
            else:
                print(f"❌ Math question pool failed: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"❌ Math question pool error: {e}")
        
        # Test math topics
        print("\n6. Testing math topics...")
        try:
            response = await client.get(f"{BASE_URL}/api/v1/math/questions/topics", headers=headers)
            
            if response.status_code == 200:
                print("✅ Math topics successful")
                topics_data = response.json()
                print(f"   Total topics: {topics_data.get('total_topics', 0)}")
                print(f"   All topics: {topics_data.get('all_topics', [])[:5]}...")  # Show first 5
                
                topics_by_level = topics_data.get('topics_by_level', {})
                for level in ['1', '2', '3']:
                    if level in topics_by_level:
                        print(f"   Level {level} topics: {len(topics_by_level[level])}")
            else:
                print(f"❌ Math topics failed: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"❌ Math topics error: {e}")
        
        # Test difficulty distribution
        print("\n7. Testing difficulty distribution...")
        try:
            response = await client.get(f"{BASE_URL}/api/v1/math/questions/difficulty-distribution", headers=headers)
            
            if response.status_code == 200:
                print("✅ Difficulty distribution successful")
                dist_data = response.json()
                print(f"   Total questions: {dist_data.get('total_questions', 0)}")
                print(f"   User level: {dist_data.get('user_level')}")
                
                distribution = dist_data.get('distribution', {})
                for level in range(1, 6):
                    count = distribution.get(str(level), 0)
                    print(f"   Level {level}: {count} questions")
            else:
                print(f"❌ Difficulty distribution failed: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"❌ Difficulty distribution error: {e}")
        
        # Test random question
        print("\n8. Testing random question...")
        try:
            response = await client.get(
                f"{BASE_URL}/api/v1/math/questions/random/2",
                headers=headers
            )
            
            if response.status_code == 200:
                print("✅ Random question successful")
                random_data = response.json()
                question = random_data.get('question', {})
                print(f"   Question content: {question.get('content', '')[:50]}...")
                print(f"   Difficulty: {random_data.get('difficulty_level')}")
                print(f"   Type: {question.get('question_type')}")
            else:
                print(f"❌ Random question failed: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"❌ Random question error: {e}")
        
        # Test question statistics
        print("\n9. Testing question statistics...")
        try:
            response = await client.get(f"{BASE_URL}/api/v1/math/questions/stats", headers=headers)
            
            if response.status_code == 200:
                print("✅ Question statistics successful")
                stats = response.json()
                print(f"   Total questions: {stats.get('total_questions', 0)}")
                print(f"   Average difficulty: {stats.get('average_difficulty', 0):.1f}")
                
                by_subject = stats.get('by_subject', {})
                print(f"   Math questions: {by_subject.get('math', 0)}")
                
                by_type = stats.get('by_type', {})
                for qtype, count in by_type.items():
                    if count > 0:
                        print(f"   {qtype}: {count}")
            else:
                print(f"❌ Question statistics failed: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"❌ Question statistics error: {e}")
        
        # Test get specific question (if we have questions)
        print("\n10. Testing get specific question...")
        try:
            # First get a question ID from search
            search_response = await client.get(
                f"{BASE_URL}/api/v1/math/questions/search?limit=1",
                headers=headers
            )
            
            if search_response.status_code == 200:
                search_data = search_response.json()
                if search_data.get('questions'):
                    question_id = search_data['questions'][0]['id']
                    
                    response = await client.get(
                        f"{BASE_URL}/api/v1/math/questions/{question_id}",
                        headers=headers
                    )
                    
                    if response.status_code == 200:
                        print("✅ Get specific question successful")
                        question = response.json()
                        print(f"   Question ID: {question.get('id')}")
                        print(f"   Content: {question.get('content', '')[:50]}...")
                        print(f"   Difficulty: {question.get('difficulty_level')}")
                    else:
                        print(f"❌ Get specific question failed: {response.status_code} - {response.text}")
                else:
                    print("⚠️  No questions found for specific question test")
            else:
                print("⚠️  Could not get question for specific question test")
        except Exception as e:
            print(f"❌ Get specific question error: {e}")
        
        # Test new RAG endpoints
        print("\n12. Testing new RAG endpoints...")
        
        # Test generate endpoint
        try:
            generate_data = {
                "topic": "quadratic equations",
                "difficulty_level": 3,
                "question_type": "multiple_choice",
                "n": 2
            }
            
            response = await client.post(
                f"{BASE_URL}/api/v1/math/rag/generate",
                json=generate_data,
                headers=headers
            )
            
            if response.status_code == 200:
                print("✅ Math RAG generate successful")
                generate_result = response.json()
                print(f"   Generated questions: {len(generate_result.get('items', []))}")
            else:
                print(f"❌ Math RAG generate failed: {response.status_code}")
        except Exception as e:
            print(f"❌ Math RAG generate error: {e}")
        
        # Test solve endpoint
        try:
            solve_data = {
                "problem": "Solve for x: 2x + 5 = 13",
                "show_steps": True
            }
            
            response = await client.post(
                f"{BASE_URL}/api/v1/math/rag/solve",
                json=solve_data,
                headers=headers
            )
            
            if response.status_code == 200:
                print("✅ Math RAG solve successful")
                solve_result = response.json()
                print(f"   Solution: {solve_result.get('solution', '')[:50]}...")
            else:
                print(f"❌ Math RAG solve failed: {response.status_code}")
        except Exception as e:
            print(f"❌ Math RAG solve error: {e}")
        
        # Test check endpoint
        try:
            check_data = {
                "question": "What is 2 + 2?",
                "user_answer": "4",
                "answer_key": "4",
                "require_explanation": True
            }
            
            response = await client.post(
                f"{BASE_URL}/api/v1/math/rag/check",
                json=check_data,
                headers=headers
            )
            
            if response.status_code == 200:
                print("✅ Math RAG check successful")
                check_result = response.json()
                print(f"   Correct: {check_result.get('correct', False)}")
            else:
                print(f"❌ Math RAG check failed: {response.status_code}")
        except Exception as e:
            print(f"❌ Math RAG check error: {e}")

        # Test grounded query endpoint
        try:
            grounded_data = {
                "query": "How do I solve quadratic equations?",
                "namespace": "math",
                "max_results": 3,
                "min_similarity": 0.6
            }
            
            response = await client.post(
                f"{BASE_URL}/api/v1/math/rag/grounded-query",
                json=grounded_data,
                headers=headers
            )
            
            if response.status_code == 200:
                print("✅ Math RAG grounded query successful")
                grounded_result = response.json()
                print(f"   Answerable: {grounded_result.get('answerable', False)}")
                print(f"   Confidence: {grounded_result.get('confidence', 0.0)}")
                print(f"   Citations: {len(grounded_result.get('citations', []))}")
            else:
                print(f"❌ Math RAG grounded query failed: {response.status_code}")
        except Exception as e:
            print(f"❌ Math RAG grounded query error: {e}")

        # Test validation stats endpoint
        try:
            response = await client.get(
                f"{BASE_URL}/api/v1/math/rag/validation-stats",
                headers=headers
            )
            
            if response.status_code == 200:
                print("✅ Math RAG validation stats successful")
                stats_result = response.json()
                print(f"   Validation settings: {stats_result.get('validation_settings', {})}")
            else:
                print(f"❌ Math RAG validation stats failed: {response.status_code}")
        except Exception as e:
            print(f"❌ Math RAG validation stats error: {e}")
        
        print("\n🎉 Math API test completed!")

if __name__ == "__main__":
    asyncio.run(test_math_api())