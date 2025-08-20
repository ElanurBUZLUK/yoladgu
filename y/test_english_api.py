#!/usr/bin/env python3
"""
Comprehensive English API test script
"""
import asyncio
import httpx
import json

BASE_URL = "http://localhost:8000"

async def test_english_api():
    """Test all English API endpoints"""
    
    async with httpx.AsyncClient() as client:
        print("🧪 Testing English API Endpoints...")
        
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
        
        # Test English question generation
        print("\n2. Testing English question generation...")
        try:
            generation_data = {
                "difficulty_level": 2,
                "question_type": "multiple_choice",
                "topic_focus": "past_tense",
                "error_patterns": ["past_tense_error"],
                "learning_style": "mixed",
                "count": 2
            }
            
            response = await client.post(
                f"{BASE_URL}/api/v1/english/questions/generate?save_to_db=true", 
                json=generation_data, 
                headers=headers
            )
            
            if response.status_code == 200:
                print("✅ English question generation successful")
                generated = response.json()
                print(f"   Generated questions: {len(generated)}")
                
                for i, q in enumerate(generated, 1):
                    print(f"   Question {i}:")
                    print(f"     Method: {q.get('generation_method')}")
                    print(f"     Quality: {q.get('quality_score', 0):.2f}")
                    print(f"     Saved to DB: {q.get('saved_to_database')}")
                    
                    question_data = q.get('generated_question', {})
                    print(f"     Content: {question_data.get('content', '')[:50]}...")
            else:
                print(f"❌ English question generation failed: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"❌ English question generation error: {e}")
        
        # Test English question recommendation
        print("\n3. Testing English question recommendation...")
        try:
            recommendation_data = {
                "subject": "english",
                "user_level": 2,
                "preferred_difficulty": 2,
                "limit": 3,
                "exclude_recent": True,
                "learning_style": "mixed",
                "error_patterns": ["past_tense_error"]
            }
            
            response = await client.post(
                f"{BASE_URL}/api/v1/english/questions/recommend", 
                json=recommendation_data, 
                headers=headers
            )
            
            if response.status_code == 200:
                print("✅ English question recommendation successful")
                recommendations = response.json()
                print(f"   Recommended questions: {len(recommendations.get('questions', []))}")
                print(f"   Reason: {recommendations.get('recommendation_reason', 'N/A')}")
                print(f"   User level: {recommendations.get('user_level', 'N/A')}")
                
                # Show first question
                if recommendations.get('questions'):
                    first_q = recommendations['questions'][0]
                    print(f"   First question: {first_q.get('content', '')[:50]}...")
                    print(f"   Source: {first_q.get('source_type')}")
            else:
                print(f"❌ English question recommendation failed: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"❌ English question recommendation error: {e}")
        
        # Test English question search
        print("\n4. Testing English question search...")
        try:
            search_params = {
                "difficulty_level": 2,
                "topic_category": "past_tense",
                "limit": 3
            }
            
            response = await client.get(
                f"{BASE_URL}/api/v1/english/questions/search",
                params=search_params,
                headers=headers
            )
            
            if response.status_code == 200:
                print("✅ English question search successful")
                search_results = response.json()
                print(f"   Found questions: {len(search_results.get('questions', []))}")
                print(f"   Search criteria: {search_results.get('search_criteria', {})}")
                
                # Show question details
                if search_results.get('questions'):
                    first_q = search_results['questions'][0]
                    print(f"   First question topic: {first_q.get('topic_category')}")
                    print(f"   Question type: {first_q.get('question_type')}")
            else:
                print(f"❌ English question search failed: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"❌ English question search error: {e}")
        
        # Test English topics
        print("\n5. Testing English topics...")
        try:
            response = await client.get(f"{BASE_URL}/api/v1/english/questions/topics", headers=headers)
            
            if response.status_code == 200:
                print("✅ English topics successful")
                topics_data = response.json()
                print(f"   Total topics: {topics_data.get('total_topics', 0)}")
                print(f"   Existing topics: {len(topics_data.get('existing_topics', []))}")
                print(f"   Generatable topics: {len(topics_data.get('generatable_topics', []))}")
                
                # Show some topics
                all_topics = topics_data.get('all_topics', [])
                print(f"   Sample topics: {all_topics[:5]}...")
            else:
                print(f"❌ English topics failed: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"❌ English topics error: {e}")
        
        # Test question validation
        print("\n6. Testing question validation...")
        try:
            sample_question = {
                "content": "Choose the correct form: I ____ to school yesterday.",
                "question_type": "multiple_choice",
                "correct_answer": "went",
                "options": ["go", "went", "going", "goes"],
                "explanation": "Use past tense for yesterday."
            }
            
            response = await client.post(
                f"{BASE_URL}/api/v1/english/questions/validate",
                json=sample_question,
                headers=headers
            )
            
            if response.status_code == 200:
                print("✅ Question validation successful")
                validation = response.json()
                validation_result = validation.get('validation_result', {})
                print(f"   Is valid: {validation_result.get('is_valid')}")
                print(f"   Quality score: {validation_result.get('quality_score', 0):.2f}")
                print(f"   Errors: {len(validation_result.get('errors', []))}")
                print(f"   Warnings: {len(validation_result.get('warnings', []))}")
                
                recommendations = validation.get('recommendations', [])
                if recommendations:
                    print(f"   Recommendations: {recommendations[0]}")
            else:
                print(f"❌ Question validation failed: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"❌ Question validation error: {e}")
        
        # Test generation statistics
        print("\n7. Testing generation statistics...")
        try:
            response = await client.get(f"{BASE_URL}/api/v1/english/questions/generation-stats", headers=headers)
            
            if response.status_code == 200:
                print("✅ Generation statistics successful")
                stats = response.json()
                print(f"   Total English questions: {stats.get('total_english_questions', 0)}")
                print(f"   Generated questions: {stats.get('generated_questions', 0)}")
                print(f"   Manual questions: {stats.get('manual_questions', 0)}")
                print(f"   Generation percentage: {stats.get('generation_percentage', 0):.1f}%")
                
                by_difficulty = stats.get('generated_by_difficulty', {})
                for level in range(1, 6):
                    count = by_difficulty.get(str(level), 0)
                    if count > 0:
                        print(f"   Generated Level {level}: {count}")
            else:
                print(f"❌ Generation statistics failed: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"❌ Generation statistics error: {e}")
        
        # Test manual question creation
        print("\n8. Testing manual question creation...")
        try:
            new_question = {
                "subject": "english",
                "content": "What is the opposite of 'hot'?",
                "question_type": "multiple_choice",
                "difficulty_level": 1,
                "topic_category": "vocabulary",
                "correct_answer": "cold",
                "options": ["warm", "cold", "cool", "freezing"],
                "source_type": "manual",
                "question_metadata": {
                    "estimated_time": 30,
                    "tags": ["test", "vocabulary", "opposites"]
                }
            }
            
            response = await client.post(
                f"{BASE_URL}/api/v1/english/questions/",
                json=new_question,
                headers=headers
            )
            
            if response.status_code == 201:
                print("✅ Manual question creation successful")
                created_question = response.json()
                print(f"   Created question ID: {created_question.get('id')}")
                print(f"   Content: {created_question.get('content')}")
                print(f"   Topic: {created_question.get('topic_category')}")
            else:
                print(f"❌ Manual question creation failed: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"❌ Manual question creation error: {e}")
        
        # Test get specific English question
        print("\n9. Testing get specific English question...")
        try:
            # First get a question ID from search
            search_response = await client.get(
                f"{BASE_URL}/api/v1/english/questions/search?limit=1",
                headers=headers
            )
            
            if search_response.status_code == 200:
                search_data = search_response.json()
                if search_data.get('questions'):
                    question_id = search_data['questions'][0]['id']
                    
                    response = await client.get(
                        f"{BASE_URL}/api/v1/english/questions/{question_id}",
                        headers=headers
                    )
                    
                    if response.status_code == 200:
                        print("✅ Get specific English question successful")
                        question = response.json()
                        print(f"   Question ID: {question.get('id')}")
                        print(f"   Content: {question.get('content', '')[:50]}...")
                        print(f"   Difficulty: {question.get('difficulty_level')}")
                        print(f"   Topic: {question.get('topic_category')}")
                    else:
                        print(f"❌ Get specific English question failed: {response.status_code} - {response.text}")
                else:
                    print("⚠️  No English questions found for specific question test")
            else:
                print("⚠️  Could not get English question for specific question test")
        except Exception as e:
            print(f"❌ Get specific English question error: {e}")
        
        # Test LLM Gateway health (if available)
        print("\n10. Testing LLM integration health...")
        try:
            # This is a simple test to see if LLM services are responsive
            simple_generation = {
                "difficulty_level": 1,
                "question_type": "multiple_choice",
                "count": 1
            }
            
            response = await client.post(
                f"{BASE_URL}/api/v1/english/questions/generate", 
                json=simple_generation, 
                headers=headers,
                timeout=10.0  # Short timeout for health check
            )
            
            if response.status_code == 200:
                generated = response.json()
                if generated and len(generated) > 0:
                    method = generated[0].get('generation_method', 'unknown')
                    print(f"✅ LLM integration healthy (method: {method})")
                else:
                    print("⚠️  LLM integration responded but no questions generated")
            else:
                print(f"⚠️  LLM integration may have issues: {response.status_code}")
        except Exception as e:
            print(f"⚠️  LLM integration test error: {e}")
        
        print("\n🎉 English API test completed!")

if __name__ == "__main__":
    asyncio.run(test_english_api())