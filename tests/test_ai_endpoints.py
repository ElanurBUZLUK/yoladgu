import pytest
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


@pytest.mark.asyncio
async def test_generate_hint_endpoint():
    """Test generate hint endpoint"""
    with patch('app.api.v1.endpoints.ai.llm_service') as mock_llm:
        mock_llm.generate_question_hint.return_value = "Use the formula: a² + b² = c²"
        
        response = client.post(
            "/api/v1/ai/generate-hint",
            json={
                "question_content": "What is the hypotenuse?",
                "subject": "mathematics"
            },
            headers={"Authorization": "Bearer test_token"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["hint"] == "Use the formula: a² + b² = c²"


@pytest.mark.asyncio
async def test_generate_explanation_endpoint():
    """Test generate explanation endpoint"""
    with patch('app.api.v1.endpoints.ai.llm_service') as mock_llm:
        mock_llm.generate_question_explanation.return_value = "Step 1: Apply the formula..."
        
        response = client.post(
            "/api/v1/ai/generate-explanation",
            json={
                "question_content": "What is 2+2?",
                "correct_answer": "4",
                "subject": "mathematics"
            },
            headers={"Authorization": "Bearer test_token"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["explanation"] == "Step 1: Apply the formula..."


@pytest.mark.asyncio
async def test_generate_feedback_endpoint():
    """Test generate feedback endpoint"""
    with patch('app.api.v1.endpoints.ai.llm_service') as mock_llm:
        mock_llm.generate_ai_feedback.return_value = "Great job! You're doing well."
        
        response = client.post(
            "/api/v1/ai/generate-feedback",
            json={
                "is_correct": True,
                "question_topic": "algebra",
                "student_level": 3
            },
            headers={"Authorization": "Bearer test_token"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["feedback"] == "Great job! You're doing well."


@pytest.mark.asyncio
async def test_generate_study_recommendation_endpoint():
    """Test generate study recommendation endpoint"""
    with patch('app.api.v1.endpoints.ai.llm_service') as mock_llm:
        mock_llm.generate_study_recommendation.return_value = "Focus on algebra fundamentals."
        
        response = client.post(
            "/api/v1/ai/generate-study-recommendation",
            json={
                "weak_topics": ["algebra"],
                "strong_topics": ["geometry"]
            },
            headers={"Authorization": "Bearer test_token"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["recommendation"] == "Focus on algebra fundamentals."


@pytest.mark.asyncio
async def test_analyze_question_difficulty_endpoint():
    """Test analyze question difficulty endpoint"""
    with patch('app.api.v1.endpoints.ai.llm_service') as mock_llm:
        mock_llm.analyze_question_difficulty.return_value = {
            "difficulty_level": 3,
            "required_knowledge": ["algebra"],
            "solution_steps": 4,
            "grade_level": "9-10",
            "explanation": "Medium difficulty"
        }
        
        response = client.post(
            "/api/v1/ai/analyze-question-difficulty",
            json={
                "question_content": "Solve for x: 2x + 5 = 13",
                "subject": "mathematics"
            },
            headers={"Authorization": "Bearer test_token"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["analysis"]["difficulty_level"] == 3


@pytest.mark.asyncio
async def test_adaptive_hint_endpoint():
    """Test adaptive hint endpoint"""
    with patch('app.api.v1.endpoints.ai.recommendation_service') as mock_rec:
        mock_rec.get_adaptive_hint.return_value = "Adaptive hint for struggling student"
        
        response = client.post(
            "/api/v1/ai/adaptive-hint",
            json={
                "question_id": 1,
                "student_id": 1
            },
            headers={"Authorization": "Bearer test_token"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["hint"] == "Adaptive hint for struggling student"


@pytest.mark.asyncio
async def test_contextual_explanation_endpoint():
    """Test contextual explanation endpoint"""
    with patch('app.api.v1.endpoints.ai.recommendation_service') as mock_rec:
        mock_rec.get_contextual_explanation.return_value = "I see you chose B, but the correct answer is A because..."
        
        response = client.post(
            "/api/v1/ai/contextual-explanation",
            json={
                "question_id": 1,
                "student_id": 1,
                "student_answer": "B"
            },
            headers={"Authorization": "Bearer test_token"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["explanation"] == "I see you chose B, but the correct answer is A because..."


@pytest.mark.asyncio
async def test_batch_enrich_questions_endpoint():
    """Test batch enrich questions endpoint"""
    with patch('app.api.v1.endpoints.ai.llm_service') as mock_llm:
        mock_llm.analyze_question_difficulty.return_value = {
            "difficulty_level": 3,
            "required_knowledge": ["algebra"],
            "solution_steps": 4,
            "grade_level": "9-10",
            "explanation": "Medium difficulty"
        }
        mock_llm.generate_question_hint.return_value = "Use the formula"
        mock_llm.generate_question_explanation.return_value = "Step by step solution"
        
        with patch('app.api.v1.endpoints.ai.question_ingestion_service') as mock_ingestion:
            mock_ingestion._get_subject_id.return_value = 1
            
            with patch('app.api.v1.endpoints.ai.db') as mock_db:
                mock_question = Mock()
                mock_question.content = "Test question"
                mock_question.correct_answer = "A"
                mock_question.difficulty_level = 2
                mock_question.hint = None
                mock_question.explanation = None
                
                mock_db.query.return_value.filter.return_value.all.return_value = [mock_question]
                
                response = client.post(
                    "/api/v1/ai/batch-enrich-questions",
                    json={
                        "subject": "mathematics",
                        "topic": "algebra"
                    },
                    headers={"Authorization": "Bearer test_token"}
                )
                
                assert response.status_code == 200
                data = response.json()
                assert data["success"] is True
                assert data["enriched_count"] == 1


@pytest.mark.asyncio
async def test_llm_status_endpoint():
    """Test LLM status endpoint"""
    with patch('app.api.v1.endpoints.ai.llm_service') as mock_llm:
        mock_llm._get_api_key.side_effect = ['test_openai_key', 'test_hf_key']
        mock_llm.provider.value = 'openai'
        
        response = client.get(
            "/api/v1/ai/llm-status",
            headers={"Authorization": "Bearer test_token"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["openai_configured"] is True
        assert data["huggingface_configured"] is True
        assert data["status"] == "ready"


@pytest.mark.asyncio
async def test_ingest_from_website_endpoint():
    """Test ingest from website endpoint"""
    with patch('app.api.v1.endpoints.ai.question_ingestion_service') as mock_ingestion:
        mock_ingestion.scrape_from_website.return_value = [
            {
                'content': 'Test question',
                'options': ['A', 'B', 'C', 'D'],
                'correct_answer': 'A',
                'difficulty_level': 2,
                'subject_id': 1,
                'question_type': 'multiple_choice',
                'tags': ['math'],
                'created_by': 1,
                'is_active': True
            }
        ]
        mock_ingestion.save_questions_to_database.return_value = 1
        
        response = client.post(
            "/api/v1/ai/ingest-from-website",
            json={
                "url": "https://example.com",
                "subject": "mathematics",
                "topic": "algebra"
            },
            headers={"Authorization": "Bearer test_token"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["saved_count"] == 1


@pytest.mark.asyncio
async def test_ingest_from_csv_endpoint():
    """Test ingest from CSV endpoint"""
    with patch('app.api.v1.endpoints.ai.question_ingestion_service') as mock_ingestion:
        mock_ingestion.ingest_from_csv.return_value = [
            {
                'content': 'Test question',
                'options': ['A', 'B', 'C', 'D'],
                'correct_answer': 'A',
                'difficulty_level': 2,
                'subject_id': 1,
                'question_type': 'multiple_choice',
                'tags': ['math'],
                'created_by': 1,
                'is_active': True
            }
        ]
        mock_ingestion.save_questions_to_database.return_value = 1
        
        response = client.post(
            "/api/v1/ai/ingest-from-csv",
            json={
                "file_path": "questions.csv",
                "subject": "mathematics"
            },
            headers={"Authorization": "Bearer test_token"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["saved_count"] == 1


@pytest.mark.asyncio
async def test_error_handling():
    """Test error handling in AI endpoints"""
    with patch('app.api.v1.endpoints.ai.llm_service') as mock_llm:
        mock_llm.generate_question_hint.side_effect = Exception("API Error")
        
        response = client.post(
            "/api/v1/ai/generate-hint",
            json={
                "question_content": "Test question",
                "subject": "mathematics"
            },
            headers={"Authorization": "Bearer test_token"}
        )
        
        assert response.status_code == 500
        data = response.json()
        assert "API Error" in data["detail"] 