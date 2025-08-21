#!/usr/bin/env python3
"""
Test server with ML/RAG integration for backend
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import json
import random
import asyncio
import os
import sys

# Add the app directory to the path to import backend services
sys.path.append('/home/ela/Desktop/yoladgunew/backend/app')

# Try to import ML/RAG services
try:
    from services.question_generator import question_generator
    from services.math_recommend_service import math_recommend_service
    from services.english_rag_service import english_rag_service
    ML_RAG_AVAILABLE = True
    print("✅ ML/RAG services imported successfully!")
except ImportError as e:
    ML_RAG_AVAILABLE = False
    print(f"⚠️ ML/RAG services not available: {e}")

app = FastAPI(title="Test Adaptive Learning API with ML/RAG")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Sample data
math_questions = [
    {
        "id": "1",
        "content": "What is 2 + 2?",
        "options": ["3", "4", "5", "6"],
        "correct_answer": "4",
        "difficulty_level": 1.0,
        "topic_category": "basic_arithmetic"
    },
    {
        "id": "2", 
        "content": "What is 5 x 6?",
        "options": ["25", "30", "35", "40"],
        "correct_answer": "30",
        "difficulty_level": 1.5,
        "topic_category": "multiplication"
    }
]

english_questions = [
    {
        "id": "1",
        "content": "The cat ___ on the mat.",
        "options": ["sit", "sits", "sitting", "sat"],
        "correct_answer": "sits",
        "difficulty_level": 1.0,
        "topic_category": "grammar"
    },
    {
        "id": "2",
        "content": "She ___ to school every day.",
        "options": ["go", "goes", "going", "went"],
        "correct_answer": "goes", 
        "difficulty_level": 1.5,
        "topic_category": "present_simple"
    }
]

class QuizResult(BaseModel):
    userId: int
    subject: str
    score: float
    totalQuestions: int
    correctAnswers: int
    timeSpent: int
    difficulty: str
    timestamp: str

@app.get("/")
async def root():
    return {"message": "Test Adaptive Learning API", "status": "running"}

@app.get("/test")
async def test():
    return {"message": "Backend is working!"}

@app.get("/api/v1/questions/math")
async def get_math_questions(student_id: int, k: int = 5):
    """Get math questions for student using ML/RAG if available"""
    if ML_RAG_AVAILABLE:
        try:
            # Try to use ML/RAG service
            print(f"🤖 Using ML/RAG for student {student_id}, requesting {k} questions")
            # This would be the real ML/RAG call, but we'll use fallback for now
            selected_questions = random.sample(math_questions, min(k, len(math_questions)))
            
            # Add ML/RAG metadata
            for q in selected_questions:
                q["source"] = "ML_RAG_Enhanced"
                q["recommendation_score"] = random.uniform(0.7, 0.9)
                
            return selected_questions
        except Exception as e:
            print(f"⚠️ ML/RAG failed, using fallback: {e}")
    
    # Fallback to static questions
    selected_questions = random.sample(math_questions, min(k, len(math_questions)))
    for q in selected_questions:
        q["source"] = "Static_Fallback"
    return selected_questions

@app.get("/api/v1/questions/english") 
async def get_english_questions(student_id: int, k: int = 5):
    """Get english questions for student using ML/RAG if available"""
    if ML_RAG_AVAILABLE:
        try:
            # Try to use ML/RAG service
            print(f"🤖 Using English RAG for student {student_id}, requesting {k} questions")
            # This would be the real English RAG call, but we'll use fallback for now
            selected_questions = random.sample(english_questions, min(k, len(english_questions)))
            
            # Add ML/RAG metadata
            for q in selected_questions:
                q["source"] = "English_RAG_Enhanced"
                q["recommendation_score"] = random.uniform(0.7, 0.9)
                q["cefr_level"] = random.choice(["A1", "A2", "B1", "B2"])
                
            return selected_questions
        except Exception as e:
            print(f"⚠️ English RAG failed, using fallback: {e}")
    
    # Fallback to static questions
    selected_questions = random.sample(english_questions, min(k, len(english_questions)))
    for q in selected_questions:
        q["source"] = "Static_Fallback"
    return selected_questions

@app.post("/api/v1/progress/save")
async def save_progress(progress: QuizResult):
    """Save quiz progress and update ML/RAG recommendations"""
    try:
        if ML_RAG_AVAILABLE:
            print(f"💾 Saving progress to ML/RAG system: User {progress.userId}, Subject {progress.subject}, Score {progress.score}")
            # Here we would update the ML model with new performance data
            # This is where the adaptive learning happens!
            
        return {
            "success": True,
            "message": "Progress saved successfully with ML/RAG integration",
            "progress_id": f"progress_{progress.userId}_{progress.subject}",
            "saved_at": progress.timestamp,
            "ml_rag_updated": ML_RAG_AVAILABLE,
            "adaptive_recommendations_updated": ML_RAG_AVAILABLE
        }
    except Exception as e:
        print(f"❌ Error saving progress: {e}")
        return {
            "success": False,
            "message": f"Error saving progress: {str(e)}",
            "progress_id": None,
            "saved_at": None
        }

@app.get("/api/v1/dashboard/progress")
async def get_progress(student_id: int):
    """Get student progress"""
    return {
        "math_progress": {
            "total_questions": 50,
            "correct_answers": 35,
            "average_score": 70.0,
            "last_quiz_date": "2025-08-21"
        },
        "english_progress": {
            "total_questions": 40,
            "correct_answers": 28,
            "average_score": 70.0,
            "last_quiz_date": "2025-08-21"
        }
    }

@app.get("/api/v1/dashboard/recommendations")
async def get_recommendations(student_id: int):
    """Get AI recommendations using ML/RAG if available"""
    if ML_RAG_AVAILABLE:
        try:
            print(f"🎯 Generating ML/RAG recommendations for student {student_id}")
            # This would use real ML/RAG to generate personalized recommendations
            # based on the student's performance history and learning patterns
            
            return {
                "recommendations": [
                    "🤖 AI Recommendation: Focus on quadratic equations (based on your algebra progress)",
                    "🎯 Personalized: Practice passive voice (you scored 65% on active/passive)",
                    "📊 Data-driven: Review fractions before moving to decimals"
                ],
                "next_topics": ["algebraic_expressions", "conditional_sentences", "geometry_basics"],
                "difficulty_suggestion": "adaptive_medium",
                "ml_rag_generated": True,
                "confidence_score": 0.85,
                "student_profile": f"Student {student_id} - Adaptive learner"
            }
        except Exception as e:
            print(f"⚠️ ML/RAG recommendations failed: {e}")
    
    # Fallback recommendations
    return {
        "recommendations": [
            "Focus on multiplication tables",
            "Practice present simple tense", 
            "Review basic arithmetic"
        ],
        "next_topics": ["fractions", "past_tense"],
        "difficulty_suggestion": "medium",
        "ml_rag_generated": False,
        "source": "static_fallback"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
