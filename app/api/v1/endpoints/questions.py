from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List, Optional
from app.db.database import get_db
from app.db.models import Question, Subject, Skill, QuestionSkill, StudentResponse
from app.schemas.question import QuestionCreate, QuestionUpdate, QuestionResponse, QuestionRecommendation
from app.services.recommendation_service import recommendation_service
from app.crud.user import get_current_user
from app.db.models import User
import time

router = APIRouter()

@router.get("/questions/", response_model=List[QuestionResponse])
def get_questions(
    skip: int = 0,
    limit: int = 100,
    subject_id: Optional[int] = None,
    difficulty_level: Optional[int] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get all questions with optional filtering"""
    query = db.query(Question).filter(Question.is_active == True)
    
    if subject_id:
        query = query.filter(Question.subject_id == subject_id)
    
    if difficulty_level:
        query = query.filter(Question.difficulty_level == difficulty_level)
    
    questions = query.offset(skip).limit(limit).all()
    return questions

@router.get("/questions/{question_id}", response_model=QuestionResponse)
def get_question(
    question_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get a specific question by ID"""
    question = db.query(Question).filter(Question.id == question_id).first()
    if not question:
        raise HTTPException(status_code=404, detail="Question not found")
    return question

@router.post("/questions/", response_model=QuestionResponse)
def create_question(
    question: QuestionCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Create a new question (teachers and admins only)"""
    if current_user.role.value not in ["teacher", "admin"]:
        raise HTTPException(status_code=403, detail="Not authorized to create questions")
    
    # Verify subject exists
    subject = db.query(Subject).filter(Subject.id == question.subject_id).first()
    if not subject:
        raise HTTPException(status_code=404, detail="Subject not found")
    
    db_question = Question(
        content=question.content,
        question_type=question.question_type,
        difficulty_level=question.difficulty_level,
        subject_id=question.subject_id,
        created_by=current_user.id,
        options=question.options,
        correct_answer=question.correct_answer,
        explanation=question.explanation,
        tags=question.tags
    )
    
    db.add(db_question)
    db.commit()
    db.refresh(db_question)
    
    # Add skill associations if provided
    if question.skill_ids:
        for skill_id, weight in question.skill_ids.items():
            skill = db.query(Skill).filter(Skill.id == skill_id).first()
            if skill:
                question_skill = QuestionSkill(
                    question_id=db_question.id,
                    skill_id=skill_id,
                    weight=weight
                )
                db.add(question_skill)
        
        db.commit()
    
    return db_question

@router.put("/questions/{question_id}", response_model=QuestionResponse)
def update_question(
    question_id: int,
    question_update: QuestionUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Update a question (teachers and admins only)"""
    if current_user.role.value not in ["teacher", "admin"]:
        raise HTTPException(status_code=403, detail="Not authorized to update questions")
    
    db_question = db.query(Question).filter(Question.id == question_id).first()
    if not db_question:
        raise HTTPException(status_code=404, detail="Question not found")
    
    # Update fields
    for field, value in question_update.dict(exclude_unset=True).items():
        setattr(db_question, field, value)
    
    db.commit()
    db.refresh(db_question)
    return db_question

@router.delete("/questions/{question_id}")
def delete_question(
    question_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Delete a question (admins only)"""
    if current_user.role.value != "admin":
        raise HTTPException(status_code=403, detail="Not authorized to delete questions")
    
    db_question = db.query(Question).filter(Question.id == question_id).first()
    if not db_question:
        raise HTTPException(status_code=404, detail="Question not found")
    
    # Soft delete by setting is_active to False
    db_question.is_active = False
    db.commit()
    
    return {"message": "Question deleted successfully"}

@router.post("/questions/{question_id}/answer")
def submit_answer(
    question_id: int,
    answer: str,
    confidence_level: Optional[int] = None,
    feedback: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Submit an answer to a question (students only)"""
    if current_user.role.value != "student":
        raise HTTPException(status_code=403, detail="Only students can submit answers")
    
    # Get the question
    question = db.query(Question).filter(Question.id == question_id).first()
    if not question:
        raise HTTPException(status_code=404, detail="Question not found")
    
    # Check if answer is correct
    is_correct = answer.lower().strip() == question.correct_answer.lower().strip()
    
    # Calculate response time (this would be better handled on the frontend)
    response_time = 30.0  # Placeholder - should be calculated from frontend
    
    # Create student response
    student_response = StudentResponse(
        student_id=current_user.id,
        question_id=question_id,
        answer=answer,
        is_correct=is_correct,
        response_time=response_time,
        confidence_level=confidence_level,
        feedback=feedback
    )
    
    db.add(student_response)
    db.commit()
    db.refresh(student_response)
    
    # Process response for recommendation system
    try:
        recommendation_service.process_student_response(
            db, current_user.id, question_id, answer, is_correct, response_time
        )
    except Exception as e:
        print(f"Error processing response for recommendations: {e}")
    
    return {
        "is_correct": is_correct,
        "correct_answer": question.correct_answer,
        "explanation": question.explanation,
        "response_id": student_response.id
    }

@router.get("/recommendations/", response_model=List[QuestionRecommendation])
def get_recommendations(
    n_recommendations: int = 10,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get personalized question recommendations (students only)"""
    if current_user.role.value != "student":
        raise HTTPException(status_code=403, detail="Only students can get recommendations")
    
    try:
        recommendations = recommendation_service.get_recommendations(
            db, current_user.id, n_recommendations
        )
        
        # Convert to response format
        response_recommendations = []
        for rec in recommendations:
            response_recommendations.append(QuestionRecommendation(
                question_id=rec['question_id'],
                score=rec['score'],
                river_score=rec['river_score'],
                difficulty_match=rec['difficulty_match'],
                skill_match=rec['skill_match'],
                question=rec['question']
            ))
        
        return response_recommendations
        
    except Exception as e:
        print(f"Error getting recommendations: {e}")
        raise HTTPException(status_code=500, detail="Error getting recommendations")

@router.get("/recommendations/next-question")
def get_next_question(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Öğrenciye önerilen bir sonraki soruyu döndürür (daha önce çözmediği en yüksek skorlu soru).
    """
    if current_user.role.value != "student":
        raise HTTPException(status_code=403, detail="Only students can get next question recommendation")
    try:
        recommendations = recommendation_service.get_recommendations(db, current_user.id, n_recommendations=20)
        # Öğrencinin daha önce çözdüğü soruları bul
        solved_question_ids = set(
            r.question_id for r in db.query(StudentResponse).filter(StudentResponse.student_id == current_user.id)
        )
        # Daha önce çözülmemiş ilk soruyu bul
        for rec in recommendations:
            if rec['question_id'] not in solved_question_ids:
                return {
                    "question_id": rec['question_id'],
                    "score": rec['score'],
                    "question": rec['question']
                }
        raise HTTPException(status_code=404, detail="No new recommended questions found")
    except Exception as e:
        print(f"Error getting next question: {e}")
        raise HTTPException(status_code=500, detail="Error getting next question")

@router.get("/subjects/")
def get_subjects(db: Session = Depends(get_db)):
    """Get all subjects"""
    subjects = db.query(Subject).all()
    return subjects

@router.get("/skills/")
def get_skills(
    subject_id: Optional[int] = None,
    db: Session = Depends(get_db)
):
    """Get all skills with optional subject filtering"""
    query = db.query(Skill)
    if subject_id:
        query = query.filter(Skill.subject_id == subject_id)
    
    skills = query.all()
    return skills 