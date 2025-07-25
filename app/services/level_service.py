from sqlalchemy.orm import Session
from app.db.models import StudentProfile

def update_student_level(db: Session, student_id: int, difficulty: float, is_correct: bool):
    profile = db.query(StudentProfile).filter_by(student_id=student_id).first()
    if not profile:
        profile = StudentProfile(student_id=student_id, level=difficulty, min_level=difficulty, max_level=difficulty)
        db.add(profile)
        db.commit()
        return profile
    if is_correct and difficulty > (profile.max_level or 0):
        profile.max_level = difficulty
    if not is_correct and (profile.min_level is None or difficulty < profile.min_level):
        profile.min_level = difficulty
    profile.level = (profile.min_level + profile.max_level) / 2
    db.commit()
    return profile 