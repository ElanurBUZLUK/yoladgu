from app.db.models import StudentProfile
from sqlalchemy.orm import Session


def update_student_level(
    db: Session, student_id: int, difficulty: float, is_correct: bool
):
    profile = db.query(StudentProfile).filter_by(student_id=student_id).first()
    if not profile:
        profile = StudentProfile(
            student_id=student_id,
            level=difficulty,
            min_level=difficulty,
            max_level=difficulty,
        )
        db.add(profile)
        db.commit()
        return profile

    # Get current values - use getattr to avoid type issues
    current_max = getattr(profile, "max_level", None)
    current_min = getattr(profile, "min_level", None)

    current_max = current_max if current_max is not None else 0.0
    current_min = current_min if current_min is not None else difficulty

    # Update max level if correct answer and difficulty is higher
    if is_correct and difficulty > current_max:
        setattr(profile, "max_level", difficulty)
        current_max = difficulty

    # Update min level if incorrect answer and difficulty is lower
    if not is_correct and difficulty < current_min:
        setattr(profile, "min_level", difficulty)
        current_min = difficulty

    # Calculate new level as average of min and max
    new_level = (current_min + current_max) / 2
    setattr(profile, "level", new_level)

    db.commit()
    return profile
