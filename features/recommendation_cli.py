import sys
import os

# Ensure project root is on the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.db.database import SessionLocal
from app.services.recommendation_service import RecommendationService


def main():
    try:
        student_id = int(input("Enter student ID: "))
    except ValueError:
        print("Invalid student ID")
        return

    service = RecommendationService()
    db = SessionLocal()
    try:
        recommendations = service.get_recommendations(db, student_id)
    finally:
        db.close()

    print("Recommended questions:")
    for rec in recommendations:
        question = rec.get('question', {})
        print(f"- ID {question.get('id')}: {question.get('content')}")


if __name__ == "__main__":
    main() 