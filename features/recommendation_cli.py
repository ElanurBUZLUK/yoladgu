import sys
import os
import argparse

# Ensure project root is on the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.db.database import SessionLocal
from app.services.recommendation_service import RecommendationService


def test_model(service, db, student_id, test_data):
    correct = 0
    total = 0
    for entry in test_data:
        qid = entry['question_id']
        is_correct = entry['is_correct']
        # Modeli güncelle
        service.process_student_response(db, student_id, qid, answer=None, is_correct=is_correct, response_time=1000)
        # Öneri üret
        recs = service.get_recommendations(db, student_id, n_recommendations=1)
        if recs and recs[0]['question_id'] == qid:
            correct += 1
        total += 1
    print(f"Model accuracy: {correct}/{total} = {correct/total if total else 0:.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['river', 'linucb'], default='river', help='Model type to use')
    parser.add_argument('--test', action='store_true', help='Run test mode with sample data')
    args = parser.parse_args()
    try:
        student_id = int(input("Enter student ID: "))
    except ValueError:
        print("Invalid student ID")
        return
    service = RecommendationService(model_type=args.model)
    db = SessionLocal()
    try:
        if args.test:
            # Örnek test datası (gerçek veriyle değiştirilebilir)
            test_data = [
                {'question_id': 1, 'is_correct': True},
                {'question_id': 2, 'is_correct': False},
                {'question_id': 3, 'is_correct': True},
            ]
            test_model(service, db, student_id, test_data)
        else:
            recommendations = service.get_recommendations(db, student_id)
            print("Recommended questions:")
            for rec in recommendations:
                question = rec.get('question', {})
                print(f"- ID {question.get('id')}: {question.get('content')}")
    finally:
        db.close()


if __name__ == "__main__":
    main() 