import pytest
import redis
import time
from multiprocessing import Process
from unittest.mock import patch
from app.services.stream_consumer import consume_loop, DLQ_STREAM

def test_stream_consumer_integration(monkeypatch):
    # Dummy RecommendationService.process_student_response
    called = {}
    def dummy_process(db, student_id, question_id, answer, is_correct, response_time, feedback=None):
        called.update(locals())
    monkeypatch.setattr("app.services.recommendation_service.RecommendationService.process_student_response", dummy_process)

    # Redis setup
    r = redis.Redis()
    r.delete("student_responses_stream")
    event = {
        "student_id": "1",
        "question_id": "2",
        "is_correct": "True",
        "response_time": "12.3",
        "feedback": "Harika"
    }
    r.xadd("student_responses_stream", event)

    # Consumer'ı kısa süreli başlat
    p = Process(target=consume_loop)
    p.start()
    time.sleep(2)
    p.terminate()
    # Test: Dummy fonksiyon çağrıldı mı?
    assert called.get("student_id") == 1
    assert called.get("feedback") == "Harika" 

def test_stream_consumer_idempotency(monkeypatch):
    called = {"count": 0}
    def dummy_process(db, student_id, question_id, answer, is_correct, response_time, feedback=None, request_id=None):
        called["count"] += 1
    monkeypatch.setattr("app.services.recommendation_service.RecommendationService.process_student_response", dummy_process)
    r = redis.Redis()
    r.delete("student_responses_stream")
    r.delete(DLQ_STREAM)
    event = {
        "student_id": "1",
        "question_id": "2",
        "is_correct": "True",
        "response_time": "12.3",
        "feedback": "Harika"
    }
    eid1 = r.xadd("student_responses_stream", event)
    eid2 = r.xadd("student_responses_stream", event)  # Duplicate
    p = Process(target=consume_loop)
    p.start()
    time.sleep(2)
    p.terminate()
    assert called["count"] == 1  # Sadece bir kez işlenmeli

def test_stream_consumer_dlq(monkeypatch):
    called = {"count": 0}
    def dummy_process(db, student_id, question_id, answer, is_correct, response_time, feedback=None, request_id=None):
        called["count"] += 1
        raise ValueError("Test retry")
    monkeypatch.setattr("app.services.recommendation_service.RecommendationService.process_student_response", dummy_process)
    r = redis.Redis()
    r.delete("student_responses_stream")
    r.delete(DLQ_STREAM)
    event = {
        "student_id": "1",
        "question_id": "2",
        "is_correct": "True",
        "response_time": "12.3",
        "feedback": "Harika"
    }
    r.xadd("student_responses_stream", event)
    p = Process(target=consume_loop)
    p.start()
    time.sleep(2)
    p.terminate()
    # DLQ'ya event yazıldı mı?
    dlq_len = r.xlen(DLQ_STREAM)
    assert dlq_len >= 1

def test_stream_consumer_dlq_malformed(monkeypatch):
    r = redis.Redis()
    r.delete("student_responses_stream")
    r.delete(DLQ_STREAM)
    r.xadd("student_responses_stream", {"bad": "data"})
    p = Process(target=consume_loop)
    p.start()
    time.sleep(2)
    p.terminate()
    dlq_len = r.xlen(DLQ_STREAM)
    assert dlq_len >= 1 