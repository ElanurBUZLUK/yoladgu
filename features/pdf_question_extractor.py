import os
import pdfplumber
import re
import requests

PDF_FOLDER = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'pdf_uploads')
API_URL = "http://localhost:8000/api/v1/questions/"
TOPIC_API_URL = "http://localhost:8000/api/v1/topics/"
DEFAULT_SUBJECT_ID = 1

# Eğer authentication gerekiyorsa, buraya token ekleyebilirsiniz
API_HEADERS = {
    'accept': 'application/json',
    'Content-Type': 'application/json',
    # 'Authorization': 'Bearer <TOKEN>'
}

def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def split_questions(text):
    pattern = r'(?m)^(?:Soru\s*)?(\d+)[\.\)]\s*'
    parts = re.split(pattern, text)[1:]
    questions = []
    for i in range(0, len(parts), 2):
        num, body = parts[i], parts[i+1]
        questions.append(f"{num}. {body.strip()}")
    return questions

def get_or_create_topic(topic_name, subject_id=DEFAULT_SUBJECT_ID):
    # 1. Var olan konuları çek
    try:
        resp = requests.get(TOPIC_API_URL, headers=API_HEADERS)
        if resp.status_code == 200:
            topics = resp.json()
            for topic in topics:
                if topic['name'].lower() == topic_name.lower():
                    return topic['id']
        else:
            print(f"[HATA] Konular çekilemedi: {resp.status_code} - {resp.text}")
    except Exception as e:
        print(f"[HATA] Konu çekme hatası: {e}")
    # 2. Yoksa yeni oluştur
    data = {
        "name": topic_name,
        "description": f"{topic_name} konusu",
        "subject_id": subject_id
    }
    try:
        resp = requests.post(TOPIC_API_URL, json=data, headers=API_HEADERS)
        if resp.status_code in [200, 201]:
            topic = resp.json()
            print(f"[OK] Yeni konu oluşturuldu: {topic_name}")
            return topic['id']
        else:
            print(f"[HATA] Konu oluşturulamadı: {resp.status_code} - {resp.text}")
    except Exception as e:
        print(f"[HATA] Konu oluşturma hatası: {e}")
    return None

def post_question_to_api(question_text, topic_id):
    data = {
        "content": question_text,
        "question_type": "open_ended",
        "difficulty_level": 1,
        "subject_id": DEFAULT_SUBJECT_ID,
        "topic_id": topic_id,
        "correct_answer": "?",
        "options": None,
        "explanation": None,
        "tags": None,
        "skill_ids": None
    }
    try:
        resp = requests.post(API_URL, json=data, headers=API_HEADERS)
        if resp.status_code == 200 or resp.status_code == 201:
            print("[OK] Soru kaydedildi.")
        else:
            print(f"[HATA] Soru kaydedilemedi: {resp.status_code} - {resp.text}")
    except Exception as e:
        print(f"[HATA] API isteği başarısız: {e}")

def main():
    pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.lower().endswith('.pdf')]
    if not pdf_files:
        print("pdf_uploads klasöründe PDF bulunamadı.")
        return
    for pdf_file in pdf_files:
        topic_name = os.path.splitext(pdf_file)[0]
        print(f"\n--- {pdf_file} (Konu: {topic_name}) ---")
        topic_id = get_or_create_topic(topic_name)
        if not topic_id:
            print(f"[HATA] {topic_name} konusu bulunamadı/oluşturulamadı, sorular atlanıyor.")
            continue
        pdf_path = os.path.join(PDF_FOLDER, pdf_file)
        raw_text = extract_text_from_pdf(pdf_path)
        questions = split_questions(raw_text)
        for idx, soru in enumerate(questions, 1):
            print(f"--- Soru {idx} ---")
            print(soru[:200], "...\n")
            post_question_to_api(soru, topic_id)

if __name__ == "__main__":
    main() 