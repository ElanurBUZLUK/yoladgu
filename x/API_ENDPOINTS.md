# API Endpoints Documentation

## Math RAG Endpoints

### Core RAG Endpoints

#### 1. Generate Questions
**POST** `/api/v1/math/rag/generate`

LLM kullanarak matematik soruları üretir.

**Request Body:**
```json
{
  "topic": "quadratic equations",
  "difficulty_level": 3,
  "question_type": "multiple_choice",
  "n": 2
}
```

**Response:**
```json
{
  "items": [
    {
      "kind": "multiple_choice",
      "stem": "What is the solution to x² + 5x + 6 = 0?",
      "options": ["x = -2, -3", "x = 2, 3", "x = -1, -6", "x = 1, 6"],
      "answer": {"correct": "x = -2, -3"},
      "meta": {"topic": "quadratic_equations", "difficulty": 3}
    }
  ],
  "usage": {
    "tokens_used": 150,
    "cost": 0.002
  }
}
```

#### 2. Solve Problem
**POST** `/api/v1/math/rag/solve`

Matematik problemlerini çözer ve adımları gösterir.

**Request Body:**
```json
{
  "problem": "Solve for x: 2x + 5 = 13",
  "show_steps": true
}
```

**Response:**
```json
{
  "solution": "x = 4",
  "steps": "1. Subtract 5 from both sides: 2x = 8\n2. Divide both sides by 2: x = 4",
  "usage": {
    "tokens_used": 120,
    "cost": 0.001
  }
}
```

#### 3. Check Answer
**POST** `/api/v1/math/rag/check`

Kullanıcı cevaplarını değerlendirir.

**Request Body:**
```json
{
  "question": "What is 2 + 2?",
  "user_answer": "4",
  "answer_key": "4",
  "require_explanation": true
}
```

**Response:**
```json
{
  "correct": true,
  "explanation": "Your answer is correct. 2 + 2 = 4.",
  "usage": {
    "tokens_used": 80,
    "cost": 0.001
  }
}
```

### Existing RAG Endpoints

#### 4. Next Question
**POST** `/api/v1/math/rag/next-question`

Öğrenci için bir sonraki matematik sorusunu seçer.

#### 5. Submit Answer
**POST** `/api/v1/math/rag/submit-answer`

Öğrenci cevabını gönderir ve değerlendirir.

#### 6. Get Profile
**GET** `/api/v1/math/rag/profile`

Öğrenci matematik profilini getirir.

### Analytics Endpoints

#### 7. Performance Analytics
**GET** `/api/v1/math/rag/analytics/performance`

Öğrenci performans analizini getirir.

#### 8. Learning Analytics
**GET** `/api/v1/math/rag/analytics/learning`

Öğrenme analizini getirir.

### Monitoring Endpoints

#### 9. System Health
**GET** `/api/v1/math/rag/monitoring/health`

Sistem sağlık durumunu kontrol eder.

#### 10. Performance Monitoring
**GET** `/api/v1/math/rag/monitoring/performance`

Sistem performans metriklerini getirir.

### Personalization Endpoints

#### 11. Get Recommendations
**GET** `/api/v1/math/rag/personalization/recommendations`

Kişiselleştirilmiş önerileri getirir.

#### 12. Update Preferences
**POST** `/api/v1/math/rag/personalization/preferences`

Kullanıcı tercihlerini günceller.

### Advanced Retrieval Endpoints

#### 13. Advanced Retrieval
**POST** `/api/v1/math/rag/retrieval/advanced`

Gelişmiş soru retrieval işlemi.

#### 14. Query Expansion
**POST** `/api/v1/math/rag/retrieval/query-expansion`

Sorgu genişletme işlemi.

### Quality Assurance Endpoints

#### 15. Question Quality Check
**POST** `/api/v1/math/rag/quality/check-question`

Soru kalitesini kontrol eder.

#### 16. Session Quality Check
**POST** `/api/v1/math/rag/quality/check-session`

Oturum kalitesini kontrol eder.

### A/B Testing Endpoints

#### 17. Create Experiment
**POST** `/api/v1/math/rag/ab-testing/create-experiment`

A/B test deneyi oluşturur.

#### 18. Record Event
**POST** `/api/v1/math/rag/ab-testing/record-event`

Deney olayını kaydeder.

#### 19. Get Experiment Status
**GET** `/api/v1/math/rag/ab-testing/experiment-status/{experiment_id}`

Deney durumunu getirir.

#### 20. Analyze Experiment
**POST** `/api/v1/math/rag/ab-testing/analyze-experiment`

Deneyi analiz eder.

#### 21. List Experiments
**GET** `/api/v1/math/rag/ab-testing/list-experiments`

Deneyleri listeler.

## Math Questions Endpoints

### Core Question Endpoints

#### 1. Recommend Questions
**POST** `/api/v1/math/questions/recommend`

Kullanıcı profiline göre soru önerisi.

#### 2. Search Questions
**GET** `/api/v1/math/questions/search`

Soru arama.

#### 3. Get Questions by Level
**GET** `/api/v1/math/questions/by-level/{level}`

Seviyeye göre sorular.

#### 4. Get Question Pool Stats
**GET** `/api/v1/math/questions/pool`

Soru havuzu istatistikleri.

#### 5. Get Topics
**GET** `/api/v1/math/questions/topics`

Konu listesi.

#### 6. Get Difficulty Distribution
**GET** `/api/v1/math/questions/difficulty-distribution`

Zorluk dağılımı.

#### 7. Get Random Question
**GET** `/api/v1/math/questions/random/{level}`

Rastgele soru.

#### 8. Get Question Stats
**GET** `/api/v1/math/questions/stats`

Soru istatistikleri.

#### 9. Get Specific Question
**GET** `/api/v1/math/questions/{question_id}`

Belirli soru.

## Authentication

Tüm endpoint'ler JWT token gerektirir. Token'ı Authorization header'ında gönderin:

```
Authorization: Bearer <your_jwt_token>
```

## Error Responses

Tüm endpoint'ler standart hata formatı kullanır:

```json
{
  "detail": "Error message",
  "status_code": 400
}
```

## Rate Limiting

API rate limiting uygulanır. Limit aşıldığında 429 status code döner.

## Usage Tracking

LLM kullanan endpoint'ler usage bilgisi döner:

```json
{
  "usage": {
    "tokens_used": 150,
    "cost": 0.002,
    "model": "gpt-4"
  }
}
```
