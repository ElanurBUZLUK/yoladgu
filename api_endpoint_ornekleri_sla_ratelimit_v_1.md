# API Endpoint Örnekleri + SLA/Ratelimit — v1

Bu doküman, mevcut **API ENDPOINTS — No‑Code Spec** ve **Task/Plan** ile uyumlu olacak şekilde **örnek istek/yanıt JSON yükleri** ve **SLA / oran sınırlama (ratelimit) politikaları** içerir. Kod yoktur.

> Ortak başlıklar: `Authorization: Bearer <JWT>`, `X-Request-ID: <uuid>`, `Content-Type: application/json`

---

## 0) Simgeler ve Kısaltmalar
- CEFR: A1–C1 İngilizce seviye etiketi
- θ (theta): IRT öğrenci yetenek tahmini
- |θ−b|: hedef soru zorluğu ile öğrenci seviyesi farkı

---

## 1) Örnek Yükler (Requests/Responses)

### 1.1 GET /health
**Response 200**
```json
{ "status": "up", "version": "1.3.2", "time": "2025-08-31T10:07:18Z" }
```

### 1.2 GET /version
**Response 200**
```json
{ "app": "adaptive-rec-svc", "models": { "retrieval": "retr_v4.1", "rerank": "ce_int8_v2", "llm": "llama3-70b-q4", "bandit": "linucb_v1.3" } }
```

### 1.3 POST /auth/login
**Request**
```json
{ "usernameOrEmail": "ayse@example.com", "password": "S3cure!pass" }
```
**Response 200**
```json
{ "access_token": "eyJhbGciOi...", "refresh_token": "eyJhbGciOi...", "expires_in": 3600 }
```
**Response 401**
```json
{ "code": "AUTH_INVALID", "message": "Invalid credentials" }
```

### 1.4 POST /auth/register
**Request**
```json
{ "username": "ayse", "email": "ayse@example.com", "password": "S3cure!pass", "grade": "9", "lang": "tr", "consent_flag": "personalization" }
```
**Response 201**
```json
{ "user_id": "u_12345", "created_at": "2025-08-31T10:10:00Z" }
```
**Response 409**
```json
{ "code": "EMAIL_TAKEN", "message": "Email already registered" }
```

### 1.5 GET /v1/profile/{user_id}
**Response 200**
```json
{ "user_id": "u_12345", "grade": "9", "lang": "tr", "theta_math": 0.15, "theta_en": -0.20, "error_profiles": { "math": { "sign_error": 0.32, "ratio_misuse": 0.18 }, "en": { "prepositions": 0.45, "articles": 0.28 } }, "segments": ["student"], "preferences": {"tone":"concise"} }
```
**Response 403**
```json
{ "code": "RBAC_DENIED", "message": "Not allowed to access this profile" }
```

### 1.6 POST /v1/profile/update
**Request**
```json
{ "user_id": "u_12345", "updates": { "theta_math": 0.22, "error_profiles": { "math": { "sign_error": 0.28 } } } }
```
**Response 200**
```json
{ "success": true, "updated_fields": ["theta_math","error_profiles.math.sign_error"] }
```
**Response 403** (öğrenci kendi θ’sını güncelliyor)
```json
{ "code": "FORBIDDEN_UPDATE", "message": "Only teacher/service can update model fields" }
```

### 1.7 POST /v1/search
**Request**
```json
{ "query": "linear equations practice", "lang": "en", "k": 200 }
```
**Response 200**
```json
{ "candidates": [
  { "item_id": "m_901", "type": "math", "retriever_scores": { "bm25": 11.8, "dense": 0.73 }, "metadata": { "skills": ["linear_equation"], "b": 0.1, "lang": "en" } },
  { "item_id": "m_902", "type": "math", "retriever_scores": { "bm25": 10.4, "dense": 0.69 }, "metadata": { "skills": ["linear_equation"], "b": 0.3, "lang": "en" } }
], "generated_at": "2025-08-31T10:12:33Z" }
```

### 1.8 POST /v1/rerank
**Request**
```json
{ "query_repr": { "skills": ["linear_equation"], "target_diff": 0.2 }, "candidates": [ { "item_id": "m_901" }, { "item_id": "m_902" } ], "max_k": 40 }
```
**Response 200**
```json
{ "ranked": [ { "item_id": "m_902", "score": 1.42 }, { "item_id": "m_901", "score": 1.31 } ], "model_version": "ce_int8_v2" }
```

### 1.9 POST /v1/recommend/next
**Request**
```json
{ "user_id": "u_12345", "target_skills": ["linear_equation","proportions"], "constraints": { "difficulty_window": [-0.1, 0.3], "diversity": { "topic_min_coverage": 0.8 }, "slate_k": 3 }, "personalization": { "use_peers": true, "avoid_repeats": true } }
```
**Response 200**
```json
{ "items": [
  { "item_id": "m_902", "reason_tags": ["skill:linear_equation","match:diff"], "propensity": 0.22 },
  { "generated_item": { "type": "en_cloze", "level": "B1", "passage": "..." }, "reason_tags": ["en:prepositions","personalized"], "propensity": 0.18 },
  { "item_id": "m_777", "reason_tags": ["peer:hard_for_similar"], "propensity": 0.15 }
], "policy_id": "personalized_linucb_v2", "bandit_version": "linucb_v1.3", "request_id": "req_abc" }
```
**Response 409** (kısıtlar sağlanamıyor)
```json
{ "code": "CONSTRAINT_UNSATISFIED", "message": "Cannot satisfy diversity and difficulty jointly", "details": { "violations": ["diversity:topic_min_coverage"] } }
```

### 1.10 POST /v1/generate/math
**Request**
```json
{ "template_id": "linear_eq_v1", "params_hint": { "a": {"min": 1, "max": 9, "exclude": [0]}, "b": {"min": -10, "max": 10}, "c": {"min": -10, "max": 10} }, "target_difficulty": 0.2, "language": "tr", "rationale_required": true }
```
**Response 200**
```json
{ "item": { "stem": "3x + 4 = 19 denklemini çözünüz.", "choices": ["x=5","x=4","x=3","x=2"], "answer_key": "x=5", "solution": "3x+4=19 → 3x=15 → x=5", "skills": ["linear_equation"], "difficulty_estimate": { "a": 0.9, "b": 0.2 }, "QA_checks": { "solver_passed": true, "single_gold": true } }, "generator": { "template_id": "linear_eq_v1", "version": "1.0.3" } }
```
**Response 409** (tek-altın sağlanamadı)
```json
{ "code": "MULTIPLE_GOLDS", "message": "Ambiguous answer key detected" }
```

### 1.11 POST /v1/generate/en_cloze
**Request**
```json
{ "level_cefr": "B1", "target_error_tags": ["prepositions"], "topic": "travel", "personalization": { "user_id": "u_12345", "use_error_history": true }, "ambiguity_tolerance": "strict" }
```
**Response 200**
```json
{ "item": { "passage": "We arrived __ the airport at 7 pm.", "blanks": [ { "span": "__", "answer": "at", "distractors": ["to","in","on"], "skill_tag": "prepositions" } ], "QA_checks": { "grammar_valid": true, "single_answer": true } }, "generator": { "version": "en_cloze_0.9" } }
```
**Response 409** (çoklu doğru)
```json
{ "code": "AMBIGUOUS_CLOZE", "message": "Multiple valid answers found under strict tolerance" }
```

### 1.12 POST /v1/attempt
**Request**
```json
{ "user_id": "u_12345", "item_id": "m_902", "answer": "x=5", "correct": true, "time_ms": 28000, "hints_used": 0, "context": { "device": "mobile", "session_id": "s_789" } }
```
**Response 200**
```json
{ "stored": true, "updated_theta": { "math": 0.24 }, "reward_components": { "correct": 1.0, "completion": 1.0, "dwell": 0.3 } }
```

### 1.13 POST /v1/feedback
**Request**
```json
{ "user_id": "u_12345", "item_id": "m_902", "rating": 4, "flags": { "too_easy": true }, "comment": "Benzer sorular daha zor olabilir." }
```
**Response 200**
```json
{ "stored": true }
```

### 1.14 GET /v1/admin/metrics
**Response 200**
```json
{ "p95_latency": 720, "error_rate": 0.006, "cache_hit_rate": 0.58, "faithfulness": 0.95, "difficulty_match": 0.82, "coverage": 0.86, "exploration_ratio": 0.22 }
```

### 1.15 GET /v1/admin/decisions/{request_id}
**Response 200**
```json
{ "policy_id": "personalized_linucb_v2", "bandit_version": "linucb_v1.3", "arms": [ { "arm_id": "arm_m_902", "item_id": "m_902", "propensity": 0.22, "scores": { "linucb": 1.42, "heuristic": 0.88 } } ], "chosen_arm_id": "arm_m_902", "created_at": "2025-08-31T10:15:00Z" }
```

---

## 2) SLA / Ratelimit Politikaları

### 2.1 Hizmet Seviyesi Hedefleri (SLO)
| Endpoint | p50 (ms) | p95 (ms) | p99 (ms) | Hedef Uptime | Not |
|---|---:|---:|---:|---:|---|
| GET /health | 10 | 30 | 50 | 99.99% | Cache’li versiyon bilgisi |
| GET /version | 15 | 40 | 60 | 99.99% | Statik/ConfigMap |
| POST /auth/login | 40 | 120 | 200 | 99.9% | Rate limit hassas |
| POST /auth/register | 60 | 180 | 300 | 99.9% | E‑posta kontrolü |
| GET /v1/profile/{id} | 40 | 120 | 200 | 99.9% | RBAC kontrolleri |
| POST /v1/profile/update | 60 | 150 | 250 | 99.9% | Sadece öğretmen/servis |
| POST /v1/search | 80 | 200 | 300 | 99.9% | Hybrid retrieval |
| POST /v1/rerank | 120 | 250 | 350 | 99.9% | Cross‑encoder INT8 |
| POST /v1/recommend/next | 300 | 700 | 1000 | 99.9% | Retrieval+rerank+bandit |
| POST /v1/generate/math | 500 | 1200 | 1800 | 99.5% | Solver+LLM |
| POST /v1/generate/en_cloze | 500 | 1200 | 1800 | 99.5% | Grammar check+LLM |
| POST /v1/attempt | 50 | 150 | 250 | 99.99% | DB write + θ update |
| POST /v1/feedback | 40 | 120 | 200 | 99.99% | Basit write |
| GET /v1/admin/metrics | 80 | 200 | 300 | 99.9% | Özetlenmiş metrik tablosu |
| GET /v1/admin/decisions/{id} | 80 | 200 | 300 | 99.9% | Tekil kayıt |

### 2.2 Oran Sınırları (Rate Limits)
| Endpoint | Varsayılan Kullanıcı | Service Account | Admin |
|---|---:|---:|---:|
| /auth/* | 30 req/dk | 60 req/dk | 60 req/dk |
| /v1/search | 120 req/dk | 300 req/dk | 120 req/dk |
| /v1/rerank | 60 req/dk | 180 req/dk | 60 req/dk |
| /v1/recommend/next | 30 req/dk | 90 req/dk | 60 req/dk |
| /v1/generate/math | 10 req/dk | 30 req/dk | 20 req/dk |
| /v1/generate/en_cloze | 10 req/dk | 30 req/dk | 20 req/dk |
| /v1/attempt | 120 req/dk | 300 req/dk | 120 req/dk |
| /v1/feedback | 120 req/dk | 300 req/dk | 120 req/dk |
| /v1/admin/* | — | — | 60 req/dk |

> **Burst**: 2× kısa süreli patlama (leaky bucket). **IP tabanlı** ve **kullanıcı‑ID** tabanlı limit birlikte uygulanır.

### 2.3 Kotalar & Boyut Sınırları
- **Request body**: max 128 KB (generate uçlarında 256 KB)
- **Response**: max 1 MB (long passages için sayfalama)
- **Storage**: Attempt/event günlükleri 365 gün (TTL), PII yok

### 2.4 Hata Kodları Özet
| Kod | Açıklama | Tipik Sebep |
|---|---|---|
| AUTH_INVALID | Kimlik doğrulama başarısız | Yanlış şifre/eksik token |
| RBAC_DENIED | Yetkisiz erişim | Farklı kullanıcı profili |
| CONSTRAINT_UNSATISFIED | Kısıtlar sağlanamadı | diversity/difficulty çakışması |
| MULTIPLE_GOLDS | Tek-altın ihlali | Math template sonucu çoklu cevap |
| AMBIGUOUS_CLOZE | Çoklu doğru | Strict modda cloze belirsiz |
| FORBIDDEN_UPDATE | Yetkisiz alan değişimi | Öğrencinin θ güncellemesi |
| SCHEMA_INVALID | Şema doğrulama hatası | Eksik/yanlış alan |

---

## 3) İzleme Göstergeleri (Dashboard Sözlüğü)
- **retrieval_ms / rerank_ms / llm_ms**: pipeline segment süreleri
- **p50/p95/p99 latency**: endpoint bazlı
- **cache_hit_rate**: retrieval ve semantic cache
- **faithfulness**: RAG cevaplarının kaynak uyumu (offline denetimli set)
- **difficulty_match**: |θ−b|≤0.3 oranı
- **coverage**: curriculum/topic kapsaması (7g)
- **exploration_ratio**: bandit keşif payı
- **error_rate**: 5xx + şema/guardrail hataları

---

## 4) Notlar
- Örnekler temsilidir; gerçek `model_version`, `policy_id` vb. dağıtım sırasında doldurulur.
- Ratelimit ve SLA değerleri başlangıç hedefleridir; prod trafik verisine göre yeniden kalibre edilir.

