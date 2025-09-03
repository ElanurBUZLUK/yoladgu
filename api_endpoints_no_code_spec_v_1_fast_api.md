# API ENDPOINTS — No‑Code Spec v1 (FastAPI)

> Amaç: Kod olmadan, üretime hazır **endpoint kataloğu**. Her endpoint için: amaç, kimlik/doğrulama, istek/yanıt şeması alan listesi, validasyon notları, hata kodları ve oran sınırlamaları.

---

## 0) Ortak İlkeler
- **Kimlik**: Tüm `/v1/*` uçları **JWT Bearer** gerektirir. Scope’lar: `student`, `teacher`, `admin`, `service`.
- **İzleme**: Her istekte `request_id` (UUID) header’ı loglanır. Cevaplarda `X-Request-ID` döner.
- **Sürümleme**: `/v1` ana sürümdür; geriye dönük kırıcı değişiklikler için `/v2` açılır.
- **Oran Sınırlama**: Varsayılan 60 req/dk (kullanıcı), 600 req/dk (service).
- **Hata Gövdesi (ortak)**: `{ code: string, message: string, details?: object }`.

---

## 1) Sağlık & Meta
### 1.1 `GET /health`
- **Auth**: Yok
- **Yanıt**: `{ status: 'up'|'down', version: string, time: ISO8601 }`
- **Hatalar**: 503 (down)

### 1.2 `GET /version`
- **Auth**: Yok
- **Yanıt**: `{ app: string, models: { retrieval: string, rerank: string, llm: string, bandit: string } }`

---

## 2) Kimlik & Profil
### 2.1 `POST /auth/login`
- **Auth**: Yok
- **İstek**: `{ usernameOrEmail: string, password: string }`
- **Yanıt**: `{ access_token: string, refresh_token: string, expires_in: number }`
- **Hatalar**: 401 (invalid), 423 (locked)

### 2.2 `POST /auth/register`
- **Auth**: Yok
- **İstek**: `{ username: string, email: string, password: string, grade?: string, lang?: 'tr'|'en', consent_flag?: 'none'|'analytics'|'personalization' }`
- **Yanıt**: `{ user_id: string, created_at: ISO8601 }`
- **Hatalar**: 409 (email taken), 400 (zayıf parola)

### 2.3 `GET /v1/profile/{user_id}`
- **Auth**: Bearer (`student`/`teacher`/`admin`)
- **Yanıt**: `{ user_id, grade, lang, theta_math?: number, theta_en?: number, error_profiles?: { math?: object, en?: object }, segments?: string[], preferences?: object }`
- **Hatalar**: 404 (not found), 403 (RBAC)

### 2.4 `POST /v1/profile/update`
- **Auth**: Bearer
- **İstek**: `{ user_id: string, updates: { theta_math?: number, theta_en?: number, error_profiles?: { math?: object, en?: object }, preferences?: object } }`
- **Yanıt**: `{ success: boolean, updated_fields: string[] }`
- **Not**: Öğrenci kendi theta’sını güncelleyemez; yalnız servis/öğretmen.

---

## 3) Retrieval & Re‑rank
### 3.1 `POST /v1/search`
- **Amaç**: Aday küme (K≈200) döndürür.
- **Auth**: Bearer
- **İstek**: `{ query?: string, goals?: { skills?: string[], level?: string, topic?: string }, lang?: 'tr'|'en', k?: number (≤ 200) }`
- **Yanıt**: `{ candidates: [{ item_id: string, type: 'math'|'en', retriever_scores: { bm25?: number, dense?: number }, metadata?: object }], generated_at: ISO8601 }`
- **Hatalar**: 400 (geçersiz parametre)

### 3.2 `POST /v1/rerank`
- **Amaç**: İlk 200’ü 40’a indirir (cross‑encoder + heuristik).
- **Auth**: Bearer
- **İstek**: `{ query_repr: object|string, candidates: [{ item_id: string, features?: object }], max_k?: number (≤ 40) }`
- **Yanıt**: `{ ranked: [{ item_id: string, score: number }], model_version: string }`

---

## 4) Öneri Akışı
### 4.1 `POST /v1/recommend/next`
- **Amaç**: Öğrenciye bir sonraki soru(lar)ı önerir (slate K).
- **Auth**: Bearer
- **İstek**: `{
  user_id: string,
  target_skills?: string[],
  constraints?: { difficulty_window?: [number, number], diversity?: { topic_min_coverage?: number }, slate_k?: number },
  personalization?: { use_peers?: boolean, avoid_repeats?: boolean }
}`
- **Yanıt**: `{
  items: [ { item_id?: string, generated_item?: object, reason_tags: string[], propensity?: number } ],
  policy_id: string, bandit_version: string, request_id: string
}`
- **Hatalar**: 412 (profil yetersiz), 409 (kısıtlar sağlanamıyor)

---

## 5) İçerik Üretimi — Matematik
### 5.1 `POST /v1/generate/math`
- **Amaç**: Parametrik şablondan **doğrulanmış** matematik sorusu üretmek.
- **Auth**: Bearer (`teacher`/`service`)
- **İstek**: `{
  template_id: string,
  params_hint?: object,               // aralıklar veya kısıtlar
  target_difficulty?: number,         // θ‑b hedefi
  language?: 'tr'|'en',
  rationale_required?: boolean
}`
- **Yanıt**: `{
  item: { stem: string, choices?: string[], answer_key: string|number, solution?: string, skills: string[], bloom?: string, difficulty_estimate?: { a?: number, b?: number }, QA_checks: { solver_passed: boolean, single_gold: boolean } },
  generator: { template_id: string, version: string }
}`
- **Hatalar**: 422 (parametre çakışması), 409 (tek‑altın sağlanamadı)

---

## 6) İçerik Üretimi — İngilizce (Cloze)
### 6.1 `POST /v1/generate/en_cloze`
- **Amaç**: Hedef hata türlerine göre cloze sorusu üretmek.
- **Auth**: Bearer (`teacher`/`service`)
- **İstek**: `{
  level_cefr: 'A1'|'A2'|'B1'|'B2'|'C1',
  target_error_tags: string[],        // prepositions, articles, SVA, ...
  topic?: string,
  personalization?: { user_id?: string, use_error_history?: boolean },
  ambiguity_tolerance?: 'strict'|'lenient'
}`
- **Yanıt**: `{
  item: { passage: string, blanks: [ { span: string, answer: string, distractors: string[], skill_tag: string } ], QA_checks: { grammar_valid: boolean, single_answer: boolean } },
  generator: { version: string }
}`
- **Hatalar**: 409 (çoklu doğru), 422 (etiket uyuşmazlığı)

---

## 7) Deneyimler, Geri Bildirim & Ödüller
### 7.1 `POST /v1/attempt`
- **Amaç**: Öğrenci yanıtını kaydetmek, θ güncellemesini tetiklemek.
- **Auth**: Bearer
- **İstek**: `{ user_id: string, item_id: string, answer: string, correct: boolean, time_ms?: number, hints_used?: number, context?: { device?: string, session_id?: string } }`
- **Yanıt**: `{ stored: boolean, updated_theta?: { math?: number, en?: number }, reward_components?: object }`

### 7.2 `POST /v1/feedback`
- **Auth**: Bearer
- **İstek**: `{ user_id: string, item_id: string, rating?: 1|2|3|4|5, flags?: { too_hard?: boolean, too_easy?: boolean, misleading?: boolean }, comment?: string }`
- **Yanıt**: `{ stored: boolean }`

---

## 8) Yönetim & Gözlemlenebilirlik
### 8.1 `GET /v1/admin/metrics`
- **Auth**: Bearer (`admin`)
- **Süzgeç**: `?from=ISO&to=ISO&tenant=string`
- **Yanıt**: `{ p95_latency: number, error_rate: number, cache_hit_rate: number, faithfulness: number, difficulty_match: number, coverage: number, exploration_ratio: number }`

### 8.2 `GET /v1/admin/decisions/{request_id}`
- **Auth**: Bearer (`admin`)
- **Yanıt**: `{ policy_id: string, bandit_version: string, arms: [{ arm_id: string, item_id: string, propensity: number, scores: object }], chosen_arm_id: string, created_at: ISO8601 }`

---

## 9) Validasyon & İş Kuralları (Endpoint Düzeyi)
- **`/v1/recommend/next`**: `slate_k` 1–10; `difficulty_window` sayısal ve simetrik; kısıtlar sağlanamıyorsa 409.
- **`/v1/generate/math`**: `params_hint` çakışmaları 422; `solver_passed=false` ise öğe **kaydolmaz**.
- **`/v1/generate/en_cloze`**: `ambiguity_tolerance='strict'` ise birden fazla doğru saptanırsa 409.
- **`/v1/attempt`**: `correct` yalnızca boolean; `time_ms ≥ 0`.

---

## 10) Güvenlik Notları
- **PII Redaction**: E‑posta/IP gibi PII log’larda maskelenir.
- **RBAC**: Öğrenci yalnız kendi verisini görebilir; öğretmen sınıfındakileri; admin tüm tenant.
- **Audit**: Tüm `/v1/recommend`, `/v1/generate/*`, `/v1/attempt` çağrıları karar izleriyle kayda alınır (model_version, policy_id, bandit_version, propensity).

---

## 11) SLA & Oran Sınırları (Endpoint Bazlı Öneri)
- **/v1/search**: p95 ≤ 200 ms, 120 req/dk
- **/v1/rerank**: p95 ≤ 250 ms, 60 req/dk
- **/v1/recommend/next**: p95 ≤ 700 ms, 30 req/dk
- **/v1/generate/math**: p95 ≤ 1200 ms, 10 req/dk
- **/v1/generate/en_cloze**: p95 ≤ 1200 ms, 10 req/dk
- **/v1/attempt**: p95 ≤ 150 ms, 120 req/dk

---

**Not**: Bu şema, mevcut “Backend Spec (No‑Code)” ve “Task List” dokümanlarıyla uyumludur. İstenirse her endpoint için **örnek istek/yanıt JSON**ları eklenebilir (kodsuz).

