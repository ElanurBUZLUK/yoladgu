# Backend Spec (No‑Code) — Yapı, DDL Özellikleri, OpenAPI Şemaları

Bu belge, **kod içermeden** aşağıdaki üç alanı üretime hazır olacak şekilde tanımlar:
1) Klasör/Modül Yapısı (FastAPI)
2) Veritabanı Şeması (DDL özellikleri; tablo ve alan tanımları)
3) OpenAPI Endpoint Şemaları (alan listeleri, doğrulama kuralları)

---

## 1) Klasör / Modül Yapısı (FastAPI) — Kod Yok

**Kök dizin:**
- .env.example, README.md, BACKEND_TASKS.md, LICENSE, docs/, infra/, scripts/, tests/, app/

**app/** altı:
- **core/**: uygulama ayarları, güvenlik politikaları, middleware’ler, hata/validasyon stratejileri
- **api/**: router tanımları ve endpoint modülleri (health, auth, recommend, generate, feedback, admin)
- **models/**: pydantic model tanımları (istek/yanıt şemaları, domain nesneleri)
- **db/**: bağlantı yönetimi, repository arayüzleri, migration metadata
- **services/**: iş kuralları (profil, IRT/BKT/DKT, bandit, diversification, evaluation)
- **retrieval/**: hybrid arama (sparse+dense), metadata filtreleri, candidate provider’lar
- **ranking/**: heuristic füzyon, cross‑encoder arayüzü, LTR entegrasyonu
- **generation/**: matematik şablonları, çözücü entegrasyonu, EN cloze üretim mantığı
- **bandit/**: epsilon‑greedy, LinUCB/LinTS, kısıtlı bandit politika tanımları ve kayıt/propensity günlükleri
- **observability/**: logging, tracing, metrics; panel entegrasyonları
- **security/**: PII redaction, guardrails, RBAC kuralları, audit kayıt stratejisi
- **routers/**: API yüzeyine göre gruplanmış yönlendirmeler (api/v1 içinde modüler örgü)
- **main entry**: uygulama başlatma, yaşam döngüsü, CORS, kompresyon, sürüm bilgisi

**docs/**: açık metin spesifikasyonları, sözleşmeler, ölçüm planları

**infra/**: veritabanı şema tanımı belgeleri, izleme ve dağıtım konfigürasyonları (kod değil)

**tests/**: birim, sözleşme ve yük testi senaryolarının tanımlarının yer aldığı açıklama dosyaları

---

## 2) Veritabanı Şeması (DDL Özellikleri) — Kod Yok

### 2.1 Temel İlkeler
- **Kimlikler ve ayrıştırma:** tenant_id, user_id, request_id, item_id; tarih bazlı partisyonlar.
- **Gizlilik:** PII saklanmaz veya ayrı kasada; maskelenmiş alanlar kullanılır.
- **Sürümleme:** model_version, policy_version, bandit_version alanları ilişkilendirilir.
- **İndeksleme:** sık sorgulanan alanlar (user_id, item_id, event_time, topic, skill, level_cefr) için bileşik indeksler.

### 2.2 Tablolar ve Alanlar

**users**
- id (kimlik), tenant_id, created_at, grade, lang, consent_flag
- profil alanları: segments (liste), preferences (dil/format)
- öğrenme durumu: theta_math, theta_en, error_profile_math (json), error_profile_en (json)

**items_math**
- id, tenant_id, created_at, updated_at, status (draft/active/retired)
- stem (metin), params (json), solution (metin), answer_key (seçenek/değer), skills (liste), bloom (seviye)
- difficulty_a, difficulty_b (IRT parametreleri), lang, source, generator (şablon adı)
- kalite: pedagogy_flags (json), review_status (öğretmen onayı bilgisi)

**items_en**
- id, tenant_id, created_at, updated_at, status
- passage (metin), blanks (json: {span, skill_tag, answer, distractors[], rationale?})
- level_cefr (A1–C1), topic, lang, source, generator
- kalite: ambiguity_flag, review_status, pedagogy_flags

**attempts**
- id, user_id, item_id, ts (zaman), answer (metin), correct (bool), time_ms, hints_used (sayı)
- context: device, session_id, ip_country (maskeli)

**retrieval_logs**
- id, request_id, user_id, query_repr (özet), candidates (json: doc_id, skorlar), features (json), ts

**decisions**
- id, request_id, user_id, policy_id, bandit_version
- arms (json: arm_id, item_id, propensity, skorlar), chosen_arm_id
- serving_time_ms, ts

**events**
- id, request_id, user_id, type (impression/click/completion/correct/feedback), payload (json), ts

**metrics_daily** (türev tablo)
- tarih, tenant_id, p95_latency, error_rate, cache_hit_rate, faithfulness, difficulty_match, coverage, exploration_ratio

### 2.3 İlişkiler
- users(1) → attempts(*), decisions(*), events(*)
- items_math/items_en(1) → attempts(*), retrieval_logs(*), decisions(*)
- decisions(1) → events(*) (aynı request akışı)

### 2.4 Validasyon Kuralları (DDL Seviyesi)
- status için izinli değerler; level_cefr aralığı; difficulty_a>0, difficulty_b gerçek sayı
- attempts.correct yalnızca {true,false}; time_ms ≥ 0
- items_en.blanks içinde en az bir boşluk; distractors benzersiz

---

## 3) OpenAPI Endpoint Şemaları — Kod Yok

Aşağıdaki her endpoint için **istek/yanıt alanları** ve doğrulama kuralları verilmiştir. Şema, pydantic/OpenAPI’ye dönüştürülmeye uygun **alan listeleri** olarak tanımlıdır.

### 3.1 Health & Meta
- **GET /health**
  - Yanıt: status (up/down), version, time
- **GET /version**
  - Yanıt: app_version, model_versions {retrieval, rerank, llm, bandit}

### 3.2 Kimlik & Profil
- **POST /auth/login**
  - İstek: username/email, password
  - Yanıt: access_token (JWT), refresh_token, expires_in
- **POST /auth/register**
  - İstek: username, email, password, grade, lang, consent_flag
  - Yanıt: user_id, created_at
- **GET /v1/profile/{user_id}**
  - Yanıt: user_id, grade, lang, theta_math, theta_en, error_profiles {math, en}, segments, preferences
- **POST /v1/profile/update**
  - İstek: user_id, updates {theta_math?, theta_en?, error_profiles?}
  - Yanıt: success, updated_fields

### 3.3 Retrieval & Re‑rank
- **POST /v1/search**
  - İstek: query (metin veya {skills, level, topic}), lang, k (varsayılan 200)
  - Yanıt: candidates[] {item_id, type (math/en), retriever_scores, metadata}
- **POST /v1/rerank**
  - İstek: query_repr, candidates[] (en fazla 200), features (opsiyonel)
  - Yanıt: ranked[] (ilk 40), scores (cross‑encoder), reasons (ops.)

### 3.4 Öneri Akışı
- **POST /v1/recommend/next**
  - İstek: user_id, target_skills (liste) veya hedefler (curriculum boşlukları), constraints {difficulty_window, diversity}, slate_k
  - Yanıt: items[] (k adet) {item_id veya generated_item, reason_tags [skill, difficulty, peer], propensity}

### 3.5 Üretim — Matematik
- **POST /v1/generate/math**
  - İstek: template_id, params_hint (aralıklar), target_difficulty (|θ−b| hedefi), language, rationale_required (bool)
  - Yanıt: item_math {stem, choices?, answer_key, solution, skills, bloom, difficulty_estimate, QA_checks {solver_passed, single_gold}}

### 3.6 Üretim — İngilizce (Cloze)
- **POST /v1/generate/en_cloze**
  - İstek: level_cefr, target_error_tags (ör. prepositions/articles), topic (ops.), personalization (öğrencinin geçmiş hataları), ambiguity_tolerance
  - Yanıt: item_en {passage, blanks[{span, answer, distractors[], skill_tag}], QA_checks {grammar_valid, single_answer}}

### 3.7 Deneyimler & Ödüller
- **POST /v1/attempt**
  - İstek: user_id, item_id, answer, correct, time_ms, hints_used, context {device, session_id}
  - Yanıt: stored (bool), updated_theta? (math/en), reward_components
- **POST /v1/feedback**
  - İstek: user_id, item_id, rating (1–5), flags {too_hard, too_easy, misleading}, comment
  - Yanıt: stored (bool)

### 3.8 Yönetim & Gözlemlenebilirlik
- **GET /v1/admin/metrics**
  - Süzgeç: tarih aralığı, tenant
  - Yanıt: p95_latency, error_rate, cache_hit_rate, faithfulness, difficulty_match, coverage, exploration_ratio
- **GET /v1/admin/decisions/{request_id}**
  - Yanıt: policy_id, bandit_version, arms (propensity ve skorlar), chosen_arm_id

### 3.9 Güvenlik & Uygunluk
- **Her istekte**: auth zorunlu, consent kontrolü, PII redaction uygulandığına dair işaret
- **Ortak hata yanıtı**: code, message, details (şema ihlali, yetki, hız sınırı)

---

## 4) Doğrulama & İş Kuralları — Kod Yok
- Zorunlu alanlar açıkça belirtilir; sayısal alanlar için alt/üst sınırlar.
- Seviye eşlemesi: matematik için |θ−b| penceresi; İngilizce için CEFR uyumu.
- Distractor benzersizliği ve pedagojik uygunluk (öğrenci geçmiş hatalarıyla tutarlılık).
- Propensity logging zorunlu (bandit politikaları için).
- Audit izleri: her öneri ve üretim kararında model/policy versiyonu.

---

Bu doküman, **kod yazmadan** proje ekipleri arasında anlaşılabilir bir sözleşme sağlar. İhtiyaç halinde her bölüm, ilgili ekibe ayrıntılı teknik şemalara (DDL, OpenAPI YAML, pydantic modelleri) dönüştürülebilir.

