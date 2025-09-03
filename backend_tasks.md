# BACKEND_TASKS.md — FastAPI Tabanlı Adaptif Soru Öneri Sistemi

Aşağıdaki görev listesi, **Matematik & İngilizce adaptif soru öneri/üretim** projesinin FastAPI tabanlı backend’ini uçtan uca üretime götürmek için hazırlanmıştır. Görevler **P0 (kritik)**, **P1 (önemli)**, **P2 (iyileştirme)** olarak etiketlenmiştir.

> Not: Her görev için `Owner`, `Status` (Todo/Doing/Done/Blocked) alanı proje takip aracında (Jira/Linear) tutulmalıdır.

---

## 0) Proje İskeleti & Standartlar
- [ ] **P0**: Repo iskeleti
  - [ ] `/app` (FastAPI), `/infra`, `/scripts`, `/tests`, `/docs`, `/data`
  - [ ] Modül yapısı: `app/api`, `app/core`, `app/models`, `app/services`, `app/retrieval`, `app/ranking`, `app/generation`, `app/bandit`, `app/observability`
- [ ] **P0**: Kod standartları & pre-commit
  - [ ] Ruff/Black/isort, mypy (ops.)
  - [ ] Commitlint + Conventional Commits
- [ ] **P0**: .env yönetimi
  - [ ] pydantic Settings, `.env.example` oluştur
- [ ] **P1**: `Makefile` / `justfile` komutları (run/test/lint)
- [ ] **P1**: Dokümantasyon şablonu (`/docs`)

---

## 1) FastAPI Uygulaması
- [ ] **P0**: Uygulama bootstrap
  - [ ] `main.py` (lifespan, routers, CORS, GZip, rate limit)
  - [ ] OpenAPI (title, version, tags, servers)
- [ ] **P0**: API ana router’lar
  - [ ] `health`, `auth`, `recommend`, `generate`, `feedback`, `admin`
- [ ] **P1**: Exception/validation handler’ları (HTTPException, pydantic errors)
- [ ] **P1**: Versioning (`/v1`), deprecation notları

---

## 2) Kimlik, Güvenlik & RBAC
- [ ] **P0**: AuthN: JWT (access/refresh) veya OIDC entegrasyonu
- [ ] **P0**: RBAC: roller (`student`, `teacher`, `admin`, `service`)
- [ ] **P0**: **PII redaction** middleware (loglarda maskeleme)
- [ ] **P1**: Rate limiting (Redis tabanlı) & IP allowlist (admin/ops)
- [ ] **P1**: Secrets management (Vault/SM), key rotasyonu
- [ ] **P2**: Audit log (kimin, ne zaman, neye erişti)

---

## 3) Veri Katmanı (SQL + Feature Store)
- [ ] **P0**: SQL şema (PostgreSQL)
  - [ ] `users(id, grade, lang, consent, created_at, ...)`
  - [ ] `items_math(id, stem, params, solution, answer_key, skills[], bloom, difficulty_a, difficulty_b, lang, status)`
  - [ ] `items_en(id, passage, blanks(jsonb), level_cefr, topic, lang, status)`
  - [ ] `attempts(id, user_id, item_id, ts, answer, correct, time_ms, hints_used)`
  - [ ] `events(id, request_id, user_id, type, payload, ts)`
- [ ] **P0**: ORM (SQLModel/SQLAlchemy) modelleri
- [ ] **P1**: Feature Store
  - [ ] `user_theta_math`, `user_theta_en`, `error_profile_math`, `error_profile_en`
  - [ ] Online (Redis) + Offline (Postgres/BigQuery) senk akışı
- [ ] **P1**: Alembic migration’ları

---

## 4) Arama & Vektör DB (Retrieval)
- [ ] **P0**: Dense + Sparse hibrit
  - [ ] Sparse: OpenSearch/Vespa **veya** Postgres+BM25 (pgroonga/pg_trgm)
  - [ ] Dense: Qdrant/Weaviate/Milvus (HNSW ayarları)
- [ ] **P0**: Bölütleme & indeksleme job’ları (batch)
- [ ] **P0**: Metadata filtreleri (RBAC, dil, seviye, beceri)
- [ ] **P1**: Query rewrite (paraphrase/HyDE – koşullu)
- [ ] **P1**: Retrieval cache (TTL, invalidation stratejisi)

---

## 5) Re‑ranking Katmanı
- [ ] **P0**: Heuristic skor füzyonu (cosine+BM25+tazelik+otorite)
- [ ] **P0**: Cross‑encoder servis (ONNX/INT8/4‑bit, batch inference)
- [ ] **P1**: LTR (LambdaMART/LightGBM) eğitimi ve çevrimiçi entegrasyon
- [ ] **P2**: LLM‑scoring (dar K≤10, kritik durumlar)

---

## 6) Öğrenci Modelleme (IRT/BKT/DKT)
- [ ] **P0**: IRT 2PL çevrimiçi güncelleme servisi
- [ ] **P1**: BKT/DKT (seq) eğitim & tahmin
- [ ] **P1**: Error embedding (user2vec/item2vec) job’ları
- [ ] **P2**: Peer‑aware komşuluk (k‑NN) API’si

---

## 7) Bandit Politikaları
- [ ] **P0**: Epsilon‑greedy politika (quickstart)
- [ ] **P1**: LinUCB/LinTS context schema (θ, skill one‑hot, device, recency)
- [ ] **P1**: Constrained bandit (min başarı, coverage≥80%)
- [ ] **P1**: Propensity logging + IPS/DR offline değerlendirme job’ları

---

## 8) İçerik Üretimi — Matematik
- [ ] **P0**: Parametrik şablon motoru (doğrusal denklem, oran‑orantı, geometri vb.)
- [ ] **P0**: Programatik çözücü (sympy) ile **tek‑altın** doğrulama
- [ ] **P1**: Yanılgı tabanlı distractor üretimi (kural + LLM destekli)
- [ ] **P1**: LLM rationale üretimi (explanation), stil kontrolü
- [ ] **P2**: Otomatik zorluk kalibrasyonu (|θ−b|≈0.3 hedefleyen parametrizasyon)

---

## 9) İçerik Üretimi — İngilizce (Cloze)
- [ ] **P0**: Hata taksonomisi & confusion sets (articles, prepositions, SVA, collocations…)
- [ ] **P0**: Pasaj seçim & seviye (CEFR) sınıflayıcı
- [ ] **P0**: Cloze üretim pipeline’ı (kural → LLM)
- [ ] **P0**: Tek‑cevap garantisi (grammar check + ambiguity test)
- [ ] **P1**: Kişiselleştirilmiş distractor (geçmiş hata profilinden)
- [ ] **P2**: Metin çeşitlendirme (topic rotation, MMR)

---

## 10) LLM Serving & Orkestrasyon
- [ ] **P0**: Model router (küçük ↔ büyük model; cost‑aware)
- [ ] **P0**: vLLM/TGI deployment (open‑source modeller)
- [ ] **P0**: Prompt şablonları (RAG/Template), JSON schema enforcement
- [ ] **P1**: Tool‑use: math solver, grammar checker, SQL retriever
- [ ] **P1**: Semantic cache (prompt_sig→cevap), eşik ayarı

---

## 11) API Yüzeyleri
- [ ] **P0**: `GET /health`
- [ ] **P0**: `POST /v1/recommend` (girdi: user_id, hedef beceri/konu; çıktı: aday liste)
- [ ] **P0**: `POST /v1/generate/math` (parametrik şablon + opsiyonel LLM rationale)
- [ ] **P0**: `POST /v1/generate/en_cloze` (hedef hata türü + seviye)
- [ ] **P0**: `POST /v1/attempt` (cevap kaydı)
- [ ] **P1**: `POST /v1/feedback` (öğrenci/öğretmen geri bildirimi)
- [ ] **P1**: `GET /v1/profile/{user_id}` (θ, hata profili, öneri geçmişi)
- [ ] **P2**: `GET /v1/admin/metrics` (SLO/KPI)

---

## 12) Observability (Log/Trace/Metric)
- [ ] **P0**: OpenTelemetry entegrasyonu (FastAPI middleware)
- [ ] **P0**: Metrikler: `retrieval_ms`, `rerank_ms`, `llm_ms`, `p95_latency`, `cache_hit_rate`, `faithfulness`
- [ ] **P1**: Request/Decision log şemaları (propensity, chosen_arm, scores)
- [ ] **P1**: Alert kuralları (latency spike, error rate, faithfulness drop)

---

## 13) Test & Kalite
- [ ] **P0**: Unit test’ler (retrieval, re‑rank, generators, bandit)
- [ ] **P0**: Contract test’ler (API şemaları, JSON schema)
- [ ] **P1**: Eval harness (Recall@K, nDCG, MAP, Difficulty‑Match, Faithfulness)
- [ ] **P1**: Adversarial/robustness test’leri (prompt injection, ambiguities)
- [ ] **P2**: Load test (Locust/k6), p95/p99 raporları

---

## 14) CI/CD & Dağıtım
- [ ] **P0**: GitHub Actions/GitLab CI (lint, test, build, SBOM)
- [ ] **P0**: Dockerfile + docker‑compose (dev), Helm chart (prod)
- [ ] **P0**: Env matrisleri (dev/stage/prod), secrets mount
- [ ] **P1**: Canary deploy + feature flags (bandit/policy switch)
- [ ] **P2**: Blue‑green / rollback playbook

---

## 15) Güvenlik & Uygunluk
- [ ] **P0**: Moderation/guardrails (toxicity, jailbreak, schema validation)
- [ ] **P0**: Erişim günlükleri (audit), veri yaşlandırma (TTL)
- [ ] **P1**: Privacy by design (min veri, şifreleme), DPIA şablonu
- [ ] **P2**: Differential Privacy / Federated (opsiyonel)

---

## 16) Operasyonel SLO/SLA & Runbook
- [ ] **P0**: SLO’lar: p95 latency, uptime, error rate, cache hit, difficulty‑match≥80%
- [ ] **P0**: Olay yönetimi: incident triage (retrieval vs. rerank vs. LLM), iletişim şablonları
- [ ] **P1**: Haftalık kalite review: learning gain, coverage, fairness dilimleri

---

## 17) Yol Haritası Fazları (kaba sıra)
- [ ] **Faz 1**: FastAPI iskelet, SQL şema+ORM, hibrit retrieval, cross‑enc, temel IRT, ε‑greedy, Math basic templates, Cloze MVP, logging.
- [ ] **Faz 2**: LinUCB/LinTS, kişiselleştirilmiş distractor, DKT, curriculum diversification, semantic cache, tool‑use.
- [ ] **Faz 3**: Constrained bandit, LTR, human‑in‑the‑loop editör UI, RWKV/Mamba (ops.), ileri gözlemlenebilirlik ve A/B testleri.

---

## 18) Ekler
- [ ] DDL taslakları (Postgres) — `/infra/sql`
- [ ] OpenAPI şeması — `/docs/openapi.yaml`
- [ ] Prompt şablonları — `/docs/prompts`
- [ ] Değerlendirme defterleri — `/docs/eval`

