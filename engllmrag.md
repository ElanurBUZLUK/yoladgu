# EN – RAG+LLM Soru Üretimi Planı (Hata‑Odaklı)

Bu doküman, öğrencinin geçmiş **yanlışları** ve **hata desenleri** kullanılarak İngilizce soru üretimi için uçtan uca mimari, algoritmalar, teknoloji seçenekleri ve senaryoları içerir. P0–P3 omurgasıyla tam uyumludur (pgvector, LLM router, UsageLogger, GDPR, offline eval, SFT/DPO).

---

## 1) Hedefler

- **Kişiselleştirme:** Öğrencinin sık yaptığı hataları hedefleyen sorular üretmek.
- **Kalite:** Şema/doğruluk/okunabilirlik garantileri (guardrail) ile üretim.
- **Maliyet & Gecikme:** Token bütçesi ve p95 gecikme sınırları içinde çalışmak.
- **İzlenebilirlik:** Soru üretiminde kullanılan bağlam, model sürümü, kalite skorları ve maliyeti kayıt altına almak.

---

## 2) Girdi Sinyalleri (Minimal çekirdek)

- **Incorrect logları:** `attempts (is_correct=false)` + hata türleri (`grammar_error`, `vocabulary_error`, `article_usage`, `tense` …).
- **Zaman bilgisi:** Recency (son 7/14 gün), streak, response time.
- **Seviye:** CEFR tahmini (MCP.web `cefr_estimate` veya klasik kurallar) + platform seviyeleri.
- **Kısıtlar:** İstenilen format (MCQ/Fill‑Blank/Transform), hedef token bütçesi, hedef seviye.

---

## 3) Retrieval Katmanı (RAG)

- **Varlıklar:**
  - `english_texts`, `grammar_rules`, `vocab_lexicon`, `generated_questions` (benzer soru/distractor havuzu).
- **Depo:** `pgvector` (mevcut). Alternatifler: **Qdrant**, **Milvus** (çok büyük veri setinde).
- **Embedding seçenekleri:**
  - Hosted: `text-embedding-3-small` (OpenAI) – iyi kalite, kolay.
  - OSS: `bge-base-en` / `bge-small-en` – düşük maliyet, lokal vLLM ile uyum.
- **BT (Birim/Token) bütçesi için özetleme:** Çoklu chunk’ı **hierarchical compression** (map→reduce) ile 800–1200 token’a sıkıştır.
- **Hybrid arama:** Dense (ANN) + BM25 → **MMR (λ=0.7)** ile çeşitlendirme.
- **Filtreler:** CEFR bandı ±1, yakın tarih, hata türü eşleşmesi.

> **Not:** P3 canvas’taki `vector_index_manager.get_active_slot` ile **blue/green** slot okuması korunur.

---

## 4) Üretim Kalıbı (Patterns)

- **Structured output / JSON Schema**: MCQ, Cloze, Error‑Correction için ayrı şemalar.
- **Critic & Revise**: 2‑aşamalı üretim; 1) taslak, 2) eleştirmen (rubric) ile düzeltme.
- **Self‑consistency @k**: 2–3 örnek arasında çoğunluk oylaması (hafif).
- **Constrained decoding**: JSON uyumluluğu (regex/`json_repair`), CEFR‑okunabilirlik kontrolü.
- **Safety/Moderasyon**: toksisite, hassas içerik, uygun uzunluk.

---

## 5) Prompt Tasarımı (özet)

**System:**

- “You are an English pedagogy expert. Produce a single JSON conforming to the schema. Keep CEFR at {target\_cefr}. Avoid rare idioms. Ensure one unique correct answer.”

**User (dinamik):**

- `error_focus`: en sık 1–3 hata (örn. `article_usage`, `verb_tense`)
- `compressed_context`: RAG sıkıştırılmış özetleri (örnek yanlışlar, açıklamalar)
- `format`: `mcq|cloze|error_correction`
- `difficulty`: 1–5 (CEFR ile eşlenir)

**Rubric (critic aşaması):**

- “Reject if: schema invalid, >N tokens, ambiguous key, distractors too similar, CEFR drift.”

---

## 6) Algoritmalar (çekirdek seçim & üretim)

### 6.1 Hata‑Odaklı Bağlam İnşası

- **Recency×Frequency skoru**: \(s = 0.6·freq_z + 0.4·recency_z\) (z‑normalize).
- En yüksek skorlu 2–3 hata türü hedeflenir; her biri için 1–2 örnek yanlış (öğrenci geçmişi) + 1 referans açıklama (grammar\_rules) alınır.

### 6.2 Bağlam Sıkıştırma (Adaptive Context Compression)

- Importance filter → CEFR/konu uyumlu cümleleri koru.
- Nuggetization: her chunk’tan **kısa “öğretici iddia”** çıkar (max 2 cümle).
- Token > bütçe ise **dedup & merge** (Levenshtein/Jaccard) uygula.

### 6.3 Zorluk Kalibrasyonu

- CEFR hedefi ±1 aralıkta; **LLM‑judge** ile “target\_cefr?” sahte sorusu sorularak self‑eval (yes/no). Uygun değilse revise.

### 6.4 Distractor Üretimi (MCQ)

- **Minimal pair** yaklaşımı (a/an, much/many, tense shifts).
- **Common misconception mining:** `generated_questions` + yanlış cevap dağılımından popüler hatalar.
- **Semantic distance**: doğru cevaba yakın ama ayırt edici (cosine 0.65–0.85).

### 6.5 Kalite Kapıları

- JSON şema doğrulama → LLM output repair → CEFR check → Moderation → **De‑dup** (vektörel benzerlik < 0.92).

---

## 7) Teknoloji Seçenekleri

- **LLM yönlendirme:** P3 `llm_router` (OpenAI/Anthropic/Vertex/vLLM), `MAX_USD_PER_REQUEST` ≤ 0.05\$.
- **Embedding:** OpenAI (`text-embedding-3-small`) / OSS `bge-small-en`.
- **Reranker:** BGE‑reranker‑base veya LLM‑judge @low‑tokens.
- **Guardrails:** `pydantic` schema + `jsonschema` + hızlı okunabilirlik/CEFR heuristic.
- **Moderation:** OpenAI mod veya basit kural seti (yasak kelimeler + uzunluk sınırları).
- **Cache:** Semantic cache (prompt\_hash → output, TTL 24h), RAG context cache 30m.

---

## 8) API Sözleşmesi (öneri)

**POST** `/api/v1/english/rag/generate-question`

```json
{
  "format": "mcq|cloze|error_correction",
  "difficulty": 3,
  "target_cefr": "B1",
  "max_tokens": 600,
  "constraints": {"unique_correct": true, "max_options": 4}
}
```

**200**

```json
{
  "question": {"id":"uuid","format":"mcq","content":"...","options":["A","B","C","D"],"answer":"B"},
  "provenance": {"model":"gpt-4o-mini","provider":"openai","ctx_items":3,"prompt_hash":"...","slot":"green"},
  "quality": {"schema_ok":true,"cefr":"B1","toxicity":0.0,"novelty":0.11},
  "cost_usd": 0.0041,
  "latency_ms": 820
}
```

---

## 9) Pipeline Pseudocode (servis içi)

```python
ctx = build_context(user_id, error_focus, cefr, budget_tokens)
compressed = compress_context(ctx, budget=900)
first = llm_router.run(render_prompt(compressed, format, difficulty))
fixed = repair_json(first["text"])  # jsonrepair/regex
if not schema_valid(fixed):
    second = critic_revision(fixed, rubric)
    fixed = repair_json(second)
checks = [cefr_check(fixed, target_cefr), moderation(fixed), novelty_check(fixed)]
if all(checks):
    save_question(fixed, provenance=…)
    return fixed
else:
    fallback = generate_minimal_pair(error_focus)
    return fallback
```

---

## 10) Değerlendirme & İzleme

- **Offline Eval:** accept\@1 (doğru cevap eşleşmesi), **distractor plausibility** (insan etiketli alt set), **novelty** (vektör benzerlik eşik).
- **Canlı Telemetri:** üretim başarı oranı, schema‑repair oranı, CEFR drift %’si, benzersizlik, p95 latency, token maliyeti.
- **A/B Deneyleri:** Prompt şablonu, critic‑revise açık/kapalı, reranker varyantı.

---

## 11) Veri Yönetimi & GDPR

- `generated_questions` satırlarına **provenance**: `{prompt_hash, context_ids, model_ver, policy_ver, creation_ts}`.
- **PII Scrub:** öğrenci metinlerinden email/telefon maskesi (P3 `build_sft_corpus.py` yaklaşımı).
- **Export/Delete:** `privacy_requests` ile uyumlu; üretimde kullanılan öğrenci‑özel metinler **anonimlenmiş** referanslarla tutulur.

---

## 12) Senaryolar (akış örnekleri)

1. **Yanlış Tekrarı (A/B articles)**: son 14 günde `article_usage` hatası yüksek → cloze MCQ üret, minimal‑pair distractor’lar.
2. **Zaman Hatası (Past Tense)**: `verb_tense` hata skoru yüksek → kısa bağlam + tense odaklı MCQ, CEFR=B1.
3. **Hızlı Etüt (Low budget)**: `max_tokens=350` → tek örnek yanlış + kısa kural özeti; critic aşaması kapalı.
4. **İyileşme Testi**: art arda 3 doğru → CEFR +1, aynı hatadan yeni cümle varyantı; novelty zorunlu.
5. **Düşük Motivasyon**: son 5 soruda uzun süre/yanlış → kolaylaştırılmış cloze + pozitif geri bildirim şablonu.

---

## 13) Yapılandırma (örnek)

```yaml
qg:
  target_cefr_default: B1
  max_tokens_default: 600
  schema: mcq
  critic:
    enabled: true
    retries: 1
  reranker:
    kind: "bge-reranker-base"
    top_k: 6
  novelty:
    similarity_threshold: 0.92
  retrieval:
    k_dense: 12
    k_bm25: 6
    mmr_lambda: 0.7
  budget:
    max_usd_per_request: 0.05
```

---

## 14) Yol Haritası (kısa)

- **v1 (2 hafta):** Hybrid RAG + critic&revise + schema repair + CEFR kontrol + temel moderasyon.
- **v1.1:** Minimal‑pair generator kütüphanesi; novelty/dup kontrolün iyileştirilmesi.
- **v1.2:** Reranker’ı LLM‑judge ile hibrit; self‑consistency @k; semantic cache 24h.
- **v2:** Öğrenci‑özel hata sınıflandırıcı (tiny classifier) + otomatik seviye drift düzeltme.

> Bu plan, mevcut P3 router, UsageLogger, GDPR ve offline eval boru hattına doğrudan bağlanır; tek dosyalık API eklemesi ile yayınlanabilir.



---

# 15) Uçtan Uca Akış (DB → RAG → LLM → Guardrails → Kayıt → Yanıt)

**İstek Girdisi**: `{user_id, target_cefr, format(mcq|cloze|error_correction), difficulty(1-5), max_tokens, error_focus?(opsiyonel)}`

1. **Profil & Hata Verisi Çekimi (DB)**

   - `attempts` tablosundan son 90 gün `is_correct=false` kayıtları (yanıt, süre, zaman damgası).
   - `error_patterns` özetleri (tür, sıklık, son görülme).
   - `generated_questions` (benzer soru + yanlış dağılımı) → **distractor mining**.
   - **Örnek SQL**:

   ```sql
   -- Yanlışlar (recency ağırlıklı seçim için)
   SELECT question_id, student_answer, error_tags, created_at, time_taken
   FROM attempts
   WHERE user_id = :uid AND is_correct = false AND created_at >= NOW() - INTERVAL '90 days'
   ORDER BY created_at DESC
   LIMIT 500;

   -- Hata özetleri
   SELECT error_type, COUNT(*) AS freq, MAX(created_at) AS last_seen
   FROM attempts
   WHERE user_id = :uid AND is_correct = false
   GROUP BY error_type
   ORDER BY freq DESC;

   -- Popüler yanlış seçenekler (misconception mining)
   SELECT g.id, g.content, g.options, g.answer_idx, g.stats_wrong_counts
   FROM generated_questions g
   WHERE g.cefr BETWEEN :low_cefr AND :high_cefr
   ORDER BY g.created_at DESC
   LIMIT 200;
   ```

2. **Hedef Hata Kümesi Seçimi**

   - Skor: `s = 0.6 * z(freq) + 0.4 * z(recency)` (son 14 gün için recency).
   - En yüksek 1–3 hata etiketi → `error_focus`.

3. **Retrieval (Hybrid)**

   - **Dense**: `pgvector` ile `english_texts.content_embedding <=> :query_vec` sıralı seçim.
   - **Sparse**: `tsvector/tsquery` (BM25 benzeri) ile eşleştirme.
   - **Fusion**: RRF (k=60) + **MMR** (λ=0.7) ile çeşitlendirme.
   - **Filtre**: CEFR band ±1, `created_at` ≤ 365g, `length` & `readability` eşiği.
   - **Örnek SQL (dense)**:

   ```sql
   SELECT id, title, content, 1 - (content_embedding <=> :qvec) AS sim
   FROM english_texts
   WHERE content_embedding IS NOT NULL
   ORDER BY content_embedding <=> :qvec ASC
   LIMIT 50;
   ```

   - **Örnek SQL (sparse)**:

   ```sql
   SELECT id, title, content,
          ts_rank_cd(to_tsvector('english', content), plainto_tsquery('english', :q)) AS rank
   FROM english_texts
   WHERE to_tsvector('english', content) @@ plainto_tsquery('english', :q)
   ORDER BY rank DESC
   LIMIT 50;
   ```

4. **Bağlam Sıkıştırma (Map→Reduce)**

   - Map: her pasajdan 1–2 cümlelik **öğretici nugget** çıkar.
   - Reduce: tekrar eden cümleleri **dedup/merge** (Jaccard/Levenshtein), hedef **≤ 900–1200 token**.
   - CEFR filtresi ve **hata-etiketi** kapsama kontrolü.
   - Çıktı: `compressed_context` (metin + `source_ids`).

5. **Üretim (LLM) – Taslak**

   - Prompt şablonu (system+user) → tek bir **JSON** döndür.
   - Şema seçimi: `mcq`, `cloze`, `error_correction`.
   - `temperature≈0.5–0.7`, `max_tokens≤config`.
   - Yanıt `text` → **JSON repair** (regex/`jsonrepair`), `schema.validate()`.

6. **Critic & Revise**

   - Critic rubric: `one_correct`, **CEFR**, bağlama dayalı, **leakage yok**, uzunluk & okunabilirlik.
   - `valid=false` ise **revise** prompt’u ile tek revizyon turu.
   - `self-consistency@k` (opsiyonel, k=2–3) → çoğunluk oyu.

7. **Guardrails**

   - **Schema**: alanların tipi ve zorunluluk kontrolü.
   - **Only-one-correct**: cevap ve distraktörler `!=` & semantik mesafe (0.65–0.9).
   - **CEFR**: sözlük tabanlı/heuristic veya hafif sınıflandırıcı.
   - **Moderation**: toksik kelime listesi, uygunsuz içerik.
   - **Novelty/De-dup**: önceki içeriklere vektör benzerlik `< 0.92`.

8. **Kayıt & İndeks Güncelleme**

   - `generated_questions` satırı: `content, options, answer_idx, explanation, tags, cefr, provenance{prompt_hash, context_ids, model, policy_ver, latency_ms, cost_usd}`.
   - Yeni sorunun embedding’i hesaplanır → `pgvector` kolonu güncellenir → **ANN index** sıcak kalır.
   - Semantic cache: `prompt_hash → output` (TTL 24h).

9. **Yanıtın İletilmesi**

   - API 200: `{question, provenance, quality, cost_usd, latency_ms}`.
   - Header: `ETag: prompt_hash`, `Cache-Control: private, max-age=300`.
   - Usage telemetry: `token_usage` yazımı + OTel span’ları.

---

# 16) Veri Sözleşmesi (Minimal Şemalar)

**generated\_questions**\
`(id uuid, format text, content text, options jsonb, answer_idx int, explanation text, cefr text, tags text[], context_ids text[], model text, provider text, policy_ver text, prompt_hash text, created_at timestamptz, content_embedding vector(1536))`

**attempts**\
`(id uuid, user_id uuid, question_id uuid, is_correct bool, student_answer text, error_tags text[], time_taken int, created_at timestamptz)`

**english\_texts**\
`(id uuid, title text, content text, cefr text, created_at timestamptz, content_embedding vector(1536))`

**error\_patterns**\
`(id uuid, user_id uuid, error_type text, freq int, last_seen timestamptz, embedding vector(1536))`

---

# 17) Retrieval Ayrıntıları (RRF + MMR)

**RRF (Reciprocal Rank Fusion)**\
`score(d) = Σ 1/(k + rank_list_i(d))` (k≈60). Dense ve sparse listeleri tek skorla birleştir.

**MMR**\
Seçim döngüsü: `argmax_i [ λ·sim(q, d_i) − (1−λ)·max_{d_j∈S} sim(d_i, d_j) ]`\
`λ=0.7`, `|S|=top_k`.

**Pseudocode**

```python
candidates = rrf(dense_top50, sparse_top50, k=60)
selected = []
while len(selected) < top_k:
  best = argmax(candidates, key=lambda d: λ*sim(q,d) - (1-λ)*max_sim(d, selected))
  selected.append(best); candidates.remove(best)
```

---

# 18) Bağlam Sıkıştırma (Map→Reduce)

**Map**: her pasajdan `nugget = 1–2 cümle` (kural/örnek/kontrast).\
**Reduce**: `dedup_merge(nuggets)`, `limit_tokens(≤1200)`, **CEFR filtre**.\
**Pseudocode**

```python
nuggets = [extract_nugget(p) for p in passages]
unique = dedup_by_sim(nuggets, thr=0.9)
compressed = truncate_to_budget(unique, budget_tokens)
```

---

# 19) Üretim Şemaları ve Örnekleri

**MCQ JSON Şema (özet)**

```json
{"type":"object","required":["content","options","answer_idx","explanation","tags","cefr"],
 "properties":{"content":{"type":"string"},
  "options":{"type":"array","minItems":4,"maxItems":4,"items":{"type":"string"}},
  "answer_idx":{"type":"integer","minimum":0,"maximum":3},
  "explanation":{"type":"string"},
  "tags":{"type":"array","items":{"type":"string"}},
  "cefr":{"type":"string"}}}
```

**Cloze & Error‑Correction**: seçenek alanı yerine `blanks` veya `corrections` dizisi.

**Constrained decoding ipuçları**: `json-only`, sayı alanlarına `min/max`, açıklama ≤ 120 token.

---

# 20) Critic → Revise Rubric (JSON)

```json
{
  "reject_if":{
    "schema_invalid": true,
    "multiple_correct": true,
    "cefr_drift": true,
    "context_not_used": true,
    "toxic_or_biased": true
  },
  "checks":["one_unique_answer","distractor_plausibility","readability","length<=120t"],
  "suggest": ["simplify_vocab","strengthen_contrast","fix_leakage"]
}
```

**Akış**: draft → critic(JSON) → `valid?` evet→geç; hayır→revise(prompt+critique) → tekrar **schema/guardrail**.

---

# 21) Guardrail Doğrulamaları (Detay)

- **Schema**: `jsonschema` ile.
- **One‑correct**: `options[answer_idx]` benzersiz; diğerleri ≠ doğru; `semantic_sim(distractor, correct)` aralık 0.65–0.9.
- **CEFR**: kelime listeleri + hafif sınıflandırıcı/heuristic; `drift` olduğunda reject.
- **Moderation**: yasaklı kelimeler + kısa heurisikler (imperative/toxic).
- **Novelty**: `cosine(new.content_embedding, historical) < 0.92`.
- **Length**: soru ≤ 80 token; açıklama ≤ 120 token (ayarlandabilir).

---

# 22) Kayıt & İndeksleme & Kimlik

- **Provenance**: `{prompt_hash, context_ids, model, provider, policy_ver, creation_ts}`.
- **Idempotency**: `prompt_hash` üzerinde benzersizlik; aynı istekte tekrar üretim yerine **cache hit**.
- **Embedding**: üretim sonrası `content_embedding` güncelle; `ivfflat` index sıcak.
- **TTL & Cache**: semantic cache 24h; RAG context 30m.

---

# 23) API Katmanı (Ayrıntı)

**İstek**\
`POST /api/v1/english/rag/generate-question`

```json
{
  "format": "mcq",
  "difficulty": 3,
  "target_cefr": "B1",
  "max_tokens": 600,
  "constraints": {"unique_correct": true}
}
```

**Başlıklar**: `Authorization: Bearer …`, `X-Request-Id`, `X-User-Id`.

**Yanıt (200)**: `{question, provenance, quality, cost_usd, latency_ms}`\
**Hatalar**: 400 (schema), 409 (idempotency conflict), 429 (rate), 500 (provider).\
**SSE/Streaming (opsiyonel)**: critic aşamasını akışla yayınla; finalde **ETag** gönder.

---

# 24) Gözlemlenebilirlik & Maliyet

- **OTel Spans**: `db.fetch_incorrect`, `rag.retrieve`, `compress.map`, `llm.generate`, `critic`, `guardrails`, `persist`.
- **Prometheus**: `qg_latency_ms`, `qg_repair_ratio`, `qg_cefr_drift_ratio`, `qg_novelty_violation`, `qg_cost_usd`.
- **SLO**: p95 < 1.2s, başarısızlık oranı < %2, maliyet/istek < \$0.05.

---

# 25) Hata Senaryoları & Geri Dönüşler

- **LLM time‑out** → router fallback (ikinci sağlayıcı) → olmazsa **rule‑based minimal‑pair** üret.
- **Schema parse** başarısız → `jsonrepair` + tek revise; yine olmazsa 400.
- **CEFR drift** → revise ile basitleştir; başarısızsa `target_cefr-1`.
- **Novelty çakışması** → yeni cümle şablonu ile re‑generate (aynı hata deseninde).
- **DB kilidi/erişim** → retry w/ jitter, DLQ’ye yaz.

---

# 26) Offline Eval & SFT/DPO

- **Eval**: accept\@1, distractor‑plausibility (insan puanı), novelty@τ, cefr‑drift@%.
- **Kayıt**: `eval_runs` + özet metrikler.
- **SFT**: `sft_corpus`’a `(input:compressed_ctx+rubric, output:final_json)`; PII scrub.
- **DPO**: critic‑rejected vs accepted ikilileri ile tercih verisi.

---

# 27) Güvenlik & GDPR

- **PII Scrub**: e‑posta/telefon maskesi; öğrenci serbest metinlerinden kimlikleyici kaldır.
- **Export/Delete**: üretimde kullanılan `context_ids` yalnızca anonim referanslar; isteğe bağlı `privacy_requests`.
- **Rate Limit**: auth bazlı anahtar ile 120/min; kritik uçlarda 2000/saat.
- **İzinler**: öğretmen/öğrenci/adm rol ayrımı; üretim uçları sadece yetkiliye.

---

# 28) Yapılandırma Örneği (YAML – geniş)

```yaml
qg:
  cefr_default: B1
  format_default: mcq
  max_tokens: 600
  retrieval:
    dense_top_k: 50
    sparse_top_k: 50
    rrf_k: 60
    mmr_lambda: 0.7
    cefr_band: 1
    max_age_days: 365
  compression:
    target_tokens: 1000
    max_tokens: 1200
  critic:
    enabled: true
    retries: 1
  novelty:
    threshold: 0.92
  moderation:
    enabled: true
  cache:
    semantic_ttl_sec: 86400
    context_ttl_sec: 1800
  budget:
    max_usd_per_request: 0.05
```

---

# 29) Ayrıntılı Pseudocode (E2E)

```python
async def generate_for_user(user_id, target_cefr, fmt, difficulty, max_tokens):
  with otel_span('qg.request') as span:
    # 1) DB → geçmiş yanlışlar & özetler
    wrongs = fetch_incorrect(user_id)
    errs = summarize_errors(wrongs)  # recency×frequency top-3

    # 2) Query formülasyonu
    qtext = build_query_text(errs, target_cefr, difficulty)
    qvec  = embed(qtext)

    # 3) Hybrid retrieval
    dense = dense_search(qvec)
    sparse = sparse_search(qtext)
    fused = rrf(dense, sparse, k=60)
    picked = mmr(fused, qvec, λ=0.7, top_k=8)

    # 4) Compression
    compressed = compress(picked, budget=1000, cefr=target_cefr)

    # 5) LLM draft
    prompt = render_prompt(compressed, errs, target_cefr, fmt, difficulty)
    draft = llm_router(prompt, max_tokens=max_tokens)
    json1 = json_repair(draft.text)
    if not schema_ok(json1):
      critic = run_critic(json1, compressed, target_cefr)
      revised = llm_router(revise_prompt(json1, critic, compressed))
      json1 = json_repair(revised.text)

    # 6) Guardrails
    if not all([
      schema_ok(json1),
      one_correct(json1),
      cefr_ok(json1, target_cefr),
      moderation_ok(json1)
    ]):
      return fallback_minimal_pair(errs, target_cefr, fmt)

    # 7) Novelty & persist
    emb = embed(json1['content'])
    if is_duplicate(emb, thr=0.92):
      json1 = regenerate_variant(json1, errs)
      emb = embed(json1['content'])

    qid = save_generated(json1, provenance(compressed, prompt))
    update_embedding(qid, emb)

    # 8) Yanıt
    return response(qid, json1, quality_scores(json1), costs(), latency())
```

---

# 30) Kontrol Listesi (Yayın Öncesi)

-

