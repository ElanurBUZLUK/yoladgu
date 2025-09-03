# Adaptif Soru Öneri Sistemi (Matematik & İngilizce) — Tam Mimari Plan v1

Bu belge, öğrencinin geçmiş performansına dayanarak **Matematik** ve **İngilizce** için yeni sorular **önermek** ve/veya **üretmek** üzere uçtan uca bir sistemin tasarımını içerir. Hedef: seviyeye uygunluk, hatalardan öğrenme, pedagojik çeşitlilik ve üretim güvencesi.

---

## 0) Hedefler & Kapsam
- **Matematik**:
  - Öğrenci yeteneğini (θ) IRT/BKT/DKT ile tahmin etmek.
  - **Doğru yaptığı sorulardan** temel seviye tahmini + **yanlış örüntülerden** kavram yanılgılarını tespit.
  - Benzer hataları yapan akranların zorlandığı soruları **öner** (peer‑aware collaborative sinyal).
  - Üretim: Şablon + LLM ile parametrik soru üretimi; çözüm denetimi için programatik çözümleyici.
- **İngilizce**:
  - Geçmiş hata türlerine göre (grammar, vocab, preposition, collocation) **boşluk doldurma (cloze)** soruları üret.
  - Öğrenciye özgü yanlış seçenek (distractor) türetme; seviye uyumlu metinler.
- **Ortak**: Kişiselleştirme, çeşitlendirme (MMR/curriculum), kısıtlı bandit ile adaptif akış, güvenlik & kalite kontrol, çevrimiçi/çevrimdışı değerlendirme.

---

## 1) Üst Düzey Mimari (metinle diyagram)
```
[İstemci] -> [API Gateway] -> [Auth/RBAC] -> [PII Redactor]
      |                                   \
      |                                    -> [Telemetry/Event Bus]
      v
[Orkestratör]
  ├─ [Profil/Geçmiş]  -> User Store (θ, hata vektörü, segmentler)
  ├─ [İçerik Katmanı] -> Item Bank (etiketli) + Vektör DB (dense/sparse)
  ├─ [Retrieval]      -> Hybrid (BM25+dense) + metadata filtresi
  ├─ [Re-ranking]     -> Cross-encoder (200→40) + LTR (ops.)
  ├─ [Çeşitlendirme]  -> MMR + curriculum gap filler
  ├─ [Zorluk Uyum]    -> IRT/BKT/DKT + kısıtlı bandit (LinUCB/LinTS)
  ├─ [Üretim (LLM)]   -> Template+LLM / RAG (gerekirse) / Matematik çözücü
  ├─ [Kalite Güvencesi] -> Schema, answer-key, pedagojik kurallar, toxicity
  └─ [Teslim]         -> UI + açıklama + gerekirse ipucu

[Feedback Döngüsü] -> (doğru/yanlış, süre, ipucu kullanımı, zorlanma) -> Feature Store
```

---

## 2) Veri Şemaları (özet)
- **User**: {user_id, grade, lang, θ_math, θ_en, error_profile_math{skill→rate}, error_profile_en{tag→rate}, consent}
- **Item (Math)**: {item_id, stem, params, solution, answer_key, skills[], bloom, difficulty{IRT a,b}, lang, generator, citations?, status}
- **Item (EN)**: {item_id, passage, blanks[{span,skill,answer,confusions[]}], level(CEFR), topic, lang}
- **Attempt**: {user_id, item_id, ts, choice/answer, correct, time_ms, hints_used, feedback}
- **Retrieval Log**: {request_id, candidates[doc_id, scores], features}
- **Decision**: {request_id, policy_id, bandit_version, arms[], chosen_arm_id, propensity}

---

## 3) Matematik Modülü
### 3.1 Beceri Ontolojisi & Etiketleme
- Kategoriler: Aritmetik, Cebir, Fonksiyon, Geometri, Olasılık, İstatistik…
- Beceri/alt-beceri (örn. "doğrusal denklem çözme", "benzerlik").
- Her soruda `skills[]` ve Bloom düzeyi (remember→create) etiketleri.

### 3.2 Seviye Tahmini
- **IRT 2PL (a,b)** başlangıç; veri yeterliyse 3PL (guessing c) eklenir.
- **BKT/DKT**: uzun dönem beceri kazanımı ve unutma modellemesi.
- θ güncelleme: çevrimiçi **EM** veya **stochastic gradient** ile per-oturum.

### 3.3 Akran‑Benzerlik ve Hata Madenciliği
- **Error embedding**: Öğrenci×Beceri matrisinden (doğru/yanlış/süre) **item2vec/user2vec** türet.
- **Peer-aware**: Benzer hata vektörlerine sahip komşuların (k‑NN) başarısız olduğu, fakat hedef öğrencinin θ ile uyumlu `b≈θ` soruları öne çıkar.
- **Misconception taxonomy**: Örn. "işlem önceliği karışıklığı", "işaret hatası", "benzerlik oranı".

### 3.4 Soru Üretimi (Template + LLM)
- **Parametrik şablonlar**: Jeneratif değil, önce kurallı üretim → güvenilirlik.
  - Örn. lineer denklem: `ax + b = c` → param üret, `a≠0`, `a,b,c ∈ Z`, kontrol: çözüm tam sayı.
- **Çözümleme**: Programatik çözücü (sympy/py) ile **tek doğru** doğrulanır.
- **Açıklama (rationale)**: LLM ile dilsel açıklama (kaynak: şablon adımları).
- **Zorluk kalibrasyonu**: Param aralıklarını `b` çevresine göre ayarla (|θ−b|≈0.3).
- **Distractor üretimi**: Yanılgı tabanlı – tipik yanlış adımların çıktıları distractor.

### 3.5 Aday Seçimi & Sıralama
1) **Retrieval**: Item Bank’tan hybrid (BM25+dense) + beceri & seviye filtresi.
2) **Re‑rank**: Cross‑encoder (query: hedef beceri + öğrenci bağlamı, doc: item). Signal set: dense, BM25, topic match, difficulty gap |θ−b|, popularity decay.
3) **Çeşitlilik**: MMR + Curriculum coverage (eksik becerileri doldur).
4) **Bandit**: LinUCB/LinTS ile adaptif öğrenme; kısıt: başarı oranı 0.6–0.85 aralığı.

---

## 4) İngilizce Modülü (Cloze Generation)
### 4.1 Hata Profili
- Etiketler: Tense, Subject‑Verb Agreement, Articles, Prepositions, Collocations, Word Form, Spelling.
- **Confusion sets**: {to/at/in/on}, {a/an/—}, {make/do/take}, {affect/effect}…

### 4.2 Metin Kaynağı & Seviyeleme
- Kaynak pasajlar: seviye‑etiketli (CEFR A1–C1), konu etiketleri.
- Otomatik seviye: readability (FKGL), sözlük sıklığı, CEFR sınıflayıcı.

### 4.3 Cloze Üretimi
- **Kural + LLM**:
  - Kural: hedef hata etiketi + POS/lemma tespiti → uygun boşluk adayları.
  - LLM: bağlama uygun **doğru cevap** + **kişiselleştirilmiş distractor** üretir (confusion set + öğrencinin geçmiş hataları).
- **Doğrulama**:
  - Dilbilgisi denetimi (grammar checker) → çoklu doğru olma riskine karşı **tek-altın** kuralı.
  - Ambiguity check: birden çok doğru ihtimal; gerekirse `multi-answer` flag + öğretmen onayı.

### 4.4 Sıralama & Adaptasyon
- Retrieval: konu/seviye uyumlu pasajlar (dense + BM25 + CEFR filtre).
- Re‑rank: Cross‑encoder (soru‑pasaj uyumu, hedef hata etiketi uygunluğu).
- Çeşitlilik: konu & hata türü dağılımı.
- Bandit: LinUCB/LinTS; ödül: doğru/yanlış + completion + dwell + feedback.

---

## 5) Bandit Politikaları (özet)
- **Hızlı başlangıç**: ε‑greedy(0.1) + popularity fallback.
- **Kişiselleştirme**: LinUCB(α=1.0, λ=1e‑2), context: {θ, hedef beceri one‑hot, cihaz, geçmiş başarı şeridi}.
- **Kısıtlı**: Constrained‑TS, kısıtlar: min başarı 0.6, haftalık konu coverage ≥0.8.

---

## 6) Kalite Güvencesi
- **Schema validation** (item JSON).
- **Answer‑key check** (tek altın, distractor farklı ve pedagojiye uygun).
- **Math solver** doğrulaması (programatik çözüm).
- **Grammar toxicity/bias** kontrolleri (EN).
- **Human‑in‑the‑loop**: yeni şablonlar ve şüpheli öğeler `draft` → öğretmen onayı.

---

## 7) Değerlendirme
- **Offline**:
  - Retrieval: Recall@50, nDCG@20.
  - Ranking: MAP, MRR, Difficulty‑Match(|θ−b|≤0.3) oranı.
  - Üretim kalite: Answer‑key validity, Rationale helpfulness (insan puanı).
- **Online**:
  - Öğrenme KPI: 2‑4 hafta içinde benzer becerilerde doğru oranındaki artış (learning gain).
  - Akış KPI: completion, dwell, hint kullanımı, frustration (çok zor/çok kolay sinyali).
  - Bandit KPI: exploration ratio, reward drift, constraint satisfaction.

---

## 8) Altyapı & Serving
- **LLM Serving**: vLLM/TGI (Llama‑3/Mistral) + opsiyonel API (GPT‑4o) için router.
- **Embedding & Arama**: Qdrant/Weaviate (dense) + OpenSearch/Vespa (BM25); hibrit füzyon.
- **ANN**: HNSW (M=32, efSearch=120); metadata filtre: dil, seviye, beceri, RBAC.
- **Orkestrasyon**: Python/Go microservices; LangGraph/DSPy ile akışlar.
- **Cache**: retrieval cache (TTL=24h), semantic cache (sim≥0.92), KV cache reuse.
- **Observability**: tracing (retrieval_ms, rerank_ms, llm_ms), logs (prompt hash, item_id), metrics (p95 latency, hit‑rate, faithfulness, |θ−b| match).

---

## 9) Güvenlik & Gizlilik
- PII redaction, kullanıcı rızası (`consent_flag`), çocuk/öğrenci verisinde sıkı veri minimizasyonu.
- Prompt injection/jailbreak guardrails (özellikle LLM üretim modülünde).
- RBAC/tenant izolasyonu; audit log (karar izleri, model/policy versiyonları).

---

## 10) Akış Pseudo‑Kod (özet)
```
def recommend_next(user, mode):
    θ = estimate_theta(user)               # IRT/BKT/DKT
    target_skills = detect_gaps(user)      # curriculum & error profile

    # 1) Retrieve
    Dd = dense.search(target_skills, topk=200)
    Ds = bm25.search(target_skills, topk=200)
    C  = fuse(Dd, Ds, w=(0.6,0.4), dedupe=True)
    C  = meta_filter(C, level≈θ, lang=user.lang)

    # 2) Re‑rank
    top = cross_enc.rank(user_context(θ, target_skills), C[:200])[:40]

    # 3) Diversify & curriculum
    cand = diversify(top, mmr=0.7, k=10)
    cand = fill_curriculum_gaps(cand, target_skills)

    # 4) Bandit selection
    arm  = bandit.choose(user, cand)       # LinUCB/LinTS with constraints

    # 5) If generate (need cloze/math template)
    if need_generation(arm):
        item = generate_item(user, arm)    # Math template+solver or EN cloze
        validate(item)                     # schema+answer key+toxicity
    else:
        item = fetch_item(arm.item_id)

    log_decision(user, arm)
    return item
```

---

## 11) Yol Haritası (Uygulama Fazları)
- **Faz 1 (4–6 hafta)**: Item Bank şeması, hybrid retrieval, cross‑encoder re‑rank, IRT tahmini, temel bandit (ε‑greedy), matematik şablonlarının ilk seti, EN cloze MVP, kalite kontrolleri.
- **Faz 2 (6–10 hafta)**: LinUCB/LinTS, misconception‑aware distractors, DKT, curriculum diversification, human‑in‑the‑loop editör UI.
- **Faz 3 (10+ hafta)**: Constrained bandit, Graph‑RAG (opsiyonel), RWKV/Mamba ile uzun metin üretimi/özetleme (içerik genişletme), kapsamlı eval + A/B test.

---

## 12) Riskler & Mitigasyon
- **Ambiguous EN cloze** → multi‑answer flag + öğretmen onayı.
- **Hatalı math answer key** → programatik çözümleyici + property‑based tests.
- **Yanlılık/adalet** → segment bazlı kalite raporu (yaş/seviye/dil); bandit kısıtları.
- **Maliyet/latency** → quantization, caching, routing, distilled cross‑encoder.

---

Bu plan, doğrudan üretime götürülebilir bir çekirdek mimari sunar. Sonraki versiyonda, **şema DDL** (Postgres/BigQuery), **bandit parametre preset’leri** ve **örnek math/EN şablon paketleri**ni ekleyebiliriz.

