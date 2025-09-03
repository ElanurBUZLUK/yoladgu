# Adaptif Soru Öneri Sistemi (Matematik & İngilizce) — Task List + Endpoint Şemaları + Bandit Politikaları + Observability + Evaluation & Lifecycle

Bu dosya, projenin **FastAPI backend** üzerinde geliştirilecek görev listesini, tanımlı **API endpoint şemalarını**, her endpoint için **örnek istek/yanıt JSON’larını**, SLA & oran sınırlamalarını, **bandit politika sözleşmelerini**, **observability dashboard taslağını** ve yeni eklenen **evaluation + data pipeline + model lifecycle** bölümlerini içerir.

---

## ✅ Faz 1 — Çekirdek Altyapı (4–6 Hafta)
- [ ] **FastAPI projesi iskeleti** → `/health`, `/version`
- [ ] **Kullanıcı Yönetimi & Auth** → `/auth/login`, `/auth/register`
- [ ] **Veri Şeması & DB Kurulumu** → PostgreSQL + pgvector, Alembic migration
- [ ] **Hybrid Retrieval Katmanı** → `/search`
- [ ] **Re-ranking Servisi** → `/rerank`
- [ ] **IRT tabanlı seviye tahmini (Math)** → `/profile/update`
- [ ] **Cloze üretimi (EN) MVP** → `/generate/cloze`

## ✅ Faz 2 — Adaptasyon & Kişiselleştirme (6–10 Hafta)
- [ ] **Bandit Orkestrasyonu** → `/recommend/next`
- [ ] **Çeşitlendirme Katmanı (MMR + curriculum)**
- [ ] **Math Şablon + Çözücü** → `/generate/math`
- [ ] **Distractor üretimi (EN)** → cloze distractor’lu
- [ ] **Feedback loop** → `/attempt`, `/feedback`

## ✅ Faz 3 — Gelişmiş Modüller (10+ Hafta)
- [ ] **Constrained Bandit** (başarı≥60%, coverage≥80%)
- [ ] **Peer-aware hata madenciliği (Math)**
- [ ] **Claim-level verification**
- [ ] **Human-in-the-loop editör UI**
- [ ] **Observability & Monitoring** (Prometheus+Grafana)
- [ ] **Evaluation & Test Sets** (Math & EN)
- [ ] **Data Pipeline & Feature Store**
- [ ] **Model Lifecycle & Versioning**

## 🚀 Bonus (Opsiyonel)
- RWKV/Mamba entegrasyonu
- Federated personalization
- Guardrails (NeMo/Llama Guard)

---

## 📘 Endpoint Şemaları + JSON Örnekleri

*(Mevcut tüm endpoint’ler ve JSON örnekleri korunmuştur — bkz. önceki sürüm.)*

---

## 📊 SLA & Oran Limitleri
*(Önceki sürümde belirtilen değerler korunmuştur)*

---

## 📂 Bandit Politika Sözleşmeleri
*(Önceki sürümde detaylı sözleşmeler korunmuştur)*

---

## 📊 Observability Dashboard Taslağı (Grafana)
*(Önceki sürümdeki metrikler ve paneller korunmuştur)*

---

## 🧪 Evaluation & Test Sets

### 1. Ground-Truth Test Sets
- **Math**: Etiketli soru-cevap + zorluk parametreleri (IRT a,b). Alt gruplar: algebra, geometry, trigonometry.
- **EN Cloze**: CEFR seviyeli pasajlar + boşluk doğru cevabı + distractor setleri.
- **Kapsama**: Tüm skill-tags için ≥30 soru.

### 2. Retrieval & RAG Metrikleri
- **Recall@K**: Soruya en az bir doğru pasaj geliyor mu?
- **Faithfulness**: Yanıtın verilen kaynaklarla tutarlılık oranı.
- **Coverage**: Curriculum/topic çeşitliliği.
- **Difficulty-Match**: Öğrenci θ’sına uygunluk (|θ−b| ≤ 0.3).

### 3. Bandit Offline Replay Değerlendirmesi
- **IPS (Inverse Propensity Scoring)**: reward/propensity tabanlı unbiased metrik.
- **DR (Doubly Robust)**: IPS + model-based tahmin.
- **Coverage Slice**: Skill-topic/dil/grade bazlı analiz.
- **Fairness Slice**: Cinsiyet, cihaz, dil bazında reward dağılımı.

---

## 📂 Data Pipeline / Feature Store

### 1. Feature Kümeleri
- **User features**: θ_math, θ_en, error_profiles, recency_stats, grade, lang.
- **Item features**: skill_tags, difficulty_a/b, CEFR, peer_cluster.
- **Context features**: device, session, day_of_week.

### 2. Update Frekansları
- **Online**: attempt sonrası θ güncellemesi (IRT/BKT/DKT).
- **Batch (daily)**: error profile aggregation, peer cluster mining.
- **Batch (weekly)**: curriculum coverage refresh.

### 3. Veri Akışı (Kafka/Kinesis Topics)
- `attempts_stream`: user_id, item_id, answer, correct, time_ms, ts.
- `feedback_stream`: rating, flags, ts.
- `decisions_stream`: policy_id, arms, propensity, chosen_arm_id, ts.
- `metrics_stream`: latency, cache_hit, error_rate, coverage.

### 4. Feature Store (ör. Feast)
- **Online store**: Redis (düşük latency, son attempt’ler).
- **Offline store**: BigQuery/Parquet (model training, evaluation).
- **Join key**: user_id, item_id, timestamp.

---

## 📦 Model Lifecycle & Versioning

### 1. Versiyonlama Politikası
- **Retrieval**: retr_model_vX.Y
- **Re-rank**: crossenc_vX.Y
- **Bandit**: linucb_vX.Y
- **Generation**: math_template_vX.Y, en_cloze_vX.Y

### 2. Model Registry
- MLflow veya Hugging Face Hub + metadata (dataset, config, eval metrics).
- Version promotion: staging → canary → production.

### 3. Deploy Stratejileri
- **Canary Deploy**: %5 kullanıcı → yeni model, SLA & reward stabil → rollout ↑.
- **Shadow Deploy**: Yeni model parallel çalışır, log alınır, sonuç gösterilmez.
- **Rollback**: SLA breach veya reward düşüşünde önceki versiyona otomatik dönüş.

---

## 🧪 Ground-Truth Test Setleri (Math & EN)

### 1) Amaç ve Kapsam
- Math: beceri (skill), konu, Bloom düzeyi ve IRT zorluk (b) dağılımlarını dengeli kapsayan; şablon kökenli değil, final item seviyesinde doğrulanmış sorular.
- EN (Cloze): CEFR seviyelerine (A1–C1) ve hata etiketlerine (prepositions, articles, SVA, collocations…) dengeli dağıtılmış, tek-altın garantili sorular.

### 2) Örnek Boyut ve Dağıtım (öneri)
- Math: her ana beceri için ≥ 200 madde (toplam 2–5k); b-binleri: (-inf,-1.0], (-1.0,-0.3], (-0.3,0.3], (0.3,1.0], (1.0,inf) eşit pay.
- EN: her CEFR seviyesi x her hata etiketi için ≥ 100 madde (toplam 1–3k). Konu çeşitliliği (travel, school, work…).
- Split: train/dev/test = 70/10/20; öğrenci ve şablon sızıntılarını önlemek için by-templategroup ve by-user ayrımı.

### 3) Etiketleme ve Doğrulama Protokolü
- Math: programatik çözücüyle (solver) tek doğru kontrolü; geometri/şekil gerektirenlerde insan onayı; ambiguity_flag alanı zorunlu.
- EN: grammar checker + kural denetimi; confusion-set’lerle distractor kontrolü; single_answer=true sağlanamazsa ambiguous etiketi ve test dışı.
- Metaveri: {skill[], bloom, b_est, a_est, lang, source, review_status}.

### 4) Kalite Barajları
- Item rejection nedenleri: çoklu altın, hatalı anahtar, pedagojik uygunsuzluk, toksisite/bias, seviye uyumsuzluğu.
- Set kabul: coverage≥90%, difficulty bin dengesizliği ≤ %10 sapma, inter-annotator agreement kappa ≥ 0.7 (uygulandığı yerlerde).

---

## 📏 Retrieval ve RAG Metrikleri

### 1) Retrieval
- Recall@K: gold içeren item’lerin K içinde bulunma oranı (altın: item veya span).
- nDCG@K: dereceli alaka (gold ağırlıklandırması) ile normalleştirilmiş DCG.
- MRR: ilk doğru item’in ters sırası; açık-uçlu Q/A’da yardımcı.
- Context Precision: gönderilen pasajların içinde cevapla ilgili span oranı.

### 2) RAG Cevap Kalitesi
- Faithfulness: iddia → kaynak span eşleşmesi (otomatik + insan denetimi); yüzde doğrulanan iddia.
- Answer Relevance: soru-cevap uygunluğu (insan veya LLM-as-judge).
- Citation P/R: verilen alıntıların doğruluğu/eksikliği.
- Difficulty-Match (Math): sunulan soruların |theta - b| ≤ Delta (varsayılan 0.3) oranı; ağırlıklı versiyon: öğrencinin oturum başına etkisiyle ağırlıklandır.

### 3) Raporlama
- Dilimleme: dil (tr/en), seviye/grade, cihaz, topic/skill ve tenant bazında tüm metrikler.
- Kontrol grafikleri: Recall@K ve nDCG@K için haftalık trend + yüzde 95 CI.

---

## 🎲 Bandit Offline Replay — IPS/DR ve Fairness

### 1) Günlükleme Gereksinimleri
- Her öneride arms (adaylar), scores, chosen_arm_id ve propensity zorunlu.
- Reward bileşenleri: correct, completion, dwell_norm, learning_gain (gecikmeli olabilir; bağla).
- Context: {theta, skill_tags, device, recency, peer_cluster}.

### 2) Değerlendirme
- IPS: (1/N) * sum_i [ indicator(a_i = pi(x_i)) * r_i / p_i ].
- SNIPS: normalize IPS (varyans düşürme).
- DR (Doubly Robust): IPS + outcome model düzeltmesi (reward modeli; ör. GBM/NN).
- Coverage/Fairness Slice: segment bazlı avg_reward, success_rate, coverage.

### 3) Veri Hijyeni
- Propensity > 0 olmayan kayıtlar dışlanır (division by zero önlenir).
- Eğitim-değerlendirme sızıntısı önlenir (zaman/öğrenci ayrımı).
- Duplicates/late events temizlenir; gecikmeli ödüller bağlanır (join by request_id chain).

---

## 📂 Data Pipeline ve Feature Store Spesifikasyonu

### 1) Streaming Topic’leri (Kafka/Kinesis – öneri adlandırma)
- events.attempts.v1 — key: tenant_id:user_id, retention: 365g, PII yok
- events.feedback.v1 — key: tenant_id:user_id, retention: 365g
- decisions.recommend.v1 — policy logları (arms, scores, propensity), retention: 180g
- retrieval.candidates.v1 — top-K aday ve skorlar, retention: 7g
- generation.items.v1 — üretilen item ve QA-checks, retention: 90g
- profile.updates.v1 — theta ve hata profili güncellemeleri, retention: 180g
- features.online_write.v1 — online store’a yazma istekleri, retention: 7g

Partitioning: tenant_id ve olay tarihi. Ordering: key-bazlı (user oturumu ardışık).

### 2) Online Feature Store (Redis/Memcached)
- SLO: p95 < 10 ms; TTL: 7–30g (auto-refresh).
- Ana Anahtar: tenant_id:user_id.
- Örnek Alanlar: theta_math, theta_en, recent_error_rates{skill→p}, recency_features{t_since_last, streak}, peer_cluster_id, last_seen_skills, device_pref.

### 3) Offline Feature Store (Warehouse: Postgres/BigQuery)
- Uzun dönem eğriler: theta_trend, coverage_7d/30d, learning_gain
- Item istatistikleri: irt_a, irt_b, popularity_decay, confusion_power (EN)
- Embedding tablosu: item_vec, skill_vec (versiyonlu: emb_v)
- Peer cluster: k-means/UMAP tablosu peer_cluster_id, centroid.

### 4) Güncelleme Frekansları
- Online: theta güncellemesi her attempt sonrası; recency/streak anlık.
- Batch: DKT günde 1x; item IRT parametreleri haftalık; embedding’ler ve indeks gece; peer cluster haftalık.
- Indeks Yenileme: append-only + saatlik incremental, gece compaction.

---

## 🧾 Versiyonlama Politikası (Retrieval, Re-rank, Bandit, Generation)

### 1) Retrieval
- embedding_model_version (örn. e5-multilingual-v2), chunking_version (bölütleme), index_version (dense/sparse ayrı izlenir).
- Değişimlerde A/B veya shadow zorunlu; rollback: eski indeks saklı.

### 2) Re-rank
- cross_encoder_version (örn. ce-int8-v2), feature_schema_version, ltr_model_version (varsa).
- Gating: nDCG@20 artışı veya p95 latency hedefi.

### 3) Bandit
- policy_id + bandit_version; reward_config_version; constraints_version.
- Gating: offline DR ≥ kontrol ve constraint ihlali yok.

### 4) Generation
- Math: template_id + template_version, solver_version.
- EN: cloze_generator_version, confusion_set_version.
- Gating: single-gold pass rate ≥ %99; pedagojik onay.

### 5) Dataset/Testset
- dataset_id (örn. math_en_eval_2025w35); değişim günlüğü; donuk tutulur.

---

## 📚 Model Registry (MLflow örneği)
- Model adları: retrieval-embedder, reranker, bandit-policy, math-generator, en-cloze.
- Stages: Staging → Production → Archived.
- Artifact’lar: model dosyaları, requirements, eval_report.json (Recall@K, nDCG, DR, latency), model_card.md (amaç, veri, riskler, etik notlar).
- Lineage: run id, dataset_id, code commit, param hash; otomatik bağlanır.
- Promotion kriterleri: gating metrikleri + SLO koşulları.

---

## 🚦 Canary vs Shadow Deployment Stratejileri

### Shadow (okuma kopyası)
- Trafiğin %100’ü gölgelenir; yeni model cevapları dönmez.
- Toplanan metrikler: latency, Recall@K (simule), DR (off-policy), faithfulness.
- Süre: 3–7 gün; risk: yok; amaç: regresyon yakalama.

### Canary (kademeli yayılım)
- Başlangıç %5–10 kullanıcı (hash: tenant_id:user_id ile stabil kova).
- Gates (24–48 saat): p95 latency ≤ hedef; avg_reward ≥ kontrol; difficulty-match ≥ %80; constraint violations = 0.
- Başarılı → %25 → %50 → %100; başarısız → roll back (feature flag ile anında geri dönüş).

### Rollback Kriterleri
- Error rate > %2 (30 dk), p95 latency > SLA x 1.5, DR düşüşü > %5, faithfulness < 0.85, constraint ihlali.

---

## 📂 Ekler
- Ground-truth örnek havuzu yönergeleri (etiketleme kılavuzu)
- Offline değerlendirme şablonları (IPS/SNIPS/DR hesap adımları)
- Topic/Feature şemaları (warehouse + online store alan listeleri)
- Versiyonlama ve model kartı şablonları (MLflow uyumlu)

