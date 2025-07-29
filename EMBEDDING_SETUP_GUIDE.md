# 🚀 Embedding ve Ensemble Sistemleri Kurulum Kılavuzu

Bu kılavuz, projeye eklenen pgvector, SBERT embedding ve ensemble scoring sistemlerinin kurulumu ve kullanımını açıklar.

## 📋 İçindekiler

1. [pgvector Kurulumu](#1-pgvector-kurulumu)
2. [SBERT Entegrasyonu](#2-sbert-entegrasyonu)
3. [Toplu Embedding Güncelleme](#3-toplu-embedding-güncelleme)
4. [Ensemble Scoring Sistemi](#4-ensemble-scoring-sistemi)
5. [Test ve Doğrulama](#5-test-ve-doğrulama)
6. [Performans Optimizasyonu](#6-performans-optimizasyonu)

## 1. pgvector Kurulumu

### 1.1 PostgreSQL Extension Kurulumu

```bash
# PostgreSQL superuser olarak giriş yapın
sudo -u postgres psql -d yoladgu

# pgvector extension'ını kurun
CREATE EXTENSION IF NOT EXISTS vector;

# Çıkış
\q
```

### 1.2 Otomatik Kurulum Scripti

```bash
# Kurulum scriptini çalıştırın
python scripts/setup_pgvector.py
```

Bu script şunları yapar:
- ✅ pgvector extension'ını kurar
- ✅ `questions` tablosuna `embedding` sütunu ekler
- ✅ Cosine similarity index'i oluşturur
- ✅ Yardımcı fonksiyonları ekler

### 1.3 Kurulum Doğrulama

```bash
# Kurulum durumunu kontrol edin
python scripts/setup_pgvector.py --verify
```

## 2. SBERT Entegrasyonu

### 2.1 Gerekli Paketlerin Kurulumu

```bash
# SBERT ve gerekli paketleri kurun
pip install sentence-transformers pgvector psycopg2-binary

# veya requirements.txt'ye ekleyin
echo "sentence-transformers>=2.2.0" >> requirements.txt
echo "pgvector>=0.2.0" >> requirements.txt
```

### 2.2 Model Konfigürasyonu

```python
# app/core/config.py'de
USE_EMBEDDING = True
EMBEDDING_MODEL = "paraphrase-MiniLM-L6-v2"  # 384 boyutlu
EMBEDDING_DIM = 384
```

### 2.3 Embedding Service Kullanımı

```python
from app.services.embedding_service import compute_embedding, find_similar_questions

# Tek metin için embedding
embedding = compute_embedding("What is 2+2?")

# Benzer soruları bul
similar = find_similar_questions(embedding, threshold=0.8, limit=10)
```

## 3. Toplu Embedding Güncelleme

### 3.1 Tüm Soruları Güncelle

```bash
# Tüm soruların embedding'lerini güncelle
python scripts/batch_embedding_update.py

# Batch boyutunu ayarla
python scripts/batch_embedding_update.py --batch-size 100

# Maksimum batch sayısını sınırla
python scripts/batch_embedding_update.py --max-batches 10
```

### 3.2 Belirli Ders İçin Güncelleme

```bash
# Matematik dersi (ID: 1) için embedding'leri güncelle
python scripts/batch_embedding_update.py --subject 1
```

### 3.3 Tek Soru Güncelleme

```bash
# Belirli bir soruyu güncelle
python scripts/batch_embedding_update.py --question 123
```

### 3.4 İstatistikleri Görüntüle

```bash
# Embedding kapsama oranını kontrol et
python scripts/batch_embedding_update.py --stats
```

## 4. Ensemble Scoring Sistemi

### 4.1 Sistem Bileşenleri

Ensemble scoring sistemi şu bileşenleri birleştirir:

- **River ML Model** (35%): Online öğrenme modeli
- **SBERT Embedding** (25%): Semantik benzerlik
- **Skill Mastery** (20%): Beceri uyumu
- **Difficulty Match** (15%): Zorluk seviyesi uyumu
- **Neo4j Similarity** (5%): Graph tabanlı benzerlik

### 4.2 Dinamik Ağırlık Ayarlama

```python
from app.services.ensemble_service import adjust_weights_dynamically

# Öğrenci performansına göre ağırlıkları ayarla
adjust_weights_dynamically(student_performance=0.3, question_count=5)
```

**Performans Bazlı Ayarlar:**
- **Düşük Performans (<0.4)**: Embedding ağırlığı artırılır
- **Yüksek Performans (>0.8)**: Skill mastery ağırlığı artırılır
- **Normal Performans**: Standart ağırlıklar kullanılır

### 4.3 Threshold Filtreleme

```python
from app.services.ensemble_service import filter_questions_by_thresholds

# Threshold'lara göre soruları filtrele
filtered_questions = filter_questions_by_thresholds(
    questions, student_level=3.0, student_recent_performance=0.6
)
```

**Threshold Değerleri:**
- `min_similarity`: 0.6 (Minimum embedding benzerliği)
- `max_difficulty_gap`: 2 (Maksimum zorluk farkı)
- `min_skill_mastery`: 0.3 (Minimum skill mastery)

## 5. Test ve Doğrulama

### 5.1 Unit Testler

```bash
# Embedding ve ensemble testlerini çalıştır
pytest tests/test_embedding_ensemble.py -v

# Neo4j entegrasyon testleri
pytest tests/test_neo4j_integration.py -v
```

### 5.2 Entegrasyon Testleri

```bash
# Tüm testleri çalıştır
pytest tests/ -v --cov=app --cov-report=html
```

### 5.3 Performans Testleri

```python
# Embedding hesaplama performansı
import time
from app.services.embedding_service import compute_embedding

start_time = time.time()
embedding = compute_embedding("Test question")
duration = time.time() - start_time
print(f"Embedding hesaplama süresi: {duration:.3f}s")
```

## 6. Performans Optimizasyonu

### 6.1 Database Index'leri

```sql
-- Embedding index'i (otomatik oluşturulur)
CREATE INDEX questions_embedding_idx 
ON questions USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- Soru metni index'i
CREATE INDEX question_content_idx ON questions USING gin(to_tsvector('turkish', content));
```

### 6.2 Caching Stratejisi

```python
# Redis cache kullanımı
cache_key = f"recommendations:{student_id}:ensemble"
cached = redis_client.get(cache_key)
if cached:
    return json.loads(cached.decode('utf-8'))
```

### 6.3 Batch Processing

```python
# Toplu embedding hesaplama
embeddings = compute_embeddings_batch(texts)  # Tek seferde birden fazla metin
```

## 7. Monitoring ve Logging

### 7.1 Structured Logging

```python
import structlog

logger = structlog.get_logger()
logger.info("ensemble_score_calculated", 
           question_id=123, 
           student_id=456,
           ensemble_score=0.85)
```

### 7.2 Prometheus Metrikleri

```python
# Embedding hesaplama metrikleri
embedding_computation_duration.observe(duration)
embedding_computation_total.inc()
```

## 8. Troubleshooting

### 8.1 Yaygın Hatalar

**pgvector Extension Hatası:**
```bash
# Çözüm: Superuser olarak kurulum yapın
sudo -u postgres psql -d yoladgu -c "CREATE EXTENSION vector;"
```

**SBERT Model Yükleme Hatası:**
```bash
# Çözüm: Model cache'ini temizleyin
rm -rf ~/.cache/torch/sentence_transformers/
```

**Memory Hatası:**
```bash
# Çözüm: Batch boyutunu azaltın
python scripts/batch_embedding_update.py --batch-size 25
```

### 8.2 Debug Modu

```python
# Debug modunda çalıştır
import logging
logging.basicConfig(level=logging.DEBUG)

# Embedding service debug
from app.services.embedding_service import embedding_service
embedding_service._load_model()  # Model yükleme sürecini izle
```

## 9. Production Deployment

### 9.1 Environment Variables

```bash
# .env dosyası
USE_EMBEDDING=true
EMBEDDING_MODEL=paraphrase-MiniLM-L6-v2
EMBEDDING_DIM=384
USE_ENSEMBLE_SCORING=true
```

### 9.2 Docker Deployment

```dockerfile
# Dockerfile'da SBERT modelini önceden yükle
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('paraphrase-MiniLM-L6-v2')"
```

### 9.3 Health Checks

```python
# Health endpoint'inde embedding kontrolü
@app.get("/health")
def health():
    try:
        embedding = compute_embedding("test")
        return {"embedding": "ok", "status": "healthy"}
    except Exception as e:
        return {"embedding": "error", "status": "unhealthy"}
```

## 10. Sonraki Adımlar

1. **Model Fine-tuning**: Türkçe sorular için SBERT modelini fine-tune edin
2. **A/B Testing**: Ensemble vs tek model performansını karşılaştırın
3. **Real-time Updates**: Yeni sorular eklendiğinde otomatik embedding hesaplama
4. **Advanced Features**: Multi-modal embedding (resim + metin) desteği

---

## 📊 Performans Beklentileri

- **Embedding Hesaplama**: ~50ms/soru
- **Benzerlik Arama**: ~10ms (index ile)
- **Ensemble Scoring**: ~100ms/soru
- **Toplu Güncelleme**: ~100 soru/saniye

## 🔧 Konfigürasyon Örnekleri

```python
# app/core/config.py
class Settings:
    USE_EMBEDDING: bool = True
    EMBEDDING_MODEL: str = "paraphrase-MiniLM-L6-v2"
    EMBEDDING_DIM: int = 384
    USE_ENSEMBLE_SCORING: bool = True
    ENSEMBLE_WEIGHTS: Dict[str, float] = {
        'river_score': 0.35,
        'embedding_similarity': 0.25,
        'skill_mastery': 0.20,
        'difficulty_match': 0.15,
        'neo4j_similarity': 0.05
    }
```

Bu kılavuz ile embedding ve ensemble sistemlerini başarıyla kurup kullanabilirsiniz! 🎯 