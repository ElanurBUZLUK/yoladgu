# ML System Integration

Bu proje, kapsamlı bir Machine Learning sistemi entegrasyonu içerir. Sistem şu bileşenlerden oluşur:

## 🚀 Özellikler

### 1. Semantic Retrieval (Semantik Arama)
- **SBERT Embeddings**: `paraphrase-MiniLM-L6-v2` modeli kullanır
- **Redis Cache**: Embedding'leri cache'ler (TTL: 7200s)
- **FAISS Vector Store**: IVF+PQ index ile O(log N) arama
- **INT8 Quantization**: Model optimizasyonu

### 2. Collaborative Filtering (CF)
- **NMF Factorization**: Kullanıcı-soru etkileşim matrisini faktörize eder
- **FAISS Search**: User/Item factors üzerinde inner-product arama
- **Redis Persistence**: Model JSON olarak saklanır (TTL: 86400s)

### 3. Contextual Bandit (LinUCB)
- **Per-Question Weights**: Her soru için ayrı ağırlık vektörü
- **Feature Extraction**: 15 boyutlu feature vektörü
- **Redis Storage**: `bandit:weights:{question_id}` anahtarıyla

### 4. Online Learning (Neural Network)
- **2-Layer Tanh NN**: Her öğrenci için ayrı ağırlık seti
- **Backpropagation**: Gerçek zamanlı öğrenme
- **Redis Storage**: `online:weights:{student_id}` anahtarıyla

### 5. Ensemble System
- **Weighted Combination**: Dinamik ağırlıklarla model birleştirme
- **Performance-Based**: Öğrenci performansına göre ağırlık ayarlama
- **Real-time Adaptation**: Gerçek zamanlı ağırlık güncelleme

## 📁 Dosya Yapısı

```
app/
├── services/
│   ├── embedding_service.py      # SBERT + Redis cache
│   ├── vector_store_service.py   # FAISS IVF+PQ
│   ├── cf_model_service.py       # Collaborative Filtering
│   ├── bandit_model_service.py   # Contextual Bandit
│   ├── online_model_service.py   # Online Learning
│   ├── ensemble_service.py       # Ensemble System
│   └── faiss_index.py           # FAISS Index Management
├── api/v1/endpoints/
│   ├── recommendations.py        # ML API endpoints
│   ├── vectors.py               # Vector operations
│   └── faiss.py                # FAISS operations
├── schemas/
│   └── recommendation.py        # Pydantic schemas
└── core/
    ├── dependencies.py          # Service dependencies
    └── config.py               # ML settings
```

## 🔧 Kurulum

### 1. Gereksinimler
```bash
pip install -r requirements.txt
```

### 2. Redis Kurulumu
```bash
# Ubuntu/Debian
sudo apt-get install redis-server

# macOS
brew install redis

# Docker
docker run -d -p 6379:6379 redis:latest
```

### 3. Environment Variables
```bash
# .env dosyasına ekleyin
REDIS_URL=redis://localhost:6379/0
FAISS_INDEX_PATH=/data/faiss.index
FAISS_INDEX_DIR=/data/faiss
```

## 🚀 Kullanım

### 1. API Endpoints

#### CF Önerileri
```bash
GET /api/v1/recommend/cf/{student_id}?top_k=10
```

#### Bandit Önerileri
```bash
POST /api/v1/recommend/bandit
{
  "user_features": {...},
  "question_features": {...},
  "question_id": 123
}
```

#### Online Önerileri
```bash
POST /api/v1/recommend/online
{
  "student_id": 1,
  "user_features": {...},
  "question_features": {...}
}
```

#### Ensemble Önerileri
```bash
POST /api/v1/recommend/ensemble
{
  "student_id": 1,
  "user_features": {...},
  "question_features": {...},
  "question_id": 123,
  "weights": {"cf": 0.3, "bandit": 0.4, "online": 0.3}
}
```

### 2. Model Güncelleme

#### Bandit Update
```bash
POST /api/v1/recommend/update/bandit
{
  "student_id": 1,
  "question_id": 123,
  "user_features": {...},
  "question_features": {...},
  "reward": 1.0
}
```

#### Online Update
```bash
POST /api/v1/recommend/update/online
{
  "student_id": 1,
  "question_id": 123,
  "user_features": {...},
  "question_features": {...},
  "reward": 1.0
}
```

### 3. Model İstatistikleri

```bash
GET /api/v1/recommend/stats/cf
GET /api/v1/recommend/stats/bandit
GET /api/v1/recommend/stats/online
GET /api/v1/recommend/stats/ensemble
```

### 4. Model Temizleme

```bash
DELETE /api/v1/recommend/clear/cf
DELETE /api/v1/recommend/clear/bandit
DELETE /api/v1/recommend/clear/online
```

## 🧪 Test

### Test Dosyası Çalıştırma
```bash
python test_ml_services.py
```

### Manuel Test
```python
from app.services.bandit_model_service import BanditModelService
from app.services.online_model_service import OnlineModelService
from app.services.ensemble_service import EnsembleService

# Bandit test
bandit_svc = BanditModelService()
score = bandit_svc.predict(user_features, question_features, question_id)

# Online test
online_svc = OnlineModelService()
score = online_svc.predict(student_id, user_features, question_features)

# Ensemble test
ensemble_svc = EnsembleService()
result = ensemble_svc.predict(student_id, user_features, question_features, question_id)
```

## ⚙️ Konfigürasyon

### ML Model Ayarları (`app/core/config.py`)

```python
# CF Settings
CF_N_FACTORS: int = 50
CF_N_NEIGHBORS: int = 100

# Bandit Settings
BANDIT_ALPHA: float = 0.1
BANDIT_LR: float = 0.01

# Online Settings
ONLINE_HIDDEN_SIZE: int = 64
ONLINE_LR: float = 0.001

# FAISS Settings
FAISS_DIM: int = 384
FAISS_NLIST: int = 128
FAISS_NPROBE: int = 10
```

## 📊 Monitoring

### Logging
Tüm servisler `structlog` kullanır:
```python
logger.info("prediction_completed", 
           student_id=student_id,
           score=score)
```

### Metrics
- Model performans metrikleri
- Cache hit/miss oranları
- Response time'lar
- Error rate'ler

## 🔄 Workflow

### 1. Öneri Süreci
1. Kullanıcı özellikleri çıkarılır
2. Soru özellikleri çıkarılır
3. Her model tahmin yapar
4. Ensemble ağırlıklarla birleştirilir
5. Final skor döndürülür

### 2. Öğrenme Süreci
1. Kullanıcı soruya cevap verir
2. Reward hesaplanır (1.0: doğru, 0.0: yanlış)
3. Bandit ve Online modeller güncellenir
4. Ensemble ağırlıkları ayarlanır

### 3. Cache Yönetimi
1. Embedding'ler Redis'te cache'lenir
2. Model ağırlıkları Redis'te saklanır
3. FAISS index disk'te persist edilir
4. Lazy deletion ile silme işlemleri

## 🚨 Hata Yönetimi

### Exception Handling
```python
try:
    result = service.predict(...)
except Exception as e:
    logger.error("prediction_failed", error=str(e))
    return fallback_result
```

### Fallback Mechanisms
- Model yüklenemezse varsayılan değerler
- Redis bağlantısı koparsa in-memory cache
- FAISS index bozuksa rebuild

## 🔮 Gelecek Geliştirmeler

1. **A/B Testing**: Farklı model kombinasyonları
2. **AutoML**: Otomatik hyperparameter tuning
3. **Distributed Training**: Çoklu sunucu eğitimi
4. **Real-time Monitoring**: Prometheus + Grafana
5. **Model Versioning**: MLflow entegrasyonu

## 📝 Notlar

- Tüm servisler thread-safe'dir
- Redis bağlantıları connection pooling kullanır
- FAISS index'leri lazy loading ile yüklenir
- Model ağırlıkları JSON formatında saklanır
- Ensemble ağırlıkları öğrenci bazında ayarlanır
