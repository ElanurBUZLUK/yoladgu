# Yoladgu - Öğrenci Merkezli Dinamik Soru Öneri Sistemi

Bu proje, öğrencilerin bilgi seviyelerini gerçek zamanlı takip eden, kişiselleştirilmiş soru önerileri sunan ve makine öğrenmesi ile sürekli kendini geliştiren bir eğitim platformudur.

## 🚀 Özellikler

### Ana Özellikler
- **Kişiselleştirilmiş Soru Önerileri**: Öğrencinin geçmiş performansına göre uygun sorular önerir
- **Gerçek Zamanlı Öğrenme**: Her cevap ile model anında güncellenir
- **Dinamik Zorluk Ayarı**: Öğrenci seviyesine göre soru zorluğu otomatik ayarlanır
- **LLM Destekli İpuçları**: Yapay zeka ile kişiselleştirilmiş ipucu ve açıklamalar
- **Çoklu Model Yaklaşımı**: River (online learning) + LinUCB (bandit) hibrit algoritma

### Gelişmiş Özellikler
- **Asenkron İşleme**: Redis Streams ile yüksek performanslı event işleme
- **Embedding Tabanlı Öneriler**: SBERT ile semantik soru benzerliği
- **Graph Tabanlı Özellikler**: Neo4j ile skill centrality analizi
- **Öneri Çeşitliliği**: Embedding benzerliği ile diversity filtering
- **İzlenebilirlik**: Prometheus metrikleri ve structured logging
- **Güvenilirlik**: DLQ, retry, idempotency ile hata toleransı

## 🏗️ Mimari

### Teknoloji Stack
- **Backend**: FastAPI (Python)
- **Veritabanı**: PostgreSQL + pgvector (embedding storage)
- **Cache/Queue**: Redis + Redis Streams
- **Graph Database**: Neo4j (skill relationships)
- **ML Models**: River (online learning), LinUCB (bandit)
- **Embedding**: Sentence Transformers (SBERT)
- **Monitoring**: Prometheus + FastAPI Instrumentator
- **Logging**: Structured logging (structlog)

### Sistem Mimarisi
```
Frontend (React/Angular) 
    ↓
FastAPI Backend
    ↓
├── PostgreSQL (users, questions, responses)
├── Redis Streams (async processing)
├── Neo4j (skill graph)
└── ML Models (River + LinUCB)
```

## 📊 API Endpoints

### Soru ve Öneri
- `GET /api/v1/questions/recommendations/` - Kişiselleştirilmiş soru önerileri
- `GET /api/v1/questions/recommendations/next-question` - Bir sonraki önerilen soru
- `POST /api/v1/questions/{id}/answer` - Cevap gönderme (asenkron işleme)
- `GET /api/v1/questions/` - Soru listesi
- `POST /api/v1/questions/` - Yeni soru ekleme (embedding otomatik hesaplanır)

### LLM Destekli Özellikler
- `POST /api/v1/ai/adaptive-hint` - Kişiselleştirilmiş ipucu
- `POST /api/v1/ai/contextual-explanation` - Bağlamsal açıklama
- `POST /api/v1/ai/analyze-question-difficulty` - Zorluk analizi
- `POST /api/v1/ai/batch-enrich-questions` - Toplu soru zenginleştirme

### Sistem Durumu
- `GET /health` - Sistem sağlık kontrolü (Redis, DB, Neo4j)
- `GET /metrics` - Prometheus metrikleri

## 🔧 Kurulum

### Gereksinimler
- Python 3.8+
- PostgreSQL 14+ (pgvector extension ile)
- Redis 6+
- Neo4j 4+

### Adım Adım Kurulum

1. **Repository'yi klonlayın**
```bash
git clone https://github.com/ElanurBUZLUK/yoladgu.git
cd yoladgu
```

2. **Bağımlılıkları yükleyin**
```bash
pip install -r requirements.txt
```

3. **Veritabanını hazırlayın**
```bash
# PostgreSQL'de pgvector extension'ını yükleyin (superuser gerekli)
sudo -u postgres psql -d yoladgu -c "CREATE EXTENSION IF NOT EXISTS vector;"

# Migration'ları çalıştırın
alembic upgrade head
```

4. **Environment değişkenlerini ayarlayın**
```bash
cp .env.example .env
# .env dosyasını düzenleyin
```

5. **Servisleri başlatın**
```bash
# Redis ve Neo4j'yi başlatın
docker-compose up -d redis neo4j

# Stream consumer'ı başlatın (arka planda)
python -m app.services.stream_consumer &

# Ana uygulamayı başlatın
uvicorn app.main:app --reload
```

## 🧪 Test

### Test Ortamı
```bash
# Test ortamını başlatın
docker-compose up -d redis neo4j

# Testleri çalıştırın
pytest

# Coverage raporu
pytest --cov=app --cov-report=html
```

### Test Kapsamı
- **Unit Tests**: API endpoints, servisler, modeller
- **Integration Tests**: Redis Streams, DLQ, retry mekanizması
- **End-to-End Tests**: Tam öneri akışı

## 📈 İzleme ve Metrikler

### Prometheus Metrikleri
- `model_update_total` - Model güncelleme sayısı
- `model_update_duration_seconds` - Model güncelleme süresi
- `stream_consumer_lag` - Redis stream lag
- FastAPI otomatik metrikleri (request count, duration, etc.)

### Logging
- **Structured Logging**: Her log context (request_id, student_id, question_id) ile
- **Log Seviyeleri**: INFO, WARNING, ERROR
- **Merkezi Log Entegrasyonu**: ELK, Loki, vs. için hazır

### Health Checks
- `/health` endpoint'i Redis, PostgreSQL ve Neo4j bağlantılarını kontrol eder
- Kubernetes probe'ları için uygun

## 🔄 Asenkron İşleme

### Redis Streams
- Öğrenci cevapları Redis Stream'e yazılır
- Arka planda consumer bu event'leri işler
- Model güncellemeleri asenkron yapılır

### DLQ (Dead Letter Queue)
- Başarısız event'ler `student_responses_dlq` stream'ine yazılır
- Retry mekanizması ile 3 kez denenir
- Malformed event'ler otomatik olarak DLQ'ya yönlendirilir

### Idempotency
- Aynı event'in tekrar işlenmesi engellenir
- Event ID tabanlı duplicate detection

## 🤖 Makine Öğrenmesi

### Hibrit Model Yaklaşımı
1. **River (Online Learning)**
   - Her cevap ile anında öğrenme
   - Logistic Regression pipeline
   - Öğrenci ve soru özelliklerini birleştirir

2. **LinUCB (Contextual Bandit)**
   - Keşif/istismar dengesi
   - Exploration: Yeni soru tipleri dener
   - Exploitation: En iyi performans gösteren soruları seçer

### Özellik Mühendisliği
- **Öğrenci Özellikleri**: Başarı oranı, cevap süresi, konu bazlı doğruluk
- **Soru Özellikleri**: Zorluk, tip, konu, skill ağırlıkları
- **Graph Özellikleri**: Neo4j'den skill centrality
- **Embedding Özellikleri**: SBERT ile semantik benzerlik

### Ensemble Skor
```python
final_score = 0.7 * river_score + 0.3 * embedding_similarity
```

## 🔧 Feature Flags

Sistem davranışını kontrol etmek için environment değişkenleri:
- `USE_NEO4J` - Graph tabanlı özellikleri aktif/pasif
- `USE_EMBEDDING` - Embedding tabanlı önerileri aktif/pasif
- `USE_DIVERSITY_FILTER` - Öneri çeşitliliği filtreleme
- `USE_DLQ` - Dead Letter Queue aktif/pasif
- `USE_PROMETHEUS_HISTOGRAM` - Detaylı metrikler

## 📚 Kullanım Örnekleri

### Öğrenci Akışı
1. Öğrenci giriş yapar
2. Sistem kişiselleştirilmiş soru önerisi sunar
3. Öğrenci soruyu çözer ve cevabını gönderir
4. Cevap asenkron olarak işlenir, model güncellenir
5. Bir sonraki soru önerisi daha da kişiselleştirilmiş olur

### Öğretmen Akışı
1. Öğretmen yeni soru ekler
2. Sistem otomatik olarak embedding hesaplar
3. Soru Neo4j'de skill ilişkileri ile bağlanır
4. Soru öneri havuzuna eklenir

## 🚀 Performans

### Ölçeklenebilirlik
- **Redis Streams**: Yüz binlerce event/saniye işleyebilir
- **Asenkron Model Güncelleme**: API yanıt süresini etkilemez
- **Horizontal Scaling**: Birden fazla consumer instance
- **Caching**: Redis ile hızlı öneri erişimi

### Optimizasyonlar
- Model cache'leme (thread-safe)
- Embedding pre-computation
- Batch processing
- Connection pooling

## 🔒 Güvenlik

- JWT tabanlı authentication
- Role-based access control (student, teacher, admin)
- Input validation ve sanitization
- Rate limiting

## 🤝 Katkıda Bulunma

1. Fork yapın
2. Feature branch oluşturun (`git checkout -b feature/amazing-feature`)
3. Commit yapın (`git commit -m 'Add amazing feature'`)
4. Push yapın (`git push origin feature/amazing-feature`)
5. Pull Request oluşturun

## 📄 Lisans

Bu proje MIT lisansı altında lisanslanmıştır.

## 📞 İletişim

- GitHub: [@ElanurBUZLUK](https://github.com/ElanurBUZLUK)
- Proje Linki: [https://github.com/ElanurBUZLUK/yoladgu](https://github.com/ElanurBUZLUK/yoladgu)

---

⭐ Bu projeyi beğendiyseniz yıldız vermeyi unutmayın! 