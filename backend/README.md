# Adaptive Question Recommendation System - Backend

Bu proje, öğrencilerin matematik ve İngilizce alanlarında kişiselleştirilmiş soru önerileri alan adaptif bir öğrenme sistemidir. Sistem, öğrencinin geçmiş performansını analiz ederek IRT (Item Response Theory) ve bandit algoritmaları kullanarak en uygun soruları önerir ve gerektiğinde yeni sorular üretir.

## 🚀 Özellikler

- **Kişiselleştirilmiş Öneriler**: IRT tabanlı öğrenci modelleme
- **Adaptif Öğrenme**: LinUCB/LinTS bandit algoritmaları
- **Hibrit Arama**: Dense + Sparse retrieval sistemi
- **Soru Üretimi**: Matematik ve İngilizce için otomatik soru üretimi
- **Kalite Güvencesi**: Programatik doğrulama ve grammar checking
- **Gerçek Zamanlı**: FastAPI ile yüksek performanslı API
- **Ölçeklenebilir**: Mikroservis mimarisi ve caching

## 🏗️ Mimari

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   İstemci       │───▶│   API Gateway   │───▶│  Orchestrator   │
│   Uygulaması    │    │   (FastAPI)     │    │    Service      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
                       ┌────────────────────────────────┼────────────────────────────────┐
                       │                                │                                │
                       ▼                                ▼                                ▼
              ┌─────────────────┐              ┌─────────────────┐              ┌─────────────────┐
              │ Profile Service │              │Retrieval Service│              │ Bandit Service  │
              │   (IRT/BKT)     │              │ (Hybrid Search) │              │  (LinUCB/TS)    │
              └─────────────────┘              └─────────────────┘              └─────────────────┘
                       │                                │                                │
                       ▼                                ▼                                ▼
              ┌─────────────────┐              ┌─────────────────┐              ┌─────────────────┐
              │   PostgreSQL    │              │ Vector DB       │              │ Generation      │
              │ (User Profiles) │              │ (Qdrant)        │              │ Service (LLM)   │
              └─────────────────┘              └─────────────────┘              └─────────────────┘
```

## 🛠️ Teknoloji Stack

- **Backend Framework**: FastAPI
- **Database**: PostgreSQL + Redis
- **Vector Database**: Qdrant
- **Search Engine**: Elasticsearch
- **ML/AI**: Sentence Transformers, SymPy, Transformers
- **Monitoring**: Prometheus, OpenTelemetry
- **Containerization**: Docker, Docker Compose

## 📋 Gereksinimler

- Python 3.11+
- Docker & Docker Compose
- PostgreSQL 15+
- Redis 7+
- Qdrant
- Elasticsearch 8+

## 🚀 Kurulum

### 1. Repository'yi klonlayın
```bash
git clone <repository-url>
cd backend
```

### 2. Environment dosyasını oluşturun
```bash
cp .env.example .env
# .env dosyasını düzenleyerek gerekli konfigürasyonları yapın
```

### 3. Docker Compose ile servisleri başlatın
```bash
docker-compose up -d
```

### 4. Veritabanı migration'larını çalıştırın
```bash
docker-compose exec app alembic upgrade head
```

### 5. API dokümantasyonuna erişin
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 📁 Proje Yapısı

```
backend/
├── app/
│   ├── api/                 # API endpoints
│   │   └── v1/             # API v1 routes
│   ├── core/               # Core configuration
│   ├── db/                 # Database models & repositories
│   ├── models/             # Pydantic models
│   ├── services/           # Business logic services
│   ├── utils/              # Utility functions
│   └── main.py            # FastAPI application
├── tests/                  # Test files
├── docs/                   # Documentation
├── scripts/               # Utility scripts
├── requirements.txt       # Python dependencies
├── Dockerfile            # Docker configuration
├── docker-compose.yml    # Docker Compose setup
└── README.md            # This file
```

## 🔧 Geliştirme

### Local Development
```bash
# Virtual environment oluşturun
python -m venv venv
source venv/bin/activate  # Linux/Mac
# veya
venv\Scripts\activate     # Windows

# Dependencies yükleyin
pip install -r requirements.txt

# Uygulamayı başlatın
uvicorn app.main:app --reload
```

### Testing
```bash
# Testleri çalıştırın
pytest

# Coverage raporu
pytest --cov=app tests/
```

### Code Quality
```bash
# Code formatting
black app/
isort app/

# Linting
flake8 app/
mypy app/
```

## 📊 API Endpoints

### Authentication
- `POST /api/v1/auth/login` - Kullanıcı girişi
- `POST /api/v1/auth/register` - Kullanıcı kaydı
- `POST /api/v1/auth/refresh` - Token yenileme

### Recommendations
- `POST /api/v1/recommend/next` - Soru önerisi
- `GET /api/v1/profile/{user_id}` - Kullanıcı profili
- `POST /api/v1/profile/update` - Profil güncelleme

### Generation
- `POST /api/v1/generate/math` - Matematik sorusu üretimi
- `POST /api/v1/generate/en_cloze` - İngilizce cloze sorusu üretimi

### Tracking
- `POST /api/v1/attempt` - Cevap kaydı
- `POST /api/v1/feedback` - Geri bildirim

### Administration
- `GET /api/v1/admin/metrics` - Sistem metrikleri
- `GET /api/v1/admin/decisions/{request_id}` - Karar audit'i

## 🔍 Monitoring

### Health Checks
- `GET /health` - Sistem sağlığı
- `GET /version` - Versiyon bilgisi

### Metrics
- Latency (p50, p95, p99)
- Error rate
- Cache hit rate
- Faithfulness score
- Difficulty match rate
- Coverage metrics
- Exploration ratio

## 🔒 Güvenlik

- JWT tabanlı authentication
- RBAC (Role-Based Access Control)
- PII redaction
- Rate limiting
- Input validation
- CORS protection

## 📈 Performance

### SLA Targets
- `/recommend/next`: p95 < 700ms
- `/generate/*`: p95 < 1200ms
- `/attempt`: p95 < 150ms
- Uptime: 99.9%

### Caching Strategy
- Retrieval cache: 24h TTL
- Profile cache: 1h TTL
- Semantic cache: similarity > 0.92

## 🤝 Katkıda Bulunma

1. Fork yapın
2. Feature branch oluşturun (`git checkout -b feature/amazing-feature`)
3. Commit yapın (`git commit -m 'Add amazing feature'`)
4. Push yapın (`git push origin feature/amazing-feature`)
5. Pull Request açın

## 📄 Lisans

Bu proje MIT lisansı altında lisanslanmıştır. Detaylar için [LICENSE](LICENSE) dosyasına bakın.

## 📞 İletişim

Sorularınız için issue açabilir veya proje maintainer'larına ulaşabilirsiniz.