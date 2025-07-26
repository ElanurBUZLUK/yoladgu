# Question Recommendation System

Akıllı soru öneri sistemi - öğrencilerin seviyelerine uygun sorular öneren yapay zeka tabanlı platform.

## 🚀 Özellikler

- **Akıllı Öneri Sistemi**: River ve LinUCB algoritmaları ile kişiselleştirilmiş soru önerileri
- **Seviye Takibi**: Öğrenci performansına göre dinamik seviye güncellemesi
- **Çoklu Model Desteği**: River (online learning) ve LinUCB (bandit) modelleri
- **Gerçek Zamanlı Öğrenme**: Öğrenci cevaplarına göre sürekli model güncellemesi
- **Kapsamlı API**: RESTful API ile tam entegrasyon
- **Test Sistemi**: Kapsamlı unit ve integration testleri

## 🏗️ Teknoloji Stack

### Backend
- **FastAPI**: Modern, hızlı web framework
- **SQLAlchemy**: ORM ve database yönetimi
- **PostgreSQL**: Ana veritabanı
- **Redis**: Cache ve session yönetimi
- **River**: Online machine learning kütüphanesi
- **Pydantic**: Data validation ve serialization

### Frontend
- **Angular**: Modern web framework
- **TypeScript**: Type-safe JavaScript
- **Material Design**: UI/UX framework

### DevOps & Testing
- **Pytest**: Test framework
- **Uvicorn**: ASGI server
- **Docker**: Containerization (opsiyonel)

## 📦 Kurulum

### Gereksinimler
- Python 3.12+
- PostgreSQL 12+
- Redis 6+
- Node.js 18+ (frontend için)

### Backend Kurulumu

1. **Repository'yi klonlayın:**
```bash
git clone https://github.com/ElanurBUZLUK/yoladgu.git
cd yoladgu
```

2. **Virtual environment oluşturun:**
```bash
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# veya
venv\Scripts\activate  # Windows
```

3. **Bağımlılıkları yükleyin:**
```bash
pip install -r backend/requirements.txt
```

4. **Environment değişkenlerini ayarlayın:**
```bash
cp .env.example .env
# .env dosyasını düzenleyin
```

5. **Veritabanını kurun:**
```bash
# PostgreSQL'de veritabanı oluşturun
createdb veritabani

# Migration'ları çalıştırın
alembic upgrade head
```

6. **Backend'i başlatın:**
```bash
# Ana dizinden
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# veya backend dizininden
cd backend
python run.py
```

### Frontend Kurulumu

1. **Frontend dizinine gidin:**
```bash
cd frontend
```

2. **Bağımlılıkları yükleyin:**
```bash
npm install
```

3. **Development server'ı başlatın:**
```bash
ng serve
```

## 🧪 Testler

### Test Çalıştırma
```bash
# Tüm testleri çalıştır
pytest tests/ -v

# Belirli test dosyasını çalıştır
pytest tests/test_api.py -v

# Coverage ile çalıştır
pytest tests/ --cov=app --cov-report=html
```

### Test Kategorileri
- **Unit Tests**: `tests/test_basic.py`
- **API Tests**: `tests/test_api.py`
- **CRUD Tests**: `tests/test_crud.py`
- **Service Tests**: `tests/test_services.py`

## 📚 API Dokümantasyonu

Backend çalıştıktan sonra API dokümantasyonuna erişin:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Ana Endpoint'ler
- `GET /health` - Sistem durumu
- `POST /api/v1/auth/login` - Kullanıcı girişi
- `GET /api/v1/questions/` - Soru listesi
- `POST /api/v1/recommendations/` - Soru önerisi
- `POST /api/v1/responses/` - Cevap kaydetme

## 🤖 Machine Learning Modelleri

### River Model
- **Amaç**: Online learning ile sürekli öğrenme
- **Algoritma**: Adaptive Random Forest
- **Özellikler**: Gerçek zamanlı güncelleme, drift detection

### LinUCB Model
- **Amaç**: Multi-armed bandit ile keşif-sömürü dengesi
- **Algoritma**: Linear Upper Confidence Bound
- **Özellikler**: Contextual bandit, confidence intervals

## 📊 Özellik Çıkarımı

Sistem şu özellikleri çıkarır:

### Öğrenci Özellikleri
- Toplam soru sayısı
- Doğru cevap oranı
- Ortalama cevap süresi
- Güven seviyesi
- Saat bazlı performans
- Konu bazlı başarı oranları

### Soru Özellikleri
- Zorluk seviyesi
- Soru tipi
- Konu bilgisi
- Skill ağırlıkları
- Etiketler

## 🔧 Konfigürasyon

### Environment Variables
```env
# Database
DATABASE_URL=postgresql://user:pass@localhost:5432/dbname
POSTGRES_SERVER=localhost
POSTGRES_USER=user
POSTGRES_PASSWORD=pass
POSTGRES_DB=dbname

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=

# Security
SECRET_KEY=your-secret-key
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# ML Models
MODEL_CACHE_DIR=./models
RECOMMENDATION_BATCH_SIZE=100
LEARNING_RATE=0.01
```

## 🚀 Production Deployment

### Docker ile (Önerilen)
```bash
# Docker Compose ile tüm servisleri başlat
docker-compose up -d

# Sadece backend
docker build -t yoladgu-backend .
docker run -p 8000:8000 yoladgu-backend
```

### Manuel Deployment
1. **Production server'da kurulum yapın**
2. **Environment variables'ları ayarlayın**
3. **Gunicorn ile başlatın:**
```bash
gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker
```

## 📈 Performans

### Test Sonuçları
- **API Response Time**: < 100ms
- **Model Inference**: < 50ms
- **Database Queries**: < 20ms
- **Test Coverage**: > 80%

### Ölçeklenebilirlik
- **Concurrent Users**: 1000+
- **Questions/Second**: 100+
- **Model Updates/Second**: 50+

## 🤝 Katkıda Bulunma

1. Fork yapın
2. Feature branch oluşturun (`git checkout -b feature/amazing-feature`)
3. Değişikliklerinizi commit edin (`git commit -m 'Add amazing feature'`)
4. Branch'inizi push edin (`git push origin feature/amazing-feature`)
5. Pull Request oluşturun

## 📝 Lisans

Bu proje MIT lisansı altında lisanslanmıştır. Detaylar için [LICENSE](LICENSE) dosyasına bakın.

## 👥 Geliştirici

- **Elanur Buzluk** - [GitHub](https://github.com/ElanurBUZLUK)

## 🙏 Teşekkürler

- River ML kütüphanesi ekibine
- FastAPI geliştiricilerine
- Açık kaynak topluluğuna

## 📞 İletişim

- **GitHub**: [@ElanurBUZLUK](https://github.com/ElanurBUZLUK)
- **Email**: [email protected]

---

⭐ Bu projeyi beğendiyseniz yıldız vermeyi unutmayın! 