# 🎓 Adaptive Question System

**AI-Powered Adaptive Learning Platform for Mathematics and English**

[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-13+-blue.svg)](https://www.postgresql.org/)
[![Redis](https://img.shields.io/badge/Redis-6+-red.svg)](https://redis.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 📋 Proje Özeti

Adaptive Question System, öğrencilerin matematik ve İngilizce derslerinde kişiselleştirilmiş öğrenme deneyimi yaşamasını sağlayan AI destekli bir eğitim platformudur. Sistem, öğrenci performansını analiz ederek dinamik seviye ayarlaması yapar ve aralıklı tekrar algoritması ile öğrenmeyi optimize eder.

## ✨ Özellikler

### 🧠 AI Destekli Öğrenme
- **LLM Entegrasyonu**: OpenAI GPT-4 ve Anthropic Claude ile akıllı soru üretimi
- **MCP (Model Context Protocol)**: Gelişmiş AI araçları entegrasyonu
- **PDF İşleme**: PDF'lerden otomatik soru çıkarma
- **Akıllı Cevap Değerlendirme**: AI destekli cevap analizi

### 📊 Dinamik Seviye Yönetimi
- **Performance-Based Level Adjustment**: Performansa dayalı seviye ayarlaması
- **Spaced Repetition**: SM-2 algoritması ile aralıklı tekrar
- **Error Pattern Analysis**: Hata analizi ve pattern recognition
- **Personalized Recommendations**: Kişiselleştirilmiş öneriler

### 🎯 Ders Desteği
- **Matematik**: Çoktan seçmeli ve açık uçlu sorular
- **İngilizce**: Grammar, vocabulary ve comprehension soruları
- **PDF Tabanlı Öğrenme**: PDF'lerden soru üretimi
- **Interactive Learning**: Etkileşimli öğrenme deneyimi

### 📈 Analytics ve Raporlama
- **Student Performance Analytics**: Öğrenci performans analizi
- **Progress Tracking**: İlerleme takibi
- **Similar Student Detection**: Benzer öğrenci tespiti
- **Trend Analysis**: Trend analizi

## 🏗️ Sistem Mimarisi

```
Adaptive Question System
├── 🚀 FastAPI Backend
│   ├── 📊 Database Layer (PostgreSQL)
│   ├── 💾 Cache Layer (Redis)
│   ├── 🤖 AI Services (LLM Gateway, MCP)
│   ├── 📚 Business Logic Services
│   └── 🔐 Authentication & Security
├── 📱 Frontend (React/Vue - Planlanan)
└── 🐳 Docker & Deployment
```

## 🛠️ Teknolojiler

### Backend
- **FastAPI**: Modern, hızlı web framework
- **PostgreSQL**: Güçlü ilişkisel veritabanı
- **Redis**: Yüksek performanslı cache
- **SQLAlchemy**: ORM ve database management
- **Alembic**: Database migration
- **Pydantic**: Data validation ve serialization

### AI & ML
- **OpenAI GPT-4**: Gelişmiş dil modeli
- **Anthropic Claude**: Claude AI entegrasyonu
- **MCP (Model Context Protocol)**: AI araçları protokolü
- **SM-2 Algorithm**: Spaced repetition algoritması

### DevOps & Tools
- **Docker**: Containerization
- **Git**: Version control
- **Alembic**: Database migrations
- **Pytest**: Testing framework

## 🚀 Kurulum

### Gereksinimler
- Python 3.12+
- PostgreSQL 13+
- Redis 6+
- Git

### 1. Repository'yi Klonlayın
```bash
git clone https://github.com/ElanurBUZLUK/yoladgu.git
cd yoladgu
```

### 2. Backend Kurulumu
```bash
cd backend

# Virtual environment oluşturun
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# Dependencies yükleyin
pip install -r requirements.txt
```

### 3. Environment Konfigürasyonu
```bash
# .env dosyası oluşturun
cp .env.example .env

# .env dosyasını düzenleyin
nano .env
```

Gerekli environment değişkenleri:
```env
# Database
DATABASE_URL=postgresql://username:password@localhost:5432/adaptive_question_system

# Redis
REDIS_URL=redis://localhost:6379/0

# Security
SECRET_KEY=your-secret-key-here
ENCRYPTION_KEY=your-encryption-key-here

# LLM Providers
OPENAI_API_KEY=your-openai-api-key
ANTHROPIC_API_KEY=your-anthropic-api-key
```

### 4. Veritabanı Kurulumu
```bash
# Database migration'ları çalıştırın
alembic upgrade head

# Sistem başlatma scriptini çalıştırın
python scripts/init_system.py --sample-data
```

### 5. Uygulamayı Başlatın
```bash
# Development server
python run_dev.py

# Veya
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

## 📚 API Dokümantasyonu

Uygulama başlatıldıktan sonra API dokümantasyonuna erişebilirsiniz:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health

## 🧪 Testler

```bash
# Tüm testleri çalıştırın
cd backend
python -m pytest

# Belirli test dosyalarını çalıştırın
python test_task_11_2_complete.py
python test_task_11_1_complete.py
```

## 📁 Proje Yapısı

```
yoladgu/
├── backend/
│   ├── app/
│   │   ├── api/v1/           # API endpoints
│   │   ├── core/             # Core configurations
│   │   ├── models/           # Database models
│   │   ├── services/         # Business logic
│   │   ├── schemas/          # Pydantic schemas
│   │   ├── middleware/       # Custom middleware
│   │   ├── repositories/     # Data access layer
│   │   └── mcp/              # MCP tools
│   ├── scripts/              # Utility scripts
│   ├── tests/                # Test files
│   ├── alembic/              # Database migrations
│   └── requirements.txt      # Python dependencies
├── .kiro/                    # Project specifications
├── docs/                     # Documentation
└── README.md
```

## 🔧 Sistem Bileşenleri

### Core Services
- **User Service**: Kullanıcı yönetimi ve authentication
- **Question Service**: Soru yönetimi ve öneriler
- **Answer Evaluation Service**: Cevap değerlendirme
- **Level Adjustment Service**: Dinamik seviye ayarlama
- **Spaced Repetition Service**: Aralıklı tekrar algoritması
- **Analytics Service**: Performans analizi
- **PDF Processing Service**: PDF işleme
- **LLM Gateway Service**: AI entegrasyonu

### API Endpoints
- **Authentication**: `/api/v1/auth/*`
- **Users**: `/api/v1/users/*`
- **Math Questions**: `/api/v1/math/*`
- **English Questions**: `/api/v1/english/*`
- **Answers**: `/api/v1/answers/*`
- **Analytics**: `/api/v1/analytics/*`
- **PDF**: `/api/v1/pdf/*`
- **System**: `/api/v1/system/*`

## 🤝 Katkıda Bulunma

1. Fork yapın
2. Feature branch oluşturun (`git checkout -b feature/amazing-feature`)
3. Değişikliklerinizi commit edin (`git commit -m 'Add amazing feature'`)
4. Branch'inizi push edin (`git push origin feature/amazing-feature`)
5. Pull Request oluşturun

## 📄 Lisans

Bu proje MIT lisansı altında lisanslanmıştır. Detaylar için [LICENSE](LICENSE) dosyasına bakın.

## 👥 Geliştirici

**Elanur Buzluk**
- GitHub: [@ElanurBUZLUK](https://github.com/ElanurBUZLUK)
- Email: elanur.buzluk@example.com

## 🆘 Destek

Herhangi bir sorun yaşarsanız:
1. [Issues](https://github.com/ElanurBUZLUK/yoladgu/issues) sayfasını kontrol edin
2. Yeni bir issue oluşturun
3. Email ile iletişime geçin

## 🗺️ Roadmap

### ✅ Tamamlanan Özellikler
- [x] Backend API geliştirme
- [x] Database schema ve migrations
- [x] AI entegrasyonu (LLM, MCP)
- [x] PDF işleme sistemi
- [x] Analytics ve reporting
- [x] Test coverage
- [x] System initialization

### 🚧 Geliştirilmekte Olan Özellikler
- [ ] Frontend web uygulaması
- [ ] Mobile app
- [ ] Advanced analytics dashboard
- [ ] Real-time notifications
- [ ] Multi-language support

### 📋 Planlanan Özellikler
- [ ] Video ders entegrasyonu
- [ ] Gamification elements
- [ ] Parent/Teacher portal
- [ ] Advanced AI features
- [ ] Cloud deployment

---

⭐ Bu projeyi beğendiyseniz yıldız vermeyi unutmayın!
