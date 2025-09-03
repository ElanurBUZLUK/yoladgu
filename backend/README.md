# 🚀 Adaptive Question Recommendation System - Backend

**AI-Powered Personalized Question Recommendation System for Math and English**

Bu proje, öğrencilerin matematik ve İngilizce alanlarında kişiselleştirilmiş soru önerileri alan adaptif bir öğrenme sistemidir. Sistem, öğrencinin geçmiş performansını analiz ederek IRT (Item Response Theory), bandit algoritmaları ve **Vector Database** teknolojileri kullanarak en uygun soruları önerir ve gerektiğinde yeni sorular üretir.

## ✨ **Yeni Özellikler (v2.0)**

### 🎯 **Multi-Backend Vector Indexing**
- **Qdrant**: Production-ready vector database
- **HNSW**: High-performance approximate nearest neighbor search
- **FAISS**: Facebook AI Similarity Search for exact search
- **ML-based Backend Selection**: Intelligent backend routing

### 🧠 **Enhanced AI Generation**
- **English E2E Pipeline**: Complete cloze generation workflow
- **Vector Examples Enhancement**: Real examples for better distractors
- **Error-Aware Recommendations**: Student error profile analysis
- **Hybrid Retrieval**: Dense + Sparse search combination

### 📊 **Advanced Monitoring**
- **Comprehensive Dashboard**: Vector + Search + Performance metrics
- **Health Checks**: Multi-backend status monitoring
- **Performance Analytics**: Latency, accuracy, and quality metrics

## 🏗️ **Mimari (Güncellenmiş)**

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
              │   PostgreSQL    │              │ Vector Manager  │              │ Generation      │
              │ (User Profiles) │              │ (Qdrant/HNSW/   │              │ Service (LLM)   │
              └─────────────────┘              │  FAISS)         │              └─────────────────┘
                       │                                │                                │
                       ▼                                ▼                                ▼
              ┌─────────────────┐              ┌─────────────────┐              ┌─────────────────┐
              │   Redis Cache   │              │ Elasticsearch   │              │ Vector Examples │
              │   (Profiles)    │              │  (Sparse Search) │              │   Service       │
              └─────────────────┘              └─────────────────┘              └─────────────────┘
```

## 🛠️ **Teknoloji Stack (Güncellenmiş)**

### **Core Framework**
- **Backend**: FastAPI + SQLModel
- **Database**: PostgreSQL + Redis
- **ORM**: SQLAlchemy + Alembic

### **Vector & Search**
- **Vector Databases**: Qdrant, HNSW, FAISS
- **Search Engine**: Elasticsearch
- **Embeddings**: Sentence Transformers (multilingual)
- **Vector Manager**: Custom multi-backend orchestration

### **AI/ML Services**
- **Language Models**: Sentence Transformers
- **Math Processing**: SymPy, NumPy
- **ML Backend Selector**: RandomForest classifier
- **Recommendation**: Error-aware + neighbor-lift scoring

### **Infrastructure**
- **Containerization**: Docker + Docker Compose
- **Monitoring**: Custom metrics + health checks
- **Testing**: Pytest + coverage
- **CI/CD**: GitHub Actions ready

## 📋 **Gereksinimler**

### **System Requirements**
- Python 3.11+
- Docker & Docker Compose
- PostgreSQL 15+
- Redis 7+
- Elasticsearch 8+

### **Python Dependencies**
```bash
# Core ML/AI
sentence-transformers>=2.2.2
scikit-learn>=1.7.1
numpy>=1.24.0
scipy>=1.11.0

# Vector Databases
qdrant-client>=1.7.0
hnswlib>=0.8.0
faiss-cpu>=1.7.4

# Search & Cache
elasticsearch>=9.1.0
redis>=6.4.0

# Web Framework
fastapi>=0.104.0
uvicorn>=0.24.0
sqlmodel>=0.0.24
```

## 🚀 **Kurulum**

### **1. Repository'yi klonlayın**
```bash
git clone https://github.com/ElanurBUZLUK/yoladgu.git
cd yoladgu/backend
```

### **2. Environment dosyasını oluşturun**
```bash
cp .env.example .env
# .env dosyasını düzenleyerek gerekli konfigürasyonları yapın
```

### **3. Virtual Environment oluşturun**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# veya
venv\Scripts\activate     # Windows
```

### **4. Dependencies yükleyin**
```bash
pip install -r requirements.txt
```

### **5. Docker Compose ile servisleri başlatın**
```bash
docker-compose up -d
```

### **6. Veritabanı migration'larını çalıştırın**
```bash
alembic upgrade head
```

### **7. ML Backend Selector'ı eğitin**
```bash
python train_backend_selector.py
```

### **8. Uygulamayı başlatın**
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### **9. API dokümantasyonuna erişin**
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health

## 🔧 **Vector Database Yapılandırması**

### **1. FAISS Index Oluşturma**
```python
from app.services.index_backends.faiss_flat_index import FAISSFlatIndexBackend

# FAISS backend'i başlat
faiss_backend = FAISSFlatIndexBackend(
    vector_size=384,  # sentence-transformers boyutu
    metric="ip"       # Inner Product (cosine similarity)
)

# Initialize
await faiss_backend.initialize()
```

### **2. HNSW Index Yapılandırması**
```python
from app.services.index_backends.hnsw_index import HNSWIndexBackend

# HNSW backend'i başlat
hnsw_backend = HNSWIndexBackend(
    vector_size=384,
    max_elements=10000,
    ef_construction=200,
    m=16
)

# Initialize
await hnsw_backend.initialize()
```

### **3. Qdrant Bağlantısı**
```python
from app.services.index_backends.qdrant_index import QdrantIndexBackend

# Qdrant backend'i başlat
qdrant_backend = QdrantIndexBackend(
    collection_name="english_items",
    vector_size=384,
    url="http://localhost:6333"
)

# Initialize
await qdrant_backend.initialize()
```

### **4. Embedding Model Yapılandırması**
```python
from sentence_transformers import SentenceTransformer

# Multilingual model (384 dimensions)
encoder = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# Enhanced embedding with metadata
enhanced_text = f"{text} [ERROR_TAGS]: {', '.join(error_tags)} [LEVEL]: {cefr_level}"
vector = encoder.encode(enhanced_text)
```

## 📁 **Proje Yapısı (Güncellenmiş)**

```
backend/
├── app/
│   ├── api/                    # API endpoints
│   │   └── v1/                # API v1 routes
│   │       ├── generate.py    # Question generation
│   │       ├── vector.py      # Vector operations
│   │       ├── recommendations.py # Error-aware recommendations
│   │       └── admin.py       # Admin operations
│   ├── core/                  # Core configuration
│   ├── db/                    # Database models & repositories
│   ├── models/                # Pydantic models
│   ├── services/              # Business logic services
│   │   ├── index_backends/    # Vector database backends
│   │   │   ├── base.py        # Abstract base class
│   │   │   ├── qdrant_index.py
│   │   │   ├── hnsw_index.py
│   │   │   └── faiss_flat_index.py
│   │   ├── ml/                # ML services
│   │   │   ├── backend_selector.py
│   │   │   └── feature_extractor.py
│   │   ├── recommenders/      # Recommendation engines
│   │   │   └── error_aware.py
│   │   └── vector_service.py  # Vector operations
│   └── main.py                # FastAPI application
├── tests/                     # Test files
├── data/                      # Data files
│   └── ml/                    # ML models
│       └── backend_selector.joblib
├── requirements.txt           # Python dependencies
├── Dockerfile                 # Docker configuration
├── docker-compose.yml         # Docker Compose setup
├── train_backend_selector.py  # ML model training
└── README.md                  # This file
```

## 📊 **API Endpoints (Güncellenmiş)**

### **🔐 Authentication**
- `POST /api/v1/auth/login` - Kullanıcı girişi
- `POST /api/v1/auth/register` - Kullanıcı kaydı

### **🎯 Question Generation**
- `POST /api/v1/generate/en_cloze` - İngilizce cloze sorusu üretimi
- `POST /api/v1/generate/math` - Matematik sorusu üretimi
- `GET /api/v1/templates/english` - İngilizce şablonları

### **🔍 Vector Operations**
- `GET /api/v1/vector/health` - Vector backends health
- `GET /api/v1/vector/stats` - Vector service statistics
- `GET /api/v1/vector/monitoring` - Comprehensive monitoring
- `POST /api/v1/vector/search` - Vector search
- `POST /api/v1/vector/add` - Add vector item
- `POST /api/v1/vector/add-batch` - Batch add items

### **📚 Recommendations**
- `POST /api/v1/recommendations/error-aware` - Error-aware recommendations
- `POST /api/v1/recommendations/error-aware/batch` - Batch recommendations
- `GET /api/v1/recommendations/error-aware/stats` - Service statistics

### **⚙️ Administration**
- `POST /api/v1/admin/reindex_english` - Reindex English items
- `POST /api/v1/admin/reindex_math` - Reindex Math items

## 🧪 **Testing & Development**

### **Local Development**
```bash
# Virtual environment
source venv/bin/activate

# Run tests
pytest

# Run specific test
pytest tests/test_vector_service.py -v

# Coverage report
pytest --cov=app tests/ --cov-report=html
```

### **Vector Service Testing**
```bash
# Test vector operations
python test_vector_index_manager.py

# Test ML backend selector
python train_backend_selector.py

# Test English generation
python test_english_cloze_generation.py
```

### **Code Quality**
```bash
# Code formatting
black app/
isort app/

# Linting
flake8 app/
mypy app/
```

## 🔍 **Monitoring & Health Checks**

### **Health Endpoints**
```bash
# System health
GET /health

# Vector backends health
GET /api/v1/vector/health

# Comprehensive monitoring
GET /api/v1/vector/monitoring
```

### **Performance Metrics**
- **Latency**: p50, p95, p99
- **Throughput**: requests/second
- **Error Rate**: percentage
- **Vector Search**: accuracy, recall
- **Backend Usage**: per-backend statistics

## 🚨 **Troubleshooting**

### **Vector Database Issues**

#### **FAISS Connection Error**
```bash
# Check FAISS installation
python -c "import faiss; print('FAISS OK')"

# Reinstall if needed
pip uninstall faiss-cpu
pip install faiss-cpu
```

#### **HNSW Index Error**
```bash
# Check HNSW installation
python -c "import hnswlib; print('HNSW OK')"

# Reinstall if needed
pip uninstall hnswlib
pip install hnswlib
```

#### **Qdrant Connection Error**
```bash
# Check Qdrant service
docker-compose ps qdrant

# Restart if needed
docker-compose restart qdrant
```

### **Embedding Issues**

#### **Model Download Error**
```bash
# Clear cache
rm -rf ~/.cache/torch/sentence_transformers/

# Reinstall sentence-transformers
pip uninstall sentence-transformers
pip install sentence-transformers
```

#### **Memory Issues**
```bash
# Reduce batch size in vector_service.py
BATCH_SIZE = 100  # Default: 1000

# Use smaller embedding model
model_name = "sentence-transformers/all-MiniLM-L6-v2"  # 384d instead of 768d
```

### **Database Issues**

#### **Migration Errors**
```bash
# Reset migrations
alembic downgrade base
alembic upgrade head

# Check database connection
python -c "from app.db.base import async_engine; print('DB OK')"
```

## 📈 **Performance Optimization**

### **Vector Search Optimization**
```python
# Use appropriate backend for query type
if k > 100:
    backend = "faiss"  # Exact search for large k
elif filters:
    backend = "qdrant"  # Filtered search
else:
    backend = "hnsw"    # Fast approximate search

# Enable hybrid search for better results
results = await vector_service.search(
    query=query,
    limit=k,
    use_hybrid=True,
    use_mmr=True,
    mmr_lambda=0.7
)
```

### **Caching Strategy**
```python
# Redis caching for frequently accessed data
CACHE_TTL = {
    "user_profile": 3600,      # 1 hour
    "vector_results": 86400,    # 24 hours
    "recommendations": 1800,    # 30 minutes
}
```

## 🤝 **Katkıda Bulunma**

### **Development Workflow**
1. **Fork** yapın
2. **Feature branch** oluşturun (`git checkout -b feature/amazing-feature`)
3. **Commit** yapın (`git commit -m 'Add amazing feature'`)
4. **Push** yapın (`git push origin feature/amazing-feature`)
5. **Pull Request** açın

### **Code Standards**
- **Python**: PEP 8, type hints
- **API**: OpenAPI 3.0 specification
- **Testing**: 90%+ coverage required
- **Documentation**: Docstrings for all functions

## 📄 **Lisans**

Bu proje MIT lisansı altında lisanslanmıştır. Detaylar için [LICENSE](LICENSE) dosyasına bakın.

## 📞 **İletişim & Support**

### **GitHub Issues**
- **Bug Reports**: [New Issue](https://github.com/ElanurBUZLUK/yoladgu/issues/new)
- **Feature Requests**: [New Issue](https://github.com/ElanurBUZLUK/yoladgu/issues/new)
- **Documentation**: [Wiki](https://github.com/ElanurBUZLUK/yoladgu/wiki)

### **Maintainers**
- **Elanur Buzluk** - [@ElanurBUZLUK](https://github.com/ElanurBUZLUK)

---

## 🎯 **Quick Start Checklist**

- [ ] Repository cloned
- [ ] Environment configured (.env)
- [ ] Dependencies installed
- [ ] Docker services running
- [ ] Database migrated
- [ ] ML model trained
- [ ] Application started
- [ ] Health checks passed
- [ ] API documentation accessible

**🚀 Ready to build amazing AI-powered learning experiences!**