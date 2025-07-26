# 🤖 LLM Servis Kurulum ve Test Rehberi

Bu rehber, Yoladgu projesinde LLM (Large Language Model) servisinin kurulumu ve test edilmesi için gerekli adımları içerir.

## 📋 Gereksinimler

### 1. API Anahtarları
- **OpenAI API Key**: https://platform.openai.com/api-keys
- **HuggingFace API Token**: https://huggingface.co/settings/tokens

### 2. Python Paketleri
Gerekli paketler `backend/requirements.txt` dosyasında tanımlı:
- `requests==2.31.0`
- `fastapi==0.104.1`
- `uvicorn[standard]==0.24.0`

## 🚀 Kurulum Adımları

### 1. Environment Dosyası Oluştur

```bash
# Backend dizinine git
cd backend

# .env dosyası oluştur
cp env.example .env
```

### 2. API Anahtarlarını Ekle

`backend/.env` dosyasını düzenle:

```env
# LLM API Keys
OPENAI_API_KEY=your_openai_api_key_here
HUGGINGFACE_API_TOKEN=your_huggingface_token_here
HUGGINGFACE_MODEL=gpt2
```

### 3. Virtual Environment Aktifleştir

```bash
# Backend dizininde
source venv/bin/activate

# Veya yeni venv oluştur
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 4. Server'ı Başlat

```bash
# Backend dizininde
python run.py

# Veya root dizinden
./start-backend.sh
```

## 🧪 Test Adımları

### 1. LLM Servis Testi

```bash
# Root dizinde
python test_llm.py
```

Bu script:
- Environment variables kontrolü yapar
- API anahtarlarını doğrular
- LLM servisini test eder
- Özel fonksiyonları (ipucu, açıklama, zorluk analizi) test eder

### 2. API Endpoint Testi

```bash
# Root dizinde (server çalışırken)
python test_llm_endpoints.py
```

Bu script:
- Server durumunu kontrol eder
- LLM durum endpoint'ini test eder
- Tüm LLM endpoint'lerini test eder

### 3. Manuel API Testi

```bash
# LLM durumu kontrol et
curl http://localhost:8000/api/v1/ai/llm-status

# İpucu üret
curl -X POST http://localhost:8000/api/v1/ai/generate-hint \
  -H "Content-Type: application/json" \
  -d '{"question": "2x + 5 = 13 denklemini çöz"}'

# Açıklama üret
curl -X POST http://localhost:8000/api/v1/ai/generate-explanation \
  -H "Content-Type: application/json" \
  -d '{"question": "2x + 5 = 13 denklemini çöz", "answer": "x = 4"}'

# Zorluk analizi
curl -X POST http://localhost:8000/api/v1/ai/analyze-question-difficulty \
  -H "Content-Type: application/json" \
  -d '{"question": "2x + 5 = 13 denklemini çöz"}'
```

## 🔧 Sorun Giderme

### 1. "No such file or directory: run.py" Hatası

```bash
# Doğru dizinde olduğunuzdan emin olun
cd backend
python run.py
```

### 2. API Key Hatası

```bash
# Environment variables kontrol et
echo $OPENAI_API_KEY
echo $HUGGINGFACE_API_TOKEN

# .env dosyasını kontrol et
cat backend/.env
```

### 3. Import Hatası

```bash
# Python path'ini kontrol et
export PYTHONPATH="${PYTHONPATH}:$(pwd)/backend:$(pwd)/app"

# Virtual environment aktif mi kontrol et
which python
```

### 4. Server Bağlantı Hatası

```bash
# Port kullanımını kontrol et
netstat -tulpn | grep :8000

# Server loglarını kontrol et
tail -f backend/logs/app.log
```

## 📊 LLM Servis Özellikleri

### Batch İşlemleri (İçe Aktarma Sırasında)
- **İpucu Üretimi**: `generate_question_hint()`
- **Açıklama Üretimi**: `generate_question_explanation()`
- **Zorluk Analizi**: `analyze_question_difficulty()`

### Runtime İşlemleri (Öğrenci Etkileşimi)
- **Adaptif İpucu**: `generate_adaptive_hint()`
- **Bağlamsal Açıklama**: `generate_contextual_explanation()`
- **Dinamik Zorluk Ayarı**: `adjust_difficulty_runtime()`

## 🔗 API Endpoint'leri

### Temel Endpoint'ler
- `GET /api/v1/ai/llm-status` - LLM durumu
- `POST /api/v1/ai/generate-hint` - İpucu üret
- `POST /api/v1/ai/generate-explanation` - Açıklama üret
- `POST /api/v1/ai/analyze-question-difficulty` - Zorluk analizi

### Gelişmiş Endpoint'ler
- `POST /api/v1/ai/adaptive-hint` - Adaptif ipucu
- `POST /api/v1/ai/contextual-explanation` - Bağlamsal açıklama
- `POST /api/v1/ai/batch-enrich-questions` - Toplu zenginleştirme

### Eski Endpoint'ler (Geriye Uyumluluk)
- `POST /api/v1/ai/hint` - Eski ipucu endpoint'i
- `POST /api/v1/ai/explanation` - Eski açıklama endpoint'i
- `POST /api/v1/ai/difficulty` - Eski zorluk endpoint'i

## 🎯 Test Senaryoları

### 1. Basit Test
```python
# test_llm.py çalıştır
python test_llm.py
```

### 2. Endpoint Testi
```python
# Server çalışırken
python test_llm_endpoints.py
```

### 3. Manuel Test
```bash
# Server'ı başlat
cd backend && python run.py

# Yeni terminal'de test et
curl http://localhost:8000/api/v1/ai/llm-status
```

## 📝 Notlar

- LLM servisi hem OpenAI hem de HuggingFace API'lerini destekler
- API anahtarları environment variables olarak tanımlanmalı
- Server 8000 portunda çalışır
- Test scriptleri root dizinde çalıştırılmalı
- Virtual environment aktif olmalı

## 🆘 Destek

Sorun yaşarsanız:
1. Test scriptlerini çalıştırın
2. Log dosyalarını kontrol edin
3. Environment variables'ları doğrulayın
4. API anahtarlarının geçerli olduğundan emin olun 