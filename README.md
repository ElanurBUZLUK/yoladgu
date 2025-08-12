## LearnAI Monorepo (Full E2E v2)

Bu depo; Backend (FastAPI + Postgres + Redis), Frontend (statik HTML/JS), ML & Öneri Sistemi (LinUCB + FTRL + Vektör Arama HNSW/FAISS/Qdrant seçilebilir), Blue/Green Index Yönetimi, Öğretmen-Öğrenci onay akışı, RBAC ve örnek MCP konfigürasyonları içerir.

### Hızlı Başlangıç (Docker Compose ile)

Önkoşullar: Docker ve Docker Compose

```bash
# 1) Tüm servisleri ayağa kaldır
docker compose up -d --build

# 2) Veritabanı şemasını yükle (iki seçenek)
# 2A) Host makinede psql varsa (port haritalaması: 55433)
psql postgresql://yoladgu:password@localhost:55433/yoladgu -f db/init.sql

# 2B) psql yoksa: SQL'i container içine pipe et
docker compose exec -T postgres psql -U yoladgu -d yoladgu -f /dev/stdin < db/init.sql

# 3) ML/Vector seed + index build + mavi/yeşil swap
docker compose exec -T backend python tools/seed_embeddings.py --n 1000
docker compose exec -T backend python tools/batch_indexer.py
curl -X POST http://localhost:8001/api/v1/index/swap

# 4) Sağlık kontrolü
curl http://localhost:8001/health

# 5) Frontend
# Docker ile geldi: http://localhost:8080
```

### Demo Kullanıcı Oluşturma (Register ile)

Ön tanımlı kullanıcı yoktur; demoyu Register ile oluşturun.

```bash
curl -X POST http://localhost:8001/api/v1/auth/register \
  -H 'Content-Type: application/json' \
  -d '{"email":"student1@yoladgu.local","password":"pass123","full_name":"Demo Student","role":"student"}'
```

Ardından bu kullanıcı ile login olabilir ve Frontend üzerinde oturum açabilirsiniz.

Not: Geliştirme kolaylığı için `.local` domain’li e-postalar desteklenir. Bu amaçla `app/schemas.py` içinde `RegisterRequest.email` ve `LoginRequest.email` tipleri `EmailStr` yerine `str` olarak tanımlanmıştır. Değişikliği konteynere almak için backend imajını yeniden derleyin:

```bash
docker compose up -d --build backend
```

## Mimari ve Bileşenler

- Backend: FastAPI, SQLAlchemy (async), JWT (access/refresh), RBAC (admin/teacher/student)
- Depolama: Postgres (55433→5432), Redis (16380→6379)
- Vektör Arama: HNSW (varsayılan) veya FAISS/Qdrant
- Blue/Green Index: `questions_blue` / `questions_green` alias yönetimi
- Frontend: Basit statik HTML/JS (8080)

### Proje Dizini (Özet)

```
backend/        # FastAPI uygulaması
frontend/       # Statik HTML/JS
db/init.sql     # Veritabanı şeması
tools/          # Seed ve index araçları
docker-compose.yml
docker/         # Alternatif compose örnekleri
```

## Çalıştırma Ayrıntıları

### API Temeli

- Base URL: `http://localhost:8001/api/v1`
- Sağlık: `GET /health`

### Auth

- Register: `POST /auth/register`
- Login: `POST /auth/login`
- Refresh: `POST /auth/refresh`

Örnek login:

```bash
curl -X POST http://localhost:8001/api/v1/auth/login \
  -H 'Content-Type: application/json' \
  -d '{"email":"student1@yoladgu.local","password":"pass123"}'
```

### Vektör ve Öneri

- Vektör arama: `POST /vectors/search` body: `{ text, k }`
- Index swap (blue/green): `POST /index/swap`

### Embedding Sağlayıcıları (OpenAI/Cohere veya Lokal SBERT)

- Dış API (önerilir):
  - `.env`:
    - `EMBEDDING_PROVIDER=openai` ve `OPENAI_API_KEY=...` (veya `EMBEDDING_PROVIDER=cohere` ve `COHERE_API_KEY=...`)
    - `EMBEDDING_MODEL_ID` varsayılanı: `text-embedding-3-small`
- Lokal SBERT:
  - `backend/requirements.txt` içinde `sentence-transformers` ve `torch` kurulu olmalı (eklendi).
  - `.env`:
    - `EMBEDDING_PROVIDER=sbert`
    - `EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2`
    - `EMBED_DIM` model ile uyumlu olmalı (MiniLM-L6-v2 için `384`).
  - `EMBEDDING_FALLBACK_MODE=sbert` kullanıyorsanız, SBERT bağımlılıkları gereklidir.

### CF (Collaborative Filtering) Etkinleştirme

1) Etkileşim datası hazırlayın (`jsonl` satırları: `{user_id, question_id, label}`)
2) Modeli eğitin:
```bash
docker compose exec -T backend python tools/train_cf.py --data /path/to/interactions.jsonl --k 16 --it 10
```
3) `.env` içine `CF_MODEL_PATH=backend/app/ml/models/cf.npz` ekleyin (varsayılan path budur).
4) `/recommend/ensemble` çağrıları CF skorunu da kullanır (model yoksa 0’a fallback eder).

### Üretim: Vektör İndeks Akışı ve Hata Senaryoları

Önerilen akış:
```bash
# 1) Embedding’leri üretin (dış API veya lokal SBERT)
docker compose exec -T backend python tools/seed_embeddings.py --n 1000

# 2) İnaktif slota indeks oluşturun
docker compose exec -T backend python tools/batch_indexer.py

# 3) İndeksi swap edin (blue/green)
curl -X POST http://localhost:8001/api/v1/index/swap

# 4) Sağlık ve istatistik
curl http://localhost:8001/api/v1/index/stats
```
Hata senaryoları:
- `swap failed: inactive slot not built`: İnaktif slota indeks inşa edilmeden swap istendi; önce batch_indexer çalıştırın.
- `embedding dim mismatch`: Indeks dosyasının meta’sı ile `EMBED_DIM` uyumsuz; doğru boyutla yeniden inşa edin.
- Redis bağlantı hatası: `REDIS_URL` kontrol edin; yeniden deneyin.

## Ortam Değişkenleri (Backend `.env`)

Varsayılanlar (Compose ile uyumlu):

```env
PROJECT_NAME=LearnAI
API_V1_STR=/api/v1
ENV=dev

JWT_SECRET=supersecretjwt
JWT_ALG=HS256
ACCESS_TOKEN_EXPIRE_MIN=30
REFRESH_TOKEN_EXPIRE_DAYS=7

DATABASE_URL=postgresql+asyncpg://yoladgu:password@postgres:5432/yoladgu
REDIS_URL=redis://redis:6379/0

VECTOR_BACKEND=hnsw
VECTOR_INDEX_DIR=../data/indices
EMBED_DIM=384
EMBEDDING_PROVIDER=hash
```

## Sorun Giderme

- InvalidPasswordError (DB bağlantısı): `DATABASE_URL` ile `docker-compose.yml` eşleşmiyorsa backend yeniden yaratın: `docker compose up -d --force-recreate backend`
- Port karmaşası: Dışarıya açık port 8001’dir (container içi 8000). API çağrılarında `http://localhost:8001` kullanın.
- CORS: Frontend 8080’da çalışır; backend tüm origin’lere açıktır (geliştirme için).

## Notlar

- Varsayılan vektör backend: HNSW, embed boyutu: 384 (demo hash-embedding)
- Qdrant: Konfig ve client mevcut; ancak backend implementasyonu yoktur. Şu an yalnızca HNSW/FAISS desteklenir. Qdrant desteği eklemek isterseniz `app/services/index_backends/` altına yeni backend ekleyin ve `VectorIndexManager` içinde anahtarlayın.
- Gözlemlenebilirlik: `GET /api/v1/admin/metrics/events/by_type` ve `.../by_type_24h` ile event sayımları; model/embedding aktif yapılandırması ve indeks slotu için `GET /api/v1/index/stats` kullanılabilir. Prometheus entegrasyonu eklenmesi önerilir.
