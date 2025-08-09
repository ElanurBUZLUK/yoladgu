# LearnAI Monorepo (Full E2E v2)

Bu depo; **Backend (FastAPI + Postgres + Redis)**, **Frontend (statik HTML/JS)**, 
**ML & Öneri Sistemi (LinUCB + FTRL + Vektör Arama HNSW/FAISS/Qdrant seçilebilir)**, 
**Blue/Green Index Yönetimi**, **Öğretmen-Öğrenci onay akışı**, **RBAC**, **MCP örnek konfigleri** içerir.

## Hızlı Başlangıç

```bash
# 1) Docker ile Postgres + Redis (Qdrant opsiyonel)
docker compose -f docker/docker-compose.yml up -d

# 2) DB şema
psql postgresql://learnai:learnai@localhost:5432/learnai -f db/init.sql

# 3) Backend
cd backend
cp .env.example .env
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload

# 4) ML/Vector seed + build + swap
cd ..
python tools/seed_embeddings.py --n 1000
python tools/batch_indexer.py
curl -X POST http://localhost:8000/api/v1/index/swap

# 5) Frontend
# frontend/index.html dosyasını tarayıcıda açın (CORS için backend 8000 portunda)
```

### Varsayılanlar
- Vektör backend: **HNSW** (`.env` → `VECTOR_BACKEND=hnsw`)
- Embed boyutu: 384 (demo hash-embedding)
- Auth: JWT Access/Refresh
- RBAC: `admin`, `teacher`, `student` rolleri

> Not: Bu iskelet "çalışır örnek" mantığıyla hazırlanmıştır. Üretime alırken güvenlik/observability/CI konularını genişletin.
