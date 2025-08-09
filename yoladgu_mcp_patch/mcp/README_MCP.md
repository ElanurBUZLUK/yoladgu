# MCP Patch (Postgres/Redis/Filesystem/S3 + Köprü Endpoint’leri)

Bu klasör, IDE’nizde (Cursor/Claude/Windsurf) **MCP server**’ları tanımlayıp projeyle birlikte kullanmanız için
JSON konfigleri ve FastAPI tarafında yardımcı **köprü endpointlerini** içerir.

## 1) JSON konfigleri IDE’ye ekleme
- Cursor: *Settings → Tools & Integrations → MCP Servers → New* → ilgili JSON’u yapıştırın.
- Claude Desktop/Windsurf: MCP ayarlarında “Add Server” ve JSON’u girin.
- Gerekli ortam değişkenlerini `.env` dosyanıza ekleyin (aşağıya bakınız).

Konfigler `mcp/configs/` içinde. Paket adları/komutları sürüme göre değişebilir;
güncel liste: `https://github.com/modelcontextprotocol/servers`

## 2) Backend köprü endpointleri
`app/api/v1/endpoints/mcp_bridge.py` dosyası aşağıdaki uçları sağlar:

- `GET  /api/v1/mcp/configs` → Mevcut JSON konfigleri ve eksik env’ler
- `POST /api/v1/mcp/redis/keys` → Pattern ile Redis anahtar listeleme (rate-limit’li)
- `GET  /api/v1/mcp/redis/queue-stats?keys=k1,k2` → Basit kuyruk/stream uzunluğu
- `POST /api/v1/mcp/s3/list` → S3 prefix listeleme
- `POST /api/v1/mcp/filesystem/import` → Bir klasörde *.pdf|*.md tarama (dry-run)
- `POST /api/v1/mcp/sql/run` → **SELECT-only** güvenli Postgres sorgu (demo)

> Not: Bu uçlar prod’da yetki/RBAC kontrolüyle kullanılmalıdır.

## 3) Kurulum
1. Bu patch’i reponuzun köküne kopyalayın (dizin yapısını koruyun).
2. `requirements_mcp.txt` içeriğini ana `requirements.txt`’inize ekleyin ve yükleyin:
   ```bash
   pip install -r requirements.txt
   ```
3. FastAPI’ye router ekleyin (`app/main.py`):
   ```python
   from app.api.v1.endpoints import mcp_bridge
   app.include_router(mcp_bridge.router, prefix=settings.API_V1_STR)
   ```
4. `.env` dosyanıza aşağıdaki MCP anahtarlarını ekleyin (örnek: `.env.mcp.example`).

## 4) .env anahtarları
- **Postgres**: `PG_DSN=postgresql://user:pass@host:5432/dbname`
- **Redis**: `REDIS_URL=redis://localhost:6379/0`
- **S3**: `AWS_ACCESS_KEY_ID=...`, `AWS_SECRET_ACCESS_KEY=...`, `AWS_DEFAULT_REGION=us-east-1`
- **S3** ek: `S3_BUCKET_NAME=your-bucket`
- **Filesystem import** için server’ın dosya erişim izinlerini ayarlayın (Docker’da volume mount).

## 5) Güvenlik
- `sql/run` yalnızca `SELECT` izinlidir ve `LIMIT` zorunludur.
- Redis/S3/FS uçlarında minimal rate-limit bulunur. Prod’da ek gateway rate-limit ve RBAC şarttır.
