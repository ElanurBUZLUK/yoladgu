# Teknoloji Seçimi ve AI Modelleri

Bu doküman, Dinamik Soru Öneri Sistemi için seçilen teknoloji yığınını ve açık kaynak AI modellerini özetlemektedir.

## 1. Altyapı & Veri Katmanları

| Katman                          | Teknoloji              | Notlar                                                                     |
|---------------------------------|------------------------|-----------------------------------------------------------------------------|
| Akış Veri İşleme**           | River                  | `partial_fit` ile her cevabı anında işler (hız, doğruluk, tip).            |
| Olay Kuyruğu**                | Redis Streams          | Düşük gecikme; Neo4j ve PostgreSQL tüketicileri için event dağıtımı.        |
| Kalıcı İlişkisel Depo**       | PostgreSQL             | Öğrenci, Soru, Beceri, Cevap tabloları; ACID, zengin SQL analitik.          |
| Graf Tavsiye Deposu**         | Neo4j                  | Soru–beceri–öğrenci grafı; 1–3 hop Cypher sorguları için optimize.         |
| Öneri Motoru**                | LightFM + Neo4j Hybrid | LightFM’den aday → Neo4j multi-hop skorlama → nihai sıralama.              |
| API & Mikroservis**           | FastAPI (Python)       | Asenkron uç noktalar, Swagger/OpenAPI, WebSocket destekli.                 |
| Container & CI/CD**           | Docker + GitLab CI/CD  | Tüm servisler konteynerde; otomatik test ve deploy pipeline.               |
| Monitoring & Metrics**        | Prometheus + Grafana   | Latency, throughput, öneri başarı oranı, sistem sağlık dashboard.           |

## 2. Frontend

| Çerçeve / Kitaplık               | Teknoloji             | Notlar                                                                     |
|----------------------------------|-----------------------|-----------------------------------------------------------------------------|
|   |   |       |
| Alternatif**                   | Angular               | Kurumsal özellikler; form doğrulama, DI, router.                          |

## 3. AI & ML Modelleri

| Görev                            | Model / Kütüphane     | Notlar                                                                     |
|----------------------------------|-----------------------|-----------------------------------------------------------------------------|
| Knowledge Tracing**            | pyKT (SAKT)           | Transformer tabanlı; mastery probability hesaplama.                        |
| Online Adaptasyon**            | River ML              | Basit online regresyon / karar ağaçları; gerçek zamanlı `partial_fit`.     |
| Soru Öneri Motoru**            | LightFM               | Hibrit MF: içerik + collaborative filtering.                                |
| Graf Tabanlı Tavsiye**         | Neo4j + Cypher        | Multi-hop ilişkiler ile ilişki skorlama.                                    |
| Metin İşleme & Etiketleme**    | HuggingFace BERT      | Soru metni sınıflandırma; alt beceri etiketleri.                           |
| Kümeleme (Ön Etiketleme)**     | SBERT + KMeans        | Soruları embedding ile kümele; hızlı keşif.                                 |
