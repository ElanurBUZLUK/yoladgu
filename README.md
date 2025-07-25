# yoladgu

Bu proje, öğrencilerin performansına göre dinamik olarak soru öneren bir eğitim platformudur. Sistem, AI ve makine öğrenmesi teknolojilerini kullanarak kişiselleştirilmiş öğrenme deneyimi sunar.

## Özellikler

- **Dinamik Soru Önerisi**: Öğrenci performansına göre kişiselleştirilmiş soru önerileri
- **Gerçek Zamanlı Öğrenme**: River ML ile anlık öğrenme adaptasyonu
- **Çoklu Rol Desteği**: Öğrenci, Öğretmen ve Admin rolleri
- **Graf Tabanlı Analiz**: Neo4j ile ilişkisel veri analizi
- **Modern Web Arayüzü**: Angular tabanlı responsive tasarım
- **Recommendation CLI**: Komut satırından öneri almak için `features/recommendation_cli.py` dosyasını çalıştırabilirsiniz.

## Teknoloji Yığını

### Backend
- **FastAPI**: Modern Python web framework
- **PostgreSQL**: Ana veritabanı
- **Neo4j**: Graf veritabanı (ilişkisel analiz için)
- **Redis**: Önbellek ve kuyruk yönetimi
- **River ML**: Online makine öğrenmesi
- **LightFM**: Hibrit öneri sistemi

### Frontend
- **Angular 20**: Modern web framework
- **TypeScript**: Tip güvenli JavaScript
- **SCSS**: Gelişmiş CSS ön işlemcisi

## Kurulum

### Backend Kurulumu

1. Python sanal ortamı oluşturun:
```bash
cd backend
python -m venv venv
source venv/bin/activate  # Linux/Mac
# veya
venv\Scripts\activate  # Windows
```

2. Bağımlılıkları yükleyin:
```bash
pip install -r requirements.txt
```

3. Environment değişkenlerini ayarlayın:
```bash
cp .env.example .env
# .env dosyasını düzenleyin
```
#### LLM Entegrasyonu
`.env` dosyasına LLM servisleri için gerekli API anahtarlarını ekleyin:
```bash
HUGGINGFACE_API_TOKEN=YOUR_TOKEN
# veya
OPENAI_API_KEY=YOUR_KEY
```

4. Veritabanını başlatın:
```bash
alembic upgrade head
```

5. Uygulamayı çalıştırın:
```bash
python run.py
```

### Frontend Kurulumu

1. Node.js bağımlılıklarını yükleyin:
```bash
cd frontend
npm install
```

2. Uygulamayı çalıştırın:
```bash
npm start
```

## Kullanım

1. Tarayıcınızda `http://localhost:4200` adresine gidin
2. Kayıt olun veya giriş yapın
3. Rolünüze göre (öğrenci/öğretmen/admin) ilgili dashboard'a yönlendirileceksiniz
4. Öğrenci olarak soru çözmeye başlayabilir, öğretmen olarak öğrenci performanslarını takip edebilirsiniz

## Features

- **Recommendation CLI**: Komut satırından öneri almak için `features/recommendation_cli.py` dosyasını çalıştırabilirsiniz.

```bash
python features/recommendation_cli.py
```

Script sizden bir öğrenci ID'si ister ve önerilen soruları listeler.

## API Dokümantasyonu

Backend API dokümantasyonuna `http://localhost:8000/docs` adresinden erişebilirsiniz.

## Katkıda Bulunma

1. Bu repository'yi fork edin
2. Feature branch oluşturun (`git checkout -b feature/amazing-feature`)
3. Değişikliklerinizi commit edin (`git commit -m 'Add some amazing feature'`)
4. Branch'inizi push edin (`git push origin feature/amazing-feature`)
5. Pull Request oluşturun

## Lisans

Bu proje MIT lisansı altında lisanslanmıştır. 