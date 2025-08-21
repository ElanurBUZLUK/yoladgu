# Matematik Soruları JSON Yükleme Rehberi

Bu dizin matematik sorularını JSON formatında yüklemek için kullanılır.

## 📁 Dosya Yapısı

```
backend/data/questions/
├── README.md                    # Bu dosya
├── example_math_questions.json  # Örnek soru dosyası
└── your_questions.json          # Kendi sorularınızı buraya ekleyin
```

## 📋 JSON Formatı

Her soru aşağıdaki formatta olmalıdır:

```json
{
  "subject": "math",
  "content": "Soru metni buraya",
  "question_type": "multiple_choice",
  "difficulty_level": 1,
  "topic_category": "addition",
  "correct_answer": "Doğru cevap",
  "options": ["Seçenek 1", "Seçenek 2", "Seçenek 3", "Seçenek 4"],
  "source_type": "manual",
  "question_metadata": {
    "estimated_time": 30,
    "learning_objectives": ["Hedef 1", "Hedef 2"],
    "tags": ["tag1", "tag2"]
  }
}
```

## 🔧 Desteklenen Alanlar

### Zorunlu Alanlar
- `content`: Soru metni (10-2000 karakter)
- `question_type`: Soru tipi
- `difficulty_level`: Zorluk seviyesi (1-5)
- `topic_category`: Konu kategorisi

### İsteğe Bağlı Alanlar
- `correct_answer`: Doğru cevap
- `options`: Çoktan seçmeli sorular için seçenekler
- `source_type`: Kaynak tipi
- `question_metadata`: Ek bilgiler

## 📝 Soru Tipleri

| Tip | Açıklama |
|-----|----------|
| `multiple_choice` | Çoktan seçmeli |
| `fill_blank` | Boşluk doldurma |
| `open_ended` | Açık uçlu |
| `true_false` | Doğru/Yanlış |

## 🎯 Zorluk Seviyeleri

| Seviye | Açıklama |
|--------|----------|
| 1 | Temel aritmetik |
| 2 | Orta seviye aritmetik |
| 3 | Kesirler ve ondalıklar |
| 4 | Temel cebir |
| 5 | İleri konular |

## 📚 Konu Kategorileri

- `addition` - Toplama
- `subtraction` - Çıkarma
- `multiplication` - Çarpma
- `division` - Bölme
- `fractions` - Kesirler
- `decimals` - Ondalıklar
- `algebra` - Cebir
- `geometry` - Geometri
- `patterns` - Örüntüler
- `word_problems` - Sözel problemler

## 🚀 Kullanım Yöntemleri

### 1. Dosya Yerleştirme (Otomatik Yükleme)
1. JSON dosyanızı `backend/data/questions/` dizinine koyun
2. Uygulama başlatıldığında otomatik olarak yüklenecektir
3. Dosya adı `.json` ile bitmelidir

### 2. API ile Yükleme
```bash
curl -X POST "http://localhost:8000/api/v1/math/questions/upload-json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "file=@your_questions.json"
```

### 3. Örnek Kullanım
```bash
# Örnek dosyayı kopyalayın
cp example_math_questions.json my_questions.json

# Dosyayı düzenleyin
nano my_questions.json

# Uygulamayı yeniden başlatın
python -m app.main
```

## ✅ Doğrulama

Yüklenen sorular otomatik olarak doğrulanır:
- Zorunlu alanların varlığı
- Veri tiplerinin uygunluğu
- Değer aralıklarının kontrolü

## 🐛 Hata Ayıklama

Hatalar log dosyalarında görüntülenir:
```bash
tail -f logs/app.log
```

## 📊 İstatistikler

Yükleme sonrası şu bilgiler döner:
- Başarıyla yüklenen soru sayısı
- Başarısız olan soru sayısı
- Hata detayları

## 🔄 Güncelleme

Mevcut soruları güncellemek için:
1. JSON dosyasını düzenleyin
2. Uygulamayı yeniden başlatın
3. Veya API endpoint'ini kullanın

## 📞 Destek

Sorun yaşarsanız:
1. Log dosyalarını kontrol edin
2. JSON formatını doğrulayın
3. Örnek dosyayı referans alın
