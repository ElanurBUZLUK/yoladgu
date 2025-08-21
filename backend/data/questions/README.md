# Matematik ve İngilizce Soruları JSON Yükleme Rehberi

Bu dizin matematik ve İngilizce sorularını JSON formatında yüklemek için kullanılır.

## 📁 Dosya Yapısı

```
backend/data/questions/
├── README.md                           # Bu dosya
├── example_math_questions.json         # Örnek matematik soru dosyası
├── math_questions_enhanced.json        # Gelişmiş matematik soru dosyası
├── english_questions_enhanced.json     # Gelişmiş İngilizce soru dosyası
└── your_questions.json                 # Kendi sorularınızı buraya ekleyin
```

## 📋 JSON Formatı

### 🆕 Gelişmiş Format (Önerilen)

Her soru aşağıdaki formatta olmalıdır:

```json
{
  "stem": "What is 2 + 2?",
  "options": {
    "A": "1",
    "B": "2",
    "C": "3",
    "D": "4"
  },
  "correct_answer": "D",
  "topic": "arithmetic",
  "subtopic": "addition",
  "difficulty": 0.5,
  "source": "seed",
  "metadata": {
    "estimated_time": 30,
    "learning_objectives": ["basic addition"],
    "tags": ["arithmetic", "basic"],
    "cefr_level": "A1"
  }
}
```

### 📝 Eski Format (Hala Desteklenir)

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

### 🆕 Gelişmiş Format - Zorunlu Alanlar
- `stem`: Soru metni (10-2000 karakter)
- `options`: Seçenekler objesi (A, B, C, D...)
- `correct_answer`: Doğru cevap harfi (A, B, C, D...)
- `topic`: Ana konu kategorisi

### 🆕 Gelişmiş Format - İsteğe Bağlı Alanlar
- `subtopic`: Alt konu kategorisi
- `difficulty`: Sürekli zorluk (0.0-2.0)
- `source`: Kaynak bilgisi
- `metadata`: Detaylı bilgiler

### 📝 Eski Format - Zorunlu Alanlar
- `content`: Soru metni (10-2000 karakter)
- `question_type`: Soru tipi
- `difficulty_level`: Zorluk seviyesi (1-5)
- `topic_category`: Konu kategorisi

### 📝 Eski Format - İsteğe Bağlı Alanlar
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

### 🆕 Gelişmiş Format - Sürekli Zorluk (0.0-2.0)
| Zorluk | Açıklama | Eski Seviye |
|--------|----------|-------------|
| 0.0-0.5 | Çok kolay | 1 |
| 0.5-1.0 | Kolay | 2 |
| 1.0-1.5 | Orta | 3 |
| 1.5-1.8 | Zor | 4 |
| 1.8-2.0 | Çok zor | 5 |

### 📝 Eski Format - Kesikli Seviyeler (1-5)
| Seviye | Açıklama |
|--------|----------|
| 1 | Temel aritmetik |
| 2 | Orta seviye aritmetik |
| 3 | Kesirler ve ondalıklar |
| 4 | Temel cebir |
| 5 | İleri konular |

## 📚 Konu Kategorileri

### 🧮 Matematik Konuları
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

### 🇬🇧 İngilizce Konuları
- `grammar` - Dilbilgisi
- `vocabulary` - Kelime bilgisi
- `present_tense` - Şimdiki zaman
- `past_tense` - Geçmiş zaman
- `present_perfect` - Şimdiki zamanın hikayesi
- `conditionals` - Koşul cümleleri
- `articles` - Tanımlıklar
- `prepositions` - Edatlar
- `plurals` - Çoğul yapılar
- `antonyms` - Zıt anlamlılar
- `synonyms` - Eş anlamlılar

## 🚀 Kullanım Yöntemleri

### 1. Dosya Yerleştirme (Otomatik Yükleme)
1. JSON dosyanızı `backend/data/questions/` dizinine koyun
2. Uygulama başlatıldığında otomatik olarak yüklenecektir
3. Dosya adı `.json` ile bitmelidir

### 2. API ile Yükleme

#### Matematik Soruları
```bash
curl -X POST "http://localhost:8000/api/v1/math/questions/upload-json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "file=@your_math_questions.json"
```

#### İngilizce Soruları
```bash
curl -X POST "http://localhost:8000/api/v1/english/questions/upload-json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "file=@your_english_questions.json"
```

### 3. Script ile Yükleme (Embedding ile)

#### Matematik Soruları
```bash
# Tek dosya yükleme
python scripts/load_math_questions_json.py --json data/questions/math_questions_enhanced.json

# Dizin yükleme
python scripts/load_math_questions_json.py --dir data/questions/
```

#### İngilizce Soruları
```bash
# Tek dosya yükleme
python scripts/load_english_questions_json.py --json data/questions/english_questions_enhanced.json

# Dizin yükleme
python scripts/load_english_questions_json.py --dir data/questions/
```

### 4. Örnek Kullanım

#### Matematik Soruları
```bash
# Gelişmiş örnek dosyayı kopyalayın
cp math_questions_enhanced.json my_math_questions.json

# Dosyayı düzenleyin
nano my_math_questions.json

# Script ile yükleyin (embedding ile)
python scripts/load_math_questions_json.py --json my_math_questions.json
```

#### İngilizce Soruları
```bash
# Gelişmiş örnek dosyayı kopyalayın
cp english_questions_enhanced.json my_english_questions.json

# Dosyayı düzenleyin
nano my_english_questions.json

# Script ile yükleyin (embedding ile)
python scripts/load_english_questions_json.py --json my_english_questions.json
```

#### Genel
```bash
# Veya uygulamayı yeniden başlatın (otomatik yükleme)
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
