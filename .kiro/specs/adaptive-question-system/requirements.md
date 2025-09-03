# Requirements Document

## Introduction

Bu proje, öğrencilerin matematik ve İngilizce alanlarında kişiselleştirilmiş soru önerileri alan adaptif bir öğrenme sistemidir. Sistem, öğrencinin geçmiş performansını analiz ederek IRT (Item Response Theory) ve bandit algoritmaları kullanarak en uygun soruları önerir ve gerektiğinde yeni sorular üretir.

## Requirements

### Requirement 1

**User Story:** Bir öğrenci olarak, seviyeme uygun matematik soruları almak istiyorum ki öğrenme sürecim optimize olsun.

#### Acceptance Criteria

1. WHEN öğrenci sisteme giriş yaptığında THEN sistem öğrencinin mevcut matematik seviyesini (θ) IRT modeli ile hesaplamalı
2. WHEN öğrenci bir matematik sorusunu çözdüğünde THEN sistem öğrencinin θ değerini güncellemeli
3. WHEN sistem soru önerisi yaptığında THEN önerilen soruların zorluk seviyesi öğrencinin θ değerine uygun olmalı (|θ-b| ≤ 0.3)
4. IF öğrenci bir soruyu yanlış yaparsa THEN sistem hata türünü analiz etmeli ve error_profile güncellenmeli

### Requirement 2

**User Story:** Bir öğrenci olarak, İngilizce dilbilgisi konularında eksik olduğum alanlarda cloze (boşluk doldurma) soruları almak istiyorum ki zayıf olduğum konuları güçlendirebiliyim.

#### Acceptance Criteria

1. WHEN öğrenci İngilizce soru çözdüğünde THEN sistem hata türlerini (prepositions, articles, SVA, collocations) kategorize etmeli
2. WHEN sistem İngilizce soru önerisi yaptığında THEN öğrencinin geçmiş hata profiline göre hedeflenmiş sorular sunmalı
3. WHEN sistem cloze sorusu ürettiğinde THEN CEFR seviyesine uygun metinler kullanmalı
4. WHEN cloze sorusu oluşturulduğunda THEN tek doğru cevap garantisi sağlanmalı

### Requirement 3

**User Story:** Bir öğretmen olarak, öğrencilerimin öğrenme ilerlemesini takip etmek istiyorum ki hangi konularda zorlandıklarını görebiliyim.

#### Acceptance Criteria

1. WHEN öğretmen dashboard'a eriştiğinde THEN öğrencilerin θ değerleri ve hata profilleri görüntülenebilmeli
2. WHEN öğretmen bir öğrencinin detaylarını incelediğinde THEN öğrencinin beceri bazlı performans analizi sunulmalı
3. IF öğretmen yeni soru şablonu eklerse THEN sistem şablonu doğrulayıp onaya sunmalı

### Requirement 4

**User Story:** Sistem yöneticisi olarak, sistemin performansını ve kalitesini izlemek istiyorum ki SLA hedeflerini karşıladığımızdan emin olabiliyim.

#### Acceptance Criteria

1. WHEN sistem çalışırken THEN p95 latency değerleri endpoint bazlı izlenmeli
2. WHEN soru önerileri yapılırken THEN bandit algoritmasının exploration/exploitation oranı loglanmalı
3. WHEN sistem metrikleri sorgulandığında THEN faithfulness, difficulty_match, coverage değerleri raporlanmalı
4. IF sistem hatası oluşursa THEN audit log'lar decision trail ile birlikte kaydedilmeli

### Requirement 5

**User Story:** Bir öğrenci olarak, kişisel verilerimin güvenli şekilde işlendiğinden emin olmak istiyorum ki gizliliğim korunsun.

#### Acceptance Criteria

1. WHEN öğrenci kaydolurken THEN consent_flag ile veri işleme izni alınmalı
2. WHEN sistem log tutarken THEN PII veriler maskelenmeli veya redact edilmeli
3. WHEN öğrenci verilerine erişim sağlanırken THEN RBAC kuralları uygulanmalı
4. IF öğrenci hesabını silerse THEN kişisel veriler GDPR uyumlu şekilde silinmeli

### Requirement 6

**User Story:** Sistem geliştiricisi olarak, matematik soruları için programatik doğrulama yapmak istiyorum ki üretilen soruların tek doğru cevabı olduğundan emin olabiliyim.

#### Acceptance Criteria

1. WHEN matematik sorusu üretildiğinde THEN programatik çözücü (sympy) ile doğrulama yapılmalı
2. WHEN soru şablonu kullanıldığında THEN parametreler belirlenen aralıklarda olmalı
3. IF çoklu doğru cevap tespit edilirse THEN soru reddedilmeli
4. WHEN distractor üretildiğinde THEN yanılgı tabanlı yanlış seçenekler oluşturulmalı

### Requirement 7

**User Story:** Sistem olarak, bandit algoritmaları kullanarak adaptif soru seçimi yapmak istiyorum ki öğrencinin öğrenme deneyimi optimize olsun.

#### Acceptance Criteria

1. WHEN soru önerisi yapılırken THEN LinUCB/LinTS algoritması kullanılmalı
2. WHEN bandit kararı verilirken THEN propensity değerleri loglanmalı
3. WHEN kısıtlı bandit çalışırken THEN minimum başarı oranı (%60) ve coverage (%80) sağlanmalı
4. IF exploration ratio düşükse THEN sistem yeni soru türlerini keşfetmeli

### Requirement 8

**User Story:** Sistem olarak, hibrit arama (dense + sparse) kullanarak en uygun soru adaylarını bulmak istiyorum ki öğrenciye en alakalı sorular sunulabilsin.

#### Acceptance Criteria

1. WHEN soru arama yapılırken THEN BM25 ve dense embedding skorları birleştirilmeli
2. WHEN aday sorular filtrelenirken THEN metadata (dil, seviye, beceri) kriterleri uygulanmalı
3. WHEN re-ranking yapılırken THEN cross-encoder modeli kullanılmalı
4. IF cache'de sonuç varsa THEN retrieval cache'den dönülmeli (TTL: 24h)