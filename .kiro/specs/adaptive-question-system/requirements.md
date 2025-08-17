# Requirements Document

## Introduction

Bu sistem, öğrencilerin geçmiş hatalarını analiz ederek kişiselleştirilmiş sorular öneren akıllı bir eğitim platformudur. Sistem, matematik ve İngilizce alanlarında öğrencilerin yaptığı hataları takip eder, seviyelerini belirler ve benzer hatalar yapan diğer öğrencilerin zorlandığı konulardan sorular üretir.

## Requirements

### Requirement 1: Öğrenci Hata Takip Sistemi

**User Story:** Bir öğretmen olarak, öğrencilerimin yaptığı hataları sistematik olarak takip etmek istiyorum ki onların zayıf yönlerini belirleyebileyim.

#### Acceptance Criteria

1. WHEN öğrenci bir soruyu yanlış cevapladığında THEN sistem hata detaylarını (soru tipi, hata kategorisi, tarih) kaydetmelidir
2. WHEN öğrenci bir matematik sorusunu yanlış yaptığında THEN sistem hangi matematik konusunda (cebir, geometri, vb.) hata yaptığını kategorize etmelidir
3. WHEN öğrenci bir İngilizce sorusunu yanlış yaptığında THEN sistem hangi grammar kuralı veya vocabulary alanında hata yaptığını kaydetmelidir
4. IF öğrenci aynı tipte hatayı tekrar yaparsa THEN sistem bu hatayı "tekrarlayan hata" olarak işaretlemelidir

### Requirement 2: Dinamik Seviye ve Zorluk Belirleme

**User Story:** Bir öğrenci olarak, performansıma göre hem seviyemin hem de soru zorluklarının dinamik olarak ayarlanmasını istiyorum ki sürekli uygun zorlukta sorularla karşılaşayım.

#### Acceptance Criteria

1. WHEN öğrenci ilk defa sisteme girdiğinde THEN sistem seviye belirleme testi sunmalıdır
2. WHEN öğrenci soruları çözdükçe THEN sistem dinamik olarak seviyesini güncellemeli
3. IF öğrenci son 10 sorunun %80'ini doğru yaparsa THEN seviyesi bir üst seviyeye çıkmalıdır
4. IF öğrenci son 10 sorunun %40'ından azını doğru yaparsa THEN seviyesi bir alt seviyeye inmeli
5. WHEN öğrenci seviyesi 4 iken zorluk 5 soruları kolayca çözerse THEN o soruların zorluk seviyesi 4'e düşürülmelidir
6. WHEN öğrenci seviyesi 5 iken zorluk 4 sorularda zorlanırsa THEN o soruların zorluk seviyesi 5'e yükseltilmelidir
7. WHEN soru zorluk seviyesi değiştiğinde THEN sistem bu değişikliği kaydetmeli ve gelecek önerilerde kullanmalıdır
8. WHEN seviye veya zorluk değiştiğinde THEN sistem öğrenciyi bilgilendirmeli

### Requirement 3: Akıllı Soru Öneri ve Üretim Sistemi

**User Story:** Bir öğrenci olarak, seviyeme uygun matematik soruları ve hatalarıma odaklanan İngilizce soruları almak istiyorum ki etkili öğrenebilim.

#### Acceptance Criteria

1. WHEN matematik sorusu önerildiğinde THEN öğrencinin mevcut seviyesine uygun hazır sorular soru havuzundan seçilmelidir
2. WHEN matematik sorusu seçilirken THEN sorunun güncel zorluk seviyesi öğrencinin seviyesi ile karşılaştırılmalıdır
3. WHEN İngilizce sorusu üretildiğinde THEN öğrencinin yanlış yaptığı grammar kuralları ve vocabulary'yi içeren yeni sorular LLM ile oluşturulmalıdır
4. WHEN sistem başlatıldığında THEN hem matematik hem İngilizce için örnek soru havuzu yüklenmelidir
5. IF öğrenci matematik konusunda hata yaparsa THEN o konudan daha fazla hazır soru önerilmelidir
6. WHEN soru önerilirken THEN sorunun geçmiş performans verilerine göre güncellenmiş zorluk seviyesi kullanılmalıdır
7. WHEN İngilizce soru üretildiğinde THEN öğrencinin son hata yaptığı grammar/vocabulary alanları kullanılmalıdır

### Requirement 4: Benzer Öğrenci Analizi

**User Story:** Bir öğrenci olarak, benimle aynı zorlukları yaşayan diğer öğrencilerin çözemediği sorularla karşılaşmak istiyorum ki daha etkili öğrenebilim.

#### Acceptance Criteria

1. WHEN sistem soru önerdiğinde THEN benzer hata profiline sahip öğrencilerin zorlandığı soruları öncelemeli
2. WHEN öğrenci profili analiz edildiğinde THEN aynı hata kategorilerinde zorlanan öğrenciler bulunmalıdır
3. IF benzer profilli öğrencilerin %70'i bir soruyu yanlış yapmışsa THEN o soru "zorlu soru" olarak işaretlenmeli
4. WHEN zorlu sorular önerildiğinde THEN öğrenci uyarılmalı ve ekstra destek sunulmalıdır

### Requirement 5: Performans Takibi ve Raporlama

**User Story:** Bir öğretmen olarak, öğrencilerimin gelişimini takip etmek istiyorum ki hangi alanlarda daha fazla desteğe ihtiyaç duyduklarını görebilim.

#### Acceptance Criteria

1. WHEN öğretmen rapor talep ettiğinde THEN öğrencinin son 30 günlük performans grafiği gösterilmelidir
2. WHEN rapor oluşturulduğunda THEN en çok hata yapılan konular listelenmelidir
3. WHEN öğrenci gelişimi analiz edildiğinde THEN hangi konularda ilerleme kaydettiği gösterilmelidir
4. IF öğrenci 7 gün boyunca aynı tipte hata yapmaya devam ederse THEN sistem öğretmeni uyarmalıdır

### Requirement 6: Soru Havuzu Yönetimi ve PDF Entegrasyonu

**User Story:** Bir öğretmen olarak, PDF formatındaki sorularımı sisteme yüklemek ve öğrencilerin bunları frontend'de görmesini istiyorum ki mevcut soru arşivimi kullanabileyim.

#### Acceptance Criteria

1. WHEN öğretmen PDF soru dosyası yüklediğinde THEN sistem PDF'i parse ederek soruları veritabanına kaydetmelidir
2. WHEN PDF'den sorular çıkarıldığında THEN her soru için zorluk seviyesi, konu kategorisi ve soru tipi belirlenmelidir
3. WHEN öğrenci soruyu çözerken THEN PDF'deki orijinal soru formatı frontend'de görüntülenmelidir
4. WHEN sistem başlatıldığında THEN hem matematik hem İngilizce için örnek soru havuzu yüklenmelidir
5. WHEN matematik sorusu önerildiğinde THEN aynı tipte art arda 3'ten fazla soru sunmamalıdır
6. WHEN İngilizce soruları üretildiğinde THEN grammar, vocabulary ve reading kategorilerinden dengeli sorular oluşturulmalıdır
7. WHEN PDF yükleme işlemi tamamlandığında THEN sistem yüklenen soru sayısını ve kategorilerini raporlamalıdır

### Requirement 7: PDF Soru Yönetimi ve Görüntüleme

**User Story:** Bir öğrenci olarak, PDF'den gelen soruları net ve okunabilir şekilde görmek istiyorum ki soruları rahatça çözebilim.

#### Acceptance Criteria

1. WHEN PDF soru yüklendiğinde THEN sistem soruları metin ve görsel olarak ayrıştırmalıdır
2. WHEN soru PDF'den geliyorsa THEN frontend'de orijinal formatında (resim/metin) gösterilmelidir
3. WHEN matematik sorusu PDF'den gelirse THEN formüller ve şekiller doğru şekilde render edilmelidir
4. WHEN öğrenci PDF sorusunu cevaplarken THEN cevap alanı soru formatına uygun olmalıdır
5. IF PDF'de çoktan seçmeli soru varsa THEN seçenekler tıklanabilir butonlar olarak gösterilmelidir
6. WHEN PDF soru görüntülenirken THEN zoom ve kaydırma özellikleri aktif olmalıdır

### Requirement 8: Sistem Performansı ve Ölçeklenebilirlik

**User Story:** Bir sistem yöneticisi olarak, sistemin yüksek kullanıcı sayısında da hızlı çalışmasını istiyorum ki öğrenci deneyimi etkilenmesin.

#### Acceptance Criteria

1. WHEN soru üretimi talep edildiğinde THEN sistem 2 saniye içinde yanıt vermelidir
2. WHEN 1000 eşzamanlı kullanıcı olduğunda THEN sistem performansı %20'den fazla düşmemelidir
3. WHEN hata analizi yapılırken THEN sistem diğer işlemleri bloklamamalıdır
4. IF sistem yükü %80'i geçerse THEN otomatik ölçeklendirme devreye girmelidir