# Kullanıcı Akışı

## 1. Giriş ve Oturum Açma
1. Kullanıcı (öğrenci) web veya mobil uygulamayı açar.
2. Kullanıcı kimlik bilgilerini girerek sisteme oturum açar.
3. Başarıyla oturum açıldıktan sonra ana ekrana yönlendirilir.

## 2. Soru Çözme Modülü
1. Ana ekranda "Teste Başla" butonuna tıklar.
2. Uygulama, arka planda anlık olarak:
   - **River** aracılığıyla öğrenci etkileşimini dinlemeye başlar.
   - **Redis Streams** kuyruklarına cevap olayını kaydeder.
3. Kullanıcı ekrandaki soruyu görür, cevabı işaretler ve "Gönder"e tıklar.
4. Cevap verildiğinde:
   - River akışındaki consumer, Neo4j grafındaki `:FAILED` veya `:PASSED` kenarını günceller.
   - PostgreSQL Consumer, detaylı log kaydını `student_responses` tablosuna ekler.
5. Uygulama kullanıcıya anında geri bildirim (doğru/yanlış) gösterir.

## 3. Öneri Görüntüleme Modülü
1. Her cevap işleminden sonra veya kullanıcı "Yeni Öneri" seçeneğine tıkladığında:
   - **Öneri Motoru** çalışır:
     1. Neo4j’den multi-hop sorgu ile aday soruları çeker.
     2. PostgreSQL’den soru metası ve zorluk bilgisini alır.
     3. LightFM ile içerik-bazlı sıralama yapar.
     4. Nihai listede RedisGraph veya Neo4j fail-skoro eklenir.
   - Öneri listesi uygulamaya API üzerinden iletilir.
2. Kullanıcı ekranda önerilen soruları görür.
3. İstediği soruyu seçerek yeniden çözüm modülüne döner.

## 4. Geri Bildirim ve İzleme
1. Kullanıcı "Geri Bildirim" sekmesinden her soruya yönelik "Kolaydı" / "Zordu" gibi geri bildirimde bulunabilir.
2. Bu geri bildirim Redis Streams’e kaydedilir.
3. Periyodik batch süreçleri bu veriyi PostgreSQL’e aktarır.

## 5. Raporlama & Analiz
1. Öğretmen / öğrenci rehberliği paneline giriş yapar.
2. PostgreSQL üzerinde çalışan raporlama sorguları (ör. başarı oranı, çözüm süreleri) dashboard’da gösterilir.
3. Neo4j Graph Data Science modülü kullanılarak öğrenciler arası başarı/başarısızlık benzerlikleri analiz edilebilir.

---
