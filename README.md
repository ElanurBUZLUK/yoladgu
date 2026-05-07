## Projenin Amacı

Yoladgu projesinin amacı, öğrencilerin Matematik ve İngilizce alanlarındaki öğrenme süreçlerini daha kişisel, ölçülebilir ve veriye dayalı hale getiren adaptif bir soru öneri sistemi geliştirmektir.

Geleneksel soru çözme platformlarında öğrenciye çoğunlukla konu başlığına, sınıf seviyesine veya genel zorluk derecesine göre sorular sunulur. Bu yaklaşım temel düzeyde faydalı olsa da her öğrencinin öğrenme ihtiyacını tam olarak karşılamaz. Çünkü aynı konuda yanlış yapan iki öğrenci, aslında farklı nedenlerle hata yapıyor olabilir. Bir öğrenci kavramı hiç anlamamış olabilirken, başka bir öğrenci işlem hatası yapıyor, dikkatsiz davranıyor veya belirli bir soru tipinde zorlanıyor olabilir.

Yoladgu, bu problemi daha ayrıntılı ele almayı hedefler. Sistem, öğrencinin yalnızca doğru veya yanlış cevap verip vermediğine bakmaz; aynı zamanda öğrencinin geçmiş performansını, tekrar eden hata türlerini, konu bazlı eksiklerini ve gelişim ihtiyacını analiz eder. Böylece öğrenciye rastgele ya da yalnızca zorluk seviyesine göre değil, kendi öğrenme profiline daha uygun sorular önerilebilir.

Bu projenin temel hedefleri şunlardır:

- Öğrencinin çözdüğü sorulardan anlamlı bir performans profili oluşturmak.
- Öğrencinin hangi konularda ve hangi becerilerde zorlandığını tespit etmek.
- Yanlış cevaplardan hata örüntüleri çıkarmak.
- Öğrencinin seviyesine ve eksiklerine uygun soru önerileri sunmak.
- Matematik ve İngilizce alanlarında adaptif öğrenme deneyimi sağlamak.
- Soru öneri sürecini yalnızca sabit kurallara değil, veri ve model tabanlı analizlere dayandırmak.
- Öğrencinin zaman içindeki gelişimini takip ederek önerileri dinamik biçimde güncellemek.
- Öğrenme sürecini daha hedefli, kişiselleştirilmiş ve sürdürülebilir hale getirmek.

Bu yaklaşım sayesinde Yoladgu, öğrencinin sadece daha fazla soru çözmesini değil, doğru eksik alanlara yönelmesini amaçlar. Böylece sistem, klasik bir soru bankası mantığından çıkarak öğrenciyi tanıyan ve ona göre yönlendirme yapan akıllı bir öğrenme destek sistemine dönüşür.

---

## Temel Özellikler

Yoladgu, adaptif öğrenme ve kişiselleştirilmiş soru önerisi yaklaşımını destekleyen farklı bileşenlerden oluşur. Projede hem backend tarafında öneri sistemi mantığı hem de frontend tarafında kullanıcıyla etkileşim kurabilecek bir yapı hedeflenmiştir.

### Kişiselleştirilmiş Soru Önerisi

Sistem, öğrencinin geçmiş cevaplarını ve performansını analiz ederek ona daha uygun sorular önermeyi amaçlar. Bu öneriler yalnızca genel zorluk seviyesine göre değil, öğrencinin hata yaptığı konulara, beceri eksiklerine ve gelişim ihtiyacına göre şekillenir.

Bu yapı sayesinde her öğrenci için farklı bir öğrenme akışı oluşturulabilir.

### Matematik ve İngilizce Desteği

Proje, Matematik ve İngilizce alanlarında soru önerisi yapabilecek şekilde tasarlanmıştır.

Matematik tarafında öğrencinin işlem, kavram ve problem çözme becerileri değerlendirilebilir. İngilizce tarafında ise kelime bilgisi, gramer, cloze test yapısı ve anlam ilişkileri gibi alanlar üzerinden öğrencinin eksikleri analiz edilebilir.

### Hata Profili Analizi

Yoladgu’nun en önemli özelliklerinden biri, öğrencinin yaptığı hataları ayrı bir öğrenme sinyali olarak ele almasıdır.

Sistem, öğrencinin yanlış cevaplarını inceleyerek hangi hata türlerinin tekrar ettiğini anlamaya çalışır. Örneğin öğrenci belirli bir gramer yapısında, matematiksel işlemde veya kavramsal ilişkide sık sık hata yapıyorsa, bu durum öğrencinin hata profiline eklenir.

Bu hata profili daha sonra öneri sisteminin karar verme sürecinde kullanılır.

### Error-Aware Recommendation Yaklaşımı

Projede error-aware recommendation yani hata duyarlı öneri sistemi yaklaşımı kullanılmaktadır.

Bu yaklaşımda öğrencinin yaptığı hatalar, öneri sisteminin merkezine alınır. Sistem yalnızca öğrencinin başarı oranına bakmaz; öğrencinin hangi hataları yaptığına, bu hataların hangi konularla ilişkili olduğuna ve benzer hata profiline sahip öğrencilerin hangi sorularla gelişim gösterdiğine de odaklanır.

Amaç, öğrencinin eksik olduğu alanı daha hızlı ve doğru şekilde tespit ederek ona gerçekten fayda sağlayacak soruları önermektir.

### Vector Search Tabanlı Soru Arama

Projede soru arama ve öneri süreçlerini güçlendirmek için vector search yaklaşımına uygun bir altyapı tasarlanmıştır.

Sorular, yalnızca metin olarak değil, anlamsal temsiller üzerinden de değerlendirilebilir. Bu sayede sistem, birebir kelime eşleşmesi olmasa bile anlam olarak benzer veya ilişkili soruları bulabilir.

Bu yaklaşım özellikle büyük soru havuzlarında daha esnek ve güçlü bir arama deneyimi sağlar.

### Hybrid Retrieval Mantığı

Yoladgu, soru arama sürecinde yalnızca tek bir arama yöntemine bağlı kalmayan hibrit bir yaklaşımı destekler.

Hybrid retrieval yapısı ile hem dense search yani embedding tabanlı anlamsal arama hem de sparse search yani anahtar kelime ve klasik arama sinyalleri birlikte kullanılabilir.

Bu sayede sistem hem anlamsal benzerliği hem de doğrudan kelime eşleşmelerini dikkate alarak daha kaliteli sonuçlar üretebilir.

### Backend Servis Mimarisi

Backend tarafında FastAPI tabanlı modüler bir yapı hedeflenmiştir. API endpointleri, öneri servisleri, veri tabanı işlemleri, model yapıları ve vector search bileşenleri ayrı katmanlar halinde ele alınır.

Bu mimari, projenin ilerleyen aşamalarda daha kolay genişletilmesini ve farklı servislerin bağımsız şekilde geliştirilmesini sağlar.

### Angular Tabanlı Frontend

Frontend tarafında Angular kullanılarak kullanıcıyla etkileşim kurabilecek bir arayüz geliştirilmesi hedeflenmiştir.

Bu arayüz üzerinden öğrencinin sisteme giriş yapması, soru çözmesi, önerilen soruları görüntülemesi ve öğrenme sürecindeki ilerlemesini takip etmesi amaçlanabilir.

### Docker ile Çalıştırılabilir Servis Yapısı

Projede Docker desteği ile backend, veritabanı, cache ve diğer servislerin daha düzenli şekilde çalıştırılması hedeflenmiştir.

Bu yapı, geliştirme ortamının daha kolay kurulmasını ve projenin farklı makinelerde daha tutarlı çalışmasını sağlar.

### Desteklenen Altyapı Bileşenleri

Projede aşağıdaki altyapı bileşenleri desteklenmektedir:

- PostgreSQL ile kalıcı veri saklama
- Redis ile cache ve hızlı erişim desteği
- Elasticsearch ile sparse search desteği
- Qdrant ile vector database kullanımı
- HNSW ve FAISS ile hızlı benzerlik araması
- Docker ile servis yönetimi
- Pytest ile test altyapısı

---

## Kullanılan Teknolojiler

Bu proje hem backend hem frontend bileşenlerinden oluşan tam kapsamlı bir uygulama mimarisi hedefler. Kullanılan teknolojiler, öneri sistemi, veri yönetimi, API geliştirme, frontend arayüzü ve servis yönetimi ihtiyaçlarına göre seçilmiştir.

### Backend

Backend tarafında Python ve FastAPI merkezli bir mimari kullanılmıştır.

- **Python:** Projenin ana backend geliştirme dilidir. Veri işleme, öneri sistemi, makine öğrenmesi ve API geliştirme süreçlerinde kullanılır.
- **FastAPI:** REST API geliştirmek için kullanılır. Hızlı, modern ve OpenAPI dokümantasyon desteği güçlü bir framework olduğu için tercih edilmiştir.
- **SQLModel:** Veritabanı modellerini ve veri doğrulama yapılarını daha düzenli kurmak için kullanılır.
- **SQLAlchemy:** Veritabanı işlemleri ve ORM katmanı için kullanılır.
- **Alembic:** Veritabanı migration işlemlerini yönetmek için kullanılır.
- **PostgreSQL:** Öğrenci bilgileri, soru kayıtları, cevap geçmişi ve sistem verilerinin saklanması için kullanılabilecek ilişkisel veritabanıdır.
- **Redis:** Cache, hızlı veri erişimi ve performans optimizasyonu için kullanılabilir.
- **Sentence Transformers:** Soru metinlerini vektör temsillerine dönüştürmek ve semantic search işlemlerini desteklemek için kullanılır.
- **scikit-learn:** Makine öğrenmesi tabanlı yardımcı modeller, sınıflandırma veya seçim mekanizmaları için kullanılabilir.
- **NumPy:** Sayısal işlemler ve vektör hesaplamaları için kullanılır.
- **SciPy:** Bilimsel hesaplama ve istatistiksel işlemler için kullanılabilir.
- **SymPy:** Matematiksel ifadelerin işlenmesi ve doğrulanması için kullanılabilir.
- **Qdrant:** Vektör veritabanı olarak semantic search işlemlerini destekler.
- **HNSW:** Approximate nearest neighbor search için kullanılabilecek hızlı bir indexleme yaklaşımıdır.
- **FAISS:** Büyük ölçekli vektör benzerlik aramaları için kullanılan güçlü bir arama kütüphanesidir.
- **Elasticsearch:** Sparse search ve klasik arama işlemleri için kullanılabilir.
- **Docker:** Servislerin container yapısı içinde daha kolay çalıştırılmasını sağlar.
- **Pytest:** Backend testlerini yazmak ve çalıştırmak için kullanılır.

### Frontend

Frontend tarafında Angular tabanlı bir yapı kullanılmıştır.

- **Angular:** Kullanıcı arayüzünü geliştirmek için kullanılan frontend frameworküdür.
- **TypeScript:** Angular geliştirmelerinde kullanılan ana programlama dilidir.
- **HTML:** Sayfa yapılarının oluşturulması için kullanılır.
- **SCSS:** Arayüz stillerini daha düzenli ve modüler yazmak için kullanılır.
- **RxJS:** Angular içinde asenkron veri akışlarını ve event tabanlı işlemleri yönetmek için kullanılır.

---

## Proje Yapısı

Proje, backend ve frontend olmak üzere iki ana bölümden oluşur. Backend tarafı API, veritabanı, öneri sistemi ve vector search servislerini içerirken; frontend tarafı kullanıcı arayüzü geliştirmelerine ayrılmıştır.

```text
yoladgu/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   ├── core/
│   │   ├── db/
│   │   ├── models/
│   │   ├── services/
│   │   └── main.py
│   ├── tests/
│   ├── requirements.txt
│   ├── Dockerfile
│   └── README.md
│
├── frontend/
│   ├── src/
│   ├── angular.json
│   ├── package.json
│   └── tsconfig.json
│
├── ERROR_AWARE_RECOMMENDER_README.md
├── VECTOR_INDEX_IMPROVEMENTS.md
├── backend_tasks.md
├── LICENSE
└── README.md
