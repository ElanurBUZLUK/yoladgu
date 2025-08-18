# Matematik Soru Seçimi — Bağımsız Sistem Kanvası (v1)

> Bu kanvas, **yalnızca matematik sorularının** öğrenciye uygun şekilde seçimi için tasarlanmış **bağımsız** bir strateji ve uygulama kılavuzudur. Konu/ünite dağılımı **kapsam dışıdır**; odak, *zorluk uyumu*, *tekrar planlama* ve *öğrenci akışı (flow)* üzerinedir.

---

## 1) Amaç ve İlkeler

- **Amaç:** Her istekte, öğrenciye ne çok kolay ne de çok zor, *tam akış bölgesinde* bir soru sunmak; öğrenmeyi hızlandırırken morali korumak.
- **İlkeler:**
  - *Basit çekirdek, güçlü sinyaller:* Zorluk uyumu + tekrar + keşif (bandit) üçlüsü.
  - *Hızlı çevrim:* Seçim süresi < **100 ms** (p95), her yanıt sonrası anlık profil güncelleme.
  - *Geri kazanım:* Tıkanma anında kolaylaştırılmış “kurtarma” adımı.
  - *Unutma eğrisi:* SM‑2 Lite ile aralıklı tekrar.
  - *Konudan bağımsızlık:* Seçim yalnızca zorluk, performans ve tazelik sinyalleriyle yapılır.

---

## 2) Minimal Veri Modeli

**Question**

- `id: str`
- `estimated_difficulty: float`  (sürekli ölçek, örn. 0.0–5.0)
- `freshness_score: float`  (0–1; yeni/az görülmüş sorulara +)
- `last_seen_at: datetime | None`
- `quality_flags: dict`  (örn. `{ambiguous: False, reviewed: True}`)

**StudentProfile**

- `global_skill: float` (öğrenci yetenek kestirimi)
- `difficulty_factor: float` (dinamik çarpan; 0.1–1.5 aralığı)
- `ema_accuracy: float` (üstel hareketli ortalama)
- `ema_speed: float` (normalize edilmiş hız skoru)
- `streak_right: int`, `streak_wrong: int`
- `last_k_outcomes: deque[bool]`
- `srs_queue: list[QuestionRef]` (SM‑2 Lite planı)
- `bandit_arms: dict[delta→(alpha,beta)]`  (örn. `{-1.0,-0.5,0,+0.5,+1.0}`)

> Bellek bütçesi hedefi: **≤10 KB** / öğrenci (float32 + bit‑packing).

---

## 3) Seçici (Selector) Durum Makinesi

1. **Kurtarma Modu** *(Recovery)* — tetikleyiciler:
   - `streak_wrong ≥ 2` **veya** son 5’de doğruluk < 0.4 **veya** p(doğru) < 0.35 **veya** aşırı yavaş yanıtlar.
   - **Aksiyon:** Hedef zorluk, `global_skill − γ` (örn. γ=0.6). Kısa çözüm ipucu opsiyonel.
2. **Tekrar Modu** *(SRS)* — tetikleyiciler:
   - `srs_queue` içinde **vadesi gelmiş** en az bir kart.
   - **Aksiyon:** Öncelikle **1 adet** SRS kartı sun; sonra normale dön.
3. **Normal Akış** *(Adaptive + Bandit)* — stabil durum:
   - Hedef zorluk aralığı hesapla → adayları filtrele → puanla → seç.

> Öncelik sırası: **Kurtarma > SRS > Normal**.

---

## 4) Çekirdek Algoritmalar

### 4.1 Hedef Zorluk Hesabı (Adaptive Difficulty)

- Taban değer: `base = global_skill × difficulty_factor`.
- Son 5 doğruluk > 0.85 ise aralığı **yukarı** kaydır (örn. +0.7…+1.3), aksi halde **stabil** aralık (−0.3…+0.5).
- Zaman sinyali: `time_ratio = time_taken / expected_time` → çok hızlıysa mikro artış.

**Güncelleme (her yanıt sonrası):**

```python
# correct ∈ {0,1}; time_ratio ~ 0..2
delta = 0.02 * (1 if correct else -1)
delta += 0.01 * (1 - min(time_ratio, 2.0))  # hızlıysa +
profile.difficulty_factor = clip(profile.difficulty_factor + delta, 0.1, 1.5)
```

### 4.2 Thompson Sampling (Konusuz Keşif)

- Kollar: zorluk farkı deltalari `Δ ∈ {-1.0,-0.5,0,+0.5,+1.0}` (hedef zorluğa ofset).
- Her Δ için `Beta(α,β)`; doğru → α++, yanlış → β++.
- Seçimde her Δ için örnekle → **en yüksek örnek** kazansın.

### 4.3 SM‑2 Lite (Aralıklı Tekrar)

- Basitleştirilmiş aralıklar: `[0, 1, 3, 7, 16]` gün.
- Kart başarıyla çözülürse seviye +1, zorlanırsa sabit kalır / geriler.
- Günlük görev: vadesi gelen en fazla **1** kartı normal akıştan **önce** göster.

### 4.4 Skorlama ve Çeşitlilik

- Aday puanı = `|target_difficulty − q.estimated_difficulty|` (daha küçük daha iyi)
  - `λ * (1 − freshness_score)`  (λ≈0.2)
  - `ε` küçük rastgelelik.
- **Yineleme koruması:** Son N’de gösterilen `id`’leri dışla (örn. N=10).

---

## 5) Seçim Akışı — Pseudocode

```python
class MathSelector:
    def select(self, profile, question_pool):
        if self._needs_recovery(profile):
            return self._pick_recovery(profile, question_pool)
        if self._has_due_srs(profile):
            return self._pick_srs(profile)
        target_low, target_high = self._target_range(profile)
        cands = self._filter(question_pool, target_low, target_high)
        if profile.total_interactions < 50:
            return self._epsilon_greedy(cands)
        delta = self._thompson_arm(profile)  # {-1.0..+1.0}
        adj_range = (target_low+delta, target_high+delta)
        cands2 = self._filter(question_pool, *adj_range)
        return self._score_and_pick(cands2 or cands, profile)
```

---

## 6) Yanıt Sonrası Güncellemeler

**Doğruluk & Hız (EMA):**

```python
profile.ema_accuracy = 0.9*profile.ema_accuracy + 0.1*correct
profile.ema_speed    = 0.8*profile.ema_speed    + 0.2*speed_score
```

**Yetenek (global\_skill):**

```python
# Basit ve hızlı: hedefe göre küçük adım
err = (q.estimated_difficulty - profile.global_skill)
profile.global_skill += 0.1 * (1 if correct else -1) * sigmoid(err)
```

**SRS:** başarı → seviye↑, tekrar aralığı güncelle; başarısız → erken tekrar.

**Bandit:** seçilen Δ için α/β’yi güncelle.

**Streak:** doğru/yanlış serileri güncelle; kurtarma tetiklerini kontrol et.

---

## 7) Guardrail’ler ve Kenar Durumları

- **Zorluk sınırları:** `estimated_difficulty ∈ [min_d, max_d]` zorlaması.
- **Burnout koruması:** 3 yanlış üst üste → zorluğu kademeli düşür, kısa ipucu aç.
- **Duplicate/Leakage:** Son N’deki soruları at; aynı kalıbın varyantlarına cooldown.
- **Timeout/Guess:** Çok hızlı ve yanlış → tahmin; bir sonraki soru kolaylaştır.
- **Kısmi puan:** Kısmi puanı `correct` ağırlığına çevir (`0.0–1.0`).

---

## 8) API Taslağı (opsiyonel)

- **POST** `/math/next` → `{question_id, target_difficulty, rationale}`
- **POST** `/math/submit` → `{question_id, correct, time_taken, partial_credit}`
- **GET** `/math/profile` → `{global_skill, ema_accuracy, due_srs}`

> Token, oran sınırlama ve günlükleme bu kanvas kapsamı dışı; entegre edilebilir.

---

## 9) Parametreler (Önerilen Varsayılanlar)

| Parametre           | Varsayılan | Açıklama                 |
| ------------------- | ---------- | ------------------------ |
| `window_size`       | 20         | Son etkileşim penceresi  |
| `λ` (freshness)     | 0.2        | Tazelik ağırlığı         |
| `ε` (random)        | 0.01       | Rastgelelik genliği      |
| `γ` (recovery drop) | 0.6        | Kurtarmada zorluk düşümü |
| `EMA_acc`           | 0.9        | Doğruluk EMA katsayısı   |
| `EMA_speed`         | 0.8        | Hız EMA katsayısı        |
| `bandit_deltas`     | −1.0..+1.0 | Zorluk ofset kolları     |

---

## 10) Ölçüm ve Başarı Kriterleri

- **Zorluk Uyumu:** hedef vs sunulan zorluk yakınlığı → hedef **%65–75**.
- **Kurtarma Başarısı:** kurtarma sonrası ilk soruda doğruluk **>%80**.
- **Öğrenme Hızı:** haftalık doğru sayısında **+%20** artış.
- **Tekrar Etkinliği:** SRS kartlarının doğru oranı **>%70**.
- **Seçim Gecikmesi:** p95 **<100 ms**.

---

## 11) Offline Simülasyon & A/B Planı

**Simülasyon:**

- Öğrenci zemini `θ*` ve her soruya `p(correct)=σ(a(θ*−b))` modeli.
- 1k adımda adaptasyon süresi, doğru oranı ve *flow* metriği ölçülür.

**A/B parametreleri:**

- `recovery γ ∈ {0.4, 0.6, 0.8}`
- `bandit_deltas ∈ {±0.5, ±1.0}`
- `λ ∈ {0.1, 0.2, 0.3}`

---

## 12) Test Stratejisi (Kısa)

- **Birim:** hedef aralık hesabı, bandit güncelleme, SRS planlayıcı.
- **Sahte Havuz:** 200 soruluk sentetik havuzla seçim deterministikliği.
- **Regresyon:** Düz hatalı seçim paternleri için fixture’lar.

---

## 13) Uygulama İskele Kodları

**Seçici:**

```python
import random, math

def clip(x,a,b):
    return max(a, min(b, x))

def sigmoid(x):
    return 1/(1+math.exp(-x))

class Selector:
    def __init__(self, cfg):
        self.cfg = cfg

    def target_range(self, p):
        base = p.global_skill * p.difficulty_factor
        if p.last5_acc > 0.85:
            return (base+0.7, base+1.3)
        return (base-0.3, base+0.5)

    def thompson_arm(self, arms):
        best = None; best_s = -1
        for d,(a,b) in arms.items():
            s = random.betavariate(a,b)
            if s>best_s: best_s, best = s, d
        return best

    def score(self, q, target, lam):
        dist = abs(q.estimated_difficulty - target)
        freshness = 1 - getattr(q, 'freshness_score', 0)
        return dist + lam*freshness + random.random()*0.01
```

**Güncelleme:**

```python
def update_after_answer(profile, q, correct, time_ratio, delta_used):
    profile.ema_accuracy = 0.9*profile.ema_accuracy + 0.1*(1 if correct else 0)
    profile.ema_speed    = 0.8*profile.ema_speed    + 0.2*(1 - min(time_ratio,2))
    err = (q.estimated_difficulty - profile.global_skill)
    profile.global_skill += 0.1 * (1 if correct else -1) * (1/(1+math.exp(-err)))
    # bandit
    a,b = profile.bandit_arms[delta_used]
    profile.bandit_arms[delta_used] = (a + (1 if correct else 0), b + (0 if correct else 1))
    # difficulty factor
    df_delta = 0.02*(1 if correct else -1) + 0.01*(1 - min(time_ratio,2))
    profile.difficulty_factor = clip(profile.difficulty_factor + df_delta, 0.1, 1.5)
```

---

## 14) Yayına Alma Yolu (MVP → V1)

- **MVP:** Normal akış (adaptive + ε‑greedy), basit SRS, temel guardrail’ler.
- **V0.9:** Thompson Sampling, kurtarma modu, telemetri paneli.
- **V1.0:** A/B parametreleştirme, otomatik hiperparametre ayarı, offline simülatör.

---

## 15) Notlar

- Bu kanvas, dış sistemlere bağımlı olmadan uygulanabilir. DB, kimlik doğrulama ve RAG benzeri katmanlar opsiyoneldir.
- Kod örnekleri iskele amaçlıdır; tipler/IO ve kalıcılık katmanı ekip tercihine göre şekillenir.

