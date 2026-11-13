# Gece raporu: 2026-11-13 (VoxHOD nano + SARD verisi; VoxScout int8 ölçümü)

Plan ve karar kuralı sonuçlardan önce yazıldı: [PLAN.md](PLAN.md) (06:15 eki dahil). Analiz: [analysis/](analysis/).
**Hiçbir VoxHOD modeli yayınlanmadı.** VoxScout'un README'si ölçülen int8 sayılarıyla güncellendi.

## Kısa sonuç

1. **Kural geçmedi, ama bugüne kadarki en güçlü sinyal bu.** SARD (CC BY 4.0, 1.586 eğitim karesi) eklenen nano (S, 5 tohum) ile eklenmeyen nano (K, 5 tohum) karşılaştırıldı:
   - **B (görmediği arazi) geçti:** Bütçe eşiğinde NOMAD yanlış alarmı 1,02'den **0,26**'ya iniyor, fark −0,76 [−0,91, −0,63].
   - **A (kişi bulma) kıl payı geçmedi:** 0,3 FA/karo'da HERIDAL yakalama oranı **+3,4 puan** [−0,4, +7,0].
   - **Bir koruma ölçütü bozuldu, sonuç bu yüzden "geçmedi":** Bütçe eşiğinde HERIDAL yakalama oranı −8,7 puan [−14,4, −3,2]. Nedeni muhtemelen şu: S'ler val bölmesinde (HERIDAL'in ayrılmış 5 konumu) çok yanlış alarm veriyor. Bu yüzden val bütçesi onları çok yüksek eşiklere (0,42–0,47) itiyor ve testte yakalama oranı düşüyor.
   - **Recall eşleştirilmiş noktada (tanımlayıcı):** HERIDAL yakalama oranı aynı (+0,6 puan), HERIDAL yanlış alarmı −0,04, NOMAD yanlış alarmı **yarıya iniyor** (0,79'dan 0,39'a, [−0,53, −0,26]). WiSARD top-2 isabeti +1,2 puan.
2. **VoxScout int8 ölçüldü.** int8, float'a göre yanlış alarmı yaklaşık iki katına çıkarıyor (%78 yakalamada 0,036'dan 0,072'ye), ama yine de nanodan 4–5 kat iyi. Önerilen int8 eşiği 0,23. VoxScout README'si ve HF kartı güncellendi.

## 1. Karar tablosu (önceden yazılmış kural)

`arms_hier.py`, `ARMS_PAIRED=1`, K1L–S1b … K5–S5b tohuma göre eşleştirildi, 2000 çekiliş. Eşik val'de: FA bütçesi 0,610/karo; recall hedefi 0,781.

| S − K (5'er çalıştırma) | K | S | Hiyerarşik %95 | Kural | Sonuç |
|---|---:|---:|---:|---|---|
| **A: HERIDAL recall @ 0,3 FA/karo** | 77,0% | 80,4% | −0,4 / +7,0 puan | alt > 0 | geçmedi (kıl payı) |
| **B: NOMAD FA/karo, bütçe eşiği** | 1,020 | 0,257 | −0,910 / −0,628 | üst < 0 | **geçti** |
| Koruma: HERIDAL recall, bütçe eşiği | 78,4% | 69,7% | −14,4 / −3,2 puan | alt ≥ −2 | **bozuldu** |
| Koruma: HERIDAL FA/karo, bütçe eşiği | 0,335 | 0,123 | −0,275 / −0,145 | nokta ≤ +0,05 | geçti |
| Koruma: WiSARD top-2 kare isabeti | 88,8% | 90,0% | — | ≥ −3 puan | geçti |
| (bilgi) Recall kuralı: HERIDAL recall / FA | 75,9% / 0,243 | 76,5% / 0,201 | −3,0/+3,9 ; −0,11/+0,03 | | |
| (bilgi) Recall kuralı: NOMAD FA/karo | 0,792 | 0,394 | −0,530 / −0,259 | | |
| (bilgi) HERIDAL recall @ 0,2 / 0,5 FA/karo | 74,0 / 81,0% | 76,2 / 84,0% | −1,6/+6,3 ; −0,2/+6,4 | | |

SARD'ın ayrılmış test bloğu (son %20 kare, aynı video, tanımlayıcı): 0,3 FA/karo'da yakalama oranı K %57,7, S %85,1.

## 2. Çalıştırma notları

- **İlk deneme geçersiz:** S1–S5 ikişer ikişer çalıştı. WSL belleği yetmedi (12.307 karo, muhtemelen takas), S1–S4 30 dakikada yalnızca 21–25 epok alabildi; K'ler ~110 epok almıştı.
- **06:15 eki** (sonuçlara bakmadan önce): S1b–S5b tek tek ve 18 dk float eğitimle yeniden çalıştırıldı. Bu süre K'lerin adım sayısına (~27 bin) eşitlemek için seçildi. Gerçekleşen: 80–85 epok × 385 adım ≈ 32 bin adım, yani K'lerden ~1,2 kat fazla. S'ler bu yüzden biraz avantajlı.
- K kolu (K1L–K3L, K4, K5) başka gecelerde ikişer ikişer eğitilmişti.
- SARD karoları `scripts/prepare_sard.py` ile üretildi: HERIDAL eğitim karolarıyla aynı kişi boyu dağılımı (medyan 20 px'e karşı 22 px), kare başına 2 kişi karosu + 1 rastgele karo, kare numarasına göre %80/%20 bölme.
- VoxScout int8 ölçümü: `sweep_voxscout_int8.json`.

## 3. Öneri (karar sende)

- Kural geçmediği için yayın yok. Ama sinyal güçlü: görmediği arazide yanlış alarm yarıya ya da çeyreğe iniyor, HERIDAL'de kişi kaybı yok gibi.
- Kuralı bozan şey veri değil, eşik seçimi: S'ler val'de çok alarm veriyor. Sıradaki deney için öneri:
  - (a) val bölmesindeki bu alarmlara bakmak (hangi arazi, neye benziyor);
  - (b) tam HERIDAL + SARD ile v3 ve nanoyu yeniden eğitip yayındaki modellerle aynı kuralla karşılaştırmak.
  Ama bunun kuralı da önceden yazılmalı.

## 4. Ek (08:50): S'ler val'de neye alarm veriyor? (tanımlayıcı, karardan sonra)

Val'in boş karolarında, eşik 0,3'te karo başına tespit sayısı K'de **0,40**, S'de **1,46**. S'nin K'ye göre en çok fazla alarm verdiği 12 boş karo
[analysis/val_false_alarms_S.png](analysis/val_false_alarms_S.png) dosyasında (HERIDAL, CC BY 3.0; kırmızı halkalar S3b'nin tespitleri).
Hepsi **otlu karst arazide küçük beyaz taşlar**. SARD'daki insanlar çoğunlukla yeşil çimende ve açık renk giysili. Görünüşe göre
nano "küçük parlak leke = insan" kuralını öğrenmiş (hipotez). HERIDAL'in val konumları bu tür arazide, test konumları daha az.

Sıradaki deney için somut öneri: SARD'la birlikte **beyaz taşlı boş karst karolarını** (HERIDAL eğitim fotoğraflarının boş kısımlarından)
daha fazla eklemek; kuralı önceden yazıp aynı 5'e 5 düzenle tekrar etmek.
