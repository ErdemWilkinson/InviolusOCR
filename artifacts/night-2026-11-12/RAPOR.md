# Gece raporu: 2026-11-12 (VoxHOD, önceden eğitilmiş öğretmenden damıtma)

Plan ve karar kuralı sonuçlardan önce yazıldı: [PLAN.md](PLAN.md) (03:15 eki dahil: 5 yerine 8 tohum, tek karar).
Analiz dosyaları: [analysis/](analysis/). **Hiçbir model yayınlanmadı.**

## Kısa sonuç

1. **Damıtma geçmedi.** Ana ölçüt (HERIDAL test recall, 0,3 yanlış alarm/karo): D − K = **−0,1 puan [−2,9, +2,4]**. Alt sınır 0'ın üstünde değil.
   Ayrıca iki koruma ölçütü de aşıldı: NOMAD yanlış alarmı +0,070/karo (sınır +0,05) ve WiSARD'da top-2 kare isabeti −3,1 puan (sınır −3).
   8'e 8 tohumla hiçbir ölçüde D lehine bir eğilim yok.
2. **Öğretmen gerçekten çok daha iyi, sadece val'de değil testte de.** Aynı val recall hedefinde (0,781) T0'ın yanlış alarmı şöyle:
   - HERIDAL testinde **0,042/karo**, nanolarda ~0,25.
   - NOMAD'da **0,14**, nanolarda ~0,9.
   Yani bilgi var, ama nano modele aktarılamadı.
3. **Neden (hipotez, test edilmedi):** Kapasite. D öğrencilerinde eğitim kaybı ~1,97, K'lerde ~0,67. Damıtma terimi ~1,3'te kalıyor, yani küçülmüyor. Öğrenci öğretmenin haritasını taklit edemiyor.
   2026-11-10'daki "nano yanlış alarmı kapasite sınırında" bulgusuyla tutarlı.
4. **Top-K rolünde de fark yok.** HERIDAL top-2 fotoğraf isabeti K 90,5%, D 90,1%. WiSARD'da K 90,9%, D 87,8%.

## 1. Karar tablosu (önceden yazılmış kural)

`arms_hier.py`, `ARMS_PAIRED=1`, tohum çiftleri (K1L–D1 … K8–D8) ve ardından test fotoğrafları/NOMAD kareleri yeniden örneklendi (2000 kez). Eşikler val'de seçildi: val yanlış alarm bütçesi 0,610/karo, val hedef recall 0,781 (referans GA1 0,20'de).

| D − K (8'er çalıştırma) | K | D | Hiyerarşik %95 | Kural | Sonuç |
|---|---:|---:|---:|---|---|
| **HERIDAL recall @ 0,3 FA/karo (eğri)** | 77,6% | 77,5% | −2,9 / +2,4 puan | alt > 0 | **geçmedi** |
| HERIDAL recall, bütçe eşiği | 79,0% | 79,4% | −1,9 / +2,9 puan | alt ≥ −2 | geçti |
| NOMAD yanlış alarm/karo, bütçe eşiği | 1,047 | 1,117 | −0,09 / +0,22 | nokta ≤ +0,05 | **geçmedi** (+0,070) |
| WiSARD top-2 kare isabeti (tek klip, tanımlayıcı) | 90,9% | 87,8% | — | ≥ −3 puan | **geçmedi** (−3,1) |
| HERIDAL yanlış alarm/karo, bütçe eşiği | 0,335 | 0,371 | −0,01 / +0,08 | (bilgi) | |
| HERIDAL recall @ 0,2 / 0,5 FA/karo | 74,7 / 81,6% | 74,3 / 81,2% | −3,8/+2,4 ; −2,4/+1,8 | (bilgi) | |

Çalıştırma bazında tablo: `analysis/arms_hier_DvK.json`. Top-K: `analysis/topk_DvK.json`.

## 2. Öğretmenin kendi test sonuçları (tanımlayıcı, karar sonrası)

`threshold_sweep.py`, T0 son epok (float, Keras, 320 px), val recall hedefi 0,781'de seçilen eşik 0,34. Dosya: `analysis/sweep_T0.json`.

| | GA1 (nano ref) | K6 | D6 | **T0** |
|---|---:|---:|---:|---:|
| HERIDAL test recall | 80,1% | 80,1% | 76,9% | 79,8% |
| HERIDAL test yanlış alarm/karo | 0,569 | 0,234 | 0,279 | **0,042** |
| HERIDAL test precision | 13% | 27% | 23% | **67%** |
| NOMAD yanlış alarm/karo | 1,56 | 0,96 | 0,85 | **0,14** |
| WiSARD recall | 65,4% | 63,5% | 59,6% | 53,6% |
| WiSARD yanlış alarm/karo | 0,84 | 0,37 | 0,31 | **0,03** |

Öğretmen WiSARD'da küçük insanları daha az yakalıyor (küçük kişi recall 23%, nanolarda ~30%). Bunun nedeni muhtemelen 160 px karoyu 320'ye büyütüp görmesi.

## 3. Çalıştırma notları

- 13 eğitimin hepsi hatasız bitti (30 dk eğitim + 10 dk QAT, GPU'da ikişerli).
- **Kesinti:** 04:14'te WSL, Claude oturumuyla birlikte kapandı. K4 ve K5 epok ~87'de kesildi ve baştan yeniden çalıştırıldı.
  Kuyruk o andan sonra Windows'tan bağımsız süreç olarak başlatıldı (`Start-Process wsl.exe`).
- **Eşitsizlik:** Süre bütçesi dakika cinsinden, bu yüzden epok sayısı GPU paylaşımına bağlı.
  - D'ler 103–121 epok aldı (damıtma biraz yavaş), K4–K7 108–111 epok.
  - K8 son 40 dakikada GPU'da tek başına kaldığı için 172 epok aldı.
  - K8 K kolunun en iyisi değil, sonucu değiştirmiyor.
- Öğrenciler hâlâ %55'lik bölmede (voxhod_g) eğitildi. NOMAD eğitimde yok.

## 4. Öneri (karar sende)

- **Bu tarifle damıtmayı bırakmak.** Nano, öğretmenin haritasını taklit edemiyor; daha fazla tohum ya da süre bunu değiştirmez.
- Öğretmenin yanlış alarmı 6–7 kat düşük olduğu için asıl soru şu: **hedef cihaza nano değil, daha büyük ama hâlâ P4'e sığan bir model** mi koymalı?
  "Dağıtılan modelin boyutu büyümesin" kuralı geçerli olduğu için bu senin kararın.
  Ara seçenekler:
  - (a) Arada bir boyut: v3 boyutu ya da genişlik 1,0. Ayrıca önceden eğitilmiş bir gövde.
  - (b) İki aşama: nano tüm karoları sıralar, P4 yalnızca ilk 2 karoda büyük modeli çalıştırır. Top-K çalışmasıyla uyumlu; 2 karo/kare için gecikme ölçülmeli.
- Denenmemiş damıtma varyantları da var: ara özellik damıtması, daha yüksek çözünürlüklü öğrenci, uzun eğitim. Ancak 3. maddedeki kapasite işareti bunlardan büyük kazanç beklememeyi söylüyor.
