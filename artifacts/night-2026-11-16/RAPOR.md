# Gece 2026-11-16/17: nano + Sırbistan boş drone kareleri (T), sonra ASR budama

Plan: `PLAN.md` (sonuçlardan önce yazıldı, iki ek 01:05 ve 01:48).

## Sonuç: T kapıyı geçemedi, nano donduruluyor (durdurma kuralı)

T = R'nin verisi (voxhod_g + SARD + karst zor negatifleri) + Sırbistan'dan 672 boş drone karesi.
Kontrol K = K1L, K2L, K3L, K4, K5. Önceden kayıtlı kural, `arms_hier.py`, `ARMS_PAIRED=1`.

| T − K | K | T | Fark [95% GA] | Koşul | Durum |
|---|---:|---:|---|---|---|
| A: HERIDAL recall, 0,3 FA/karede | 77,0% | 80,7% | +3,7 [+0,4, +6,3] | alt > 0 | **geçti** |
| B: NOMAD FA/kare, bütçe eşiği | 1,020 | 0,402 | −0,617 [−0,760, −0,496] | üst < 0 | **geçti** |
| Koruma: HERIDAL recall, bütçe eşiği | 78,4% | 75,2% | −3,2 [−7,1, +0,4] | alt ≥ −2 | **kaldı** |
| Koruma: WiSARD top-2 kare isabeti | 88,8% | 85,1% | −3,7 | ≥ −3 | **kaldı** |
| Koruma: HERIDAL FA/kare, bütçe eşiği | 0,335 | 0,158 | −0,176 | ≤ +0,05 | geçti |

İki ana ölçüt de geçti ama iki koruma kaldı, kural gereği sonuç **başarısız**. Durdurma kuralının son veri
adımıydı; nano (VoxHOD nano) bu haliyle donduruluyor, yayımlanan model değişmiyor.

Koşu bazında WiSARD top-2 kare isabeti: K 79,3 81,6 96,9 88,9 97,3; T 96,9 88,5 84,3 74,7 81,2. T'nin
koşular arası dağılımı geniş (T1 en iyi, T4 en kötü).

## Neden yine de önemli: R'ye göre belirgin iyileşme

| | K | R (dün) | T (bu gece) |
|---|---:|---:|---:|
| Boş doğrulama karesi başına tespit, 0,3 eşik (beyaz kayalar) | 0,40 | 1,32 | **0,67** |
| HERIDAL recall, bütçe eşiği | 78,4% | 67,9% | 75,2% |
| HERIDAL recall, 0,3 FA/karede | 77,0% | 78,6% | 80,7% |
| NOMAD FA/kare, bütçe eşiği | 1,020 | 0,243 | 0,402 |
| WiSARD top-2 kare isabeti | 88,8% | 90,8% | 85,1% |

T − R (tanımlayıcı): bütçe eşiğinde HERIDAL recall +7,3 [+0,3, +15,0]; NOMAD FA +0,160 [+0,060, +0,245].
Benzer zeminden gelen boş kareler beyaz kaya alarmlarını yarıya indirdi ve R'nin büyük recall kaybını
çoğunlukla geri getirdi. Sabit FA'da (0,2 / 0,3 / 0,5) T, K'den her noktada daha yüksek recall veriyor.
Kalan kayıp eşik seçiminden geliyor: doğrulamadaki beyaz kayalar eşiği yukarı itiyor, bu da test recall'unu
düşürüyor.

## Çekinceler

- **T, K'den fazla eğitildi.** Bütçe 30 dk (PLAN ek 01:05/01:48). Epoch başına 466 adım: T1 90 epoch
  (41,9 bin adım), T2 102 (47,5 bin), T3 107 (49,9 bin), T4 80 (37,3 bin), T5 114 (53,1 bin); ortalama
  ~46 bin = K'nin ~27 binin **1,7 katı**. Önceden yazıldığı gibi: T kaldığı için sonuç geçerli (daha çok
  eğitimle bile kaldı).
- T2-T5 eğitimi sırasında önceki koşunun taraması (4 iş parçacığı) paralel çalıştı; epoch süreleri 16-20 sn.
- Sırbistan karelerinin 764'ünden 92'si elle incelemede atıldı (insan, olası insan, içinde insan olabilecek
  araç, net olmayan nesne); 672 kaldı. Fotoğraflar etiketsiz olduğu için bu inceleme tek güvence.

## Veri ve lisans

- "Comprehensive dataset from high resolution UAV land cover mapping of diverse natural environments in
  Serbia", Zenodo 14169485, **CC BY 4.0**. 22,8 GB RAR'ın tamamı indirilmedi: sıkıştırma katı değil, HTTP
  range ile yalnızca seçilen 191 RGB fotoğraf (949 MB, 14 yer) çekildi.
- Yer seçimi model puanlarından önce gözle yapıldı (dağ merası, kayalık çayır, çalılık). Su, tarla, yerleşim
  ağırlıklı 6 yer dışarıda kaldı.
- Kareler 160/240/320 px kırpıldı (GSD HERIDAL'ın yaklaşık iki katı), 160'a küçültüldü; fotoğraf başına S3b ile
  puanlanan 40 adaydan en yüksek 3 + 1 rastgele. Araç: `voxocr/scripts/mine_external.py`.
- Model bu veriyle yayımlanmadı. Yayımlanırsa MODEL_LICENSE'a CC BY 4.0 atfı eklenmeli.

## Dosyalar

- `analysis/arms_hier_TvK.json`/`.txt` (kapı), `analysis/arms_hier_TvR.json`/`.txt` (tanımlayıcı),
  `analysis/topk_TvK.json`/`.txt` (WiSARD/test top-k).
- `train_T*.log`, `qat_T*_gpu.log`, `status.txt`.

## ASR: budama + yeni veri seti (FLEURS)

İstek: "asr ı buda + yeni veri seti seç ardından bu yeni veri seti ile onu eğit".

- **Budama:** `voice/scripts/prune_asr.py`, v11'in en önemli %75'lik conv filtreleri, GRU ve dense birimleri
  (L1 ağırlık) tutuldu: 63.955 → **38.375 parametre (%40 daha küçük)**.
- **Yeni veri:** Google FLEURS Türkçe (tr_tr), **CC BY 4.0**, 2.526 eğitim klibi (×4 tekrar). v11'in verisiyle
  (Common Voice CC0 + ISSAI MIT) birlikte 110.883 klip. MediaSpeech yayın telif hakkı nedeniyle alınmadı.
- **Eğitim:** CPU'da 04:33-08:31, 3 epoch. Epoch süresi 36 → 75 → 117 dk'ya çıktı (paylaşılan CPU).

Greedy CER, her test setinden aynı 500 klip:

| Model | Parametre | Common Voice | ISSAI | FLEURS |
|---|---:|---:|---:|---:|
| v11 (yayımlanan) | 63.955 | 45,6% | 46,3% | 43,7% |
| budanmış, eğitimden önce | 38.375 | — | — | 85,7% |
| budanmış + 3 epoch | 38.375 | 51,2% | 50,9% | 47,4% |

Beklendiği gibi (v11 zaten yetersiz öğreniyor) küçük model 4-6 puan daha kötü. Fark en az FLEURS'ta (3,7 puan),
yeni veri orada biraz işe yaramış görünüyor. Doğrulama kaybı hâlâ düşüyordu (59,7 → 57,9 → 57,5); GPU'da daha
uzun eğitimle fark kapanabilir ama %40 küçük modelin v11'i geçmesi beklenmez. **Yayımlanmadı**; v11 temel
model olarak kalıyor. Dosyalar: `voice/artifacts/asr_pruned_1117/` (repo dışı), README notu commit dc0abee.

## Commitler

- VoxHOD 92d12d4: `scripts/mine_external.py`, README T1-T5 satırı ve yol haritası (nano donduruldu).
- turkish-asr dc0abee: `scripts/prune_asr.py`, README budama deneyi, FLEURS lisans satırı.
- InviolusOCR: bu klasör.
