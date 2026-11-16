# 2026-11-16 raporu: nano + SARD + karst zor negatifleri (R)

Plan: [PLAN.md](PLAN.md) (17:55'te, veri ve sonuç görülmeden yazıldı). Kural 2026-11-13 ile aynı.

## Sonuç: geçmedi

| Ölçü (R − K, 5'e 5 eğitim, hiyerarşik %95 GA) | K | R | Fark | Kural | Durum |
|---|---:|---:|---:|---|---|
| A: HERIDAL test recall, 0,3 FA/karede | 77,0% | 78,6% | +1,6 [−2,1, +5,3] | alt sınır > 0 | geçmedi |
| B: NOMAD FA/kare, bütçe eşiğinde | 1,020 | 0,243 | −0,777 [−0,950, −0,621] | üst sınır < 0 | **geçti** |
| Koruma: HERIDAL recall, bütçe eşiğinde | 78,4% | 67,9% | −10,5 [−18,3, −3,9] | alt sınır ≥ −2 puan | **başarısız** |
| Koruma: WiSARD top-2 kare isabeti | 88,8% | 90,8% | +2,0 | −3 puandan kötü değil | tamam |
| Koruma: HERIDAL FA/kare, bütçe eşiğinde | 0,335 | 0,134 | −0,201 | +0,05'ten fazla artmamalı | tamam |

B geçti ama koruma başarısız olduğu için karar **geçmedi**. Tablo, 2026-11-13'teki S (sadece SARD) ile
neredeyse aynı: NOMAD'da yanlış alarm dörtte bire iniyor, ama doğrulama yerlerinde yanlış alarm çok olduğu
için bütçe eşiği yükseliyor (R: 0,36-0,44, K: 0,23-0,29) ve HERIDAL test recall düşüyor.

## Zor negatifler S'ye bir şey katmadı (betimleyici, R − S)

| Ölçü | S | R | Fark [%95 GA] |
|---|---:|---:|---:|
| HERIDAL recall, bütçe eşiğinde | 69,7% | 67,9% | −1,8 [−8,6, +5,7] |
| NOMAD FA/kare, bütçe eşiğinde | 0,257 | 0,243 | −0,014 [−0,094, +0,079] |
| HERIDAL recall, 0,3 FA/karede | 80,4% | 78,6% | −1,8 [−5,3, +1,4] |
| WiSARD top-2 kare isabeti | 90,0% | 90,8% | +0,8 |

## Beyaz kaya sorunu çözülmedi

Doğrulamadaki 4.000 boş karede, 0,3 eşiğinde kare başına tespit (5 eğitimin ortalaması):

| Kol | Boş kare başına tespit | Eğitimler |
|---|---:|---|
| K (HERIDAL) | 0,40 | 0,36 0,49 0,38 0,47 0,31 |
| S (+ SARD) | 1,46 | 1,21 1,86 0,87 1,47 1,90 |
| R (+ SARD + 1.920 zor negatif) | 1,32 | 0,90 1,20 1,20 1,42 1,90 |

Eğitim fotoğraflarından çıkarılan zor negatifler bu sorunu yalnızca %10 azalttı. Doğrulama yerlerindeki
karst çayırı ve beyaz kayalar eğitim fotoğraflarında yeterince temsil edilmiyor görünüyor; yerlere göre
ayrılmış bölme tam olarak bu tür zemin farkını yakalamak için kuruldu.

## Çekinceler

- **R, K'den daha az eğitildi.** Kullanıcı CPU ısınmasını bildirince eğitim 4 çekirdeğe ve en düşük
  önceliğe sabitlendi. Epoch başına 445 adım: R1 36 epoch (~16,0 bin adım), R2 51 (~22,7 bin),
  R3 37 (~16,5 bin), R4 29 (~12,9 bin), R5 30 (~13,4 bin). Ortalama K'nin (~27 bin) yaklaşık %60'ı;
  bu R'nin aleyhine. Ama en çok adım alan R2 de bütçe eşiğinde 71,8% recall'a düştü ve boş karede 1,20
  tespit verdi; ayrıca S (~32 bin adım) da aynı yerde takıldı. Adım farkı sonucu açıklamıyor.
- Zor negatifler bir SARD modeliyle (S3b) seçildi. 1.920 karenin birinde etiketsiz bir insan olabilir
  (gözle bakıldı). Etiket gürültüsü küçük.
- Ölçümleri başlatan izleyici CPU kısıtlaması sırasında durmuştu. 21:37'de yeniden başlatıldı, ölçümler
  21:42'de bitti. Sonuçları etkilemiyor.

## Ne anlama geliyor

Durdurma kuralının 1. adımı geçmedi. Kalan tek veri denemesi: OpenAerialMap (CC BY 4.0) etiketsiz
havadan görüntüyle ön eğitim. O da geçmezse nano dondurulur ve iş sisteme kayar (iki aşamalı tarama,
P4, saha testi). Bu karar sahibinindir.

Dosyalar: `analysis/arms_hier_RvK.json`, `analysis/arms_hier_RvS.json`, `analysis/topk_RvK.json`,
`analysis/topk_RvK.txt`, `status.txt`, `train_R*.log`, `qat_R*_gpu.log`.
