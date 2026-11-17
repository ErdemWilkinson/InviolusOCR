# Gün 2026-11-17: iki aşama, nano → VoxScout (C)

Plan: `PLAN.md` (sayılardan önce yazıldı). Eğitim yok: önbellekteki ısı haritalarından kuruldu
(`voxocr/scripts/two_stage.py`). Nano (K1L..K5) her kareyi tarar; en yüksek puanı τ'nun üstündeki kareleri
VoxScout yeniden kontrol eder, tespitler yalnızca VoxScout'tan gelir. τ yalnızca doğrulamada seçildi:
geçen kareler doğrulamadaki insanların ≥ %97'sini içeriyor (τ = 0,13-0,17).

## Sonuç: kural gereği kaldı (bir koruma), ama eşit noktalarda çok daha iyi

| C − K | K | C | Fark [95% GA] | Koşul | Durum |
|---|---:|---:|---|---|---|
| A: HERIDAL recall, 0,3 FA/karede | 77,0% | 87,4% | +10,4 [+6,5, +14,9] | alt > 0 | **geçti** |
| B: NOMAD FA/kare, bütçe eşiği | 1,020 | 0,986 | −0,034 [−0,169, +0,096] | üst < 0 | kaldı |
| Koruma: HERIDAL recall, bütçe eşiği | 78,4% | 89,4% | +11,0 [+6,8, +15,3] | alt ≥ −2 | geçti |
| Koruma: WiSARD top-2 kare isabeti | 88,8% | 100% | +11,2 | ≥ −3 | geçti |
| Koruma: HERIDAL FA/kare, bütçe eşiği | 0,335 | 0,674 | +0,340 | ≤ +0,05 | **kaldı** |

A geçti, ama HERIDAL FA koruması kaldı, bu yüzden önceden yazılan kurala göre sonuç **başarısız**.

**Neden kaldı:** bütçe kuralı her modele doğrulamada aynı yanlış alarm sayısını (0,61/kare) harcatıyor.
VoxScout o kadar az alarm veriyor ki bütçeyi doldurmak için eşik ölçeğin en altına (0,04) iniyor; bu bütçeyi
tamamen recall'a çeviriyor (+11 puan). O düşük eşikte doğrulamadan teste aktarım da tutmuyor (test FA 0,67,
bütçenin üstünde). Kural benzer modeller için kurulmuştu; bunu sonuç görüldükten sonra değiştirmiyorum.

**Aynı recall'da (doğrulamada eşleşen nokta, tanımlayıcı):**

| | K | C |
|---|---:|---:|
| HERIDAL recall | 75,9% | 79,2% (+3,3 [−1,0, +8,0]) |
| HERIDAL yanlış alarm/kare | 0,243 | **0,039** (−%84) |
| NOMAD yanlış alarm/kare | 0,792 | **0,141** (−%82) |
| WiSARD recall | 57,5% | 54,5% |
| WiSARD yanlış alarm/kare | 0,264 | 0,030 |

Yanlış alarmlar 6 kat azalıyor, HERIDAL recall düşmüyor. WiSARD'da recall ~3 puan düşük (VoxScout küçük
insanlarda zayıf, 2026-11-12'de görülmüştü). Kapı (nano) neredeyse hiçbir şey kaybettirmiyor: top-k
sonuçları VoxScout'un tek başına tüm karelere bakmasıyla aynı (test top-2 insan 54,3% / 54,3%).

## Maliyet: kapı kareleri yeterince elemiyor

%97 kapsama için nano karelerin büyük kısmını geçiriyor: test %44-59, WiSARD %49-78, NOMAD %49-61.
VoxScout P4'te ölçülmedi (int8 5,2 MB; dizüstü CPU'da ~0,4 sn/kare). Nano 151 ms/kare. Yani cihazda kare
başına süre kabaca nano + yarısı kadar VoxScout; VoxScout P4'te birkaç saniye sürerse tarama çok yavaşlar.

## Çekinceler

- Beş C koşusu tek bir VoxScout modelini paylaşıyor; onun koşudan koşuya değişkenliği aralıkta yok.
- VoxScout sayıları float; int8 doğruluğu ve P4 süresi/RAM'i ölçülmedi.
- Bu bir VoxScout projesi hattı olur (omurga kuralı); VoxHOD nano dosyası değişmiyor.

## Sonraki adım (sahibin kararı)

1. Kuralı önceden yazıp değiştirmek: bütçe yerine "aynı doğrulama recall'u" kuralı. Ancak bu sonuç görüldükten
   sonra seçilmiş olur; onay için yeni veri ya da yeni VoxScout eğitimleri gerekir.
2. Maliyeti ölçmek: VoxScout int8'i P4'te çalıştırmak (süre, RAM) ve int8 doğruluğunu test etmek (CPU'da ~1 saat).

## Dosyalar

`analysis/arms_hier_CvK.json`/`.txt`, `analysis/topk_CvK.json`/`.txt`, `analysis/two_stage_T0.json` (τ ve
geçen kare oranları).

## Ek (akşam): cihazdaki dosya, VoxScout int8 (C8) — kapıyı geçti, çekinceli

Sahip: "devam et otopilot". Cihazda float değil int8 çalışır. Plan eki 18:15 (int8 sayısından önce yazıldı):
yayımlanan int8 dosyası (5,2 MB) dört bölümün hepsinde çalıştırıldı (4 iş parçacığı, 18:07-19:28), C8_1..C8_5
aynı kapı ve aynı kuralla kuruldu.

| C8 − K | K | C8 | Fark [95% GA] | Koşul | Durum |
|---|---:|---:|---|---|---|
| A: HERIDAL recall, 0,3 FA/karede | 77,0% | 83,1% | +6,1 [+1,6, +10,5] | alt > 0 | **geçti** |
| B: NOMAD FA/kare, bütçe eşiği | 1,020 | 0,844 | −0,175 [−0,316, −0,046] | üst < 0 | **geçti** |
| Koruma: HERIDAL recall, bütçe eşiği | 78,4% | 84,6% | +6,2 [+2,2, +10,6] | alt ≥ −2 | geçti |
| Koruma: WiSARD top-2 kare isabeti | 88,8% | 100% | +11,2 | ≥ −3 | geçti |
| Koruma: HERIDAL FA/kare, bütçe eşiği | 0,335 | 0,334 | −0,001 | ≤ +0,05 | geçti |

Aynı recall'da (tanımlayıcı): HERIDAL FA 0,243 → 0,056, NOMAD FA 0,792 → 0,223 (yaklaşık 4 kat az).

**int8 kaybı (VoxScout int8 − float, tek başına):** 0,3 FA'da recall −4,0 puan [−6,7, −1,5]; aynı recall'da
NOMAD FA 0,139 → 0,232. int8 belirgin biçimde daha kötü, ama hâlâ nano'nun çok önünde.

**Çekinceler (önemli):**
- **Aynı test setine ikinci bakış.** Float C kaldı, int8 C8 geçti. int8 adımı sayılardan önce yazıldı, ama
  geçmesinin bir nedeni eşik kalibrasyonu: int8'in doğrulama FA'sı biraz daha yüksek olduğu için bütçe eşiği
  0,04 yerine 0,06'ya düştü ve testte bütçe aşılmadı. Yani sonuç eşik seçimine duyarlı; tek başına kanıt sayılmaz.
- **WiSARD recall düşüyor.** Kural top-2 kare isabetine bakıyor (100%), ama doğrulamada eşleşen eşikte WiSARD
  insan recall'u K 57,5% → C8 50,7% (−6,8 puan). VoxScout küçük insanlarda zayıf. Operatör kare kare bakıyorsa
  sorun değil; tek tek insan sayılıyorsa sorun.
- Beş koşu tek VoxScout modelini paylaşıyor; aralık onun değişkenliğini içermiyor.
- **Cihaz maliyeti ölçülmedi:** nano karelerin %44-78'ini geçiriyor; VoxScout'un P4'teki süresi ve RAM'i
  bilinmiyor (dizüstü CPU 4 iş parçacığında 0,30 sn/kare).

**Karar:** plana göre geçiş yayımlamak değil, bir sonraki ölçümü haklı çıkarır: VoxScout int8'i P4'te çalıştırmak
(süre, RAM; kart bağlı olmalı). Hat bir VoxScout projesi ürünü olur; VoxHOD nano dosyası değişmez.

Dosyalar: `analysis/arms_hier_C8vK.*`, `analysis/arms_hier_VS8vT0.*`, `analysis/topk_C8vK.*`,
`analysis/two_stage_VS8.json`, `analysis/sweep_VS8.log`, `status.txt`.
