# Çalıştırılmış kabul sonuçları

Python 3.12.14, sentetik `scenario.json`; yerel test sayısı **8**, tümü başarılı. Ham test günlüğü `test-log.txt`.

Eğitim 142 kayıt; Brier 0.1353; maliyet 52; confusion {'tp': 3, 'tn': 36, 'fp': 2, 'fn': 10}

Sonuçlar yalnız bu örneğe aittir; üretim doğruluğu veya performans garantisi olarak yorumlanmamalıdır. Ölçülen iş sonucunu kontrol edin; örnek negatif vaka içeriyorsa ret beklenir. Tam çıktı `sample-report.json`.

## Sınanan davranışlar

| Test | Kontrol |
|---|---|
| `test_training_cutoff` | training cutoff |
| `test_holdout_leakage` | holdout leakage |
| `test_probability` | probability |
| `test_shape` | shape |
| `test_one_class` | one class |
| `test_deterministic` | deterministic |
| `test_nonfinite_training` | nonfinite training |
| `test_future_label_excluded` | future label excluded |

## Gelişmiş deney planı

1. Müşteri kimliği ekleyip zamansal holdoutta grup sızıntısı kontrol edin.
2. Ayrı geçmiş validation penceresinde maliyet bazlı eşik seçin.
3. Model parametrelerini ve scalerı JSON sürümüyle dışa aktarın.
4. Gerçek etiketli veriyle kalibrasyon ve segment başına hata maliyetini raporlayın.
