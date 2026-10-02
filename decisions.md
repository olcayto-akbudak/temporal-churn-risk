# Tasarım kararları

## Alan motorunu CLI'dan ayırmak

`run(config)` orkestrasyonu kaynak kodun doğrudan test edilebilmesini sağlar. JSON arayüz taşınabilirliği artırır; bu sürüm kullanıcı arayüzü barındırmaz.

## Seçilen yöntem

Lojistik regresyon, label olgunlaşması, holdout. Etiket zamanı cutoff öncesinde olmalıdır; holdout gözlem zamanı cutoff sonrasındadır.

## Bilinçli sınır

Sentetik veri ve basit modeldir; gerçek müşteride grup ayrımı, kalibrasyon ve model yönetişimi gerekir.

## Önerilen sonraki doğrulama

Gerçek kullanım hacmiyle testten önce mevcut kabul ve ret örneklerinin alan uzmanı tarafından onaylanması gerekir. Sonraki sürüm performans ölçümleri, dış adaptör sözleşmesi ve üretim gözlemlenebilirliğini ayrı karar kayıtlarında ele almalıdır.
