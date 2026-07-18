# Experiment 19 Notes

## Tujuan Experiment 19 Revised v3

Experiment 19 revised v3 bertujuan mencari titik tengah antara baseline side
lama dan revised v2 full balanced. Fokus revisi ini adalah membuat side view
sedikit lebih seimbang ke arah `safe_driving`, tetapi tidak terlalu agresif
hingga merusak kelas `phone_use` secara besar-besaran.

## Riwayat Singkat Experiment 19

### 1. Experiment 19 awal gagal

- Side Macro F1 = `0.58625`
- Fusion Macro F1 = `0.76827`
- Indikasi utama: konfigurasi terlalu konservatif dan tidak cocok untuk side
  view.

### 2. Experiment 19 revisi pertama

- Side Macro F1 = `0.62023`
- Fusion Macro F1 = `0.78059`
- Hasil ini berhasil memulihkan performa ke baseline Side13, tetapi belum
  memberi peningkatan baru.

### 3. Experiment 19 revised v2 full balanced

- Class weight full balanced:
  - safe_driving = `3.0795`
  - phone_use = `0.5969`
- Side Macro F1 = `0.55772`
- Fusion Macro F1 = `0.79166`
- Arah perbaikannya benar di level fusion, karena `safe->phone` membaik dari
  `20` menjadi `14`.
- Namun trade-off-nya terlalu agresif karena `phone->safe` memburuk dari `4`
  menjadi `13`.

### 4. Experiment 19 revised v3 mild dan high-gamma class weight

- Revisi ini mencoba titik tengah menggunakan rumus:
  `weight_mild = weight_balanced ** gamma`
- Gamma yang diuji:
  - `0.25`
  - `0.50`
  - `0.75`
  - `0.85`
  - `0.90`
  - `0.95`

### 5. Experiment 19 narrow gamma sweep di sekitar kandidat terbaik

- Karena `gamma=0.95` menjadi kandidat terbaik pada sweep sebelumnya, dilakukan
  sweep sempit tambahan di sekitar titik tersebut.
- Gamma tambahan yang diuji:
  - `0.93`
  - `0.94`
  - `0.96`
  - `0.97`
  - `0.98`

## Konfigurasi Training Revised v3

Seluruh gamma memakai konfigurasi yang sama, kecuali bobot kelas:

- View: `side`
- Backbone: `EfficientNetV2-S`
- Optimizer: `AdamW`
- Learning rate: `3e-5`
- Weight decay: `1e-4`
- Dropout: `0.3`
- Label smoothing: `0.0`
- Frozen stages: `4`
- Scheduler: `ReduceLROnPlateau`
- Scheduler factor: `0.5`
- Scheduler patience: `1`
- Early stopping patience: `5`
- Batch size: `32`
- Max epochs: `30`
- Best checkpoint dipilih berdasarkan validation Macro F1

Distribusi train split yang dipakai untuk menghitung balanced class weight:
- safe_driving = `176`
- phone_use = `908`

Balanced class weight dasar:
- safe_driving = `3.0795`
- phone_use = `0.5969`

## Ringkasan Hasil Gamma

### Gamma 0.25

- Class weight:
  - safe_driving = `1.3247`
  - phone_use = `0.8790`
- Best epoch = `1`
- Train loss best = `0.57971`
- Val loss best = `0.52143`
- Loss gap = `-0.05828`
- Side Macro F1 = `0.45000`
- Side confusion = `[[0, 40], [0, 180]]`
- Side safe recall = `0.000`
- Side phone recall = `1.000`
- Side safe->phone = `40`
- Side phone->safe = `0`
- Fusion Macro F1 = `0.56681`
- Fusion accuracy = `0.84091`
- Fusion confusion = `[[5, 35], [0, 180]]`
- Fusion safe->phone = `35`
- Fusion phone->safe = `0`
- Fusion tembus 0.80 = `Tidak`

Interpretasi:
- Gamma `0.25` terlalu lemah. Model side kembali sangat bias ke `phone_use`
  dan gagal mengenali `safe_driving`.

### Gamma 0.50

- Class weight:
  - safe_driving = `1.7549`
  - phone_use = `0.7726`
- Best epoch = `8`
- Train loss best = `0.45549`
- Val loss best = `0.50143`
- Loss gap = `0.04595`
- Side Macro F1 = `0.64349`
- Side confusion = `[[13, 27], [13, 167]]`
- Side safe recall = `0.325`
- Side phone recall = `0.92778`
- Side safe->phone = `27`
- Side phone->safe = `13`
- Fusion Macro F1 = `0.77992`
- Fusion accuracy = `0.88636`
- Fusion confusion = `[[21, 19], [6, 174]]`
- Fusion safe->phone = `19`
- Fusion phone->safe = `6`
- Fusion tembus 0.80 = `Tidak`

Interpretasi:
- Gamma `0.50` adalah titik tengah paling stabil.
- Side Macro F1 naik di atas baseline Side13 (`0.62023` -> `0.64349`).
- Namun fusion masih sedikit di bawah baseline lama `0.78059`.

### Gamma 0.75

- Class weight:
  - safe_driving = `2.3247`
  - phone_use = `0.6791`
- Best epoch = `5`
- Train loss best = `0.58708`
- Val loss best = `0.60428`
- Loss gap = `0.01720`
- Side Macro F1 = `0.58696`
- Side confusion = `[[9, 31], [13, 167]]`
- Side safe recall = `0.225`
- Side phone recall = `0.92778`
- Side safe->phone = `31`
- Side phone->safe = `13`
- Fusion Macro F1 = `0.77436`
- Fusion accuracy = `0.88636`
- Fusion confusion = `[[20, 20], [5, 175]]`
- Fusion safe->phone = `20`
- Fusion phone->safe = `5`
- Fusion tembus 0.80 = `Tidak`

Interpretasi:
- Gamma `0.75` lebih baik daripada full balanced v2 pada `phone->safe`, tetapi
  tidak cukup membantu `safe_driving`, sehingga fusion juga tetap di bawah
  baseline utama.

### Gamma 0.85

- Class weight:
  - safe_driving = `2.6014`
  - phone_use = `0.6450`
- Best epoch = `5`
- Train loss best = `0.60389`
- Val loss best = `0.62038`
- Loss gap = `0.01649`
- Side Macro F1 = `0.62212`
- Side confusion = `[[13, 27], [18, 162]]`
- Side safe recall = `0.325`
- Side phone recall = `0.900`
- Side safe->phone = `27`
- Side phone->safe = `18`
- Fusion Macro F1 = `0.77327`
- Fusion accuracy = `0.87727`
- Fusion confusion = `[[22, 18], [9, 171]]`
- Fusion safe->phone = `18`
- Fusion phone->safe = `9`
- Fusion tembus 0.80 = `Tidak`

Interpretasi:
- Gamma `0.85` mulai mendekati perilaku full balanced, tetapi peningkatan ke
  arah `safe_driving` masih belum cukup kuat untuk mengungguli baseline fusion.

### Gamma 0.90

- Class weight:
  - safe_driving = `2.7519`
  - phone_use = `0.6285`
- Best epoch = `5`
- Train loss best = `0.61789`
- Val loss best = `0.63450`
- Loss gap = `0.01661`
- Side Macro F1 = `0.57535`
- Side confusion = `[[18, 22], [45, 135]]`
- Side safe recall = `0.450`
- Side phone recall = `0.750`
- Side safe->phone = `22`
- Side phone->safe = `45`
- Fusion Macro F1 = `0.77256`
- Fusion accuracy = `0.87273`
- Fusion confusion = `[[23, 17], [11, 169]]`
- Fusion safe->phone = `17`
- Fusion phone->safe = `11`
- Fusion tembus 0.80 = `Tidak`

Interpretasi:
- Gamma `0.90` membuat side lebih berani ke kelas `safe_driving`, tetapi biaya
  kesalahannya pada `phone_use` menjadi terlalu besar sehingga fusion tidak ikut
  naik.

### Gamma 0.95

- Class weight:
  - safe_driving = `2.9111`
  - phone_use = `0.6125`
- Best epoch = `5`
- Train loss best = `0.62352`
- Val loss best = `0.63856`
- Loss gap = `0.01504`
- Side Macro F1 = `0.55346`
- Side confusion = `[[19, 21], [54, 126]]`
- Side safe recall = `0.475`
- Side phone recall = `0.700`
- Side safe->phone = `21`
- Side phone->safe = `54`
- Fusion Macro F1 = `0.79323`
- Fusion accuracy = `0.88182`
- Fusion confusion = `[[25, 15], [11, 169]]`
- Fusion safe->phone = `15`
- Fusion phone->safe = `11`
- Fusion tembus 0.80 = `Tidak`

Interpretasi:
- Gamma `0.95` memberi hasil fusion terbaik dari seluruh sweep gamma yang
  dicoba pada revised v3. Meskipun side single-view turun, kombinasi dengan
  front justru menghasilkan Macro F1 fusion tertinggi.

### Gamma 0.93

- Class weight:
  - safe_driving = `2.8464`
  - phone_use = `0.6189`
- Best epoch = `5`
- Train loss best = `0.62137`
- Val loss best = `0.63709`
- Loss gap = `0.01572`
- Side Macro F1 = `0.57397`
- Side confusion = `[[19, 21], [48, 132]]`
- Side safe recall = `0.475`
- Side phone recall = `0.73333`
- Side safe->phone = `21`
- Side phone->safe = `48`
- Fusion Macro F1 = `0.78301`
- Fusion accuracy = `0.87727`
- Fusion confusion = `[[24, 16], [11, 169]]`
- Fusion safe->phone = `16`
- Fusion phone->safe = `11`
- Fusion tembus 0.80 = `Tidak`

Interpretasi:
- Gamma `0.93` masih berada dekat dengan zona kandidat terbaik, tetapi Macro F1
  fusion turun cukup jelas dibanding `gamma=0.95`.

### Gamma 0.94

- Class weight:
  - safe_driving = `2.8786`
  - phone_use = `0.6157`
- Best epoch = `5`
- Train loss best = `0.62247`
- Val loss best = `0.63783`
- Loss gap = `0.01537`
- Side Macro F1 = `0.56706`
- Side confusion = `[[19, 21], [50, 130]]`
- Side safe recall = `0.475`
- Side phone recall = `0.72222`
- Side safe->phone = `21`
- Side phone->safe = `50`
- Fusion Macro F1 = `0.78301`
- Fusion accuracy = `0.87727`
- Fusion confusion = `[[24, 16], [11, 169]]`
- Fusion safe->phone = `16`
- Fusion phone->safe = `11`
- Fusion tembus 0.80 = `Tidak`

Interpretasi:
- Gamma `0.94` menghasilkan outcome fusion yang sama dengan `0.93`, sehingga
  belum mendekati hasil terbaik `0.95`.

### Gamma 0.96

- Class weight:
  - safe_driving = `2.9441`
  - phone_use = `0.6094`
- Best epoch = `14`
- Train loss best = `0.53113`
- Val loss best = `0.58362`
- Loss gap = `0.05249`
- Side Macro F1 = `0.58393`
- Side confusion = `[[24, 16], [57, 123]]`
- Side safe recall = `0.600`
- Side phone recall = `0.68333`
- Side safe->phone = `16`
- Side phone->safe = `57`
- Fusion Macro F1 = `0.77629`
- Fusion accuracy = `0.86818`
- Fusion confusion = `[[25, 15], [14, 166]]`
- Fusion safe->phone = `15`
- Fusion phone->safe = `14`
- Fusion tembus 0.80 = `Tidak`

Interpretasi:
- Gamma `0.96` mendorong side lebih kuat ke arah `safe_driving`, tetapi
  kesalahan `phone->safe` pada side membesar sehingga fusion justru turun.

### Gamma 0.97

- Class weight:
  - safe_driving = `2.9774`
  - phone_use = `0.6062`
- Best epoch = `8`
- Train loss best = `0.57887`
- Val loss best = `0.61286`
- Loss gap = `0.03399`
- Side Macro F1 = `0.56805`
- Side confusion = `[[22, 18], [57, 123]]`
- Side safe recall = `0.550`
- Side phone recall = `0.68333`
- Side safe->phone = `18`
- Side phone->safe = `57`
- Fusion Macro F1 = `0.78184`
- Fusion accuracy = `0.87273`
- Fusion confusion = `[[25, 15], [13, 167]]`
- Fusion safe->phone = `15`
- Fusion phone->safe = `13`
- Fusion tembus 0.80 = `Tidak`

Interpretasi:
- Gamma `0.97` sedikit membaik dibanding `0.96`, tetapi masih belum dapat
  melampaui kandidat utama `0.95`.

### Gamma 0.98

- Class weight:
  - safe_driving = `3.0110`
  - phone_use = `0.6031`
- Best epoch = `8`
- Train loss best = `0.57962`
- Val loss best = `0.61342`
- Loss gap = `0.03381`
- Side Macro F1 = `0.56113`
- Side confusion = `[[22, 18], [59, 121]]`
- Side safe recall = `0.550`
- Side phone recall = `0.67222`
- Side safe->phone = `18`
- Side phone->safe = `59`
- Fusion Macro F1 = `0.79172`
- Fusion accuracy = `0.87727`
- Fusion confusion = `[[26, 14], [13, 167]]`
- Fusion safe->phone = `14`
- Fusion phone->safe = `13`
- Fusion tembus 0.80 = `Tidak`

Interpretasi:
- Gamma `0.98` sangat dekat dengan full balanced `1.0`, dan hasil fusion-nya
  juga nyaris identik. Namun hasil ini masih sedikit di bawah `gamma=0.95`.

## Tabel Ringkas Gamma

| Gamma | W_safe | W_phone | Side F1 | Side safe recall | Side phone recall | Fusion F1 | Fusion safe->phone | Fusion phone->safe |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.25 | 1.3247 | 0.8790 | 0.45000 | 0.000 | 1.000 | 0.56681 | 35 | 0 |
| 0.50 | 1.7549 | 0.7726 | 0.64349 | 0.325 | 0.92778 | 0.77992 | 19 | 6 |
| 0.75 | 2.3247 | 0.6791 | 0.58696 | 0.225 | 0.92778 | 0.77436 | 20 | 5 |
| 0.85 | 2.6014 | 0.6450 | 0.62212 | 0.325 | 0.90000 | 0.77327 | 18 | 9 |
| 0.90 | 2.7519 | 0.6285 | 0.57535 | 0.450 | 0.75000 | 0.77256 | 17 | 11 |
| 0.93 | 2.8464 | 0.6189 | 0.57397 | 0.475 | 0.73333 | 0.78301 | 16 | 11 |
| 0.94 | 2.8786 | 0.6157 | 0.56706 | 0.475 | 0.72222 | 0.78301 | 16 | 11 |
| 0.95 | 2.9111 | 0.6125 | 0.55346 | 0.475 | 0.70000 | 0.79323 | 15 | 11 |
| 0.96 | 2.9441 | 0.6094 | 0.58393 | 0.600 | 0.68333 | 0.77629 | 15 | 14 |
| 0.97 | 2.9774 | 0.6062 | 0.56805 | 0.550 | 0.68333 | 0.78184 | 15 | 13 |
| 0.98 | 3.0110 | 0.6031 | 0.56113 | 0.550 | 0.67222 | 0.79172 | 14 | 13 |

## Kandidat Terbaik

Pemilihan kandidat dilakukan dengan aturan:
1. Fusion Macro F1 tertinggi
2. Jika seri, pilih `phone->safe` lebih rendah
3. Jika masih seri, pilih `safe->phone` lebih rendah

Berdasarkan aturan tersebut, kandidat terbaik dari sweep gamma `0.25` sampai
`0.98` tetap:
- **gamma = 0.95**
- Fusion Macro F1 = `0.79323`
- Fusion confusion = `[[25, 15], [11, 169]]`

## Apakah Ada Yang Tembus 0.80

Tidak. Tidak ada gamma `0.25`, `0.50`, `0.75`, `0.85`, `0.90`, `0.93`,
`0.94`, `0.95`, `0.96`, `0.97`, atau `0.98` yang menembus `0.80`.

## Trade-off Safe->Phone dan Phone->Safe

Perbandingan dengan revised v2 full balanced:
- Revised v2 full balanced:
  - Fusion Macro F1 = `0.79166`
  - safe->phone = `14`
  - phone->safe = `13`

Perbandingan kandidat terbaik current sweep (`gamma=0.95`):
- Fusion Macro F1 naik tipis menjadi `0.79323`
- safe->phone sedikit memburuk dari `14` menjadi `15`
- phone->safe membaik dari `13` menjadi `11`

Interpretasi:
- Sweep high-gamma menunjukkan bahwa titik yang sangat dekat dengan full
  balanced lebih menjanjikan daripada mild gamma yang lebih kecil.
- Gamma `0.95` memberi kompromi yang sedikit lebih baik daripada full balanced
  `1.0`: `phone->safe` turun dari `13` menjadi `11`, sementara `safe->phone`
  hanya naik tipis dari `14` menjadi `15`.
- Karena itu, Macro F1 fusion ikut naik tipis dari `0.79166` menjadi `0.79323`,
  tetapi masih belum mencapai `0.80`.
- Narrow sweep di sekitar `0.95` tidak menemukan kandidat yang lebih tinggi.
  Titik terdekat yang paling kompetitif adalah `0.98` dengan Macro F1
  `0.79172`, yang tetap berada di bawah `0.95`.

## Kesimpulan Sementara

- Gamma `0.25` terlalu lemah dan tidak layak dipakai.
- Gamma `0.50` tetap menjadi mild candidate terbaik jika fokus utamanya adalah
  menahan `phone->safe`, tetapi fusion-nya tidak melampaui baseline utama.
- Sweep high-gamma menunjukkan bahwa performa fusion terbaik saat ini justru
  datang dari **gamma `0.95`** dengan Macro F1 `0.79323`.
- Sweep sempit `0.93`, `0.94`, `0.96`, `0.97`, dan `0.98` tidak berhasil
  melampaui `gamma=0.95`.
- Dibanding full balanced `1.0`, gamma `0.95` memberi perbaikan tipis pada
  Macro F1 fusion dan menurunkan `phone->safe`, dengan konsekuensi kenaikan
  kecil pada `safe->phone`.
- Meski demikian, **belum ada kandidat** yang menembus `0.80`.
- Dengan demikian, hasil terbaru mendukung bahwa class weight yang mendekati
  full balanced masih paling potensial, tetapi ruang peningkatannya tetap
  terbatas pada konfigurasi fusion saat ini.
