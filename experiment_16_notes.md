# Experiment 16 - Qualitative Sample Review

## Tujuan

Memilih sampel error paling representatif dari output Experiment 15 untuk
review visual front-vs-side dan pembahasan kualitatif di laporan.

## Output

- `results/exp16_selected_failure_cases.csv`
- `results/exp16_failure_case_images/`
- `experiment_16_notes.md`

## Ringkasan Seleksi

- Total sampel terpilih: 20
- Komposisi label: 11 safe_driving, 9 phone_use
- Komposisi case: {'both_wrong': 5, 'fusion_broke_front_correct': 5, 'fusion_fixed_front_error': 5, 'side_only_correct': 5}

## Temuan Awal Untuk Dilaporkan

- Front salah cukup sering terjadi pada sampel `safe_driving` yang condong diprediksi `phone_use` (6 dari 20 sampel terpilih).
- Side membantu terutama pada sampel aman yang ambigu dari front view (2 sampel pada `side_only_correct` + `fusion_fixed_front_error`).
- Fusion merusak terutama ketika front sudah benar tetapi side memberi sinyal kuat ke kelas lawan (5 sampel aman di `fusion_broke_front_correct`).
- Kasus `both_wrong` masih didominasi sampel aman yang sangat mirip `phone_use` (4 sampel), cocok untuk menjawab pertanyaan apakah kelas safe sering mirip phone_use.

## Observasi Visual Awal

- Pada beberapa `safe_driving`, front view menampilkan tangan kanan dekat wajah, setir, atau konsol tengah sehingga gesture aman terlihat mirip aktivitas memegang ponsel.
- Pada beberapa `phone_use`, objek ponsel tidak terlalu jelas dari front karena tertutup tangan, rendah di area lap, atau menyatu dengan pose tubuh; side view lebih mudah menangkap siluet tangan-ke-telinga atau perangkat di samping tubuh.
- Beberapa kasus `fusion_broke_front_correct` menunjukkan side view terlalu percaya diri pada sinyal yang salah, misalnya tangan di area gear/console atau komposisi penumpang belakang yang menambah clutter visual.
- Kasus `both_wrong` cenderung borderline: probabilitas front dan side sama-sama dekat ambang 0.5, menandakan frame memang ambigu secara visual, bukan sekadar kesalahan threshold tunggal.

## Empat Kategori Untuk Pembahasan Skripsi

### `fusion_fixed_front_error`

Kategori ini menunjukkan kondisi ketika front view gagal, tetapi side view
memberi sinyal tambahan yang cukup kuat sehingga fusion mengoreksi prediksi
menjadi benar. Secara visual, kasus ini sering muncul saat ponsel kurang jelas
dari depan, tetapi posisi tangan, sudut kepala, atau kedekatan perangkat dengan
telinga terlihat lebih tegas dari samping.

### `fusion_broke_front_correct`

Kategori ini penting untuk menjelaskan risiko fusion. Pada beberapa sampel
`safe_driving`, front view sebenarnya sudah benar, tetapi side view menangkap
pose tangan yang mirip pola penggunaan ponsel sehingga fusion bergeser ke kelas
`phone_use`. Kasus ini berkontribusi pada false alarm di kelas aman.

### `side_only_correct`

Kategori ini menegaskan bahwa side view memang membawa informasi komplementer.
Walau fusion tidak selalu cukup kuat untuk membalik keputusan akhir, side view
sendiri sudah bisa membaca konteks postur atau posisi lengan yang tidak
tertangkap jelas oleh front view.

### `both_wrong`

Kategori ini menunjukkan batas kemampuan kedua view pada frame yang sangat
ambigu. Beberapa sampel aman terlihat mirip aktivitas ponsel dari depan maupun
dari samping, sedangkan beberapa sampel `phone_use` tidak menampilkan perangkat
secara eksplisit. Kasus ini cocok dipakai untuk menekankan bahwa sebagian error
bersifat visual-intrinsik, bukan hanya kelemahan strategi fusion.

## Kalimat Inti Laporan

Analisis visual menunjukkan bahwa side view berperan sebagai informasi
komplementer pada beberapa kasus phone-use ketika front view kurang jelas.
Namun, fusion juga meningkatkan kecenderungan false alarm pada kelas safe
driving, terutama ketika pose tangan pada side view menyerupai pola penggunaan
ponsel. Hal ini menjelaskan mengapa fusion meningkatkan Macro F1 secara
keseluruhan, tetapi safe recall menurun dibanding front single-view.

## Checklist Review Visual

- Cek apakah tangan/ponsel tertutup atau keluar frame pada front.
- Cek apakah pose tubuh atau arah kepala lebih jelas pada side.
- Cek apakah front menonjolkan gesture yang mirip phone_use walau label sebenarnya safe.
- Cek apakah fusion gagal karena side terlalu percaya diri pada prediksi yang salah.

## Kesimpulan Cepat

- Experiment 15 sudah memberi daftar failure case per-sample.
- Experiment 16 ini melanjutkan ke tahap kurasi visual sehingga sampel bisa langsung dipakai untuk pembahasan kualitatif.
- Fokus utama pembahasan sebaiknya pada confusion `safe_driving -> phone_use`, kontribusi side saat front ambigu, dan beberapa contoh saat fusion justru menurunkan keputusan yang awalnya benar.
