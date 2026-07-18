# Draft Bab Hasil dan Pembahasan

## Ringkasan Skenario Eksperimen

Bagian ini menyajikan hasil akhir penelitian berdasarkan eksperimen yang sudah
diselesaikan, yaitu Experiment 14A revisi untuk model front single-view,
Experiment 13 sebagai pembanding side single-view, Experiment 15 untuk analisis
error numerik per-sample, dan Experiment 16 untuk analisis visual sampel
kualitatif. Pada tahap ini tidak dilakukan training ulang, tuning tambahan,
perubahan dataset, perubahan arsitektur model, maupun perubahan metode fusion.

Fokus utama evaluasi adalah **Macro F1-score**. Pemilihan metrik ini didasarkan
pada kebutuhan untuk menilai performa secara lebih seimbang pada dua kelas,
yaitu `safe_driving` dan `phone_use`. Dalam konteks klasifikasi biner dengan
karakter error yang tidak simetris, Macro F1 lebih representatif dibanding
akurasi semata karena tidak hanya menilai jumlah prediksi benar total, tetapi
juga mempertimbangkan keseimbangan precision dan recall antar kelas.

## Hasil Single-View Front

Model front single-view hasil Experiment 14A revisi memperoleh Macro F1 sebesar
**0.76626**, accuracy **0.86364**, precision macro **0.77183**, dan recall
macro **0.76111**. Dibandingkan eksperimen front sebelumnya, hasil ini menjadi
baseline final yang dipakai pada tahap analisis lanjutan.

Dari sisi confusion matrix, model front mengklasifikasikan 24 sampel
`safe_driving` dengan benar dan salah memprediksi 16 sampel aman sebagai
`phone_use`. Untuk kelas `phone_use`, model ini berhasil memprediksi 166 sampel
dengan benar, sedangkan 14 sampel lainnya salah diprediksi sebagai
`safe_driving`. Pola ini menunjukkan bahwa front single-view sudah cukup kuat
dalam mengenali aktivitas `phone_use`, tetapi masih menghasilkan false alarm
yang cukup terlihat pada kelas aman.

Selain metrik klasifikasi, front single-view juga memiliki nilai ECE sebesar
**0.12513** dan Brier Score **0.11311**. Pada penelitian ini, calibration
belum menjadi fokus utama pembahasan. Namun demikian, kedua nilai tersebut
tetap penting dicatat karena menunjukkan bahwa confidence model masih dapat
ditelaah lebih lanjut pada studi lanjutan, terutama jika penelitian ingin
mengevaluasi reliabilitas probabilitas prediksi secara lebih mendalam.

## Hasil Single-View Side

Model side single-view dari Experiment 13 menghasilkan Macro F1 sebesar
**0.62023**, accuracy **0.81364**, precision macro **0.66553**, dan recall
macro **0.60417**. Secara umum, nilai ini berada di bawah model front
single-view pada seluruh metrik utama.

Temuan ini menunjukkan bahwa side view tidak cukup kuat jika dipakai sebagai
classifier tunggal. Meski demikian, hasil side tidak berarti tidak berguna.
Justru pada tahap fusion dan analisis visual, side view terbukti tetap memiliki
peran sebagai sumber informasi komplementer pada kasus-kasus tertentu yang
kurang jelas dari sudut pandang depan.

## Perbandingan Front vs Side

Perbandingan antara front dan side menunjukkan bahwa front single-view adalah
modalitas yang lebih andal sebagai model utama. Selisih Macro F1 antara keduanya
cukup besar, yaitu **0.76626** pada front berbanding **0.62023** pada side.
Perbedaan ini mengindikasikan bahwa informasi visual dari sudut depan lebih
konsisten untuk membedakan `safe_driving` dan `phone_use` dalam skenario
penelitian ini.

Walaupun demikian, hasil side tidak dapat langsung dianggap gagal. Jika hanya
dilihat sebagai model tunggal, performanya memang lebih rendah. Namun jika
dilihat sebagai modalitas pendamping, side view masih bernilai karena dapat
menangkap postur lengan, arah kepala, atau posisi perangkat yang pada beberapa
sampel tidak tampak cukup jelas dari front view. Dengan kata lain, front lebih
baik sebagai basis keputusan utama, sedangkan side lebih relevan sebagai
informasi tambahan.

## Hasil Average Fusion

Average fusion menghasilkan Macro F1 sebesar **0.78059**, accuracy **0.89091**,
precision macro **0.86565**, dan recall macro **0.73889**. Dibanding baseline
front single-view, hasil ini menunjukkan peningkatan Macro F1 dari **0.76626**
menjadi **0.78059**. Karena Macro F1 dijadikan metrik utama, peningkatan ini
menjadi dasar bahwa fusion memberikan manfaat pada performa keseluruhan.

Pada confusion matrix fusion, kelas `phone_use` menunjukkan perbaikan yang
jelas. Sebanyak 176 sampel `phone_use` berhasil diprediksi benar, dan hanya 4
sampel yang salah menjadi `safe_driving`. Angka ini lebih baik daripada front
single-view yang masih memiliki 14 error `phone_use -> safe`. Dengan demikian,
fusion membantu mengurangi missed detection pada kelas `phone_use`.

Namun, perbaikan tersebut disertai trade-off pada kelas aman. Pada fusion,
hanya 20 sampel `safe_driving` yang diprediksi benar, sementara 20 sampel aman
salah diprediksi sebagai `phone_use`. Dibanding hasil front, jumlah error
`safe -> phone` justru meningkat dari 16 menjadi 20. Oleh sebab itu,
peningkatan Macro F1 tidak boleh ditafsirkan sebagai perbaikan merata di semua
aspek, melainkan sebagai perbaikan keseluruhan dengan konsekuensi tertentu.

## Hasil Adaptive Fusion

Adaptive fusion menghasilkan nilai yang sama dengan average fusion, yaitu Macro
F1 **0.78059**, accuracy **0.89091**, precision macro **0.86565**, dan recall
macro **0.73889**. Dengan hasil tersebut, adaptive fusion juga berada di atas
front single-view dalam hal Macro F1, tetapi tidak memberi tambahan keuntungan
dibanding average fusion.

Secara konseptual, adaptive fusion dirancang untuk memberi bobot berbeda pada
masing-masing view berdasarkan confidence prediksi. Namun, pada hasil akhir
penelitian ini, strategi tersebut belum menghasilkan keluaran yang lebih baik
dibanding rata-rata sederhana. Hal ini penting dicatat agar pembahasan tetap
proporsional: adaptive fusion bukan gagal, tetapi pada konfigurasi dan data
yang digunakan, manfaat tambahannya belum terlihat secara kuantitatif.

## Perbandingan Average Fusion dan Adaptive Fusion

Perbandingan langsung antara average fusion dan adaptive fusion menunjukkan
bahwa keduanya menghasilkan performa identik pada seluruh metrik utama yang
dipakai, termasuk Macro F1, accuracy, precision macro, dan recall macro.
Dengan demikian, adaptive fusion **belum mengungguli** average fusion pada
tahap ini.

Temuan tersebut memiliki dua implikasi. Pertama, average fusion dapat dianggap
sebagai metode yang lebih sederhana tetapi sudah cukup efektif untuk skenario
penelitian ini. Kedua, penggunaan adaptive fusion belum memberikan bukti
empiris bahwa pembobotan berbasis confidence menghasilkan keputusan yang lebih
baik dibanding rata-rata biasa. Karena itu, klaim bahwa adaptive fusion lebih
unggul tidak dapat didukung oleh hasil yang tersedia.

## Analisis Error Numerik dari Experiment 15

Experiment 15 dilakukan untuk memahami pola error pada level sampel. Dari total
220 sampel uji, front single-view memprediksi benar 190 sampel, side 179
sampel, sedangkan average dan adaptive fusion masing-masing benar pada 196
sampel. Hasil ini kembali menguatkan bahwa fusion memberi perbaikan kuantitatif
dibanding model front tunggal.

Jika dilihat lebih rinci, terdapat 163 sampel yang benar pada front dan side
secara bersamaan. Sebanyak 27 sampel hanya benar pada front, sedangkan 16
sampel hanya benar pada side. Selain itu, terdapat 14 sampel yang salah pada
kedua view. Distribusi ini menunjukkan bahwa meskipun side lebih lemah sebagai
model tunggal, ia tetap membawa informasi yang tidak sepenuhnya redundant
terhadap front.

Kategori `fusion_fixed_front_error` berjumlah 11 sampel. Artinya, terdapat 11
kasus ketika front salah, tetapi fusion berhasil memperbaiki keputusan akhir.
Sebaliknya, kategori `fusion_broke_front_correct` berjumlah 5 sampel, yaitu
kasus ketika front sebenarnya sudah benar, tetapi fusion justru mengubahnya
menjadi salah. Perbandingan 11 lawan 5 ini mendukung alasan mengapa fusion
masih memberikan peningkatan Macro F1 secara keseluruhan.

Dari sisi arah kesalahan, front menghasilkan 16 error `safe -> phone` dan 14
error `phone -> safe`. Pada fusion, error `phone -> safe` turun tajam menjadi
4, tetapi error `safe -> phone` naik menjadi 20. Ini menunjukkan bahwa fusion
cenderung menjadi lebih sensitif terhadap indikasi `phone_use`, sehingga lebih
jarang melewatkan aktivitas `phone_use`, tetapi lebih sering memberi alarm
palsu pada kelas aman. Interpretasi ini penting karena memperlihatkan bentuk
trade-off yang lebih konkret daripada sekadar perubahan satu angka Macro F1.

## Analisis Visual dari Experiment 16

Experiment 16 melanjutkan hasil numerik Experiment 15 ke tahap pembacaan
kualitatif. Sampel visual dipilih dari empat kategori, yaitu
`fusion_fixed_front_error`, `fusion_broke_front_correct`, `side_only_correct`,
dan `both_wrong`. Keempat kategori ini dipakai agar pembahasan tidak hanya
menjelaskan kapan fusion membantu, tetapi juga kapan fusion merugikan dan kapan
kedua view sama-sama gagal.

Pada kategori `fusion_fixed_front_error`, beberapa sampel `phone_use`
menunjukkan bahwa objek ponsel atau gestur pemakaian tidak selalu jelas dari
front view. Dalam kondisi seperti ini, side view lebih informatif karena dapat
menangkap siluet tangan ke telinga, posisi lengan, atau keberadaan perangkat di
samping tubuh. Temuan ini mendukung gagasan bahwa side view memiliki peran
komplementer, khususnya saat bukti visual dari depan kurang kuat.

Pada kategori `fusion_broke_front_correct`, ditemukan situasi sebaliknya.
Beberapa sampel `safe_driving` sebenarnya sudah diprediksi benar oleh front,
tetapi side view memberikan sinyal yang menyerupai pola penggunaan ponsel,
misalnya posisi tangan di area wajah, setir, atau konsol tengah. Saat sinyal
tersebut ikut digabungkan, fusion cenderung bergeser ke kelas `phone_use`.
Kasus ini menjelaskan peningkatan false alarm pada kelas aman.

Kategori `side_only_correct` memperlihatkan bahwa side view memang memuat
informasi yang tidak selalu tertangkap oleh front. Walaupun tidak semua kasus
ini berhasil dibalik oleh fusion, keberadaan kategori tersebut menunjukkan
bahwa kontribusi side bukan sekadar tambahan yang pasif, melainkan informasi
yang pada beberapa contoh benar-benar relevan untuk pengenalan aktivitas.

Sementara itu, kategori `both_wrong` memperlihatkan batas kemampuan sistem.
Pada beberapa sampel, baik front maupun side sama-sama ambigu. Ada frame aman
yang secara visual menyerupai `phone_use`, dan ada pula frame `phone_use` yang
tidak memperlihatkan perangkat secara cukup eksplisit. Dengan demikian, tidak
semua error dapat dijelaskan hanya sebagai kelemahan fusion; sebagian memang
bersumber dari ambiguitas visual frame itu sendiri.

Secara keseluruhan, analisis visual menunjukkan bahwa side view berperan
sebagai informasi komplementer pada beberapa kasus phone-use ketika front view
kurang jelas. Namun, fusion juga meningkatkan kecenderungan false alarm pada
kelas safe driving, terutama ketika pose tangan pada side view menyerupai pola
penggunaan ponsel. Hal ini menjelaskan mengapa fusion meningkatkan Macro F1
secara keseluruhan, tetapi safe recall menurun dibanding front single-view.

## Analisis Adaptive Fusion pada Experiment 17

Experiment 17 dilakukan sebagai analisis lanjutan pada level decision-level
fusion. Tujuan utamanya bukan untuk mengganti hasil utama penelitian, melainkan
untuk memahami mengapa adaptive fusion lama menghasilkan performa identik
dengan average fusion, serta untuk menguji apakah variasi sharpening bobot
berbasis confidence dapat memunculkan perilaku yang berbeda tanpa mengubah
dataset, checkpoint, model, maupun rumus adaptive fusion lama.

Pada analisis bobot adaptive fusion lama, diperoleh mean bobot front sebesar
**0.52368** dan mean bobot side **0.47632**. Simpangan baku bobot front hanya
**0.03445**, dengan rentang dari **0.42955** sampai **0.61073**. Selain itu,
sebanyak **167 dari 220 sampel** memiliki `weight_front` pada rentang
**0.45-0.55**. Temuan ini menunjukkan bahwa meskipun adaptive fusion secara
matematis tidak selalu memberi bobot 0.5 persis, dalam praktik bobot yang
dihasilkan tetap sangat dekat dengan average fusion pada sebagian besar sampel.

Hal tersebut diperkuat oleh hasil prediksi akhir. Pada Experiment 17A, jumlah
prediksi adaptive fusion lama yang berbeda dari average fusion adalah
**0 sampel**. Dengan kata lain, variasi bobot yang muncul belum cukup besar
untuk menggeser skor gabungan melewati threshold keputusan secara berbeda.
Karena keputusan klasifikasi tetap sama pada seluruh sampel uji, maka seluruh
metrik evaluasi adaptive fusion lama juga tetap identik dengan average fusion.

Experiment 17B kemudian mencoba adaptive sharpening dengan faktor
`alpha = 1, 2, 3, 5, 10`, menggunakan rumus `weight = softmax(alpha *
confidence)` pada level fusion saja. Hasilnya menunjukkan bahwa seluruh nilai
alpha tersebut tetap menghasilkan **Macro F1 = 0.78059**, sama dengan average
fusion dan adaptive fusion lama. Pada seluruh nilai alpha, arah kesalahan juga
tetap sama, yaitu `safe->phone = 20` dan `phone->safe = 4`. Tidak ada satu pun
konfigurasi alpha yang menghasilkan perubahan prediksi akhir terhadap average
fusion.

Temuan ini memberi penjelasan yang lebih jelas mengenai identitas hasil average
dan adaptive. Perbedaan confidence antara front dan side memang ada, tetapi
belum cukup tajam atau belum cukup informatif untuk mengubah keputusan akhir
setelah digabungkan. Bahkan setelah bobot dipertajam dengan alpha yang lebih
besar, skor fusion tetap tidak berpindah sisi threshold pada sampel mana pun.
Dengan demikian, penyebab identitas hasil bukan sekadar karena implementasi
adaptive yang salah, tetapi karena distribusi confidence aktual pada pasangan
prediksi front dan side belum menghasilkan keputusan yang berbeda.

Berdasarkan hasil tersebut, dapat disimpulkan bahwa adaptive fusion belum
mengungguli average fusion pada konfigurasi penelitian ini. Average fusion
tetap menjadi baseline fusion yang lebih sederhana, sedangkan adaptive fusion
lama maupun adaptive sharpening alpha 1, 2, 3, 5, dan 10 belum memberikan
peningkatan Macro F1 di atas **0.78059**. Karena itu, hasil utama penelitian
tetap tidak berubah, yaitu average fusion dan adaptive fusion sama-sama berada
pada Macro F1 **0.78059**.

## Analisis Threshold Fusion pada Experiment 18

Experiment 18 dilakukan sebagai analisis lanjutan berbasis validation set untuk
melihat apakah threshold keputusan pada output fusion dapat disesuaikan tanpa
training ulang. Pada eksperimen ini tidak dilakukan perubahan model, dataset,
split, checkpoint, maupun rumus average fusion dan adaptive fusion lama.
Analisis hanya dilakukan pada level probabilitas output decision-level fusion.

Threshold untuk kelas `phone_use` dicari pada validation set dalam rentang
0.30 sampai 0.80 dengan step 0.01. Setelah threshold terbaik dipilih
berdasarkan Macro F1 validation, threshold tersebut dievaluasi satu kali pada
test set. Hasil pencarian menunjukkan bahwa threshold terbaik untuk
`average_fusion` dan `adaptive_fusion` sama-sama berada pada **0.49**.

Meskipun demikian, threshold terbaik dari validation set tersebut tidak
memberikan peningkatan pada test set. Pada baseline threshold **0.50**, fusion
memperoleh Macro F1 **0.78059**, accuracy **0.89091**, dengan `safe->phone =
20` dan `phone->safe = 4`. Ketika threshold diubah menjadi **0.49**, Macro F1
test justru turun menjadi **0.72982** dengan accuracy **0.87273**. Pada saat
yang sama, false alarm `safe->phone` meningkat dari **20** menjadi **24**,
sedangkan `phone->safe` tetap **4**.

Temuan ini menunjukkan bahwa threshold tuning berbasis validation pada
Experiment 18 belum berhasil menghasilkan perbaikan lanjutan terhadap hasil
fusion yang sudah ada. Target untuk mendorong Macro F1 ke kisaran **0.80**
tidak tercapai, dan threshold terpilih justru memperburuk error pada kelas
`safe_driving`. Selain itu, average fusion dan adaptive fusion tetap identik,
baik pada threshold baseline **0.50** maupun pada threshold terpilih
**0.49**.

Berdasarkan hasil tersebut, threshold **0.50** tetap dipertahankan sebagai
hasil utama untuk fusion, dengan Macro F1 **0.78059**. Dengan kata lain,
Experiment 18 menunjukkan bahwa peningkatan lebih lanjut belum berhasil dicapai
melalui threshold tuning pada konfigurasi ini, walaupun analisis ini tetap
bermanfaat untuk menegaskan bahwa batas performa fusion saat ini tidak mudah
didorong hanya dengan menggeser threshold keputusan.

## Experiment 19: Side Class-Weighting

Experiment 19 dilakukan karena upaya analisis pada level fusion di Experiment
17 dan 18 belum berhasil meningkatkan hasil utama. Arah perbaikannya kemudian
dipindahkan ke model side single-view melalui class-weighting sweep, tanpa
mengubah front checkpoint, dataset, split, backbone, arsitektur model, maupun
metode fusion. Tujuannya adalah memperkuat kontribusi side view agar informasi
komplementer yang dibawa ke fusion menjadi lebih bermanfaat secara kuantitatif.

Motivasi eksperimen ini cukup jelas jika melihat baseline side. Pada
Experiment 13, side single-view hanya mencapai Macro F1 **0.62023** dan secara
umum lebih lemah dibanding front. Salah satu indikasi pentingnya adalah recall
kelas `safe_driving` yang relatif rendah, sehingga model side cenderung mudah
mengarahkan sampel aman ke kelas `phone_use`. Kelemahan ini ikut membatasi
seberapa besar kontribusi side ketika digabungkan dengan front.

Experiment 19 kemudian menguji class-weighting berbasis rumus
`weight = balanced_weight ** gamma` pada model side, dengan beberapa rentang
gamma sampai akhirnya ditemukan kandidat terbaik pada **gamma = 0.95**.
Pada konfigurasi ini, hasil fusion antara Front14A dan Side hasil class
weighting mencapai Macro F1 **0.79323**, lebih tinggi dibanding baseline
fusion Front14A + Side13 yang berada pada **0.78059**. Dengan demikian, class
weighting pada side dapat dikatakan berhasil meningkatkan Macro F1 fusion,
meskipun peningkatannya tetap terbatas dan belum menembus **0.80**.

Dilihat dari confusion matrix, baseline fusion lama memiliki bentuk
`[[20, 20], [4, 176]]`, sedangkan kandidat terbaik Experiment 19 pada
`gamma = 0.95` menghasilkan confusion matrix `[[25, 15], [11, 169]]`.
Perubahan ini menunjukkan trade-off yang cukup penting. Di satu sisi, error
`safe->phone` membaik dari **20** menjadi **15**, sehingga false alarm pada
kelas aman menurun. Namun di sisi lain, error `phone->safe` memburuk dari
**4** menjadi **11**, yang berarti lebih banyak sampel `phone_use` terlewat
dibanding baseline fusion lama. Karena itu, hasil Experiment 19 tidak boleh
ditafsirkan sebagai perbaikan seragam pada seluruh aspek.

Pencarian lanjutan di sekitar kandidat terbaik juga sudah dilakukan melalui
narrow sweep pada gamma `0.93`, `0.94`, `0.96`, `0.97`, dan `0.98`. Tidak ada
satu pun konfigurasi tersebut yang melampaui performa `gamma = 0.95`.
Sebagai pembanding, `gamma = 1.00` menghasilkan Macro F1 fusion **0.79166**
dengan confusion matrix `[[26, 14], [13, 167]]`, sehingga tetap sedikit di
bawah `gamma = 0.95`. Temuan ini menegaskan bahwa **gamma = 0.95** merupakan
kandidat terbaik final pada Experiment 19 dalam konfigurasi yang diuji.

Secara keseluruhan, Experiment 19 memberi temuan penting bahwa peningkatan
lebih lanjut masih mungkin dicapai dengan memperbaiki kualitas modalitas side,
bukan hanya dengan memodifikasi aturan fusion. Akan tetapi, pola hasilnya juga
menunjukkan bahwa peningkatan Macro F1 fusion tetap datang bersama perubahan
karakter error. Oleh sebab itu, hasil terbaik Experiment 19 lebih tepat
diposisikan sebagai peningkatan yang nyata tetapi tetap memiliki trade-off,
bukan sebagai solusi yang memperbaiki seluruh komponen performa secara bersamaan.

## Pembahasan Trade-off Fusion

Jika hanya melihat Macro F1, fusion merupakan hasil terbaik pada penelitian
ini. Nilai Macro F1 meningkat dari **0.76626** pada front single-view menjadi
**0.78059** pada baseline fusion, dan kemudian meningkat lagi menjadi
**0.79323** pada kandidat terbaik Experiment 19. Karena Macro F1 adalah
metrik utama, maka hasil ini cukup untuk menyatakan bahwa fusion memberikan
peningkatan performa akhir.

Namun, pembahasan hasil tidak boleh berhenti pada kesimpulan tersebut saja.
Angka confusion matrix dan analisis error menunjukkan bahwa peningkatan Macro F1
dicapai bersamaan dengan perubahan karakter kesalahan. Pada baseline fusion,
sistem menjadi lebih kuat dalam mengenali `phone_use`, yang terlihat dari
turunnya error `phone -> safe` dari 14 menjadi 4. Akan tetapi, fusion juga
membuat sistem lebih mudah menganggap sampel aman sebagai `phone_use`,
sehingga error `safe -> phone` naik dari 16 menjadi 20. Pada Experiment 19,
arah trade-off berubah: kandidat terbaik `gamma = 0.95` justru menurunkan
error `safe -> phone` menjadi 15, tetapi menaikkan `phone -> safe` menjadi 11.

Dengan demikian, trade-off fusion dapat dirumuskan secara hati-hati sebagai
berikut: fusion memang meningkatkan performa keseluruhan menurut Macro F1,
tetapi distribusi kesalahannya dapat berubah tergantung kualitas modalitas side
yang digunakan. Dalam konteks penelitian ini, hasil tersebut tetap dapat
dianggap positif karena metrik utama meningkat, tetapi interpretasinya harus
disertai penjelasan bahwa perbaikan tidak bersifat seragam pada semua kelas.

## Keterbatasan Hasil

Beberapa keterbatasan perlu dicatat agar interpretasi hasil tidak berlebihan.
Pertama, adaptive fusion belum menunjukkan keunggulan dibanding average fusion,
sehingga belum ada dasar empiris untuk menyimpulkan bahwa pembobotan berbasis
confidence lebih efektif daripada rata-rata sederhana.

Kedua, calibration belum menjadi fokus utama penelitian ini. Nilai ECE dan
Brier Score pada front single-view sudah dicatat, tetapi belum dianalisis lebih
jauh untuk menghubungkan reliabilitas confidence dengan perilaku fusion pada
setiap kategori error. Oleh karena itu, aspek ini lebih tepat diposisikan
sebagai peluang analisis lanjutan, bukan sebagai klaim utama hasil sekarang.

Ketiga, analisis visual pada Experiment 16 bersifat kualitatif dan berbasis
sampel terpilih. Hasilnya sangat berguna untuk menjelaskan pola kesalahan, tetapi
tetap tidak dimaksudkan sebagai generalisasi absolut terhadap seluruh populasi
data di luar sampel yang dianalisis.

Keempat, meskipun Experiment 19 berhasil meningkatkan Macro F1 fusion menjadi
**0.79323**, tidak ada konfigurasi class-weighting yang berhasil menembus
**0.80**. Selain itu, peningkatan tersebut juga dibayar dengan memburuknya
error `phone->safe` dibanding baseline fusion lama. Karena itu, hasil ini lebih
tepat dipahami sebagai perbaikan terarah dengan trade-off yang masih perlu
dipertimbangkan, bukan sebagai titik akhir yang sepenuhnya optimal.

## Kesimpulan Sementara

Berdasarkan seluruh hasil yang tersedia, front single-view tetap menjadi
baseline tunggal terkuat, sedangkan side single-view berperan lebih efektif
sebagai modalitas komplementer daripada sebagai classifier utama. Baseline
fusion average maupun adaptive meningkatkan Macro F1 dari **0.76626** menjadi
**0.78059**, dan Experiment 19 kemudian menunjukkan bahwa perbaikan lebih lanjut
masih dapat dicapai melalui side class-weighting hingga mencapai **0.79323**
pada kandidat terbaik `gamma = 0.95`.

Meski demikian, adaptive fusion belum mengungguli average fusion karena
hasil keduanya identik. Selain itu, peningkatan performa fusion tetap disertai
trade-off yang berubah bentuk antar konfigurasi. Pada baseline fusion, masalah
utama berada pada false alarm `safe_driving`, sedangkan pada kandidat terbaik
Experiment 19, false alarm tersebut membaik tetapi error `phone_use -> safe`
justru membesar. Oleh sebab itu, kesimpulan yang paling tepat bukan bahwa
seluruh aspek performa selalu membaik, melainkan bahwa fusion memberi
peningkatan performa keseluruhan menurut Macro F1 sambil tetap membawa
konsekuensi tertentu pada distribusi kesalahan.
