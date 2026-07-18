# Experiment 17 Notes

## Tujuan Experiment 17

- Menganalisis kenapa adaptive fusion lama identik dengan average fusion.
- Melihat distribusi bobot adaptive lama pada checkpoint front Exp14A revisi dan side Exp13.
- Mencoba adaptive sharpening dengan faktor alpha di level decision-level fusion saja.
- Menjaga hasil utama tetap sama sampai ada bukti bahwa variasi baru benar-benar lebih baik.

## Alasan Adaptive Fusion Lama Identik Dengan Average Fusion

- Mean bobot front: 0.52368; mean bobot side: 0.47632.
- Simpangan baku bobot front: 0.03445; rentang bobot front: 0.42955 sampai 0.61073.
- Jumlah sampel dengan bobot front di antara 0.45-0.55: 167 dari 220.
- Jumlah prediksi adaptive lama yang berbeda dari average fusion: 0.
- Adaptive fusion lama identik dengan average fusion karena bobot yang dihasilkan tetap sangat dekat ke 0.5, sehingga skor gabungan tidak cukup bergeser untuk mengubah keputusan threshold.

## Hipotesis Sharpening Alpha

- Dengan menaikkan alpha pada softmax(confidence), bobot diharapkan menjadi lebih tajam.
- Jika perbedaan confidence front dan side memang informatif, alpha yang lebih besar seharusnya dapat menghasilkan prediksi yang berbeda dari average fusion.
- Namun, sharpening tidak otomatis memperbaiki hasil; ia juga bisa memperbesar bias error dari view yang salah tetapi terlalu percaya diri.

## Hasil Tiap Alpha

- alpha=1: Macro F1 0.78059, accuracy 0.89091, safe->phone 20, phone->safe 4, beda vs average 0, trade-off besar tidak menonjol.
- alpha=2: Macro F1 0.78059, accuracy 0.89091, safe->phone 20, phone->safe 4, beda vs average 0, trade-off besar tidak menonjol.
- alpha=3: Macro F1 0.78059, accuracy 0.89091, safe->phone 20, phone->safe 4, beda vs average 0, trade-off besar tidak menonjol.
- alpha=5: Macro F1 0.78059, accuracy 0.89091, safe->phone 20, phone->safe 4, beda vs average 0, trade-off besar tidak menonjol.
- alpha=10: Macro F1 0.78059, accuracy 0.89091, safe->phone 20, phone->safe 4, beda vs average 0, trade-off besar tidak menonjol.

## Apakah Ada Alpha Yang Benar-Benar Memperbaiki Macro F1

- Baseline average fusion: 0.78059.
- Baseline adaptive fusion lama: 0.78059.
- Tidak ada alpha yang benar-benar memperbaiki Macro F1 di atas baseline average/adaptive lama (0.78059). Nilai terbaik tetap 0.78059 pada alpha=1.

## Trade-off Safe Recall dan Phone Recall

- Perubahan alpha perlu dibaca bersama perubahan `safe->phone` dan `phone->safe`, bukan hanya Macro F1.
- Jika alpha yang lebih besar menurunkan `phone->safe` tetapi menaikkan `safe->phone`, maka peningkatan sensitivitas terhadap phone_use terjadi dengan trade-off false alarm pada safe_driving.
- Sebaliknya, jika alpha tertentu menurunkan false alarm safe tetapi menaikkan `phone->safe`, maka sistem menjadi lebih konservatif terhadap deteksi phone_use.

## Catatan Penting

- Experiment 17 belum otomatis mengganti hasil utama penelitian.
- Hasil utama tetap: Average Fusion = 0.78059 dan Adaptive Fusion lama = 0.78059.
- Variasi adaptive sharpening pada experiment ini bersifat analisis fusion-level dan perlu direview sebelum dipertimbangkan lebih lanjut.
