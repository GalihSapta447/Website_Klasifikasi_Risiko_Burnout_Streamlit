# Website Klasifikasi Risiko Burnout — Streamlit App

Aplikasi Streamlit untuk klasifikasi risiko burnout (4 kelas: Low / Moderate / High /
Severe) menggunakan model **XGBoost** dan **Random Forest**, dengan navigasi dan alur
sesuai sitemap & flowchart yang sudah disepakati.

## 📁 Struktur Folder

```
burnout_app/
├── app.py                  # Aplikasi utama Streamlit
├── requirements.txt
├── README.md
├── models/
│   ├── xgboost_burnout_pipeline.joblib
│   └── random_forest_burnout_pipeline.joblib
└── riwayat_prediksi.json    # dibuat otomatis saat pertama kali prediksi disimpan
```

## 🚀 Cara Menjalankan

```bash
cd Website_Klasifikasi_Risiko_Burnout_Streamlit-main
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Pastikan kedua file `.joblib` sudah ada di folder `models/` (nama file harus persis
seperti di atas, atau ubah `MODEL_PATHS` di `app.py`).

## ⚠️ PENTING — Verifikasi Sebelum Dipakai

beberapa hal berikut adalah yang perlu di cek ulang
di environment lokal:

1. **Urutan & nama kolom fitur** — didefinisikan di `FEATURE_ORDER` (app.py), diambil
   dari 27 variabel yang kamu berikan. Jika pipeline kamu (ColumnTransformer) memakai
   nama kolom berbeda, sesuaikan.

2. **Nilai kategori (string) yang dikirim ke model** — didefinisikan di
   `VARIABLE_CONFIG` (app.py), contoh: `gender` dikirim sebagai `"Male"`/`"Female"`/`"Other"`,
   `therapy_access`/`uses_therapy`/`ai_tools_daily` dikirim sebagai `"Yes"`/`"No"`.
   **Jika encoder yang dipakai saat training mengharapkan nilai berbeda** (misalnya huruf
   kecil semua, singkatan lain, atau kategori tambahan), prediksi bisa gagal atau salah.

3. **Skala skor** (`manager_support_score`, `stress_score`, dll.) diasumsikan **1–10**.
   Sesuaikan `min`/`max` di `VARIABLE_CONFIG` kalau skala aslinya berbeda (mis. 1–5).

4. **`company_size`** diasumsikan kategorikal (`Small/Medium/Large/Enterprise`). Kalau
   di dataset training kamu ini berupa angka jumlah karyawan, ubah tipe field-nya
   menjadi `"number"` di `VARIABLE_CONFIG`.

5. **Urutan kelas output** (`classes_`) — kode sudah mencoba membaca otomatis dari
   pipeline (`model.classes_`), dengan fallback ke `CLASS_ORDER = ["Low","Moderate",
   "High","Severe"]`. Jalankan `inspect_model.py` untuk memastikan urutannya benar.



## 🧭 Struktur Navigasi

| Menu | Isi |
|---|---|
| 🏠 Beranda | Informasi aplikasi, Tujuan sistem, Petunjuk penggunaan |
| 🧮 Klasifikasi Risiko | Input 27 variabel → Validasi → Pilih model → Hasil kategori & probabilitas |
| 📚 Informasi Kelompok Variabel | Penjelasan tiap variabel per kategori (Demografi, Konteks Kerja, Gaya Hidup, Budaya Tempat Kerja) |
| 🕘 Riwayat Prediksi | Lihat riwayat, Ekspor CSV, Hapus satu/semua riwayat |

Alur pada halaman **Klasifikasi Risiko** mengikuti flowchart: input semua variabel →
validasi (kalau tidak lengkap/valid → tampil pesan error) → pilih model → transformasi
input → jalankan klasifikasi → tampilkan hasil → simpan ke riwayat (persisten di
`riwayat_prediksi.json`).
