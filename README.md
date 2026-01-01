# Proyek Vibe Coding Sederhana Klasifikasi Depresi dengan Jinja2, Pelatihan Model Random Forest dan Visualisasi Dataset CSV

## 🚀 Fitur-Fitur

- Menampilkan form input untuk klasifikasi
- Memilih satu diantara dua model Random Forest
- Mendapatkan hasil klasifikasi dan probabilitas
- Melatih model Random Forest untuk format csv yang sama (antarmuka belum tersedia)
- Memvisualisasikan dan menyimpan data csv ke beberapa format figur grafik

## 🔧 Instalasi
### 1. Mengekstrak Proyek

Jika dari ZIP:
```bash
unzip random-forest-jinja2.zip
cd random-forest-jinja2.zip
```
Anda juga bisa menggunakan software editor zip.

### 2. Instalasi Dependensi

Pastikan Anda sudah memiliki ekstensi python sebelum instal dependensi.

```bash
pip install -r requirements.txt
```

## 🚀 Cara Menjalankan Proyek
### 1. Menjalankan Proyek

Jalankan di direktori root proyek:
```bash
uvicorn app.main:app --reload
```

Anda juga bisa menambah opsi --host untuk jaringan lokal dan/atau --port untuk menggunakan port yang berbeda:
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8080
```

### 2. Mengakses Proyek Melalui Browser

Melalui browser, Anda bisa akses localhost:8000 atau port yang Anda tentukan sebelumnya saat menjalankan proyek.

## Cara Melatih Model Random Forest
### 1. Menyiapkan Dataset CSV

Langkah-langkah menyiapkan dataset CSV:
- Nama kolom-kolom Dataset CSV harus sama seperti "data\Student Mental health.csv".
- Dataset CSV harus disimpan di folder data.

### 2. Melatih Model Random Forest

Jalankan perintah berikut dari direktori root proyek untuk memulai pelatihan:
```bash
python scripts\klasifikasi.py
```

Skrip ini akan:
- Membersihkan dan memproses data.
- Menggunakan pipeline scikit-learn untuk preprocessing fitur numerik dan kategorikal.
- Melatih model Random Forest menggunakan GridSearchCV dengan Stratified K-Fold cross-validation.
- Mengevaluasi performa model menggunakan metrik akurasi, presisi, recall dan F1-score.

Waktu pelatihan bisa bervariasi tergantung spesifikasi perangkat Anda. Silahkan tunggu hingga proses selesai.

Setelah selesai, hasil evaluasi dan prediksi dummy akan ditampilkan di terminal.

> ⚠️ Preprocessing fitur-fitur kategorikal pipeline dalam skrip ini disesuaikan khusus untuk distribusi nilai unik dataset `Student Mental Health.csv`.
> Silahkan cek dahulu distribusi nilai unik fitur-fitur kategorikal dataset Anda, lalu sesuaikan secara manual preprocessing di file scripts\klasifikasi.py.

### 3. Model Tersimpan di Folder pkl\

Model yang dilatih ada 2:
- nested_cv.pkl

nested_cv adalah model Random Forest yang dihasilkan setelah dievaluasi dengan Nested Cross-Validation dengan menggabungkan cross-validation dan GridSearchCV untuk menghindari bias evaluasi dan data leakage saat memilih hyperparameter.

- best_model.pkl

best_model adalah model Random Forest terbaik yang dipilih otomatis oleh GridSearchCV berdasarkan performa tertinggi.

## Cara Membuat Figur Untuk Visualisasi Data Baru
### 1. Menyiapkan Dataset CSV

Langkah-langkah menyiapkan dataset CSV:
- Nama kolom-kolom dataset CSV harus sama seperti "data\Student Mental health.csv".
- Dataset CSV harus disimpan di folder data.

### 2. Menghasilkan Figur Grafik Visualisasi Dataset CSV

Jalankan perintah berikut dari direktori root proyek untuk memulai menghasilkan figur:
```bash
python scripts\visual.py
```

Skrip ini akan:
- Membersihkan dan memproses data.
- Menghasilkan figur grafik visualisasi dataset csv

### 4. Figur Tersimpan di Folder outputs\figures

8 figur akan disimpan secara otomatis:
- `Distribusi Mahasiswa dengan/tanpa Depresi.png`
- `Indeks Prestasi Kumulatif (IPK) vs Depresi.png`
- `Jenis Kelamin vs Depresi.png`
- `Jumlah Mahasiswa dengan Depresi.png`
- `Rasio Mahasiswa Depresi.png`
- `Status Pernikahan vs Depresi.png`
- `Tahun Perkuliahan vs Depresi.png`
- `Usia vs Depresi.png`

**📜 Lisensi**
Proyek ini disusun untuk keperluan akademik dan tidak digunakan untuk komersial.

**Sekilas tentang file dan folder proyek**
Beberapa folder dan file belum terisi sesuatu yang benar-benar berarti, saya sengaja membiarkannya sebagai pengingat untuk mempelajari yang saya rencanakan sebelumnya.

Beberapa folder dan file yang termasuk dalam konteks tersebut yaitu:
- app\config.py
- app\routes
- notebooks (jupyter)
- semua file dengan nama `__init__.py`

---

Terima kasih sudah menggunakan proyek ini.