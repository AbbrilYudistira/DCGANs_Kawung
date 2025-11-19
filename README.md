# Generator Batik Kawung dengan DCGAN

Aplikasi web ini menggunakan *Deep Convolutional Generative Adversarial Network* (DCGAN) untuk menghasilkan gambar motif Batik Kawung. Aplikasi ini dibangun dengan Streamlit dan TensorFlow, memungkinkan pengguna untuk membuat gambar batik baru secara acak atau mengunggah gambar untuk dianalisis oleh model diskriminator.

Model yang digunakan dilatih berdasarkan notebook `batik_kawung_FID300.ipynb`.

## Fitur

* **Generator**: Menghasilkan gambar motif Batik Kawung baru dari vektor laten acak.
* **Diskriminator**: Menganalisis gambar yang diunggah pengguna untuk memprediksi apakah gambar tersebut asli (berdasarkan dataset latihan) atau palsu.
* **Antarmuka Web**: Tampilan yang sederhana dan interaktif menggunakan Streamlit.

## Tampilan Aplikasi

![Contoh Tampilan Aplikasi](https://i.imgur.com/placeholder.png) ## Instalasi

Untuk menjalankan aplikasi ini di lingkungan lokal Anda, ikuti langkah-langkah berikut.

**1. Prasyarat**
* Python 3.8 atau yang lebih baru.
* `pip` untuk manajemen paket.
* (Opsional tapi direkomendasikan) Virtual environment seperti `venv` atau `conda`.

**2. Kloning atau Unduh Proyek**
Unduh semua file proyek (`app.py`, `generator_final.h5`, `discriminator_final.h5`) ke dalam satu direktori.

**3. Buat Virtual Environment (Direkomendasikan)**
```bash
# Buat environment
python -m venv venv

# Aktifkan di Windows
venv\Scripts\activate

# Aktifkan di macOS/Linux
source venv/bin/activate
```

**4. Instal Dependensi**
Instal semua pustaka Python yang dibutuhkan menggunakan file `requirements.txt`.
```bash
pip install -r requirements.txt
```

## Cara Menjalankan Aplikasi

1.  Pastikan Anda berada di direktori utama proyek.
2.  Pastikan model `generator_final.h5` dan `discriminator_final.h5` berada di direktori yang sama dengan `app.py`.
3.  Jalankan perintah berikut di terminal Anda:
    ```bash
    streamlit run app.py
    ```
4.  Aplikasi akan terbuka secara otomatis di browser web Anda.

## Struktur Proyek

```
.
├── app.py                  # Kode utama aplikasi Streamlit
├── generator_final.h5      # Model generator yang sudah dilatih
├── discriminator_final.h5  # Model diskriminator yang sudah dilatih
├── requirements.txt        # Daftar pustaka Python yang dibutuhkan
└── README.md               # File ini
```

## Requirements

Berikut adalah daftar pustaka yang diperlukan untuk menjalankan proyek ini.

* **streamlit**: Untuk membangun antarmuka aplikasi web.
* **tensorflow**: Framework utama untuk memuat dan menjalankan model machine learning.
* **numpy**: Untuk operasi numerik dan manipulasi array.
* **Pillow**: Untuk pemrosesan gambar.
* **scipy**: (Digunakan dalam notebook pelatihan) Diperlukan untuk beberapa perhitungan ilmiah.
* **matplotlib**: (Digunakan dalam notebook pelatihan) Untuk visualisasi data.
