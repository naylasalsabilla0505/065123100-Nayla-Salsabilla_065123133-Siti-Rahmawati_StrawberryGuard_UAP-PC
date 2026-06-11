# StrawberryGuard

StrawberryGuard adalah aplikasi web berbasis Flask yang digunakan untuk mendeteksi penyakit pada daun stroberi secara real-time menggunakan metode Machine Learning. Sistem dapat mengklasifikasikan kondisi daun ke dalam tiga kategori, yaitu Healthy, Leaf Spot, dan Powdery Mildew.

## Fitur Utama

- Upload gambar daun stroberi
- Deteksi penyakit daun secara otomatis
- Menampilkan hasil klasifikasi beserta rekomendasi penanganan
- Antarmuka web yang sederhana dan mudah digunakan
- Implementasi dua model Machine Learning:
  - K-Nearest Neighbor (KNN)
  - Naive Bayes (NB)

## Kategori Klasifikasi

| Kelas | Deskripsi |
|---------|---------|
| Healthy | Daun stroberi dalam kondisi sehat |
| Leaf Spot | Penyakit bercak daun akibat infeksi jamur |
| Powdery Mildew | Penyakit embun tepung pada daun stroberi |

## Metode yang Digunakan

### Preprocessing
- Median Filter untuk reduksi noise

### Segmentasi Citra
- HSV Color Segmentation

### Ekstraksi Fitur
- Fitur warna HSV
- Fitur tekstur GLCM (Gray Level Co-occurrence Matrix)

### Klasifikasi
- K-Nearest Neighbor (KNN)
- Naive Bayes (NB)

## Teknologi yang Digunakan

- Python
- Flask
- OpenCV
- NumPy
- Scikit-image
- HTML
- CSS
- JavaScript

## Struktur Project

```text
strawberry-detection/
│
├── app.py
├── model_knn.pkl
├── model_nb.pkl
├── requirements.txt
│
├── static/
│   ├── css/
│   │   └── style.css
│   └── js/
│       └── main.js
│
└── templates/
    └── index.html
```

## Instalasi

### 1. Clone Repository

```bash
git clone https://github.com/naylasalsabilla0505/065123100-Nayla-Salsabilla_065123133-Siti-Rahmawati_StrawberryGuard_UAP-PC.git
```

### 2. Masuk ke Folder Project

```bash
cd strawberry-detection
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Jalankan Aplikasi

```bash
python app.py
```

### 5. Buka Browser

```text
http://127.0.0.1:5000
```

## Cara Penggunaan

1. Buka aplikasi melalui browser.
2. Upload gambar daun stroberi.
3. Sistem akan melakukan preprocessing dan ekstraksi fitur.
4. Model Machine Learning melakukan klasifikasi.
5. Hasil prediksi dan rekomendasi penanganan akan ditampilkan.

## Anggota Kelompok

- Nayla Salsabilla (065123100)
- Siti Rahmawati (065123133)

## Mata Kuliah

Ujian Akhir Praktikum - Pengolahan Citra 
