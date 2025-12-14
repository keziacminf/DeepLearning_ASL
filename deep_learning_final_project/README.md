# ASL Hand Sign Recognition: Real-time System

[![GitHub language](https://img.shields.io/github/languages/top/keziacminf/DeepLearning_ASL?style=flat-square)](https://github.com/keziacminf/DeepLearning_ASL)

---

## 1. Overview Project

Project ini mengimplementasikan sistem pengenalan isyarat tangan ASL (American Sign Language) secara _end-to-end_. Menggunakan arsitektur Deep Learning untuk klasifikasi gestur dan diintegrasikan dengan WebRTC untuk demonstrasi _real-time_ via _webcam_ (Real-time Webcam).

- **Tujuan:** Klasifikasi 28 isyarat tangan ASL (A-Z, DELETE, SPACE).
- **Aplikasi:** Interaktif Streamlit App yang menyajikan demo _live_ dan detail evaluasi.

---

## 2. Metodologi & Arsitektur

### 2.1. Dataset

- **Sumber:** Mendeley ASL Alphabet Dataset.
- **Ukuran:** Total **11.200** gambar.
- **Kelas:** 28 Kelas (A-Z, DELETE, SPACE).

### 2.2. Model Inti

- **Arsitektur:** **MobileNetV2** (Transfer Learning).
- **Framework:** TensorFlow / Keras.
- **Strategi:** Pelatihan dua fase (Fine-tuning).

---

## 3. Hasil Kunci & Performa

Model mencapai akurasi yang teruji pada _Test Set_ yang belum pernah dilihat:

| Metrik    | Validation Acc. | Validation Loss | Test Accuracy | Test Loss  |
| :-------- | :-------------- | :-------------- | :------------ | :--------- |
| **Nilai** | **88.48%**      | **0.4315**      | **86.61%**    | **0.5121** |

### 3.1. Metrik Per Kelas

Model menunjukkan performa seimbang di seluruh kelas:

- **Macro Avg F1-Score:** 0.8671
- **Weighted Avg F1-Score:** 0.8671

➡️ **Lihat Visualisasi:** Plot History Pelatihan (Loss & Accuracy) dan Confusion Matrix lengkap tersedia di mode **About** dalam aplikasi.

---

## 4. Panduan Eksekusi Lokal

### 4.1. Instalasi

Clone repositori ini dan install semua dependensi dari `requirements.txt`:

```bash
# 1. Clone repositori
git clone [https://github.com/keziacminf/DeepLearning_ASL.git](https://github.com/keziacminf/DeepLearning_ASL.git)
cd DeepLearning_ASL

# 2. Instalasi
pip install -r requirements.txt
```

### 4.2. Jalankan Aplikasi

Jalankan perintah berikut. Aplikasi akan otomatis terbuka di browser.

```bash
streamlit run app.py
```
