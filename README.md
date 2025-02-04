# Vehicle-AccidentYoloApp

## 🚗 Tentang Proyek

**Vehicle-AccidentYoloApp** adalah aplikasi berbasis **YOLO** yang digunakan untuk **deteksi kendaraan dan kecelakaan** dari video atau secara **real-time**. Aplikasi ini dapat mengidentifikasi berbagai jenis kendaraan, menghitung kecepatan kendaraan, serta memberikan peringatan ketika terjadi kecelakaan.

Proyek ini dideploy secara **lokal** pada perangkat pengguna untuk menghindari biaya tinggi dalam menyewa sumber daya GPU secara online.

---

## 🖥️ Persyaratan Sistem

Untuk menjalankan aplikasi ini dengan optimal, spesifikasi perangkat yang disarankan adalah:

### **Hardware:**

-   **CPU**: AMD Ryzen 5 5600 (6 Core, 12 Thread, 3.5 GHz - 4.4 GHz)
-   **GPU**: NVIDIA GeForce RTX 3060 (8GB GDDR6, CUDA Core 3584)
-   **RAM**: 16GB (Dual Channel, 3200MHz)
-   **Storage**: SSD NVMe 500GB (PCIe Gen 3x4)
-   **OS**: Windows 11 (versi 24H2)

### **Software & Library:**

-   **Python**: 3.12.7
-   **Code Editor**: Visual Studio Code (versi 1.96.4)
-   **Library yang diperlukan:**
    -   NumPy (2.2.2)
    -   OpenCV (4.10.0.84)
    -   Pandas (2.2.3)
    -   Pillow (11.1.0)
    -   Plotly (5.24.1)
    -   Streamlit (1.40.2)
    -   Supervision (0.25.1)
    -   Torch (2.5.1)
    -   Ultralytics (8.3.40)

---

## 📦 Instalasi

Ikuti langkah-langkah berikut untuk menginstal dan menjalankan aplikasi:

### **1. Clone Repository**

```sh
git clone https://github.com/rai-pramana/Vehicle-AccidentYoloApp.git
cd Vehicle-AccidentYoloApp
```

### **2. Buat Virtual Environment (Opsional, tapi disarankan)**

```sh
python -m venv venv
source venv/bin/activate  # Untuk Linux/Mac
venv\Scripts\activate  # Untuk Windows
```

### **3. Install Dependencies**

```sh
pip install -r requirements.txt
```

Jika `requirements.txt` belum ada, bisa diinstal satu per satu:

```sh
pip install numpy opencv-python pandas pillow plotly streamlit supervision torch ultralytics
```

### **4. Install CUDA**

Untuk instalasi NVIDIA CUDA, silahkan ikuti tutorial berikut [Setup-NVIDIA-GPU-for-Deep-Learning](https://github.com/entbappy/Setup-NVIDIA-GPU-for-Deep-Learning)

### **5. Jalankan Aplikasi**

```sh
streamlit run app.py
```

Akses aplikasi melalui browser di **http://localhost:8501**.

---

## 🔧 Cara Penggunaan

1. **Pilih mode deteksi**:
    - **Deteksi Video**: Memproses semua frame dari video input untuk mendapatkan hasil yang lebih akurat.
    - **Deteksi Real-Time**: Menggunakan webcam untuk deteksi langsung dengan FPS terbatas berdasarkan kemampuan perangkat.
2. **Input video atau nyalakan webcam**.
3. **Aplikasi akan menampilkan hasil deteksi, kecepatan kendaraan, dan peringatan kecelakaan** jika terdeteksi.
4. **Data kecepatan dan kecelakaan akan dicatat dalam laporan**.

---

## ⚡ Perbedaan Fitur Deteksi Video vs. Real-Time

-   **Deteksi Video**:
    -   Memproses semua frame dalam video.
    -   Lebih akurat dalam estimasi kecepatan.
    -   Waktu pemrosesan lebih lama.
-   **Deteksi Real-Time**:
    -   Memproses lebih sedikit frame (pada perangkat ini hanya 8.3 FPS).
    -   Lebih cepat karena ada frame skipping.
    -   Akurasi estimasi kecepatan lebih rendah dibanding deteksi video.

---

## 📌 Catatan

Aplikasi ini **di-deploy secara lokal** karena menjalankan model deteksi kendaraan dan kecelakaan membutuhkan **sumber daya GPU** yang tinggi, yang jika dilakukan secara online akan memerlukan biaya besar.

---

## 📜 Lisensi

Proyek ini menggunakan lisensi **MIT License**. Silakan gunakan dan kembangkan sesuai kebutuhan.

---

## 📬 Kontak

Jika ada pertanyaan atau kendala, silakan hubungi:

-   **GitHub**: [Vehicle-AccidentYoloApp](https://github.com/rai-pramana/Vehicle-AccidentYoloApp)
-   **Email**: rai.pramana46@gmail.com

---

🚀 Selamat mencoba! Semoga aplikasi ini bermanfaat untuk deteksi kendaraan dan kecelakaan secara efektif. 😊
