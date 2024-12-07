import streamlit as st
import cv2
import numpy as np
from collections import defaultdict, deque
from ultralytics import YOLO
import supervision as sv
from tempfile import NamedTemporaryFile
import torch
import plotly.express as px
import pandas as pd
import os
from datetime import datetime, timedelta
from io import BytesIO
import zipfile

st.set_page_config(
    page_title="Deteksi Kendaraan dan Estimasi Kecepatan - Dengan Statistik",
    page_icon="🚗",
)

st.title("Deteksi Kendaraan dan Estimasi Kecepatan - Dengan Statistik")
st.markdown("""
## Langkah-langkah Menggunakan Aplikasi:

1. **Pilih Model**: Pilih model YOLO yang tersedia dari sidebar.
2. **Upload Video**: Unggah video yang ingin dianalisis.
3. **Pengaturan Input**: 
    - Tentukan lebar dan tinggi target area dalam meter.
    - Pilih lokasi atau masukkan koordinat sumber secara manual.
    - Atur threshold kepercayaan dan IoU.
    - Masukkan waktu mulai video.
4. **Kontrol Video**: Gunakan tombol Start/Reset, Continue, dan Stop untuk mengontrol pemrosesan video.
5. **Lihat Hasil**: 
    - Frame video yang dianotasi akan ditampilkan secara real-time.
    - Statistik kendaraan dan kecelakaan akan diperbarui secara real-time.
    - Grafik kecepatan kendaraan akan ditampilkan.
6. **Unduh Hasil**: Setelah pemrosesan selesai, unduh hasil deteksi dalam format ZIP yang berisi video dan file Excel.

## Catatan:
- Pastikan format koordinat dan waktu yang dimasukkan sesuai dengan format yang ditentukan.
- Gunakan model yang sesuai dengan jenis kendaraan yang ingin dideteksi.
""")
