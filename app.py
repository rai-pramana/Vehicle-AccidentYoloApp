import streamlit as st
from modules import detectImage, detectVideo, detectRealTime

# Sidebar navigasi
st.sidebar.title("Navigasi")
page = st.sidebar.selectbox(
    "Pilih Mode Deteksi",
    ["Deteksi Gambar", "Deteksi Video", "Deteksi Real-Time"]
)

# Logika untuk menampilkan halaman sesuai pilihan
if page == "Deteksi Gambar":
    detectImage.main()  # Panggil fungsi utama dari modul detectPhoto
elif page == "Deteksi Video":
    detectVideo.main()  # Panggil fungsi utama dari modul detectVideo
elif page == "Deteksi Real-Time":
    detectRealTime.main()  # Panggil fungsi utama dari modul detectRealTime
