import streamlit as st
from modules import detectImage, detectVideo, detectRealTime

# Fungsi untuk membersihkan session state
def reset_session_state():
    st.session_state.clear()

st.set_page_config(
    page_title="Aplikasi Deteksi Kendaraan dan Kecelakaan",
    page_icon="🚗",
    layout="centered",
    initial_sidebar_state="auto"
)

# Sidebar navigasi
st.sidebar.title("Navigasi")
page = st.sidebar.selectbox(
    "Pilih Mode Deteksi",
    ["Deteksi Gambar", "Deteksi Video", "Deteksi Real-Time"],
    key="current_page"
)

# Deteksi perubahan halaman
if "last_page" not in st.session_state:
    st.session_state.last_page = page  # Inisialisasi halaman awal

if st.session_state.last_page != page:
    reset_session_state()  # Bersihkan session state jika halaman berubah
    st.session_state.last_page = page  # Perbarui halaman terakhir

# Logika untuk menampilkan halaman sesuai pilihan
if page == "Deteksi Gambar":
    detectImage.main()  # Panggil fungsi utama dari modul detectImage
elif page == "Deteksi Video":
    detectVideo.main()  # Panggil fungsi utama dari modul detectVideo
elif page == "Deteksi Real-Time":
    detectRealTime.main()  # Panggil fungsi utama dari modul detectRealTime
