import streamlit as st
import cv2
import numpy as np
import os
from datetime import datetime, timedelta
import sys
import time as t

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.coordinateUtils import parse_coordinates
from utils.saveOutputs import save_and_provide_download_button
from utils.displayStatistics import update_real_time_statistics
from utils.state import reset_state, init_state
from utils.components import load_model_and_initialize_components
from utils.detect import process_frame

def main():
    # Streamlit UI
    st.title("Deteksi Kendaraan dan Estimasi Kecepatan - Webcam")

    # Dapatkan daftar model yang tersedia di folder 'models/'
    current_dir = os.path.dirname(os.path.abspath(__file__))

    model_path = os.path.join(current_dir, '..', 'models')
    model_files = [f for f in os.listdir(model_path) if f.endswith('.pt')]
    selected_model = st.sidebar.selectbox("Pilih Model", model_files)

    # Mengatur pengambilan gambar dari webcam
    camera = cv2.VideoCapture(1)  # Gunakan webcam dengan ID 1 atau 2 (Kamera Virtual OBS)
    if not camera.isOpened():
        st.error("Webcam tidak tersedia")
        return

    frame_width = 640
    frame_height = 360
    resolution_wh = (frame_width, frame_height)

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    # Dapatkan FPS dari webcam
    # FPS = camera.get(cv2.CAP_PROP_FPS) #5
    FPS = 25 / 2.2  # Simpang Pidada
    # FPS = 25 / 1.88 # Batubulan

    stframe = st.empty()  # Tempat untuk menampilkan frame
    vehicle_stats_placeholder = st.empty()  # Placeholder untuk statistik kendaraan
    vehicle_speed_placeholder = st.empty()  # Placeholder untuk grafik kecepatan kendaraan
    accident_stats_placeholder = st.empty()  # Placeholder untuk statistik kecelakaan
    
    # Input sidebar
    st.sidebar.header("Pengaturan Input")
    
    # Tombol kontrol
    if 'status' not in st.session_state:
        st.session_state.status = 'stopped'
    
    init_state()
    
    # Tombol kontrol
    if st.sidebar.button("Mulai"):
        st.session_state.status = 'running'
    if st.sidebar.button("Jeda"):
        st.session_state.status = 'paused'
    if st.sidebar.button("Berhenti"):
        st.session_state.status = 'stopped'

    # Input untuk Target Width dan Height
    target_width = st.sidebar.number_input("Lebar Target (meter)", min_value=1.0, max_value=100.0, value=13.56, step=0.01)
    target_height = st.sidebar.number_input("Panjang Target (meter)", min_value=1.0, max_value=500.0, value=20.95, step=0.01)
    
    # Dropdown untuk memilih lokasi
    location = st.sidebar.selectbox(
        "Lokasi",
        ["Simpang Pidada", "Batubulan", "Fullscreen 360p", "Kustom"]
    )

    # Tentukan koordinat berdasarkan lokasi yang dipilih
    if location == "Simpang Pidada":
        source_coordinates = "290,197;516,211;484,358;100,333"
    elif location == "Batubulan":
        source_coordinates = "310,52;406,45;544,342;315,358"
    elif location == "Kustom":
        source_coordinates = st.sidebar.text_input(
            "Koordinat Sumber (format: x1,y1;x2,y2;x3,y3;x4,y4)",
            value="",
            placeholder="Masukkan koordinat kustom"
        )
    elif location == "Fullscreen 360p":
        source_coordinates = "0,0;640,0;640,360;0,360"
    else:
        source_coordinates = "0,0;640,0;640,360;0,360"

    # Tampilkan koordinat yang dipilih
    if location != "Kustom":
        st.sidebar.text_input(
            "Koordinat Sumber (format: x1,y1;x2,y2;x3,y3;x4,y4)",
            value=source_coordinates,
            disabled=True
        )

    # Parse Source Coordinates
    SOURCE = parse_coordinates(source_coordinates)
    TARGET = np.array([[0, 0], [target_width - 1, 0], [target_width - 1, target_height - 1], [0, target_height - 1]])

    # Confidence dan IoU Threshold
    confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.3, 0.05)
    iou_threshold = st.sidebar.slider("IoU Threshold", 0.0, 1.0, 0.7, 0.05)

    # Input untuk waktu mulai video
    start_time_str = st.sidebar.text_input("Waktu Mulai Video (dd-mm-yyyy hh:mm:ss)", value="27-06-2023 06:40:10")

    # Parse waktu mulai video
    try:
        start_time = datetime.strptime(start_time_str, "%d-%m-%Y %H:%M:%S")
    except ValueError:
        st.error("Format waktu tidak valid. Gunakan format: dd-mm-yyyy hh:mm:ss")
        start_time = None

    start_processing_time = t.time()

    model, byte_track, box_annotator, label_annotator, trace_annotator, polygon_zone, view_transformer, coordinates, vehicle_classes, accident_classes, thickness, text_scale = load_model_and_initialize_components(model_path, selected_model, resolution_wh, confidence_threshold, SOURCE, TARGET, FPS)

    plot_counter = 0  # Counter untuk key yang unik

    while st.session_state.status == 'running':
        ret, frame = camera.read()
        if not ret:
            st.error("Gagal membaca frame dari webcam")
            break

        elapsed_time = t.time() - start_processing_time
        current_time = start_time + timedelta(seconds=elapsed_time)
        
        process_frame(
            frame, 
            model, 
            byte_track, 
            polygon_zone, 
            view_transformer, 
            coordinates, 
            vehicle_classes, 
            accident_classes, 
            start_time, 
            elapsed_time, 
            current_time, 
            frame_height, 
            box_annotator, 
            label_annotator, 
            trace_annotator, 
            stframe, 
            thickness, 
            FPS, 
            SOURCE, 
            confidence_threshold, 
            iou_threshold, 
            text_scale
        )

        plot_counter = update_real_time_statistics(
            vehicle_stats_placeholder, 
            accident_stats_placeholder, 
            vehicle_speed_placeholder, 
            plot_counter, 
            vehicle_classes
        )

    # Rilis webcam setelah selesai
    camera.release()

    # Simpan file video dan Excel dan sediakan tombol unduh untuk file ZIP
    if st.session_state.annotated_frames and st.session_state.status == 'stopped':        
        save_and_provide_download_button(current_dir, FPS, model)

        reset_state()
    
    # Tampilkan frame terakhir dan grafik saat status 'paused'
    if st.session_state.status == 'paused' and st.session_state.last_frame is not None:
        stframe.image(st.session_state.last_frame, channels="RGB")

        plot_counter = update_real_time_statistics(
            vehicle_stats_placeholder, 
            accident_stats_placeholder, 
            vehicle_speed_placeholder, 
            plot_counter, 
            vehicle_classes
        )
        
        # Tampilkan semua accident_messages yang disimpan dalam st.session_state
        for message in st.session_state.accident_messages:
            st.error(message)
