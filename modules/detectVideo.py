import streamlit as st
import numpy as np
import supervision as sv
from tempfile import NamedTemporaryFile
import os
from datetime import datetime, timedelta
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.coordinateUtils import parse_coordinates
from utils.saveOutputs import save_and_provide_download_button
from utils.displayStatistics import update_real_time_statistics
from utils.state import reset_state, init_state
from utils.components import load_model_and_initialize_components
from utils.detect import process_frame

def main():
    # Streamlit UI
    st.title("Deteksi Kendaraan dan Kecelakaan - Video")

    # Dapatkan daftar model yang tersedia di folder 'models/'
    current_dir = os.path.dirname(os.path.abspath(__file__))

    model_path = os.path.join(current_dir, '..', 'models')
    model_files = [f for f in os.listdir(model_path) if f.endswith('.pt')]
    selected_model = st.sidebar.selectbox("Pilih Model", model_files)

    # Input sidebar
    st.sidebar.header("Pengaturan Input")

    # Upload video
    uploaded_video = st.sidebar.file_uploader("Upload Video", type=["mp4", "avi", "mov", "mkv"])

    if uploaded_video is not None:
        stframe = st.empty()  # Tempat untuk menampilkan frame
        vehicle_stats_placeholder = st.empty()  # Placeholder untuk statistik kendaraan
        vehicle_speed_placeholder = st.empty()  # Placeholder untuk grafik kecepatan kendaraan
        accident_stats_placeholder = st.empty()  # Placeholder untuk statistik kecelakaan
        
        if 'last_uploaded_video' not in st.session_state or st.session_state.last_uploaded_video != uploaded_video:
            # Reset state jika video baru diunggah
            st.session_state.status = 'paused'
            reset_state()

            # Simpan video yang baru diunggah
            st.session_state.last_uploaded_video = uploaded_video

        # Tombol kontrol
        if 'status' not in st.session_state:
            st.session_state.status = 'paused'
        
        init_state()
        
        # Tombol kontrol
        if st.sidebar.button("Mulai/Ulang"):
            st.session_state.status = 'running'
            reset_state()
        if st.sidebar.button("Lanjut"):
            st.session_state.status = 'running'
        if st.sidebar.button("Jeda"):
            st.session_state.status = 'paused'

        # Input untuk Target Width dan Height
        target_width = st.sidebar.number_input("Lebar Target (meter)", min_value=1.0, max_value=100.0, value=13.56, step=0.01)
        target_height = st.sidebar.number_input("Panjang Target (meter)", min_value=1.0, max_value=500.0, value=20.95, step=0.01)

        # Dropdown untuk memilih lokasi
        location = st.sidebar.selectbox(
            "Lokasi",
            ["Simpang Pidada", "Batubulan", "Fullscreen 720p", "Kustom"]
        )

        # Tentukan koordinat berdasarkan lokasi yang dipilih
        if location == "Simpang Pidada":
            source_coordinates = "550,394;1032,423;968,717;130,666"
        elif location == "Batubulan":
            source_coordinates = "620,104;812,90;1089,684;630,716"
        elif location == "Kustom":
            source_coordinates = st.sidebar.text_input(
                "Koordinat Sumber (format: x1,y1;x2,y2;x3,y3;x4,y4)",
                value="",
                placeholder="Masukkan koordinat kustom"
            )
        elif location == "Fullscreen 720p":
            source_coordinates = "0,0;1280,0;1280,720;0,720"
        else:
            source_coordinates = "0,0;1280,0;1280,720;0,720"

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

        if SOURCE is not None and start_time is not None:
            tfile = NamedTemporaryFile(delete=False)
            tfile.write(uploaded_video.read())

            video_info = sv.VideoInfo.from_video_path(video_path=tfile.name)
            frame_height = video_info.height
            resolution_wh = video_info.resolution_wh
            FPS = video_info.fps

            model, byte_track, box_annotator, label_annotator, trace_annotator, polygon_zone, view_transformer, coordinates, vehicle_classes, accident_classes, thickness, text_scale = load_model_and_initialize_components(model_path, selected_model, resolution_wh, confidence_threshold, SOURCE, TARGET, FPS)

            plot_counter = 0  # Counter untuk key yang unik

            frames = list(sv.get_video_frames_generator(source_path=tfile.name))

            while st.session_state.status == 'running' and st.session_state.frame_index < len(frames):
                frame = frames[st.session_state.frame_index]
                st.session_state.frame_index += 1

                elapsed_time = st.session_state.frame_index / FPS
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
            
            st.session_state.status == 'paused'

            # Simpan file video dan Excel dan sediakan tombol unduh untuk file ZIP
            if st.session_state.annotated_frames:
                save_and_provide_download_button(current_dir, FPS, model)

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
