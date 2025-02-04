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
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.videoUtils import save_video
from utils.coordinateUtils import parse_coordinates
from utils.viewTransformer import ViewTransformer

def main():
    # Streamlit UI
    st.title("Deteksi Kendaraan dan Estimasi Kecepatan - Video")

    # Dapatkan daftar model yang tersedia di folder 'models/'
    current_dir = os.path.dirname(os.path.abspath(__file__))

    model_path = os.path.join(current_dir, '..', 'models')
    model_files = [f for f in os.listdir(model_path) if f.endswith('.pt')]
    selected_model = st.sidebar.selectbox("Pilih Model", model_files)

    # Sidebar inputs
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
            st.session_state.frame_index = 0
            st.session_state.vehicle_count = defaultdict(int)
            st.session_state.accident_count = defaultdict(int)
            st.session_state.counted_ids = set()
            st.session_state.last_frame = None
            st.session_state.vehicle_accident_data = []
            st.session_state.annotated_frames = []
            st.session_state.accident_messages = []
            st.session_state.notified_accident_ids = set()

            # Simpan video yang baru diunggah
            st.session_state.last_uploaded_video = uploaded_video

        # Tombol kontrol
        if 'status' not in st.session_state:
            st.session_state.status = 'paused'
        if 'frame_index' not in st.session_state:
            st.session_state.frame_index = 0
        if 'vehicle_count' not in st.session_state:
            st.session_state.vehicle_count = defaultdict(int)
        if 'accident_count' not in st.session_state:
            st.session_state.accident_count = defaultdict(int)
        if 'counted_ids' not in st.session_state:
            st.session_state.counted_ids = set()
        if 'last_frame' not in st.session_state:
            st.session_state.last_frame = None
        if 'vehicle_accident_data' not in st.session_state:
            st.session_state.vehicle_accident_data = []
        if 'annotated_frames' not in st.session_state:
            st.session_state.annotated_frames = []
        if 'accident_messages' not in st.session_state:
            st.session_state.accident_messages = []
        if 'notified_accident_ids' not in st.session_state:
            st.session_state.notified_accident_ids = set()
        
        # Tombol kontrol
        if st.sidebar.button("Start/Reset"):
            st.session_state.status = 'running'
            st.session_state.frame_index = 0
            st.session_state.vehicle_count = defaultdict(int)
            st.session_state.accident_count = defaultdict(int)
            st.session_state.counted_ids = set()
            st.session_state.last_frame = None
            st.session_state.vehicle_accident_data = []
            st.session_state.annotated_frames = []
            st.session_state.accident_messages = []
            st.session_state.notified_accident_ids = set()
        if st.sidebar.button("Continue"):
            st.session_state.status = 'running'
        if st.sidebar.button("Pause"):
            st.session_state.status = 'paused'

        # Input untuk Target Width dan Height
        target_width = st.sidebar.number_input("Target Width (meter)", min_value=1.0, max_value=100.0, value=13.56, step=0.01)
        target_height = st.sidebar.number_input("Target Height (meter)", min_value=1.0, max_value=500.0, value=20.95, step=0.01)

        # Dropdown untuk memilih lokasi
        location = st.sidebar.selectbox(
            "Location",
            ["Simpang Pidada", "Custom"]
        )

        # Tentukan koordinat berdasarkan lokasi yang dipilih
        if location == "Simpang Pidada":
            source_coordinates = "550,394;1032,423;968,717;130,666"
        elif location == "Custom":
            source_coordinates = st.sidebar.text_input(
                "Source Coordinates (format: x1,y1;x2,y2;x3,y3;x4,y4)",
                value="",
                placeholder="Enter custom coordinates"
            )
        else:
            source_coordinates = ""

        # Tampilkan koordinat yang dipilih
        if location != "Custom":
            st.sidebar.text_input(
                "Source Coordinates (format: x1,y1;x2,y2;x3,y3;x4,y4)",
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

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Load YOLO model
            model = YOLO(f'{model_path}/{selected_model}').to(device)  # Ganti dengan model yang dipilih

            # Ambil nama class dari model
            all_classes = model.names.values()
            vehicle_classes = {"bus", "car", "motorcycle", "truck"}
            accident_classes = set(all_classes) - vehicle_classes

            byte_track = sv.ByteTrack(
                frame_rate=video_info.fps,
                track_activation_threshold=confidence_threshold
            )

            thickness = int(sv.calculate_optimal_line_thickness(video_info.resolution_wh) / 2)
            text_scale = sv.calculate_optimal_text_scale(video_info.resolution_wh)
            
            box_annotator = sv.BoxAnnotator(thickness=thickness)
            label_annotator = sv.LabelAnnotator(
                text_scale=text_scale,
                text_thickness=thickness,
                text_position=sv.Position.BOTTOM_CENTER,
            )
            trace_annotator = sv.TraceAnnotator(
                thickness=thickness,
                trace_length=video_info.fps * 2,
                position=sv.Position.BOTTOM_CENTER,
            )

            polygon_zone = sv.PolygonZone(polygon=SOURCE)
            view_transformer = ViewTransformer(source=SOURCE, target=TARGET)
            coordinates = defaultdict(lambda: deque(maxlen=video_info.fps))

            plot_counter = 0  # Counter untuk key yang unik

            frames = list(sv.get_video_frames_generator(source_path=tfile.name))

            while st.session_state.status == 'running' and st.session_state.frame_index < len(frames):
                frame = frames[st.session_state.frame_index]
                st.session_state.frame_index += 1
                
                # Ubah frame ke grayscale
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray_frame_rgb = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)

                result = model(gray_frame_rgb)[0]
                detections = sv.Detections.from_ultralytics(result)
                detections = detections[detections.confidence > confidence_threshold]
                detections = detections[polygon_zone.trigger(detections)]
                detections = detections.with_nms(threshold=iou_threshold)
                detections = byte_track.update_with_detections(detections=detections)

                points = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
                points = view_transformer.transform_points(points=points).astype(int)

                for tracker_id, [_, y], class_id in zip(detections.tracker_id, points, detections.class_id):
                    class_name = model.names[class_id]
                    if tracker_id not in st.session_state.counted_ids:
                        if class_name in vehicle_classes:
                            st.session_state.vehicle_count[class_name] += 1
                        elif class_name in accident_classes:
                            st.session_state.accident_count[class_name] += 1
                        st.session_state.counted_ids.add(tracker_id)  # Tambahkan ID ke set setelah dihitung

                    if class_name in vehicle_classes:
                        coordinates[tracker_id].append(y)

                labels = []
                for tracker_id, class_id in zip(detections.tracker_id, detections.class_id):
                    class_name = model.names[class_id]
                    timestamp = start_time + timedelta(seconds=st.session_state.frame_index / video_info.fps)

                    if class_name in vehicle_classes and len(coordinates[tracker_id]) >= video_info.fps / 2:
                        coordinate_start = coordinates[tracker_id][-1]
                        coordinate_end = coordinates[tracker_id][0]
                        distance = abs(coordinate_start - coordinate_end)
                        time = len(coordinates[tracker_id]) / video_info.fps
                        speed = distance / time * 3.6  # Kecepatan dalam km/h
                        labels.append(f"#{tracker_id} {int(speed)} km/h")
                        st.session_state.vehicle_accident_data.append({
                            "Detik": st.session_state.frame_index / video_info.fps, 
                            "Timestamp": timestamp, 
                            "ID": tracker_id, 
                            "Kelas": class_name, 
                            "Kecepatan (km/h)": speed
                        })

                    else:
                        labels.append(f"#{tracker_id}")

                        if class_name in accident_classes:
                            speed = None
                            st.session_state.vehicle_accident_data.append({
                                "Detik": st.session_state.frame_index / video_info.fps, 
                                "Timestamp": timestamp, 
                                "ID": tracker_id, 
                                "Kelas": class_name, 
                                "Kecepatan (km/h)": speed
                            })

                            if st.session_state.vehicle_accident_data:
                                latest_accident = st.session_state.vehicle_accident_data[-1]  # Ambil data kecelakaan terbaru (elemen terakhir)

                                # Periksa apakah tracker_id sudah ada dalam notified_accident_ids
                                if tracker_id not in st.session_state.notified_accident_ids:
                                    # Format pesan notifikasi
                                    accident_message = (
                                        f"🚨 **Kecelakaan Terdeteksi!**\n\n"
                                        f"🔹 **Detik:** {latest_accident['Detik']:.0f}\n"
                                        f"🕒 **Timestamp:** {latest_accident['Timestamp'].strftime('%Y-%m-%d %H:%M:%S')}\n"
                                        f"🆔 **ID:** {latest_accident['ID']}\n"
                                        f"🚗 **Kelas:** {latest_accident['Kelas']}"
                                    )

                                    # Tambahkan accident_message ke array dalam st.session_state
                                    st.session_state.accident_messages.append(accident_message)

                                    # Tambahkan tracker_id ke notified_accident_ids
                                    st.session_state.notified_accident_ids.add(tracker_id)

                                    # Tampilkan notifikasi kecelakaan
                                    st.toast(accident_message)
                                    st.error(accident_message)

                annotated_frame = frame.copy()

                cv2.polylines(
                    annotated_frame,
                    [SOURCE],
                    isClosed=True,
                    color=(0, 255, 0),
                    thickness=thickness,
                )

                annotated_frame = trace_annotator.annotate(scene=annotated_frame, detections=detections)
                annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections)
                annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)

                # Tambahkan anotasi teks untuk jumlah kendaraan dan kecelakaan
                y_offset = int(30 * video_info.height / 360)
                for vehicle_class in vehicle_classes:
                    count = st.session_state.vehicle_count[vehicle_class]
                    text = f"{vehicle_class.capitalize()}: {count}"
                    (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, text_scale, thickness)
                    cv2.rectangle(annotated_frame, (10, y_offset - text_height - baseline), (10 + text_width, y_offset + baseline), (0, 0, 0), cv2.FILLED)
                    cv2.putText(annotated_frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, text_scale, (255, 255, 255), thickness, cv2.LINE_AA)
                    y_offset += int(20 * video_info.height / 360)
                
                total_accidents = sum(st.session_state.accident_count.values())
                text = f"Accidents: {total_accidents}"
                (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, text_scale, thickness)
                cv2.rectangle(annotated_frame, (10, y_offset - text_height - baseline), (10 + text_width, y_offset + baseline), (0, 0, 0), cv2.FILLED)
                cv2.putText(annotated_frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, text_scale, (255, 255, 255), thickness, cv2.LINE_AA)

                # Tambahkan anotasi teks untuk timestamp
                current_time = start_time + timedelta(seconds=st.session_state.frame_index / video_info.fps)
                timestamp_text = current_time.strftime("%d-%m-%Y %H:%M:%S")
                (text_width, text_height), baseline = cv2.getTextSize(timestamp_text, cv2.FONT_HERSHEY_SIMPLEX, text_scale, thickness)
                cv2.rectangle(annotated_frame, (10, y_offset + int(20 * video_info.height / 360) - text_height - baseline), (10 + text_width, y_offset + int(20 * video_info.height / 360) + baseline), (0, 0, 0), cv2.FILLED)
                cv2.putText(annotated_frame, timestamp_text, (10, y_offset + int(20 * video_info.height / 360)), cv2.FONT_HERSHEY_SIMPLEX, text_scale, (255, 255, 255), thickness, cv2.LINE_AA)

                # Convert annotated frame to RGB for display
                annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                st.session_state.annotated_frames.append(annotated_frame)  # Add frame to list

                # Display frame
                stframe.image(annotated_frame, channels="RGB")
                st.session_state.last_frame = annotated_frame  # Simpan frame terakhir

                # Update real-time statistics
                vehicle_stats_df = pd.DataFrame(list(st.session_state.vehicle_count.items()), columns=["Kelas", "Jumlah"])
                accident_stats_df = pd.DataFrame(list(st.session_state.accident_count.items()), columns=["Kelas", "Jumlah"])

                # Filter DataFrame untuk menghapus baris dengan jumlah 0
                vehicle_stats_df = vehicle_stats_df[vehicle_stats_df["Jumlah"] > 0]
                accident_stats_df = accident_stats_df[accident_stats_df["Jumlah"] > 0]

                # Tampilkan grafik hanya jika ada data kendaraan
                if not vehicle_stats_df.empty:
                    with vehicle_stats_placeholder.container():
                        st.subheader("Statistik Kendaraan")
                        fig_vehicle = px.bar(vehicle_stats_df, x="Jumlah", y="Kelas", orientation='h', title="Jumlah Kendaraan per Kelas")
                        st.plotly_chart(fig_vehicle, use_container_width=True, key=f"vehicle_stats_{plot_counter}")
                        plot_counter += 1  # Tingkatkan counter untuk key yang unik

                # Tampilkan grafik hanya jika ada data kecelakaan
                if not accident_stats_df.empty:
                    with accident_stats_placeholder.container():
                        st.subheader("Statistik Kecelakaan")
                        fig_accident = px.bar(accident_stats_df, x="Jumlah", y="Kelas", orientation='h', title="Jumlah Kecelakaan per Kelas")
                        st.plotly_chart(fig_accident, use_container_width=True, key=f"accident_stats_{plot_counter}")
                        plot_counter += 1  # Tingkatkan counter untuk key yang unik

                # Update real-time vehicle speed graph
                if st.session_state.vehicle_accident_data:
                    df = pd.DataFrame(st.session_state.vehicle_accident_data)
                    
                    # Filter hanya untuk kelas yang termasuk dalam vehicle_classes
                    df = df[df["Kelas"].isin(vehicle_classes)]
                    
                    if not df.empty:  # Pastikan tidak membuat grafik jika tidak ada data kendaraan
                        # Hitung rata-rata kecepatan per kelas
                        avg_speed_df = df.groupby("Kelas")["Kecepatan (km/h)"].mean().reset_index()
                        avg_speed_df.columns = ["Kelas", "Rata-Rata Kecepatan (km/h)"]
                        
                        # Buat grafik rata-rata kecepatan
                        fig_speed = px.bar(
                            avg_speed_df, x="Rata-Rata Kecepatan (km/h)", y="Kelas", 
                            orientation='h', title="Rata-rata Kecepatan Kendaraan per Kelas"
                        )
                        
                        # Perbarui grafik di placeholder
                        vehicle_speed_placeholder.empty()  # Kosongkan placeholder
                        vehicle_speed_placeholder.plotly_chart(fig_speed, key=f"vehicle_speed_{plot_counter}")
                        plot_counter += 1  # Tingkatkan counter untuk key yang unik
            
            st.session_state.status == 'paused'

            # Save the video and Excel files and provide a download button for the ZIP file
            if st.session_state.annotated_frames:
                # Menentukan jalur absolut dari direktori saat ini
                video_file_path = os.path.join(current_dir, '..', 'outputTest', 'output_video.mp4')

                save_video(st.session_state.annotated_frames, video_file_path, video_info.fps)

                # Buat DataFrame untuk kecepatan setiap ID terdeteksi
                vehicle_accident_df = pd.DataFrame(st.session_state.vehicle_accident_data)

                # Ambil semua nama kelas dari model
                all_classes = list(model.names.values())

                # Buat DataFrame untuk jumlah kendaraan dan kecelakaan
                vehicle_count_df = pd.DataFrame(list(st.session_state.vehicle_count.items()), columns=["Kelas", "Count_vehicle"])
                accident_count_df = pd.DataFrame(list(st.session_state.accident_count.items()), columns=["Kelas", "Count_accident"])

                # Gabungkan semua kelas kendaraan dan kecelakaan dengan nilai 0 untuk yang tidak terdeteksi
                vehicle_count_df = vehicle_count_df.set_index("Kelas").reindex(all_classes, fill_value=0).reset_index()
                accident_count_df = accident_count_df.set_index("Kelas").reindex(all_classes, fill_value=0).reset_index()

                # Gabungkan DataFrame kendaraan dan kecelakaan
                count_df = pd.merge(vehicle_count_df, accident_count_df, on="Kelas", how="outer", suffixes=("_vehicle", "_accident"))

                # Gabungkan kedua kolom Count menjadi satu kolom "Jumlah"
                count_df["Jumlah"] = count_df["Count_vehicle"] + count_df["Count_accident"]

                # Drop kolom tambahan (Count_vehicle dan Count_accident)
                count_df = count_df.drop(columns=["Count_vehicle", "Count_accident"])

                # Hapus duplikasi jika ada
                count_df = count_df.drop_duplicates(subset=["Kelas"])

                # Hitung rata-rata kecepatan per kelas dan gabungkan dengan count_df
                avg_speed_df = vehicle_accident_df.groupby("Kelas")["Kecepatan (km/h)"].mean().reset_index()
                avg_speed_df.columns = ["Kelas", "Rata-Rata Kecepatan (km/h)"]
                count_df = pd.merge(count_df, avg_speed_df, on="Kelas", how="left")
                
                # Tulis kedua DataFrame ke dalam satu file Excel dengan dua sheet
                output = BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    vehicle_accident_df.to_excel(writer, index=False, sheet_name='Kejadian')
                    count_df.to_excel(writer, index=False, sheet_name='Rangkuman')
                excel_data = output.getvalue()

                # Create a ZIP file containing both the video and Excel files
                zip_buffer = BytesIO()
                with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
                    zip_file.write(video_file_path, os.path.basename(video_file_path))
                    zip_file.writestr('hasil_deteksi.xlsx', excel_data)

                # Provide a download button for the ZIP file
                st.sidebar.download_button(
                    label="Download Hasil Deteksi",
                    data=zip_buffer.getvalue(),
                    file_name='output_files.zip',
                    mime='application/zip'
                )

                # Delete the video file after download
                os.remove(video_file_path)


        # Tampilkan frame terakhir dan grafik saat status 'paused'
        if st.session_state.status == 'paused' and st.session_state.last_frame is not None:
            stframe.image(st.session_state.last_frame, channels="RGB")

            vehicle_stats_df = pd.DataFrame(list(st.session_state.vehicle_count.items()), columns=["Kelas", "Jumlah"])
            accident_stats_df = pd.DataFrame(list(st.session_state.accident_count.items()), columns=["Kelas", "Jumlah"])

            # Filter DataFrame untuk menghapus baris dengan jumlah 0
            vehicle_stats_df = vehicle_stats_df[vehicle_stats_df["Jumlah"] > 0]
            accident_stats_df = accident_stats_df[accident_stats_df["Jumlah"] > 0]

            with vehicle_stats_placeholder.container():
                st.subheader("Statistik Kendaraan")
                fig_vehicle = px.bar(vehicle_stats_df, x="Jumlah", y="Kelas", orientation='h', title="Jumlah Kendaraan per Kelas")
                st.plotly_chart(fig_vehicle, use_container_width=True, key=f"vehicle_stats_{plot_counter}")

            with accident_stats_placeholder.container():
                st.subheader("Statistik Kecelakaan")
                fig_accident = px.bar(accident_stats_df, x="Jumlah", y="Kelas", orientation='h', title="Jumlah Kecelakaan per Kelas")
                st.plotly_chart(fig_accident, use_container_width=True, key=f"accident_stats_{plot_counter}")

            if st.session_state.vehicle_accident_data:
                df = pd.DataFrame(st.session_state.vehicle_accident_data)
                
                # Filter hanya untuk kelas yang termasuk dalam vehicle_classes
                df = df[df["Kelas"].isin(vehicle_classes)]
                
                if not df.empty:  # Pastikan tidak membuat grafik jika tidak ada data kendaraan
                    # Hitung rata-rata kecepatan per kelas
                    avg_speed_df = df.groupby("Kelas")["Kecepatan (km/h)"].mean().reset_index()
                    avg_speed_df.columns = ["Kelas", "Rata-Rata Kecepatan (km/h)"]
                    
                    # Buat grafik rata-rata kecepatan
                    fig_speed = px.bar(
                        avg_speed_df, x="Rata-Rata Kecepatan (km/h)", y="Kelas", 
                        orientation='h', title="Rata-rata Kecepatan Kendaraan per Kelas"
                    )
                    
                    # Perbarui grafik di placeholder
                    vehicle_speed_placeholder.empty()  # Kosongkan placeholder
                    vehicle_speed_placeholder.plotly_chart(fig_speed, key=f"vehicle_speed_{plot_counter}")
                    plot_counter += 1  # Tingkatkan counter untuk key yang unik
            
            # Tampilkan semua accident_messages yang disimpan dalam st.session_state
            for message in st.session_state.accident_messages:
                st.error(message)
