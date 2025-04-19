import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv
import plotly.express as px
import pandas as pd
from io import BytesIO
import os
import torch
from PIL import Image

def main():
    st.title("Deteksi Objek pada Gambar")

    # Sidebar - Input Model
    st.sidebar.header("Pengaturan")
    model_path = "models"  # Pastikan folder model tersedia
    model_files = [f for f in os.listdir(model_path) if f.endswith('.pt')]
    selected_model = st.sidebar.selectbox("Pilih Model", model_files)

    # Sidebar - Confidence dan IoU Threshold
    confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.3, 0.05)
    iou_threshold = st.sidebar.slider("IoU Threshold", 0.0, 1.0, 0.7, 0.05)

    # Sidebar - Upload Gambar
    uploaded_image = st.sidebar.file_uploader("Upload Gambar", type=["jpg", "jpeg", "png"])

    if uploaded_image:
        # Tampilkan gambar yang diunggah
        st.subheader("Gambar Asli")
        file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        st.image(image, channels="BGR")

        # Convert gambar ke grayscale untuk deteksi
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Konversi grayscale ke RGB (salin channel grayscale ke 3 channel)
        gray_image_rgb = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2RGB)

        height, width, _ = image.shape
        resolution_wh = (width, height)

        # Load model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = YOLO(f'{model_path}/{selected_model}').to(device)

        # Deteksi objek pada gambar grayscale yang sudah dikonversi ke RGB
        results = model(gray_image_rgb)[0]
        detections = sv.Detections.from_ultralytics(results)
        detections = detections[detections.confidence > confidence_threshold]
        detections = detections.with_nms(threshold=iou_threshold)

        thickness = sv.calculate_optimal_line_thickness(resolution_wh)
        text_scale = sv.calculate_optimal_text_scale(resolution_wh)

        # Tambahkan label berdasarkan nama kelas
        labels = [
            f"{model.names[class_id]} ({confidence:.2f})"
            for class_id, confidence in zip(detections.class_id, detections.confidence)
        ]

        # Tambahkan anotasi pada gambar berwarna
        box_annotator = sv.BoxAnnotator(thickness=thickness)
        label_annotator = sv.LabelAnnotator(
            text_scale=text_scale,
            text_thickness=thickness,
            text_position=sv.Position.BOTTOM_CENTER,
        )

        annotated_image = box_annotator.annotate(scene=image_rgb.copy(), detections=detections)
        annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)

        # Tampilkan gambar hasil anotasi
        st.subheader("Gambar dengan Anotasi")
        st.image(annotated_image, channels="RGB")

        # Statistik deteksi
        class_counts = pd.DataFrame(detections.class_id, columns=["Kelas"])
        class_counts["Kelas"] = class_counts["Kelas"].map(model.names)
        counts_df = class_counts["Kelas"].value_counts().reset_index()
        counts_df.columns = ["Kelas", "Jumlah"]

        # Tampilkan grafik
        st.subheader("Grafik Jumlah Objek Terdeteksi")
        fig = px.bar(counts_df, x="Jumlah", y="Kelas", orientation='h', title="Jumlah Deteksi per Kelas")
        st.plotly_chart(fig)

        # Simpan gambar hasil untuk unduhan
        # Convert to PIL Image (RGB)
        pil_image = Image.fromarray(annotated_image)

        # Simpan gambar ke buffer dalam format JPEG
        buffer = BytesIO()
        pil_image.save(buffer, format="JPEG")
        buffer.seek(0)

        # Tombol unduh di sidebar
        st.sidebar.download_button(
            label="Download Gambar Hasil",
            data=buffer,
            file_name="annotated_image.jpg",
            mime="image/jpeg"
        )
