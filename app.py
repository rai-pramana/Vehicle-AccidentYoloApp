import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import tempfile
import torch

def get_text_color(bg_color):
    luminance = (0.299 * bg_color[2] + 0.587 * bg_color[1] + 0.114 * bg_color[0])
    return (0, 0, 0) if luminance > 128 else (255, 255, 255)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load YOLO model
model = YOLO('models/vehicle-accident.pt').to(device)  # Ganti dengan model Anda

class_names = model.names
class_colors = {
    0: (255, 0, 0), 1: (0, 255, 0), 2: (0, 0, 255), 3: (255, 255, 0), 4: (255, 0, 255),
    5: (0, 255, 255), 6: (128, 0, 0), 7: (0, 128, 0), 8: (0, 0, 128), 9: (128, 128, 0),
    10: (128, 0, 128), 11: (0, 128, 128), 12: (64, 0, 0), 13: (0, 64, 0), 14: (0, 0, 64),
    15: (64, 64, 0), 16: (64, 0, 64), 17: (0, 64, 64), 18: (192, 0, 0), 19: (0, 192, 0),
    20: (0, 0, 192), 21: (192, 192, 0)
}

# Streamlit layout
st.title("YOLO Object Detection with Streamlit")
st.sidebar.title("Upload Image or Video")

# Pilihan input
uploaded_file = st.sidebar.file_uploader("Upload a file (image or video)", type=["jpg", "jpeg", "png", "mp4", "avi", "mov", "mkv"])

# Jika file diunggah
if uploaded_file:
    file_ext = uploaded_file.name.split('.')[-1].lower()

    # Jika input adalah gambar
    if file_ext in ["jpg", "jpeg", "png"]:
        # Baca gambar
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # Konversi gambar ke grayscale sebelum deteksi
        grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        grayscale_image_rgb = cv2.cvtColor(grayscale_image, cv2.COLOR_GRAY2RGB)

        # Deteksi objek
        results = model(grayscale_image_rgb)

        # Gambarkan bounding box hasil deteksi pada gambar asli berwarna
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = box.conf[0]
                class_id = int(box.cls[0])
                label = class_names[class_id]
                color = class_colors[class_id]
                idx = box.id
                # Gambar bounding box
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
                
                # Tentukan ukuran teks
                text = f'ID {idx}: {label} {confidence:.2f}'
                (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.8, 3)
                
                # Gambar latar belakang teks
                cv2.rectangle(image, (x1, y1 - text_height - baseline), (x1 + text_width, y1), color, -1)
                
                # Tentukan warna teks berdasarkan luminansi latar belakang
                text_color = get_text_color(color)
                
                # Tambahkan teks di atas latar belakang
                cv2.putText(image, text, (x1, y1 - baseline), cv2.FONT_HERSHEY_SIMPLEX, 1.8, text_color, 3)

        # Tampilkan hasil di Streamlit
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Detected Image", use_container_width=True)

    # Jika input adalah video
    elif file_ext in ["mp4", "avi", "mov", "mkv"]:
        # Simpan video ke file sementara
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        tfile.close()

        # Baca video
        cap = cv2.VideoCapture(tfile.name)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        # Buat VideoWriter untuk menyimpan video hasil
        out = cv2.VideoWriter('outputTest/output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

        stframe = st.empty()
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Konversi frame ke grayscale sebelum deteksi
            grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            grayscale_frame_rgb = cv2.cvtColor(grayscale_frame, cv2.COLOR_GRAY2RGB)

            # Deteksi objek
            results = model(grayscale_frame_rgb)

            # Gambarkan bounding box hasil deteksi pada frame asli berwarna
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = box.conf[0]
                    class_id = int(box.cls[0])
                    label = class_names[class_id]
                    color = class_colors[class_id]
                    idx = box.id
                    # Gambar bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                    
                    # Tentukan ukuran teks
                    text = f'ID {idx}: {label} {confidence:.2f}'
                    (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.8, 3)
                    
                    # Gambar latar belakang teks
                    cv2.rectangle(frame, (x1, y1 - text_height - baseline), (x1 + text_width, y1), color, -1)
                    
                    # Tentukan warna teks berdasarkan luminansi latar belakang
                    text_color = get_text_color(color)
                    
                    # Tambahkan teks di atas latar belakang
                    cv2.putText(frame, text, (x1, y1 - baseline), cv2.FONT_HERSHEY_SIMPLEX, 1.8, text_color, 3)

            stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)

            # Tulis frame hasil ke video output
            out.write(frame)

        # Release resources
        cap.release()
        out.release()

        # Tampilkan video hasil di Streamlit
        with open('outputTest/output.mp4', "rb") as file:
            st.download_button(label="Download Video", data=file, file_name="output.mp4", mime="video/mp4")
