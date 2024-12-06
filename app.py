import streamlit as st
import cv2
import numpy as np
from collections import defaultdict, deque
from ultralytics import YOLO
import supervision as sv
from tempfile import NamedTemporaryFile
import torch

# Fungsi untuk mengubah input string ke array numpy
def parse_coordinates(coord_string):
    try:
        points = [list(map(int, point.split(','))) for point in coord_string.split(';')]
        return np.array(points)
    except:
        st.error("Format koordinat tidak valid. Gunakan format: x1,y1;x2,y2;x3,y3;x4,y4")
        return None


class ViewTransformer:
    def __init__(self, source: np.ndarray, target: np.ndarray) -> None:
        source = source.astype(np.float32)
        target = target.astype(np.float32)
        self.m = cv2.getPerspectiveTransform(source, target)

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        if points.size == 0:
            return points
        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)
        return transformed_points.reshape(-1, 2)


# Streamlit UI
st.title("Deteksi Kendaraan dan Estimasi Kecepatan - Dengan Statistik")

# Upload video
uploaded_video = st.file_uploader("Upload Video", type=["mp4", "avi", "mov", "mkv"])

# Input untuk Target Width dan Height
target_width = st.number_input("Target Width (meter)", min_value=1.0, max_value=100.0, value=13.56, step=0.01)
target_height = st.number_input("Target Height (meter)", min_value=1.0, max_value=500.0, value=20.95, step=0.01)

# Input untuk Source Coordinates
source_coordinates = st.text_input(
    "Source Coordinates (format: x1,y1;x2,y2;x3,y3;x4,y4)",
    value="619,394;1032,423;968,717;240,666"
)

# Parse Source Coordinates
SOURCE = parse_coordinates(source_coordinates)
TARGET = np.array([[0, 0], [target_width - 1, 0], [target_width - 1, target_height - 1], [0, target_height - 1]])

# Confidence dan IoU Threshold
confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.3, 0.05)
iou_threshold = st.slider("IoU Threshold", 0.0, 1.0, 0.7, 0.05)

if uploaded_video and SOURCE is not None:
    tfile = NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())

    video_info = sv.VideoInfo.from_video_path(video_path=tfile.name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load YOLO model
    model = YOLO('models/vehicle-accident.pt').to(device)  # Ganti dengan model Anda

    # Ambil nama class dari model
    all_classes = model.names.values()
    vehicle_classes = {"bus", "car", "motorcycle", "truck"}
    accident_classes = set(all_classes) - vehicle_classes

    byte_track = sv.ByteTrack(
        frame_rate=video_info.fps,
        track_activation_threshold=confidence_threshold
    )

    thickness = sv.calculate_optimal_line_thickness(video_info.resolution_wh)
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

    vehicle_count = defaultdict(int)
    accident_count = defaultdict(int)
    counted_ids = set()  # Set untuk melacak ID yang sudah dihitung

    stframe = st.empty()  # Tempat untuk menampilkan frame
    for frame in sv.get_video_frames_generator(source_path=tfile.name):
        result = model(frame)[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = detections[detections.confidence > confidence_threshold]
        detections = detections[polygon_zone.trigger(detections)]
        detections = detections.with_nms(threshold=iou_threshold)
        detections = byte_track.update_with_detections(detections=detections)

        points = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
        points = view_transformer.transform_points(points=points).astype(int)

        for tracker_id, [_, y], class_id in zip(detections.tracker_id, points, detections.class_id):
            class_name = model.names[class_id]
            if tracker_id not in counted_ids:
                if class_name in vehicle_classes:
                    vehicle_count[class_name] += 1
                elif class_name in accident_classes:
                    accident_count[class_name] += 1
                counted_ids.add(tracker_id)  # Tambahkan ID ke set setelah dihitung

            if class_name in vehicle_classes:
                coordinates[tracker_id].append(y)

        labels = []
        for tracker_id, class_id in zip(detections.tracker_id, detections.class_id):
            class_name = model.names[class_id]
            if class_name in vehicle_classes and len(coordinates[tracker_id]) >= video_info.fps / 2:
                coordinate_start = coordinates[tracker_id][-1]
                coordinate_end = coordinates[tracker_id][0]
                distance = abs(coordinate_start - coordinate_end)
                time = len(coordinates[tracker_id]) / video_info.fps
                speed = distance / time * 3.6  # Kecepatan dalam km/h
                labels.append(f"#{tracker_id} {int(speed)} km/h")
            else:
                labels.append(f"#{tracker_id}")

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

        # Convert annotated frame to RGB for display
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

        # Display frame
        stframe.image(annotated_frame, channels="RGB")

    # Tampilkan hasil di sidebar
    st.sidebar.subheader("Statistik Kendaraan")
    for vehicle, count in vehicle_count.items():
        st.sidebar.write(f"{vehicle}: {count} kendaraan")

    st.sidebar.subheader("Statistik Kecelakaan")
    for accident, count in accident_count.items():
        st.sidebar.write(f"{accident}: {count} kejadian")