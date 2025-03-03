import streamlit as st
import cv2
import supervision as sv
from datetime import timedelta

from utils.annotation import add_annotations

def process_accident_detection(tracker_id, class_name, elapsed_time, timestamp):
    speed = None

    st.session_state.vehicle_accident_data.append({
        "Detik": elapsed_time, 
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

def calculate_speed(coordinates, tracker_id, FPS):
    coordinate_start = coordinates[tracker_id][-1]
    coordinate_end = coordinates[tracker_id][0]
    distance = abs(coordinate_start - coordinate_end)
    time = len(coordinates[tracker_id]) / FPS
    speed = distance / time * 3.6  # Kecepatan dalam km/h
    return speed

def process_frame(frame, model, byte_track, polygon_zone, view_transformer, coordinates, vehicle_classes, accident_classes, start_time, elapsed_time, current_time, frame_height, box_annotator, label_annotator, trace_annotator, stframe, thickness, FPS, SOURCE, confidence_threshold, iou_threshold, text_scale):
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
            st.session_state.counted_ids.add(tracker_id)

        if class_name in vehicle_classes:
            coordinates[tracker_id].append(y)

    labels = []
    for tracker_id, class_id in zip(detections.tracker_id, detections.class_id):
        class_name = model.names[class_id]
        timestamp = start_time + timedelta(seconds=elapsed_time)

        if class_name in vehicle_classes and len(coordinates[tracker_id]) >= FPS / 2:
            speed = calculate_speed(coordinates, tracker_id, FPS)
            labels.append(f"#{tracker_id} {int(speed)} km/h")
            st.session_state.vehicle_accident_data.append({
                "Detik": elapsed_time, 
                "Timestamp": timestamp, 
                "ID": tracker_id, 
                "Kelas": class_name, 
                "Kecepatan (km/h)": speed
            })

        else:
            labels.append(f"#{tracker_id}")

            if class_name in accident_classes:
                process_accident_detection(tracker_id, class_name, elapsed_time, timestamp)

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

    annotated_frame = add_annotations(annotated_frame, vehicle_classes, thickness, text_scale, frame_height, current_time)

    annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
    st.session_state.annotated_frames.append(annotated_frame)

    stframe.image(annotated_frame, channels="RGB")
    st.session_state.last_frame = annotated_frame