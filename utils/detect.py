import streamlit as st
import cv2
import supervision as sv
from datetime import timedelta

from utils.annotation import add_annotations

def process_accident_detection(tracker_id, class_name, elapsed_time, timestamp):
    speed = None

    # Check if it has been recorded before (avoid duplication)
    already_recorded = any(
        d["ID"] == tracker_id and d["Timestamp"] == timestamp and d["Class"] == class_name
        for d in st.session_state.vehicle_accident_data
    )
    if not already_recorded:
        st.session_state.vehicle_accident_data.append({
            "Sec": elapsed_time, 
            "Timestamp": timestamp, 
            "ID": tracker_id, 
            "Class": class_name, 
            "Speed (km/h)": speed
        })

        # Accident enumeration only if tracker_id has never been enumerated for an accident.
        if tracker_id not in st.session_state.counted_accident_ids:
            st.session_state.accident_count[class_name] += 1
            st.session_state.counted_accident_ids.add(tracker_id)

    if st.session_state.vehicle_accident_data:
        latest_accident = st.session_state.vehicle_accident_data[-1]  # Retrieve the latest accident data (last element)

        # Check if tracker_id is already in notified_accident_ids
        if tracker_id not in st.session_state.notified_accident_ids:
            # Format notification message
            accident_message = (
                f"🚨 **Accident Detected!**\n\n"
                f"🔹 **Sec:** {latest_accident['Sec']:.0f}\n"
                f"🕒 **Timestamp:** {latest_accident['Timestamp'].strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"🆔 **ID:** {latest_accident['ID']}\n"
                f"🚗 **Class:** {latest_accident['Class']}"
            )

            # Add accident_message to array in st.session_state
            st.session_state.accident_messages.append(accident_message)

            # Add tracker_id to notified_accident_ids
            st.session_state.notified_accident_ids.add(tracker_id)

            # Display accident notification
            st.toast(accident_message)
            st.error(accident_message)

def calculate_speed(coordinates, tracker_id, FPS):
    coordinate_start = coordinates[tracker_id][-1]
    coordinate_end = coordinates[tracker_id][0]
    distance = abs(coordinate_start - coordinate_end)
    time = len(coordinates[tracker_id]) / FPS
    speed = distance / time * 3.6  # Speed in km/h
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
        # Vehicle enumeration only if tracker_id has never been enumerated for vehicle
        if class_name in vehicle_classes and tracker_id not in st.session_state.counted_vehicle_ids:
            st.session_state.vehicle_count[class_name] += 1
            st.session_state.counted_vehicle_ids.add(tracker_id)
        if class_name in accident_classes:
            # Accident enumeration is executed in process_accident_detection
            pass
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
                "Sec": elapsed_time, 
                "Timestamp": timestamp, 
                "ID": tracker_id, 
                "Class": class_name, 
                "Speed (km/h)": speed
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