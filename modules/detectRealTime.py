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
    st.title("Vehicle and Accident Detection - Real-Time")

    # Input sidebar
    st.sidebar.header("Input Settings")

    # Get list of models available in the 'models/' folder
    current_dir = os.path.dirname(os.path.abspath(__file__))

    model_path = os.path.join(current_dir, '..', 'models')
    model_files = [f for f in os.listdir(model_path) if f.endswith('.pt')]
    selected_model = st.sidebar.selectbox("Select Model", model_files)

    # Set up webcam capture
    camera = cv2.VideoCapture(1)  # Use webcam with ID 1 or 2 (Virtual OBS Camera)
    if not camera.isOpened():
        st.error("Webcam not available")
        return

    frame_width = 640
    frame_height = 360
    resolution_wh = (frame_width, frame_height)

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    # Get FPS from webcam
    # FPS = camera.get(cv2.CAP_PROP_FPS) #5
    FPS = 25 / 2.2  # Simpang Pidada
    # FPS = 25 / 1.88 # Batubulan

    stframe = st.empty()  # Place to display the frame
    vehicle_stats_placeholder = st.empty()  # Placeholder for vehicle statistics
    vehicle_speed_placeholder = st.empty()  # Placeholder for vehicle speed chart
    accident_stats_placeholder = st.empty()  # Placeholder for accident statistics

    # Control buttons
    if 'status' not in st.session_state:
        st.session_state.status = 'stopped'
    
    init_state()

    # Control buttons
    if st.sidebar.button("Start"):
        st.session_state.status = 'running'
    if st.sidebar.button("Pause"):
        st.session_state.status = 'paused'
    if st.sidebar.button("Stop"):
        st.session_state.status = 'stopped'

    # Input for Target Width and Height
    target_width = st.sidebar.number_input("Target Width (meter)", min_value=1.0, max_value=100.0, value=13.56, step=0.01)
    target_height = st.sidebar.number_input("Target Height (meter)", min_value=1.0, max_value=500.0, value=20.95, step=0.01)

    # Dropdown for selecting location
    location = st.sidebar.selectbox(
        "Location",
        ["Simpang Pidada", "Batubulan", "Fullscreen 360p", "Custom"]
    )

    # Specify coordinates based on the selected location
    if location == "Simpang Pidada":
        source_coordinates = "290,197;516,211;484,358;100,333"
    elif location == "Batubulan":
        source_coordinates = "310,52;406,45;544,342;315,358"
    elif location == "Custom":
        source_coordinates = st.sidebar.text_input(
            "Source Coordinates (format: x1,y1;x2,y2;x3,y3;x4,y4)",
            value="",
            placeholder="Enter custom coordinates"
        )
    elif location == "Fullscreen 360p":
        source_coordinates = "0,0;640,0;640,360;0,360"
    else:
        source_coordinates = "0,0;640,0;640,360;0,360"

    # Display selected coordinates
    if location != "Custom":
        st.sidebar.text_input(
            "Source Coordinates (format: x1,y1;x2,y2;x3,y3;x4,y4)",
            value=source_coordinates,
            disabled=True
        )

    # Parse Source Coordinates
    SOURCE = parse_coordinates(source_coordinates)
    TARGET = np.array([[0, 0], [target_width - 1, 0], [target_width - 1, target_height - 1], [0, target_height - 1]])

    # Confidence and IoU Threshold
    confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.3, 0.05)
    iou_threshold = st.sidebar.slider("IoU Threshold", 0.0, 1.0, 0.7, 0.05)

    # Input for video start time
    start_time_str = st.sidebar.text_input("Video Start Time (dd-mm-yyyy hh:mm:ss)", value="27-06-2023 06:40:10")

    # Parse video start time
    try:
        start_time = datetime.strptime(start_time_str, "%d-%m-%Y %H:%M:%S")
    except ValueError:
        st.error("Invalid time format. Use format: dd-mm-yyyy hh:mm:ss")
        start_time = None

    start_processing_time = t.time()

    model, byte_track, box_annotator, label_annotator, trace_annotator, polygon_zone, view_transformer, coordinates, vehicle_classes, accident_classes, thickness, text_scale = load_model_and_initialize_components(model_path, selected_model, resolution_wh, confidence_threshold, SOURCE, TARGET, FPS)

    plot_counter = 0  # Counter for unique keys

    while st.session_state.status == 'running':
        ret, frame = camera.read()
        if not ret:
            st.error("Failed to read frame from webcam")
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

    # Release the webcam when finished
    camera.release()

    # Save video and xlsx files and provide a download button for ZIP files
    if st.session_state.annotated_frames and st.session_state.status == 'stopped':
        save_and_provide_download_button(current_dir, FPS, model)

        reset_state()

    # Display last frame and chart when status 'paused'
    if st.session_state.status == 'paused' and st.session_state.last_frame is not None:
        stframe.image(st.session_state.last_frame, channels="RGB")

        plot_counter = update_real_time_statistics(
            vehicle_stats_placeholder, 
            accident_stats_placeholder, 
            vehicle_speed_placeholder, 
            plot_counter, 
            vehicle_classes
        )
        
        # Display all accident_messages stored in st.session_state
        for message in st.session_state.accident_messages:
            st.error(message)
