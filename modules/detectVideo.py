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
    st.title("Vehicle and Accident Detection - Video")

    # Sidebar input
    st.sidebar.header("Input Settings")

    # Get list of models available in the 'models/' folder
    current_dir = os.path.dirname(os.path.abspath(__file__))

    model_path = os.path.join(current_dir, '..', 'models')
    model_files = [f for f in os.listdir(model_path) if f.endswith('.pt')]
    selected_model = st.sidebar.selectbox("Select Model", model_files)

    # Upload video
    uploaded_video = st.sidebar.file_uploader("Upload Video", type=["mp4", "avi", "mov", "mkv"])

    if uploaded_video is not None:
        stframe = st.empty()  # Place to display the frame
        vehicle_stats_placeholder = st.empty()  # Placeholder for vehicle statistics
        vehicle_speed_placeholder = st.empty()  # Placeholder for vehicle speed chart
        accident_stats_placeholder = st.empty()  # Placeholder for accident statistics

        if 'last_uploaded_video' not in st.session_state or st.session_state.last_uploaded_video != uploaded_video:
            # Reset state if a new video is uploaded
            st.session_state.status = 'paused'
            reset_state()

            # Save the newly uploaded video
            st.session_state.last_uploaded_video = uploaded_video

        # Control buttons
        if 'status' not in st.session_state:
            st.session_state.status = 'paused'
        
        init_state()

        # Control buttons
        if st.sidebar.button("Start/Restart"):
            st.session_state.status = 'running'
            reset_state()
        if st.sidebar.button("Continue"):
            st.session_state.status = 'running'
        if st.sidebar.button("Pause"):
            st.session_state.status = 'paused'

        # Input for Target Width and Height
        target_width = st.sidebar.number_input("Target Width (meter)", min_value=1.0, max_value=100.0, value=13.56, step=0.01)
        target_height = st.sidebar.number_input("Target Height (meter)", min_value=1.0, max_value=500.0, value=20.95, step=0.01)

        # Dropdown for selecting location
        location = st.sidebar.selectbox(
            "Location",
            ["Simpang Pidada", "Batubulan", "Fullscreen 720p", "Custom"]
        )

        # Specify coordinates based on the selected location
        if location == "Simpang Pidada":
            source_coordinates = "550,394;1032,423;968,717;130,666"
        elif location == "Batubulan":
            source_coordinates = "620,104;812,90;1089,684;630,716"
        elif location == "Custom":
            source_coordinates = st.sidebar.text_input(
                "Source Coordinates (format: x1,y1;x2,y2;x3,y3;x4,y4)",
                value="",
                placeholder="Enter custom coordinates"
            )
        elif location == "Fullscreen 720p":
            source_coordinates = "0,0;1280,0;1280,720;0,720"
        else:
            source_coordinates = "0,0;1280,0;1280,720;0,720"

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

        if SOURCE is not None and start_time is not None:
            tfile = NamedTemporaryFile(delete=False)
            tfile.write(uploaded_video.read())

            video_info = sv.VideoInfo.from_video_path(video_path=tfile.name)
            frame_height = video_info.height
            resolution_wh = video_info.resolution_wh
            FPS = video_info.fps

            model, byte_track, box_annotator, label_annotator, trace_annotator, polygon_zone, view_transformer, coordinates, vehicle_classes, accident_classes, thickness, text_scale = load_model_and_initialize_components(model_path, selected_model, resolution_wh, confidence_threshold, SOURCE, TARGET, FPS)

            plot_counter = 0  # Counter for unique keys

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

            # Save video and xlsx files and provide a download button for ZIP files
            if st.session_state.annotated_frames:
                save_and_provide_download_button(current_dir, FPS, model)

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
