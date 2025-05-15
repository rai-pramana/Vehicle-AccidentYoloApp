import pandas as pd
import zipfile
import os
from io import BytesIO
import streamlit as st
import cv2

def save_video(frames, output_path, fps):
    if frames:
        height, width, layers = frames[0].shape
        video = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

        for frame in frames:
            video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        video.release()

def save_and_zip_results(annotated_frames, video_file_path, video_info_fps, vehicle_accident_data, vehicle_count, accident_count, model_names):
    save_video(annotated_frames, video_file_path, video_info_fps)

    vehicle_accident_df = pd.DataFrame(vehicle_accident_data)
    all_classes = list(model_names.values())

    vehicle_count_df = pd.DataFrame(list(vehicle_count.items()), columns=["Class", "Count_vehicle"])
    accident_count_df = pd.DataFrame(list(accident_count.items()), columns=["Class", "Count_accident"])

    vehicle_count_df = vehicle_count_df.set_index("Class").reindex(all_classes, fill_value=0).reset_index()
    accident_count_df = accident_count_df.set_index("Class").reindex(all_classes, fill_value=0).reset_index()

    count_df = pd.merge(vehicle_count_df, accident_count_df, on="Class", how="outer", suffixes=("_vehicle", "_accident"))
    count_df["Count"] = count_df["Count_vehicle"] + count_df["Count_accident"]
    count_df = count_df.drop(columns=["Count_vehicle", "Count_accident"])
    count_df = count_df.drop_duplicates(subset=["Class"])

    if "Class" in vehicle_accident_df.columns:
        avg_speed_df = vehicle_accident_df.groupby("Class")["Speed (km/h)"].mean().reset_index()
        avg_speed_df.columns = ["Class", "Average Speed (km/h)"]
        count_df = pd.merge(count_df, avg_speed_df, on="Class", how="left")

    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        vehicle_accident_df.to_excel(writer, index=False, sheet_name='Events')
        count_df.to_excel(writer, index=False, sheet_name='Summary')
    excel_data = output.getvalue()

    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
        zip_file.write(video_file_path, os.path.basename(video_file_path))
        zip_file.writestr('detection_result.xlsx', excel_data)

    return zip_buffer.getvalue()

def save_and_provide_download_button(current_dir, FPS, model):
    video_file_path = os.path.join(current_dir, '..', 'outputTest', 'output_video.mp4')
    zip_data = save_and_zip_results(
        st.session_state.annotated_frames, 
        video_file_path, 
        FPS, 
        st.session_state.vehicle_accident_data, 
        st.session_state.vehicle_count, 
        st.session_state.accident_count, 
        model.names
    )
    
    # Provide download button for ZIP files
    st.sidebar.download_button(
        label="Download Detection Results",
        data=zip_data,
        file_name='output_files.zip',
        mime='application/zip'
    )

    # Remove video file after downloading
    os.remove(video_file_path)