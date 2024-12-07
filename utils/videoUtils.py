# video_utils.py
import cv2
import numpy as np

def save_video(frames, output_path, fps):
    if frames:
        height, width, layers = frames[0].shape
        video = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

        for frame in frames:
            video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        video.release()