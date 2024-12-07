# coordinate_utils.py
import numpy as np
import streamlit as st

def parse_coordinates(coord_string):
    try:
        points = [list(map(int, point.split(','))) for point in coord_string.split(';')]
        return np.array(points)
    except:
        st.error("Format koordinat tidak valid. Gunakan format: x1,y1;x2,y2;x3,y3;x4,y4")
        return None