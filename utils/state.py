import streamlit as st
from collections import defaultdict

def reset_state():
    st.session_state.frame_index = 0
    st.session_state.vehicle_count = defaultdict(int)
    st.session_state.accident_count = defaultdict(int)
    st.session_state.counted_ids = set()
    st.session_state.last_frame = None
    st.session_state.vehicle_accident_data = []
    st.session_state.annotated_frames = []
    st.session_state.accident_messages = []
    st.session_state.notified_accident_ids = set()

def init_state():
    if 'frame_index' not in st.session_state:
        st.session_state.frame_index = 0
    if 'vehicle_count' not in st.session_state:
        st.session_state.vehicle_count = defaultdict(int)
    if 'accident_count' not in st.session_state:
        st.session_state.accident_count = defaultdict(int)
    if 'counted_ids' not in st.session_state:
        st.session_state.counted_ids = set()
    if 'last_frame' not in st.session_state:
        st.session_state.last_frame = None
    if 'vehicle_accident_data' not in st.session_state:
        st.session_state.vehicle_accident_data = []
    if 'annotated_frames' not in st.session_state:
        st.session_state.annotated_frames = []  
    if 'accident_messages' not in st.session_state:
        st.session_state.accident_messages = []
    if 'notified_accident_ids' not in st.session_state:
        st.session_state.notified_accident_ids = set()