#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   @File Name:     app.py
   @Author:        Luyao.zhang
   @Date:          2023/5/15
   @Description:
-------------------------------------------------
"""
from pathlib import Path
import hashlib
from PIL import Image
import streamlit as st

import config
from utils import load_model, infer_uploaded_image, infer_uploaded_video, infer_uploaded_webcam


def save_uploaded_model(uploaded_model):
    """Save an uploaded model file and return its local path."""
    custom_model_dir = Path(config.CUSTOM_MODEL_DIR)
    custom_model_dir.mkdir(parents=True, exist_ok=True)
    model_buffer = uploaded_model.getbuffer()
    model_hash = hashlib.sha256(model_buffer).hexdigest()[:12]
    original_name = Path(uploaded_model.name).name
    model_path = custom_model_dir / f"{Path(original_name).stem}_{model_hash}.pt"
    model_path.write_bytes(model_buffer)
    return model_path


# setting page layout
st.set_page_config(
    page_title="Interactive Interface for YOLOv8",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
    )

# main page heading
st.title("Interactive Interface for YOLOv8")

# sidebar
st.sidebar.header("DL Model Config")

# model options
task_type = st.sidebar.selectbox(
    "Select Task",
    ["Detection", "Segmentation"]
)

model_type = None
model_path = None
if task_type in ["Detection", "Segmentation"]:
    model_source = st.sidebar.radio(
        "Model Source",
        ["Built-in", "Custom Upload"]
    )
    if model_source == "Built-in":
        if task_type == "Detection":
            model_list = config.DETECTION_MODEL_LIST
            model_dir = config.DETECTION_MODEL_DIR
        else:
            model_list = config.SEGMENTATION_MODEL_LIST
            model_dir = config.SEGMENTATION_MODEL_DIR

        model_type = st.sidebar.selectbox(
            "Select Model",
            model_list
        )
        model_path = Path(model_dir, str(model_type))
    else:
        uploaded_model = st.sidebar.file_uploader(
            "Upload YOLO Model",
            type=("pt",)
        )
        if uploaded_model:
            model_path = save_uploaded_model(uploaded_model)
            st.sidebar.success(f"Loaded custom model: {model_path.name}")
        else:
            st.sidebar.warning("Upload a .pt model to continue.")
else:
    st.error("Currently only 'Detection' and 'Segmentation' functions are implemented")

confidence = float(st.sidebar.slider(
    "Select Model Confidence", 30, 100, 50)) / 100

if model_path is None:
    st.stop()

# load pretrained DL model
try:
    model = load_model(model_path)
except Exception as e:
    st.error(f"Unable to load model. Please check the specified path: {model_path}")
    st.stop()

# display mode
st.sidebar.header("Display Mode")
display_mode = st.sidebar.radio(
    "Select Display Mode",
    ["叠加显示", "对比显示"]
)
use_chinese_labels = st.sidebar.checkbox(
    "Use Chinese Detection Labels",
    value=False
)

# image/video options
st.sidebar.header("Image/Video Config")
source_selectbox = st.sidebar.selectbox(
    "Select Source",
    config.SOURCES_LIST
)

source_img = None
if source_selectbox == config.SOURCES_LIST[0]: # Image
    infer_uploaded_image(confidence, model, display_mode, use_chinese_labels)
elif source_selectbox == config.SOURCES_LIST[1]: # Video
    infer_uploaded_video(confidence, model, display_mode, use_chinese_labels)
elif source_selectbox == config.SOURCES_LIST[2]: # Webcam
    infer_uploaded_webcam(confidence, model, display_mode, use_chinese_labels)
else:
    st.error("Currently only 'Image' and 'Video' source are implemented")
