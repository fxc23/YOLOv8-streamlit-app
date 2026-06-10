#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   @File Name:     utils.py
   @Author:        Luyao.zhang
   @Date:          2023/5/16
   @Description:
-------------------------------------------------
"""
from ultralytics import YOLO
import streamlit as st
import cv2
import time
from PIL import Image
import tempfile


def _display_detected_frames(conf, model, st_frame, image):
    """
    Display the detected objects on a video frame using the YOLOv8 model.
    :param conf (float): Confidence threshold for object detection.
    :param model (YOLOv8): An instance of the `YOLOv8` class containing the YOLOv8 model.
    :param st_frame (Streamlit object): A Streamlit object to display the detected video.
    :param image (numpy array): A numpy array representing the video frame.
    :return: float: inference time in seconds
    """
    image = cv2.resize(image, (720, int(720 * (9 / 16))))

    t1 = time.time()
    res = model.predict(image, conf=conf)
    t2 = time.time()
    infer_time = t2 - t1

    res_plotted = res[0].plot()
    st_frame.image(res_plotted,
                   caption=f'Detected Video | Inference: {infer_time:.3f}s',
                   channels="BGR",
                   use_column_width=True
                   )
    return infer_time


@st.cache_resource
def load_model(model_path):
    """
    Loads a YOLO object detection model from the specified model_path.

    Parameters:
        model_path (str): The path to the YOLO model file.

    Returns:
        A YOLO object detection model.
    """
    model = YOLO(model_path)
    return model


def infer_uploaded_image(conf, model):
    """
    Execute inference for uploaded image
    :param conf: Confidence of YOLOv8 model
    :param model: An instance of the `YOLOv8` class containing the YOLOv8 model.
    :return: None
    """
    source_img = st.sidebar.file_uploader(
        label="Choose an image...",
        type=("jpg", "jpeg", "png", 'bmp', 'webp')
    )

    col1, col2 = st.columns(2)

    with col1:
        if source_img:
            uploaded_image = Image.open(source_img)
            # adding the uploaded image to the page with caption
            st.image(
                image=source_img,
                caption="Uploaded Image",
                use_column_width=True
            )

    if source_img:
        if st.button("Execution"):
            with st.spinner("Running..."):
                res = model.predict(uploaded_image,
                                    conf=conf)
                boxes = res[0].boxes
                res_plotted = res[0].plot()[:, :, ::-1]

                with col2:
                    st.image(res_plotted,
                             caption="Detected Image",
                             use_column_width=True)
                    try:
                        with st.expander("Detection Results"):
                            for box in boxes:
                                st.write(box.xywh)
                    except Exception as ex:
                        st.write("No image is uploaded yet!")
                        st.write(ex)


def infer_uploaded_video(conf, model):
    """
    Execute inference for uploaded video
    :param conf: Confidence of YOLOv8 model
    :param model: An instance of the `YOLOv8` class containing the YOLOv8 model.
    :return: None
    """
    source_video = st.sidebar.file_uploader(
        label="Choose a video..."
    )

    if source_video:
        st.video(source_video)

    if source_video:
        if 'video_running' not in st.session_state:
            st.session_state['video_running'] = False

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Execution"):
                st.session_state['video_running'] = True

        if st.session_state['video_running']:
            try:
                tfile = tempfile.NamedTemporaryFile()
                tfile.write(source_video.read())
                vid_cap = cv2.VideoCapture(tfile.name)
                total_frames = int(vid_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                st_frame = st.empty()
                progress_bar = st.progress(0)

                frame_idx = 0
                stop_processing = st.checkbox("Stop Processing", key="stop_video")
                while vid_cap.isOpened() and not stop_processing:
                    success, image = vid_cap.read()
                    if success:
                        _display_detected_frames(conf, model, st_frame, image)
                        frame_idx += 1
                        progress_bar.progress(min(frame_idx / total_frames, 1.0))
                    else:
                        vid_cap.release()
                        break
                progress_bar.progress(1.0)
                vid_cap.release()
                st.session_state['video_running'] = False
            except Exception as e:
                st.error(f"Error loading video: {e}")


def infer_uploaded_webcam(conf, model):
    """
    Execute inference for webcam.
    :param conf: Confidence of YOLOv8 model
    :param model: An instance of the `YOLOv8` class containing the YOLOv8 model.
    :return: None
    """
    try:
        flag = st.button(
            label="Stop running"
        )
        vid_cap = cv2.VideoCapture(0)
        st_frame = st.empty()
        while not flag:
            success, image = vid_cap.read()
            if success:
                _display_detected_frames(
                    conf,
                    model,
                    st_frame,
                    image
                )
            else:
                vid_cap.release()
                break
    except Exception as e:
        st.error(f"Error loading video: {str(e)}")
