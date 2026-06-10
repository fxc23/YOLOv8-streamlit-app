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


def _display_detected_frames(conf, model, st_frame, image, display_mode="叠加显示", st_frame_res=None):
    """
    Display the detected objects on a video frame using the YOLOv8 model.
    :param conf (float): Confidence threshold for object detection.
    :param model (YOLOv8): An instance of the `YOLOv8` class containing the YOLOv8 model.
    :param st_frame (Streamlit object): A Streamlit object to display the original video.
    :param image (numpy array): A numpy array representing the video frame.
    :param display_mode (str): "叠加显示" or "对比显示".
    :param st_frame_res (Streamlit object, optional): A Streamlit object to display the detected video (for 对比显示).
    :return: float: inference time in seconds
    """
    image_resized = cv2.resize(image, (720, int(720 * (9 / 16))))

    t1 = time.time()
    res = model.predict(image_resized, conf=conf)
    t2 = time.time()
    infer_time = t2 - t1

    res_plotted = res[0].plot()

    if display_mode == "对比显示" and st_frame_res is not None:
        st_frame.image(image_resized,
                       caption=f'Original | {infer_time:.3f}s',
                       channels="BGR",
                       use_column_width=True
                       )
        st_frame_res.image(res_plotted,
                           caption=f'Detected | {infer_time:.3f}s',
                           channels="BGR",
                           use_column_width=True
                           )
    else:
        st_frame.image(res_plotted,
                       caption=f'Detected | {infer_time:.3f}s',
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


def infer_uploaded_image(conf, model, display_mode="叠加显示"):
    """
    Execute inference for uploaded image
    :param conf: Confidence of YOLOv8 model
    :param model: An instance of the `YOLOv8` class containing the YOLOv8 model.
    :param display_mode: "叠加显示" or "对比显示".
    :return: None
    """
    source_img = st.sidebar.file_uploader(
        label="Choose an image...",
        type=("jpg", "jpeg", "png", 'bmp', 'webp')
    )

    if source_img:
        uploaded_image = Image.open(source_img)
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

                if display_mode == "对比显示":
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(source_img,
                                 caption="Original Image",
                                 use_column_width=True)
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
                else:
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


def infer_uploaded_video(conf, model, display_mode="叠加显示"):
    """
    Execute inference for uploaded video
    :param conf: Confidence of YOLOv8 model
    :param model: An instance of the `YOLOv8` class containing the YOLOv8 model.
    :param display_mode: "叠加显示" or "对比显示".
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
                progress_bar = st.progress(0)

                if display_mode == "对比显示":
                    col1, col2 = st.columns(2)
                    st_frame_raw = col1.empty()
                    st_frame_det = col2.empty()
                else:
                    st_frame_det = st.empty()
                    st_frame_raw = None

                frame_idx = 0
                stop_processing = st.checkbox("Stop Processing", key="stop_video")
                while vid_cap.isOpened() and not stop_processing:
                    success, image = vid_cap.read()
                    if success:
                        _display_detected_frames(conf, model, st_frame_det, image, display_mode, st_frame_raw)
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


def infer_uploaded_webcam(conf, model, display_mode="叠加显示"):
    """
    Execute inference for webcam.
    :param conf: Confidence of YOLOv8 model
    :param model: An instance of the `YOLOv8` class containing the YOLOv8 model.
    :param display_mode: "叠加显示" or "对比显示".
    :return: None
    """
    try:
        flag = st.button(
            label="Stop running"
        )
        if display_mode == "对比显示":
            col1, col2 = st.columns(2)
            st_frame_raw = col1.empty()
            st_frame_det = col2.empty()
        else:
            st_frame_det = st.empty()
            st_frame_raw = None

        vid_cap = cv2.VideoCapture(0)
        while not flag:
            success, image = vid_cap.read()
            if success:
                _display_detected_frames(
                    conf,
                    model,
                    st_frame_det,
                    image,
                    display_mode,
                    st_frame_raw
                )
            else:
                vid_cap.release()
                break
    except Exception as e:
        st.error(f"Error loading video: {str(e)}")
