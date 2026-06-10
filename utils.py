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
import csv
import cv2
import io
import os
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
    :return: tuple: inference time in seconds and the first Ultralytics result
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
    return infer_time, res[0]


def _detections_to_csv(result):
    """Convert one Ultralytics result to CSV text for download."""
    fieldnames = ["class_id", "class_name", "confidence", "x1", "y1", "x2", "y2", "width", "height"]
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()

    for box in result.boxes:
        class_id = int(box.cls[0].item())
        confidence = float(box.conf[0].item())
        x1, y1, x2, y2 = [float(value) for value in box.xyxy[0].tolist()]
        writer.writerow({
            "class_id": class_id,
            "class_name": result.names.get(class_id, str(class_id)),
            "confidence": f"{confidence:.4f}",
            "x1": f"{x1:.2f}",
            "y1": f"{y1:.2f}",
            "x2": f"{x2:.2f}",
            "y2": f"{y2:.2f}",
            "width": f"{x2 - x1:.2f}",
            "height": f"{y2 - y1:.2f}",
        })

    return output.getvalue()


def _video_detections_to_csv(rows):
    """Convert video detection rows to CSV text for download."""
    fieldnames = [
        "frame_id", "time_sec", "class_id", "class_name", "confidence",
        "x1", "y1", "x2", "y2", "width", "height"
    ]
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _append_video_detection_rows(rows, result, frame_id, time_sec):
    """Append detections from one video frame to rows."""
    for box in result.boxes:
        class_id = int(box.cls[0].item())
        confidence = float(box.conf[0].item())
        x1, y1, x2, y2 = [float(value) for value in box.xyxy[0].tolist()]
        rows.append({
            "frame_id": frame_id,
            "time_sec": f"{time_sec:.3f}",
            "class_id": class_id,
            "class_name": result.names.get(class_id, str(class_id)),
            "confidence": f"{confidence:.4f}",
            "x1": f"{x1:.2f}",
            "y1": f"{y1:.2f}",
            "x2": f"{x2:.2f}",
            "y2": f"{y2:.2f}",
            "width": f"{x2 - x1:.2f}",
            "height": f"{y2 - y1:.2f}",
        })


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
                csv_data = _detections_to_csv(res[0])

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

                st.download_button(
                    label="Download Detection CSV",
                    data=csv_data,
                    file_name="detection_results.csv",
                    mime="text/csv"
                )


def infer_uploaded_video(conf, model, display_mode="叠加显示"):
    """
    Execute inference for uploaded video
    :param conf: Confidence of YOLOv8 model
    :param model: An instance of the `YOLOv8` class containing the YOLOv8 model.
    :param display_mode: "叠加显示" or "对比显示".
    :return: None
    """
    source_video = st.sidebar.file_uploader(
        label="Choose a video...",
        type=("mp4", "avi", "mov", "mkv")
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
            tfile = None
            try:
                suffix = f".{source_video.name.rsplit('.', 1)[-1]}" if "." in source_video.name else ".mp4"
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
                tfile.write(source_video.getbuffer())
                tfile.flush()
                vid_cap = cv2.VideoCapture(tfile.name)
                total_frames = int(vid_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                progress_bar = st.progress(0)

                if display_mode == "对比显示":
                    col1, col2 = st.columns(2)
                    st_frame_raw = col1.empty()
                    st_frame_det = col2.empty()
                else:
                    st_frame_det = st.empty()
                    st_frame_raw = None

                frame_idx = 0
                detection_rows = []
                stop_processing = st.checkbox("Stop Processing", key="stop_video")
                while vid_cap.isOpened() and not stop_processing:
                    success, image = vid_cap.read()
                    if success:
                        _, result = _display_detected_frames(conf, model, st_frame_det, image, display_mode, st_frame_raw)
                        time_sec = frame_idx / fps if fps else 0
                        _append_video_detection_rows(detection_rows, result, frame_idx, time_sec)
                        frame_idx += 1
                        if total_frames:
                            progress_bar.progress(min(frame_idx / total_frames, 1.0))
                    else:
                        vid_cap.release()
                        break
                progress_bar.progress(1.0)
                vid_cap.release()
                st.session_state['video_running'] = False
                st.download_button(
                    label="Download Video Detection CSV",
                    data=_video_detections_to_csv(detection_rows),
                    file_name="video_detection_results.csv",
                    mime="text/csv"
                )
            except Exception as e:
                st.error(f"Error loading video: {e}")
            finally:
                if tfile is not None:
                    temp_name = tfile.name
                    tfile.close()
                    if os.path.exists(temp_name):
                        os.unlink(temp_name)


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
