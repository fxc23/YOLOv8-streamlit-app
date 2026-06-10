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


LABEL_ZH_ALIASES = {
    "person": "人员",
    "bicycle": "自行车",
    "car": "汽车",
    "motorcycle": "摩托车",
    "airplane": "飞机",
    "bus": "公交车",
    "train": "火车",
    "truck": "卡车",
    "boat": "船",
    "traffic light": "交通灯",
    "fire hydrant": "消防栓",
    "stop sign": "停止标志",
    "parking meter": "停车计时器",
    "bench": "长椅",
    "bird": "鸟",
    "cat": "猫",
    "dog": "狗",
    "horse": "马",
    "sheep": "羊",
    "cow": "牛",
    "elephant": "大象",
    "bear": "熊",
    "zebra": "斑马",
    "giraffe": "长颈鹿",
    "backpack": "背包",
    "umbrella": "雨伞",
    "handbag": "手提包",
    "tie": "领带",
    "suitcase": "行李箱",
    "frisbee": "飞盘",
    "skis": "滑雪板",
    "snowboard": "单板滑雪板",
    "sports ball": "球",
    "kite": "风筝",
    "baseball bat": "棒球棒",
    "baseball glove": "棒球手套",
    "skateboard": "滑板",
    "surfboard": "冲浪板",
    "tennis racket": "网球拍",
    "bottle": "瓶子",
    "wine glass": "酒杯",
    "cup": "杯子",
    "fork": "叉子",
    "knife": "刀",
    "spoon": "勺子",
    "bowl": "碗",
    "banana": "香蕉",
    "apple": "苹果",
    "sandwich": "三明治",
    "orange": "橙子",
    "broccoli": "西兰花",
    "carrot": "胡萝卜",
    "hot dog": "热狗",
    "pizza": "披萨",
    "donut": "甜甜圈",
    "cake": "蛋糕",
    "chair": "椅子",
    "couch": "沙发",
    "potted plant": "盆栽",
    "bed": "床",
    "dining table": "餐桌",
    "toilet": "马桶",
    "tv": "电视",
    "laptop": "笔记本电脑",
    "mouse": "鼠标",
    "remote": "遥控器",
    "keyboard": "键盘",
    "cell phone": "手机",
    "microwave": "微波炉",
    "oven": "烤箱",
    "toaster": "烤面包机",
    "sink": "水槽",
    "refrigerator": "冰箱",
    "book": "书",
    "clock": "时钟",
    "vase": "花瓶",
    "scissors": "剪刀",
    "teddy bear": "玩具熊",
    "hair drier": "吹风机",
    "toothbrush": "牙刷",
    "fire": "明火",
    "smoke": "烟雾",
    "water_leak": "漏水",
    "leak": "漏水",
    "obstacle": "障碍物",
}


def _apply_label_display(result, use_chinese_labels=False):
    """Apply display-only label aliases while preserving original names for rules."""
    if not use_chinese_labels:
        return result

    original_names = dict(result.names)
    result.original_names = original_names
    result.names = {
        class_id: LABEL_ZH_ALIASES.get(str(class_name).lower(), class_name)
        for class_id, class_name in original_names.items()
    }
    return result


def _display_detected_frames(
    conf,
    model,
    st_frame,
    image,
    display_mode="叠加显示",
    st_frame_res=None,
    use_chinese_labels=False,
):
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

    _apply_label_display(res[0], use_chinese_labels)
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


def _parse_alarm_labels(raw_labels):
    """Parse comma-separated alarm labels into a normalized set."""
    return {
        label.strip().lower()
        for label in raw_labels.split(",")
        if label.strip()
    }


def _render_alarm_controls():
    """Render common alarm event controls in the sidebar."""
    st.sidebar.header("Alarm Event Config")
    raw_labels = st.sidebar.text_input(
        "Alarm Labels",
        value="fire, smoke, water_leak, leak, obstacle, person"
    )
    alarm_conf = float(st.sidebar.slider(
        "Alarm Confidence", 1, 100, 50
    )) / 100
    consecutive_frames = st.sidebar.number_input(
        "Consecutive Frames",
        min_value=1,
        max_value=100,
        value=3,
        step=1
    )
    return _parse_alarm_labels(raw_labels), alarm_conf, int(consecutive_frames)


def _alarm_events_to_csv(rows):
    """Convert alarm event rows to CSV text for download."""
    fieldnames = [
        "event_id", "source", "frame_id", "time_sec", "class_id", "class_name",
        "confidence", "consecutive_frames", "x1", "y1", "x2", "y2"
    ]
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _extract_alarm_candidates(result, alarm_labels, alarm_conf):
    """Return the highest-confidence watched detection per class for one frame."""
    candidates = {}
    original_names = getattr(result, "original_names", result.names)
    for box in result.boxes:
        class_id = int(box.cls[0].item())
        original_name = original_names.get(class_id, str(class_id))
        display_name = result.names.get(class_id, str(class_id))
        normalized_name = str(original_name).lower()
        normalized_display_name = str(display_name).lower()
        confidence = float(box.conf[0].item())
        if (
            normalized_name not in alarm_labels
            and normalized_display_name not in alarm_labels
        ) or confidence < alarm_conf:
            continue

        current = candidates.get(normalized_name)
        if current is not None and current["confidence"] >= confidence:
            continue

        x1, y1, x2, y2 = [float(value) for value in box.xyxy[0].tolist()]
        candidates[normalized_name] = {
            "class_id": class_id,
            "class_name": display_name,
            "confidence": confidence,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
        }
    return candidates


def _update_alarm_events(
    events,
    streaks,
    result,
    source,
    frame_id,
    time_sec,
    alarm_labels,
    alarm_conf,
    consecutive_frames,
):
    """Append alarm events when watched labels persist for enough frames."""
    candidates = _extract_alarm_candidates(result, alarm_labels, alarm_conf)
    for label in alarm_labels:
        if label in candidates:
            streaks[label] = streaks.get(label, 0) + 1
        else:
            streaks[label] = 0

    for label, candidate in candidates.items():
        if streaks[label] != consecutive_frames:
            continue

        events.append({
            "event_id": len(events) + 1,
            "source": source,
            "frame_id": frame_id,
            "time_sec": f"{time_sec:.3f}",
            "class_id": candidate["class_id"],
            "class_name": candidate["class_name"],
            "confidence": f"{candidate['confidence']:.4f}",
            "consecutive_frames": consecutive_frames,
            "x1": f"{candidate['x1']:.2f}",
            "y1": f"{candidate['y1']:.2f}",
            "x2": f"{candidate['x2']:.2f}",
            "y2": f"{candidate['y2']:.2f}",
        })


def _display_alarm_events(events, file_name):
    """Display alarm events and provide a CSV download."""
    st.subheader("Alarm Events")
    if events:
        st.dataframe(events, use_container_width=True)
    else:
        st.info("No alarm events triggered.")

    st.download_button(
        label="Download Alarm Events CSV",
        data=_alarm_events_to_csv(events),
        file_name=file_name,
        mime="text/csv"
    )


@st.cache_resource
def load_model(model_path):
    """
    Loads a YOLO object detection model from the specified model_path.

    Parameters:
        model_path (str): The path to the YOLO model file.

    Returns:
        A YOLO object detection model.
    """
    model_source = model_path if os.path.exists(model_path) else os.path.basename(str(model_path))
    model = YOLO(model_source)
    return model


def infer_uploaded_image(conf, model, display_mode="叠加显示", use_chinese_labels=False):
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
    alarm_labels, alarm_conf, _ = _render_alarm_controls()

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
                _apply_label_display(res[0], use_chinese_labels)
                boxes = res[0].boxes
                res_plotted = res[0].plot()[:, :, ::-1]
                csv_data = _detections_to_csv(res[0])
                alarm_events = []
                alarm_streaks = {}
                _update_alarm_events(
                    alarm_events,
                    alarm_streaks,
                    res[0],
                    "image",
                    0,
                    0,
                    alarm_labels,
                    alarm_conf,
                    1
                )

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
                _display_alarm_events(alarm_events, "image_alarm_events.csv")


def infer_uploaded_video(conf, model, display_mode="叠加显示", use_chinese_labels=False):
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
    alarm_labels, alarm_conf, consecutive_frames = _render_alarm_controls()

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
                alarm_events = []
                alarm_streaks = {}
                stop_processing = st.checkbox("Stop Processing", key="stop_video")
                while vid_cap.isOpened() and not stop_processing:
                    success, image = vid_cap.read()
                    if success:
                        _, result = _display_detected_frames(
                            conf,
                            model,
                            st_frame_det,
                            image,
                            display_mode,
                            st_frame_raw,
                            use_chinese_labels
                        )
                        time_sec = frame_idx / fps if fps else 0
                        _append_video_detection_rows(detection_rows, result, frame_idx, time_sec)
                        _update_alarm_events(
                            alarm_events,
                            alarm_streaks,
                            result,
                            "video",
                            frame_idx,
                            time_sec,
                            alarm_labels,
                            alarm_conf,
                            consecutive_frames
                        )
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
                _display_alarm_events(alarm_events, "video_alarm_events.csv")
            except Exception as e:
                st.error(f"Error loading video: {e}")
            finally:
                if tfile is not None:
                    temp_name = tfile.name
                    tfile.close()
                    if os.path.exists(temp_name):
                        os.unlink(temp_name)


def infer_uploaded_webcam(conf, model, display_mode="叠加显示", use_chinese_labels=False):
    """
    Execute inference for webcam.
    :param conf: Confidence of YOLOv8 model
    :param model: An instance of the `YOLOv8` class containing the YOLOv8 model.
    :param display_mode: "叠加显示" or "对比显示".
    :return: None
    """
    camera_index = st.sidebar.number_input(
        "Camera Index",
        min_value=0,
        max_value=20,
        value=0,
        step=1
    )
    max_frames = st.sidebar.number_input(
        "Webcam Frames to Capture",
        min_value=1,
        max_value=10000,
        value=300,
        step=1
    )
    alarm_labels, alarm_conf, consecutive_frames = _render_alarm_controls()

    if "webcam_detection_rows" not in st.session_state:
        st.session_state["webcam_detection_rows"] = []
    if "webcam_alarm_events" not in st.session_state:
        st.session_state["webcam_alarm_events"] = []
    if "webcam_capture_done" not in st.session_state:
        st.session_state["webcam_capture_done"] = False

    try:
        run_capture = st.button("Execution")
        if display_mode == "对比显示":
            col1, col2 = st.columns(2)
            st_frame_raw = col1.empty()
            st_frame_det = col2.empty()
        else:
            st_frame_det = st.empty()
            st_frame_raw = None

        if run_capture:
            vid_cap = cv2.VideoCapture(int(camera_index))
            if not vid_cap.isOpened():
                st.error(f"Unable to open camera index {camera_index}")
                return

            detection_rows = []
            alarm_events = []
            alarm_streaks = {}
            captured_frames = 0
            progress_bar = st.progress(0)
            start_time = time.time()

            try:
                for frame_idx in range(int(max_frames)):
                    success, image = vid_cap.read()
                    if not success:
                        st.warning("Webcam frame capture stopped.")
                        break

                    captured_frames += 1
                    _, result = _display_detected_frames(
                        conf,
                        model,
                        st_frame_det,
                        image,
                        display_mode,
                        st_frame_raw,
                        use_chinese_labels
                    )
                    time_sec = time.time() - start_time
                    _append_video_detection_rows(
                        detection_rows,
                        result,
                        frame_idx,
                        time_sec
                    )
                    _update_alarm_events(
                        alarm_events,
                        alarm_streaks,
                        result,
                        "webcam",
                        frame_idx,
                        time_sec,
                        alarm_labels,
                        alarm_conf,
                        consecutive_frames
                    )
                    progress_bar.progress((frame_idx + 1) / int(max_frames))
            finally:
                vid_cap.release()

            st.session_state["webcam_detection_rows"] = detection_rows
            st.session_state["webcam_alarm_events"] = alarm_events
            st.session_state["webcam_capture_done"] = True
            st.success(f"Captured {captured_frames} frames and {len(detection_rows)} detection rows.")

        if st.session_state["webcam_capture_done"]:
            st.download_button(
                label="Download Webcam Detection CSV",
                data=_video_detections_to_csv(st.session_state["webcam_detection_rows"]),
                file_name="webcam_detection_results.csv",
                mime="text/csv"
            )
            _display_alarm_events(
                st.session_state["webcam_alarm_events"],
                "webcam_alarm_events.csv"
            )
    except Exception as e:
        st.error(f"Error loading video: {str(e)}")
