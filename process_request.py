import os
import cv2
import datetime
import collections
import numpy as np
from sort import Sort
from PIL import Image
from collections import deque
from flask import render_template
from werkzeug.utils import secure_filename

import config
from db import *
from utlis import *
from helper import *
from config import *
from intermediate import *
from validator import validate_nepali, validate_english


tracker = Sort(max_age=config.__TRACK_MAX_AGE__, min_hits=config.__TRACK_MIN_HITS__)
track_texts = {}

tracked_plates = {}
next_plate_id = 0

realtime_recognized_plates = {}


def process_image(image_file):
    filename = secure_filename(image_file.filename)
    filepath = os.path.join(config.__UPLOAD_FOLDER__, filename)
    image_file.save(filepath)

    img = Image.open(filepath)
    od_results = model(img)
    recognized_plates = []

    plates = []
    for result in od_results:
        boxes = result.boxes
        for box in boxes:
            if box.conf >= config.__OD_THRESHOLD__ and box.cls == 0:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                plate_img = img.crop((x1, y1, x2, y2))
                text, text_conf = process_plate(plate_img)
                plate = Plate(
                    plate_bbox=(x1, y1, x2, y2),
                    plate_conf=float(box.conf),
                    plate_color=get_plate_color(plate_img),
                    text=text,
                    text_conf=text_conf,
                    source=Source.IMAGE,
                    original_image_path=filename,
                    result_image_path=None)
                recognized_plates.append(plate)
                plates.append({'text': text, 'confidence': text_conf})

    processed_filename = f"processed_{filename}"
    processed_filepath = os.path.join(config.__PROCESSED_FOLDER__, processed_filename)
    processed_img = img.copy()

    for plate in recognized_plates:
        processed_img = draw_text(processed_img, plate.text, plate.text_conf, plate.lang, plate.plate_bbox)
        plate.recognized_plates = processed_filename
        plate.save_to_db()

    if not isinstance(processed_img, Image.Image):
        Image.fromarray(processed_img).save(processed_filepath)
    else:
        processed_img.save(processed_filepath)

    return render_template(
        'image_result.html',
        original_image=filename,
        processed_image=processed_filename,
        plates=plates
    )


def process_video(video):
    global next_plate_id, tracked_plates
    recognized_plates = []
    frame_count = 0
    next_plate_id = 0
    tracked_plates = {}

    filename = secure_filename(video.filename)
    filepath = os.path.join(config.__UPLOAD_FOLDER__, filename)
    video.save(filepath)

    cap = cv2.VideoCapture(filepath)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    processed_filename = f"processed_{filename}"
    processed_filepath = os.path.join(config.__PROCESSED_FOLDER__, processed_filename)

    out = cv2.VideoWriter(processed_filepath, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        processed_frame = process_video_frame(frame, frame_count)
        out.write(processed_frame)
        frame_count += 1

    for track_id, track_info in tracked_plates.items():
        if track_info['hits'] >= config.__TRACK_MIN_HITS__:
            if track_info['text_history']:
                text_counts = {}
                for text in track_info['text_history']:
                    if text in text_counts:
                        text_counts[text] += 1
                    else:
                        text_counts[text] = 1
                most_common_text = max(text_counts.items(), key=lambda x: x[1])[0]
                plate_lang = track_info.get('language', 'unknown')

                plate_info = (Plate(
                    plate_bbox=str(track_info.get('bbox', [])),
                    plate_conf=track_info.get('confidence', 0),
                    plate_color=track_info.get('plate_color', 'N/A'),
                    text=most_common_text,
                    text_conf=track_info.get('text_confidence', 0),
                    source=Source.VIDEO,
                    original_image_path=filename,
                    result_image_path=processed_filename
                ))
                plate_info.save_to_db()

                recognized_plates.append({
                    'track_id': track_id,
                    'text': most_common_text,
                    'confidence': track_info['confidence'],
                    'hits': track_info['hits'],
                    'language': plate_lang
                })

    cap.release()
    out.release()

    return render_template('video_result.html',
                           original_video=filename,
                           processed_video=processed_filename,
                           plates=recognized_plates)


def process_video_sort(video):
    filename = secure_filename(video.filename)
    filepath = os.path.join(config.__UPLOAD_FOLDER__, filename)
    video.save(filepath)

    cap = cv2.VideoCapture(filepath)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    processed_filename = f"processed_{filename}"
    processed_filepath = os.path.join(config.__PROCESSED_FOLDER__, processed_filename)
    out = cv2.VideoWriter(processed_filepath, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    track_texts = {}
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        od_results = model(frame)
        detections = []
        for result in od_results:
            for box in result.boxes:
                if box.conf >= config.__OD_THRESHOLD__ and box.cls == 0:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf)
                    detections.append([x1, y1, x2, y2, conf])

        dets = np.array(detections)
        if len(dets) == 0:
            dets = np.empty((0, 5))
        tracks = tracker.update(dets)

        for track in tracks:
            x1, y1, x2, y2, track_id = map(int, track)
            plate_img = frame[y1:y2, x1:x2]
            if (track_id not in track_texts) or (track_texts[track_id]['confidence'] < config.__VIDEO_OCR_THRESHOLD__ and frame_count % 10 == 0):
                text, text_conf = process_plate(plate_img)
                if track_id not in track_texts or text_conf > track_texts[track_id]['confidence']:
                    track_texts[track_id] = {'text': text, 'confidence': text_conf}
            else:
                text = track_texts[track_id]['text']
                text_conf = track_texts[track_id]['confidence']

            cv2.rectangle(frame, (x1, y1), (x2, y2), (57, 255, 20), 2)
            cv2.putText(frame, f"ID:{track_id} {text}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (77, 77, 255), 2)

        out.write(frame)
        frame_count += 1

    cap.release()
    out.release()

    recognized_plates = [
        {'track_id': tid, 'text': info['text'], 'confidence': info['confidence']}
        for tid, info in track_texts.items()
    ]

    return render_template('video_result.html',
                           original_video=filename,
                           processed_video=processed_filename,
                           plates=recognized_plates)


def generate_frames():
    # cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture(1)
    tracker = Sort(max_age=config.__TRACK_MAX_AGE__, min_hits=config.__TRACK_MIN_HITS__)
    track_texts.clear()
    while True:
        success, frame = cap.read()
        if not success:
            break

        od_results = model(frame)
        detections = []
        for result in od_results:
            for box in result.boxes:
                if box.conf >= config.__OD_THRESHOLD__ and box.cls == 0:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf)
                    detections.append([x1, y1, x2, y2, conf])

        dets = np.array(detections)
        if len(dets) == 0:
            dets = np.empty((0, 5))
        tracks = tracker.update(dets)

        for track in tracks:
            x1, y1, x2, y2, track_id = map(int, track)
            plate_img = frame[y1:y2, x1:x2]
            if track_id not in track_texts or frame_count % 10 == 0:
                text, text_conf = process_plate(plate_img)
                track_texts[track_id] = {'text': text, 'confidence': text_conf}
            else:
                text = track_texts[track_id]['text']
                text_conf = track_texts[track_id]['confidence']

            cv2.rectangle(frame, (x1, y1), (x2, y2), (57, 255, 20), 2)
            cv2.putText(frame, f"ID:{track_id} {text}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (77, 77, 255), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()


def generate_frames_sort():
    global realtime_recognized_plates

    cap = cv2.VideoCapture(0)
    tracker = Sort(max_age=config.__TRACK_MAX_AGE__, min_hits=config.__TRACK_MIN_HITS__)
    track_texts = {}
    frame_count = 0

    while True:
        success, frame = cap.read()
        if not success:
            break

        od_results = model(frame)
        detections = []
        for result in od_results:
            for box in result.boxes:
                if box.conf >= config.__OD_THRESHOLD__ and box.cls == 0:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf)
                    detections.append([x1, y1, x2, y2, conf])

        dets = np.array(detections)
        if len(dets) == 0:
            dets = np.empty((0, 5))
        tracks = tracker.update(dets)

        for track in tracks:
            x1, y1, x2, y2, track_id = map(int, track)
            plate_img = frame[y1:y2, x1:x2]
            if track_id not in track_texts or frame_count % 10 == 0:
                text, text_conf = process_plate(plate_img)
                track_texts[track_id] = {'text': text, 'confidence': text_conf}
            else:
                text = track_texts[track_id]['text']
                text_conf = track_texts[track_id]['confidence']

            cv2.rectangle(frame, (x1, y1), (x2, y2), (57, 255, 20), 2)
            cv2.putText(frame, f"ID:{track_id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (77, 77, 255), 2)

        realtime_recognized_plates.clear()
        for tid, info in track_texts.items():
            realtime_recognized_plates[tid] = {
                'text': info['text'],
                'confidence': info['confidence']
            }

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        frame_count += 1

    cap.release()


def generate_realtime_frame():
    global tracked_plates

    tracked_plates = []
    counter = 1
    active_ids = set()

    # cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture(1)
    # cap = cv2.VideoCapture(2)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25

    processed_filename = "realtime_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + ".mp4"
    processed_filepath = os.path.join(config.__PROCESSED_FOLDER__, processed_filename)
    out = cv2.VideoWriter(processed_filepath, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    while True:
        success, frame = cap.read()
        if not success:
            break

        od_results = model(frame)
        detected_boxes = []
        active_ids = set()

        for result in od_results:
            boxes = result.boxes
            for box in boxes:
                if box.conf < config.__OD_THRESHOLD__ or box.cls == 0:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detected_boxes.append(((x1, y1, x2, y2), box.conf))

        for bbox, od_conf in detected_boxes:
            tracked_id = is_plate_tracked(bbox, tracked_plates)
            if tracked_id is not None:
                active_ids.add(tracked_id)
                for plate in tracked_plates:
                    if plate['id'] == tracked_id:
                        plate['plate'].plate_bbox = bbox
                        plate['hits'] += 1
                        plate['lifespan'] = config.__TRACK_LIFESPAN__
                        break
            else:
                plate_img = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                text, text_conf = process_plate(plate_img)
                new_plate = Plate(
                    plate_bbox=bbox,
                    plate_conf=od_conf,
                    plate_color=get_plate_color(plate_img),
                    text=text,
                    text_conf=text_conf,
                    source=Source.VIDEO,
                    original_image_path=None,
                    result_image_path=processed_filename
                )
                tracked_plates.append({
                    'id': counter,
                    'plate': new_plate,
                    'hits': 1,
                    'lifespan': config.__TRACK_LIFESPAN__
                })
                active_ids.add(counter)
                counter += 1

        for plate in tracked_plates[:]:
            if plate['id'] not in active_ids:
                plate['lifespan'] -= 1
            if plate['lifespan'] <= 0:
                tracked_plates.remove(plate)

        processed_frame = frame.copy()
        for tracked_plate in tracked_plates:
            plate = tracked_plate['plate']
            processed_frame = draw_text(processed_frame, plate.text, plate.text_conf, plate.lang, plate.plate_bbox)

        out.write(processed_frame)

        ret, buffer = cv2.imencode('.jpg', processed_frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()
    out.release()


def process_plate(plate_img):
    plate_img = np.array(plate_img)

    if plate_img is None or plate_img.size == 0:
        return "", 0

    processed_plate_img = preprocess_image(plate_img)

    nep_results = ne_reader.readtext(plate_img, **ne_read_text_config)
    eng_results = en_reader.readtext(processed_plate_img, **en_read_text_config)

    validated_nep_results = validate_nepali(nep_results)
    validated_eng_results = validate_english(eng_results)

    eng_conf = max([res[2] for res in eng_results], default=0)
    nep_conf = max([res[2] for res in nep_results], default=0)

    if eng_conf >= nep_conf and validated_eng_results:
        text, confidence = validated_eng_results, eng_conf
    else:
        text, confidence = validated_nep_results, nep_conf

    return text, confidence


def process_video_frame(frame, frame_count):
    global next_plate_id, tracked_plates
    processed_frame, current_detections = process_frame(frame)
    active_ids = []
    unmatched_detections = []
    for detection in current_detections:
        x1, y1, x2, y2 = detection['coordinates']
        detection_box = [x1, y1, x2, y2]
        best_iou = config.__IOU_THRESHOLD__
        best_id = None
        for track_id, track_info in tracked_plates.items():
            if track_info['active']:
                track_box = track_info['last_box']
                iou = calculate_box_iou(detection_box, track_box)
                if iou > best_iou:
                    best_iou = iou
                    best_id = track_id
        if best_id is not None:
            tracked_plates[best_id]['last_box'] = detection_box
            tracked_plates[best_id]['text'] = detection['text']
            tracked_plates[best_id]['last_seen'] = frame_count
            tracked_plates[best_id]['hits'] += 1
            tracked_plates[best_id]['text_history'].append(detection['text'])
            tracked_plates[best_id]['language'] = detection.get('language', 'unknown')
            active_ids.append(best_id)
        else:
            unmatched_detections.append(detection)
    for detection in unmatched_detections:
        x1, y1, x2, y2 = detection['coordinates']
        tracked_plates[next_plate_id] = {
            'last_box': [x1, y1, x2, y2],
            'text': detection['text'],
            'first_seen': frame_count,
            'last_seen': frame_count,
            'hits': 1,
            'active': True,
            'text_history': deque(maxlen=config.__TRACK_MAX_SIZE__),
            'confidence': detection['confidence'],
            'language': detection.get('language', 'unknown')
        }
        tracked_plates[next_plate_id]['text_history'].append(detection['text'])
        active_ids.append(next_plate_id)
        next_plate_id += 1
    for track_id, track_info in tracked_plates.items():
        if track_info['active'] and (frame_count - track_info['last_seen'] > config.__TRACK_MAX_SIZE__):
            track_info['active'] = False
        if track_info['active'] and track_id in active_ids and track_info['hits'] >= config.__TRACK_MIN_HITS__:
            x1, y1, x2, y2 = track_info['last_box']
            if track_info['text_history']:
                text_counts = {}
                for text in track_info['text_history']:
                    if text in text_counts:
                        text_counts[text] += 1
                    else:
                        text_counts[text] = 1
                most_common_text = max(text_counts.items(), key=lambda x: x[1])[0]
                cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (57, 255, 20), 2)
                cv2.putText(
                    processed_frame,
                    f"ID:{track_id} {most_common_text}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (77, 77, 255),
                    2
                )
    return processed_frame


def process_frame(frame):
    if isinstance(frame, Image.Image):
        frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)

    od_results = model(frame)
    recognized_plates = []

    for result in od_results:
        boxes = result.boxes
        for box in boxes:
            if box.conf >= config.__OD_THRESHOLD__ and box.cls == 0:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                plate_img = frame[y1:y2, x1:x2]

                text, text_conf = process_plate(plate_img)

                plate = {
                    'coordinates': (x1, y1, x2, y2),
                    'confidence': float(box.conf),
                    'plate_color': get_plate_color(plate_img),
                    'text': text,
                    'text_confidence': text_conf,
                    'language': 'ne' if any(char in config.__ALLOWED_NEP_CHAR__ for char in text) else 'en'
                }

                recognized_plates.append(plate)

    return frame, recognized_plates