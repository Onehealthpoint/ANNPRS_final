import os
import cv2
import config
from PIL import Image
from db import *
from utlis import *
from config import *
from helper import *
from werkzeug.utils import secure_filename
# from sort.sort import *
from intermediate import *
from flask import render_template
from validator import validate_nepali, validate_english

# tracker = Sort()

tracked_plates = []
processed_plate_ids = set()
active_ids = set()
counter = 0

def process_image(image_file):
    filename = secure_filename(image_file.filename)
    filepath = os.path.join(config.__UPLOAD_FOLDER__, filename)
    image_file.save(filepath)

    img = Image.open(filepath)

    od_results = model(img)
    recognized_plates = []

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

    processed_filename = f"processed_{filename}"
    processed_filepath = os.path.join(config.__PROCESSED_FOLDER__, processed_filename)

    processed_img = img.copy()

    for plate in recognized_plates:
        processed_img = draw_text(processed_img, plate.text, plate.text_conf, plate.lang, plate.plate_bbox)
        plate.recognized_plates = processed_filename
        plate.save_to_db()

    Image.fromarray(processed_img).save(processed_filepath)

    return render_template(
        'image_result.html',
        original_image=filename,
        processed_image=processed_filename,
        plates=recognized_plates
    )

def process_video(video):
    global tracked_plates, processed_plate_ids, counter

    tracked_plates = []
    processed_plate_ids = set()
    counter = 1

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

    recognized_plates = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        processed_frames = process_video_frame(frame)
        for plate in processed_frames:
            plate.original_image_path = filename
            plate.result_image_path = processed_filename
            plate.save_to_db()

            plate_id = counter
            counter += 1
            tracked_plates.append({
                'id': plate_id,
                'plate': plate,
                'hits': 1,
                'lifespan': config.__TRACK_LIFESPAN__
            })
            processed_plate_ids.add(plate_id)

        processed_frame = frame.copy()
        for tracked_plate in tracked_plates:
            if tracked_plate['id'] in active_ids:
                plate = tracked_plate['plate']
                processed_frame = draw_text(processed_frame, plate.text, plate.text_conf, plate.lang, plate.plate_bbox)

        out.write(processed_frame)
        frame_count += 1

    cap.release()
    out.release()

    return render_template('video_result.html',
                           original_video=filename,
                           processed_video=processed_filename,
                           plates=recognized_plates)

def generate_realtime_frame():
    cap = cv2.VideoCapture(0)
    frame_count = -1

    while True:
        frame_count += 1
        success, frame = cap.read()
        if not success:
            break

        processed_frame = process_video_frame(frame)

        ret, buffer = cv2.imencode('.jpg', processed_frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

def process_plate(plate_img):
    plate_img = np.array(plate_img)

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

def process_video_frame(frame):
    global active_ids, tracked_plates

    new_recognized_plates = []
    active_ids = set()

    od_results = model(frame)
    for result in od_results:
        boxes = result.boxes
        for box in boxes:
            if box.conf < config.__OD_THRESHOLD__ or box.cls == 0:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            tracked_id = is_plate_tracked((x1, y1, x2, y2), tracked_plates)
            if tracked_id is not None:
                active_ids.add(tracked_id)
                continue

            plate_img = frame[y1:y2, x1:x2]
            text, text_conf = process_plate(plate_img)
            new_recognized_plates.append(Plate(
                plate_bbox=(x1, y1, x2, y2),
                plate_conf=float(box.conf),
                plate_color=get_plate_color(plate_img),
                text=text,
                text_conf=text_conf,
                source=Source.VIDEO,
                original_image_path=None,
                result_image_path=None
            ))

    for plate in tracked_plates:
        if plate['id'] in active_ids:
            plate['hits'] += 1
            plate['lifespan'] = config.__TRACK_LIFESPAN__
        else:
            plate['lifespan'] -= 1
        if plate['lifespan'] <= 0:
            tracked_plates.remove(plate)

    return new_recognized_plates