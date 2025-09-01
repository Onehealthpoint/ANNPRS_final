import config
import cv2
import numpy as np
from PIL import Image, ImageDraw

def draw_text(plate, text, confidence, lang, bbox):
    if plate is None or plate.size == 0 or text is None or len(text) == 0:
        return plate

    plate = np.array(plate)

    overlay = plate.copy()
    output = plate.copy()

    if lang == 'en':
        font = cv2.FONT_HERSHEY_SIMPLEX
    else:
        font = config.__NEP_FONT_PATH__
    font_scale = 0.6
    font_thickness = 2

    text_str = f"{text}%"
    text_size, _ = cv2.getTextSize(text_str, 2, font_scale, font_thickness)
    text_width, text_height = text_size

    x, y, w, h = bbox
    text_x = max(x, 0)
    text_y = max(y - 10, text_height + 10)

    cv2.rectangle(overlay, (text_x, text_y - text_height - 5), (text_x + text_width, text_y + 5), (0, 0, 0), -1)
    alpha = 0.6
    cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)

    cv2.putText(output, text_str, (text_x, text_y), 3, font_scale, (255, 255, 255), font_thickness)
    # if lang == 'en':
    #     pass
    # else:
    #     image_pil = Image.fromarray(output)
    #     draw = ImageDraw.Draw(image_pil)
    #     text_color = (255, 255, 255)  # Default to white
    #     draw.text((x, y), text, font=font, fill=text_color)
    #     output = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

    return output

def calculate_box_iou(box1, box2):
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    x_left = max(x1_1, x1_2)
    y_top = max(y1_1, y1_2)
    x_right = min(x2_1, x2_2)
    y_bottom = min(y2_1, y2_2)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection = (x_right - x_left) * (y_bottom - y_top)

    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)

    return intersection / float(box1_area + box2_area - intersection)

def is_plate_tracked(bbox, tracked_plates):
    for tracked_plate in tracked_plates:
        iou = calculate_box_iou(bbox, tracked_plate.plate.plate_bbox)
        if iou > config.__IOU_THRESHOLD__:
            return tracked_plate.id
    return None