import config
import enum
import datetime

class Source(enum.Enum):
    IMAGE = "Image"
    VIDEO = "Video"
    LIVE = "Live"

class Plate:
    def __init__(
            self,
            plate_bbox,
            plate_conf,
            plate_color,
            text,
            text_conf,
            source,
            original_image_path,
            result_image_path,
    ):
        self.plate_bbox = plate_bbox
        self.plate_conf = float(plate_conf)
        self.plate_color = plate_color
        self.text = text
        self.text_conf = float(text_conf)
        self.lang = "en" if all(c in config.__ALLOWED_ENG_CHAR__ for c in text) else "ne"
        self.source = source
        self.original_image_path = original_image_path
        self.result_image_path = result_image_path
        self.timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def clean_data(self):
        pass

    def save_to_db(self):
        self.clean_data()
        import csv
        with open(config.__DB_FILENAME__, mode='a', encoding='utf-8', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                self.plate_bbox,
                self.plate_conf,
                self.plate_color,
                self.text,
                self.text_conf,
                self.lang,
                self.source,
                self.original_image_path,
                self.result_image_path,
                self.timestamp
            ])


class PlateInfo:
    def __init__(
            self,
            plate_bbox,
            plate_conf,
            plate_color,
            text,
            text_conf,
            lang,
            source,
            original_image_path,
            result_image_path,
            timestamp
    ):
        self.plate_bbox = plate_bbox
        self.plate_conf = plate_conf
        self.plate_color = plate_color
        self.text = text
        self.text_conf = text_conf
        self.lang = lang
        self.source = source
        self.original_image_path = original_image_path
        self.result_image_path = result_image_path
        self.timestamp = timestamp


def read_from_db(
        source,
        start_date_time,
        end_date_time,
        color
):
    import csv
    plate_info_list = []
    if start_date_time is None:
        start_date_time = datetime.datetime.min
    if end_date_time is None:
        end_date_time = datetime.datetime.max
    try:
        with open(config.__DB_FILENAME__, mode='r') as file:
            reader = csv.reader(file)
            for row in reader:
                # filter by source
                if source and row[6] != source:
                    continue

                # filter by date range
                record_time = datetime.datetime.strptime(row[9], "%Y-%m-%d %H:%M:%S")
                if not (start_date_time <= record_time <= end_date_time):
                    continue

                # filter by color
                if color and row[2] != color:
                    continue

                plate_info = PlateInfo(
                    plate_bbox=row[0],
                    plate_conf=float(row[1]),
                    plate_color=row[2],
                    text=row[3],
                    text_conf=float(row[4]),
                    lang=row[5],
                    source=row[6],
                    original_image_path=row[7],
                    result_image_path=row[8],
                    timestamp=row[9]
                )
                plate_info_list.append(plate_info)
    except FileNotFoundError:
        print("Database file not found. Returning an empty list.")
    return plate_info_list[::-1]