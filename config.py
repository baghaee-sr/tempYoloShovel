# config.py

USE_CAMERA = False  # اگر True باشد، از دوربین استفاده می‌شود، در غیر این صورت از ویدیو
CAMERA_IP = "192.168.88.10"
STREAM_URL = f"rtsp://{CAMERA_IP}/mainstream"

WEIGHTS_PATH = "weights/best_new_dataset.pt"
VIDEO_PATH = "output_2025-01-31_11-48-58.mp4"

BUCKET_THRESHOLD = 270000  # مقدار آستانه برای bucket
DRAW_BOXES = False  # نمایش باکس‌های تشخیص
TEETH_THRESHOLD = 6  # حداقل تعداد teeth مورد نیاز
IOU_THRESHOLD = 0.45  # مقدار IOU برای NMS در YOLO
CONF_THRESHOLD = 0.65  # مقدار Confidence برای قبول تشخیص‌ها
IMG_SIZE = 320  # 👈 تغییر سایز ورودی مدل YOLOv5

TEETH_AREA_THRESHOLD = 1700  # حداقل مساحت مجاز برای شناسایی ناخن‌ها

MIN_DUMP_TIME = 5  # حداقل مدت زمان تخلیه به ثانیه
