config = {
    # --- تنظیمات ورودی و مدل ---
    "input_source": "files/output_2025-01-31_11-48-58.mp4",       # مسیر فایل ویدیو یا 0 برای وب‌کم
    "model_path": "assets/weights/img1024_notSorting.pt",        # مسیر فایل مدل YOLO

    # --- آستانه‌های باکت (Bucket Thresholds) ---
    "bucket_area_threshold": 165000,                # آستانه ورود به فاز تخلیه (برحسب پیکسل)
    "bucket_area_exit_threshold": 113000,           # آستانه خروج از فاز تخلیه (برحسب پیکسل)
    "bucket_min_discharge_time": 2.0,              # حداقل مدت زمان تخلیه (برحسب ثانیه)
    "bucket_min_discharge_frames": 60,             # حداقل فریم مورد نیاز برای تایید تخلیه (اختیاری)

    # --- آستانه‌های دندانه (Teeth Thresholds) ---
    "tooth_area_threshold": 2600,                   # حداقل مساحت دندانه برای شمارش (پیکسل)
    "tooth_min_count": 5,                          # حداقل تعداد دندانه بزرگ برای سلامت سیستم

    # --- آستانه تغییرات تصویر (Motion Detection) ---
    "motion_threshold": 20,                        # حداقل تغییرات تصویر برای فعال بودن ماشین (قابل تنظیم)

    # --- سایر تنظیمات پردازش ---
    "enable_day": True,                            # فعال بودن پردازش برای تصاویر روز
    "enable_night": False,                         # فعال بودن پردازش برای تصاویر شب
    "draw_boxes": True,                            # نمایش باکس دور اشیا در تصویر

    # --- تنظیمات ظاهری و عمومی ---
    "fps": 30,                                     # نرخ نمایش فریم (پیشنهادی، برحسب میلی‌ثانیه)
    "language": "fa",                              # زبان برنامه (fa/en) - قابل توسعه

    # --- تنظیمات ذخیره تصویر ---
    "capture_enabled": True,  # فعال بودن ذخیره تصویر
    "capture_interval_sec": 10,  # هر چند ثانیه یک‌بار (هنگام فعال بودن ماشین)
    "capture_image_width": 640,  # عرض تصویر ذخیره‌شده (پیکسل)
    "capture_image_height": 480,  # ارتفاع تصویر ذخیره‌شده (پیکسل)


}


