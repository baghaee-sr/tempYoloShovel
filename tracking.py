from config import BUCKET_THRESHOLD

def update_teeth_tracking(bucket_detected, bucket_size, current_teeth_positions, tracked_teeth, next_teeth_id, annotator):
    """
    مدیریت و بروزرسانی Tracking دندان‌ها (teeth)
    """
    if bucket_detected and bucket_size >= BUCKET_THRESHOLD:
        updated_tracked_teeth = {}

        for cx, cy, x1, y1, x2, y2 in current_teeth_positions:
            matched_id = None
            min_distance = float("inf")

            # بررسی دندان‌های قبلی برای یافتن نزدیک‌ترین تطابق
            for tid, (px, py) in tracked_teeth.items():
                distance = (cx - px) ** 2 + (cy - py) ** 2
                if distance < min_distance and distance < 24000:
                    min_distance = distance
                    matched_id = tid

            # اگر دندان جدید باشد، یک ID جدید اختصاص بده
            if matched_id is None:
                matched_id = next_teeth_id
                next_teeth_id += 1

            updated_tracked_teeth[matched_id] = (cx, cy)

            # نمایش ID دندان در تصویر
            annotator.text((cx, cy), str(matched_id), txt_color=(0, 0, 0))

        return updated_tracked_teeth, next_teeth_id
    else:
        # اگر bucket کمتر از حد آستانه باشد، شمارش دندان‌ها ریست شود
        return {}, 1  # مقدار خالی و مقدار 1 برای جلوگیری از خطا
