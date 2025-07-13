# core/bucket_monitor.py

import time

class BucketMonitor:
    def __init__(self, config):
        self.config = config
        self.in_discharge = False
        self.discharge_start_time = None
        self.discharge_start_frame = None
        self.max_teeth_count = 0
        self.last_status = "بررسی سیستم"
        self.current_frame_num = 0
        self._event_msg = ""  # پیام لحظه‌ای وسط

    def reset_discharge(self):
        self.in_discharge = False
        self.discharge_start_time = None
        self.discharge_start_frame = None
        self.max_teeth_count = 0

    def update(self, bucket_area, teeth_areas, motion_change):
        self.current_frame_num += 1

        if not self.in_discharge:
            if bucket_area > self.config["bucket_area_threshold"]:
                self.in_discharge = True
                self.discharge_start_time = time.time()
                self.discharge_start_frame = self.current_frame_num
                self.max_teeth_count = 0
                self._event_msg = "ورود به فاز تخلیه"
        else:
            teeth_count = sum([area > self.config["tooth_area_threshold"] for area in teeth_areas])
            if teeth_count > self.max_teeth_count:
                self.max_teeth_count = teeth_count

            if bucket_area < self.config["bucket_area_exit_threshold"]:
                discharge_time = time.time() - self.discharge_start_time
                discharge_frames = self.current_frame_num - self.discharge_start_frame

                time_ok = discharge_time >= self.config["bucket_min_discharge_time"]
                frame_ok = discharge_frames >= self.config["bucket_min_discharge_frames"]

                if (time_ok or frame_ok):
                    if self.max_teeth_count < self.config["tooth_min_count"]:
                        self.last_status = "هشدار: تعداد دندان‌ها کم است!"
                        self._event_msg = "تخلیه کامل" + " تعداد دندانه ها:" + str(self.max_teeth_count)
                    else:
                        self.last_status = "سلامت سیستم"
                        self._event_msg = "تخلیه کامل" + " تعداد دندانه ها:" + str(self.max_teeth_count)
                else:
                    # وضعیت سیستم تغییر نمی‌کند، فقط یک پیام کوتاه می‌دهیم
                    self._event_msg = "تخلیه ناقص" + " تعداد دندانه ها:" + str(self.max_teeth_count)

                self.reset_discharge()

    def get_status(self):
        return self.last_status

    def get_event(self):
        return self._event_msg

    def clear_event(self):
        self._event_msg = ""
