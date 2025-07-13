# core/utils.py
import cv2
import numpy as np

def is_day(frame):
    """یک الگوریتم خیلی ساده برای تشخیص روز/شب (قابل بهبود!)"""
    # فرض: اگر میانگین روشنایی بالاست، روز است
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)
    return brightness > 90  # این عدد قابل تغییر است
