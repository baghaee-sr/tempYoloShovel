import cv2
import time
import os
from datetime import datetime

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("دوربین باز نشد!")
    exit()

image_counter = 0

cv2.namedWindow('Camera', cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty('Camera', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

try:
    while True:
        ret, frame = cap.read()

        if not ret:
            print("ناتوان در دریافت فریم از دوربین!")
            break

        cv2.imshow('Camera', frame)

        if image_counter == 0 or time.time() - start_time >= 600:
            today_date = datetime.now().strftime("%Y-%m-%d")
            save_folder = os.path.join("images", today_date)

            if not os.path.exists(save_folder):
                os.makedirs(save_folder)

            image_name = f"image_{image_counter}.jpg"
            save_path = os.path.join(save_folder, image_name)
            cv2.imwrite(save_path, frame)
            print(f"عکس {save_path} ذخیره شد.")
            image_counter += 1
            start_time = time.time()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
