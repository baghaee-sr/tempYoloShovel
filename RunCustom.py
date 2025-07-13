def run(
    weights="best_new_dataset.pt",  # آدرس مدل از پیش آموزش داده‌شده
    source="output_2025-01-31_11-48-58.mp4",  # آدرس فایل ویدیو یا جریان (یا عدد برای وب‌کم)
    imgsz=(640, 640),  # سایز تصاویر
    conf_thres=0.25,  # حد آستانه اطمینان
    iou_thres=0.45,  # حد آستانه IOU
    max_det=1000,  # تعداد ماکزیمم دتکشن‌ها
    device="",  # دستگاه: CUDA یا CPU
    view_img=True,  # نمایش بلادرنگ
    nosave=True,  # عدم ذخیره‌سازی
    line_thickness=2,  # ضخامت خطوط جعبه
):
    import torch
    from coreYoloV5.utils.torch_utils import select_device
    from coreYoloV5.utils.dataloaders import LoadStreams, LoadImages
    from coreYoloV5.utils.general import non_max_suppression, scale_boxes
    from coreYoloV5.utils.plots import Annotator, colors
    from coreYoloV5.models.common import DetectMultiBackend
    import cv2
    from pathlib import Path

    device = select_device(device)
    model = DetectMultiBackend(weights, device=device)
    stride, names = model.stride, model.names

    if source.isnumeric() or source.startswith("rtsp://") or source.startswith("http://"):
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    for path, img, im0s, _, _ in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.float() / 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # مدل را روی تصویر اعمال کن
        pred = model(img)
        pred = non_max_suppression(pred, conf_thres, iou_thres)

        for i, det in enumerate(pred):  # برای هر دتکشن
            im0 = im0s[i].copy() if isinstance(im0s, list) else im0s
            annotator = Annotator(im0, line_width=line_thickness)

            if det is not None and len(det):
                det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], im0.shape).round()

                for *xyxy, conf, cls in reversed(det):
                    label = f"{names[int(cls)]} {conf:.2f}"
                    annotator.box_label(xyxy, label, color=colors(int(cls), True))

            im0 = annotator.result()
            cv2.imshow(str(path), im0)
            if cv2.waitKey(1) == ord("q"):  # با زدن Q خارج می‌شود
                break

    cv2.destroyAllWindows()


run(weights="weights/best_new_dataset.pt",source="output_2025-01-31_11-48-58.mp4")
