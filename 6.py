from ultralytics import YOLO
model = YOLO('runs/detect/custom_yolo23/weights/best.pt')
result_1 =model.predict(source="./videos/Video3.mp4",save=True)