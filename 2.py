from ultralytics import YOLO
model = YOLO('yolov8n.pt')
result_1 =model.predict(source="./videos/Video2.mp4",save=True)