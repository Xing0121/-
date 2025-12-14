from ultralytics import YOLO
model = YOLO('yolo11n.pt')
result_1=model.predict(source="./ultralytics/assets/bus.jpg",save=True)
result_2 =model.predict(source="./ultralytics/assets/zidane.jpg",save=True)
result_3 =model.predict(source="./images/IMG_20251122_184526.jpg",save=True)