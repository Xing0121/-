from ultralytics import YOLO
# 验证模型
model = YOLO('runs/detect/custom_yolo12/weights/best.pt')
metrics = model.val(
    data='datasets/YOLODataset/dataset.yaml',
    split='val'
)
print(f"mAP50: {metrics.box.map50}")
print(f"mAP50-95: {metrics.box.map}")
results = model('test_image.jpg')