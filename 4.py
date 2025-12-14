from ultralytics import YOLO
# 加载预训练模型
model = YOLO('yolov8n.pt')

# 训练模型
results = model.train(
    data='datasets/YOLODataset/dataset.yaml',
    epochs=50,
    imgsz=640,
    batch=16,
    name='custom_yolo',
    device='cpu'   # 使用 GPU 0，CPU 则设为 'cpu
)