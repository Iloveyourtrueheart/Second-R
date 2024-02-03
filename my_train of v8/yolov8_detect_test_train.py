from ultralytics import YOLO
model = YOLO('yolov8n.yaml')
model = YOLO('runs/detect/train9/weights/last.pt')
model = YOLO('yolov8.yaml').load('runs/detect/train7/weights/last.pt')
results = model.train(data='coco128.yaml', epochs=100, imgsz=640) 
