from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n.pt')  # load an official model
model = YOLO('runs/detect/train9/weights/last.pt')  # load a custom trained model

# Export the model
model.export(format='torchscrip')