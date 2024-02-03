from ultralytics import YOLO

# Load a model
model = YOLO('runs/detect/train7/weights/last.pt')  # load a partially trained model

# Resume training
results = model.train(resume=True, epochs=50)