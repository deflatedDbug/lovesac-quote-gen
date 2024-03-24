from ultralytics import YOLO

model = YOLO("runs/detect/train/weights/best.pt")

results = model.train(data="config.yaml", epochs=1000) 