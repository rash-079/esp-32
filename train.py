from ultralytics import YOLO
model=YOLO("yolov8m.pt")
model.train(data="data.yaml", epochs=100, imgsz=135, batch=15, device="cpu")
