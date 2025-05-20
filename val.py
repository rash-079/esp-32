from ultralytics import YOLO

model = YOLO("runs/detect/train/weights/best.pt")
metrics = model.val()  # This will print mAP, precision, recall
