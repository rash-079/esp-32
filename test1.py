# detect_webcam.py
from ultralytics import YOLO
import cv2

# Load your trained model
model = YOLO("runs/detect/train11/weights/best.pt")

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run inference
    results = model(frame)

    # Visualize results
    for r in results:
        annotated_frame = r.plot()
        cv2.imshow("Webcam Detection", annotated_frame)

    # Quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
