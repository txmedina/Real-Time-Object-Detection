# txmedina
#
# Simple Object Identifier:
# 
# implemented using Python, Co-Pilot, and YOLO model for detection.
#
# next update: Allow for more object classifications with more models

import cv2
from ultralytics import YOLO

# yolo is a pre-trained model, but we can create our own
model = YOLO("yolov8s.pt")

# allows webcam feature
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("User Webcam Undetected.")
    exit()

print("Identifying Objects in View...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    # allows visualizations of the object dimensions
    annotated_frame = results[0].plot()

    cv2.imshow("Real-Time Object Identifier", annotated_frame)

    # User ESC to exit program
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
