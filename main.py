import cv2
import torch
import time
import numpy as np

def load_object_detection_model():
    print("Loading object detection model...")
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    model.eval()
    print("Model loaded successfully.")
    return model

def run_object_detection(model, frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(rgb)
    detections = results.pandas().xyxy[0] 

    for _, row in detections.iterrows():
        x1, y1, x2, y2 = map(int, [row.xmin, row.ymin, row.xmax, row.ymax])
        label = f"{row['name']} {row['confidence']:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(frame, (x1, y1 - h - 6), (x1 + w, y1), (0, 255, 0), -1)

        cv2.putText(frame, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, lineType=cv2.LINE_AA)

    return frame

def main():
    model = load_object_detection_model()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    print("Press 'q' to exit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame.")
            break

        frame = cv2.resize(frame, (640, 480))
        frame_with_boxes = run_object_detection(model, frame.copy())

        cv2.imshow('YOLOv5 Object Detection', frame_with_boxes)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
