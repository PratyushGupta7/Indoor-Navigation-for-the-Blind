import cv2
import torch
import time
import numpy as np

# Load YOLOv5
def load_object_detection_model():
    print("Loading object detection model...")
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    model.eval()
    print("YOLOv5 model loaded.")
    return model

def load_depth_model():
    print("Loading depth estimation model...")
    model_type = "DPT_Hybrid"

    midas = torch.hub.load("intel-isl/MiDaS", model_type, pretrained=True)
    midas.eval()

    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = midas_transforms.dpt_transform if "DPT" in model_type else midas_transforms.small_transform

    print("Depth model loaded.")
    return midas, transform

def run_depth_estimation(midas, transform, image):
    input_tensor = transform(image)
    if input_tensor.dim() == 3:
        input_tensor = input_tensor.unsqueeze(0)

    with torch.no_grad():
        prediction = midas(input_tensor)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=image.shape[:2],
            mode="bicubic",
            align_corners=False
        ).squeeze()

    depth_map = prediction.cpu().numpy()
    return depth_map

def run_object_detection(model, midas, midas_transform, frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    depth_map = run_depth_estimation(midas, midas_transform, rgb)
    
    depth_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
    depth_colored = cv2.applyColorMap(depth_normalized.astype(np.uint8), cv2.COLORMAP_MAGMA)

    blended = cv2.addWeighted(frame, 0.6, depth_colored, 0.4, 0)

    results = model(rgb)
    detections = results.pandas().xyxy[0]

    for _, row in detections.iterrows():
        x1, y1, x2, y2 = map(int, [row.xmin, row.ymin, row.xmax, row.ymax])
        label = row['name']
        confidence = row['confidence']

        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        depth_value = depth_map[cy, cx]
        depth_value = np.clip(depth_value, 1e-3, None)

        label_text = f"{label} {confidence:.2f} | Depth: {depth_value:.2f}"

        cv2.rectangle(blended, (x1, y1), (x2, y2), (0, 255, 0), 2)
        (w, h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(blended, (x1, y1 - h - 6), (x1 + w, y1), (0, 255, 0), -1)
        cv2.putText(blended, label_text, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, lineType=cv2.LINE_AA)

    return blended


def main():
    model = load_object_detection_model()
    midas, midas_transform = load_depth_model()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    frame_count = 0
    annotated_frame = None

    print("Press 'q' to exit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame.")
            break

        frame = cv2.resize(frame, (640, 480))

        if frame_count % 5 == 0:
            annotated_frame = run_object_detection(model, midas, midas_transform, frame.copy())

        frame_count += 1

        if annotated_frame is not None:
            cv2.imshow('YOLOv5 + Depth Estimation', annotated_frame)
        else:
            cv2.imshow('YOLOv5 + Depth Estimation', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
