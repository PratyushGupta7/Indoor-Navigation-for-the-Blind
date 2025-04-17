import cv2
import torch
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# Load camera calibration
K = np.load('camera_matrix.npy')
dist = np.load('distortion_coeffs.npy')

device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"Using device: {device}")

def load_object_detection_model(device):
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    model.to(device)
    model.eval()
    return model

def load_depth_model(device):
    model_type = "DPT_Hybrid"
    midas = torch.hub.load("intel-isl/MiDaS", model_type, pretrained=True)
    midas.to(device)
    midas.eval()
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = midas_transforms.dpt_transform if "DPT" in model_type else midas_transforms.small_transform
    return midas, transform

def run_depth_estimation(midas, transform, image, device):
    input_tensor = transform(image).to(device)
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
    depth_map = cv2.medianBlur(depth_map.astype(np.float32), 5)
    return depth_map

def run_object_detection(model, frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(rgb)
    detections = results.pandas().xyxy[0]
    return detections

def mask_dynamic_objects(detections, frame_shape):
    """ Create a mask where dynamic objects like person, car, etc. are detected """
    mask = np.ones(frame_shape[:2], dtype=np.uint8)
    dynamic_classes = ['person', 'car', 'dog', 'cat', 'bicycle', 'motorcycle']  # add more if needed
    for _, row in detections.iterrows():
        label = row['name']
        if label in dynamic_classes:
            x1, y1, x2, y2 = map(int, [row.xmin, row.ymin, row.xmax, row.ymax])
            mask[y1:y2, x1:x2] = 0
    return mask

def draw_visual_odometry(prev_pts, curr_pts, frame):
    vo_frame = frame.copy()
    for p_old, p_new in zip(prev_pts, curr_pts):
        a, b = p_old.ravel()
        c, d = p_new.ravel()
        cv2.line(vo_frame, (int(a), int(b)), (int(c), int(d)), (255, 0, 0), 2)
        cv2.circle(vo_frame, (int(c), int(d)), 3, (0, 255, 0), -1)
    return vo_frame

def main():
    model = load_object_detection_model(device)
    midas, midas_transform = load_depth_model(device)

    orb = cv2.ORB_create(1000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    frame_count = 0
    prev_gray = None
    prev_kp = None
    prev_des = None
    prev_depth = None
    prev_mask = None
    trajectory = []
    pose = np.zeros((3, 1))

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame.")
            break

        h, w = frame.shape[:2]
        new_K, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 1, (w, h))
        undistorted = cv2.undistort(frame, K, dist, None, new_K)

        # Depth + Detection
        rgb = cv2.cvtColor(undistorted, cv2.COLOR_BGR2RGB)
        depth_map = run_depth_estimation(midas, midas_transform, rgb, device)
        detections = run_object_detection(model, undistorted)
        mask = mask_dynamic_objects(detections, frame.shape)

        # Feature detection
        gray = cv2.cvtColor(undistorted, cv2.COLOR_BGR2GRAY)
        kp, des = orb.detectAndCompute(gray, None)

        vo_frame = undistorted.copy()

        if prev_gray is not None and prev_des is not None and len(prev_kp) > 0 and len(kp) > 0:
            matches = bf.match(prev_des, des)
            matches = sorted(matches, key=lambda x: x.distance)

            pts1 = np.float32([prev_kp[m.queryIdx].pt for m in matches]).reshape(-1, 2)
            pts2 = np.float32([kp[m.trainIdx].pt for m in matches]).reshape(-1, 2)

            # Apply mask to remove dynamic object points
            valid_idx = [i for i, pt in enumerate(pts1) if prev_mask[int(pt[1]), int(pt[0])] > 0]
            pts1 = pts1[valid_idx]
            pts2 = pts2[valid_idx]

            if len(pts1) >= 8:
                # Estimate Essential Matrix
                E, mask_E = cv2.findEssentialMat(pts1, pts2, new_K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
                _, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, new_K)

                # Scale translation using depth
                scale_estimates = []
                for pt, dp in zip(pts1, prev_depth[pts1[:,1].astype(int), pts1[:,0].astype(int)]):
                    if dp > 0:
                        scale_estimates.append(dp)
                if scale_estimates:
                    avg_scale = np.median(scale_estimates)
                    t = t * avg_scale  # Scale the translation properly
                else:
                    t = t * 0.01  # fallback small translation

                # Update pose
                pose += R @ t
                trajectory.append(pose.copy())

                # Draw tracks
                vo_frame = draw_visual_odometry(pts1, pts2, vo_frame)

        # Visualization
        depth_norm = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
        depth_color = cv2.applyColorMap(depth_norm.astype(np.uint8), cv2.COLORMAP_MAGMA)
        blended = cv2.addWeighted(undistorted, 0.6, depth_color, 0.4, 0)

        cv2.imshow('YOLO + Depth', blended)
        cv2.imshow('Visual Odometry', vo_frame)

        prev_gray = gray.copy()
        prev_kp = kp
        prev_des = des
        prev_depth = depth_map.copy()
        prev_mask = mask.copy()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
