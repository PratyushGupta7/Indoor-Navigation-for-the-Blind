import cv2
import torch
import numpy as np
import warnings
from collections import deque
import math  # for atan2, degrees
import os

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


if os.environ.get("USE_TTS", "1") == "1":
    import pyttsx3
    tts_engine = pyttsx3.init()
    tts_engine.setProperty('rate', 150)  # Set speech rate (optional)
else:
    tts_engine = None

# --- Configuration ---
try:
    K = np.load('camera_matrix.npy')
    dist = np.load('distortion_coeffs.npy')
    print("Camera calibration files loaded successfully.")
except FileNotFoundError:
    print("Error: camera_matrix.npy or distortion_coeffs.npy not found.")
    print("Please run a camera calibration script first.")
    K = np.eye(3)
    dist = np.zeros(5)
    K[0, 2] = 320
    K[1, 2] = 240
    K[0, 0] = 400
    K[1, 1] = 400
    print("Using placeholder camera calibration. Results will be inaccurate.")

device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"Using device: {device}")

# --- Model Loading ---

def load_object_detection_model(device):
    try:
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        model.to(device)
        model.eval()
        print("YOLOv5 model loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading YOLOv5 model: {e}")
        exit()

def load_depth_model(device):
    try:
        model_type = "DPT_Hybrid"
        midas = torch.hub.load("intel-isl/MiDaS", model_type, pretrained=True)
        midas.to(device)
        midas.eval()
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        transform = midas_transforms.dpt_transform if "DPT" in model_type else midas_transforms.small_transform
        print("MiDaS depth model loaded successfully.")
        return midas, transform
    except Exception as e:
        print(f"Error loading MiDaS model: {e}")
        exit()

# --- Natural Language Instruction Generator ---

def generate_instructions(waypoints, step_size=15, angle_threshold=30):
    """
    Converts a list of pixel-based waypoints into simple NL steps.
    Groups consecutive identical actions and reports approximate pixel distance.
    """
    if len(waypoints) < 2:
        return ["No path computed"]
    instructions = []
    # Initial heading: "up" in image coords is toward decreasing y
    heading = np.array([0.0, -1.0])
    current_action = None
    count = 0

    for i in range(1, len(waypoints)):
        prev = np.array(waypoints[i-1], dtype=float)
        curr = np.array(waypoints[i], dtype=float)
        vec = curr - prev
        norm = np.linalg.norm(vec)
        if norm < 1e-3:
            continue
        vec_norm = vec / norm

        # Compute signed angle between heading and vec_norm
        angle = math.degrees(
            math.atan2(vec_norm[1], vec_norm[0]) - math.atan2(heading[1], heading[0])
        )
        angle = (angle + 180) % 360 - 180  # wrap to [-180,180]

        if abs(angle) < angle_threshold:
            action = "Move forward"
            # heading unchanged
        elif angle > 0:
            action = "Turn right and move forward"
            heading = vec_norm
        else:
            action = "Turn left and move forward"
            heading = vec_norm

        if action == current_action:
            count += 1
        else:
            if current_action is not None:
                dist_px = count * step_size
                instructions.append(f"{current_action} ~{dist_px:.0f} px")
            current_action = action
            count = 1

    if current_action is not None:
        dist_px = count * step_size
        instructions.append(f"{current_action} ~{dist_px:.0f} px")

    return instructions

# --- Core Processing Functions ---

def run_depth_estimation(midas, transform, image_rgb, device):
    input_tensor = transform(image_rgb).to(device)
    if input_tensor.dim() == 3:
        input_tensor = input_tensor.unsqueeze(0)
    with torch.no_grad():
        prediction = midas(input_tensor)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=image_rgb.shape[:2],
            mode="bicubic",
            align_corners=False
        ).squeeze()
    depth_map = prediction.cpu().numpy()
    depth_map = cv2.medianBlur(depth_map.astype(np.float32), 5)
    dmin, dmax = depth_map.min(), depth_map.max()
    if dmax - dmin > 1e-6:
        normalized = (depth_map - dmin) / (dmax - dmin)
    else:
        normalized = np.zeros_like(depth_map)
    return normalized

def run_object_detection(model, frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(rgb)
    df = results.pandas().xyxy[0]
    objects = []
    for _, row in df.iterrows():
        x1, y1, x2, y2 = map(int, [row['xmin'], row['ymin'], row['xmax'], row['ymax']])
        cx, cy = (x1 + x2)//2, (y1 + y2)//2
        objects.append({
            'label': row['name'],
            'confidence': row['confidence'],
            'bbox': (x1, y1, x2, y2),
            'center': (cx, cy),
            'size': (x2 - x1, y2 - y1)
        })
    return objects, df

def process_visual_odometry(prev_gray, curr_gray, prev_kp, prev_des, K):
    orb = cv2.ORB_create(1000)
    kp, des = orb.detectAndCompute(curr_gray, None)
    R = t = None
    pts1 = pts2 = None

    if prev_gray is not None and prev_des is not None and prev_kp is not None:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = sorted(bf.match(prev_des, des), key=lambda m: m.distance)
        good = matches[:max(int(len(matches)*0.7), 20)]
        if len(good) >= 8:
            pts1 = np.float32([prev_kp[m.queryIdx].pt for m in good]).reshape(-1,1,2)
            pts2 = np.float32([kp[m.trainIdx].pt for m in good]).reshape(-1,1,2)
            E, mask = cv2.findEssentialMat(pts2, pts1, K, cv2.RANSAC, 0.999, 1.0, 1000)
            if E is not None and mask is not None:
                inliers = mask.ravel()==1
                p1, p2 = pts1[inliers], pts2[inliers]
                if len(p1) >= 8:
                    _, Rr, tr, _ = cv2.recoverPose(E, p2, p1, K)
                    if np.isfinite(Rr).all() and np.isfinite(tr).all():
                        R, t = Rr, tr
                pts1, pts2 = p1, p2
    else:
        if prev_gray is None:
            print("VO: waiting for first frame.")
        elif des is None or len(kp)==0:
            print("VO: no features detected.")
        else:
            print("VO: no prev features.")

    return kp, des, R, t, pts1, pts2

def update_object_depth(objects, depth_map):
    h, w = depth_map.shape
    for o in objects:
        cx, cy = o['center']
        if 0<=cx<w and 0<=cy<h:
            o['depth'] = float(np.clip(depth_map[cy, cx], 0.0, 1.0))
        else:
            o['depth'] = None
    return objects

def plan_trajectory(objects, current_pos, depth_map, frame_shape):
    h, w = frame_shape[:2]
    obstacle_map = np.zeros((h, w), dtype=np.uint8)
    cost_base = np.zeros((h, w), dtype=np.float32)
    margin = 25
    for o in objects:
        if o.get('depth') is not None:
            x1,y1,x2,y2 = o['bbox']
            x1m, y1m = max(0, x1-margin), max(0, y1-margin)
            x2m, y2m = min(w, x2+margin), min(h, y2+margin)
            obstacle_map[y1m:y2m, x1m:x2m] = 255

    dt = cv2.distanceTransform(255-obstacle_map, cv2.DIST_L2, 5)
    cv2.normalize(dt, dt, 0, 1.0, cv2.NORM_MINMAX)
    obs_cost = (1.0-dt)*200
    depth_cost = depth_map * 55
    final_cost = cost_base + obs_cost + depth_cost
    final_cost[obstacle_map==255] = 10000.0
    cost_vis = cv2.normalize(final_cost, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    start = current_pos
    goal = (w//2, h//4)
    waypoints = [start]
    visited = {start}
    step = 15
    thresh = step*1.5
    for _ in range(100):
        if np.linalg.norm(np.array(start)-np.array(goal))<thresh:
            waypoints.append(goal)
            break
        best = float('inf')
        nxt = None
        for dx in (-step,0,step):
            for dy in (-step,0,step):
                if dx==0 and dy==0: continue
                x,y = start[0]+dx, start[1]+dy
                if 0<=x<w and 0<=y<h and obstacle_map[y,x]==0 and (x,y) not in visited:
                    cost = final_cost[y,x] + 1.5*np.linalg.norm(np.array((x,y))-np.array(goal))
                    if cost<best:
                        best, nxt = cost, (x,y)
        if nxt is None:
            break
        waypoints.append(nxt)
        visited.add(nxt)
        start = nxt
    return waypoints, obstacle_map, cost_vis

# --- Visualization with Instruction Overlay ---

def visualize_combined(frame, objects, matched1=None, matched2=None, instructions=None):
    out = frame.copy()
    h, w = out.shape[:2]
    # Draw YOLO
    for o in objects:
        x1,y1,x2,y2 = o['bbox']
        cv2.rectangle(out, (x1,y1), (x2,y2), (0,255,0), 2)
        text = f"{o['label']} {o['confidence']:.2f}"
        if o.get('depth') is not None:
            text += f" | d={o['depth']:.2f}"
        tw, th = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5,1)[0]
        cv2.rectangle(out, (x1,y1-th-4), (x1+tw+4, y1), (0,0,0), -1)
        cv2.putText(out, text, (x1+2,y1-2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255),1)

    # Draw VO matches
    if matched1 is not None and matched2 is not None:
        num = min(len(matched1), 50)
        idxs = np.linspace(0, len(matched1)-1, num, dtype=int)
        for i in idxs:
            p1 = tuple(map(int, matched1[i].ravel()))
            p2 = tuple(map(int, matched2[i].ravel()))
            if 0<=p1[0]<w and 0<=p1[1]<h and 0<=p2[0]<w and 0<=p2[1]<h:
                cv2.line(out, p1, p2, (0,165,255), 1)
                cv2.circle(out, p2, 3, (0,0,255), -1)

    # Instruction overlay
    if instructions:
        y = 30
        cv2.putText(out, "Nav Instructions:", (10,y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0),2)
        for instr in instructions[:3]:
            y += 25
            cv2.putText(out, instr, (10,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255),1)

    return out

def visualize_trajectory(frame_shape, history, current_pos, waypoints=None, obs_map=None, cost_vis=None):
    h, w = frame_shape[:2]
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    if cost_vis is not None:
        colored = cv2.applyColorMap(cost_vis, cv2.COLORMAP_JET)
        canvas = cv2.addWeighted(canvas, 0.5, colored, 0.5, 0)
    for x in range(0,w,50):
        cv2.line(canvas, (x,0), (x,h), (70,70,70),1)
    for y in range(0,h,50):
        cv2.line(canvas, (0,y), (w,y), (70,70,70),1)
    if obs_map is not None:
        overlay = np.zeros_like(canvas)
        overlay[obs_map>0] = (0,0,200)
        canvas = cv2.addWeighted(canvas, 0.7, overlay, 0.3, 0)
    pts = list(history)
    for i in range(1,len(pts)):
        cv2.line(canvas, tuple(map(int,pts[i-1])), tuple(map(int,pts[i])), (255,255,0),2)
    if waypoints and len(waypoints)>1:
        for i in range(1,len(waypoints)):
            cv2.line(canvas, tuple(map(int,waypoints[i-1])), tuple(map(int,waypoints[i])), (0,255,0),2)
        for p in waypoints:
            cv2.circle(canvas, tuple(map(int,p)),5,(0,0,255),-1)
            cv2.circle(canvas, tuple(map(int,p)),6,(255,255,255),1)
    if current_pos:
        cv2.circle(canvas, tuple(map(int,current_pos)),7,(0,255,255),-1)
        cv2.circle(canvas, tuple(map(int,current_pos)),8,(0,0,0),1)
    cv2.putText(canvas, "Trajectory View", (10,30), cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)
    return canvas

# --- Main Loop ---

YOLO = load_object_detection_model(device)
MIDAS, TRANSFORM = load_depth_model(device)

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: cannot open camera.")
        return

    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera resolution: {fw}x{fh}")

    prev_gray = prev_kp = prev_des = None
    new_K = K.copy()
    traj_hist = deque(maxlen=150)
    origin = (fw//2, fh//2)
    curr_vis = origin
    traj_hist.append(curr_vis)
    cum_R = np.eye(3)
    cum_t = np.zeros((3,1))
    scale = 1.0
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count == 0:
            h, w = frame.shape[:2]
            new_K, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w,h),1,(w,h))
            print("Optimal K:\n", new_K)

        undist = cv2.undistort(frame, K, dist, None, new_K)
        rgb = cv2.cvtColor(undist, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(undist, cv2.COLOR_BGR2GRAY)

        depth_map = run_depth_estimation(MIDAS, TRANSFORM, rgb, device)
        objects, _ = run_object_detection(YOLO, undist)
        objects = update_object_depth(objects, depth_map)

        kp, des, R, t, m1, m2 = process_visual_odometry(prev_gray, gray, prev_kp, prev_des, new_K)
        if R is not None and t is not None:
            cum_t += cum_R @ (scale * t)
            cum_R = cum_R @ R
            vis_x = int(origin[0] + cum_t[0,0])
            vis_y = int(origin[1] - cum_t[2,0])
            vis_x = np.clip(vis_x, 0, fw-1)
            vis_y = np.clip(vis_y, 0, fh-1)
            curr_vis = (vis_x, vis_y)
            traj_hist.append(curr_vis)

        waypoints, obs_map, cost_vis = plan_trajectory(objects, curr_vis, depth_map, undist.shape)
        instructions = generate_instructions(waypoints)
        print(instructions)
        if tts_engine and instructions and instructions[0] != "No path computed":
            to_say = instructions[0]  # Say the first instruction
            tts_engine.say(to_say)
            tts_engine.runAndWait()

        combined = visualize_combined(undist, objects, m1, m2, instructions)
        traj_view = visualize_trajectory(undist.shape, traj_hist, curr_vis, waypoints, obs_map, cost_vis)

        cv2.imshow('YOLO + VO View', combined)
        cv2.imshow('Trajectory Planning View', traj_view)

        prev_gray = gray.copy()
        prev_kp, prev_des = kp, des
        frame_count += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Exiting and cleaned up.")

if __name__ == "__main__":
    main()
