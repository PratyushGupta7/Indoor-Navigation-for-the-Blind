import cv2
import torch
import numpy as np
import warnings
from collections import deque
import math
import pyttsx3
import threading
import queue
import time

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# --- TTS Manager ---
class TTSManager(threading.Thread):
    def __init__(self):
        super().__init__()
        self.queue = queue.Queue()
        self.stop_event = threading.Event()
        self.tts_engine = None
        self.daemon = True
        self._initialized = threading.Event()

    def _initialize_engine(self):
        try:
            self.tts_engine = pyttsx3.init()
            self.tts_engine.setProperty('rate', 140)
            self._initialized.set()
            print("TTS Engine Initialized in its thread.")
        except Exception as e:
            print(f"Failed to initialize TTS engine: {e}")
            self.tts_engine = None

    def run(self):
        self._initialize_engine()
        if not self.tts_engine:
            print("TTS thread exiting due to initialization failure.")
            return
        while not self.stop_event.is_set():
            instruction = None
            try:
                instruction = self.queue.get(block=True, timeout=0.2)
                if instruction:
                    print(f"TTS Speaking: {instruction}")
                    self.tts_engine.say(instruction)
                    self.tts_engine.runAndWait()
                    self.queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"TTS run loop error: {type(e).__name__} - {e}")
                if instruction is not None:
                    try:
                        if self.queue.unfinished_tasks > 0:
                            self.queue.task_done()
                    except ValueError:
                        pass
                    except Exception as E:
                        print(f"Error marking task done after TTS error: {E}")
                time.sleep(0.5)

    def speak(self, instruction):
        if instruction and instruction != "No path computed":
            if not self._initialized.is_set():
                return
            self.queue.put(instruction)

    def stop(self):
        print("Stopping TTS Manager...")
        self.stop_event.set()
        self.queue.put(None)
        time.sleep(0.3)
        try:
            if self.tts_engine and self._initialized.is_set():
                self.tts_engine.stop()
        except Exception as e:
            print(f"Error stopping TTS engine: {e}")

# --- Configuration ---
try:
    K = np.load('camera_matrix.npy')
    dist = np.load('distortion_coeffs.npy')
    print("Camera calibration files loaded successfully.")
except FileNotFoundError:
    print("Warning: camera_matrix.npy or distortion_coeffs.npy not found.")
    print("Using placeholder camera calibration. Results will be inaccurate.")
    fw, fh = 640, 480
    K = np.array([[400., 0., fw/2], [0., 400., fh/2], [0., 0., 1.]])
    dist = np.zeros(5)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
if device == 'cuda':
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")

# --- Model Loading ---
def load_object_detection_model(device):
    try:
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        model.to(device)
        model.eval()
        print("YOLOv5 model loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading YOLOv5 model: {type(e).__name__} - {e}")
        exit()

def load_depth_model(device):
    try:
        model_type = "DPT_Hybrid"
        midas = torch.hub.load("intel-isl/MiDaS", model_type, pretrained=True)
        midas.to(device)
        midas.eval()
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        transform = midas_transforms.dpt_transform if "DPT" in model_type else midas_transforms.small_transform
        print(f"MiDaS ({model_type}) depth model loaded successfully.")
        return midas, transform
    except Exception as e:
        print(f"Error loading MiDaS model: {type(e).__name__} - {e}")
        exit()

# --- Natural Language Instruction Generator ---
def generate_instructions(waypoints, step_size=15, angle_threshold=40):
    if not waypoints or len(waypoints) < 2:
        return ["No path computed"]
    instructions = []
    heading = np.array([0.0, -1.0])
    current_action = None
    count = 0
    for i in range(1, len(waypoints)):
        prev = np.array(waypoints[i-1], dtype=float)
        curr = np.array(waypoints[i], dtype=float)
        vec = curr - prev
        norm = np.linalg.norm(vec)
        if norm < 1e-3: continue
        vec_norm = vec / norm
        angle_vec = math.atan2(vec_norm[1], vec_norm[0])
        angle_heading = math.atan2(heading[1], heading[0])
        turn_angle = math.degrees(angle_vec - angle_heading)
        turn_angle = (turn_angle + 180) % 360 - 180
        if abs(turn_angle) < angle_threshold:
            action = "Move forward"
        elif turn_angle > 0:
            action = "Turn left slightly and move"
            heading = vec_norm
        else:
            action = "Turn right slightly and move"
            heading = vec_norm
        if action == current_action:
            count += 1
        else:
            if current_action is not None:
                dist_px = count * step_size
                instructions.append(f"{current_action} (approx {dist_px:.0f} px)")
            current_action = action
            count = 1
    if current_action is not None:
        dist_px = count * step_size
        instructions.append(f"{current_action} (approx {dist_px:.0f} px)")
    return instructions if instructions else ["Path computed, but no instructions generated"]

# --- Core Processing Functions ---
def run_depth_estimation(midas, transform, image_rgb, device, stream):
    try:
        image_rgb = np.ascontiguousarray(image_rgb)
        input_tensor = transform(image_rgb).to(device)
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)
        with torch.no_grad():
            if stream:
                with torch.cuda.stream(stream):
                    prediction = midas(input_tensor)
            else:
                prediction = midas(input_tensor)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1), size=image_rgb.shape[:2],
                mode="bicubic", align_corners=False
            ).squeeze()
        depth_map = prediction.cpu().numpy()
        depth_map_filtered = cv2.medianBlur(depth_map.astype(np.float32), 5)
        depth_map_denoised = cv2.bilateralFilter(depth_map_filtered, d=9, sigmaColor=0.5, sigmaSpace=5)
        dmin = depth_map_denoised.min()
        dmax = depth_map_denoised.max()
        if dmax - dmin > 1e-6:
            normalized_depth = (depth_map_denoised - dmin) / (dmax - dmin)
        else:
            normalized_depth = np.zeros_like(depth_map_denoised)
        return normalized_depth
    except Exception as e:
        print(f"Depth estimation error: {type(e).__name__} - {e}")
        h, w = image_rgb.shape[:2]
        return np.zeros((h, w), dtype=np.float32)

def run_object_detection(model, frame_bgr, device, stream):
    try:
        img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        if stream:
            with torch.cuda.stream(stream):
                results = model(img_rgb)
        else:
            results = model(img_rgb)
        df = results.pandas().xyxy[0]
        objects = []
        for _, row in df.iterrows():
            if row['confidence'] > 0.4:
                x1, y1, x2, y2 = map(int, [row['xmin'], row['ymin'], row['xmax'], row['ymax']])
                cx, cy = (x1 + x2)//2, (y1 + y2)//2
                objects.append({
                    'label': row['name'], 'confidence': row['confidence'],
                    'bbox': (x1, y1, x2, y2), 'center': (cx, cy),
                    'size': (x2 - x1, y2 - y1)
                })
        return objects, df
    except Exception as e:
        return [], None

def process_visual_odometry(prev_gray, curr_gray, prev_kp, prev_des, K, dist, prev_depth_map):
    orb = cv2.ORB_create(nfeatures=2000)
    kp, des = orb.detectAndCompute(curr_gray, None)
    R_rel = np.eye(3)
    t_rel = np.zeros((3, 1))
    pts1 = pts2 = None
    if prev_gray is None or prev_kp is None or prev_des is None:
        return kp, des, R_rel, t_rel, pts1, pts2
    if des is None or len(kp) < 8:
        return kp, des, R_rel, t_rel, pts1, pts2
    if prev_des.dtype != des.dtype or prev_des.shape[1] != des.shape[1]:
        print("VO Descriptor mismatch.")
        return kp, des, R_rel, t_rel, pts1, pts2
    try:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(prev_des, des)
        matches = sorted(matches, key=lambda m: m.distance)
        good_matches = matches[:max(int(len(matches) * 0.7), 30)]
        if len(good_matches) < 8:
            return kp, des, R_rel, t_rel, pts1, pts2
        pts1 = np.float32([prev_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        pts2 = np.float32([kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        use_pnp = False
        if prev_depth_map is not None and dist is not None:
            P_3d = []
            q_2d = []
            h_prev, w_prev = prev_depth_map.shape
            for i in range(len(good_matches)):
                pt1 = pts1[i].ravel()
                x1, y1 = map(int, pt1)
                if 0 <= x1 < w_prev and 0 <= y1 < h_prev:
                    d1 = prev_depth_map[y1, x1]
                    if d1 > 0.1:
                        pt1_hom = np.array([x1, y1, 1])
                        inv_K = np.linalg.inv(K)
                        P_i_cam = d1 * (inv_K @ pt1_hom)
                        P_3d.append(P_i_cam)
                        q_2d.append(pts2[i].ravel())
            if len(P_3d) >= 4:
                P_3d = np.array(P_3d, dtype=np.float32)
                q_2d = np.array(q_2d, dtype=np.float32).reshape(-1, 1, 2)
                success, rvec, tvec = cv2.solvePnP(P_3d, q_2d, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
                if success:
                    R_rel, _ = cv2.Rodrigues(rvec)
                    t_rel = tvec
                    use_pnp = True
        if not use_pnp:
            E, mask_e = cv2.findEssentialMat(pts2, pts1, K, method=cv2.RANSAC, prob=0.999, threshold=1.5)
            if E is not None and mask_e is not None:
                _, R_rec, t_rec, mask_p = cv2.recoverPose(E, pts2, pts1, K, mask=mask_e)
                if R_rec is not None and t_rec is not None and np.isfinite(R_rec).all() and np.isfinite(t_rec).all():
                    R_rel = R_rec
                    t_rel = t_rec
        inlier_mask = mask_p.ravel() == 1 if not use_pnp else np.ones(len(pts1), dtype=bool)
        pts1_in = pts1[inlier_mask]
        pts2_in = pts2[inlier_mask]
        num_inliers = np.sum(inlier_mask)
        if num_inliers < 8:
            return kp, des, np.eye(3), np.zeros((3, 1)), pts1, pts2
        pts1, pts2 = pts1_in, pts2_in
    except cv2.error as e:
        print(f"OpenCV error during VO processing: {e}")
        return kp, des, np.eye(3), np.zeros((3, 1)), None, None
    except Exception as e:
        print(f"Generic error during VO processing: {type(e).__name__} - {e}")
        return kp, des, np.eye(3), np.zeros((3, 1)), None, None
    return kp, des, R_rel, t_rel, pts1, pts2

def update_object_depth(objects, depth_map):
    h, w = depth_map.shape
    for obj in objects:
        cx, cy = obj['center']
        if 0 <= cx < w and 0 <= cy < h:
            obj['depth'] = float(depth_map[cy, cx])
        else:
            obj['depth'] = None
    return objects

def plan_trajectory(objects, current_pos_img, depth_map, frame_shape, obstacle_disparity_threshold=0.5):
    h, w = frame_shape[:2]
    start_node = tuple(map(int, current_pos_img))
    obstacle_map = np.zeros((h, w), dtype=np.uint8)
    obstacle_margin = 30
    obstacle_cost_factor = 300.0
    depth_cost_factor = 50.0
    heuristic_weight = 1.5
    num_depth_obstacles = 0
    for obj in objects:
        obj_depth = obj.get('depth')
        is_obstacle = False
        if obj_depth is not None and obj_depth > obstacle_disparity_threshold:
            is_obstacle = True
            num_depth_obstacles += 1
        if is_obstacle and obj.get('bbox'):
            x1, y1, x2, y2 = obj['bbox']
            x1m = max(0, x1 - obstacle_margin); y1m = max(0, y1 - obstacle_margin)
            x2m = min(w, x2 + obstacle_margin); y2m = min(h, y2 + obstacle_margin)
            obstacle_map[y1m:y2m, x1m:x2m] = 255
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    obstacle_map = cv2.morphologyEx(obstacle_map, cv2.MORPH_OPEN, kernel)
    dist_to_obstacle = cv2.distanceTransform(obstacle_map, cv2.DIST_L2, 5)
    max_dist = np.max(dist_to_obstacle)
    if max_dist < 1e-6: max_dist = 1e-6
    obs_cost = obstacle_cost_factor * (1.0 - np.clip(dist_to_obstacle / max_dist, 0, 1))
    depth_cost = depth_map * depth_cost_factor
    base_cost = 0.0
    final_cost_map = base_cost + obs_cost + depth_cost
    final_cost_map[obstacle_map == 255] = 1e7
    temp_cost_vis = final_cost_map.copy()
    temp_cost_vis[obstacle_map == 255] = 0
    non_obstacle_max = temp_cost_vis.max()
    if non_obstacle_max < 1e-6: non_obstacle_max = 1.0
    temp_cost_vis = np.clip(final_cost_map, 0, non_obstacle_max)
    cost_vis = cv2.normalize(temp_cost_vis, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    cost_vis = cv2.applyColorMap(cost_vis, cv2.COLORMAP_JET)
    cost_vis[obstacle_map == 255] = [0, 0, 200]
    step_size = 15; goal_node = (w // 2, h - step_size)
    waypoints = [start_node]; visited = {start_node}; current_node = start_node
    max_steps = 100
    if not (0 <= start_node[0] < w and 0 <= start_node[1] < h) or obstacle_map[start_node[1], start_node[0]] != 0:
        return [start_node], obstacle_map, cost_vis
    for _ in range(max_steps):
        dist_to_goal = np.linalg.norm(np.array(current_node) - np.array(goal_node))
        if dist_to_goal < step_size * 1.5:
            if current_node != goal_node: waypoints.append(goal_node)
            break
        best_combined_cost = float('inf'); next_node = None
        for dx in [-step_size, 0, step_size]:
            for dy in [-step_size, 0, step_size]:
                if dx == 0 and dy == 0: continue
                potential_x = current_node[0] + dx; potential_y = current_node[1] + dy
                potential_node = (potential_x, potential_y)
                if (0 <= potential_x < w and 0 <= potential_y < h and
                        potential_node not in visited and
                        obstacle_map[potential_y, potential_x] == 0):
                    g_cost = final_cost_map[potential_y, potential_x]
                    h_cost = heuristic_weight * np.linalg.norm(np.array(potential_node) - np.array(goal_node))
                    total_cost = g_cost + h_cost
                    if total_cost < best_combined_cost:
                        best_combined_cost = total_cost; next_node = potential_node
        if next_node:
            waypoints.append(next_node); visited.add(next_node); current_node = next_node
        else:
            break
    return waypoints, obstacle_map, cost_vis

# --- Visualization ---
def visualize_combined(frame, objects, matched1=None, matched2=None, instructions=None, fps=0):
    out_frame = frame.copy(); h, w = out_frame.shape[:2]
    for obj in objects:
        x1, y1, x2, y2 = obj['bbox']; color = (0, 255, 0)
        cv2.rectangle(out_frame, (x1, y1), (x2, y2), color, 2)
        label = f"{obj['label']} ({obj['confidence']:.2f})"
        if obj.get('depth') is not None: label += f" | d={obj['depth']:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        text_y = y1 - 4
        if text_y < th: text_y = y2 + th + 4
        cv2.rectangle(out_frame, (x1, text_y - th), (x1 + tw + 4, text_y + 4), color, -1)
        cv2.putText(out_frame, label, (x1 + 2, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    if matched1 is not None and matched2 is not None and len(matched1) > 0:
        num_to_draw = min(len(matched1), 50); indices = np.random.choice(len(matched1), num_to_draw, replace=False)
        for i in indices:
            pt1 = tuple(map(int, matched1[i].ravel())); pt2 = tuple(map(int, matched2[i].ravel()))
            if 0 <= pt1[0] < w and 0 <= pt1[1] < h and 0 <= pt2[0] < w and 0 <= pt2[1] < h:
                cv2.line(out_frame, pt1, pt2, (0, 165, 255), 1); cv2.circle(out_frame, pt2, 3, (0, 0, 255), -1)
    if instructions:
        y_offset = 30; cv2.putText(out_frame, "Nav Instructions:", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)
        for i, instr in enumerate(instructions[:3]):
            y_offset += 25; cv2.putText(out_frame, f"{i+1}: {instr}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(out_frame, f"FPS: {fps:.1f}", (w - 100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
    return out_frame

def visualize_trajectory(frame_shape, history, current_pos_vis, waypoints=None, cost_map_vis=None):
    h, w = frame_shape[:2]
    if cost_map_vis is not None: canvas = cost_map_vis.copy()
    else: canvas = np.zeros((h, w, 3), dtype=np.uint8)
    grid_color = (70, 70, 70)
    for x in range(0, w, 50): cv2.line(canvas, (x, 0), (x, h), grid_color, 1)
    for y in range(0, h, 50): cv2.line(canvas, (0, y), (w, y), grid_color, 1)
    if len(history) > 1:
        pts = np.array(list(history), dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(canvas, [pts], isClosed=False, color=(255, 255, 0), thickness=2)
    if waypoints and len(waypoints) > 1:
        wp_pts = np.array(waypoints, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(canvas, [wp_pts], isClosed=False, color=(0, 255, 0), thickness=2)
        for p in waypoints:
            cv2.circle(canvas, tuple(map(int, p)), 5, (0, 0, 255), -1); cv2.circle(canvas, tuple(map(int, p)), 6, (255, 255, 255), 1)
    if current_pos_vis:
        center = tuple(map(int, current_pos_vis))
        pt1 = center; pt2 = (center[0] - 6, center[1] + 10); pt3 = (center[0] + 6, center[1] + 10)
        cv2.drawContours(canvas, [np.array([pt1, pt2, pt3])], 0, (0, 255, 255), -1)
    cv2.putText(canvas, "Trajectory & Cost View", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return canvas

# --- Main Loop ---
def main():
    tts_manager = TTSManager()
    tts_manager.start()
    if not tts_manager._initialized.wait(timeout=5.0):
        print("Warning: TTS engine did not initialize promptly.")
    yolo_model = load_object_detection_model(device)
    midas_model, midas_transform = load_depth_model(device)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open camera.")
        if tts_manager.is_alive(): tts_manager.stop(); tts_manager.join()
        exit()
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera resolution: {fw}x{fh}")
    prev_gray = None; prev_kp = None; prev_des = None; prev_depth_map = None
    new_K = K.copy()
    map_origin_vis = (fw // 2, fh * 3 // 4)
    current_pos_vis = map_origin_vis
    trajectory_history_vis = deque(maxlen=200)
    trajectory_history_vis.append(current_pos_vis)
    R_cum = np.eye(3); t_cum = np.zeros((3, 1))
    vo_scale_factor = 40.0
    frame_count = 0; last_time = time.time(); fps = 0
    stream_yolo = torch.cuda.Stream() if device == 'cuda' else None
    stream_midas = torch.cuda.Stream() if device == 'cuda' else None
    try:
        while True:
            ret, frame_bgr = cap.read()
            if not ret: print("Error: Failed to capture frame."); break
            current_time = time.time(); time_delta = current_time - last_time
            if time_delta > 0: fps = 1.0 / time_delta
            last_time = current_time
            if frame_count == 0:
                h_frame, w_frame = frame_bgr.shape[:2]
                new_K, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w_frame, h_frame), alpha=1, newImgSize=(w_frame, h_frame))
                print("Optimal New Camera Matrix (new_K) applied.")
            frame_undistorted = cv2.undistort(frame_bgr, K, dist, None, new_K)
            h, w = frame_undistorted.shape[:2]
            img_rgb = cv2.cvtColor(frame_undistorted, cv2.COLOR_BGR2RGB)
            img_gray = cv2.cvtColor(frame_undistorted, cv2.COLOR_BGR2GRAY)
            vo_result = [None] * 6; depth_result = [None]; obj_result = [None] * 2
            def vo_task():
                vo_result[:] = process_visual_odometry(prev_gray, img_gray, prev_kp, prev_des, new_K, dist, prev_depth_map)
            def depth_task():
                depth_result[0] = run_depth_estimation(midas_model, midas_transform, img_rgb, device, stream_midas)
            def obj_task():
                obj_result[:] = run_object_detection(yolo_model, frame_undistorted, device, stream_yolo)
            thread_vo = threading.Thread(target=vo_task)
            thread_depth = threading.Thread(target=depth_task)
            thread_obj = threading.Thread(target=obj_task)
            thread_vo.start(); thread_depth.start(); thread_obj.start()
            thread_vo.join(); thread_depth.join(); thread_obj.join()
            if device == 'cuda':
                torch.cuda.synchronize()
            kp, des, R_rel, t_rel, matched_pts1, matched_pts2 = vo_result
            depth_map_normalized = depth_result[0] if depth_result[0] is not None else np.zeros_like(img_gray, dtype=np.float32)
            detected_objects = obj_result[0] if obj_result[0] is not None else []
            if np.linalg.norm(t_rel) > 1e-6:
                t_update = R_cum @ t_rel
                t_cum += t_update
                R_cum = R_cum @ R_rel
            vis_x_raw = map_origin_vis[0] + t_cum[0, 0] * vo_scale_factor
            vis_y_raw = map_origin_vis[1] - t_cum[2, 0] * vo_scale_factor
            vis_x = int(np.clip(vis_x_raw, 0, w - 1))
            vis_y = int(np.clip(vis_y_raw, 0, h - 1))
            current_pos_vis = (vis_x, vis_y)
            if not trajectory_history_vis or np.linalg.norm(np.array(current_pos_vis) - np.array(trajectory_history_vis[-1])) > 2.0:
                trajectory_history_vis.append(current_pos_vis)
            objects_with_depth = update_object_depth(detected_objects, depth_map_normalized)
            planned_waypoints, obstacle_map_vis, cost_map_vis = plan_trajectory(
                objects_with_depth,
                current_pos_vis,
                depth_map_normalized,
                frame_undistorted.shape
            )
            nav_instructions = generate_instructions(planned_waypoints, step_size=15, angle_threshold=40)
            if nav_instructions:
                tts_manager.speak(nav_instructions[0])
            combined_view = visualize_combined(frame_undistorted, objects_with_depth, matched_pts1, matched_pts2, nav_instructions, fps)
            trajectory_view = visualize_trajectory(frame_undistorted.shape, trajectory_history_vis, current_pos_vis, planned_waypoints, cost_map_vis)
            cv2.imshow('Navigation View', combined_view)
            cv2.imshow('Trajectory & Cost View', trajectory_view)
            prev_gray = img_gray.copy()
            prev_kp = kp
            prev_des = des
            prev_depth_map = depth_map_normalized.copy()
            frame_count += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Exit requested.")
                break
    finally:
        print("Releasing resources...")
        cap.release()
        cv2.destroyAllWindows()
        if tts_manager.is_alive():
            tts_manager.stop()
            tts_manager.join(timeout=2.0)
            if tts_manager.is_alive():
                print("Warning: TTS thread did not exit cleanly.")
        print("Exiting script.")

if __name__ == "__main__":
    main()