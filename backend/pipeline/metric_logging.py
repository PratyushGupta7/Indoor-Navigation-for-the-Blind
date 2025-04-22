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
import pandas as pd
import uuid
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import sys

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# --- TTS Setup ---
tts_queue = queue.Queue()
tts_stop_event = threading.Event()

# --- Metrics Logger ---
class MetricsLogger:
    def __init__(self, log_file='navigation_metrics.csv'):
        self.log_file = log_file
        self.metrics = []
        self.task_id = None
        self.start_time = None

    def start_task(self):
        self.task_id = str(uuid.uuid4())
        self.start_time = time.time()
        return self.task_id

    def log_frame(self, objects, waypoints, current_pos_vis, goal_node, obstacle_map, depth_map):
        try:
            num_objects = len(objects)
            avg_confidence = np.mean([obj['confidence'] for obj in objects]) if objects else 0.0
            path_length = sum(np.linalg.norm(np.array(waypoints[i]) - np.array(waypoints[i-1])) 
                             for i in range(1, len(waypoints))) if len(waypoints) > 1 else 0.0
            min_obstacle_distance = self.compute_min_obstacle_distance(waypoints, obstacle_map)
            min_depth = self.compute_min_depth(waypoints, depth_map)
            
            self.metrics.append({
                'task_id': self.task_id,
                'timestamp': datetime.now(),
                'num_objects': num_objects,
                'avg_confidence': avg_confidence,
                'path_length': path_length,
                'min_obstacle_distance': min_obstacle_distance,
                'min_depth': min_depth
            })
        except Exception as e:
            print(f"Error logging frame metrics: {e}")

    def compute_min_obstacle_distance(self, waypoints, obstacle_map):
        if not waypoints or obstacle_map is None:
            return float('inf')
        h, w = obstacle_map.shape
        min_dist = float('inf')
        for wp in waypoints:
            x, y = map(int, wp)
            if 0 <= x < w and 0 <= y < h:
                dist = cv2.distanceTransform(obstacle_map, cv2.DIST_L2, 5)[y, x]
                min_dist = min(min_dist, dist)
        return min_dist

    def compute_min_depth(self, waypoints, depth_map):
        if not waypoints or depth_map is None:
            return float('inf')
        h, w = depth_map.shape
        min_depth = float('inf')
        for wp in waypoints:
            x, y = map(int, wp)
            if 0 <= x < w and 0 <= y < h:
                depth = depth_map[y, x]
                min_depth = min(min_depth, depth)
        return min_depth

    def end_task(self, success, collision_detected):
        try:
            end_time = time.time()
            duration = end_time - self.start_time
            self.metrics.append({
                'task_id': self.task_id,
                'timestamp': datetime.now(),
                'success': success,
                'collision_detected': collision_detected,
                'duration': duration
            })
            self.save_metrics()
        except Exception as e:
            print(f"Error ending task and saving metrics: {e}")

    def save_metrics(self):
        if self.metrics:
            try:
                df = pd.DataFrame(self.metrics)
                df.to_csv(self.log_file, index=False)
                print(f"Metrics saved to {self.log_file}")
            except Exception as e:
                print(f"Error saving metrics to CSV: {e}")

    def save_all_metrics(self):
        if self.metrics:
            try:
                df = pd.DataFrame(self.metrics)
                df.to_csv(self.log_file, index=False)
                print(f"All logged metrics saved to {self.log_file}")
            except Exception as e:
                print(f"Error saving all metrics to CSV: {e}")

def tts_worker():
    """Dedicated thread for text-to-speech synthesis with error handling."""
    print("TTS Worker started.")
    engine = None
    try:
        engine = pyttsx3.init()
        if engine is None:
            raise RuntimeError("Failed to initialize pyttsx3 engine.")
        engine.setProperty('rate', 165)
        print("TTS Engine initialized successfully.")

        while not tts_stop_event.is_set():
            try:
                text_to_say = tts_queue.get(timeout=0.1)
                if text_to_say is None:
                    print("TTS Worker received stop signal.")
                    break
                print(f"TTS Saying: '{text_to_say}'")
                engine.say(text_to_say)
                engine.runAndWait()
                tts_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"TTS Error during synthesis/speaking: {e}")
                if not tts_queue.empty():
                    try: tts_queue.get_nowait()
                    except queue.Empty: pass
                tts_queue.task_done()
    except Exception as engine_init_error:
        print(f"FATAL: Failed to initialize TTS engine: {engine_init_error}")
    finally:
        print("TTS Worker shutting down.")
        if engine:
            try: pass
            except Exception as e: print(f"Error during TTS engine cleanup: {e}")
        print("TTS Worker finished.")

def speak(text):
    """Adds text to the TTS queue if the worker is running."""
    if tts_stop_event.is_set():
        return
    if tts_queue.qsize() < 5:
        try: tts_queue.put(text)
        except Exception as e: print(f"Error putting text into TTS queue: {e}")

# --- Device Selection ---
def select_device():
    """Selects the best available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        dev = torch.device('cuda')
        print("Using device: CUDA")
        stream_depth = torch.cuda.Stream()
        stream_objdet = torch.cuda.Stream()
        return dev, stream_depth, stream_objdet
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() and torch.backends.mps.is_built():
        dev = torch.device('mps')
        print("Using device: MPS")
        return dev, None, None
    else:
        dev = torch.device('cpu')
        print("Using device: CPU")
        return dev, None, None

device, stream_depth, stream_objdet = select_device()

# --- Camera Calibration ---
def load_calibration(filepath_matrix='camera_matrix.npy', filepath_coeffs='distortion_coeffs.npy'):
    """Loads camera calibration files or returns default placeholders."""
    try:
        k = np.load(filepath_matrix)
        dist_coeffs = np.load(filepath_coeffs)
        print("Camera calibration files loaded successfully.")
        return k.astype(np.float32), dist_coeffs.astype(np.float32)
    except FileNotFoundError:
        print("Warning: Camera calibration files not found.")
        fw_placeholder, fh_placeholder = 640, 480
        k = np.eye(3, dtype=np.float32)
        dist_coeffs = np.zeros(5, dtype=np.float32)
        k[0, 2] = fw_placeholder / 2.0
        k[1, 2] = fh_placeholder / 2.0
        k[0, 0] = fw_placeholder * 0.8
        k[1, 1] = fw_placeholder * 0.8
        print(f"Using placeholder camera calibration for {fw_placeholder}x{fh_placeholder}.")
        return k, dist_coeffs
    except Exception as e:
        print(f"Error loading calibration files: {e}")
        sys.exit(1)

K, dist = load_calibration()

# --- Model Loading ---
def load_object_detection_model(dev):
    """Loads YOLOv5 model onto the specified device with error handling."""
    try:
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        model.to(dev)
        model.eval()
        print(f"YOLOv5 model loaded successfully on {dev}.")
        return model
    except Exception as e:
        print(f"FATAL: Error loading YOLOv5 model: {e}")
        speak("Error loading object detection model.")
        sys.exit(1)

def load_depth_model(dev):
    """Loads MiDaS depth model onto the specified device with error handling."""
    try:
        model_type = "DPT_Hybrid"
        print(f"Loading MiDaS model type: {model_type}...")
        midas = torch.hub.load("intel-isl/MiDaS", model_type, pretrained=True)
        midas.to(dev)
        midas.eval()
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        transform = midas_transforms.dpt_transform if "DPT" in model_type else midas_transforms.small_transform
        print(f"MiDaS {model_type} model loaded successfully on {dev}.")
        return midas, transform
    except Exception as e:
        print(f"FATAL: Error loading MiDaS model: {e}")
        speak("Error loading depth estimation model.")
        sys.exit(1)

# --- Natural Language Instruction Generator ---
def generate_instructions(waypoints, step_size=15, angle_threshold=30):
    if not waypoints or len(waypoints) < 2:
        return ["No path computed"]
    instructions = []
    heading = np.array([0.0, -1.0])
    current_action = None
    segment_start_point = np.array(waypoints[0], dtype=float)
    total_distance_current_action = 0.0
    for i in range(1, len(waypoints)):
        prev = np.array(waypoints[i-1], dtype=float)
        curr = np.array(waypoints[i], dtype=float)
        vec = curr - prev
        norm = np.linalg.norm(vec)
        if norm < 1e-3: continue
        vec_norm = vec / norm
        segment_distance = norm
        angle = math.degrees(math.atan2(np.cross(heading, vec_norm), np.dot(heading, vec_norm)))
        action = ""
        new_heading = heading
        if abs(angle) < angle_threshold: action = "Move forward"
        elif angle > 0: action = f"Turn {'left slightly' if angle < angle_threshold * 2 else 'left'} and move forward"; new_heading = vec_norm
        else: action = f"Turn {'right slightly' if angle > -angle_threshold * 2 else 'right'} and move forward"; new_heading = vec_norm
        if action == current_action: total_distance_current_action += segment_distance
        else:
            if current_action is not None: instructions.append(f"{current_action} ~{total_distance_current_action:.0f} px")
            current_action = action
            total_distance_current_action = segment_distance
            segment_start_point = prev
        heading = new_heading
    if current_action is not None: instructions.append(f"{current_action} ~{total_distance_current_action:.0f} px")
    if not instructions: return ["Path computed, but no movement needed or path too short."]
    return instructions

# --- Core Processing Functions ---
def run_depth_estimation(midas, transform, image_rgb, device, stream):
    """Runs MiDaS depth estimation with denoising, potentially on CUDA stream."""
    try:
        input_tensor_cpu = transform(image_rgb)
        input_tensor = input_tensor_cpu.to(device)
        if input_tensor.dim() == 3: input_tensor = input_tensor.unsqueeze(0)
        with torch.no_grad():
            if device.type == 'cuda' and stream:
                with torch.cuda.stream(stream):
                    prediction = midas(input_tensor)
                    prediction = torch.nn.functional.interpolate(
                        prediction.unsqueeze(1), size=image_rgb.shape[:2],
                        mode="bicubic", align_corners=False,
                    ).squeeze()
                stream.synchronize()
            else:
                prediction = midas(input_tensor)
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1), size=image_rgb.shape[:2],
                    mode="bicubic", align_corners=False,
                ).squeeze()
        depth_map = prediction.cpu().numpy().astype(np.float32)
        depth_map_filtered = cv2.medianBlur(depth_map, 5)
        depth_map_filtered = cv2.bilateralFilter(depth_map_filtered, d=7, sigmaColor=80, sigmaSpace=80)
        dmin, dmax = depth_map_filtered.min(), depth_map_filtered.max()
        if dmax - dmin > 1e-6:
            normalized_inverse = (dmax - depth_map_filtered) / (dmax - dmin)
        else:
            normalized_inverse = np.ones_like(depth_map_filtered)
        return np.clip(normalized_inverse, 0.0, 1.0).astype(np.float32)
    except Exception as e:
        print(f"Error during depth estimation: {e}")
        h, w = image_rgb.shape[:2]
        return np.zeros((h, w), dtype=np.float32)

def run_object_detection(model, frame, device, stream):
    """Runs YOLOv5 object detection with error handling."""
    objects = []
    try:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        with torch.no_grad():
            if device.type == 'cuda' and stream:
                with torch.cuda.stream(stream):
                    results = model(rgb, size=frame.shape[1])
                stream.synchronize()
            else:
                results = model(rgb, size=frame.shape[1])
            df = results.pandas().xyxy[0]
            h, w = frame.shape[:2]
            for _, row in df.iterrows():
                x1, y1, x2, y2 = map(int, [row['xmin'], row['ymin'], row['xmax'], row['ymax']])
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w - 1, x2), min(h - 1, y2)
                if x1 >= x2 or y1 >= y2:
                    continue
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                objects.append({
                    'label': row['name'],
                    'confidence': row['confidence'],
                    'bbox': (x1, y1, x2, y2),
                    'center': (cx, cy),
                    'size': (x2 - x1, y2 - y1)
                })
        return objects, df
    except Exception as e:
        print(f"Error during object detection: {e}")
        return [], None

def process_visual_odometry_pnp(
    prev_gray, curr_gray, prev_kp, prev_des, prev_depth_map_closeness,
    K_intrinsic, dist_coeffs, depth_vo_scale=5.0, min_depth_closeness=0.05
):
    """Estimates camera pose using ORB feature matching and PnP with depth information."""
    kp_curr, des_curr = None, None
    R_vo, t_vo = None, None
    pts1_inliers, pts2_inliers = None, None
    try:
        orb = cv2.ORB_create(nfeatures=1500, scaleFactor=1.2, nlevels=8, edgeThreshold=31,
                             firstLevel=0, WTA_K=2, scoreType=cv2.ORB_HARRIS_SCORE, patchSize=31, fastThreshold=20)
        kp_curr, des_curr = orb.detectAndCompute(curr_gray, None)
        if kp_curr is None or len(kp_curr) == 0 or des_curr is None:
            print("VO-PnP: No features detected in current frame.")
            return kp_curr, des_curr, R_vo, t_vo, pts1_inliers, pts2_inliers
    except cv2.error as e:
        print(f"VO-PnP: OpenCV Error during feature detection: {e}")
        return kp_curr, des_curr, R_vo, t_vo, pts1_inliers, pts2_inliers
    except Exception as e:
        print(f"VO-PnP: General Error during feature detection: {e}")
        return kp_curr, des_curr, R_vo, t_vo, pts1_inliers, pts2_inliers
    if prev_gray is None or prev_kp is None or len(prev_kp) == 0 or prev_des is None or prev_depth_map_closeness is None:
        print("VO-PnP: Missing valid previous frame data.")
        return kp_curr, des_curr, R_vo, t_vo, pts1_inliers, pts2_inliers
    try:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(prev_des, des_curr)
        matches = sorted(matches, key=lambda m: m.distance)
    except cv2.error as e:
        print(f"VO-PnP: OpenCV Error during matching: {e}")
        return kp_curr, des_curr, R_vo, t_vo, pts1_inliers, pts2_inliers
    except Exception as e:
        print(f"VO-PnP: General Error during matching: {e}")
        return kp_curr, des_curr, R_vo, t_vo, pts1_inliers, pts2_inliers
    object_points_3d = []
    image_points_2d = []
    original_indices = []
    h_prev, w_prev = prev_depth_map_closeness.shape
    fx, fy = K_intrinsic[0, 0], K_intrinsic[1, 1]
    cx, cy = K_intrinsic[0, 2], K_intrinsic[1, 2]
    epsilon = 1e-6
    min_matches_for_pnp = 6
    if len(matches) < min_matches_for_pnp:
        print(f"VO-PnP: Not enough matches ({len(matches)} < {min_matches_for_pnp}).")
        return kp_curr, des_curr, R_vo, t_vo, pts1_inliers, pts2_inliers
    try:
        for i, m in enumerate(matches):
            pt1_idx = m.queryIdx
            pt1_2d = prev_kp[pt1_idx].pt
            u1, v1 = int(round(pt1_2d[0])), int(round(pt1_2d[1]))
            if 0 <= v1 < h_prev and 0 <= u1 < w_prev:
                closeness = prev_depth_map_closeness[v1, u1]
                if closeness >= min_depth_closeness:
                    Z = depth_vo_scale * (1.0 - closeness + epsilon)
                    X = (u1 - cx) * Z / fx
                    Y = (v1 - cy) * Z / fy
                    pt2_idx = m.trainIdx
                    pt2_2d = kp_curr[pt2_idx].pt
                    object_points_3d.append([X, Y, Z])
                    image_points_2d.append(list(pt2_2d))
                    original_indices.append(i)
    except IndexError as e:
        print(f"VO-PnP: Index error during point preparation: {e}")
        return kp_curr, des_curr, R_vo, t_vo, pts1_inliers, pts2_inliers
    except Exception as e:
        print(f"VO-PnP: Error during point preparation: {e}")
        return kp_curr, des_curr, R_vo, t_vo, pts1_inliers, pts2_inliers
    num_valid_points = len(object_points_3d)
    if num_valid_points < min_matches_for_pnp:
        print(f"VO-PnP: Not enough valid 3D points ({num_valid_points} < {min_matches_for_pnp}).")
        return kp_curr, des_curr, R_vo, t_vo, pts1_inliers, pts2_inliers
    object_points_3d = np.array(object_points_3d, dtype=np.float32)
    image_points_2d = np.array(image_points_2d, dtype=np.float32)
    try:
        success, rvec, tvec, inliers_indices = cv2.solvePnPRansac(
            object_points_3d,
            image_points_2d,
            K_intrinsic,
            distCoeffs=None,
            iterationsCount=100,
            reprojectionError=4.0,
            confidence=0.99,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        if success and inliers_indices is not None and len(inliers_indices) >= min_matches_for_pnp:
            print(f"VO-PnP success: Found {len(inliers_indices)} inliers out of {num_valid_points} points.")
            R_pnp, _ = cv2.Rodrigues(rvec)
            R_vo = R_pnp
            t_vo = -R_pnp.T @ tvec
            inliers_indices = inliers_indices.flatten()
            original_inlier_match_indices = [original_indices[i] for i in inliers_indices]
            pts1_inliers = np.array([prev_kp[matches[i].queryIdx].pt for i in original_inlier_match_indices], dtype=np.float32)
            pts2_inliers = np.array([kp_curr[matches[i].trainIdx].pt for i in original_inlier_match_indices], dtype=np.float32)
        else:
            print(f"VO-PnP failed: Success={success}, Inliers={len(inliers_indices) if inliers_indices is not None else 'None'}")
    except cv2.error as e:
        print(f"VO-PnP: OpenCV Error during PnP solving: {e}")
    except Exception as e:
        print(f"VO-PnP: General Error during PnP solving: {e}")
    return kp_curr, des_curr, R_vo, t_vo, pts1_inliers, pts2_inliers

def update_object_depth(objects, depth_map_closeness):
    """Assigns closeness value (0=far, 1=close) to detected objects."""
    try:
        h, w = depth_map_closeness.shape
        for o in objects:
            cx, cy = o['center']
            if 0 <= cy < h and 0 <= cx < w:
                o['closeness'] = float(depth_map_closeness[cy, cx])
            else:
                o['closeness'] = 0.0
        return objects
    except Exception as e:
        print(f"Error updating object depth: {e}")
        for o in objects:
            if 'closeness' not in o:
                o['closeness'] = 0.0
        return objects

def plan_trajectory(objects, current_pos_px, depth_map_closeness, frame_shape):
    """Plans a simple trajectory avoiding obstacles using a cost map."""
    try:
        h, w = frame_shape[:2]
        cost_map = np.zeros((h, w), dtype=np.float32)
        obstacle_map_viz = np.zeros((h, w), dtype=np.uint8)
        obstacle_penalty = 15000.0
        margin_px = 15
        closeness_threshold_obstacle = 0.5
        depth_cost_factor = 250.0
        proximity_cost_factor = 180.0
        for o in objects:
            if o.get('closeness', 0.0) > closeness_threshold_obstacle:
                x1, y1, x2, y2 = o['bbox']
                x1m, y1m = max(0, x1 - margin_px), max(0, y1 - margin_px)
                x2m, y2m = min(w, x2 + margin_px), min(h, y2 + margin_px)
                cost_map[y1m:y2m, x1m:x2m] += obstacle_penalty
                obstacle_map_viz[y1m:y2m, x1m:x2m] = 255
        cost_map += depth_map_closeness * depth_cost_factor
        dt = cv2.distanceTransform(255 - obstacle_map_viz, cv2.DIST_L2, 5)
        cv2.normalize(dt, dt, 0, 1.0, cv2.NORM_MINMAX)
        cost_map += (1.0 - dt) * proximity_cost_factor
        cost_map[obstacle_map_viz == 255] = obstacle_penalty * 1.5
        start_point = tuple(map(int, current_pos_px))
        goal_point = (w // 2, h // 4)
        waypoints = [start_point]
        visited = {start_point}
        current_wp = start_point
        max_steps = 150
        step_size = 15
        goal_reached_threshold = step_size * 1.5
        for _ in range(max_steps):
            if np.linalg.norm(np.array(current_wp) - np.array(goal_point)) < goal_reached_threshold:
                if current_wp != goal_point:
                    waypoints.append(goal_point)
                break
            best_cost = float('inf')
            next_wp = None
            for dx in [-step_size, 0, step_size]:
                for dy in [-step_size, 0, step_size]:
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = current_wp[0] + dx, current_wp[1] + dy
                    if 0 <= nx < w and 0 <= ny < h and (nx, ny) not in visited and obstacle_map_viz[ny, nx] != 255:
                        try:
                            map_cost = cost_map[ny, nx]
                        except IndexError:
                            continue
                        heuristic_cost = 1.5 * np.linalg.norm(np.array((nx, ny)) - np.array(goal_point))
                        total_cost = map_cost + heuristic_cost
                        if total_cost < best_cost:
                            best_cost = total_cost
                            next_wp = (nx, ny)
            if next_wp is None:
                break
            waypoints.append(next_wp)
            visited.add(next_wp)
            current_wp = next_wp
        cost_vis = cv2.normalize(cost_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        cost_vis_color = cv2.applyColorMap(cost_vis, cv2.COLORMAP_JET)
        cost_vis_color[obstacle_map_viz == 255] = (0, 0, 200)
        return waypoints, obstacle_map_viz, cost_vis_color
    except Exception as e:
        print(f"Error during trajectory planning: {e}")
        h, w = frame_shape[:2]
        return [current_pos_px], np.zeros((h, w), dtype=np.uint8), np.zeros((h, w, 3), dtype=np.uint8)

# --- Visualization ---
def visualize_combined(frame, objects, matched_pts1=None, matched_pts2=None, instructions=None, fps=0):
    """Draws detections, VO matches, instructions, and FPS on the frame."""
    try:
        out_frame = frame.copy()
        h, w = out_frame.shape[:2]
        for o in objects:
            x1, y1, x2, y2 = o['bbox']
            color = (0, 255, 0)
            thickness = 2
            closeness = o.get('closeness')
            label = f"{o['label']} {o['confidence']:.2f}"
            if closeness is not None:
                label += f" | Close={closeness:.2f}"
                if closeness > 0.8:
                    color, thickness = (0, 0, 255), 3
                elif closeness > 0.5:
                    color = (0, 165, 255)
            cv2.rectangle(out_frame, (x1, y1), (x2, y2), color, thickness)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(out_frame, (x1, y1 - th - 4), (x1 + tw + 4, y1), color, -1)
            cv2.putText(out_frame, label, (x1 + 2, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        if matched_pts1 is not None and matched_pts2 is not None and len(matched_pts1) > 0:
            num_to_draw = min(len(matched_pts1), 75)
            indices = np.linspace(0, len(matched_pts1) - 1, num_to_draw, dtype=int)
            for i in indices:
                try:
                    pt1 = tuple(map(int, matched_pts1[i].ravel()))
                    pt2 = tuple(map(int, matched_pts2[i].ravel()))
                    if 0 <= pt1[0] < w and 0 <= pt1[1] < h and 0 <= pt2[0] < w and 0 <= pt2[1] < h:
                        cv2.line(out_frame, pt1, pt2, (0, 165, 255), 1)
                        cv2.circle(out_frame, pt2, 3, (255, 0, 0), -1)
                except IndexError:
                    continue
        if instructions:
            y_offset = 30
            cv2.putText(out_frame, "Nav Instructions:", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
            for i, instruction in enumerate(instructions[:4]):
                y_offset += 25
                cv2.putText(out_frame, instruction, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(out_frame, f"FPS: {fps:.2f}", (w - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        return out_frame
    except Exception as e:
        print(f"Error during combined visualization: {e}")
        return frame

def visualize_trajectory(frame_shape, history_deque, current_pos_px, waypoints=None, obstacle_map_vis=None, cost_map_vis_color=None):
    """Creates a top-down trajectory view."""
    try:
        h, w = frame_shape[:2]
        canvas = cost_map_vis_color if cost_map_vis_color is not None else np.zeros((h, w, 3), dtype=np.uint8)
        history_pts = list(history_deque)
        if len(history_pts) >= 2:
            try:
                pts_np = np.array(history_pts, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(canvas, [pts_np], isClosed=False, color=(255, 255, 0), thickness=2)
            except Exception as e:
                print(f"Error drawing history polyline: {e}")
        if waypoints and len(waypoints) >= 2:
            try:
                waypoints_np = np.array(waypoints, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(canvas, [waypoints_np], isClosed=False, color=(0, 255, 0), thickness=2)
                for p in waypoints:
                    cv2.circle(canvas, tuple(map(int, p)), 5, (0, 0, 255), -1)
                    cv2.circle(canvas, tuple(map(int, p)), 6, (255, 255, 255), 1)
            except Exception as e:
                print(f"Error drawing waypoint polyline/circles: {e}")
        if current_pos_px:
            try:
                pos_int = tuple(map(int, current_pos_px))
                cv2.circle(canvas, pos_int, 7, (0, 255, 255), -1)
                cv2.circle(canvas, pos_int, 8, (0, 0, 0), 1)
            except Exception as e:
                print(f"Error drawing current position circle: {e}")
        cv2.putText(canvas, "Trajectory View (Top-Down)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        return canvas
    except Exception as e:
        print(f"Error during trajectory visualization: {e}")
        h, w = frame_shape[:2]
        return np.zeros((h, w, 3), dtype=np.uint8)

# --- Main Execution ---
def main():
    # --- Initialization ---
    print("System initializing...")
    speak("System initializing.")
    tts_thread = threading.Thread(target=tts_worker, daemon=True)
    tts_thread.start()

    yolo_model = load_object_detection_model(device)
    midas_model, midas_transform = load_depth_model(device)

    cap = None
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise IOError("Cannot open camera.")
        fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Camera opened successfully: {fw}x{fh}")
        if K[0, 0] > 500 and K[0, 2] != fw / 2.0:
            print("Updating placeholder K principal point.")
            K[0, 2] = fw / 2.0
            K[1, 2] = fh / 2.0
    except Exception as e:
        print(f"FATAL: Error initializing camera: {e}")
        speak("Error initializing camera.")
        tts_stop_event.set()
        if tts_thread.is_alive():
            try:
                tts_queue.put(None)
                tts_thread.join(timeout=2)
            except Exception as te:
                print(f"Error stopping TTS thread: {te}")
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        sys.exit(1)

    metrics_logger = MetricsLogger(log_file='navigation_metrics.csv')
    K_intrinsic_calib = K.astype(np.float32)
    new_K, roi = cv2.getOptimalNewCameraMatrix(K_intrinsic_calib, dist, (fw, fh), alpha=0.9, newImgSize=(fw, fh))
    new_K = new_K.astype(np.float32)
    mapx, mapy = cv2.initUndistortRectifyMap(K_intrinsic_calib, dist, None, new_K, (fw, fh), cv2.CV_32FC1)
    print("Undistortion maps created.")
    print("Using Intrinsics for PnP:\n", new_K)

    prev_gray = None
    prev_kp = None
    prev_des = None
    prev_depth_map = None
    traj_hist = deque(maxlen=200)
    origin_px = (fw // 2, fh - fh // 10)
    current_pos_px = origin_px
    traj_hist.append(current_pos_px)
    cumulative_R = np.eye(3, dtype=np.float32)
    cumulative_t = np.zeros((3, 1), dtype=np.float32)
    vis_odom_scale = 40.0
    depth_pnp_scale = 5.0

    num_workers = 3
    executor = ThreadPoolExecutor(max_workers=num_workers, thread_name_prefix='Worker')
    print(f"Thread pool started with {num_workers} workers.")

    frame_count = 0
    last_instruction_time = time.time()
    instruction_speak_delay = 3.5
    fps = 0.0
    print("Starting main loop... Press 'q' to quit.")
    speak("Navigation system ready.")

    metrics_logger.start_task()

    try:
        while True:
            loop_start_time = time.time()

            try:
                ret, frame = cap.read()
                if not ret:
                    print("End of stream. Exiting.")
                    speak("Camera feed stopped.")
                    break
            except Exception as e:
                print(f"Error reading frame: {e}")
                speak("Camera error.")
                break

            try:
                undistorted_frame = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)
                rgb_frame_copy = cv2.cvtColor(undistorted_frame, cv2.COLOR_BGR2RGB)
                gray_frame = cv2.cvtColor(undistorted_frame, cv2.COLOR_BGR2GRAY)
            except cv2.error as e:
                print(f"OpenCV error during preprocessing: {e}")
                continue

            try:
                future_depth = executor.submit(run_depth_estimation, midas_model, midas_transform, rgb_frame_copy, device, stream_depth)
                future_objects = executor.submit(run_object_detection, yolo_model, undistorted_frame, device, stream_objdet)
                future_vo = executor.submit(process_visual_odometry_pnp,
                                           prev_gray, gray_frame, prev_kp, prev_des,
                                           prev_depth_map, new_K, None, depth_vo_scale=depth_pnp_scale)
            except Exception as e:
                print(f"Error submitting tasks: {e}")
                continue

            try:
                depth_map_closeness = future_depth.result()
                objects, _ = future_objects.result()
                kp, des, R_vo, t_vo, vo_pts1, vo_pts2 = future_vo.result()
            except Exception as e:
                print(f"Error retrieving results: {e}")
                depth_map_closeness = np.zeros_like(gray_frame, dtype=np.float32) if prev_depth_map is None else prev_depth_map
                objects = []
                kp, des = prev_kp, prev_des
                R_vo, t_vo, vo_pts1, vo_pts2 = None, None, None, None

            try:
                objects = update_object_depth(objects, depth_map_closeness)
                if R_vo is not None and t_vo is not None:
                    R_vo = R_vo.astype(np.float32)
                    t_vo = t_vo.astype(np.float32)
                    cumulative_t = cumulative_t + cumulative_R @ t_vo
                    cumulative_R = R_vo @ cumulative_R
                    vis_x = int(origin_px[0] + cumulative_t[0, 0] * (vis_odom_scale / depth_pnp_scale))
                    vis_y = int(origin_px[1] - cumulative_t[2, 0] * (vis_odom_scale / depth_pnp_scale))
                    vis_x = np.clip(vis_x, 0, fw - 1)
                    vis_y = np.clip(vis_y, 0, fh - 1)
                    if np.linalg.norm(np.array(current_pos_px) - np.array((vis_x, vis_y))) > 1:
                        current_pos_px = (vis_x, vis_y)
                        traj_hist.append(current_pos_px)
                        print(f"Current position updated: {current_pos_px}")
                waypoints, obstacle_map_vis, cost_map_vis_color = plan_trajectory(objects, current_pos_px, depth_map_closeness, undistorted_frame.shape)
                instructions = generate_instructions(waypoints)
                w, h = undistorted_frame.shape[1], undistorted_frame.shape[0]
                goal_node = (w // 2, h - 15)
                metrics_logger.log_frame(objects, waypoints, current_pos_px, goal_node, obstacle_map_vis, depth_map_closeness)
                distance_to_goal = np.linalg.norm(np.array(current_pos_px) - np.array(goal_node))
                goal_reached = distance_to_goal < 15
                collision_detected = False
                if 0 <= current_pos_px[0] < w and 0 <= current_pos_px[1] < h and obstacle_map_vis[current_pos_px[1], current_pos_px[0]] == 255:
                    collision_detected = True
                if goal_reached or collision_detected:
                    print(f"Task completed: goal_reached={goal_reached}, collision_detected={collision_detected}, distance_to_goal={distance_to_goal:.2f}")
                    success = goal_reached and not collision_detected
                    metrics_logger.end_task(success=success, collision_detected=collision_detected)
                    metrics_logger.start_task()
            except Exception as e:
                print(f"Error during post-processing: {e}")
                instructions = ["Processing Error"]
                waypoints, obstacle_map_vis, cost_map_vis_color = [current_pos_px], np.zeros_like(gray_frame, dtype=np.uint8), np.zeros((fh, fw, 3), dtype=np.uint8)

            try:
                very_close_obstacle = False
                obstacle_warning_threshold = 0.85
                close_obstacle_label = ""
                for obj in objects:
                    if obj.get('closeness', 0.0) > obstacle_warning_threshold:
                        very_close_obstacle = True
                        close_obstacle_label = obj.get('label', 'Obstacle')
                        speak(f"Warning! {close_obstacle_label} very close ahead!")
                        instructions = [f"STOP! {close_obstacle_label} detected!"]
                        last_instruction_time = time.time()
                        break
                current_time = time.time()
                if not very_close_obstacle and instructions and instructions[0] != "No path computed" and (current_time - last_instruction_time > instruction_speak_delay):
                    if instructions[0] not in ["Processing Error", "STOP! Obstacle detected!"]:
                        speak(instructions[0])
                        last_instruction_time = current_time
            except Exception as e:
                print(f"Error during alert/TTS processing: {e}")

            loop_end_time = time.time()
            elapsed_time = loop_end_time - loop_start_time
            fps = 0.9 * fps + 0.1 * (1.0 / elapsed_time if elapsed_time > 0 else 0)
            try:
                combined_view = visualize_combined(undistorted_frame, objects, vo_pts1, vo_pts2, instructions, fps)
                trajectory_view = visualize_trajectory(undistorted_frame.shape, traj_hist, current_pos_px, waypoints, obstacle_map_vis, cost_map_vis_color)
                cv2.imshow('Object Detection + PnP VO View', combined_view)
                cv2.imshow('Trajectory Planning View', trajectory_view)
            except cv2.error as e:
                print(f"OpenCV error during visualization: {e}")
            except Exception as e:
                print(f"General error during visualization: {e}")

            prev_gray = gray_frame
            prev_kp = kp
            prev_des = des
            prev_depth_map = depth_map_closeness.copy()

            frame_count += 1

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("'q' pressed, initiating shutdown.")
                speak("Exiting program.")
                break
            elif key == ord('s'):
                speak(f"Status OK. Position {current_pos_px}. FPS {fps:.1f}.")

    except KeyboardInterrupt:
        print("\nCtrl+C detected. Initiating graceful shutdown.")
        speak("Shutdown initiated.")
    finally:
        print("Cleaning up resources...")
        if 'executor' in locals() and executor is not None:
            print("Shutting down thread pool...")
            try:
                executor.shutdown(wait=False, cancel_futures=True)
                print("Thread pool shutdown signal sent.")
            except Exception as e:
                print(f"Error shutting down thread pool: {e}")
        print("Stopping TTS worker...")
        tts_stop_event.set()
        if 'tts_thread' in locals() and tts_thread.is_alive():
            try:
                tts_queue.put(None)
                tts_thread.join(timeout=3.0)
                if tts_thread.is_alive():
                    print("Warning: TTS thread did not terminate gracefully.")
                else:
                    print("TTS worker stopped.")
            except Exception as e:
                print(f"Error stopping TTS thread: {e}")
        if 'cap' in locals() and cap is not None and cap.isOpened():
            print("Releasing camera...")
            try:
                cap.release()
                print("Camera released.")
            except Exception as e:
                print(f"Error releasing camera: {e}")
        print("Destroying OpenCV windows...")
        try:
            cv2.destroyAllWindows()
            print("OpenCV windows destroyed.")
        except Exception as e:
            print(f"Error destroying OpenCV windows: {e}")
        print("Saving any logged metrics...")
        metrics_logger.save_all_metrics()
        print("Cleanup finished. Exiting.")
        sys.exit(0)

if __name__ == "__main__":
    main()