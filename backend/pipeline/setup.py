import cv2
import torch
import numpy as np
import warnings
import math  
import pyttsx3
import threading
import queue
import sys
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

BASE_PATH = Path(__file__).resolve().parent
CALIBRATION_PATH = BASE_PATH / 'calibration'
MODELS_DIR = BASE_PATH.parent / 'models'

#  TTS Setup
tts_queue = queue.Queue()
tts_stop_event = threading.Event()

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
        # print(f"TTS worker stopped, discarding: {text}")
        return
    if tts_queue.qsize() < 5:
        try: tts_queue.put(text)
        except Exception as e: print(f"Error putting text into TTS queue: {e}")
    # else: print(f"TTS Queue full, discarding: {text}")

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
def load_calibration(
    filepath_matrix=CALIBRATION_PATH / 'camera_matrix.npy',
    filepath_coeffs=CALIBRATION_PATH / 'distortion_coeffs.npy'
):
    """Loads camera calibration files or returns default placeholders."""
    try:
        k = np.load(filepath_matrix)
        dist_coeffs = np.load(filepath_coeffs)
        print("Camera calibration files loaded successfully.")
        # Ensure float32 type, important for OpenCV functions like PnP
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
def load_object_detection_model(device):
    """
    Load YOLOv5 from a local clone + local weights (yolov5s.pt).
    Falls back to the online hub if anything goes wrong.
    """
    repo_dir   = MODELS_DIR / "ultralytics_yolov5_master"
    weight_path = repo_dir / "yolov5s.pt"
    try:
        # 'custom' tells hubconf to load any .pt you give it via `path`
        model = torch.hub.load(
            str(repo_dir),
            'custom',
            path=str(weight_path),
            source='local'
        )
        model.to(device).eval()
        print(f"YOLOv5 loaded from local path {weight_path}")
        return model

    except Exception as e:
        print(f"Local YOLO load failed ({e}), falling back to online…")
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        model.to(device).eval()
        print("YOLOv5 loaded from ultralytics hub")
        return model

def load_depth_model(device):
    """
    Load MiDaS DPT_Hybrid from a local clone + local weights (dpt_hybrid_384.pt).
    Falls back to the online hub if anything goes wrong.
    """
    repo_dir    = MODELS_DIR / "intel-isl_MiDaS_master"
    weight_path = repo_dir / "dpt_hybrid_384.pt"
    model_type  = "DPT_Hybrid"

    try:
        # load model architecture without weights
        midas = torch.hub.load(
            str(repo_dir),
            model_type,
            pretrained=False,   # don’t auto-download
            source='local'
        )
        # now manually load your .pt
        state_dict = torch.load(str(weight_path), map_location=device)
        midas.load_state_dict(state_dict)
        midas.to(device).eval()

        # load transforms locally as before
        midas_transforms = torch.hub.load(
            str(repo_dir),
            "transforms",
            source='local'
        )
        transform = (
            midas_transforms.dpt_transform
            if "DPT" in model_type
            else midas_transforms.small_transform
        )

        print(f"MiDaS {model_type} loaded from local path {weight_path}")
        return midas, transform

    except Exception as e:
        print(f"Local MiDaS load failed ({e}), falling back to online…")
        # fallback to remote
        midas = torch.hub.load("intel-isl/MiDaS", model_type, pretrained=True)
        midas.to(device).eval()
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        transform = (
            midas_transforms.dpt_transform
            if "DPT" in model_type
            else midas_transforms.small_transform
        )
        print("MiDaS loaded from intel-isl hub")
        return midas, transform

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
        action = ""; new_heading = heading
        if abs(angle) < angle_threshold: action = "Move forward"
        elif angle > 0: action = f"Turn {'left slightly' if angle < angle_threshold * 2 else 'left'} and move forward"; new_heading = vec_norm
        else: action = f"Turn {'right slightly' if angle > -angle_threshold * 2 else 'right'} and move forward"; new_heading = vec_norm
        if action == current_action: total_distance_current_action += segment_distance
        else:
            if current_action is not None: instructions.append(f"{current_action} ~{total_distance_current_action:.0f} px")
            current_action = action; total_distance_current_action = segment_distance; segment_start_point = prev
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
            else: # MPS or CPU
                 prediction = midas(input_tensor)
                 prediction = torch.nn.functional.interpolate(
                     prediction.unsqueeze(1), size=image_rgb.shape[:2],
                     mode="bicubic", align_corners=False,
                 ).squeeze()

        depth_map = prediction.cpu().numpy().astype(np.float32) # Ensure float32

        # Denoising
        depth_map_filtered = cv2.medianBlur(depth_map, 5)
        depth_map_filtered = cv2.bilateralFilter(depth_map_filtered, d=7, sigmaColor=80, sigmaSpace=80)

        # Normalization (Inverse Depth: 1.0 = Close, 0.0 = Far)
        dmin, dmax = depth_map_filtered.min(), depth_map_filtered.max()
        if dmax - dmin > 1e-6:
            normalized_inverse = (dmax - depth_map_filtered) / (dmax - dmin)
        else:
            normalized_inverse = np.ones_like(depth_map_filtered)

        return np.clip(normalized_inverse, 0.0, 1.0).astype(np.float32) # Ensure float32 output

    except Exception as e:
        print(f"Error during depth estimation: {e}")
        h, w = image_rgb.shape[:2]
        return np.zeros((h, w), dtype=np.float32) # Return zero map (max distance)


def run_object_detection(model, frame, device, stream):
    """Runs YOLOv5 object detection with error handling."""
    objects = []
    try:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        with torch.no_grad():
            if device.type == 'cuda' and stream:
                 with torch.cuda.stream(stream): results = model(rgb, size=frame.shape[1])
                 stream.synchronize()
            else: results = model(rgb, size=frame.shape[1])

            df = results.pandas().xyxy[0]
            h, w = frame.shape[:2]
            for _, row in df.iterrows():
                x1, y1, x2, y2 = map(int, [row['xmin'], row['ymin'], row['xmax'], row['ymax']])
                x1, y1 = max(0, x1), max(0, y1); x2, y2 = min(w - 1, x2), min(h - 1, y2)
                if x1 >= x2 or y1 >= y2: continue
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                objects.append({'label': row['name'], 'confidence': row['confidence'],
                                'bbox': (x1, y1, x2, y2), 'center': (cx, cy),
                                'size': (x2 - x1, y2 - y1)})
        return objects, df # df might be useful elsewhere
    except Exception as e:
        print(f"Error during object detection: {e}")
        return [], None

# =============================================================================
# === Depth-Aided Visual Odometry using PnP ===
# =============================================================================
def process_visual_odometry_pnp(
    prev_gray, curr_gray,
    prev_kp, prev_des,
    prev_depth_map_closeness, # Needs depth map from the PREVIOUS frame
    K_intrinsic, # Camera intrinsic matrix (new_K)
    dist_coeffs, # Distortion coefficients (use None if images are undistorted)
    depth_vo_scale=5.0, # **Crucial Tuning Parameter**: Maps normalized depth to a metric scale for PnP
    min_depth_closeness=0.05 # Ignore features corresponding to very far points (low closeness)
    ):
    """
    Estimates camera pose using ORB feature matching and PnP with depth information.

    Args:
        prev_gray: Grayscale image of the previous frame.
        curr_gray: Grayscale image of the current frame.
        prev_kp: Keypoints detected in the previous frame.
        prev_des: Descriptors for keypoints in the previous frame.
        prev_depth_map_closeness: Normalized inverse depth map (1=close, 0=far) of the previous frame.
        K_intrinsic: Camera intrinsic matrix (3x3 NumPy float32 array).
        dist_coeffs: Camera distortion coefficients (NumPy array), or None if undistorted.
        depth_vo_scale: Heuristic scale factor to convert normalized depth into a metric Z value.
        min_depth_closeness: Minimum closeness value (0-1) to consider a point for 3D projection.

    Returns:
        Tuple: (kp, des, R_vo, t_vo, pts1_inliers, pts2_inliers)
            kp: Keypoints detected in the current frame.
            des: Descriptors for keypoints in the current frame.
            R_vo: Estimated rotation matrix (3x3) transforming points from previous to current frame coords. None on failure.
            t_vo: Estimated translation vector (3x1) of current cam origin relative to previous cam origin,
                  expressed in previous cam coords. None on failure.
            pts1_inliers: 2D keypoints from previous frame identified as inliers by PnP-RANSAC.
            pts2_inliers: Corresponding 2D keypoints from current frame identified as inliers.
    """
    kp_curr, des_curr = None, None
    R_vo, t_vo = None, None
    pts1_inliers, pts2_inliers = None, None

    # --- Feature Detection (Current Frame) ---
    try:
        orb = cv2.ORB_create(nfeatures=1500, scaleFactor=1.2, nlevels=8, edgeThreshold=31, firstLevel=0, WTA_K=2, scoreType=cv2.ORB_HARRIS_SCORE, patchSize=31, fastThreshold=20)
        # orb = cv2.ORB_create(nfeatures=1000) # Simpler ORB
        kp_curr, des_curr = orb.detectAndCompute(curr_gray, None)
        if kp_curr is None or len(kp_curr) == 0 or des_curr is None:
            # print("VO-PnP: No features detected in current frame.")
            return kp_curr, des_curr, R_vo, t_vo, pts1_inliers, pts2_inliers
    except cv2.error as e:
        print(f"VO-PnP: OpenCV Error during feature detection: {e}")
        return kp_curr, des_curr, R_vo, t_vo, pts1_inliers, pts2_inliers
    except Exception as e:
        print(f"VO-PnP: General Error during feature detection: {e}")
        return kp_curr, des_curr, R_vo, t_vo, pts1_inliers, pts2_inliers


    # --- Check Previous Frame Data ---
    if prev_gray is None or prev_kp is None or len(prev_kp) == 0 or prev_des is None or prev_depth_map_closeness is None:
        # print("VO-PnP: Missing valid previous frame data (image, kp, des, or depth).")
        return kp_curr, des_curr, R_vo, t_vo, pts1_inliers, pts2_inliers

    # --- Feature Matching ---
    try:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(prev_des, des_curr)
        matches = sorted(matches, key=lambda m: m.distance) # Sort by quality
        # Filter matches based on distance (optional but can help)
        # good_distance_threshold = 50 # Example threshold, adjust based on descriptor quality
        # matches = [m for m in matches if m.distance < good_distance_threshold]

    except cv2.error as e:
        print(f"VO-PnP: OpenCV Error during matching: {e}")
        return kp_curr, des_curr, R_vo, t_vo, pts1_inliers, pts2_inliers
    except Exception as e:
        print(f"VO-PnP: General Error during matching: {e}")
        return kp_curr, des_curr, R_vo, t_vo, pts1_inliers, pts2_inliers


    # --- Prepare Points for PnP ---
    # We need 3D points from the PREVIOUS frame and corresponding 2D points from the CURRENT frame.
    object_points_3d = [] # 3D points in previous camera coordinate system
    image_points_2d = []  # Corresponding 2D points in current image
    original_indices = [] # Keep track of original match indices for inlier filtering later

    h_prev, w_prev = prev_depth_map_closeness.shape
    fx, fy = K_intrinsic[0, 0], K_intrinsic[1, 1]
    cx, cy = K_intrinsic[0, 2], K_intrinsic[1, 2]
    epsilon = 1e-6 # To avoid division by zero or Z=0

    min_matches_for_pnp = 6 # Minimum required points for PnP RANSAC

    if len(matches) < min_matches_for_pnp:
         # print(f"VO-PnP: Not enough matches ({len(matches)} < {min_matches_for_pnp}).")
         return kp_curr, des_curr, R_vo, t_vo, pts1_inliers, pts2_inliers

    try:
        for i, m in enumerate(matches):
            # Get 2D point from previous frame (queryIdx)
            pt1_idx = m.queryIdx
            pt1_2d = prev_kp[pt1_idx].pt # (u, v) coordinates
            u1, v1 = int(round(pt1_2d[0])), int(round(pt1_2d[1]))

            # Check if point is within depth map bounds
            if 0 <= v1 < h_prev and 0 <= u1 < w_prev:
                # Get closeness value from previous depth map
                closeness = prev_depth_map_closeness[v1, u1]

                # Filter points that are too far (low closeness) or have invalid depth
                if closeness >= min_depth_closeness:
                    # Convert closeness (0-1) to a scaled Z depth (metric estimate)
                    # Z = scale * (1 - closeness) --> Larger closeness means smaller Z
                    Z = depth_vo_scale * (1.0 - closeness + epsilon)

                    # Unproject 2D point (u1, v1) to 3D (X, Y, Z) using intrinsics
                    X = (u1 - cx) * Z / fx
                    Y = (v1 - cy) * Z / fy

                    # Get corresponding 2D point from current frame (trainIdx)
                    pt2_idx = m.trainIdx
                    pt2_2d = kp_curr[pt2_idx].pt

                    # Add the pair
                    object_points_3d.append([X, Y, Z])
                    image_points_2d.append(list(pt2_2d))
                    original_indices.append(i) # Store index of the match

    except IndexError as e:
         print(f"VO-PnP: Index error during point preparation: {e}. Match index {i}, pt1_idx {pt1_idx}, pt2_idx {pt2_idx}")
         # This might happen if keypoint indices are somehow invalid.
         return kp_curr, des_curr, R_vo, t_vo, pts1_inliers, pts2_inliers
    except Exception as e:
         print(f"VO-PnP: Error during point preparation: {e}")
         return kp_curr, des_curr, R_vo, t_vo, pts1_inliers, pts2_inliers


    # Check if we have enough valid points for PnP
    num_valid_points = len(object_points_3d)
    if num_valid_points < min_matches_for_pnp:
        # print(f"VO-PnP: Not enough valid 3D points generated ({num_valid_points} < {min_matches_for_pnp}).")
        return kp_curr, des_curr, R_vo, t_vo, pts1_inliers, pts2_inliers

    # Convert lists to NumPy arrays (required by solvePnPRansac)
    object_points_3d = np.array(object_points_3d, dtype=np.float32)
    image_points_2d = np.array(image_points_2d, dtype=np.float32)

    # --- Solve PnP using RANSAC ---
    try:
        # Use None for distortion coeffs if images used for kp_curr are already undistorted
        # Common flags: cv2.SOLVEPNP_ITERATIVE, cv2.SOLVEPNP_EPNP, cv2.SOLVEPNP_AP3P
        # RANSAC parameters might need tuning:
        iterations_count = 100
        reprojection_error_threshold = 4.0 # Pixels - adjust based on image resolution and feature quality
        confidence = 0.99 # Probability that the solution is correct

        success, rvec, tvec, inliers_indices = cv2.solvePnPRansac(
            object_points_3d,
            image_points_2d,
            K_intrinsic,
            distCoeffs=None, # Assuming undistorted points fed to PnP
            iterationsCount=iterations_count,
            reprojectionError=reprojection_error_threshold,
            confidence=confidence,
            flags=cv2.SOLVEPNP_ITERATIVE # Or EPNP
        )

        if success and inliers_indices is not None and len(inliers_indices) >= min_matches_for_pnp:
             # PnP gives rotation (rvec) and translation (tvec) that transform points
             # from the WORLD (previous camera) frame to the CURRENT camera frame.

             # Convert rotation vector to rotation matrix
             R_pnp, _ = cv2.Rodrigues(rvec)

             # Calculate the relative motion (R_vo, t_vo)
             # R_vo is the rotation from prev to curr frame: R_vo = R_pnp
             R_vo = R_pnp

             # t_vo is the translation of the current camera origin relative to the previous one,
             # expressed in the previous camera's coordinate system.
             # tvec is the translation of the *previous* origin in the *current* frame coords.
             # t_vo = -R_pnp^T * tvec
             t_vo = -R_pnp.T @ tvec

             # --- Filter original matched keypoints based on PnP inliers ---
             if inliers_indices is not None:
                 inliers_indices = inliers_indices.flatten() # Ensure it's a flat array
                 # Get the original match indices corresponding to the PnP inliers
                 original_inlier_match_indices = [original_indices[i] for i in inliers_indices]

                 # Extract the corresponding 2D points from prev (pts1) and current (pts2) frames
                 pts1_inliers = np.array([prev_kp[matches[i].queryIdx].pt for i in original_inlier_match_indices], dtype=np.float32)
                 pts2_inliers = np.array([kp_curr[matches[i].trainIdx].pt for i in original_inlier_match_indices], dtype=np.float32)
                 # print(f"VO-PnP Success: Found {len(inliers_indices)} inliers out of {num_valid_points} potential points.")
             else:
                 # Should not happen if success is true and len(inliers_indices) is checked, but good practice
                 print("VO-PnP: Success reported but inliers_indices is None.")
                 R_vo, t_vo = None, None # Mark as failure
                 pts1_inliers, pts2_inliers = None, None

        # else: print(f"VO-PnP: solvePnPRansac failed. Success={success}, Inliers={len(inliers_indices) if inliers_indices is not None else 'None'}")


    except cv2.error as e:
        print(f"VO-PnP: OpenCV Error during PnP solving: {e}")
        # R_vo, t_vo remain None
    except Exception as e:
        print(f"VO-PnP: General Error during PnP solving: {e}")
        # R_vo, t_vo remain None

    # Return current keypoints/descriptors, estimated pose (R_vo, t_vo), and inlier points
    return kp_curr, des_curr, R_vo, t_vo, pts1_inliers, pts2_inliers

# =============================================================================


def update_object_depth(objects, depth_map_closeness):
    """Assigns closeness value (0=far, 1=close) to detected objects."""
    try:
        h, w = depth_map_closeness.shape
        for o in objects:
            cx, cy = o['center']
            if 0 <= cy < h and 0 <= cx < w:
                o['closeness'] = float(depth_map_closeness[cy, cx])
            else: o['closeness'] = 0.0
        return objects
    except Exception as e:
        print(f"Error updating object depth: {e}")
        for o in objects:
            if 'closeness' not in o: o['closeness'] = 0.0
        return objects


def plan_trajectory(objects, current_pos_px, depth_map_closeness, frame_shape):
    """Plans a simple trajectory avoiding obstacles using a cost map."""
    try:
        h, w = frame_shape[:2]
        cost_map = np.zeros((h, w), dtype=np.float32)
        obstacle_map_viz = np.zeros((h, w), dtype=np.uint8)
        obstacle_penalty = 15000.0; margin_px = 15; closeness_threshold_obstacle = 0.5
        depth_cost_factor = 250.0; proximity_cost_factor = 180.0

        for o in objects: # Obstacle cost
            if o.get('closeness', 0.0) > closeness_threshold_obstacle:
                x1, y1, x2, y2 = o['bbox']
                x1m, y1m = max(0, x1 - margin_px), max(0, y1 - margin_px)
                x2m, y2m = min(w, x2 + margin_px), min(h, y2 + margin_px)
                cost_map[y1m:y2m, x1m:x2m] += obstacle_penalty
                obstacle_map_viz[y1m:y2m, x1m:x2m] = 255

        cost_map += depth_map_closeness * depth_cost_factor # Depth cost
        dt = cv2.distanceTransform(255 - obstacle_map_viz, cv2.DIST_L2, 5)
        cv2.normalize(dt, dt, 0, 1.0, cv2.NORM_MINMAX)
        cost_map += (1.0 - dt) * proximity_cost_factor # Proximity cost
        cost_map[obstacle_map_viz == 255] = obstacle_penalty * 1.5 # Ensure max cost

        # Greedy Path Finding
        start_point = tuple(map(int, current_pos_px)); goal_point = (w // 2, h // 4)
        waypoints = [start_point]; visited = {start_point}; current_wp = start_point
        max_steps = 150; step_size = 15; goal_reached_threshold = step_size * 1.5
        for _ in range(max_steps):
            if np.linalg.norm(np.array(current_wp) - np.array(goal_point)) < goal_reached_threshold:
                if current_wp != goal_point: waypoints.append(goal_point); break
            best_cost = float('inf'); next_wp = None
            for dx in [-step_size, 0, step_size]:
                for dy in [-step_size, 0, step_size]:
                    if dx == 0 and dy == 0: continue
                    nx, ny = current_wp[0] + dx, current_wp[1] + dy
                    if 0 <= nx < w and 0 <= ny < h and (nx, ny) not in visited and obstacle_map_viz[ny, nx] != 255:
                        try: map_cost = cost_map[ny, nx]
                        except IndexError: continue
                        heuristic_cost = 1.5 * np.linalg.norm(np.array((nx, ny)) - np.array(goal_point))
                        total_cost = map_cost + heuristic_cost
                        if total_cost < best_cost: best_cost = total_cost; next_wp = (nx, ny)
            if next_wp is None: break
            waypoints.append(next_wp); visited.add(next_wp); current_wp = next_wp

        cost_vis = cv2.normalize(cost_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        cost_vis_color = cv2.applyColorMap(cost_vis, cv2.COLORMAP_JET)
        cost_vis_color[obstacle_map_viz == 255] = (0, 0, 200) # Overlay obstacles
        return waypoints, obstacle_map_viz, cost_vis_color
    except Exception as e:
        print(f"Error during trajectory planning: {e}")
        h, w = frame_shape[:2]
        return [current_pos_px], np.zeros((h, w), dtype=np.uint8), np.zeros((h, w, 3), dtype=np.uint8)


# --- Visualization ---
def visualize_combined(frame, objects, matched_pts1=None, matched_pts2=None, instructions=None, fps=0):
    """Draws detections, VO matches, instructions, and FPS on the frame."""
    try:
        out_frame = frame.copy(); h, w = out_frame.shape[:2]
        for o in objects: # Detections
            x1, y1, x2, y2 = o['bbox']; color = (0, 255, 0); thickness = 2
            closeness = o.get('closeness'); label = f"{o['label']} {o['confidence']:.2f}"
            if closeness is not None:
                label += f" | Close={closeness:.2f}"
                if closeness > 0.8: color, thickness = (0, 0, 255), 3
                elif closeness > 0.5: color = (0, 165, 255)
            cv2.rectangle(out_frame, (x1, y1), (x2, y2), color, thickness)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(out_frame, (x1, y1 - th - 4), (x1 + tw + 4, y1), color, -1)
            cv2.putText(out_frame, label, (x1 + 2, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        if matched_pts1 is not None and matched_pts2 is not None and len(matched_pts1) > 0: # VO Matches
            num_to_draw = min(len(matched_pts1), 75) # Draw more points for PnP maybe
            indices = np.linspace(0, len(matched_pts1) - 1, num_to_draw, dtype=int)
            for i in indices:
                try:
                    pt1 = tuple(map(int, matched_pts1[i].ravel())); pt2 = tuple(map(int, matched_pts2[i].ravel()))
                    if 0 <= pt1[0] < w and 0 <= pt1[1] < h and 0 <= pt2[0] < w and 0 <= pt2[1] < h:
                        cv2.line(out_frame, pt1, pt2, (0, 165, 255), 1) # Orange lines
                        cv2.circle(out_frame, pt2, 3, (255, 0, 0), -1)    # Blue circles for PnP inliers
                except IndexError: continue
        if instructions: # Instructions
            y_offset = 30; cv2.putText(out_frame, "Nav Instructions:", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
            for i, instruction in enumerate(instructions[:4]): y_offset += 25; cv2.putText(out_frame, instruction, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(out_frame, f"FPS: {fps:.2f}", (w - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA) # FPS
        return out_frame
    except Exception as e: print(f"Error during combined visualization: {e}"); return frame

def visualize_trajectory(frame_shape, history_deque, current_pos_px, waypoints=None, obstacle_map_vis=None, cost_map_vis_color=None):
    """Creates a top-down trajectory view."""
    try:
        h, w = frame_shape[:2]
        canvas = cost_map_vis_color if cost_map_vis_color is not None else np.zeros((h, w, 3), dtype=np.uint8)
        history_pts = list(history_deque) # History Path
        if len(history_pts) >= 2:
            try: pts_np = np.array(history_pts, dtype=np.int32).reshape((-1, 1, 2)); cv2.polylines(canvas, [pts_np], isClosed=False, color=(255, 255, 0), thickness=2)
            except Exception as e: print(f"Error drawing history polyline: {e}")
        if waypoints and len(waypoints) >= 2: # Planned Path
            try:
                waypoints_np = np.array(waypoints, dtype=np.int32).reshape((-1, 1, 2)); cv2.polylines(canvas, [waypoints_np], isClosed=False, color=(0, 255, 0), thickness=2)
                for p in waypoints: cv2.circle(canvas, tuple(map(int, p)), 5, (0, 0, 255), -1); cv2.circle(canvas, tuple(map(int, p)), 6, (255, 255, 255), 1)
            except Exception as e: print(f"Error drawing waypoint polyline/circles: {e}")
        if current_pos_px: # Current Position
            try: pos_int = tuple(map(int, current_pos_px)); cv2.circle(canvas, pos_int, 7, (0, 255, 255), -1); cv2.circle(canvas, pos_int, 8, (0, 0, 0), 1)
            except Exception as e: print(f"Error drawing current position circle: {e}")
        cv2.putText(canvas, "Trajectory View (Top-Down)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        return canvas
    except Exception as e: print(f"Error during trajectory visualization: {e}"); h, w = frame_shape[:2]; return np.zeros((h, w, 3), dtype=np.uint8)

# --- Main Execution ---
YOLO = load_object_detection_model(device)
MIDAS, TRANSFORM = load_depth_model(device)