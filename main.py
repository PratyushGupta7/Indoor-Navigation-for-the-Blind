import cv2
import torch
import numpy as np
import warnings
from collections import deque

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
    model_type = "DPT_Hybrid"  # More accurate than Small but still fast enough
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
    depth_map = cv2.medianBlur(depth_map.astype(np.float32), 5)  # Smooth depth map
    return depth_map

def run_object_detection(model, frame, depth_map):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(rgb)
    detections = results.pandas().xyxy[0]
    objects = []

    for _, row in detections.iterrows():
        x1, y1, x2, y2 = map(int, [row.xmin, row.ymin, row.xmax, row.ymax])
        label = row['name']
        conf = row['confidence']
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        
        # Extract depth information
        if 0 <= cx < depth_map.shape[1] and 0 <= cy < depth_map.shape[0]:
            depth = depth_map[cy, cx]
            depth = np.clip(depth, 1e-3, None)
        else:
            depth = None
            
        # Calculate object dimensions
        width = x2 - x1
        height = y2 - y1
        
        objects.append({
            'label': label,
            'confidence': conf,
            'bbox': (x1, y1, x2, y2),
            'center': (cx, cy),
            'depth': depth,
            'size': (width, height)
        })
        
    return objects, detections

def visualize_detection_and_depth(frame, detections, depth_map):
    depth_norm = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
    depth_color = cv2.applyColorMap(depth_norm.astype(np.uint8), cv2.COLORMAP_MAGMA)
    blended = cv2.addWeighted(frame, 0.6, depth_color, 0.4, 0)

    for _, row in detections.iterrows():
        x1, y1, x2, y2 = map(int, [row.xmin, row.ymin, row.xmax, row.ymax])
        label = row['name']
        conf = row['confidence']
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        if 0 <= cx < depth_map.shape[1] and 0 <= cy < depth_map.shape[0]:
            depth = depth_map[cy, cx]
            depth = np.clip(depth, 1e-3, None)
        else:
            depth = None
        label_text = f"{label} {conf:.2f}"
        if depth is not None:
            label_text += f" | {depth:.2f}m"
        cv2.rectangle(blended, (x1, y1), (x2, y2), (0, 255, 0), 2)
        (w, h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(blended, (x1, y1 - h - 5), (x1 + w, y1), (0, 255, 0), -1)
        cv2.putText(blended, label_text, (x1, y1 - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return blended

def process_visual_odometry(prev_gray, gray, prev_kp=None, prev_des=None, new_K=None):
    orb = cv2.ORB_create(1000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    kp, des = orb.detectAndCompute(gray, None)
    
    R = None
    t = None
    matched_pts1 = None
    matched_pts2 = None
    
    if prev_gray is not None and prev_des is not None and len(prev_kp) > 0 and len(kp) > 0:
        matches = bf.match(prev_des, des)
        matches = sorted(matches, key=lambda x: x.distance)
        
        # Get enough good matches
        good_matches = matches[:max(int(len(matches) * 0.7), 15)]
        
        pts1 = np.float32([prev_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        pts2 = np.float32([kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        matched_pts1 = pts1
        matched_pts2 = pts2
        
        if len(pts1) >= 8:  # Need at least 8 points for a good estimate
            # Calculate Essential matrix
            E, mask = cv2.findEssentialMat(pts2, pts1, new_K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
            
            if E is not None and mask is not None:
                # Filter points using the mask
                inlier_mask = mask.ravel() == 1
                pts1_inliers = pts1[inlier_mask]
                pts2_inliers = pts2[inlier_mask]
                
                if len(pts1_inliers) >= 8:
                    # Recover pose (camera movement)
                    _, R, t, _ = cv2.recoverPose(E, pts2_inliers, pts1_inliers, new_K)
    
    return kp, des, R, t, matched_pts1, matched_pts2

def draw_visual_odometry(prev_pts, curr_pts, frame):
    vo_frame = frame.copy()
    if prev_pts is not None and curr_pts is not None:
        for i, (p_old, p_new) in enumerate(zip(prev_pts, curr_pts)):
            a, b = p_old.ravel()
            c, d = p_new.ravel()
            cv2.line(vo_frame, (int(a), int(b)), (int(c), int(d)), (255, 0, 0), 2)
            cv2.circle(vo_frame, (int(c), int(d)), 3, (0, 255, 0), -1)
    return vo_frame

def plan_trajectory(objects, current_pose, depth_map, frame_shape):
    """
    Plan trajectory based on detected objects, current pose and depth map.
    Returns waypoints, obstacle map, and cost map for visualization.
    """
    h, w = frame_shape[:2]
    obstacle_map = np.zeros((h, w), dtype=np.uint8)
    cost_map = np.ones((h, w), dtype=np.float32) * 255
    
    # Mark detected objects as obstacles with safety margin
    for obj in objects:
        if obj['depth'] is not None and obj['depth'] < 10.0:  # Filter by depth
            x1, y1, x2, y2 = obj['bbox']
            # Add safety margin around objects
            margin = 20
            x1_safe = max(0, x1 - margin)
            y1_safe = max(0, y1 - margin)
            x2_safe = min(w, x2 + margin)
            y2_safe = min(h, y2 + margin)
            
            # Mark obstacle area
            obstacle_map[y1_safe:y2_safe, x1_safe:x2_safe] = 255
            
            # Higher cost near obstacles for path planning
            cv2.rectangle(cost_map, (x1_safe, y1_safe), (x2_safe, y2_safe), 0, -1)
            cv2.rectangle(cost_map, (x1, y1), (x2, y2), 0, -1)
    
    # Create distance transform for smoother cost gradient
    dist_transform = cv2.distanceTransform(255 - obstacle_map, cv2.DIST_L2, 3)
    dist_transform = cv2.normalize(dist_transform, None, 0, 1, cv2.NORM_MINMAX)
    
    # Convert to cost (closer to obstacles = higher cost)
    smooth_cost = (1 - dist_transform) * 255
    
    # Apply depth information to cost map (closer = higher cost)
    depth_cost = cv2.normalize(depth_map, None, 0, 0.5, cv2.NORM_MINMAX)
    depth_cost = (1 - depth_cost) * 127  # Scale to half intensity
    
    # Combine costs
    final_cost = cv2.addWeighted(smooth_cost, 0.7, depth_cost, 0.3, 0)
    
    # Simple goal points (center bottom of frame as target for demo)
    start = (w // 2, h - 30)
    goal = (w // 2, h // 4)
    
    # Generate waypoints (simplified for demonstration)
    waypoints = []
    waypoints.append(start)
    
    # Add intermediate waypoints (simplified A* approach)
    # For real implementation, you'd use proper A* or RRT algorithm
    current = start
    step_size = 20
    max_steps = 20
    
    for _ in range(max_steps):
        best_cost = float('inf')
        best_point = None
        
        # Look in 8 directions
        for dx, dy in [(0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1)]:
            nx, ny = current[0] + dx * step_size, current[1] + dy * step_size
            
            if 0 <= nx < w and 0 <= ny < h:
                # Don't select points in obstacles
                if obstacle_map[ny, nx] == 0:
                    # Calculate cost: distance to goal + point cost
                    dist_cost = np.sqrt((nx - goal[0])**2 + (ny - goal[1])**2)
                    point_cost = final_cost[ny, nx]
                    total_cost = dist_cost + point_cost * 0.5
                    
                    if total_cost < best_cost:
                        best_cost = total_cost
                        best_point = (nx, ny)
        
        if best_point is None:
            break
            
        waypoints.append(best_point)
        current = best_point
        
        # If close to goal, stop
        if np.sqrt((current[0] - goal[0])**2 + (current[1] - goal[1])**2) < step_size:
            waypoints.append(goal)
            break
    
    return waypoints, obstacle_map, final_cost.astype(np.uint8)

def visualize_trajectory(frame, trajectory_points, position, waypoints=None, obstacle_map=None, cost_map=None):
    """
    Create a visualization of the trajectory and planned path
    """
    # Create maps for trajectory visualization
    h, w = frame.shape[:2]
    traj_map = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Visualize cost map in background
    if cost_map is not None:
        cost_color = cv2.applyColorMap(cost_map, cv2.COLORMAP_JET)
        traj_map = cv2.addWeighted(traj_map, 0.3, cost_color, 0.7, 0)
    
    # Visualize obstacles
    if obstacle_map is not None:
        obstacle_vis = np.zeros((h, w, 3), dtype=np.uint8)
        obstacle_vis[obstacle_map > 0] = [0, 0, 255]  # Red for obstacles
        traj_map = cv2.addWeighted(traj_map, 0.7, obstacle_vis, 0.3, 0)
    
    # Draw the trajectory history
    for i in range(1, len(trajectory_points)):
        pt1 = (int(trajectory_points[i-1][0]), int(trajectory_points[i-1][1]))
        pt2 = (int(trajectory_points[i][0]), int(trajectory_points[i][1]))
        cv2.line(traj_map, pt1, pt2, (0, 255, 255), 2)
    
    # Draw current position
    cv2.circle(traj_map, position, 5, (255, 255, 0), -1)
    
    # Draw planned waypoints
    if waypoints and len(waypoints) > 1:
        for i in range(1, len(waypoints)):
            cv2.line(traj_map, waypoints[i-1], waypoints[i], (0, 255, 0), 2)
        for point in waypoints:
            cv2.circle(traj_map, point, 3, (0, 0, 255), -1)
    
    # Add title and legend
    title_bg = np.zeros((40, w, 3), dtype=np.uint8)
    cv2.putText(title_bg, "Trajectory Planning", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    legend_bg = np.zeros((60, w, 3), dtype=np.uint8)
    cv2.putText(legend_bg, "Yellow: Current Pos | Green: Planned Path | Cyan: History", (10, 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(legend_bg, "Red: Obstacles | Color Map: Cost (Blue=Low, Red=High)", (10, 45), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Combine
    full_vis = np.vstack([title_bg, traj_map, legend_bg])
    
    return full_vis

def combine_visualizations(yolo_depth, vo_frame):
    """Combine YOLO+Depth and VO frames into one visualization"""
    h, w = yolo_depth.shape[:2]
    combined = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Blend the two visualizations
    combined = cv2.addWeighted(yolo_depth, 0.7, vo_frame, 0.3, 0)
    
    # Add labels
    cv2.putText(combined, "YOLO + Depth + VO", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    return combined

def main():
    model = load_object_detection_model(device)
    midas, midas_transform = load_depth_model(device)

    orb = cv2.ORB_create(1000)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    
    # Get camera properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # For visual odometry
    prev_gray = None
    prev_kp = None
    prev_des = None
    
    # For trajectory visualization
    trajectory_points = deque(maxlen=100)  # Store last 100 points
    current_position = (frame_width // 2, frame_height // 2)  # Start in the middle
    scale_factor = 5.0  # Scale factor for trajectory visualization
    
    # Initialize cumulative translation and rotation
    cum_translation = np.zeros((3, 1))
    cum_rotation = np.eye(3)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame.")
            break

        h, w = frame.shape[:2]
        new_K, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 1, (w, h))
        undistorted = cv2.undistort(frame, K, dist, None, new_K)

        rgb = cv2.cvtColor(undistorted, cv2.COLOR_BGR2RGB)
        
        # Run depth estimation
        depth_map = run_depth_estimation(midas, midas_transform, rgb, device)
        
        # Run object detection
        objects, detections = run_object_detection(model, undistorted, depth_map)
        
        # Create visualization with YOLO detections and depth
        yolo_depth_vis = visualize_detection_and_depth(undistorted, detections, depth_map)
        
        # Process visual odometry
        gray = cv2.cvtColor(undistorted, cv2.COLOR_BGR2GRAY)
        kp, des, R, t, matched_pts1, matched_pts2 = process_visual_odometry(
            prev_gray, gray, prev_kp, prev_des, new_K)
        
        # Update trajectory if we have motion
        if t is not None and R is not None:
            # Update cumulative pose (camera movement)
            cum_rotation = cum_rotation.dot(R)
            cum_translation = cum_translation + scale_factor * cum_rotation.dot(t)
            
            # Update current position from cumulative translation
            tx, ty, tz = cum_translation.ravel()
            current_position = (
                int(frame_width // 2 + tx * 10),  # Scale for visibility
                int(frame_height // 2 + tz * 10)  # Use x and z for top-down view
            )
            
            # Add to trajectory history
            trajectory_points.append(current_position)
        
        # Draw visual odometry on frame
        vo_frame = draw_visual_odometry(matched_pts1, matched_pts2, undistorted)
        
        # Plan trajectory based on detected objects and current position
        waypoints, obstacle_map, cost_map = plan_trajectory(objects, current_position, depth_map, undistorted.shape)
        
        # Create trajectory visualization
        trajectory_vis = visualize_trajectory(
            undistorted, trajectory_points, current_position, waypoints, obstacle_map, cost_map)
        
        # Combine YOLO+Depth+VO visualizations
        combined_vis = combine_visualizations(yolo_depth_vis, vo_frame)

        # Show the two windows
        cv2.imshow('YOLO + Depth + VO', combined_vis)
        cv2.imshow('Trajectory Planning', trajectory_vis)

        # Update previous frame data
        prev_gray = gray.copy()
        prev_kp = kp
        prev_des = des

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
