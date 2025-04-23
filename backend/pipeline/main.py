import cv2
import numpy as np
from collections import deque
import threading
import time
from concurrent.futures import ThreadPoolExecutor
import sys 

from backend.pipeline.setup import *

if __name__ == "__main__":
    print("System initializing...")
    speak("System initializing.")
    tts_thread = threading.Thread(target=tts_worker, daemon=True)
    tts_thread.start()

    yolo_model = load_object_detection_model(device)
    midas_model, midas_transform = load_depth_model(device)

    cap = None
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened(): raise IOError("Cannot open camera.")
        fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Camera opened successfully: {fw}x{fh}")
        # Update placeholder K if needed
        if K[0, 0] > 500 and K[0, 2] != fw/2.0 :
            print("Updating placeholder K principal point.")
            K[0, 2] = fw / 2.0
            K[1, 2] = fh / 2.0
    except Exception as e:
        print(f"FATAL: Error initializing camera: {e}")
        speak("Error initializing camera.")
        tts_stop_event.set()
        if tts_thread.is_alive():
            try:
                tts_queue.put(None); tts_thread.join(timeout=2)
            except Exception as te: print(f"Error stopping TTS thread: {te}")
        if cap is not None: cap.release(); cv2.destroyAllWindows(); sys.exit(1)

    # Ensure K is float32 for PnP
    K_intrinsic_calib = K.astype(np.float32)
    new_K, roi = cv2.getOptimalNewCameraMatrix(K_intrinsic_calib, dist, (fw, fh), alpha=0.9, newImgSize=(fw, fh))
    new_K = new_K.astype(np.float32) # Ensure new_K is also float32
    mapx, mapy = cv2.initUndistortRectifyMap(K_intrinsic_calib, dist, None, new_K, (fw, fh), cv2.CV_32FC1)
    print("Undistortion maps created.")
    print("Using Intrinsics for PnP:\n", new_K)

    # --- State Variables ---
    prev_gray = None
    prev_kp = None
    prev_des = None
    prev_depth_map = None # <<< Added for PnP VO
    traj_hist = deque(maxlen=200)
    origin_px = (fw // 2, fh - fh // 10)
    current_pos_px = origin_px
    traj_hist.append(current_pos_px)
    cumulative_R = np.eye(3, dtype=np.float32) # Use float32 for consistency
    cumulative_t = np.zeros((3, 1), dtype=np.float32)
    # === VO Tuning Parameters ===
    vis_odom_scale = 40.0 # Scale factor for VISUALIZING trajectory from VO translation
    depth_pnp_scale = 5.0 # ** CRUCIAL **: Heuristic scale for PnP depth (maps normalized closeness -> metric Z)
    # This depth_pnp_scale needs tuning based on typical scene depth & MiDaS output range.
    # A larger value means normalized depth differences map to larger metric Z differences.
    # ==========================

    num_workers = 3
    executor = ThreadPoolExecutor(max_workers=num_workers, thread_name_prefix='Worker')
    print(f"Thread pool started with {num_workers} workers.")

    # --- Main Loop ---
    frame_count = 0; last_instruction_time = time.time(); instruction_speak_delay = 3.5; fps = 0.0
    print("Starting main loop... Press 'q' to quit.")
    speak("Navigation system ready.")

    try:
        while True:
            loop_start_time = time.time()

            # 1. Read Frame
            try:
                ret, frame = cap.read();
                if not ret: print("End of stream. Exiting."); speak("Camera feed stopped."); break
            except Exception as e: print(f"Error reading frame: {e}"); speak("Camera error."); break

            # 2. Preprocessing
            try:
                undistorted_frame = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)
                rgb_frame_copy = cv2.cvtColor(undistorted_frame, cv2.COLOR_BGR2RGB)
                gray_frame = cv2.cvtColor(undistorted_frame, cv2.COLOR_BGR2GRAY)
            except cv2.error as e: print(f"OpenCV error during preprocessing: {e}"); continue

            # 3. Submit Tasks (Pass PREVIOUS depth map to VO)
            try:
                future_depth = executor.submit(run_depth_estimation, midas_model, midas_transform, rgb_frame_copy, device, stream_depth)
                future_objects = executor.submit(run_object_detection, yolo_model, undistorted_frame, device, stream_objdet)
                # <<< Pass previous depth map to PnP VO function >>>
                future_vo = executor.submit(process_visual_odometry_pnp,
                                            prev_gray, gray_frame, prev_kp, prev_des,
                                            prev_depth_map, # Pass previous depth map
                                            new_K, # Use undistorted intrinsics
                                            None, # Distortion handled by remap
                                            depth_vo_scale=depth_pnp_scale) # Pass PnP scale
            except Exception as e: print(f"Error submitting tasks: {e}"); continue

            # 4. Retrieve Results
            try:
                depth_map_closeness = future_depth.result() # Current depth map
                objects, _ = future_objects.result()
                kp, des, R_vo, t_vo, vo_pts1, vo_pts2 = future_vo.result()
            except Exception as e:
                print(f"Error retrieving results: {e}")
                depth_map_closeness = np.zeros_like(gray_frame, dtype=np.float32) if prev_depth_map is None else prev_depth_map
                objects = []; kp, des = prev_kp, prev_des; R_vo, t_vo, vo_pts1, vo_pts2 = None, None, None, None

            # 5. Process Results
            try:
                objects = update_object_depth(objects, depth_map_closeness)

                # Update Visual Odometry Pose
                if R_vo is not None and t_vo is not None:
                     # Ensure float32 for matrix operations
                    R_vo = R_vo.astype(np.float32)
                    t_vo = t_vo.astype(np.float32)

                    # Accumulate Pose: T_world_curr = T_prev_curr * T_world_prev
                    cumulative_t = cumulative_t + cumulative_R @ t_vo # Compact form
                    cumulative_R = R_vo @ cumulative_R # R_vo is rotation from prev to curr

                    # Update visualization position using the VISUAL ODOMETRY scale
                    # Adjust viz scale relative to PnP scale - may still need tuning
                    viz_scale_factor = (vis_odom_scale / depth_pnp_scale) if depth_pnp_scale != 0 else 1.0
                    vis_x = int(origin_px[0] + cumulative_t[0, 0] * viz_scale_factor)
                    vis_y = int(origin_px[1] - cumulative_t[2, 0] * viz_scale_factor) # Map Z change to screen Y change
                    vis_x = np.clip(vis_x, 0, fw - 1); vis_y = np.clip(vis_y, 0, fh - 1)
                    if np.linalg.norm(np.array(current_pos_px) - np.array((vis_x, vis_y))) > 1: # Sensitivity reduced
                        current_pos_px = (vis_x, vis_y); traj_hist.append(current_pos_px)


                # Plan trajectory
                waypoints, obstacle_map_vis, cost_map_vis_color = plan_trajectory(objects, current_pos_px, depth_map_closeness, undistorted_frame.shape)
                instructions = generate_instructions(waypoints)

            except Exception as e:
                print(f"Error during post-processing: {e}")
                instructions = ["Processing Error"]
                waypoints, obstacle_map_vis, cost_map_vis_color = [current_pos_px], np.zeros_like(gray_frame, dtype=np.uint8), np.zeros((fh, fw, 3), dtype=np.uint8)

            # 6. Speak Alerts / Instructions
            try:
                very_close_obstacle = False; obstacle_warning_threshold = 0.85
                for obj in objects:
                    if obj.get('closeness', 0.0) > obstacle_warning_threshold:
                        very_close_obstacle = True
                        # --- MODIFICATION HERE ---
                        # Speak generic warning instead of specific object label
                        speak("Warning! Object very close ahead!")
                        instructions = ["STOP! Obstacle detected!"] # Keep instruction generic too
                        # -------------------------
                        last_instruction_time = time.time(); break # Reset timer and exit loop

                current_time = time.time()
                # Speak navigation instructions periodically if no immediate danger
                if not very_close_obstacle and instructions and instructions[0] != "No path computed" and (current_time - last_instruction_time > instruction_speak_delay):
                     if instructions[0] not in ["Processing Error", "STOP! Obstacle detected!"]:
                          speak(instructions[0]); last_instruction_time = current_time
            except Exception as e: print(f"Error during alert/TTS processing: {e}")

            # 7. Visualization
            loop_end_time = time.time(); elapsed_time = loop_end_time - loop_start_time
            fps = 0.9 * fps + 0.1 * (1.0 / elapsed_time if elapsed_time > 0 else 0)
            try:
                combined_view = visualize_combined(undistorted_frame, objects, vo_pts1, vo_pts2, instructions, fps)
                trajectory_view = visualize_trajectory(undistorted_frame.shape, traj_hist, current_pos_px, waypoints, obstacle_map_vis, cost_map_vis_color)
                cv2.imshow('Object Detection + PnP VO View', combined_view) # Window title updated
                cv2.imshow('Trajectory Planning View', trajectory_view)
            except cv2.error as e: print(f"OpenCV error during visualization: {e}")
            except Exception as e: print(f"General error during visualization: {e}")

            # --- Update State for Next Loop ---
            prev_gray = gray_frame # No copy needed if gray_frame isn't modified elsewhere before next loop
            prev_kp = kp
            prev_des = des
            prev_depth_map = depth_map_closeness.copy() # <<< MUST copy the depth map for use in the next frame's VO

            frame_count += 1

            # 8. Exit Condition
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): print("'q' pressed, initiating shutdown."); speak("Exiting program."); break
            elif key == ord('s'): speak(f"Status OK. Position {current_pos_px}. FPS {fps:.1f}.")

    except KeyboardInterrupt: print("\nCtrl+C detected. Initiating graceful shutdown."); speak("Shutdown initiated.")
    finally:
        # --- Cleanup ---
        print("Cleaning up resources...")

        # 1. Signal threads to stop and shutdown executor
        if 'executor' in locals() and executor is not None:
            print("Shutting down thread pool...")
            try:
                executor.shutdown(wait=False, cancel_futures=True) # Don't wait, try to cancel running
                print("Thread pool shutdown signal sent.")
            except Exception as e:
                print(f"Error shutting down thread pool: {e}")

        # 2. Stop TTS thread
        print("Stopping TTS worker...")
        tts_stop_event.set()
        if 'tts_thread' in locals() and tts_thread.is_alive():
            try:
                 tts_queue.put(None) # Send sentinel
                 tts_thread.join(timeout=3.0) # Wait for thread to finish
                 if tts_thread.is_alive():
                      print("Warning: TTS thread did not terminate gracefully.")
                 else:
                      print("TTS worker stopped.")
            except Exception as e:
                 print(f"Error stopping TTS thread: {e}")

        # 3. Release Camera
        if 'cap' in locals() and cap is not None and cap.isOpened():
            print("Releasing camera...")
            try:
                cap.release()
                print("Camera released.")
            except Exception as e:
                print(f"Error releasing camera: {e}")

        # 4. Destroy OpenCV Windows
        print("Destroying OpenCV windows...")
        try:
            cv2.destroyAllWindows()
            # Add a small wait for windows to actually close on some systems
            # cv2.waitKey(50)
            print("OpenCV windows destroyed.")
        except Exception as e:
            print(f"Error destroying OpenCV windows: {e}")

        print("Cleanup finished. Exiting.")
        # Explicit exit call might be needed if daemon threads delay exit
        sys.exit(0)
