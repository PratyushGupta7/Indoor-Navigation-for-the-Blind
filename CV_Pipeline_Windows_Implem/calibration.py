import glob
import cv2
import numpy as np

# Define chessboard size (number of internal corners)
chessboard_size = (6, 8)  # For a 7x9 square chessboard (width x height)

# Prepare object points
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

# Arrays to store object points and image points
objpoints = []
imgpoints = []

# Load calibration images
images = glob.glob('calibration_images/*.jpg')
if not images:
    print("Error: No JPEG images found in calibration_images folder.")
    exit()

for fname in images:
    img = cv2.imread(fname)
    if img is None:
        print(f"Warning: Failed to load image {fname}")
        continue
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    if ret:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), 
                                   criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        imgpoints.append(corners2)
        cv2.drawChessboardCorners(img, chessboard_size, corners2, ret)
        cv2.imshow('Chessboard Corners', img)
        cv2.waitKey(500)
    else:
        print(f"Warning: Chessboard corners not found in {fname}")

cv2.destroyAllWindows()

if not objpoints or not imgpoints:
    print("Error: No valid chessboard images processed. Calibration aborted.")
    exit()

# Calibrate camera
ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

if ret:
    print("Camera matrix K:\n", K)
    print("Distortion coefficients:\n", dist)
    # Save to files
    np.save('camera_matrix.npy', K)
    np.save('distortion_coeffs.npy', dist)
    print("Calibration files saved successfully.")
else:
    print("Error: Camera calibration failed.")
    exit()