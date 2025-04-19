from pathlib import Path


# assume this file lives at backend/pipeline/main.py
BASE_DIR = Path(__file__).resolve().parent.parent   # points to backend/
print(BASE_DIR)

MODELS_DIR = BASE_DIR / "models"
CALIB_DIR  = BASE_DIR / "pipeline" / "calibration"

# K = np.load(CALIB_DIR / "camera_matrix.npy")
# dist = np.load(CALIB_DIR / "distortion_coeffs.npy")

# # …

# repo_yolo = MODELS_DIR / "ultralytics_yolov5_master"
# repo_midas = MODELS_DIR / "intel-isl_MiDaS_master"
