# Dockerfile (place in the project root, next to `backend/`)
FROM python:3.13-slim

# 1) Install system deps for OpenCV, TTS, etc.
RUN apt-get update && apt-get install -y \
    build-essential libxext6 libsm6 libgl1-mesa-glx libglib2.0-0 espeak \
  && rm -rf /var/lib/apt/lists/*

# 2) Set workdir to /app, where we'll copy the code
WORKDIR /app

# 3) Install Python deps
COPY backend/requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# 4) Copy your entire backend/ folder into the image
COPY backend ./backend

# 5) Ensure Python can import 'backend' as a top‐level package
ENV PYTHONPATH=/app

# 6) Expose Uvicorn port
EXPOSE 10000

# 7) Run Uvicorn exactly as you do locally
CMD ["uvicorn", "backend.api.api:app", "--host", "0.0.0.0", "--port", "10000"]
