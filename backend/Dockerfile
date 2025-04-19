FROM python:3.13-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    libxext6 \
    libsm6 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    espeak \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . ./backend

ENV PYTHONPATH=/app/backend

EXPOSE 8000

COPY pipeline/calibration /app/backend/pipeline/calibration

CMD ["uvicorn", "backend.api.api:app", "--host", "0.0.0.0", "--port", "8000"]
