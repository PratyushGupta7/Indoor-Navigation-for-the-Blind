# uvicorn backend.api.api:app --host 0.0.0.0 --port 8000 --reload
# ngrok http 8000

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import cv2
import numpy as np
from backend.pipeline.setup import *

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class DetectionResponse(BaseModel):
    objects: list
    waypoints: list
    instructions: list

@app.post("/predict", response_model=DetectionResponse)
async def predict(file: UploadFile = File(...)):
    # read image
    data = await file.read()
    nparr = np.frombuffer(data, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # 1) depth
    depth = run_depth_estimation(MIDAS, TRANSFORM, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), device, None)

    # 2) objects
    objects, _ = run_object_detection(YOLO, frame, device, None)
    objects = update_object_depth(objects, depth)

    # 3) vo & pose‑tracking state would normally be stored per‑session; 
    #    for stateless demos you could skip VO and just re‑plan from center
    curr_pos = (frame.shape[1]//2, frame.shape[0]//2)
    waypoints, _, _ = plan_trajectory(objects, curr_pos, depth, frame.shape)

    # 4) instructions
    instr = generate_instructions(waypoints)

    return DetectionResponse(objects=objects,
                             waypoints=waypoints,
                             instructions=instr)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
