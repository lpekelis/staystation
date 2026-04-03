import io

from fastapi import FastAPI, File, UploadFile
from PIL import Image
from ultralytics import YOLO

app = FastAPI()
model = YOLO("/ultralytics/yolo26s_ncnn_model", task="detect")


@app.post("/detect")
async def detect(file: UploadFile = File(...), confidence: float = 0.5):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))

    results = model.predict(source=image, conf=confidence, verbose=False)

    detections = []
    for r in results:
        for box in r.boxes:
            detections.append(
                {
                    "class_id": int(box.cls[0]),
                    "class_name": r.names[int(box.cls[0])],
                    "confidence": float(box.conf[0]),
                    "bbox": box.xyxy[0].tolist(),  # [x1, y1, x2, y2]
                }
            )

    return {"detections": detections}


@app.get("/health")
async def health():
    return {"status": "ok"}
