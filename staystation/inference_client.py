import io

import numpy as np
import requests
from PIL import Image

INFERENCE_URL = "http://localhost:8080"


def detect(frame: np.ndarray, confidence: float = 0.5) -> list[dict]:
    """Send a numpy frame (from picamera2) to the inference server."""
    image = Image.fromarray(frame)
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=85)
    buf.seek(0)

    response = requests.post(
        f"{INFERENCE_URL}/detect",
        files={"file": ("frame.jpg", buf, "image/jpeg")},
        params={"confidence": confidence},
        timeout=10,
    )
    response.raise_for_status()
    return response.json()["detections"]


def health_check() -> bool:
    try:
        r = requests.get(f"{INFERENCE_URL}/health", timeout=3)
        return r.status_code == 200
    except requests.ConnectionError:
        return False
