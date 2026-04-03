"""OpenCV GUI test script — requires a display (local or ssh -X)."""

import time

import cv2

from staystation.camera import Camera
from staystation.inference_client import detect, health_check


def draw_detections(frame, detections):
    for d in detections:
        x1, y1, x2, y2 = [int(v) for v in d["bbox"]]
        label = f'{d["class_name"]} {d["confidence"]:.2f}'
        color = (0, 255, 0) if d["class_name"] == "cat" else (0, 165, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return frame


def main():
    assert health_check(), "Inference server not running — start Docker first"

    cam = Camera(resolution=(640, 480))
    cam.start()
    time.sleep(1)  # warm-up

    try:
        while True:
            frame = cam.capture_frame()
            detections = detect(frame, confidence=0.4)

            annotated = draw_detections(frame.copy(), detections)
            cv2.imshow("StayStation - Detection Test", cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cam.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
