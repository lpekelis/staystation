from staystation.inference_client import detect
from staystation.motor import Motor


class Conditioning:
    def __init__(self, motor: Motor, confidence_threshold: float = 0.5):
        self.motor = motor
        self.confidence_threshold = confidence_threshold

    def step(self, frame) -> list[dict]:
        """Run one detect → decide → reward cycle. Returns detections."""
        detections = detect(frame, confidence=self.confidence_threshold)
        cat_detections = [d for d in detections if d["class_name"] == "cat"]

        if cat_detections:
            print(f"Cat detected (conf={cat_detections[0]['confidence']:.2f}) — dispensing treat")
            self.motor.dispense()

        return detections
