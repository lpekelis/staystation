import time

from staystation.camera import Camera
from staystation.conditioning import Conditioning
from staystation.inference_client import health_check
from staystation.motor import Motor


def main() -> None:
    assert health_check(), "Inference server not reachable — start Docker first"

    motor = Motor()
    camera = Camera()
    conditioning = Conditioning(motor)

    camera.start()
    time.sleep(1)  # warm-up

    try:
        while True:
            frame = camera.capture_frame()
            conditioning.step(frame)
    except KeyboardInterrupt:
        pass
    finally:
        camera.stop()
        motor.cleanup()


if __name__ == "__main__":
    main()
