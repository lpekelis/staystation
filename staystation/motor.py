import time

import RPi.GPIO as GPIO

# 5.625*(1/64) per step, 4096 steps is 360°
STEPS_PER_REVOLUTION = 4096

# A treat is dispenses every 90° (quarter revolution), so 1024 steps
STEPS_PER_TREAT = STEPS_PER_REVOLUTION // 4

# 8-step sequence (http://www.4tronix.co.uk/arduino/Stepper-Motors.php)
_STEP_SEQUENCE = [
    [1, 0, 0, 1],
    [1, 0, 0, 0],
    [1, 1, 0, 0],
    [0, 1, 0, 0],
    [0, 1, 1, 0],
    [0, 0, 1, 0],
    [0, 0, 1, 1],
    [0, 0, 0, 1],
]


class Motor:
    def __init__(
        self, in1: int = 17, in2: int = 18, in3: int = 27, in4: int = 22, step_sleep: float = 0.002
    ) -> None:
        # careful lowering step_sleep — mechanical limits kick in quickly
        self.pins = [in1, in2, in3, in4]
        self.step_sleep = step_sleep
        self._step_counter = 0

        GPIO.setmode(GPIO.BCM)
        for pin in self.pins:
            GPIO.setup(pin, GPIO.OUT)
            GPIO.output(pin, GPIO.LOW)

    def dispense(self, steps: int = STEPS_PER_TREAT) -> None:
        """Advance the motor a given number of steps (one full revolution by default)."""
        for _ in range(steps):
            for i, pin in enumerate(self.pins):
                GPIO.output(pin, _STEP_SEQUENCE[self._step_counter][i])
            self._step_counter = (self._step_counter + 1) % 8
            time.sleep(self.step_sleep)

    def cleanup(self) -> None:
        for pin in self.pins:
            GPIO.output(pin, GPIO.LOW)
        GPIO.cleanup()
