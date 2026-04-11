import time

import RPi.GPIO as GPIO

BUZZER_PIN = 12


class Buzzer:
    def __init__(self, pin: int = BUZZER_PIN) -> None:
        self.pin = pin
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.pin, GPIO.OUT)
        GPIO.output(self.pin, GPIO.LOW)

    def tone(self, frequency: int, duration: float) -> None:
        """Play a tone at the given frequency (Hz) for the given duration (seconds)."""
        period = 1.0 / frequency
        half_period = period / 2.0
        cycles = int(duration / period)
        for _ in range(cycles):
            GPIO.output(self.pin, GPIO.HIGH)
            time.sleep(half_period)
            GPIO.output(self.pin, GPIO.LOW)
            time.sleep(half_period)

    def silence(self, duration: float) -> None:
        """Stay silent for the given duration (seconds)."""
        time.sleep(duration)

    def off(self) -> None:
        """Immediately stop output."""
        GPIO.output(self.pin, GPIO.LOW)

    def cleanup(self) -> None:
        self.off()
        GPIO.cleanup()


# Note frequencies (Hz)
B5 = 988
E6 = 1319


def mario_coin(buzzer: Buzzer) -> None:
    """Play the Mario coin sound: a quick B5 then E6 accent."""
    buzzer.tone(B5, 0.075)
    buzzer.tone(E6, 0.125)
